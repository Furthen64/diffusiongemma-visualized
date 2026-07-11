"""Core simulation engine for DiffusionGemma's diffusion process.

This module simulates the discrete diffusion denoising loop without requiring
the actual model.  It is designed to produce visually realistic behavior that
mirrors the real DiffusionGemma architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Small default vocabulary — enough for visual variety, small enough to render
# ---------------------------------------------------------------------------
DEFAULT_VOCAB: list[str] = [
    "the", "a", "is", "it", "to", "and", "of", "in", "for", "that",
    "on", "with", "as", "at", "by", "an", "be", "this", "from", "or",
    "was", "are", "has", "not", "but", "had", "can", "will", "do", "if",
    "so", "no", "up", "my", "we", "he", "me", "her", "his", "its",
    "one", "all", "would", "there", "their", "what", "about", "which",
    "when", "make", "like", "time", "just", "know", "take", "people",
    "into", "year", "your", "some", "them", "than", "other", "could",
]

NOISE_TOKEN = "\u2588"  # block character for random/noised positions


@dataclass
class StepSnapshot:
    """Snapshot of one denoising step."""
    step: int
    canvas: np.ndarray  # token indices
    logits: np.ndarray  # (canvas_size, vocab_size)
    entropy: np.ndarray  # per-position entropy
    accepted_mask: np.ndarray  # bool — which positions were kept
    argmax_canvas: np.ndarray  # current best-guess
    converged: bool
    self_cond_gate: float = 0.0
    cumulative_entropy: float = 0.0
    budget_used: float = 0.0


@dataclass
class BlockResult:
    """Full history for one 256-token block (or smaller in viz)."""
    steps: List[StepSnapshot] = field(default_factory=list)
    target_tokens: np.ndarray = field(default_factory=lambda: np.array([]))
    committed_tokens: np.ndarray = field(default_factory=lambda: np.array([]))
    num_accepted_per_step: List[int] = field(default_factory=list)


class DiffusionSim:
    """Configurable discrete diffusion simulation."""

    def __init__(
        self,
        canvas_size: int = 16,
        vocab_size: int = 0,
        entropy_bound: float = 0.1,
        max_steps: int = 20,
        convergence_threshold: int = 2,
        entropy_threshold: float = 0.3,
        temperature: float = 1.0,
        seed: int | None = None,
    ):
        self.canvas_size = canvas_size
        self.entropy_bound = entropy_bound
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
        self.entropy_threshold = entropy_threshold
        self.temperature = temperature

        self.rng = np.random.default_rng(seed)

        # Vocabulary
        self.vocab = DEFAULT_VOCAB[:vocab_size] if vocab_size > 0 else DEFAULT_VOCAB
        self.vocab_size = len(self.vocab)

        # Hidden target — the "true" text the diffusion converges toward
        self.target = self.rng.integers(0, self.vocab_size, size=canvas_size)

        # Per-position "convergence speed" — some positions lock in faster
        self._convergence_rates = self.rng.uniform(0.3, 1.0, size=canvas_size)

        # Self-conditioning state
        self._prev_softmax: np.ndarray | None = None
        self._gate_values: list[float] = []

    def vocab_token(self, idx: int) -> str:
        if idx < 0 or idx >= self.vocab_size:
            return NOISE_TOKEN
        return self.vocab[idx]

    def canvas_tokens(self, canvas: np.ndarray) -> list[str]:
        return [self.vocab_token(int(i)) for i in canvas]

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def _compute_logits(
        self,
        step: int,
        canvas: np.ndarray,
        self_cond: np.ndarray | None,
    ) -> np.ndarray:
        """Simulate logits for each position.

        The simulation blends:
        1. A target distribution (increases with step number)
        2. Random noise (decreases with step number)
        3. Local context smoothing (increases with step number)
        4. Self-conditioning signal (if provided)
        """
        n = self.canvas_size
        v = self.vocab_size
        progress = min(step / max(self.max_steps - 1, 1), 1.0)

        logits = np.zeros((n, v), dtype=np.float64)

        # --- Target signal (increases with step) ---
        for i in range(n):
            target_tok = int(self.target[i])
            rate = self._convergence_rates[i]
            strength = rate * progress ** 0.7
            # Put a strong peak on the target token
            logits[i, target_tok] += strength * 3.0
            # Small probability mass on a few neighboring vocab tokens
            # to simulate "almost right" predictions
            nearby = self.rng.choice(v, size=min(3, v), replace=False)
            for nb in nearby:
                logits[i, nb] += strength * 0.3

        # --- Noise (decreases with step) ---
        noise_level = (1.0 - progress ** 0.6) * 2.0
        noise = self.rng.standard_normal((n, v)) * noise_level
        logits += noise

        # --- Local context smoothing ---
        context_strength = progress * 0.5
        for i in range(n):
            for di in [-2, -1, 1, 2]:
                j = i + di
                if 0 <= j < n:
                    neighbor_tok = int(canvas[j])
                    logits[i, neighbor_tok] += context_strength

        # --- Self-conditioning ---
        if self_cond is not None and self_cond.shape == (n, v):
            gate = min(progress * 1.5, 0.8)
            logits += self_cond * gate * 0.4

        return logits

    def _logits_to_softmax(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / max(self.temperature, 0.01)
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def _per_position_entropy(self, softmax: np.ndarray) -> np.ndarray:
        eps = 1e-12
        log_p = np.log(softmax + eps)
        return -np.sum(softmax * log_p, axis=-1)

    def entropy_bound_accept(
        self,
        entropy: np.ndarray,
        budget: float,
    ) -> tuple[np.ndarray, float]:
        """Sort positions by confidence (lowest entropy first), accept until
        accumulated entropy exceeds *budget*.

        Returns (accepted_mask, budget_used).
        """
        order = np.argsort(entropy)  # most confident first
        cumulative = 0.0
        accepted = np.zeros(len(entropy), dtype=bool)
        for idx in order:
            if cumulative + entropy[idx] <= budget:
                accepted[idx] = True
                cumulative += entropy[idx]
            else:
                break
        return accepted, cumulative

    def _renoise(self, canvas: np.ndarray, accepted: np.ndarray) -> np.ndarray:
        """Replace rejected positions with fresh random tokens."""
        new_canvas = canvas.copy()
        rejected = ~accepted
        n_rejected = int(np.sum(rejected))
        if n_rejected > 0:
            new_canvas[rejected] = self.rng.integers(
                0, self.vocab_size, size=n_rejected
            )
        return new_canvas

    def check_convergence(
        self,
        argmax_history: list[np.ndarray],
        mean_entropy: float,
    ) -> bool:
        if len(argmax_history) < self.convergence_threshold:
            return False
        recent = argmax_history[-self.convergence_threshold :]
        all_same = all(np.array_equal(recent[0], r) for r in recent[1:])
        return all_same and mean_entropy < self.entropy_threshold

    def denoise_step(
        self,
        step: int,
        canvas: np.ndarray,
        prev_logits: np.ndarray | None = None,
    ) -> StepSnapshot:
        """Run one denoising step and return a snapshot."""
        # Self-conditioning from previous step
        self_cond = None
        gate = 0.0
        if prev_logits is not None:
            self._prev_softmax = self._logits_to_softmax(prev_logits)
            gate = min((step / max(self.max_steps - 1, 1)) * 1.5, 0.8)
            self_cond = self._prev_softmax

        logits = self._compute_logits(step, canvas, self_cond)
        softmax = self._logits_to_softmax(logits)
        entropy = self._per_position_entropy(softmax)

        accepted, budget_used = self.entropy_bound_accept(
            entropy, self.entropy_bound
        )

        argmax_canvas = np.argmax(softmax, axis=-1)
        mean_entropy = float(np.mean(entropy))

        new_canvas = self._renoise(canvas, accepted)

        self._gate_values.append(gate)

        return StepSnapshot(
            step=step,
            canvas=canvas.copy(),
            logits=logits,
            entropy=entropy,
            accepted_mask=accepted,
            argmax_canvas=argmax_canvas,
            converged=False,  # set later
            self_cond_gate=gate,
            cumulative_entropy=budget_used,
            budget_used=budget_used,
        )

    def run_full_block(self) -> BlockResult:
        """Run the full denoising loop for one block."""
        canvas = self.rng.integers(0, self.vocab_size, size=self.canvas_size)
        result = BlockResult(target_tokens=self.target.copy())
        argmax_history: list[np.ndarray] = []
        prev_logits: np.ndarray | None = None

        for step in range(self.max_steps):
            snapshot = self.denoise_step(step, canvas, prev_logits)

            argmax_history.append(snapshot.argmax_canvas)
            snapshot.converged = self.check_convergence(
                argmax_history, float(np.mean(snapshot.entropy))
            )

            result.steps.append(snapshot)
            result.num_accepted_per_step.append(int(np.sum(snapshot.accepted_mask)))

            canvas = self._renoise(snapshot.canvas, snapshot.accepted_mask)
            prev_logits = snapshot.logits

            if snapshot.converged:
                break

        # Final committed tokens are the argmax of the last step
        if result.steps:
            result.committed_tokens = result.steps[-1].argmax_canvas.copy()
        return result


# ---------------------------------------------------------------------------
# Attention mask generators
# ---------------------------------------------------------------------------

def causal_mask(seq_len: int) -> np.ndarray:
    """Upper-triangular causal mask (True = masked)."""
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


def bidirectional_mask(seq_len: int) -> np.ndarray:
    """Fully unmasked (nothing is masked)."""
    return np.zeros((seq_len, seq_len), dtype=bool)


def sliding_window_mask(seq_len: int, window: int) -> np.ndarray:
    """Causal with a symmetric sliding window of radius *window*."""
    mask = np.ones((seq_len, seq_len), dtype=bool)
    for i in range(seq_len):
        lo = max(0, i - window)
        hi = min(seq_len, i + window + 1)
        mask[i, lo:hi] = False
    return mask


def per_seq_causal_mask(lengths: list[int]) -> np.ndarray:
    """Block-diagonal causal mask for a batch of sequences.
    Returns a single large (sum(lengths) x sum(lengths)) mask."""
    total = sum(lengths)
    mask = np.ones((total, total), dtype=bool)
    offset = 0
    for L in lengths:
        mask[offset : offset + L, offset : offset + L] = causal_mask(L)
        offset += L
    return mask
