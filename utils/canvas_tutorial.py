"""Deterministic, prompt-conditioned simulation for the canvas tutorial.

This is intentionally not a language model. It produces plausible distributions
around a scripted continuation so the real sampling loop can be inspected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TutorialScenario:
    name: str
    prompt: str
    continuation: str
    seed_offset: int


SCENARIOS = {
    "creative": TutorialScenario(
        name="Creative writing",
        prompt="The story about a squirrel starts with",
        continuation="""a tiny red squirrel named Pip waking before sunrise in the oldest oak of Bramble Wood. A silver acorn lay beside his nest, although he was certain it had not been there the night before. When he touched it, the acorn gave a warm hum and pointed like a compass toward the northern trees. Pip packed three hazelnuts, tied on his green scarf, and followed.

At the river he found the stepping stones hidden beneath rushing water. Pip wove fallen reeds into a narrow bridge, then stopped halfway to rescue a field mouse clinging to a branch. Her name was Moss, and she knew where the silver acorn wanted to go.

Together they reached a hollow tree whose little door opened only for the acorn. Inside waited a store of golden seeds holding every spring song in the forest. Pip and Moss rolled them home in a walnut shell and scattered them from the ridge. Birds sang, buds opened, and Pip learned that even the smallest traveler could carry a whole season home.""",
        seed_offset=0,
    ),
    "technical": TutorialScenario(
        name="Technical explanation",
        prompt="Explain why a database index makes queries faster:",
        continuation="""A database index is a separate data structure that maps selected column values to the rows containing them. Without an index, the database may need to inspect every row to answer a filter such as WHERE email = 'user@example.com'. This is a full table scan, and its cost grows roughly with the number of rows.

A common B-tree index keeps keys ordered in a balanced tree. The database follows a small number of branches to locate a key, so lookup cost grows logarithmically rather than linearly. Range queries also benefit because neighboring keys are stored in order. Hash indexes provide fast equality lookup but generally do not support ordered ranges.

Indexes are not free. They consume storage, must be updated whenever indexed data changes, and can slow inserts or writes. Query planners compare these costs with table-scan costs and choose an index only when it is likely to reduce total work. Effective indexing therefore focuses on columns used frequently for filtering, joining, and ordering, while avoiding redundant indexes.""",
        seed_offset=101,
    ),
    "legal": TutorialScenario(
        name="Contract analysis",
        prompt="Explain this contract clause in plain English: The agreement renews automatically for successive one-year terms unless either party gives 60 days' written notice.",
        continuation="""The contract continues for another year automatically when the current term ends. Either party can prevent renewal, but it must send written notice at least 60 days before the expiration date. If notice arrives later than that deadline, the agreement may already have renewed for the next one-year period.

In practice, each party should identify the exact end date, calculate the notice deadline, and confirm which delivery methods and addresses count as written notice under the agreement. It is also important to check whether another section permits termination after renewal, requires a particular notice format, or imposes an early-termination fee.

This clause does not require either party to negotiate a new agreement each year. Silence results in renewal. The practical risk is missing the notice window and remaining bound for an additional term. This plain-language summary is educational and is not a substitute for legal advice about a specific agreement or jurisdiction.""",
        seed_offset=202,
    ),
}

DISTRACTORS = (
    "the a an and or but then when while under over beside through toward away "
    "quiet bright dark quick slow rabbit fox badger moon river mountain garden "
    "castle bicycle cloud whispered wondered suddenly perhaps never always"
).split()


def _tokens(text: str) -> list[str]:
    return text.replace("\n", " ").split()


@dataclass(frozen=True)
class TutorialStep:
    step: int
    input_canvas: tuple[str, ...]
    argmax_tokens: tuple[str, ...]
    sampled_tokens: tuple[str, ...]
    output_canvas: tuple[str, ...]
    entropy: np.ndarray
    accepted: np.ndarray
    accepted_count: int
    budget_used: float
    changed_from_previous: int
    argmax_stable: bool
    mean_entropy: float
    self_conditioning: float


@dataclass(frozen=True)
class TutorialRun:
    prompt: str
    target: tuple[str, ...]
    vocabulary: tuple[str, ...]
    initial_canvas: tuple[str, ...]
    steps: tuple[TutorialStep, ...]
    entropy_budget: float


def run_tutorial(
    *, canvas_size: int = 256, num_steps: int = 12, entropy_budget: float = 32.0,
    temperature: float = 1.0, seed: int = 7, scenario_id: str = "creative",
) -> TutorialRun:
    """Run the educational denoising loop and retain every intermediate value."""
    scenario = SCENARIOS[scenario_id]
    target = _tokens(scenario.continuation)
    if len(target) >= canvas_size:
        raise ValueError(f"Scenario {scenario_id!r} must leave room for an <eos> token")
    target.extend(["<eos>"] + ["<pad>"] * (canvas_size - len(target) - 1))
    vocabulary = list(dict.fromkeys(target + DISTRACTORS + ["<eos>", "<pad>"]))
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    target_ids = np.array([token_to_id[token] for token in target])
    vocab_size = len(vocabulary)
    rng = np.random.default_rng(seed + scenario.seed_offset)
    canvas_ids = rng.integers(0, vocab_size, size=canvas_size)
    initial = tuple(vocabulary[index] for index in canvas_ids)
    # Positions become easy at different times, avoiding an artificial left-to-right reveal.
    reveal_at = rng.uniform(0.18, 0.82, size=canvas_size)
    previous_probs: np.ndarray | None = None
    previous_argmax: np.ndarray | None = None
    steps: list[TutorialStep] = []

    for step in range(num_steps):
        progress = (step + 1) / num_steps
        input_ids = canvas_ids.copy()
        logits = rng.normal(0.0, 0.75 * (1.0 - progress) + 0.12, (canvas_size, vocab_size))
        target_strength = 1.0 + 10.0 / (1.0 + np.exp(-(progress - reveal_at) * 13.0))
        logits[np.arange(canvas_size), target_ids] += target_strength

        self_conditioning = 0.0 if previous_probs is None else min(0.8, progress * 0.9)
        if previous_probs is not None:
            logits += previous_probs * (2.2 * self_conditioning)

        scaled = logits / max(temperature, 0.05)
        scaled -= scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled)
        probs /= probs.sum(axis=1, keepdims=True)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        argmax_ids = np.argmax(probs, axis=1)

        gumbel = -np.log(-np.log(rng.uniform(1e-9, 1.0 - 1e-9, probs.shape)))
        sampled_ids = np.argmax(np.log(probs + 1e-12) + gumbel, axis=1)
        accepted = np.zeros(canvas_size, dtype=bool)
        budget_used = 0.0
        for index in np.argsort(entropy):
            if budget_used + entropy[index] > entropy_budget:
                break
            accepted[index] = True
            budget_used += float(entropy[index])

        canvas_ids = sampled_ids.copy()
        rejected = ~accepted
        canvas_ids[rejected] = rng.integers(0, vocab_size, size=int(rejected.sum()))
        stable = previous_argmax is not None and np.array_equal(argmax_ids, previous_argmax)
        changed = canvas_size if previous_argmax is None else int(np.sum(argmax_ids != previous_argmax))
        steps.append(TutorialStep(
            step=step,
            input_canvas=tuple(vocabulary[index] for index in input_ids),
            argmax_tokens=tuple(vocabulary[index] for index in argmax_ids),
            sampled_tokens=tuple(vocabulary[index] for index in sampled_ids),
            output_canvas=tuple(vocabulary[index] for index in canvas_ids),
            entropy=entropy.copy(),
            accepted=accepted.copy(),
            accepted_count=int(accepted.sum()),
            budget_used=budget_used,
            changed_from_previous=changed,
            argmax_stable=stable,
            mean_entropy=float(entropy.mean()),
            self_conditioning=self_conditioning,
        ))
        previous_probs = probs
        previous_argmax = argmax_ids

    return TutorialRun(
        prompt=scenario.prompt,
        target=tuple(target),
        vocabulary=tuple(vocabulary),
        initial_canvas=initial,
        steps=tuple(steps),
        entropy_budget=entropy_budget,
    )
