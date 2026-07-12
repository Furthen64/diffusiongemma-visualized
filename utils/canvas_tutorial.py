"""Deterministic, prompt-conditioned simulation for the canvas tutorial.

This is intentionally not a language model. It produces plausible distributions
around a scripted continuation so the real sampling loop can be inspected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PROMPT = "The story about a squirrel starts with"

STORY = """a tiny red squirrel named Pip waking before sunrise in the oldest oak of Bramble Wood. A silver acorn lay beside his nest, although he was certain it had not been there the night before. When he touched it, the acorn gave a warm little hum and pointed, like a compass, toward the northern trees. Pip packed three hazelnuts, tied on his green scarf, and followed.

The forest was still blue with morning when he reached a stream. The usual stepping stones had vanished beneath rushing water. Pip noticed reeds bending together at the bank, so he wove them into a narrow bridge. Halfway across, he heard a frightened squeak. A field mouse clung to a wet branch below. Pip lowered his scarf, pulled her up, and together they crossed. Her name was Moss, and she knew the path north.

Beyond the stream they found a clearing where every bird was silent. At its center stood a hollow tree with a door no bigger than Pip's paw. The silver acorn hummed again. Pip placed it in a round notch, and the door opened onto a chamber full of golden seeds. An old owl explained that the seeds held the forest's spring songs, stolen and hidden by the winter wind.

Pip could have carried only one seed home, but Moss found an abandoned walnut shell. They filled it, pushed it uphill, and scattered the songs from the ridge. At once, finches, thrushes, and wrens began to sing. Buds opened across Bramble Wood. Pip returned to his oak at sunset, tired and muddy, with one ordinary acorn in his pocket and a new friend beside him. From then on, whenever the forest faced a puzzle, its smallest neighbors remembered that courage could begin with one careful step."""

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
    temperature: float = 1.0, seed: int = 7,
) -> TutorialRun:
    """Run the educational denoising loop and retain every intermediate value."""
    target = _tokens(STORY)
    if len(target) < canvas_size:
        target.extend(["<eos>"] + ["<pad>"] * (canvas_size - len(target) - 1))
    target = target[:canvas_size]
    vocabulary = list(dict.fromkeys(target + DISTRACTORS + ["<eos>", "<pad>"]))
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    target_ids = np.array([token_to_id[token] for token in target])
    vocab_size = len(vocabulary)
    rng = np.random.default_rng(seed)
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
        prompt=PROMPT,
        target=tuple(target),
        vocabulary=tuple(vocabulary),
        initial_canvas=initial,
        steps=tuple(steps),
        entropy_budget=entropy_budget,
    )
