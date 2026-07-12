# Canvas Denoising Improvements

## Goal

Turn the canvas-denoising page into a faithful, inspectable tutorial of one
DiffusionGemma block. The page should make the distinction between a denoising
**pass** and the operations performed **inside each pass** explicit, demonstrate
non-causal parallel refinement, and clearly identify which values are simulated.

The tutorial must remain lightweight and deterministic. It does not need to run
the 26B-parameter model, but its control flow should match the documented
DiffusionGemma sampling algorithm.

## 1. Separate Pass and Stage Navigation

The current phase strip incorrectly maps the five algorithm stages across the
full sequence of denoising passes. In reality, every pass performs every stage.

Add two independent navigation dimensions:

- **Pass:** `0..N`, where pass 0 is canvas initialization and passes `1..N` are
  complete decoder passes.
- **Stage:** input canvas, model prediction, Gumbel-max sample, entropy-bound
  acceptance/re-noising, and self-conditioning output.

Recommended state:

```python
st.session_state.canvas_pass: int
st.session_state.canvas_stage: Literal[
    "input", "predict", "sample", "accept", "self_condition"
]
```

The main canvas should change with the selected stage rather than showing all
intermediate values only in tabs. Previous/next pass controls should preserve
the selected stage. A separate stage stepper should explain the transformation
being displayed.

Acceptance criteria:

- Every denoising pass exposes all five stages.
- Moving between stages does not change the selected pass.
- Pass 0 exposes initialization only and disables irrelevant stages.
- Browser reruns do not desynchronize buttons and sliders.

## 2. Implement Actual Convergence Decisions

The tutorial currently commits only because it reaches the configured step cap.
Implement the documented convergence rule:

```python
stable = all(
    np.array_equal(recent_argmax[0], candidate)
    for candidate in recent_argmax[1:]
)
confident = mean_entropy < entropy_threshold
converged = stable and confident
should_commit = converged or step == max_steps - 1
```

Expose `stability_steps` and `entropy_threshold` as advanced controls, with
reasonable defaults such as two stable passes and `0.1` mean entropy. Store the
reason for termination in the result:

```python
termination_reason: Literal["converged", "step_cap"]
```

The UI should display two live convergence checks:

- Argmax stable for `x / required` consecutive comparisons.
- Mean entropy `value < threshold`.

The commit message must distinguish natural convergence from a forced step-cap
commit. Do not describe a step-cap result as converged.

Acceptance criteria:

- The run can terminate before `max_steps`.
- Changing temperature, seed, or thresholds can change the termination pass.
- The UI states exactly why the block committed.
- The clean argmax canvas, not the noisy carried canvas, is committed.

## 3. Produce an Exact 256-Token Continuation

The scripted story currently exceeds the canvas and is cut at token 256. Replace
it with a continuation that has a natural ending before the block boundary.

Token construction should be explicit:

```python
story_tokens = tokenize(STORY)
assert len(story_tokens) < CANVAS_SIZE
target = story_tokens + ["<eos>"]
target += ["<pad>"] * (CANVAS_SIZE - len(target))
assert len(target) == CANVAS_SIZE
```

Because the tutorial tokenizer is whitespace-based rather than Gemma's real
SentencePiece tokenizer, label counts as **tutorial tokens**. Keep `CANVAS_SIZE`
as a named constant and add a test that prevents story edits from silently
truncating.

Render `<eos>` as an explicit end marker and visually mute `<pad>` positions.
The readable continuation should stop at `<eos>` and never print padding.

Acceptance criteria:

- The story ends grammatically before position 256.
- Exactly one `<eos>` appears in the target.
- Every position after `<eos>` is `<pad>`.
- No slice operation silently truncates target text.

## 4. Add a Position Inspector

Allow the user to choose a canvas position using a number input or by clicking a
token if Streamlit component support makes that practical. Clicking is optional;
the numeric selector is the reliable baseline.

For the selected position, show:

- Input token carried into this pass.
- Top five predicted tokens and probabilities.
- Argmax token used for convergence.
- Gumbel-max sampled candidate.
- Entropy in nats and confidence rank among all 256 positions.
- Whether the candidate fit under the cumulative entropy budget.
- Output token passed to the next iteration.
- Whether that output was accepted or freshly randomized.

The simulator must retain either the full probability matrix per pass or the
top-k values required by the inspector. Prefer storing top-k token IDs and
probabilities to reduce cached result size:

```python
top_token_ids: np.ndarray       # shape: (256, 5)
top_probabilities: np.ndarray   # shape: (256, 5)
confidence_rank: np.ndarray     # shape: (256,)
```

Also store the random replacement separately from the sampled candidate so the
UI never conflates sampling with re-noising.

## 5. Add Purposeful Playback

Add an optional play mode that advances denoising passes at a configurable delay.
Playback should not continuously animate every token or add decorative motion.
Each transition should highlight information that changed:

- Newly correct argmax positions.
- Argmax positions that changed since the previous pass.
- Accepted candidates.
- Re-noised positions.
- Newly padded positions after the model predicts the end marker.

Precompute boolean masks on `TutorialStep`:

```python
newly_correct: np.ndarray
argmax_changed: np.ndarray
accepted: np.ndarray
renoised: np.ndarray
```

Use a speed control between roughly 250 and 1500 milliseconds. Playback must
stop at the actual commit pass and remain fully usable with reduced-motion
preferences. Manual navigation is the primary interaction and must work without
playback.

## 6. Make Parallel Emergence Obvious

The visual should directly refute the assumption that generation proceeds from
left to right.

Add a canvas color mode selector:

- Acceptance state.
- Entropy/confidence heatmap.
- Newly correct this pass.
- Changed since previous pass.

Show position indices in tooltips and optionally every sixteenth cell in the
grid. Add a compact scatter plot with canvas position on the x-axis and the pass
where that position's argmax first became stable on the y-axis. A non-monotonic
pattern makes parallel, scattered convergence visible.

Do not define correctness as part of the real algorithm. “Correct” is available
only because the simulator has a scripted target and must be labeled as a
tutorial-only diagnostic. The actual convergence check uses argmax stability
and entropy, not access to a target answer.

## 7. Visualize Attention Context for One Position

Add an “Attention context” section connected to the position inspector. It
should explain the decoder/denoise pass for a selected query position:

- The prefilled prompt and prior committed blocks are readable from the KV cache.
- All 256 positions in the current canvas attend bidirectionally to each other.
- Denoising does not advance the KV-cache position.
- The final causal encoder/commit pass writes the clean block to the KV cache.

A simplified attention diagram is sufficient; do not invent attention weights.
Use a binary reachability mask unless real weights are available. Visually
separate cached context from the current canvas, and highlight the selected query
and all keys it may read.

If sliding-window behavior is included, make it an explicit optional mode and
reuse the mask helpers in `utils/diffusion_sim.py`. Avoid implying that every
layer necessarily has unrestricted global attention.

## 8. Support Editable Prompts Through Presets

Provide several curated scenarios rather than pretending the simulator can
understand arbitrary prompts:

- Story continuation.
- Fill-in-the-middle sentence.
- Short planning response.
- Small code completion.

Represent each scenario as structured data:

```python
@dataclass(frozen=True)
class TutorialScenario:
    name: str
    prompt: str
    continuation: str
    explanation: str
    seed: int
```

The prompt field may be editable for experimentation, but changing arbitrary
text must show a notice that the scripted probability generator is not a
language model and will still converge toward the selected preset continuation.
Do not claim that the output is conditioned semantically on custom input.

Changing scenarios should reset pass/stage navigation and invalidate only the
relevant cached simulation. Include the scenario identifier and edited prompt
in the cache key so stale runs cannot be displayed under a different prompt.

## Simulator Data Model

Extend `TutorialStep` rather than adding page-local calculations. A suggested
shape is:

```python
@dataclass(frozen=True)
class TutorialStep:
    step: int
    input_canvas: tuple[str, ...]
    argmax_tokens: tuple[str, ...]
    sampled_tokens: tuple[str, ...]
    random_replacements: tuple[str | None, ...]
    output_canvas: tuple[str, ...]
    entropy: np.ndarray
    accepted: np.ndarray
    confidence_rank: np.ndarray
    top_token_ids: np.ndarray
    top_probabilities: np.ndarray
    newly_correct: np.ndarray
    argmax_changed: np.ndarray
    mean_entropy: float
    stable_comparisons: int
    converged: bool
    self_conditioning: float
    budget_used: float
```

`TutorialRun` should additionally contain the scenario, effective step count,
termination reason, EOS position, and the configured convergence thresholds.

## Testing

Add unit tests for the simulator and in-process Streamlit interaction tests.
The minimum test matrix should cover:

- Deterministic output for identical seed and controls.
- Different seeds producing different initial canvases.
- Target length, EOS, and padding invariants.
- Entropy acceptance never exceeding its shared budget.
- Rejected positions receiving explicit random replacements.
- Confidence ranks being a permutation of `0..255`.
- Top-k probabilities sorted in descending order.
- Natural convergence and step-cap termination paths.
- Commit output matching the final argmax canvas.
- Previous/next, slider, stage selector, scenario switch, and jump-to-commit UI
  interactions without Streamlit state exceptions.

Use `streamlit.testing.v1.AppTest` for page-level tests so validation does not
require opening a server socket.

## Suggested Implementation Order

1. Fix exact target sizing and add simulator invariant tests.
2. Add convergence state and termination reasons.
3. Refactor the UI into independent pass and stage navigation.
4. Retain top-k and re-noising data, then add the position inspector.
5. Add parallel-emergence color modes and the stabilization scatter plot.
6. Add the attention-context diagram.
7. Add curated scenarios and editable-prompt disclosure.
8. Add playback last, after navigation and state transitions are stable.

This order prioritizes algorithmic correctness before adding richer interaction.
