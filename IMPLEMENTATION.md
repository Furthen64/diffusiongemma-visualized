# DiffusionGemma Visualization — Implementation Plan

## Overview

An interactive Streamlit application that visualizes all core concepts of
DiffusionGemma using simulated data.  Six visualization modules, each on its own
page, with a sidebar for navigation and shared utilities.

## Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Framework | Streamlit | Interactive widgets, fast prototyping |
| Charts | Plotly | Interactive heatmaps, bar charts, animations |
| Simulation | NumPy | Lightweight, no GPU required |
| Layout | Streamlit multipage | One `.py` per visualization page |
| Style | Custom CSS via `st.markdown` | Consistent look across pages |

## Project Structure

```
diffusiongemma-visualized/
├── app.py                       # Entry point, sidebar nav, landing page
├── pages/
│   ├── 1_canvas_denoising.py    # Visualization 1
│   ├── 2_entropy_bound.py       # Visualization 2
│   ├── 3_self_conditioning.py   # Visualization 3
│   ├── 4_block_sampling.py      # Visualization 4
│   ├── 5_encoder_decoder.py     # Visualization 5
│   └── 6_attention_mechanisms.py # Visualization 6
├── utils/
│   ├── __init__.py
│   ├── diffusion_sim.py         # Core simulation engine
│   ├── plot_helpers.py          # Reusable Plotly figure builders
│   └── styles.py                # Shared CSS / color palette
├── pyproject.toml
├── IMPLEMENTATION.md            # This file
└── README.md
```

## Simulation Engine (`utils/diffusion_sim.py`)

A single module all pages import.  Provides:

- `DiffusionSim` class — configurable with `canvas_size`, `vocab_size`,
  `entropy_bound`, `max_steps`.
- `initialize_canvas()` — random tokens from vocabulary.
- `denoise_step(canvas, self_cond_probs=None)` — one denoise pass producing
  logits (simulated via soft token-similarity), sampling tokens via Gumbel-max,
  computing per-position entropy, accepting/rejecting via entropy bound.
- `entropy_bound_accept(logits, budget)` — sort positions by confidence, accept
  until accumulated entropy exceeds budget.
- `self_conditioning_step(prev_logits)` — convert previous softmax to
  probability-weighted embeddings, pass through a gate.
- `check_convergence(argmax_history, mean_entropy, threshold)` — return converged
  bool.
- `run_full_block(canvas_size, entropy_bound, max_steps)` — return full
  step-by-step history as a list of dicts.
- Attention mask generators — `causal_mask`, `bidirectional_mask`,
  `sliding_window_mask`, `per_seq_causal_mask`.

All randomness is seeded via Streamlit `@st.cache_data` so the same parameters
produce the same simulation.

## Visualization Pages

### 1. Canvas Denoising (`pages/1_canvas_denoising.py`)

**Concept:** How a canvas goes from random noise to coherent text over denoising
steps.

**Controls:**
- Canvas size (8, 16, 32 tokens)
- Number of denoising steps (1–20) with play/pause/step buttons
- Entropy bound (0.05–1.0)
- "Run all" button with speed slider

**Main area:**
- Grid heatmap — each cell is a token position, colored by entropy (green =
  confident, red = noisy). Token text as annotations.
- Token text panel — actual text of current canvas state.
- Entropy bar chart — per-position entropy with horizontal budget line.
- Step counter — "Step 3/12" with forward/back buttons.
- Convergence indicator — green/red badge.

**Animation:** "Play" button auto-advances steps with configurable delay.

### 2. Entropy-Bound Acceptance (`pages/2_entropy_bound.py`)

**Concept:** Sorting positions by confidence and accepting until entropy budget
is exhausted.

**Controls:**
- Number of positions (8–32)
- Entropy budget (slider)
- Vocabulary size
- Temperature

**Main area:**
- Sorted entropy bar chart — positions sorted lowest→highest. Accepted = green,
  rejected = red. Vertical cutoff line at budget boundary.
- Acceptance curve — cumulative entropy vs position index.
- Before/After grid — canvas before step and after (accepted highlighted,
  rejected renoised).
- Info panel — "Accepted 14/32 positions. Accumulated entropy: 0.087 < 0.10."

### 3. Self-Conditioning (`pages/3_self_conditioning.py`)

**Concept:** Previous step's softmax feeds back into next step via gated MLP.

**Controls:**
- Step to visualize (1–10)
- Gate initial value (0.0–1.0)
- Show raw logits vs softmax

**Main area:**
- Flow diagram — annotated boxes: `Step N logits → Softmax → Weighted
  Embeddings → Gated MLP → + Canvas Embeddings → Step N+1 input`.
- Distribution comparison — side-by-side histograms of Step N softmax vs Step
  N+1 logits.
- Per-position influence heatmap.
- Gate value over steps — animated bar chart.

### 4. Block Sampling Loop (`pages/4_block_sampling.py`)

**Concept:** The high-level loop: Prefill → [Denoise → Accept/Commit] → Next
Block.

**Controls:**
- Number of blocks (1–5)
- Tokens per block (8–16)
- Prompt length

**Main area:**
- Timeline/Gantt chart — colored segments: blue = Prefill, orange = Denoising (N
  steps), green = Commit.
- KV Cache growth — horizontal bar filling as blocks are committed.
- Current block detail — click to expand denoising grid for that block.
- Long text assembly — final text with block boundaries highlighted.

### 5. Encoder/Decoder Mode Toggle (`pages/5_encoder_decoder.py`)

**Concept:** Same backbone in two modes: causal (encoder) and bidirectional
(decoder).

**Controls:**
- Mode toggle (Encoder / Decoder)
- Canvas size
- Show KV cache state

**Main area:**
- Mode diagram — large toggle visualization with attention mask shape shown.
- Attention mask heatmap — full (tokens × tokens) matrix. Encoder = upper
  triangle masked. Decoder = full.
- KV cache state — table showing read/write per token.
- Forward pass walkthrough — numbered steps for the current mode.

### 6. Attention Mechanisms (`pages/6_attention_mechanisms.py`)

**Concept:** Causal, bidirectional, and sliding window attention, including
per-sequence mixing in a batch.

**Controls:**
- Sequence lengths for up to 3 requests
- Window size
- Mode per request (prefill / denoise / commit)

**Main area:**
- Per-sequence attention masks — 3 heatmaps (one per request).
- Combined batch view — single large heatmap with block-diagonal structure.
- Sliding window detail — zoomed-in view of one query's attention span.
- Interactive query selector — click position to highlight its attention span.

## Color Palette

| Role | Color | Hex |
|------|-------|-----|
| Confident / Accepted | Green | `#2ecc71` |
| Noisy / Rejected | Red | `#e74c3c` |
| Encoder mode | Blue | `#3498db` |
| Decoder mode | Orange | `#e67e22` |
| Masked | Gray | `#95a5a6` |
| Neutral | Light gray | `#ecf0f1` |

## Implementation Order

| Phase | What | Depends on |
|-------|------|------------|
| 1 | Scaffold: `app.py`, `utils/`, `pyproject.toml` | — |
| 2 | Canvas Denoising (page 1) | Phase 1 |
| 3 | Entropy-Bound Acceptance (page 2) | Phase 1 |
| 4 | Self-Conditioning (page 3) | Phase 1 |
| 5 | Block Sampling Loop (page 4) | Pages 1–3 |
| 6 | Encoder/Decoder Mode (page 5) | Phase 1 |
| 7 | Attention Mechanisms (page 6) | Phase 1 |

## Dependencies

```toml
[project]
name = "diffusiongemma-visualized"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "streamlit>=1.35",
    "plotly>=5.20",
    "numpy>=1.26",
]
```

## Running

```bash
pip install -e .
streamlit run app.py
```
