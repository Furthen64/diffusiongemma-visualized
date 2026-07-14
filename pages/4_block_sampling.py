import streamlit as st
import numpy as np

from utils.glossary import glossary_link
from utils.navigation import render_learning_path
from utils.styles import inject_styles, COLORS, render_description
from utils.diffusion_sim import DiffusionSim
from utils.plot_helpers import timeline_gantt, kv_cache_bar

st.set_page_config(page_title="Block Sampling Loop", page_icon="🔗", layout="wide")
inject_styles()

st.title("🔗 Block Sampling Loop")
render_description(
    """
    This page zooms out from one denoising pass to the full generation loop.
    DiffusionGemma works block by block: prefill existing context, denoise the
    current block for several steps, commit the finished block, then repeat.

    Use the timeline to see the phase order. The block detail cards show how
    each block converges before it is appended to the running text and added to
    the KV cache for future blocks.
    """,
    references=(
        f"{glossary_link('Block sampling', 'Block sampling')} · "
        f"{glossary_link('Prefill', 'Prefill')} · "
        f"{glossary_link('Denoising', 'Denoising')} · "
        f"{glossary_link('Commit', 'Commit')} · "
        f"{glossary_link('KV cache', 'KV cache')} · "
        f"{glossary_link('Convergence', 'Convergence')} · "
        f"{glossary_link('Argmax', 'Argmax')}"
    ),
)
render_learning_path("pages/4_block_sampling.py")

# --- Sidebar ---
st.sidebar.markdown("### Controls")
num_blocks = st.sidebar.slider("Number of blocks", 1, 5, 3)
tokens_per_block = st.sidebar.select_slider("Tokens per block (viz)", options=[8, 12, 16], value=12)
prompt_len = st.sidebar.slider("Prompt length", 0, 20, 8)
max_denoise_steps = st.sidebar.slider("Max denoising steps", 3, 15, 8)
entropy_bound = st.sidebar.slider("Entropy budget", 0.01, 1.0, 0.2, 0.01)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

# --- Simulate all blocks ---
all_blocks = []
rng = np.random.default_rng(seed)
phases = []
current_time = 0.0
kv_committed = []
total_kv = prompt_len

# Prefill
phases.append({
    "task": "Prefill prompt",
    "start": 0.0,
    "finish": 1.0,
    "color": COLORS["prefill"],
    "block": "prefill",
})
current_time = 1.0

for b in range(num_blocks):
    sim = DiffusionSim(
        canvas_size=tokens_per_block,
        entropy_bound=entropy_bound,
        max_steps=max_denoise_steps,
        seed=int(rng.integers(0, 2**31)),
    )
    result = sim.run_full_block()
    all_blocks.append(result)

    n_denoise = len(result.steps)
    phases.append({
        "task": f"Block {b} — Denoise",
        "start": current_time,
        "finish": current_time + n_denoise,
        "color": COLORS["denoise"],
        "block": f"block_{b}",
    })
    current_time += n_denoise

    phases.append({
        "task": f"Block {b} — Commit",
        "start": current_time,
        "finish": current_time + 1,
        "color": COLORS["commit"],
        "block": f"commit_{b}",
    })
    current_time += 1
    kv_committed.append(tokens_per_block)
    total_kv += tokens_per_block

# --- Timeline ---
st.plotly_chart(
    timeline_gantt(phases),
    width="stretch",
)

# --- KV Cache ---
st.plotly_chart(
    kv_cache_bar(kv_committed, total_kv),
    width="stretch",
)

# --- Block detail ---
st.markdown("---")
st.markdown("### Block Details")

selected_block = st.selectbox(
    "Select a block to inspect",
    range(num_blocks),
    format_func=lambda i: f"Block {i}",
)

if all_blocks:
    block = all_blocks[selected_block]
    col_info, col_grid = st.columns([1, 2])

    with col_info:
        st.markdown(glossary_link("Denoising steps", "Denoising"), unsafe_allow_html=True)
        st.metric("Denoising steps value", len(block.steps), label_visibility="collapsed")
        converged = block.steps[-1].converged if block.steps else False
        st.markdown(glossary_link("Converged", "Convergence"), unsafe_allow_html=True)
        st.metric("Converged value", "Yes" if converged else "No (hit step limit)", label_visibility="collapsed")

        # Acceptance over steps
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(block.num_accepted_per_step))),
                y=block.num_accepted_per_step,
                mode="lines+markers",
                marker_color=COLORS["confident"],
                fill="tozeroy",
                fillcolor="rgba(46,204,113,0.2)",
            )
        )
        fig.update_layout(
            title=f"Block {selected_block} — Accepted per step",
            xaxis_title="Step",
            yaxis_title="Positions accepted",
            height=200,
            margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
            xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333", range=[0, tokens_per_block]),
        )
        st.plotly_chart(fig, width="stretch")

    with col_grid:
        # Show first, middle, and last step
        steps_to_show = [0]
        if len(block.steps) > 2:
            steps_to_show.append(len(block.steps) // 2)
        if len(block.steps) > 1:
            steps_to_show.append(len(block.steps) - 1)

        for si in steps_to_show:
            snap = block.steps[si]
            st.markdown(f"**Step {snap.step + 1}**")
            tokens = sim.canvas_tokens(snap.canvas)
            html = ""
            for i, tok in enumerate(tokens):
                if snap.accepted_mask[i]:
                    bg = COLORS["accepted"]
                    fg = "#000"
                else:
                    bg = COLORS["rejected"]
                    fg = "#fff"
                html += (
                    f'<span style="display:inline-block;padding:4px 8px;margin:2px;'
                    f'border-radius:4px;background:{bg};color:{fg};'
                    f'font-family:monospace;font-size:0.85rem;">{tok}</span>'
                )
            st.markdown(html, unsafe_allow_html=True)

# --- Assembled text ---
st.markdown("---")
st.markdown("### Assembled Text (with block boundaries)")
all_tokens = []
for i, block in enumerate(all_blocks):
    committed = sim.canvas_tokens(block.committed_tokens)
    all_tokens.append((i, committed))

prompt_tokens = sim.canvas_tokens(rng.integers(0, sim.vocab_size, size=prompt_len))

html = '<span style="color:#95a5a6;font-family:monospace;">'
for tok in prompt_tokens:
    html += f'<span style="padding:3px 6px;margin:1px;border-radius:3px;background:#333;color:#999;">{tok}</span>'
html += '</span>'

for block_idx, tokens in all_tokens:
    color = COLORS["commit"]
    html += (
        f'<span style="display:inline-block;padding:3px 6px;margin:1px;'
        f'border-radius:3px;border:2px dashed {color};'
        f'color:{color};font-family:monospace;font-size:0.8rem;">'
        f"[Block {block_idx}]</span>"
    )
    for tok in tokens:
        html += (
            f'<span style="display:inline-block;padding:3px 6px;margin:1px;'
            f'border-radius:3px;background:{COLORS["commit"]}22;'
            f'color:{COLORS["text"]};font-family:monospace;">{tok}</span>'
        )

st.markdown(html, unsafe_allow_html=True)

# --- Explanation ---
with st.expander("How does the block sampling loop work?"):
    st.markdown(
        f"""
    1. **{glossary_link('Prefill', 'Prefill')}**: The prompt is encoded with {glossary_link('causal attention', 'Causal attention')} and written to the {glossary_link('KV cache', 'KV cache')}.
    2. **{glossary_link('Denoise', 'Denoising')}**: A fresh 256-token {glossary_link('canvas', 'Canvas')} of random tokens is initialized. The model runs
       multiple denoising steps with {glossary_link('bidirectional attention', 'Bidirectional attention')}, iteratively refining the canvas.
    3. **{glossary_link('Commit', 'Commit')}**: Once {glossary_link('converged', 'Convergence')} (or the step limit is hit), the clean {glossary_link('argmax', 'Argmax')} prediction is
       encoded with causal attention and appended to the KV cache.
    4. **Next block**: A fresh canvas is initialized conditioned on all previously committed
       tokens, and the process repeats.
    5. Across blocks, generation is still left-to-right ({glossary_link('autoregressive', 'Autoregressive')}). Within each block,
       all 256 positions denoise in parallel (diffusion).
    """,
        unsafe_allow_html=True,
    )
