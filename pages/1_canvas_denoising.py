import streamlit as st
import numpy as np

from utils.styles import inject_styles, COLORS
from utils.diffusion_sim import DiffusionSim
from utils.plot_helpers import entropy_bar_chart

st.set_page_config(page_title="Canvas Denoising", page_icon="🧊", layout="wide")
inject_styles()

st.title("🧊 Canvas Denoising")
st.markdown(
    "DiffusionGemma initializes a canvas of random tokens and iteratively "
    "refines them.  Watch the denoising process step by step."
)

# --- Sidebar controls ---
st.sidebar.markdown("### Controls")
canvas_size = st.sidebar.select_slider(
    "Canvas size", options=[8, 12, 16, 24, 32], value=16
)
max_steps = st.sidebar.slider("Max denoising steps", 1, 25, 12)
entropy_bound = st.sidebar.slider("Entropy budget", 0.01, 1.0, 0.15, 0.01)
temperature = st.sidebar.slider("Temperature", 0.1, 3.0, 1.0, 0.1)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

# --- Run simulation ---
sim = DiffusionSim(
    canvas_size=canvas_size,
    entropy_bound=entropy_bound,
    max_steps=max_steps,
    temperature=temperature,
    seed=seed,
)
result = sim.run_full_block()
total_steps = len(result.steps)

# --- Step navigation ---
st.markdown("### Step Navigation")
step_idx = st.slider("Step", 0, total_steps - 1, 0)
snapshot = result.steps[step_idx]

# --- Status badge ---
if snapshot.converged:
    badge = '<span class="status-badge badge-converged">CONVERGED</span>'
else:
    badge = '<span class="status-badge badge-noisy">DENOISING</span>'
st.markdown(f"**Step {snapshot.step + 1}/{total_steps}** {badge}", unsafe_allow_html=True)

# --- Accepted count ---
n_accepted = int(np.sum(snapshot.accepted_mask))
st.markdown(
    f"Accepted **{n_accepted}/{canvas_size}** positions "
    f"(accumulated entropy: {snapshot.cumulative_entropy:.4f} / "
    f"budget: {entropy_bound:.4f})"
)

# --- Main visualizations ---
left, right = st.columns([3, 2])

with left:
    st.markdown("#### Canvas State")
    tokens = sim.canvas_tokens(snapshot.canvas)
    grid_cols = min(canvas_size, 8)
    grid_rows = max(1, canvas_size // grid_cols)
    for r in range(grid_rows):
        row_html = ""
        for c in range(grid_cols):
            idx = r * grid_cols + c
            if idx >= canvas_size:
                break
            tok = tokens[idx]
            if snapshot.accepted_mask[idx]:
                bg = COLORS["confident"]
                fg = "#000"
            else:
                bg = COLORS["noisy"]
                fg = "#fff"
            row_html += (
                f'<span style="display:inline-block;padding:4px 8px;margin:2px;'
                f'border-radius:4px;background:{bg};color:{fg};'
                f'font-family:monospace;font-size:0.9rem;">{tok}</span>'
            )
        st.markdown(row_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Target (hidden)")
    target_tokens = sim.canvas_tokens(result.target_tokens)
    row_html = ""
    for i, tok in enumerate(target_tokens):
        row_html += (
            f'<span style="display:inline-block;padding:4px 8px;margin:2px;'
            f'border-radius:4px;background:{COLORS["neutral"]};color:#000;'
            f'font-family:monospace;font-size:0.9rem;">{tok}</span>'
        )
    st.markdown(row_html, unsafe_allow_html=True)

    st.markdown("#### Argmax Prediction")
    argmax_tokens = sim.canvas_tokens(snapshot.argmax_canvas)
    row_html = ""
    for i, tok in enumerate(argmax_tokens):
        match = tok == target_tokens[i]
        bg = COLORS["confident"] if match else COLORS["noisy"]
        fg = "#000"
        row_html += (
            f'<span style="display:inline-block;padding:4px 8px;margin:2px;'
            f'border-radius:4px;background:{bg};color:{fg};'
            f'font-family:monospace;font-size:0.9rem;">{tok}</span>'
        )
    st.markdown(row_html, unsafe_allow_html=True)

with right:
    st.plotly_chart(
        entropy_bar_chart(
            snapshot.entropy,
            budget=entropy_bound,
            accepted=snapshot.accepted_mask,
            title=f"Entropy per Position (Step {snapshot.step + 1})",
        ),
        width="stretch",
    )

    # Accepted count over all steps
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(total_steps)),
            y=result.num_accepted_per_step,
            mode="lines+markers",
            marker_color=COLORS["encoder"],
            fill="tozeroy",
            fillcolor=f"rgba(52,152,219,0.2)",
        )
    )
    fig.add_vline(
        x=step_idx,
        line_dash="dash",
        line_color=COLORS["denoise"],
    )
    fig.update_layout(
        title="Accepted Positions per Step",
        xaxis_title="Step",
        yaxis_title="Positions accepted",
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig, width="stretch")


