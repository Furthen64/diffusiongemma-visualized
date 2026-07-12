import streamlit as st
import numpy as np

from utils.styles import inject_styles, COLORS
from utils.diffusion_sim import DiffusionSim
from utils.plot_helpers import entropy_bar_chart, acceptance_curve

st.set_page_config(page_title="Entropy-Bound Acceptance", page_icon="📊", layout="wide")
inject_styles()

st.title("📊 Entropy-Bound Acceptance")
st.markdown(
    "Each denoising step produces a probability distribution over the vocabulary "
    "for every position.  The model accepts positions from most to least confident "
    "until accumulated entropy exceeds a fixed budget."
)

# --- Sidebar ---
st.sidebar.markdown("### Controls")
num_positions = st.sidebar.slider("Number of positions", 8, 32, 16)
entropy_budget = st.sidebar.slider("Entropy budget", 0.01, 2.0, 0.5, 0.01)
temperature = st.sidebar.slider("Temperature", 0.1, 3.0, 1.0, 0.1)
step_num = st.sidebar.slider("Denoising step", 0, 15, 5)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

# --- Run a single-step simulation ---
sim = DiffusionSim(
    canvas_size=num_positions,
    entropy_bound=entropy_budget,
    max_steps=step_num + 1,
    temperature=temperature,
    seed=seed,
)
result = sim.run_full_block()
snapshot = result.steps[min(step_num, len(result.steps) - 1)]

# --- Accepted info ---
n_accepted = int(np.sum(snapshot.accepted_mask))
st.markdown(
    f"**Accepted: {n_accepted}/{num_positions}** positions  |  "
    f"Accumulated entropy: **{snapshot.cumulative_entropy:.4f}**  |  "
    f"Budget: **{entropy_budget:.4f}**"
)

# --- Main layout ---
left, right = st.columns(2)

with left:
    st.plotly_chart(
        entropy_bar_chart(
            snapshot.entropy,
            budget=entropy_budget,
            accepted=snapshot.accepted_mask,
            title="Entropy per Position (sorted by index)",
        ),
        width="stretch",
    )

with right:
    st.plotly_chart(
        acceptance_curve(snapshot.entropy, entropy_budget),
        width="stretch",
    )

# --- Before / After grid ---
st.markdown("---")
st.markdown("#### Before → After")
col_before, col_arrow, col_after = st.columns([5, 1, 5])

def render_grid(tokens, accepted, highlight=False):
    html = ""
    for i, tok in enumerate(tokens):
        if highlight and accepted is not None:
            bg = COLORS["accepted"] if accepted[i] else COLORS["rejected"]
        elif highlight:
            bg = COLORS["neutral"]
        else:
            bg = COLORS["masked"]
        fg = "#000" if bg in (COLORS["accepted"], COLORS["neutral"]) else "#fff"
        html += (
            f'<span style="display:inline-block;padding:6px 10px;margin:3px;'
            f'border-radius:4px;background:{bg};color:{fg};'
            f'font-family:monospace;font-size:0.95rem;">{tok}</span>'
        )
    return html

with col_before:
    st.markdown("**Current canvas**")
    st.markdown(
        render_grid(sim.canvas_tokens(snapshot.canvas), None),
        unsafe_allow_html=True,
    )

with col_arrow:
    st.markdown("<br><br><h2>→</h2>", unsafe_allow_html=True)

with col_after:
    st.markdown("**After accept/reject**")
    new_canvas = sim._renoise(snapshot.canvas, snapshot.accepted_mask)
    st.markdown(
        render_grid(sim.canvas_tokens(new_canvas), snapshot.accepted_mask, highlight=True),
        unsafe_allow_html=True,
    )

# --- Explanation ---
st.markdown("---")
with st.expander("How does entropy-bound acceptance work?"):
    st.markdown("""
    1. The model predicts a probability distribution for each canvas position.
    2. **Entropy** measures uncertainty — low entropy means the model is confident.
    3. Positions are sorted from most to least confident.
    4. Positions are accepted one by one until the cumulative entropy exceeds the budget.
    5. Rejected positions are replaced with fresh random tokens for the next step.
    6. Over several steps, accepted positions provide context that helps neighboring
       positions become more confident, causing the block to "snap into focus."
    """)
