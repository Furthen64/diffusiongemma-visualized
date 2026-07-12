import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils.styles import inject_styles, COLORS
from utils.diffusion_sim import DiffusionSim

st.set_page_config(page_title="Self-Conditioning", page_icon="🔄", layout="wide")
inject_styles()

st.title("🔄 Self-Conditioning")
st.markdown(
    "Between denoising steps, DiffusionGemma feeds its own previous softmax "
    "distribution back as input — through a gated MLP — giving each step a "
    "memory of what the model believed last time."
)

# --- Sidebar ---
st.sidebar.markdown("### Controls")
canvas_size = st.sidebar.select_slider("Canvas size", options=[8, 12, 16], value=12)
max_steps = st.sidebar.slider("Steps to run", 2, 15, 8)
entropy_bound = st.sidebar.slider("Entropy budget", 0.01, 1.0, 0.15, 0.01)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

# --- Run simulation ---
sim = DiffusionSim(
    canvas_size=canvas_size,
    entropy_bound=entropy_bound,
    max_steps=max_steps,
    seed=seed,
)
result = sim.run_full_block()

# --- Flow diagram ---
st.markdown("#### Self-Conditioning Flow")
flow_html = """
<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:1rem 0;">
  <span class="flow-stage" style="background:#3498db22;border:1px solid #3498db;color:#3498db;">
    Step N Logits
    <span class="flow-help" tabindex="0" aria-label="About Step N Logits"
          data-tooltip="The model's raw, unnormalized score for every possible token at each canvas position.">?</span>
  </span>
  <span style="color:#666;">→</span>
  <span class="flow-stage" style="background:#e67e2222;border:1px solid #e67e22;color:#e67e22;">
    Softmax
    <span class="flow-help" tabindex="0" aria-label="About Softmax"
          data-tooltip="Converts the raw logits into probabilities that add up to 1 for each canvas position.">?</span>
  </span>
  <span style="color:#666;">→</span>
  <span class="flow-stage" style="background:#2ecc7122;border:1px solid #2ecc71;color:#2ecc71;">
    Weighted Avg Embeddings
    <span class="flow-help" tabindex="0" aria-label="About Weighted Average Embeddings"
          data-tooltip="Blends token embeddings using those probabilities. Likely tokens contribute more, preserving uncertainty instead of choosing one token early.">?</span>
  </span>
  <span style="color:#666;">→</span>
  <span class="flow-stage" style="background:#e74c3c22;border:1px solid #e74c3c;color:#e74c3c;">
    Gated MLP (gate={:.2f})
    <span class="flow-help" tabindex="0" aria-label="About the Gated MLP"
          data-tooltip="Transforms the soft embedding. The gate controls how strongly the previous step can influence the next one; 0 means no influence.">?</span>
  </span>
  <span style="color:#666;">→</span>
  <span class="flow-stage" style="background:#ecf0f1;border:1px solid #95a5a6;color:#333;">
    + Canvas Embeddings
    <span class="flow-help" tabindex="0" aria-label="About Canvas Embeddings"
          data-tooltip="Adds the transformed memory from Step N to the embeddings of the current noisy canvas.">?</span>
  </span>
  <span style="color:#666;">→</span>
  <span class="flow-stage" style="background:#3498db22;border:1px solid #3498db;color:#3498db;">
    Step N+1 Input
    <span class="flow-help" tabindex="0" aria-label="About Step N plus 1 Input"
          data-tooltip="The next denoising step now receives both the current canvas and a soft memory of the previous prediction.">?</span>
  </span>
</div>
""".format(result.steps[-1].self_cond_gate if result.steps else 0)
st.markdown(flow_html, unsafe_allow_html=True)

# --- Gate value over steps ---
st.markdown("#### Gate Value Over Steps")
gate_vals = [s.self_cond_gate for s in result.steps]
step_numbers = [s.step + 1 for s in result.steps]
fig_gate = go.Figure()
fig_gate.add_trace(
    go.Scatter(
        x=step_numbers,
        y=gate_vals,
        mode="lines+markers",
        marker_color=COLORS["denoise"],
        fill="tozeroy",
        fillcolor="rgba(230,126,34,0.15)",
    )
)
fig_gate.update_layout(
    xaxis_title="Step",
    yaxis_title="Gate value",
    yaxis_range=[0, 1],
    height=250,
    margin=dict(l=40, r=20, t=20, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"]),
    xaxis=dict(gridcolor="#333"),
    yaxis=dict(gridcolor="#333"),
)
st.plotly_chart(fig_gate, width="stretch")

# --- Distribution comparison across steps ---
st.markdown("#### How Self-Conditioning Sharpens Distributions")
st.markdown(
    "Choose an earlier and a later denoising step. Both charts show the model's "
    "predictions for the **first token position**, so you can compare how its "
    "confidence changes over time."
)

step_a, step_b = st.columns(2)
with step_a:
    selected_step_a = st.slider(
        "Earlier step",
        1,
        len(result.steps),
        1,
        key="self_conditioning_earlier_step",
    )
with step_b:
    selected_step_b = st.slider(
        "Later step",
        1,
        len(result.steps),
        min(4, len(result.steps)),
        key="self_conditioning_later_step",
    )

idx_a = selected_step_a - 1
idx_b = selected_step_b - 1
snap_a = result.steps[idx_a]
snap_b = result.steps[idx_b]

# Show average softmax distribution for position 0
softmax_a = sim._logits_to_softmax(snap_a.logits)
softmax_b = sim._logits_to_softmax(snap_b.logits)

# Keep the token set and x-axis order fixed as the sliders move. Previously,
# the tokens were selected and sorted by Step A's probabilities, which made
# Step A look artificially smooth and Step B look randomly ordered.
top_k = min(15, sim.vocab_size)
final_softmax = sim._logits_to_softmax(result.steps[-1].logits)
top_indices = np.argsort(final_softmax[0])[-top_k:]
top_indices = np.array(sorted(top_indices, key=lambda idx: sim.vocab[idx]))
shared_y_max = max(
    float(np.max(softmax_a[0, top_indices])),
    float(np.max(softmax_b[0, top_indices])),
)
shared_y_range = [0, max(shared_y_max * 1.1, 0.05)]

st.caption(
    "Both charts use the same tokens, selected from the final step and ordered "
    "alphabetically. Their probability scales are identical."
)

col_chart_a, col_chart_b = st.columns(2)
with col_chart_a:
    fig_a = go.Figure()
    fig_a.add_trace(
        go.Bar(
            x=[sim.vocab[i] for i in top_indices],
            y=[softmax_a[0, i] for i in top_indices],
            marker_color=COLORS["encoder"],
        )
    )
    fig_a.update_layout(
        title=f"Step {selected_step_a} — First token position",
        yaxis_title="Probability",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333", range=shared_y_range),
    )
    st.plotly_chart(fig_a, width="stretch")

with col_chart_b:
    fig_b = go.Figure()
    fig_b.add_trace(
        go.Bar(
            x=[sim.vocab[i] for i in top_indices],
            y=[softmax_b[0, i] for i in top_indices],
            marker_color=COLORS["denoise"],
        )
    )
    fig_b.update_layout(
        title=f"Step {selected_step_b} — First token position",
        yaxis_title="Probability",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333", range=shared_y_range),
    )
    st.plotly_chart(fig_b, width="stretch")

# --- Entropy reduction over steps ---
st.markdown("---")
st.markdown("#### Mean Entropy Over Steps")
mean_entropies = [float(np.mean(s.entropy)) for s in result.steps]
fig_ent = go.Figure()
fig_ent.add_trace(
    go.Scatter(
        x=step_numbers,
        y=mean_entropies,
        mode="lines+markers",
        marker_color=COLORS["confident"],
        fill="tozeroy",
        fillcolor="rgba(46,204,113,0.15)",
    )
)
fig_ent.update_layout(
    xaxis_title="Step",
    yaxis_title="Mean entropy",
    height=250,
    margin=dict(l=40, r=20, t=20, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"]),
    xaxis=dict(gridcolor="#333"),
    yaxis=dict(gridcolor="#333"),
)
st.plotly_chart(fig_ent, width="stretch")

# --- Explanation ---
with st.expander("How does self-conditioning work?"):
    st.markdown("""
    1. After each denoising step, the model takes the **softmax distribution** (not the hard tokens).
    2. It computes a **probability-weighted average** of token embeddings — a soft representation.
    3. This is passed through a small **gated MLP** whose gate starts at 0 and increases over steps.
    4. The output is **added to the canvas embeddings** before the next forward pass.
    5. This gives each step a "memory" of what the model believed previously, even for positions
       that were re-noised to random tokens.
    """)
