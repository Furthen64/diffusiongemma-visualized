import streamlit as st

st.set_page_config(
    page_title="DiffusionGemma Visualized",
    page_icon="🧊",
    layout="wide",
)

from utils.styles import inject_styles, COLORS

inject_styles()

st.title("DiffusionGemma — Visualized")
st.markdown(
    "An interactive guide to understanding how **DiffusionGemma** works.  "
    "Use the sidebar to explore each concept."
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f'<div class="metric-card">'
        f'<h3 style="color:{COLORS["encoder"]}">🧊 Canvas Denoising</h3>'
        f"<p>Watch a 256-token canvas go from pure noise to coherent text "
        f"through iterative denoising steps.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f'<div class="metric-card">'
        f'<h3 style="color:{COLORS["decoder"]}">📊 Entropy-Bound Acceptance</h3>'
        f"<p>See how the model decides which positions to keep and which "
        f"to re-noise based on confidence.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f'<div class="metric-card">'
        f'<h3 style="color:{COLORS["confident"]}">🔄 Self-Conditioning</h3>'
        f"<p>Understand how the model feeds its own predictions back to "
        f"stabilize convergence.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(
        f'<div class="metric-card">'
        f'<h3 style="color:{COLORS["denoise"]}">🔗 Block Sampling Loop</h3>'
        f"<p>See how blocks chain together: Prefill → Denoise → Commit → Next.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        f'<div class="metric-card">'
        f'<h3 style="color:{COLORS["encoder"]}">⚡ Encoder / Decoder Modes</h3>'
        f"<p>One backbone, two modes: causal for encoding, bidirectional "
        f"for denoising.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col6:
    st.markdown(
        f'<div class="metric-card">'
        f'<h3 style="color:{COLORS["masked"]}">🔍 Attention Mechanisms</h3>'
        f"<p>Causal, bidirectional, and sliding window attention — including "
        f"per-sequence mixing in batches.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown(
    "Built to understand [DiffusionGemma](https://deepmind.google/models/gemma/diffusiongemma/) "
    "by Google DeepMind.  All visualizations use simulated data — no GPU required."
)
