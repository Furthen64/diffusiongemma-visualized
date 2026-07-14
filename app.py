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
    "Pick a concept below to start exploring."
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link(
        "pages/1_canvas_denoising.py",
        label="Canvas Denoising",
        icon="🧊",
    )
    st.caption("Watch a 256-token canvas go from pure noise to coherent text through iterative denoising steps.")

with col2:
    st.page_link(
        "pages/2_entropy_bound.py",
        label="Entropy-Bound Acceptance",
        icon="📊",
    )
    st.caption("See how the model decides which positions to keep and which to re-noise based on confidence.")

with col3:
    st.page_link(
        "pages/3_self_conditioning.py",
        label="Self-Conditioning",
        icon="🔄",
    )
    st.caption("Understand how the model feeds its own predictions back to stabilize convergence.")

col4, col5, col6 = st.columns(3)

with col4:
    st.page_link(
        "pages/4_block_sampling.py",
        label="Block Sampling Loop",
        icon="🔗",
    )
    st.caption("See how blocks chain together: Prefill \u2192 Denoise \u2192 Commit \u2192 Next.")

with col5:
    st.page_link(
        "pages/5_encoder_decoder.py",
        label="Encoder / Decoder Modes",
        icon="⚡",
    )
    st.caption("One backbone, two modes: causal for encoding, bidirectional for denoising.")

with col6:
    st.page_link(
        "pages/6_attention_mechanisms.py",
        label="Attention Mechanisms",
        icon="🔍",
    )
    st.caption("Causal, bidirectional, and sliding window attention \u2014 including per-sequence mixing in batches.")

glossary_col, _ = st.columns([1, 2])

with glossary_col:
    st.page_link(
        "pages/7_terms_glossary.py",
        label="Terms Glossary",
        icon="📘",
    )
    st.caption("Definitions for the core DiffusionGemma concepts used across all visualizations.")

st.markdown("---")
st.markdown(
    "Built to understand [DiffusionGemma](https://deepmind.google/models/gemma/diffusiongemma/) "
    "by Google DeepMind.  All visualizations use simulated data \u2014 no GPU required."
)
