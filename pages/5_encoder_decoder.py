import streamlit as st
import numpy as np

from utils.glossary import glossary_href, glossary_link
from utils.styles import inject_styles, COLORS
from utils.diffusion_sim import causal_mask, bidirectional_mask
from utils.plot_helpers import attention_mask_heatmap

st.set_page_config(page_title="Encoder / Decoder Modes", page_icon="⚡", layout="wide")
inject_styles()

st.title("⚡ Encoder / Decoder Modes")
st.markdown(
    "DiffusionGemma runs a single Gemma4 backbone in two modes that share the "
    "same weights — one set of layers, used two ways."
)
st.markdown(
    f"Reference: {glossary_link('Encoder / decoder modes', 'Encoder / decoder modes')} · "
    f"{glossary_link('Causal attention', 'Causal attention')} · "
    f"{glossary_link('Bidirectional attention', 'Bidirectional attention')} · "
    f"{glossary_link('KV cache', 'KV cache')} · "
    f"{glossary_link('Self-conditioning', 'Self-conditioning')} · "
    f"{glossary_link('Transformer layer', 'Transformer layer')} · "
    f"{glossary_link('MoE', 'MoE')} · "
    f"{glossary_link('FFN', 'FFN')}",
    unsafe_allow_html=True,
)

# --- Sidebar ---
st.sidebar.markdown("### Controls")
mode = st.sidebar.radio("Mode", ["Encoder (Causal)", "Decoder (Bidirectional)"])
canvas_size = st.sidebar.slider("Canvas / sequence length", 4, 16, 8)
show_kv = st.sidebar.checkbox("Show KV cache state", value=True)

is_encoder = mode.startswith("Encoder")

# --- Mode diagram ---
st.markdown("#### Mode Toggle")
if is_encoder:
    toggle_html = f"""
    <div style="display:flex;align-items:center;gap:12px;margin:1rem 0;">
      <span style="padding:12px 24px;border-radius:8px;background:{COLORS["encoder"]};
                   color:#fff;font-weight:bold;font-size:1.1rem;
                   box-shadow:0 0 12px {COLORS["encoder"]}66;">Encoder (Causal)</span>
      <span style="color:#666;font-size:1.5rem;">⇄</span>
      <span style="padding:12px 24px;border-radius:8px;background:#333;
                   color:#999;font-size:1.1rem;">Decoder (Bidirectional)</span>
    </div>
    <p style="color:#999;">Uses <b><a href="{glossary_href('Causal attention')}" style="color:inherit;">causal attention</a></b>. Writes to <a href="{glossary_href('KV cache')}" style="color:inherit;">KV cache</a>.
       Runs during prefill and commit phases.</p>
    """
else:
    toggle_html = f"""
    <div style="display:flex;align-items:center;gap:12px;margin:1rem 0;">
      <span style="padding:12px 24px;border-radius:8px;background:#333;
                   color:#999;font-size:1.1rem;">Encoder (Causal)</span>
      <span style="color:#666;font-size:1.5rem;">⇄</span>
      <span style="padding:12px 24px;border-radius:8px;background:{COLORS["decoder"]};
                   color:#fff;font-weight:bold;font-size:1.1rem;
                   box-shadow:0 0 12px {COLORS["decoder"]}66;">Decoder (Bidirectional)</span>
    </div>
    <p style="color:#999;">Uses <b><a href="{glossary_href('Bidirectional attention')}" style="color:inherit;">bidirectional attention</a></b>. Reads (but does not
       update) the <a href="{glossary_href('KV cache')}" style="color:inherit;">KV cache</a>. Runs during the denoising phase.</p>
    """
st.markdown(toggle_html, unsafe_allow_html=True)

# --- Attention mask ---
st.markdown("---")
left, right = st.columns(2)

with left:
    st.markdown("#### Attention Mask")
    if is_encoder:
        mask = causal_mask(canvas_size)
        title = "Causal Mask (Encoder) — upper triangle masked"
    else:
        mask = bidirectional_mask(canvas_size)
        title = "Bidirectional Mask (Decoder) — nothing masked"

    st.plotly_chart(
        attention_mask_heatmap(mask, title=title),
        width="stretch",
    )

with right:
    st.markdown("#### Forward Pass Walkthrough")
    if is_encoder:
        steps = [
            ("1. Embed tokens", f"Convert token IDs to dense {glossary_link('vectors', 'Embedding')}"),
            ("2. Apply causal mask", f"Each position attends only to itself and prior tokens via an {glossary_link('attention mask', 'Attention mask')}"),
            ("3. Run transformer layers", f"30 {glossary_link('transformer layers', 'Transformer layer')} of {glossary_link('MoE', 'MoE')} attention + {glossary_link('FFN', 'FFN')}"),
            ("4. Write to KV cache", f"Key/value pairs stored in the {glossary_link('KV cache', 'KV cache')} for future decoder passes"),
            ("5. Output logits", f"{glossary_link('Logits', 'Logits')} for next-step use (prefill) or commit"),
        ]
    else:
        steps = [
            ("1. Embed canvas tokens", f"Random/noised tokens from the current {glossary_link('canvas', 'Canvas')}"),
            ("2. Apply self-conditioning", f"Add the previous step's gated {glossary_link('softmax', 'Softmax')} signal"),
            ("3. Apply bidirectional mask", f"Every canvas token can attend to every other through an {glossary_link('attention mask', 'Attention mask')}"),
            ("4. Run transformer layers", f"Same 30 {glossary_link('transformer layers', 'Transformer layer')}, bidirectional attention mode"),
            ("5. Read KV cache", f"Attend to all previously committed context in the {glossary_link('KV cache', 'KV cache')}"),
            ("6. Output logits", f"New {glossary_link('logits', 'Logits')} for each canvas position"),
        ]

    for title, desc in steps:
        st.markdown(
            f'<div style="padding:8px 12px;margin:4px 0;border-radius:6px;'
            f'background:{COLORS["card_bg"]};border-left:3px solid '
            f'{COLORS["encoder"] if is_encoder else COLORS["decoder"]};">'
            f'<b style="color:{COLORS["text"]}">{title}</b><br>'
            f'<span style="color:#999;font-size:0.85rem;">{desc}</span></div>',
            unsafe_allow_html=True,
        )

# --- KV Cache state ---
if show_kv:
    st.markdown("---")
    st.markdown("#### KV Cache State")
    st.markdown(
        "The KV cache stores key/value pairs from the encoder. "
        "The decoder reads from it but never writes to it."
    )

    # Simulate a multi-block scenario
    block_size = 8
    n_committed = 3
    kv_tokens = block_size * n_committed + canvas_size  # prompt + committed blocks + current

    kv_cols = st.columns(min(kv_tokens, 16))
    for i in range(min(kv_tokens, 16)):
        with kv_cols[i]:
            if i < block_size * n_committed:
                # Committed
                st.markdown(
                    f'<div style="padding:6px;border-radius:4px;text-align:center;'
                    f'background:{COLORS["commit"]}33;border:1px solid {COLORS["commit"]};'
                    f'color:{COLORS["commit"]};font-size:0.7rem;font-family:monospace;">'
                    f'K/V<br>pos {i}</div>',
                    unsafe_allow_html=True,
                )
            elif is_encoder:
                # Encoder is writing
                st.markdown(
                    f'<div style="padding:6px;border-radius:4px;text-align:center;'
                    f'background:{COLORS["encoder"]}33;border:1px solid {COLORS["encoder"]};'
                    f'color:{COLORS["encoder"]};font-size:0.7rem;font-family:monospace;">'
                    f'WRITING<br>pos {i}</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Decoder is reading
                st.markdown(
                    f'<div style="padding:6px;border-radius:4px;text-align:center;'
                    f'background:{COLORS["decoder"]}33;border:1px solid {COLORS["decoder"]};'
                    f'color:{COLORS["decoder"]};font-size:0.7rem;font-family:monospace;">'
                    f'READING<br>pos {i}</div>',
                    unsafe_allow_html=True,
                )

    legend_html = f"""
    <div style="display:flex;gap:16px;margin-top:8px;font-size:0.8rem;">
      <span style="color:{COLORS["commit"]}">■ Committed (KV written)</span>
      {"<span style='color:" + COLORS["encoder"] + "'>■ Encoder writing</span>" if is_encoder else ""}
      {"<span style='color:" + COLORS["decoder"] + "'>■ Decoder reading</span>" if not is_encoder else ""}
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

# --- Explanation ---
with st.expander("Why two modes?"):
    st.markdown(f"""
    The same Gemma4 backbone runs in two modes sharing one set of weights:

    - **Encoder mode (causal)**: Used during {glossary_link('prefill', 'Prefill')} (ingesting the prompt) and
      {glossary_link('commit', 'Commit')} (appending a finished block to the {glossary_link('KV cache', 'KV cache')}).  {glossary_link('Causal attention', 'Causal attention')} means
      each token only sees itself and prior tokens — standard {glossary_link('autoregressive', 'Autoregressive')} behavior.

    - **Decoder mode (bidirectional)**: Used during {glossary_link('denoising', 'Denoising')}.  Every canvas token
      can attend to every other canvas token, enabling the model to refine the
      entire block in parallel and self-correct errors.

    This design lets DiffusionGemma reuse the same weights for both encoding and
    denoising, with only the attention mask changing between modes.
    """, unsafe_allow_html=True)
