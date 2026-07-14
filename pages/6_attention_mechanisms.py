import streamlit as st
import numpy as np
import plotly.graph_objects as go

from utils.glossary import glossary_link
from utils.styles import inject_styles, COLORS
from utils.diffusion_sim import (
    causal_mask,
    bidirectional_mask,
    sliding_window_mask,
)
from utils.plot_helpers import attention_mask_heatmap

st.set_page_config(page_title="Attention Mechanisms", page_icon="🔍", layout="wide")
inject_styles()

st.title("🔍 Attention Mechanisms")
st.markdown(
    "DiffusionGemma uses three attention patterns: **causal** (encoder), "
    "**bidirectional** (decoder), and **sliding window** (for efficiency). "
    "In a batch, each request can be at a different phase with its own mask."
)
st.markdown(
    f"Reference: {glossary_link('Causal attention', 'Causal attention')} · "
    f"{glossary_link('Bidirectional attention', 'Bidirectional attention')} · "
    f"{glossary_link('Sliding window attention', 'Sliding window attention')} · "
    f"{glossary_link('Prefill', 'Prefill')} · "
    f"{glossary_link('Commit', 'Commit')} · "
    f"{glossary_link('Attention mask', 'Attention mask')} · "
    f"{glossary_link('Query / key', 'Query / key')}",
    unsafe_allow_html=True,
)

# --- Sidebar ---
st.sidebar.markdown("### Controls")
view = st.sidebar.radio(
    "View",
    ["Single Mask", "Batch Mixing", "Sliding Window"],
)

if view == "Single Mask":
    mask_type = st.sidebar.radio("Mask type", ["Causal", "Bidirectional"])
    seq_len = st.sidebar.slider("Sequence length", 4, 16, 8)
    query_pos = st.sidebar.slider("Highlight query position", 0, seq_len - 1, 0)

elif view == "Batch Mixing":
    len0 = st.sidebar.slider("Request 0 length (Prefill)", 3, 10, 6)
    len1 = st.sidebar.slider("Request 1 length (Denoise)", 3, 10, 6)
    len2 = st.sidebar.slider("Request 2 length (Commit)", 3, 10, 6)
    canvas1 = st.sidebar.slider("Request 1 canvas size", 3, 8, 4)

elif view == "Sliding Window":
    seq_len = st.sidebar.slider("Sequence length", 6, 16, 10)
    window = st.sidebar.slider("Window size (W)", 1, 6, 2)
    mode = st.sidebar.radio("Mode", ["Causal + Sliding Window", "Bidirectional + Sliding Window"])

# ============================================================
# View 1: Single Mask
# ============================================================
if view == "Single Mask":
    st.markdown(f"#### Single Request {glossary_link('Attention Mask', 'Attention mask')}", unsafe_allow_html=True)

    if mask_type == "Causal":
        mask = causal_mask(seq_len)
        title = f"Causal Mask ({seq_len}x{seq_len})"
        desc = "Each query attends to itself and all prior keys. Upper triangle is masked."
    else:
        mask = bidirectional_mask(seq_len)
        title = f"Bidirectional Mask ({seq_len}x{seq_len})"
        desc = "Every query attends to every key. No masking."

    st.plotly_chart(
        attention_mask_heatmap(mask, title=title),
        width="stretch",
    )

    attended = [k for k in range(seq_len) if not mask[query_pos, k]]
    st.markdown(
        f"Attends to keys: `{attended}` — "
        f"**{len(attended)}/{seq_len}** positions"
    )

    # Show the query row as a horizontal bar
    fig = go.Figure()
    row = np.where(~mask[query_pos], 1.0, 0.0)
    colors = [
        COLORS["encoder"] if row[k] > 0.5 else COLORS["masked"]
        for k in range(seq_len)
    ]
    fig.add_trace(
        go.Bar(
            x=list(range(seq_len)),
            y=[1] * seq_len,
            marker_color=colors,
            hovertemplate="Key %{x}: %{text}<extra></extra>",
            text=["Attended" if row[k] > 0.5 else "Masked" for k in range(seq_len)],
        )
    )
    fig.update_layout(
        title=f"Query {query_pos} — Attention Span",
        xaxis_title="Key position",
        yaxis=dict(visible=False),
        height=100,
        margin=dict(l=20, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor="#333", dtick=1),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"**{desc}** This describes which {glossary_link('keys', 'Query / key')} a "
        f"{glossary_link('query', 'Query / key')} position can read.",
        unsafe_allow_html=True,
    )

# ============================================================
# View 2: Batch Mixing
# ============================================================
elif view == "Batch Mixing":
    st.markdown(
        f"#### Per-Sequence {glossary_link('Causal Attention', 'Causal attention')} in a Batch",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Three requests in the same batch, each at a different phase with "
        "its own attention pattern."
    )

    reqs = [
        {"name": "Request 0 — Prefill", "len": len0, "mode": "causal", "color": COLORS["encoder"]},
        {"name": "Request 1 — Denoise", "len": len1, "mode": "bidirectional", "color": COLORS["decoder"]},
        {"name": "Request 2 — Commit", "len": len2, "mode": "causal", "color": COLORS["encoder"]},
    ]

    # Individual masks
    cols = st.columns(3)
    for i, (col, req) in enumerate(zip(cols, reqs)):
        with col:
            if req["mode"] == "causal":
                m = causal_mask(req["len"])
            else:
                m = bidirectional_mask(req["len"])
            st.plotly_chart(
                attention_mask_heatmap(m, title=req["name"]),
                width="stretch",
            )

    # Combined block-diagonal view
    st.markdown(f"#### Combined Batch {glossary_link('Mask', 'Attention mask')}", unsafe_allow_html=True)
    total = sum(r["len"] for r in reqs)
    combined = np.ones((total, total), dtype=bool)
    offset = 0
    block_colors = []
    for req in reqs:
        L = req["len"]
        if req["mode"] == "causal":
            combined[offset:offset+L, offset:offset+L] = causal_mask(L)
        else:
            combined[offset:offset+L, offset:offset+L] = bidirectional_mask(L)
        block_colors.append((offset, L, req["color"], req["name"]))
        offset += L

    display = np.where(combined, 0.0, 1.0)
    fig = go.Figure(data=go.Heatmap(
        z=display,
        colorscale=[[0.0, COLORS["masked"]], [1.0, "#ffffff"]],
        showscale=False,
        hovertemplate="Q%{y} → K%{x}<extra></extra>",
    ))
    # Add block boundaries
    offset = 0
    for req in reqs:
        L = req["len"]
        fig.add_shape(
            type="rect",
            x0=offset, x1=offset+L,
            y0=offset, y1=offset+L,
            line=dict(color=req["color"], width=2),
        )
        offset += L

    fig.update_layout(
        title=f"Combined Batch Mask ({total}x{total})",
        xaxis_title="Key position",
        yaxis_title="Query position",
        height=400,
        width=500,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor="#333", dtick=1),
        yaxis=dict(gridcolor="#333", dtick=1, autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch")

    # Legend
    legend = " ".join(
        f'<span style="color:{r["color"]}">■ {r["name"]}</span>'
        for r in reqs
    )
    st.markdown(legend, unsafe_allow_html=True)

# ============================================================
# View 3: Sliding Window
# ============================================================
elif view == "Sliding Window":
    st.markdown(f"#### {glossary_link('Sliding Window Attention', 'Sliding window attention')}", unsafe_allow_html=True)

    is_causal = mode.startswith("Causal")

    if is_causal:
        mask = sliding_window_mask(seq_len, window)
        title = f"Causal Sliding Window (W={window})"
        desc = (
            f"Each query attends to itself and the {window} keys before it, "
            f"for a total window of {2*window+1} positions. "
            "Other positions are masked."
        )
    else:
        mask = bidirectional_mask(seq_len)
        # Override with symmetric sliding window
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) > window:
                    mask[i, j] = True
        title = f"Bidirectional Sliding Window (W={window})"
        desc = (
            f"Each query attends to the {window} keys on either side, "
            f"for a total window of {min(2*window+1, seq_len)} positions. "
            "Used for canvas tokens in diffusion layers."
        )

    col_mask, col_detail = st.columns([1, 1])

    with col_mask:
        st.plotly_chart(
            attention_mask_heatmap(mask, title=title),
            width="stretch",
        )

    with col_detail:
        st.markdown(
            f"**{desc}** This is another {glossary_link('attention mask', 'Attention mask')} "
            f"over {glossary_link('query / key', 'Query / key')} positions.",
            unsafe_allow_html=True,
        )

        # Zoomed view for one query
        query = min(seq_len // 2, seq_len - 1)
        attended = [k for k in range(seq_len) if not mask[query, k]]

        fig = go.Figure()
        row = np.where(~mask[query], 1.0, 0.0)
        colors = [
            COLORS["confident"] if abs(query - k) <= window else COLORS["masked"]
            for k in range(seq_len)
        ]
        fig.add_trace(go.Bar(
            x=list(range(seq_len)),
            y=[1] * seq_len,
            marker_color=colors,
            text=[f"{'In' if not mask[query,k] else 'Out'}" for k in range(seq_len)],
            hovertemplate="Key %{x}: %{text}<extra></extra>",
        ))
        fig.update_layout(
            title=f"Query {query} — Window [{max(0,query-window)}, {min(seq_len-1,query+window)}]",
            xaxis_title="Key position",
            yaxis=dict(visible=False),
            height=150,
            margin=dict(l=20, r=20, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text"]),
            xaxis=dict(gridcolor="#333", dtick=1),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            f"Query {query} attends to keys: `{attended}` — "
            f"**{len(attended)}/{seq_len}** positions"
        )

# --- Explanation ---
with st.expander("How do these attention mechanisms relate?"):
    st.markdown(
        f"""
    DiffusionGemma uses {glossary_link('attention masks', 'Attention mask')} dynamically per-request:

    - **Causal (full)**: Standard {glossary_link('autoregressive', 'Autoregressive')} mask. Upper triangle masked.
      Used in encoder mode for {glossary_link('prefill', 'Prefill')} and {glossary_link('commit', 'Commit')}.

    - **Bidirectional (full)**: No masking. Every position attends to every other.
      Used in decoder mode for {glossary_link('denoising', 'Denoising')} — this is what allows parallel refinement.

    - **Causal sliding window**: Only attends to the W positions before each query.
      Reduces compute for long sequences in some layers.

    - **Bidirectional sliding window**: Attends to W positions on either side.
      Used for canvas tokens in diffusion layers — symmetric because canvas
      tokens don't have a left-to-right ordering during denoising.

    In a batch, each request can be at a different phase (prefill/denoise/commit),
    so the attention mask must be dynamic per-sequence — not batch-wide.
    """,
        unsafe_allow_html=True,
    )
