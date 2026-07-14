from __future__ import annotations

import streamlit as st


PAGES = [
    {
        "path": "pages/1_canvas_denoising.py",
        "label": "Canvas Denoising",
        "icon": "🧊",
        "focus": "See the whole noisy canvas refine into text.",
    },
    {
        "path": "pages/2_entropy_bound.py",
        "label": "Entropy-Bound Acceptance",
        "icon": "📊",
        "focus": "Understand which token positions get accepted.",
    },
    {
        "path": "pages/3_self_conditioning.py",
        "label": "Self-Conditioning",
        "icon": "🔄",
        "focus": "Track how previous predictions stabilize later passes.",
    },
    {
        "path": "pages/4_block_sampling.py",
        "label": "Block Sampling Loop",
        "icon": "🔗",
        "focus": "Zoom out to prefill, denoise, commit, and repeat.",
    },
    {
        "path": "pages/5_encoder_decoder.py",
        "label": "Encoder / Decoder Modes",
        "icon": "⚡",
        "focus": "Compare the two behaviors of one shared backbone.",
    },
    {
        "path": "pages/6_attention_mechanisms.py",
        "label": "Attention Mechanisms",
        "icon": "🔍",
        "focus": "Inspect the masks that control information flow.",
    },
    {
        "path": "pages/7_terms_glossary.py",
        "label": "Terms Glossary",
        "icon": "📘",
        "focus": "Look up the vocabulary used across the visualizations.",
    },
]


def render_learning_path(current_path: str | None = None):
    if current_path is None:
        st.markdown(
            """
            <div class="learning-path">
              <div class="learning-path-kicker">Recommended path</div>
              <div class="learning-path-title">Start with the canvas, then unpack the mechanisms.</div>
              <div class="learning-path-note">The numbered pages are ordered from concrete behavior to supporting internals.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    current_index = next(
        (index for index, page in enumerate(PAGES) if page["path"] == current_path),
        None,
    )
    if current_index is None:
        return

    current = PAGES[current_index]
    st.markdown(
        f"""
        <div class="learning-path">
          <div class="learning-path-kicker">Learning path · Step {current_index + 1} of {len(PAGES)}</div>
          <div class="learning-path-title">{current["icon"]} {current["label"]}</div>
          <div class="learning-path-note">{current["focus"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prev_page = PAGES[current_index - 1] if current_index > 0 else None
    next_page = PAGES[current_index + 1] if current_index < len(PAGES) - 1 else None
    home_col, prev_col, next_col = st.columns([1, 1, 1])
    with home_col:
        st.page_link("app.py", label="Overview", icon="🏠")
    with prev_col:
        if prev_page:
            st.page_link(
                prev_page["path"],
                label=f"Previous: {prev_page['label']}",
                icon=prev_page["icon"],
            )
    with next_col:
        if next_page:
            st.page_link(
                next_page["path"],
                label=f"Next: {next_page['label']}",
                icon=next_page["icon"],
            )
