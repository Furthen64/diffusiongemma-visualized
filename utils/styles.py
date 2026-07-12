COLORS = {
    "confident": "#2ecc71",
    "noisy": "#e74c3c",
    "encoder": "#3498db",
    "decoder": "#e67e22",
    "masked": "#95a5a6",
    "neutral": "#ecf0f1",
    "accepted": "#2ecc71",
    "rejected": "#e74c3c",
    "prefill": "#3498db",
    "commit": "#2ecc71",
    "denoise": "#e67e22",
    "background": "#1a1a2e",
    "card_bg": "#16213e",
    "text": "#eaeaea",
}

PAGE_CSS = """
<style>
    .block-container { padding-top: 1rem; }
    h1 { color: #eaeaea; }
    h2 { color: #b0b0b0; }
    .metric-card {
        background: #16213e;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-converged {
        background: #2ecc7133;
        color: #2ecc71;
        border: 1px solid #2ecc71;
    }
    .badge-noisy {
        background: #e74c3c33;
        color: #e74c3c;
        border: 1px solid #e74c3c;
    }
    .step-info {
        font-size: 1.1rem;
        color: #b0b0b0;
        margin-bottom: 0.5rem;
    }
    .flow-stage {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 8px 10px 8px 14px;
        border-radius: 6px;
        font-family: monospace;
    }
    .flow-help {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.15rem;
        height: 1.15rem;
        border: 1px solid currentColor;
        border-radius: 50%;
        font-family: sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        line-height: 1;
        cursor: help;
        opacity: 0.8;
        outline-offset: 2px;
    }
    .flow-help::after {
        content: attr(data-tooltip);
        position: absolute;
        z-index: 20;
        left: 50%;
        bottom: calc(100% + 10px);
        width: 240px;
        padding: 0.65rem 0.75rem;
        transform: translateX(-50%);
        border: 1px solid #4b5563;
        border-radius: 6px;
        background: #111827;
        color: #eaeaea;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
        font-family: sans-serif;
        font-size: 0.82rem;
        font-weight: 400;
        line-height: 1.35;
        text-align: left;
        white-space: normal;
        pointer-events: none;
        opacity: 0;
        visibility: hidden;
        transition: opacity 120ms ease;
    }
    .flow-help:hover::after,
    .flow-help:focus::after {
        opacity: 1;
        visibility: visible;
    }
</style>
"""


def inject_styles():
    import streamlit as st

    st.markdown(PAGE_CSS, unsafe_allow_html=True)
