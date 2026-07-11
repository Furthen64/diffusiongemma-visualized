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
</style>
"""


def inject_styles():
    import streamlit as st

    st.markdown(PAGE_CSS, unsafe_allow_html=True)
