import streamlit as st

from utils.glossary import slugify_term
from utils.navigation import render_learning_path
from utils.styles import inject_styles, render_description

st.set_page_config(page_title="Terms Glossary", page_icon="📘", layout="wide")
inject_styles()

st.title("📘 Terms Glossary")
render_description(
    """
    This page is the reference layer for the visualizations. It defines the core
    terms used in page names, charts, controls, and hover labels.

    If you arrive here from a glossary link, the matching term opens
    automatically. Otherwise, scan the expanders for unfamiliar concepts before
    returning to the graph-heavy pages.
    """,
    expanded=True,
)
render_learning_path("pages/7_terms_glossary.py")

selected_term = st.query_params.get("term")

terms = [
    (
        "DiffusionGemma",
        "A language model from Google DeepMind that generates text by refining a "
        "whole block of noisy tokens over several denoising passes instead of "
        "producing exactly one next token at a time.",
    ),
    (
        "Canvas",
        "The current block of tokens being refined. At the start it is mostly "
        "noise or low-confidence guesses; by the end it should contain a clean "
        "text continuation.",
    ),
    (
        "Logits",
        "The raw scores the model produces for each possible token before they "
        "are converted into probabilities. Higher logits mean the model prefers "
        "that token more strongly.",
    ),
    (
        "Softmax",
        "The function that converts logits into probabilities that sum to 1. "
        "It turns raw model scores into a distribution the model can sample or "
        "analyze for confidence.",
    ),
    (
        "Argmax",
        "The highest-probability token at a position. In these visualizations, "
        "the argmax is the model's current best guess before any sampling noise "
        "or later refinement changes it.",
    ),
    (
        "Temperature",
        "A sampling control that changes how sharp or flat the model's token "
        "distribution becomes before sampling. Lower temperature makes choices "
        "more deterministic; higher temperature makes them more diverse.",
    ),
    (
        "Denoising",
        "One refinement step where the model looks at the whole canvas, predicts "
        "better tokens, keeps confident positions, and re-noises uncertain ones "
        "for another pass.",
    ),
    (
        "Entropy",
        "A measure of uncertainty in a probability distribution. Low entropy "
        "means the model strongly prefers a small number of tokens; high "
        "entropy means the model is still unsure.",
    ),
    (
        "Entropy-bound acceptance",
        "A confidence rule. Low-entropy positions have sharp, confident token "
        "distributions, so they are accepted. High-entropy positions are still "
        "uncertain, so they stay editable and get revisited in later steps.",
    ),
    (
        "Entropy budget",
        "The total amount of uncertainty the model is allowed to accept in one "
        "denoising pass. Once the running sum of accepted-position entropies "
        "crosses this budget, the remaining positions are re-noised.",
    ),
    (
        "Gumbel-max sampling",
        "A way to draw one discrete token sample from a probability distribution "
        "by adding random Gumbel noise to log-probabilities and taking the argmax.",
    ),
    (
        "Self-conditioning",
        "The model feeds information from its previous prediction back into the "
        "next denoising step. That stabilizes the refinement process and reduces "
        "step-to-step thrashing.",
    ),
    (
        "Self-conditioning gate",
        "A scalar control that sets how strongly the previous step's soft "
        "prediction influences the next denoising pass. A higher gate means "
        "more of that prior belief is mixed back into the next step.",
    ),
    (
        "Gated MLP",
        "A small learned transformation used inside self-conditioning. It "
        "reshapes the previous step's soft signal before adding it back into "
        "the next canvas input.",
    ),
    (
        "Embedding",
        "A dense vector representation of a token or position that the model "
        "can process numerically. Embeddings let the model work with meaning "
        "and context instead of raw token IDs.",
    ),
    (
        "Probability-weighted average",
        "A soft combination where each token embedding is multiplied by its "
        "predicted probability and then summed. Likely tokens contribute more, "
        "but uncertainty is preserved instead of collapsing immediately to one token.",
    ),
    (
        "Block sampling",
        "Text is generated in blocks. Each block goes through prefill, denoise, "
        "and commit. After one block is finished, the next block starts using "
        "the already committed text as context.",
    ),
    (
        "Prefill",
        "The stage where the existing prompt or already committed text is encoded "
        "into the model state before a new noisy block is denoised.",
    ),
    (
        "Commit",
        "The point where a denoised block is treated as finished and appended to "
        "the running output. After commit, later blocks can condition on it.",
    ),
    (
        "Encoder / decoder modes",
        "DiffusionGemma reuses one backbone in two behaviors. The encoder-style "
        "path handles causal context accumulation, while the decoder-style path "
        "handles bidirectional denoising over the current canvas.",
    ),
    (
        "Autoregressive",
        "A left-to-right generation pattern where each new token depends only "
        "on previously generated tokens. Standard causal language models use "
        "autoregressive generation.",
    ),
    (
        "Causal attention",
        "Each position can only attend to itself and earlier positions. This "
        "preserves left-to-right ordering and is used when processing committed "
        "text or prompts.",
    ),
    (
        "Bidirectional attention",
        "Each position can attend to tokens on both sides. This is useful during "
        "denoising because the whole block is being refined together rather than "
        "generated strictly one token at a time.",
    ),
    (
        "Sliding window attention",
        "A local attention pattern where each position only looks at nearby "
        "tokens. It reduces compute and memory cost while still letting the "
        "model use neighborhood context.",
    ),
    (
        "Attention mask",
        "A boolean pattern that decides which key positions each query position "
        "is allowed to attend to. Changing the mask changes the information flow "
        "without changing the model weights.",
    ),
    (
        "Query / key",
        "The attention mechanism compares a query vector from the current "
        "position against key vectors from candidate source positions to decide "
        "which information should be read.",
    ),
    (
        "KV cache",
        "Stored key/value attention states for already processed tokens. Keeping "
        "this cache lets later stages reuse past context efficiently instead of "
        "recomputing everything from scratch.",
    ),
    (
        "Convergence",
        "The point where repeated denoising passes stop changing much and the "
        "canvas has effectively stabilized. A converged block is ready to commit.",
    ),
    (
        "Transformer layer",
        "One repeated processing block inside the model, typically containing "
        "attention and feed-forward computation. Stacking many layers lets the "
        "model build more complex representations.",
    ),
    (
        "MoE",
        "Mixture of Experts. A model design where a routing mechanism chooses "
        "which specialized feed-forward sub-networks should handle a token, "
        "instead of using the same feed-forward path for every token.",
    ),
    (
        "FFN",
        "Feed-forward network. The per-token neural network sub-layer inside a "
        "transformer block that processes each position after attention.",
    ),
]

for term, explanation in terms:
    slug = slugify_term(term)
    st.markdown(f'<div id="{slug}"></div>', unsafe_allow_html=True)
    with st.expander(term, expanded=selected_term == slug):
        st.markdown(explanation)
