import html

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.canvas_tutorial import run_tutorial
from utils.styles import COLORS, inject_styles


st.set_page_config(page_title="Canvas Denoising", page_icon="🧊", layout="wide")
inject_styles()
st.markdown("""
<style>
.prompt-box {background:#10182e;border:1px solid #3b4b70;border-radius:12px;padding:1rem 1.2rem;font-size:1.15rem}
.blank {color:#ffbd59;font-weight:700;border-bottom:2px solid #ffbd59;padding:0 .25rem}
.phase-strip {display:grid;grid-template-columns:repeat(5,1fr);gap:.35rem;margin:.6rem 0 1rem}
.phase {padding:.55rem .35rem;text-align:center;border-radius:8px;background:#202a45;color:#9ba9c8;font-size:.8rem}
.phase.active {background:#ffbd59;color:#151827;font-weight:800}
.canvas {display:flex;flex-wrap:wrap;gap:3px;line-height:1.25;margin:.4rem 0 1rem}
.token {font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.72rem;padding:3px 5px;border-radius:4px;background:#26304b;color:#dce5fa;border:1px solid transparent}
.token.accepted {background:#153d32;border-color:#2ecc71;color:#baf5d5}
.token.rejected {background:#40232c;border-color:#e76b78;color:#ffd2d7}
.token.correct {background:#193b42;border-color:#34b9ca;color:#c7f8ff}
.token.noise {color:#77829c}
.legend-dot {display:inline-block;width:.65rem;height:.65rem;border-radius:2px;margin-right:.25rem}
@media(max-width:700px){.phase-strip{grid-template-columns:1fr}.phase{display:none}.phase.active{display:block}.token{font-size:.66rem}}
</style>
""", unsafe_allow_html=True)

st.title("🧊 Denoise a 256-token canvas")
st.markdown(
    "Follow one complete block from random tokens to a committed continuation. "
    "Every position is predicted **in parallel**, not from left to right."
)
st.markdown(
    '<div class="prompt-box">The story about a squirrel starts with '
    '<span class="blank">___</span></div>', unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Experiment")
    num_steps = st.slider("Denoising passes", 6, 20, 12)
    entropy_budget = st.slider("Entropy budget (nats)", 8.0, 80.0, 32.0, 2.0)
    temperature = st.slider("Sampling temperature", 0.4, 1.8, 1.0, 0.1)
    seed = st.number_input("Random seed", value=7, step=1)
    st.caption("The block size stays at DiffusionGemma's real 256 positions.")


@st.cache_data(show_spinner=False)
def simulate(steps: int, budget: float, temp: float, random_seed: int):
    return run_tutorial(num_steps=steps, entropy_budget=budget, temperature=temp, seed=random_seed)


run = simulate(num_steps, entropy_budget, temperature, int(seed))
state_key = "canvas_tutorial_step"
if state_key not in st.session_state:
    st.session_state[state_key] = 0
st.session_state[state_key] = min(st.session_state[state_key], num_steps)


def move_step(delta: int):
    st.session_state[state_key] = max(0, min(num_steps, st.session_state[state_key] + delta))


def jump_to_commit():
    st.session_state[state_key] = num_steps


back, slider_col, forward, final = st.columns([1, 7, 1, 1.35], vertical_alignment="bottom")
with back:
    st.button(
        "← Back", use_container_width=True, disabled=st.session_state[state_key] == 0,
        on_click=move_step, args=(-1,),
    )
with slider_col:
    st.slider("Step navigation", 0, num_steps, key=state_key)
with forward:
    st.button(
        "Next →", use_container_width=True, disabled=st.session_state[state_key] == num_steps,
        on_click=move_step, args=(1,),
    )
with final:
    st.button("Jump to commit", use_container_width=True, on_click=jump_to_commit)

step_number = st.session_state[state_key]
phases = ["1 · Canvas in", "2 · Predict", "3 · Sample", "4 · Accept / re-noise", "5 · Self-condition"]
active_phase = 0 if step_number == 0 else min(4, int((step_number - 1) / max(num_steps - 1, 1) * 5))
phase_html = "".join(
    f'<div class="phase {"active" if i == active_phase else ""}">{label}</div>'
    for i, label in enumerate(phases)
)
st.markdown(f'<div class="phase-strip">{phase_html}</div>', unsafe_allow_html=True)


def render_tokens(tokens, classes=None, limit=256):
    chunks = []
    for index, token in enumerate(tokens[:limit]):
        extra = "" if classes is None else f" {classes[index]}"
        title = html.escape(f"position {index}")
        chunks.append(f'<span class="token{extra}" title="{title}">{html.escape(token)}</span>')
    return '<div class="canvas">' + "".join(chunks) + "</div>"


if step_number == 0:
    st.subheader("Step 0 · Initialize with noise")
    st.info("After the prompt prefill, a fresh 256-position canvas is filled with random tokens. The prompt is in the KV cache; it is not part of this canvas.")
    st.markdown(render_tokens(run.initial_canvas, ["noise"] * 256), unsafe_allow_html=True)
    st.caption("Nothing is generated or committed yet. Next, all 256 positions attend bidirectionally to the prompt and to each other.")
else:
    snap = run.steps[step_number - 1]
    is_final = step_number == num_steps
    title = "Commit the clean argmax" if is_final else f"Denoising pass {step_number} of {num_steps}"
    st.subheader(title)
    accepted_pct = snap.accepted_count / 256
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accepted this pass", f"{snap.accepted_count} / 256", f"{accepted_pct:.0%}")
    m2.metric("Entropy used", f"{snap.budget_used:.1f}", f"of {entropy_budget:.0f} nats")
    m3.metric("Argmax changes", snap.changed_from_previous)
    m4.metric("Self-conditioning gate", f"{snap.self_conditioning:.2f}")

    tab_result, tab_algorithm, tab_entropy = st.tabs(["What the model sees", "Inside this pass", "Why these tokens?"])
    with tab_result:
        if is_final:
            st.success("The step cap has been reached. DiffusionGemma commits the clean argmax canvas, writes its KV entries, and advances by 256 tokens.")
            classes = ["correct" if token == target else "rejected" for token, target in zip(snap.argmax_tokens, run.target)]
            st.markdown(render_tokens(snap.argmax_tokens, classes), unsafe_allow_html=True)
        else:
            st.markdown("**Canvas carried into the next pass**")
            classes = ["accepted" if kept else "rejected" for kept in snap.accepted]
            st.markdown(render_tokens(snap.output_canvas, classes), unsafe_allow_html=True)
            st.caption("Green tokens are sampled candidates kept under the entropy budget. Red tokens were rejected and replaced with fresh random tokens.")
        with st.expander("Read the current best-guess continuation"):
            visible = [token for token in snap.argmax_tokens if token not in {"<pad>", "<eos>"}]
            st.write(" ".join(visible))

    with tab_algorithm:
        left, right = st.columns(2)
        with left:
            st.markdown("**A. Input canvas**")
            st.markdown(render_tokens(snap.input_canvas, ["noise"] * 256, 48), unsafe_allow_html=True)
            st.caption("First 48 of 256 positions. Bidirectional attention processes them together.")
            st.markdown("**B. Argmax prediction (for convergence)**")
            classes = ["correct" if token == target else "rejected" for token, target in zip(snap.argmax_tokens, run.target)]
            st.markdown(render_tokens(snap.argmax_tokens, classes, 48), unsafe_allow_html=True)
        with right:
            st.markdown("**C. Gumbel-max sample (candidate tokens)**")
            st.markdown(render_tokens(snap.sampled_tokens, ["noise"] * 256, 48), unsafe_allow_html=True)
            st.markdown("**D. Accept low entropy, re-noise the rest**")
            classes = ["accepted" if kept else "rejected" for kept in snap.accepted]
            st.markdown(render_tokens(snap.output_canvas, classes, 48), unsafe_allow_html=True)
        st.markdown(
            "**E. Self-condition:** the full softmax distribution from this pass is converted to a weighted token embedding and fed through a gated MLP into the next pass. "
            "It preserves soft beliefs even where the hard token was re-noised."
        )

    with tab_entropy:
        order = np.argsort(snap.entropy)
        colors = [COLORS["accepted"] if snap.accepted[index] else COLORS["rejected"] for index in order]
        fig = go.Figure(go.Bar(x=np.arange(256), y=snap.entropy[order], marker_color=colors, hovertemplate="rank %{x}<br>entropy %{y:.3f}<extra></extra>"))
        fig.update_layout(height=320, title="Positions ranked from most to least confident", xaxis_title="Confidence rank", yaxis_title="Entropy (nats)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=COLORS["text"], margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig, width="stretch")
        st.caption("The sampler walks left to right through this ranking and keeps candidates until adding the next position would exceed the shared entropy budget.")

st.divider()
st.markdown("#### What is exact, and what is simulated?")
st.markdown(
    "The **256-token canvas, bidirectional denoise pass, temperature-scaled Gumbel-max sampling, per-position entropy, entropy-bound acceptance, re-noising, self-conditioning, convergence check, and final argmax commit** mirror the documented algorithm. "
    "The probabilities are generated around a scripted squirrel story instead of running the 26B-parameter model, so the words and confidence values are illustrative."
)
