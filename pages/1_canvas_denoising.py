import html

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.canvas_tutorial import SCENARIOS, run_tutorial
from utils.glossary import glossary_link
from utils.navigation import render_learning_path
from utils.styles import COLORS, inject_styles, render_description


st.set_page_config(page_title="Canvas Denoising", page_icon="🧊", layout="wide")
inject_styles()
st.markdown("""
<style>
.prompt-box {background:#10182e;border:1px solid #3b4b70;border-radius:12px;padding:1rem 1.2rem;font:1.05rem ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.blank {color:#ffbd59;font-weight:700;border-bottom:2px solid #ffbd59;padding:0 .25rem}
.canvas {display:flex;flex-wrap:wrap;gap:3px;line-height:1.25;margin:.4rem 0 1rem}
.canvas.compact {gap:2px;line-height:1.1;margin:.35rem 0 .2rem}
.token {font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.72rem;padding:3px 5px;border-radius:4px;background:#26304b;color:#dce5fa;border:1px solid transparent}
.canvas.compact .token {font-size:.57rem;padding:2px 3px}
.token.accepted {background:#153d32;border-color:#2ecc71;color:#baf5d5}
.token.rejected {background:#40232c;border-color:#e76b78;color:#ffd2d7}
.token.correct {background:#193b42;border-color:#34b9ca;color:#c7f8ff}
.token.noise {color:#77829c}
.token.neutral {background:#1b2740;color:#91a2c3;border-color:#2c4068}
.legend-dot {display:inline-block;width:.65rem;height:.65rem;border-radius:2px;margin-right:.25rem}
.output-card {background:linear-gradient(135deg,#14233d,#10182e);border:1px solid #3b5a82;border-left:5px solid #34b9ca;border-radius:12px;padding:1.1rem 1.3rem;margin:.65rem 0 1rem}
.output-card.committed {border-color:#2ecc71;border-left-color:#2ecc71;background:linear-gradient(135deg,#15332d,#101d25)}
.output-kicker {color:#8fa9ca;font-size:.75rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.45rem}
.output-state {display:inline-block;border-radius:999px;padding:.18rem .55rem;margin-right:.55rem;font-weight:900}
.output-state.tentative {background:#34b9ca22;border:1px solid #34b9ca;color:#a9edf5}
.output-state.committed {background:#2ecc7122;border:1px solid #2ecc71;color:#8af0b5}
.output-text {color:#f2f6ff;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:1rem;line-height:1.9;margin:0}
.output-prompt {color:#8fa9ca}
.output-token {display:inline;padding:.08rem .12rem;border-radius:3px}
.output-token.tentative {color:#f2f6ff}
.output-token.committed {color:#c5f7da;background:#2ecc7124;border:1px solid #278a50}
.confidence-legend {display:flex;align-items:center;gap:.5rem;color:#8fa9ca;font:700 .7rem ui-monospace,SFMono-Regular,Menlo,monospace;margin-top:.65rem}
.confidence-ramp {width:9rem;height:.45rem;border-radius:999px;background:linear-gradient(90deg,hsl(0 72% 50%),hsl(60 72% 48%),hsl(120 62% 43%))}
.output-empty {color:#8fa9ca;font-style:italic}
.nav-shell {background:linear-gradient(135deg,#14233d,#0f172a);border:1px solid #35507b;border-radius:18px;padding:1rem 1rem .85rem;margin:.8rem 0 1rem;box-shadow:0 16px 36px rgba(3,8,20,.22)}
.nav-topline {display:flex;justify-content:space-between;align-items:center;gap:.75rem;flex-wrap:wrap;margin-bottom:.75rem}
.nav-kicker {color:#8fa9ca;font:800 .72rem ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;letter-spacing:.08em;text-transform:uppercase}
.nav-status {display:inline-flex;align-items:center;gap:.45rem;padding:.3rem .7rem;border-radius:999px;background:#34b9ca1c;border:1px solid #34b9ca55;color:#d9f7fb;font:800 .78rem ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.nav-status strong {font-size:.92rem;color:#ffffff}
.nav-rail {display:grid;grid-template-columns:auto 1fr auto;gap:.65rem;align-items:center}
.nav-stop {display:inline-flex;align-items:center;gap:.45rem;font:700 .74rem ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:#8fa9ca;white-space:nowrap}
.nav-dot {width:.55rem;height:.55rem;border-radius:999px;background:#34b9ca;box-shadow:0 0 0 4px #34b9ca18}
.nav-dot.commit {background:#2ecc71;box-shadow:0 0 0 4px #2ecc7118}
.pass-explainer {background:linear-gradient(180deg,#11192d,#0d1527);border:1px solid #263c61;border-radius:18px;padding:1rem 1rem .35rem;margin:.75rem 0 1rem}
.pass-kicker {color:#8fa9ca;font:800 .72rem ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.25rem}
.pass-title {color:#f2f6ff;font-size:1.1rem;font-weight:800;margin-bottom:.2rem}
.pass-subtitle {color:#9cb0d1;font-size:.92rem;margin-bottom:.8rem}
.stage-card {background:linear-gradient(180deg,#16213a,#10182b);border:1px solid #2f466e;border-radius:14px;padding:.85rem .9rem .7rem;margin-bottom:.8rem;min-height:100%}
.stage-head {display:flex;justify-content:space-between;align-items:baseline;gap:.75rem;margin-bottom:.35rem}
.stage-title {color:#f2f6ff;font:800 .95rem ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.stage-badge {display:inline-flex;align-items:center;gap:.35rem;padding:.2rem .5rem;border-radius:999px;background:#1f314f;border:1px solid #35507b;color:#8fa9ca;font:800 .67rem ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;letter-spacing:.04em;text-transform:uppercase}
.stage-body {color:#b7c7e3;font-size:.87rem;line-height:1.45}
.stage-legend {color:#7e93b8;font:700 .68rem ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;margin-top:.2rem}
.stage-arrow {color:#44648f;text-align:center;font-size:1.3rem;margin:-.2rem 0 .35rem}
@media(max-width:900px){.nav-rail{grid-template-columns:1fr}.nav-stop{justify-content:center}}
@media(max-width:700px){.token{font-size:.66rem}}
</style>
""", unsafe_allow_html=True)

st.title("🧊 Denoise a 256-token canvas")
st.caption("Watch 256 positions become text in parallel.")
render_description(
    """
    This page shows the core DiffusionGemma idea: a full 256-token canvas starts
    noisy, then several denoising passes refine many positions in parallel.

    Use the pass selector to move from initialization to final commit. The token
    colors show which positions are still noisy, which ones are accepted, and how
    confidence changes as the canvas stabilizes. The readable output card is the
    best place to track the story-level effect of each pass.
    """,
    references=(
        f"{glossary_link('Canvas', 'Canvas')} · "
        f"{glossary_link('Denoising', 'Denoising')} · "
        f"{glossary_link('Entropy-bound acceptance', 'Entropy-bound acceptance')} · "
        f"{glossary_link('Self-conditioning', 'Self-conditioning')} · "
        f"{glossary_link('Temperature', 'Temperature')} · "
        f"{glossary_link('Gumbel-max sampling', 'Gumbel-max sampling')}"
    ),
)
render_learning_path("pages/1_canvas_denoising.py")


def reset_scenario():
    st.session_state["canvas_tutorial_step"] = 0

with st.sidebar:
    st.markdown("### Experiment")
    scenario_id = st.selectbox(
        "Example",
        options=list(SCENARIOS),
        format_func=lambda key: SCENARIOS[key].name,
        on_change=reset_scenario,
    )
    num_steps = st.slider("Denoising passes", 6, 20, 12)
    entropy_budget = st.slider("Entropy budget (nats)", 8.0, 80.0, 32.0, 2.0)
    temperature = st.slider("Sampling temperature", 0.4, 1.8, 1.0, 0.1)
    seed = st.number_input("Random seed", value=7, step=1)
    st.caption(
        f"Simulated {glossary_link('logits', 'Logits')}; real 256-position block size.",
        unsafe_allow_html=True,
    )

scenario = SCENARIOS[scenario_id]
st.markdown(
    f'<div class="prompt-box">{html.escape(scenario.prompt)} '
    '<span class="blank">___</span></div>', unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def simulate(scenario_key: str, steps: int, budget: float, temp: float, random_seed: int):
    return run_tutorial(
        scenario_id=scenario_key,
        num_steps=steps,
        entropy_budget=budget,
        temperature=temp,
        seed=random_seed,
    )


run = simulate(scenario_id, num_steps, entropy_budget, temperature, int(seed))
state_key = "canvas_tutorial_step"
if state_key not in st.session_state:
    st.session_state[state_key] = 0
st.session_state[state_key] = min(st.session_state[state_key], num_steps)


def move_step(delta: int):
    st.session_state[state_key] = max(0, min(num_steps, st.session_state[state_key] + delta))


def jump_to_commit():
    st.session_state[state_key] = num_steps


step_number = st.session_state[state_key]
st.markdown(
    (
        '<div class="nav-shell">'
        '<div class="nav-topline">'
        '<div><div class="nav-kicker">Canvas pass navigator</div></div>'
        f'<div class="nav-status"><span>Step</span><strong>{step_number}</strong><span>of {num_steps}</span></div>'
        "</div>"
        '<div class="nav-rail">'
        '<div class="nav-stop"><span class="nav-dot"></span><span>Init noise</span></div>'
        '<div class="nav-stop" style="justify-content:center"><span>Scrub the pass or step one move at a time.</span></div>'
        '<div class="nav-stop" style="justify-content:flex-end"><span class="nav-dot commit"></span><span>Commit view</span></div>'
        "</div>"
        "</div>"
    ),
    unsafe_allow_html=True,
)
back, slider_col, forward, final = st.columns([1.2, 5.8, 1.2, 1.8], vertical_alignment="center")
with back:
    st.button(
        "← Back",
        use_container_width=True,
        disabled=st.session_state[state_key] == 0,
        on_click=move_step,
        args=(-1,),
    )
with slider_col:
    st.slider("Pass navigation", 0, num_steps, key=state_key, label_visibility="collapsed")
with forward:
    st.button(
        "Next →",
        use_container_width=True,
        disabled=st.session_state[state_key] == num_steps,
        on_click=move_step,
        args=(1,),
    )
with final:
    st.button("Commit view ↗", use_container_width=True, on_click=jump_to_commit)

step_number = st.session_state[state_key]
stage_labels = {
    "input": "Canvas in",
    "predict": "Predict",
    "sample": "Sample",
    "accept": "Accept / re-noise",
    "self_condition": "Self-condition",
}


def render_tokens(tokens, classes=None, limit=256, compact=False):
    chunks = []
    for index, token in enumerate(tokens[:limit]):
        extra = "" if classes is None else f" {classes[index]}"
        title = html.escape(f"position {index}")
        chunks.append(f'<span class="token{extra}" title="{title}">{html.escape(token)}</span>')
    canvas_class = "canvas compact" if compact else "canvas"
    return f'<div class="{canvas_class}">' + "".join(chunks) + "</div>"


def render_readable_output(tokens, entropy, committed=False):
    visible = []
    for index, token in enumerate(tokens):
        if token == "<eos>":
            break
        if token != "<pad>":
            visible.append((index, token))
    if committed:
        continuation = " ".join(
            f'<span class="output-token committed">{html.escape(token)}</span>'
            for _, token in visible
        )
    else:
        max_entropy = np.log(len(run.vocabulary))
        pieces = []
        for index, token in visible:
            confidence = float(np.clip(1.0 - entropy[index] / max_entropy, 0.0, 1.0))
            hue = round(confidence * 120)
            color = f"hsl({hue} 72% 48%)"
            background = f"hsl({hue} 65% 45% / 0.20)"
            pieces.append(
                f'<span class="output-token tentative" title="confidence {confidence:.0%}" '
                f'style="border:1px solid {color};background:{background}">'
                f'{html.escape(token)}</span>'
            )
        continuation = " ".join(pieces)
    token_state = "committed" if committed else "tentative"
    card_class = "output-card committed" if committed else "output-card"
    state_label = "Committed" if committed else "Denoising"
    detail = "256-token block" if committed else "current argmax guess"
    legend = "" if committed else (
        '<div class="confidence-legend"><span>uncertain</span>'
        '<span class="confidence-ramp"></span><span>confident</span></div>'
    )
    return (
        f'<div class="{card_class}"><div class="output-kicker">'
        f'<span class="output-state {token_state}">{state_label}</span>{detail}</div>'
        f'<p class="output-text"><span class="output-prompt">{html.escape(run.prompt)} </span>'
        f'{continuation}</p>{legend}</div>'
    )


def render_stage_card(number: int, label: str, description: str, tokens, classes, legend: str):
    badge = f"Stage {number}"
    return (
        '<div class="stage-card">'
        '<div class="stage-head">'
        f'<div class="stage-title">{number}. {html.escape(label)}</div>'
        f'<div class="stage-badge">{badge}</div>'
        '</div>'
        f'<div class="stage-body">{description}</div>'
        f'{render_tokens(tokens, classes=classes, compact=True)}'
        f'<div class="stage-legend">{legend}</div>'
        '</div>'
    )


if step_number == 0:
    st.subheader("Step 0 · Initialize with noise")
    st.caption("The prompt is cached; the internal canvas starts as random tokens.")
    st.markdown(
        '<div class="output-card"><div class="output-kicker">Generated output</div>'
        '<p class="output-text output-empty">Nothing yet.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(render_tokens(run.initial_canvas, ["noise"] * 256), unsafe_allow_html=True)
else:
    snap = run.steps[step_number - 1]
    is_final = step_number == num_steps
    st.subheader(f"Pass {step_number} of {num_steps} · Live canvas state")
    if is_final:
        st.success("Generation complete · clean argmax committed")
    st.markdown(
        render_readable_output(snap.argmax_tokens, snap.entropy, committed=is_final),
        unsafe_allow_html=True,
    )
    accepted_pct = snap.accepted_count / 256
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accepted this pass", f"{snap.accepted_count} / 256", f"{accepted_pct:.0%}")
    with m2:
        st.markdown(glossary_link("Entropy used", "Entropy budget"), unsafe_allow_html=True)
        st.metric(
            "Entropy used value",
            f"{snap.budget_used:.1f}",
            f"of {entropy_budget:.0f} nats",
            label_visibility="collapsed",
        )
    with m3:
        st.markdown(glossary_link("Argmax changes", "Argmax"), unsafe_allow_html=True)
        st.metric(
            "Argmax changes value",
            snap.changed_from_previous,
            label_visibility="collapsed",
        )
    with m4:
        st.markdown(glossary_link("Self-conditioning gate", "Self-conditioning gate"), unsafe_allow_html=True)
        st.metric(
            "Self-conditioning gate value",
            f"{snap.self_conditioning:.2f}",
            label_visibility="collapsed",
        )
    predict_classes = [
        "correct" if token == target else "rejected"
        for token, target in zip(snap.argmax_tokens, run.target)
    ]
    sample_classes = [
        "accepted" if sample == target else "neutral"
        for sample, target in zip(snap.sampled_tokens, run.target)
    ]
    output_classes = ["accepted" if kept else "rejected" for kept in snap.accepted]

    st.markdown(
        (
            '<div class="pass-explainer">'
            '<div class="pass-kicker">How this pass transforms the full canvas</div>'
            '<div class="pass-title">Every scrubbed step now shows the whole pipeline at once</div>'
            '<div class="pass-subtitle">Read left-to-right as one denoising pass: the model sees the noisy block, proposes a clean guess, samples candidates, then commits only the confident positions.</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            render_stage_card(
                1,
                stage_labels["input"],
                (
                    f"Incoming {glossary_link('canvas', 'Canvas')} for this pass. "
                    f"{glossary_link('Bidirectional attention', 'Bidirectional attention')} reads all 256 positions together."
                ),
                snap.input_canvas,
                ["noise"] * 256,
                "Gray-blue tokens are still noisy positions entering the pass.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown('<div class="stage-arrow">↓</div>', unsafe_allow_html=True)
        st.markdown(
            render_stage_card(
                3,
                stage_labels["sample"],
                (
                    f"One {glossary_link('Gumbel-max', 'Gumbel-max sampling')} draw per position. "
                    "This injects variation before the acceptance filter decides what survives."
                ),
                snap.sampled_tokens,
                sample_classes,
                "Green sampled tokens happen to match the scripted target; muted tokens are alternative draws.",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            render_stage_card(
                2,
                stage_labels["predict"],
                (
                    f"Parallel {glossary_link('Argmax prediction', 'Argmax')} over the whole block. "
                    "Cyan is already aligned with the target continuation; red is still wrong."
                ),
                snap.argmax_tokens,
                predict_classes,
                "Cyan means the model's best guess is right at that position.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown('<div class="stage-arrow">↓</div>', unsafe_allow_html=True)
        st.markdown(
            render_stage_card(
                4,
                stage_labels["accept"],
                (
                    f"Green positions are accepted into the next {glossary_link('canvas', 'Canvas')}; "
                    "red positions get re-noised and must be solved again next pass."
                ),
                snap.output_canvas,
                output_classes,
                "Green survived the entropy budget. Red was reset for another denoising round.",
            ),
            unsafe_allow_html=True,
        )

    order = np.argsort(snap.entropy)
    colors = [COLORS["accepted"] if snap.accepted[index] else COLORS["rejected"] for index in order]
    fig = go.Figure(go.Bar(x=np.arange(256), y=snap.entropy[order], marker_color=colors, hovertemplate="rank %{x}<br>entropy %{y:.3f}<extra></extra>"))
    fig.update_layout(height=320, title="Acceptance frontier: positions ranked from most to least confident", xaxis_title="Confidence rank", yaxis_title="Entropy (nats)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=COLORS["text"], margin=dict(l=40, r=20, t=50, b=40))
    st.plotly_chart(fig, width="stretch")
    st.progress(
        snap.self_conditioning,
        text=f"Self-conditioning feedback carried into the next pass: {snap.self_conditioning:.2f} / 0.80",
    )
    if is_final:
        st.caption(
            f"Final encoder pass writes the block to the {glossary_link('KV cache', 'KV cache')}.",
            unsafe_allow_html=True,
        )

st.caption("Tutorial simulation: the algorithm mirrors DiffusionGemma, while the story probabilities are scripted.")
