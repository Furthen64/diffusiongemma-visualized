import html

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.canvas_tutorial import SCENARIOS, run_tutorial
from utils.glossary import glossary_link
from utils.styles import COLORS, inject_styles


st.set_page_config(page_title="Canvas Denoising", page_icon="🧊", layout="wide")
inject_styles()
st.markdown("""
<style>
.prompt-box {background:#10182e;border:1px solid #3b4b70;border-radius:12px;padding:1rem 1.2rem;font:1.05rem ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.blank {color:#ffbd59;font-weight:700;border-bottom:2px solid #ffbd59;padding:0 .25rem}
.canvas {display:flex;flex-wrap:wrap;gap:3px;line-height:1.25;margin:.4rem 0 1rem}
.token {font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.72rem;padding:3px 5px;border-radius:4px;background:#26304b;color:#dce5fa;border:1px solid transparent}
.token.accepted {background:#153d32;border-color:#2ecc71;color:#baf5d5}
.token.rejected {background:#40232c;border-color:#e76b78;color:#ffd2d7}
.token.correct {background:#193b42;border-color:#34b9ca;color:#c7f8ff}
.token.noise {color:#77829c}
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
@media(max-width:700px){.token{font-size:.66rem}}
</style>
""", unsafe_allow_html=True)

st.title("🧊 Denoise a 256-token canvas")
st.caption("Watch 256 positions become text in parallel.")
st.markdown(
    f"Reference: {glossary_link('Canvas', 'Canvas')} · "
    f"{glossary_link('Denoising', 'Denoising')} · "
    f"{glossary_link('Entropy-bound acceptance', 'Entropy-bound acceptance')} · "
    f"{glossary_link('Self-conditioning', 'Self-conditioning')} · "
    f"{glossary_link('Temperature', 'Temperature')} · "
    f"{glossary_link('Gumbel-max sampling', 'Gumbel-max sampling')}",
    unsafe_allow_html=True,
)


def reset_scenario():
    st.session_state["canvas_tutorial_step"] = 0
    st.session_state["canvas_tutorial_stage"] = "input"

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
    st.session_state["canvas_tutorial_stage"] = "self_condition"


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
stage_key = "canvas_tutorial_stage"
stage_labels = {
    "input": "1 · Canvas in",
    "predict": "2 · Predict",
    "sample": "3 · Sample",
    "accept": "4 · Accept / re-noise",
    "self_condition": "5 · Self-condition",
}
if stage_key not in st.session_state:
    st.session_state[stage_key] = "input"

st.markdown("**Stage inside this pass**")
st.segmented_control(
    "Stage inside this pass",
    options=list(stage_labels),
    format_func=stage_labels.get,
    key=stage_key,
    disabled=step_number == 0,
    label_visibility="collapsed",
    width="stretch",
)
selected_stage = st.session_state[stage_key]


def render_tokens(tokens, classes=None, limit=256):
    chunks = []
    for index, token in enumerate(tokens[:limit]):
        extra = "" if classes is None else f" {classes[index]}"
        title = html.escape(f"position {index}")
        chunks.append(f'<span class="token{extra}" title="{title}">{html.escape(token)}</span>')
    return '<div class="canvas">' + "".join(chunks) + "</div>"


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
    st.subheader(f"Pass {step_number} of {num_steps} · {stage_labels[selected_stage]}")
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

    if selected_stage == "input":
        st.markdown(
            f"**{glossary_link('Canvas', 'Canvas')} entering this pass** · "
            f"{glossary_link('Bidirectional attention', 'Bidirectional attention')} "
            "reads all positions together.",
            unsafe_allow_html=True,
        )
        st.markdown(render_tokens(snap.input_canvas, ["noise"] * 256), unsafe_allow_html=True)

    elif selected_stage == "predict":
        st.markdown(
            f"**{glossary_link('Argmax prediction', 'Argmax')}** · "
            "cyan matches the scripted target; red does not.",
            unsafe_allow_html=True,
        )
        classes = [
            "correct" if token == target else "rejected"
            for token, target in zip(snap.argmax_tokens, run.target)
        ]
        st.markdown(render_tokens(snap.argmax_tokens, classes), unsafe_allow_html=True)

    elif selected_stage == "sample":
        st.markdown(
            f"**Sampled candidates** · one {glossary_link('Gumbel-max', 'Gumbel-max sampling')} "
            "draw per position.",
            unsafe_allow_html=True,
        )
        st.markdown(render_tokens(snap.sampled_tokens, ["noise"] * 256), unsafe_allow_html=True)

    elif selected_stage == "accept":
        st.markdown(
            f"**Next {glossary_link('canvas', 'Canvas')}** · green candidates are "
            "kept; red positions are re-noised.",
            unsafe_allow_html=True,
        )
        classes = ["accepted" if kept else "rejected" for kept in snap.accepted]
        st.markdown(render_tokens(snap.output_canvas, classes), unsafe_allow_html=True)
        order = np.argsort(snap.entropy)
        colors = [COLORS["accepted"] if snap.accepted[index] else COLORS["rejected"] for index in order]
        fig = go.Figure(go.Bar(x=np.arange(256), y=snap.entropy[order], marker_color=colors, hovertemplate="rank %{x}<br>entropy %{y:.3f}<extra></extra>"))
        fig.update_layout(height=320, title="Positions ranked from most to least confident", xaxis_title="Confidence rank", yaxis_title="Entropy (nats)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=COLORS["text"], margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig, width="stretch")

    else:
        st.markdown(
            f"**Soft feedback** · the full prediction distribution conditions the "
            f"next pass through {glossary_link('self-conditioning', 'Self-conditioning')}.",
            unsafe_allow_html=True,
        )
        classes = ["accepted" if kept else "rejected" for kept in snap.accepted]
        st.markdown(render_tokens(snap.output_canvas, classes), unsafe_allow_html=True)
        st.progress(
            snap.self_conditioning,
            text=f"Self-conditioning gate: {snap.self_conditioning:.2f} / 0.80",
        )
        if is_final:
            st.caption(
                f"Final encoder pass writes the block to the {glossary_link('KV cache', 'KV cache')}.",
                unsafe_allow_html=True,
            )

st.caption("Tutorial simulation: the algorithm mirrors DiffusionGemma, while the story probabilities are scripted.")
