"""Reusable Plotly figure builders for DiffusionGemma visualizations."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .styles import COLORS


def entropy_heatmap(
    entropies: np.ndarray,
    canvas_size: int,
    title: str = "Per-Position Entropy",
) -> go.Figure:
    cols = min(canvas_size, 16)
    rows = max(1, canvas_size // cols)
    grid = np.full(rows * cols, np.nan)
    grid[:canvas_size] = entropies
    grid = grid.reshape(rows, cols)

    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            colorscale=[
                [0.0, COLORS["confident"]],
                [0.5, "#f39c12"],
                [1.0, COLORS["noisy"]],
            ],
            showscale=True,
            colorbar=dict(title="Entropy"),
            hovertemplate="Position %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=200 + rows * 30,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def entropy_bar_chart(
    entropy: np.ndarray,
    budget: float | None = None,
    accepted: np.ndarray | None = None,
    title: str = "Per-Position Entropy",
) -> go.Figure:
    n = len(entropy)
    colors = [
        COLORS["accepted"] if (accepted is not None and accepted[i])
        else COLORS["rejected"]
        for i in range(n)
    ]
    if accepted is None:
        colors = [COLORS["neutral"]] * n

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(range(n)),
            y=entropy,
            marker_color=colors,
            hovertemplate="Pos %{x}: %{y:.4f}<extra></extra>",
        )
    )
    if budget is not None:
        fig.add_hline(
            y=budget,
            line_dash="dash",
            line_color=COLORS["noisy"],
            annotation_text=f"Budget = {budget:.3f}",
            annotation_font_color=COLORS["noisy"],
        )
    fig.update_layout(
        title=title,
        xaxis_title="Position",
        yaxis_title="Entropy",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
    )
    return fig


def acceptance_curve(entropy: np.ndarray, budget: float) -> go.Figure:
    order = np.argsort(entropy)
    sorted_ent = entropy[order]
    cumulative = np.cumsum(sorted_ent)

    colors = [
        COLORS["accepted"] if v <= budget else COLORS["rejected"]
        for v in cumulative
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(cumulative))),
            y=cumulative,
            mode="lines+markers",
            marker=dict(color=colors, size=6),
            line=dict(color=COLORS["neutral"], width=1),
            hovertemplate="Rank %{x}: cum=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=budget,
        line_dash="dash",
        line_color=COLORS["noisy"],
        annotation_text=f"Budget = {budget:.3f}",
        annotation_font_color=COLORS["noisy"],
    )
    n_accepted = int(np.sum(cumulative <= budget))
    fig.add_vline(
        x=n_accepted - 0.5,
        line_dash="dot",
        line_color=COLORS["accepted"],
        annotation_text=f"Accepted: {n_accepted}/{len(entropy)}",
        annotation_font_color=COLORS["accepted"],
    )
    fig.update_layout(
        title="Cumulative Entropy (sorted by confidence)",
        xaxis_title="Position rank (most confident first)",
        yaxis_title="Cumulative entropy",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
    )
    return fig


def attention_mask_heatmap(
    mask: np.ndarray,
    title: str = "Attention Mask",
    query_labels: list[str] | None = None,
    key_labels: list[str] | None = None,
) -> go.Figure:
    display = np.where(mask, 0.0, 1.0)  # 1 = attended, 0 = masked
    fig = go.Figure(
        data=go.Heatmap(
            z=display,
            colorscale=[
                [0.0, COLORS["masked"]],
                [1.0, COLORS["encoder"]],
            ],
            showscale=False,
            hovertemplate="Q%{y} → K%{x}: %{z:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Key position",
        yaxis_title="Query position",
        height=400,
        width=400,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor="#333", dtick=1),
        yaxis=dict(gridcolor="#333", dtick=1, autorange="reversed"),
    )
    return fig


def timeline_gantt(phases: list[dict]) -> go.Figure:
    """Phases: list of {task, start, finish, color, block}."""
    fig = go.Figure()
    for p in phases:
        fig.add_trace(
            go.Bar(
                x=[p["finish"] - p["start"]],
                y=[p["task"]],
                base=[p["start"]],
                orientation="h",
                marker_color=p.get("color", COLORS["encoder"]),
                name=p.get("block", ""),
                hovertemplate=(
                    f"{p['task']}<br>"
                    f"Start: {p['start']:.1f}<br>"
                    f"End: {p['finish']:.1f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Block Sampling Timeline",
        xaxis_title="Time (forward passes)",
        yaxis_title="",
        barmode="overlay",
        height=200 + len(phases) * 40,
        margin=dict(l=120, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        xaxis=dict(gridcolor="#333"),
        showlegend=False,
    )
    return fig


def kv_cache_bar(committed: list[int], total: int) -> go.Figure:
    """Horizontal stacked bar showing KV cache fill."""
    fig = go.Figure()
    offset = 0
    for i, count in enumerate(committed):
        fig.add_trace(
            go.Bar(
                x=[count],
                y=["KV Cache"],
                base=[offset],
                orientation="h",
                marker_color=COLORS["commit"],
                name=f"Block {i}",
                hovertemplate=f"Block {i}: {count} tokens<extra></extra>",
            )
        )
        offset += count

    if offset < total:
        fig.add_trace(
            go.Bar(
                x=[total - offset],
                y=["KV Cache"],
                base=[offset],
                orientation="h",
                marker_color=COLORS["masked"],
                name="Empty",
                hovertemplate="Available<extra></extra>",
            )
        )
    fig.update_layout(
        title="KV Cache Utilization",
        xaxis=dict(range=[0, total], title="Tokens", gridcolor="#333"),
        yaxis=dict(visible=False),
        barmode="overlay",
        height=80,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"]),
        showlegend=False,
    )
    return fig
