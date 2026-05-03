"""
Visualization utilities for the Echo Chamber Simulation.
Generates Plotly network graphs and pyvis HTML graphs.
"""

import numpy as np
import networkx as nx
import plotly.graph_objects as go
from typing import List, Optional, Dict
from simulation.agents import Agent, Stance


# ──────────────────────────────────────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────────────────────────────────────
STANCE_COLORS = {
    "Flat Earther":  "#FF4C4C",
    "Neutral":       "#FFD700",
    "Round Earther": "#4CAF50",
}

STANCE_SYMBOLS = {
    "Flat Earther":  "square",
    "Neutral":       "diamond",
    "Round Earther": "circle",
}

EP_BASE_COLORS = {
    "Pratyakṣa":    "#E91E63",
    "Anumāna":      "#9C27B0",
    "Śabda":        "#2196F3",
    "Upamāna":      "#00BCD4",
    "Arthāpatti":   "#4CAF50",
    "Anupalabdhi":  "#FF9800",
}

# ──────────────────────────────────────────────────────────────────────────────
# Belief-zoned layout  (REPLACED from spring layout)
#
#   Flat Earthers  → left   (x ≈ -1.0)
#   Neutral        → centre (x ≈  0.0)
#   Round Earthers → right  (x ≈ +1.0)
#
# Agents are evenly distributed vertically within each column with a small
# random jitter so nodes don't stack. The spring-layout pass is gone — it
# always collapsed the three groups together.
# ──────────────────────────────────────────────────────────────────────────────
_X_CENTER = {"Flat Earther": -1.0, "Neutral": 0.0, "Round Earther": 1.0}
_X_SPREAD = 0.28   # horizontal jitter within a column
_Y_SPREAD = 0.84   # full vertical range [-Y_SPREAD, +Y_SPREAD]


def _belief_zoned_layout(agents: List[Agent], seed: int = 42) -> Dict[int, tuple]:
    """
    Return {agent_id: (x, y)} with hard horizontal separation by stance label.
    Vertical positions are evenly spaced within each group + small jitter.
    """
    rng = np.random.RandomState(seed)

    # Bucket agents by stance label, sort for determinism
    groups: Dict[str, List[Agent]] = {
        "Flat Earther":  [],
        "Neutral":       [],
        "Round Earther": [],
    }
    for a in agents:
        label = a.get_stance_label()
        if label in groups:
            groups[label].append(a)
    for k in groups:
        groups[k].sort(key=lambda a: a.agent_id)

    pos: Dict[int, tuple] = {}
    for label, members in groups.items():
        n = len(members)
        x_center = _X_CENTER[label]
        for i, a in enumerate(members):
            # evenly spaced vertically, small random jitter
            y_base = -_Y_SPREAD + 2 * _Y_SPREAD * (i / max(n - 1, 1))
            x = x_center + rng.uniform(-_X_SPREAD, _X_SPREAD)
            y = y_base   + rng.uniform(-0.10, 0.10)
            pos[a.agent_id] = (float(x), float(y))

    return pos


def build_plotly_network(
    G: nx.DiGraph,
    agents: List[Agent],
    title: str = "Social Network",
    show_ep_base: bool = False,
) -> go.Figure:
    """
    Build an interactive Plotly network visualization with belief-zoned layout.

    Flat Earthers cluster on the LEFT, Neutral in the CENTRE,
    Round Earthers on the RIGHT — making echo chambers immediately visible.
    Cross-belief edges are drawn brighter than same-belief edges.
    Coloured zone bands + dotted dividers mark each region.
    """
    agent_map = {a.agent_id: a for a in agents}
    pos = _belief_zoned_layout(agents)

    # ── Zone background bands ──────────────────────────────────────────────
    zone_shapes = [
        dict(type="rect", xref="x", yref="paper",
             x0=-1.45, x1=-0.50, y0=0, y1=1,
             fillcolor="rgba(255,76,76,0.06)", line_width=0),
        dict(type="rect", xref="x", yref="paper",
             x0=-0.50, x1=0.50, y0=0, y1=1,
             fillcolor="rgba(255,215,0,0.04)", line_width=0),
        dict(type="rect", xref="x", yref="paper",
             x0=0.50, x1=1.45, y0=0, y1=1,
             fillcolor="rgba(76,175,80,0.06)", line_width=0),
        # dotted dividers
        dict(type="line", xref="x", yref="paper",
             x0=-0.50, x1=-0.50, y0=0.02, y1=0.98,
             line=dict(color="rgba(255,76,76,0.25)", width=1, dash="dot")),
        dict(type="line", xref="x", yref="paper",
             x0=0.50, x1=0.50, y0=0.02, y1=0.98,
             line=dict(color="rgba(76,175,80,0.25)", width=1, dash="dot")),
    ]

    zone_annotations = [
        dict(x=-0.97, y=1.07, xref="x", yref="paper",
             text="◀  FLAT-EARTHERS",
             showarrow=False,
             font=dict(color="rgba(255,76,76,0.75)", size=10, family="monospace"),
             xanchor="center"),
        dict(x=0.0, y=1.07, xref="x", yref="paper",
             text="NEUTRAL",
             showarrow=False,
             font=dict(color="rgba(255,215,0,0.75)", size=10, family="monospace"),
             xanchor="center"),
        dict(x=0.97, y=1.07, xref="x", yref="paper",
             text="ROUND-EARTHERS  ▶",
             showarrow=False,
             font=dict(color="rgba(76,175,80,0.75)", size=10, family="monospace"),
             xanchor="center"),
    ]

    # ── Edge traces — same-belief vs cross-belief ──────────────────────────
    id_to_label = {a.agent_id: a.get_stance_label() for a in agents}
    ex_same, ey_same = [], []
    ex_cross, ey_cross = [], []

    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        if id_to_label.get(u) == id_to_label.get(v):
            ex_same  += [x0, x1, None]
            ey_same  += [y0, y1, None]
        else:
            ex_cross += [x0, x1, None]
            ey_cross += [y0, y1, None]

    edge_traces = []
    if ex_same:
        edge_traces.append(go.Scatter(
            x=ex_same, y=ey_same, mode="lines",
            line=dict(width=0.4, color="rgba(150,150,150,0.18)"),
            hoverinfo="none", showlegend=False,
        ))
    if ex_cross:
        edge_traces.append(go.Scatter(
            x=ex_cross, y=ey_cross, mode="lines",
            line=dict(width=0.9, color="rgba(200,200,255,0.35)"),
            hoverinfo="none", showlegend=False,
        ))

    # ── Node traces (grouped by stance for legend) ─────────────────────────
    node_traces = []
    groups: Dict[str, list] = {label: [] for label in STANCE_COLORS}

    for node_id in G.nodes():
        if node_id not in agent_map:
            continue
        agent = agent_map[node_id]
        label = agent.get_stance_label()
        if node_id not in pos:
            continue
        x, y = pos[node_id]
        hover = (
            f"<b>{agent.name}</b><br>"
            f"Age: {agent.age} | {agent.gender}<br>"
            f"Occupation: {agent.occupation}<br>"
            f"Stance: {label} ({agent.stance_score:.2f})<br>"
            f"Epistemology: {agent.epistemological_base.value}<br>"
            f"Interests: {', '.join(agent.interests[:2])}"
        )
        groups[label].append((x, y, hover, node_id))

    for label, nodes in groups.items():
        if not nodes:
            continue
        xs     = [n[0] for n in nodes]
        ys     = [n[1] for n in nodes]
        hovers = [n[2] for n in nodes]
        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            name=label,
            marker=dict(
                size=12,
                color=STANCE_COLORS[label],
                symbol=STANCE_SYMBOLS[label],
                line=dict(width=1.5, color="white"),
                opacity=0.90,
            ),
            text=hovers,
            hoverinfo="text",
        ))

    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16, color="#E0E0E0"), x=0.5),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#E0E0E0"),
            ),
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=60),
            paper_bgcolor="#0F0F1A",
            plot_bgcolor="#0F0F1A",
            shapes=zone_shapes,
            annotations=zone_annotations,
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[-1.45, 1.45],
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[-1.05, 1.05],
            ),
            height=600,
        ),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Everything below this line is UNCHANGED from the original file
# ──────────────────────────────────────────────────────────────────────────────

def build_stance_distribution_chart(
    agents_before: List[Agent],
    agents_after_active: List[Agent],
    agents_after_passive: List[Agent],
) -> go.Figure:
    """Bar chart comparing stance distributions across scenarios."""

    def count_stances(agents):
        flat    = sum(1 for a in agents if a.stance == Stance.FLAT_EARTHER)
        neutral = sum(1 for a in agents if a.stance == Stance.NEUTRAL)
        rnd     = sum(1 for a in agents if a.stance == Stance.ROUND_EARTHER)
        return flat, neutral, rnd

    b_f, b_n, b_r = count_stances(agents_before)
    a_f, a_n, a_r = count_stances(agents_after_active)
    p_f, p_n, p_r = count_stances(agents_after_passive)

    categories = ["Initial", "After Active Nudge", "After Passive Nudge"]

    fig = go.Figure(data=[
        go.Bar(name="Flat Earthers",  x=categories, y=[b_f, a_f, p_f],
               marker_color="#FF4C4C"),
        go.Bar(name="Neutral",        x=categories, y=[b_n, a_n, p_n],
               marker_color="#FFD700"),
        go.Bar(name="Round Earthers", x=categories, y=[b_r, a_r, p_r],
               marker_color="#4CAF50"),
    ])

    fig.update_layout(
        barmode="group",
        title=dict(text="Stance Distribution Comparison",
                   font=dict(color="#E0E0E0"), x=0.5),
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#1A1A2E",
        font=dict(color="#E0E0E0"),
        legend=dict(font=dict(color="#E0E0E0")),
        xaxis=dict(gridcolor="#333366"),
        yaxis=dict(gridcolor="#333366", title="Number of Agents"),
        height=400,
    )
    return fig


def build_polarization_gauge(
    initial_pol: float,
    active_pol: float,
    passive_pol: float,
) -> go.Figure:
    """Gauge charts for polarization index comparison."""

    fig = go.Figure()

    for val, label, color, x_pos in [
        (initial_pol,  "Initial",       "#888888", 0.15),
        (active_pol,   "Active Nudge",  "#4FC3F7", 0.5),
        (passive_pol,  "Passive Nudge", "#81C784", 0.85),
    ]:
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            title={"text": label, "font": {"color": "#E0E0E0", "size": 13}},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": "#E0E0E0"},
                "bar": {"color": color},
                "bgcolor": "#1A1A2E",
                "bordercolor": "#333366",
                "steps": [
                    {"range": [0, 0.33],  "color": "#1A3A1A"},
                    {"range": [0.33, 0.66], "color": "#2A2A1A"},
                    {"range": [0.66, 1.0], "color": "#3A1A1A"},
                ],
            },
            number={"font": {"color": "#E0E0E0"}},
            domain={"x": [x_pos - 0.13, x_pos + 0.13], "y": [0, 1]},
        ))

    fig.update_layout(
        paper_bgcolor="#0F0F1A",
        height=280,
        title=dict(
            text="Polarization Index (lower = less polarized)",
            font=dict(color="#E0E0E0", size=14),
            x=0.5,
        ),
    )
    return fig


def build_stance_score_scatter(
    agents_before: List[Agent],
    agents_after: List[Agent],
    scenario_label: str = "Active Nudge",
) -> go.Figure:
    """Scatter plot showing individual stance score changes."""

    before_map = {a.agent_id: a for a in agents_before}

    x_names, y_before, y_after, colors, hover = [], [], [], [], []

    for a in sorted(agents_after, key=lambda x: x.stance_score):
        b = before_map.get(a.agent_id)
        if not b:
            continue
        x_names.append(a.name.split()[0])
        y_before.append(b.stance_score)
        y_after.append(a.stance_score)
        delta = a.stance_score - b.stance_score
        colors.append(
            "#4FC3F7" if delta > 0 else
            "#FF4C4C" if delta < 0 else
            "#888"
        )
        hover.append(
            f"{a.name}<br>"
            f"Before: {b.stance_score:.2f}<br>"
            f"After: {a.stance_score:.2f}<br>"
            f"Δ: {delta:+.2f}"
        )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_names, y=y_before,
        mode="markers",
        name="Before",
        marker=dict(size=8, color="rgba(200,200,200,0.5)", symbol="circle-open"),
        text=hover,
        hoverinfo="text",
    ))

    fig.add_trace(go.Scatter(
        x=x_names, y=y_after,
        mode="markers",
        name=f"After {scenario_label}",
        marker=dict(size=10, color=colors, symbol="circle"),
        text=hover,
        hoverinfo="text",
    ))

    fig.add_hline(y=0,     line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig.add_hline(y=-0.33, line_dash="dot",  line_color="rgba(255,76,76,0.4)",
                  annotation_text="Flat Earth boundary",
                  annotation_font_color="#FF4C4C")
    fig.add_hline(y=0.33,  line_dash="dot",  line_color="rgba(76,175,80,0.4)",
                  annotation_text="Round Earth boundary",
                  annotation_font_color="#4CAF50")

    fig.update_layout(
        title=dict(
            text=f"Individual Stance Shifts — {scenario_label}",
            font=dict(color="#E0E0E0"), x=0.5,
        ),
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#1A1A2E",
        font=dict(color="#E0E0E0"),
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=9),
            gridcolor="#333366",
            showgrid=False,
        ),
        yaxis=dict(
            title="Stance Score",
            range=[-1.1, 1.1],
            gridcolor="#333366",
        ),
        height=420,
        legend=dict(font=dict(color="#E0E0E0")),
    )
    return fig