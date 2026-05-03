"""
Echo Chamber Depolarization Simulation — Streamlit App
Main Page: Network Visualization (Initial | Active Nudge | Passive Nudge)
"""
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import json
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
from simulation.visualize import (
    build_plotly_network,
    build_stance_distribution_chart,
    build_polarization_gauge,
    build_stance_score_scatter,
    STANCE_COLORS,
)
from simulation.agents import Agent, Stance, EpistemologicalBase, create_agent_pool
from simulation.network import build_initial_network
from simulation.runner import run_full_simulation, load_simulation_state

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Echo Chamber Sim",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0F0F1A;
    color: #E0E0E0;
}

.stApp {
    background: linear-gradient(135deg, #0F0F1A 0%, #0D1B2A 50%, #0F0F1A 100%);
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    color: #7EB8F7;
}

.metric-card {
    background: rgba(30, 30, 60, 0.8);
    border: 1px solid rgba(126, 184, 247, 0.2);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    backdrop-filter: blur(10px);
}

.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #7EB8F7;
}

.metric-label {
    font-size: 0.8rem;
    color: #9E9EBB;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stance-flat { color: #FF4C4C; font-weight: 600; }
.stance-neutral { color: #FFD700; font-weight: 600; }
.stance-round { color: #4CAF50; font-weight: 600; }

.hero-banner {
    background: linear-gradient(90deg, rgba(126,184,247,0.08) 0%, rgba(79,195,247,0.05) 100%);
    border-left: 4px solid #7EB8F7;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin-bottom: 24px;
}

.pramana-tag {
    display: inline-block;
    background: rgba(126, 184, 247, 0.15);
    border: 1px solid rgba(126, 184, 247, 0.3);
    border-radius: 16px;
    padding: 3px 10px;
    font-size: 0.75rem;
    margin: 2px;
    color: #B0D4F7;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(30, 30, 60, 0.6);
    border-radius: 8px 8px 0 0;
    border: 1px solid rgba(126, 184, 247, 0.2);
    color: #9E9EBB;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    padding: 8px 20px;
}

.stTabs [aria-selected="true"] {
    background: rgba(126, 184, 247, 0.15) !important;
    color: #7EB8F7 !important;
    border-bottom-color: transparent !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1A3A5C, #0D2137);
    border: 1px solid #7EB8F7;
    color: #7EB8F7;
    font-family: 'Space Mono', monospace;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 0.85rem;
    width: 100%;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #7EB8F7, #4FC3F7);
    color: #0F0F1A;
    border-color: transparent;
}

.sidebar-nav-btn {
    margin-top: 12px;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #0F0F1A 100%);
    border-right: 1px solid rgba(126, 184, 247, 0.1);
}

div[data-testid="metric-container"] {
    background: rgba(30, 30, 60, 0.6);
    border: 1px solid rgba(126, 184, 247, 0.15);
    border-radius: 10px;
    padding: 12px;
}

div[data-testid="metric-container"] label {
    color: #9E9EBB !important;
}

div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #7EB8F7 !important;
    font-family: 'Space Mono', monospace;
}

.info-box {
    background: rgba(79, 195, 247, 0.06);
    border: 1px solid rgba(79, 195, 247, 0.25);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #C0D8F0;
}

.log-entry {
    background: rgba(20, 20, 40, 0.8);
    border-left: 3px solid #7EB8F7;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 0 8px 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# State helpers
# ──────────────────────────────────────────────────────────────────────────────
def reconstruct_agents_from_json(agent_list: list) -> list:
    """Reconstruct Agent objects from JSON data."""
    agents = []
    for d in agent_list:
        # We just need lightweight objects for visualization
        class _A:
            pass
        a = _A()
        a.agent_id = d["agent_id"]
        a.name = d["name"]
        a.age = d["age"]
        a.gender = d["gender"]
        a.occupation = d["occupation"]
        a.stance_score = d["stance_score"]
        a.interests = d.get("interests", [])
        a.personality_traits = d.get("personality", [])
        a.epistemological_base = type('obj', (object,), {'value': d["ep_base"]})()

        # Assign stance
        score = d["stance_score"]
        if score <= -0.33:
            a.stance = Stance.FLAT_EARTHER
        elif score >= 0.33:
            a.stance = Stance.ROUND_EARTHER
        else:
            a.stance = Stance.NEUTRAL

        def _label(self=a):
            if self.stance_score <= -0.33:
                return "Flat Earther"
            elif self.stance_score >= 0.33:
                return "Round Earther"
            return "Neutral"
        a.get_stance_label = _label
        a.embedding = np.zeros(16)
        agents.append(a)
    return agents


def reconstruct_graph_from_json(graph_data: dict) -> nx.DiGraph:
    """Reconstruct networkx graph from JSON."""
    G = nx.DiGraph()
    for node in graph_data["nodes"]:
        nid = node["id"]
        G.add_node(nid, **{k: v for k, v in node.items() if k != "id"})
    for edge in graph_data["edges"]:
        G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1.0))
    return G


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.1rem; color: #7EB8F7; font-weight: 700;'>
            🌍 ECHO CHAMBER<br>SIMULATION
        </div>
        <div style='font-size: 0.72rem; color: #5A5A7A; margin-top: 4px; letter-spacing: 2px;'>
            FLAT EARTH · SOCIAL NETWORK
        </div>
    </div>
    <hr style='border-color: rgba(126,184,247,0.15);'/>
    """, unsafe_allow_html=True)

    st.markdown("**Navigation**")
    if st.button("📊 Network Visualization", use_container_width=True):
        st.switch_page("app.py")

    if st.button("📜 Interaction Logs", use_container_width=True):
        st.switch_page("pages/1_Interaction_Logs.py")

    st.markdown("<hr style='border-color: rgba(126,184,247,0.1);'/>", unsafe_allow_html=True)

    st.markdown("**Simulation Controls**")

    run_sim = st.button("▶  Run New Simulation", use_container_width=True, type="primary")

    st.markdown("<hr style='border-color: rgba(126,184,247,0.1);'/>", unsafe_allow_html=True)

    # Legend
    st.markdown("**Stance Legend**")
    for label, color in STANCE_COLORS.items():
        st.markdown(
            f"<span style='color:{color}; font-size:1.1rem;'>●</span> "
            f"<span style='font-size:0.85rem;'>{label}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color: rgba(126,184,247,0.1);'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#5A5A7A; text-align:center;'>
        Based on Indian epistemology (pramāṇa theory).<br>
        50 LLM agents · Directed social graph.
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main Page
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <h1 style='margin:0; font-size:1.5rem;'>Echo Chamber Depolarization Simulator</h1>
    <p style='margin:4px 0 0 0; color:#9E9EBB; font-size:0.9rem;'>
        50 LLM agents · Flat Earth controversy · Indian epistemology (pramāṇa) · Active & Passive Nudge strategies
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Run simulation or load
# ──────────────────────────────────────────────────────────────────────────────
if "sim_state" not in st.session_state:
    st.session_state.sim_state = load_simulation_state()

if run_sim or st.session_state.sim_state is None:
    with st.status("🔬 Running simulation...", expanded=True) as status_box:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def progress_cb(msg, pct=None):
            status_text.write(f"› {msg}")
            if pct is not None:
                progress_bar.progress(pct)

        try:
            state = run_full_simulation(progress_callback=progress_cb)
            st.session_state.sim_state = state
            status_box.update(label="✅ Simulation complete!", state="complete")
        except Exception as e:
            status_box.update(label=f"❌ Error: {e}", state="error")
            st.error(f"Simulation failed: {e}")
            st.stop()

state = st.session_state.sim_state
if state is None:
    st.info("No simulation data found. Click **▶ Run New Simulation** in the sidebar to begin.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruct objects
# ──────────────────────────────────────────────────────────────────────────────
initial_agents = reconstruct_agents_from_json(state["initial"]["agents"])
active_agents = reconstruct_agents_from_json(state["active_nudge"]["agents"])
passive_agents = reconstruct_agents_from_json(state["passive_nudge"]["agents"])

initial_graph = reconstruct_graph_from_json(state["initial"]["graph"])
active_graph = reconstruct_graph_from_json(state["active_nudge"]["graph"])
passive_graph = reconstruct_graph_from_json(state["passive_nudge"]["graph"])


# ──────────────────────────────────────────────────────────────────────────────
# Summary metrics row
# ──────────────────────────────────────────────────────────────────────────────
def stance_counts(agents):
    f = sum(1 for a in agents if a.stance == Stance.FLAT_EARTHER)
    n = sum(1 for a in agents if a.stance == Stance.NEUTRAL)
    r = sum(1 for a in agents if a.stance == Stance.ROUND_EARTHER)
    return f, n, r


i_f, i_n, i_r = stance_counts(initial_agents)
a_f, a_n, a_r = stance_counts(active_agents)
p_f, p_n, p_r = stance_counts(passive_agents)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 🔵 Initial State")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Flat Earthers", i_f)
    mc2.metric("Neutral", i_n)
    mc3.metric("Round Earth", i_r)
    st.metric("Polarization Index", f"{state['initial']['polarization']:.3f}")

with col2:
    st.markdown("### 🟡 After Active Nudge")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Flat Earthers", a_f, delta=a_f - i_f)
    mc2.metric("Neutral", a_n, delta=a_n - i_n)
    mc3.metric("Round Earth", a_r, delta=a_r - i_r)
    st.metric("Polarization Index",
              f"{state['active_nudge']['polarization']:.3f}",
              delta=f"{state['active_nudge']['polarization'] - state['initial']['polarization']:.3f}",
              delta_color="inverse")

with col3:
    st.markdown("### 🟢 After Passive Nudge")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Flat Earthers", p_f, delta=p_f - i_f)
    mc2.metric("Neutral", p_n, delta=p_n - i_n)
    mc3.metric("Round Earth", p_r, delta=p_r - i_r)
    st.metric("Polarization Index",
              f"{state['passive_nudge']['polarization']:.3f}",
              delta=f"{state['passive_nudge']['polarization'] - state['initial']['polarization']:.3f}",
              delta_color="inverse")


st.markdown("<hr style='border-color: rgba(126,184,247,0.1);'/>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Network Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🌐  Initial Network",
    "⚡  After Active Nudge",
    "🌿  After Passive Nudge",
])

with tab1:
    st.markdown("""
    <div class='info-box'>
        <strong>Initial State</strong> — The social network before any intervention.
        Agents are clustered into echo chambers based on stance similarity.
        <span class='stance-flat'>■ Red = Flat Earthers</span> &nbsp;
        <span class='stance-neutral'>◆ Gold = Neutral</span> &nbsp;
        <span class='stance-round'>● Green = Round Earthers</span>
    </div>
    """, unsafe_allow_html=True)
    fig = build_plotly_network(initial_graph, initial_agents, title="Initial Social Network")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Network Statistics**")
    stats = state["initial"]["stats"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", stats["num_nodes"])
    c2.metric("Edges", stats["num_edges"])
    c3.metric("Avg Degree", f"{stats['avg_degree']:.1f}")
    c4.metric("Echo Chamber Score", f"{stats['echo_chamber_score']:.3f}")


with tab2:
    st.markdown("""
    <div class='info-box'>
        <strong>Active Nudge</strong> — Strategically paired agents with high embedding similarity
        but differing stances are introduced to each other. They engage in direct conversation.
        The intuition: people are more receptive to opposing views when they share common ground
        with the messenger.
    </div>
    """, unsafe_allow_html=True)
    fig = build_plotly_network(active_graph, active_agents, title="Network After Active Nudge")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Stance Shift Analysis**")
    fig_scatter = build_stance_score_scatter(initial_agents, active_agents, "Active Nudge")
    st.plotly_chart(fig_scatter, use_container_width=True)

    stats = state["active_nudge"]["stats"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", stats["num_nodes"])
    c2.metric("Edges", stats["num_edges"])
    c3.metric("Avg Degree", f"{stats['avg_degree']:.1f}")
    c4.metric("Echo Chamber Score", f"{stats['echo_chamber_score']:.3f}")


with tab3:
    st.markdown("""
    <div class='info-box'>
        <strong>Passive Nudge</strong> — Agents receive epistemologically-tailored content in their feed
        aligned with their <em>pramāṇa</em> (knowledge source). No direct confrontation.
        Instead, the nudge makes epistemically stronger information more salient and easier to trust.
    </div>
    """, unsafe_allow_html=True)
    fig = build_plotly_network(passive_graph, passive_agents, title="Network After Passive Nudge")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Stance Shift Analysis**")
    fig_scatter = build_stance_score_scatter(initial_agents, passive_agents, "Passive Nudge")
    st.plotly_chart(fig_scatter, use_container_width=True)

    stats = state["passive_nudge"]["stats"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", stats["num_nodes"])
    c2.metric("Edges", stats["num_edges"])
    c3.metric("Avg Degree", f"{stats['avg_degree']:.1f}")
    c4.metric("Echo Chamber Score", f"{stats['echo_chamber_score']:.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# Comparison Charts
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<hr style='border-color: rgba(126,184,247,0.1);'/>", unsafe_allow_html=True)
st.markdown("### 📊 Comparative Analysis")

col_l, col_r = st.columns(2)
with col_l:
    fig_dist = build_stance_distribution_chart(initial_agents, active_agents, passive_agents)
    st.plotly_chart(fig_dist, use_container_width=True)

with col_r:
    fig_pol = build_polarization_gauge(
        state["initial"]["polarization"],
        state["active_nudge"]["polarization"],
        state["passive_nudge"]["polarization"],
    )
    st.plotly_chart(fig_pol, use_container_width=True)
