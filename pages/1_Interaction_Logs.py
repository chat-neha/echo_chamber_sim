"""
Interaction Logs Page — Echo Chamber Simulation
Shows conversation logs for Active Nudge and Passive Nudge scenarios.
"""

import streamlit as st
import json
from pathlib import Path
from simulation.visualize import STANCE_COLORS

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interaction Logs — Echo Chamber Sim",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS (same as main page)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0F0F1A;
    color: #E0E0E0;
}
.stApp { background: linear-gradient(135deg, #0F0F1A 0%, #0D1B2A 50%, #0F0F1A 100%); }
h1, h2, h3 { font-family: 'Space Mono', monospace; color: #7EB8F7; }

.hero-banner {
    background: linear-gradient(90deg, rgba(126,184,247,0.08) 0%, rgba(79,195,247,0.05) 100%);
    border-left: 4px solid #7EB8F7;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin-bottom: 24px;
}

.conv-bubble-a {
    background: rgba(126, 184, 247, 0.1);
    border: 1px solid rgba(126, 184, 247, 0.25);
    border-radius: 4px 16px 16px 16px;
    padding: 10px 14px;
    margin: 6px 40px 6px 0;
    font-size: 0.9rem;
}

.conv-bubble-b {
    background: rgba(79, 195, 247, 0.06);
    border: 1px solid rgba(79, 195, 247, 0.2);
    border-radius: 16px 4px 16px 16px;
    padding: 10px 14px;
    margin: 6px 0 6px 40px;
    font-size: 0.9rem;
    text-align: right;
}

.speaker-a { color: #7EB8F7; font-weight: 600; font-size: 0.82rem; margin-bottom: 4px; }
.speaker-b { color: #4FC3F7; font-weight: 600; font-size: 0.82rem; margin-bottom: 4px; text-align: right; }

.pair-header {
    background: rgba(30, 30, 60, 0.8);
    border: 1px solid rgba(126, 184, 247, 0.2);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
}

.stance-flat { color: #FF4C4C; font-weight: 600; }
.stance-neutral { color: #FFD700; font-weight: 600; }
.stance-round { color: #4CAF50; font-weight: 600; }

.delta-positive { color: #4CAF50; font-weight: 600; }
.delta-negative { color: #FF4C4C; font-weight: 600; }

.nudge-card {
    background: rgba(20, 20, 40, 0.9);
    border: 1px solid rgba(126, 184, 247, 0.15);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.nudge-content {
    background: rgba(79, 195, 247, 0.06);
    border-left: 3px solid #4FC3F7;
    padding: 10px 14px;
    margin: 10px 0;
    border-radius: 0 8px 8px 0;
    font-style: italic;
    font-size: 0.88rem;
    color: #B0D4F0;
}

.reflection {
    background: rgba(255, 215, 0, 0.05);
    border-left: 3px solid #FFD700;
    padding: 10px 14px;
    margin: 10px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
}

.pramana-badge {
    display: inline-block;
    background: rgba(126, 184, 247, 0.15);
    border: 1px solid rgba(126, 184, 247, 0.3);
    border-radius: 16px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #B0D4F7;
    font-family: 'Space Mono', monospace;
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: rgba(30,30,60,0.6);
    border-radius: 8px 8px 0 0;
    border: 1px solid rgba(126,184,247,0.2);
    color: #9E9EBB;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: rgba(126,184,247,0.15) !important;
    color: #7EB8F7 !important;
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
}
.stButton > button:hover {
    background: linear-gradient(135deg, #7EB8F7, #4FC3F7);
    color: #0F0F1A;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #0F0F1A 100%);
    border-right: 1px solid rgba(126,184,247,0.1);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.1rem; color: #7EB8F7; font-weight: 700;'>
            🌍 ECHO CHAMBER<br>SIMULATION
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
    st.markdown("""
    <div style='font-size:0.72rem; color:#5A5A7A; text-align:center;'>
        Run simulation from the main page to refresh logs.
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load state
# ──────────────────────────────────────────────────────────────────────────────
def load_state():
    """Load from session state or disk."""
    if "sim_state" in st.session_state and st.session_state.sim_state:
        return st.session_state.sim_state
    path = Path("sim_data/simulation_state.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


state = load_state()


st.markdown("""
<div class='hero-banner'>
    <h1 style='margin:0; font-size:1.5rem;'>📜 Agent Interaction Logs</h1>
    <p style='margin:4px 0 0 0; color:#9E9EBB; font-size:0.9rem;'>
        Detailed conversation transcripts and reflection logs from both nudge scenarios.
    </p>
</div>
""", unsafe_allow_html=True)

if state is None:
    st.warning("⚠️ No simulation data found. Go to the **Network Visualization** page and click **▶ Run New Simulation**.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Helper: stance label to HTML
# ──────────────────────────────────────────────────────────────────────────────
def stance_html(score: float, label: str) -> str:
    cls = "stance-flat" if score <= -0.33 else ("stance-round" if score >= 0.33 else "stance-neutral")
    return f"<span class='{cls}'>{label}</span>"


def delta_html(delta: float) -> str:
    cls = "delta-positive" if delta >= 0 else "delta-negative"
    sign = "+" if delta >= 0 else ""
    return f"<span class='{cls}'>{sign}{delta:.3f}</span>"


# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_active, tab_passive = st.tabs([
    "⚡  Active Nudge Conversations",
    "🌿  Passive Nudge Reflections",
])


# ──────────────────────────────────────────────────────────────────────────────
# ACTIVE NUDGE LOGS
# ──────────────────────────────────────────────────────────────────────────────
with tab_active:
    active_logs = state["active_nudge"].get("logs", [])

    st.markdown(f"""
    <div style='background: rgba(30,30,60,0.5); border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;'>
        <strong style='color:#7EB8F7;'>Active Nudge Strategy</strong><br>
        <span style='font-size:0.88rem; color:#C0D8F0;'>
            Pairs agents with <em>high embedding similarity</em> but <em>differing stances</em> for direct conversations.
            The compatibility score balances contextual similarity with stance divergence.
        </span>
        <br><br>
        <strong style='color:#FFD700;'>{len(active_logs)}</strong> agent pairs interacted.
    </div>
    """, unsafe_allow_html=True)

    if not active_logs:
        st.info("No active nudge logs found.")
    else:
        for log in active_logs:
            pair_idx = log["pair_index"]
            a = log["agent_a"]
            b = log["agent_b"]
            compat = log.get("compatibility_score", 0)
            conversation = log.get("conversation", [])

            with st.expander(
                f"Pair {pair_idx:02d} — {a['name']} ↔ {b['name']}  |  "
                f"Compat: {compat:.3f}",
                expanded=(pair_idx == 1),
            ):
                # Agent cards
                col_a, col_mid, col_b = st.columns([5, 1, 5])

                with col_a:
                    delta_a = a["final_stance"] - a["initial_stance"]
                    st.markdown(f"""
                    <div class='pair-header'>
                        <div style='font-weight:700; color:#7EB8F7; font-size:1rem;'>{a['name']}</div>
                        <div style='font-size:0.82rem; color:#9E9EBB;'>ID #{a['id']}</div>
                        <div style='margin-top:8px;'>
                            <span style='font-size:0.8rem; color:#9E9EBB;'>Before:</span>
                            {stance_html(a['initial_stance'], a.get('initial_label', ''))} ({a['initial_stance']:.2f})
                        </div>
                        <div>
                            <span style='font-size:0.8rem; color:#9E9EBB;'>After:</span>
                            {stance_html(a['final_stance'], a['label'])} ({a['final_stance']:.2f})
                        </div>
                        <div style='margin-top:4px;'>
                            <span style='font-size:0.8rem; color:#9E9EBB;'>Shift:</span> {delta_html(delta_a)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_mid:
                    st.markdown("<div style='text-align:center; padding-top:30px; font-size:1.5rem;'>↔</div>",
                                unsafe_allow_html=True)

                with col_b:
                    delta_b = b["final_stance"] - b["initial_stance"]
                    st.markdown(f"""
                    <div class='pair-header'>
                        <div style='font-weight:700; color:#4FC3F7; font-size:1rem;'>{b['name']}</div>
                        <div style='font-size:0.82rem; color:#9E9EBB;'>ID #{b['id']}</div>
                        <div style='margin-top:8px;'>
                            <span style='font-size:0.8rem; color:#9E9EBB;'>Before:</span>
                            {stance_html(b['initial_stance'], b.get('initial_label', ''))} ({b['initial_stance']:.2f})
                        </div>
                        <div>
                            <span style='font-size:0.8rem; color:#9E9EBB;'>After:</span>
                            {stance_html(b['final_stance'], b['label'])} ({b['final_stance']:.2f})
                        </div>
                        <div style='margin-top:4px;'>
                            <span style='font-size:0.8rem; color:#9E9EBB;'>Shift:</span> {delta_html(delta_b)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                st.markdown("**💬 Conversation Transcript**")

                if not conversation:
                    st.caption("No conversation recorded.")
                else:
                    # Detect which speaker is A and which is B
                    first_speaker = conversation[0][0] if conversation else a["name"]

                    for turn_idx, turn in enumerate(conversation):
                        speaker, message = turn[0], turn[1]
                        is_a = (speaker == first_speaker)
                        if is_a:
                            st.markdown(f"""
                            <div class='conv-bubble-a'>
                                <div class='speaker-a'>{speaker}</div>
                                {message}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='conv-bubble-b'>
                                <div class='speaker-b'>{speaker}</div>
                                {message}
                            </div>
                            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PASSIVE NUDGE LOGS
# ──────────────────────────────────────────────────────────────────────────────
with tab_passive:
    passive_logs = state["passive_nudge"].get("logs", [])

    st.markdown(f"""
    <div style='background: rgba(30,30,60,0.5); border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;'>
        <strong style='color:#81C784;'>Passive Nudge Strategy</strong><br>
        <span style='font-size:0.88rem; color:#C0D8F0;'>
            Each agent receives a <em>pramāṇa-tailored</em> epistemological nudge in their feed —
            matched to their knowledge-acquisition style. No direct confrontation; instead,
            the nudge makes epistemically stronger information more salient.
        </span>
        <br><br>
        <strong style='color:#FFD700;'>{len(passive_logs)}</strong> agents received nudges.
    </div>
    """, unsafe_allow_html=True)

    # Filter controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_stance = st.selectbox(
            "Filter by initial stance",
            ["All", "Flat Earther", "Neutral", "Round Earther"],
            key="passive_filter_stance",
        )
    with col_f2:
        filter_pramana = st.selectbox(
            "Filter by pramāṇa type",
            ["All", "Pratyakṣa", "Anumāna", "Śabda", "Upamāna", "Arthāpatti", "Anupalabdhi"],
            key="passive_filter_pramana",
        )

    filtered_logs = passive_logs
    if filter_stance != "All":
        filtered_logs = [l for l in filtered_logs if l["agent"]["initial_label"] == filter_stance]
    if filter_pramana != "All":
        filtered_logs = [l for l in filtered_logs if l["nudge_type"] == filter_pramana]

    st.caption(f"Showing {len(filtered_logs)} of {len(passive_logs)} agent logs")

    if not filtered_logs:
        st.info("No logs match the selected filters.")
    else:
        for log in filtered_logs:
            agent = log["agent"]
            nudge_type = log["nudge_type"]
            nudge_content = log["nudge_content"]
            reflection = log["agent_reflection"]
            shift = log.get("stance_shift", 0)
            initial_score = agent["initial_stance"]
            final_score = agent["final_stance"]

            with st.expander(
                f"{agent['name']} ({agent['occupation']}, {agent['age']}) — "
                f"{agent['initial_label']} → {agent['final_label']}  |  "
                f"Shift: {shift:+.3f}",
                expanded=False,
            ):
                col1, col2 = st.columns([3, 2])

                with col1:
                    st.markdown(f"""
                    <div class='nudge-card'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <div>
                                <span style='font-weight:700; color:#81C784;'>{agent['name']}</span>
                                <span style='color:#9E9EBB; font-size:0.82rem;'> · {agent['occupation']} · Age {agent['age']}</span>
                            </div>
                            <span class='pramana-badge'>{nudge_type}</span>
                        </div>

                        <div style='margin-top:12px; font-size:0.85rem;'>
                            <span style='color:#9E9EBB;'>Stance before:</span>
                            {stance_html(initial_score, agent['initial_label'])} <span style='color:#9E9EBB;'>({initial_score:.2f})</span>
                        </div>
                        <div style='font-size:0.85rem;'>
                            <span style='color:#9E9EBB;'>Stance after:</span>
                            {stance_html(final_score, agent['final_label'])} <span style='color:#9E9EBB;'>({final_score:.2f})</span>
                        </div>
                        <div style='font-size:0.85rem;'>
                            <span style='color:#9E9EBB;'>Stance shift:</span> {delta_html(shift)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    # Mini bar showing stance change
                    import plotly.graph_objects as go
                    fig_mini = go.Figure()
                    fig_mini.add_trace(go.Bar(
                        x=["Before", "After"],
                        y=[initial_score, final_score],
                        marker_color=[
                            "#FF4C4C" if initial_score <= -0.33 else ("#FFD700" if initial_score <= 0.33 else "#4CAF50"),
                            "#FF4C4C" if final_score <= -0.33 else ("#FFD700" if final_score <= 0.33 else "#4CAF50"),
                        ],
                    ))
                    fig_mini.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=150,
                        margin=dict(t=10, b=10, l=10, r=10),
                        yaxis=dict(range=[-1, 1], gridcolor="#333", tickfont=dict(color="#9E9EBB", size=9)),
                        xaxis=dict(tickfont=dict(color="#9E9EBB", size=9)),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_mini, use_container_width=True)

                st.markdown("**📬 Nudge Content (seen in feed)**")
                st.markdown(f"<div class='nudge-content'>{nudge_content}</div>", unsafe_allow_html=True)

                st.markdown("**🧠 Agent's Internal Reflection**")
                st.markdown(f"<div class='reflection'>{reflection}</div>", unsafe_allow_html=True)
