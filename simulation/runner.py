"""
Main simulation runner.
Orchestrates agent creation, network building, nudge scenarios, and saves all state.
"""

import os
import json
import copy
import logging
import numpy as np
from pathlib import Path

from simulation.agents import create_agent_pool, Agent, Stance
from simulation.network import (
    build_initial_network,
    compute_network_stats,
    compute_polarization_index,
)
from simulation.nudges import run_active_nudge, run_passive_nudge

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("sim_data")


def agents_to_json(agents: list) -> list:
    """Serialize agents to JSON-safe dicts."""
    result = []
    for a in agents:
        result.append({
            "agent_id": a.agent_id,
            "name": a.name,
            "age": a.age,
            "gender": a.gender,
            "occupation": a.occupation,
            "education": a.education,
            "social_media_usage": a.social_media_usage,
            "stance": a.stance.value,
            "stance_score": round(float(a.stance_score), 4),
            "stance_label": a.get_stance_label(),
            "ep_base": a.epistemological_base.value,
            "interests": a.interests,
            "personality": a.personality_traits,
        })
    return result


def graph_to_json(G) -> dict:
    """Serialize networkx graph to JSON-safe dict."""
    return {
        "nodes": [
            {
                "id": n,
                **{k: (v if not isinstance(v, list) else v) for k, v in G.nodes[n].items()},
            }
            for n in G.nodes()
        ],
        "edges": [
            {"source": u, "target": v, "weight": round(float(d.get("weight", 1.0)), 4)}
            for u, v, d in G.edges(data=True)
        ],
    }


def run_full_simulation(progress_callback=None) -> dict:
    """
    Run the complete simulation pipeline.
    Returns a dict with all state for the Streamlit app.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    def progress(msg, pct=None):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg, pct)

    # ── Step 1: Create agents ──
    progress("Creating 50 agent personas...", 0.05)
    agents = create_agent_pool()

    # ── Step 2: Build initial network ──
    progress("Building initial social network graph...", 0.15)
    initial_graph = build_initial_network(agents, echo_chamber_strength=0.75)

    initial_stats = compute_network_stats(initial_graph, agents)
    initial_pol = compute_polarization_index(agents)

    progress("Initial network built.", 0.20)

    # ── Step 3: Active Nudge ──
    progress("Running Active Nudge scenario...", 0.30)
    active_agents, active_logs, active_graph = run_active_nudge(
        agents, num_pairs=15, interaction_turns=3
    )
    active_stats = compute_network_stats(active_graph, active_agents)
    active_pol = compute_polarization_index(active_agents)
    progress(f"Active Nudge complete. {len(active_logs)} pairs interacted.", 0.60)

    # ── Step 4: Passive Nudge ──
    progress("Running Passive Nudge scenario...", 0.65)
    passive_agents, passive_logs, passive_graph = run_passive_nudge(
        agents, target_flat_earthers=True, target_neutrals=True
    )
    passive_stats = compute_network_stats(passive_graph, passive_agents)
    passive_pol = compute_polarization_index(passive_agents)
    progress(f"Passive Nudge complete. {len(passive_logs)} agents nudged.", 0.90)

    # ── Step 5: Save all state ──
    progress("Saving simulation state...", 0.95)

    state = {
        "initial": {
            "agents": agents_to_json(agents),
            "graph": graph_to_json(initial_graph),
            "stats": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                      for k, v in initial_stats.items()},
            "polarization": float(initial_pol),
        },
        "active_nudge": {
            "agents": agents_to_json(active_agents),
            "graph": graph_to_json(active_graph),
            "stats": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                      for k, v in active_stats.items()},
            "polarization": float(active_pol),
            "logs": active_logs,
        },
        "passive_nudge": {
            "agents": agents_to_json(passive_agents),
            "graph": graph_to_json(passive_graph),
            "stats": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                      for k, v in passive_stats.items()},
            "polarization": float(passive_pol),
            "logs": passive_logs,
        },
    }

    # Sanitize logs (conversations may have tuples)
    def sanitize(obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, list):
            return [sanitize(i) for i in obj]
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    state = sanitize(state)

    with open(OUTPUT_DIR / "simulation_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    progress("Simulation complete!", 1.0)
    return state


def load_simulation_state() -> dict:
    """Load previously saved simulation state."""
    path = OUTPUT_DIR / "simulation_state.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
