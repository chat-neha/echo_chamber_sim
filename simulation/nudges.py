"""
Nudge strategies for the Echo Chamber Simulation.
Implements Active Nudge (compatibility-based pairing) and Passive Nudge (pramāṇa-based reflection).
"""

import copy
import logging
import numpy as np
from typing import List, Tuple, Dict
from simulation.agents import Agent, Stance, EpistemologicalBase
from simulation.network import cosine_similarity, build_initial_network
from simulation.llm_engine import (
    simulate_conversation,
    estimate_stance_shift,
    call_llm,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# ACTIVE NUDGE
# ──────────────────────────────────────────────────────────────────────────────

def compute_compatibility_score(agent_a: Agent, agent_b: Agent) -> float:
    """
    Compatibility score for Active Nudge pairing.
    High embedding similarity + sufficient stance difference.
    """
    sim = cosine_similarity(agent_a.embedding, agent_b.embedding)
    stance_diff = abs(agent_a.stance_score - agent_b.stance_score)
    # We want: high similarity AND meaningful difference
    score = sim * stance_diff
    return float(score)


def find_active_nudge_pairs(agents: List[Agent], num_pairs: int = 15) -> List[Tuple[Agent, Agent]]:
    """
    Find optimal pairing of agents for active nudge interactions.
    Pairs agents with similar embeddings but different stances.
    """
    pairs = []
    scored = []

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = agents[i], agents[j]
            # Only pair if stances are different
            if a.stance == b.stance:
                continue
            score = compute_compatibility_score(a, b)
            scored.append((score, a, b))

    scored.sort(key=lambda x: x[0], reverse=True)

    used = set()
    for score, a, b in scored:
        if a.agent_id in used or b.agent_id in used:
            continue
        pairs.append((a, b))
        used.add(a.agent_id)
        used.add(b.agent_id)
        if len(pairs) >= num_pairs:
            break

    return pairs


def run_active_nudge(
    agents: List[Agent],
    num_pairs: int = 15,
    interaction_turns: int = 3,
) -> Tuple[List[Agent], List[dict], "nx.DiGraph"]:
    """
    Run the Active Nudge scenario.
    Returns: (updated agents, interaction logs, updated graph)
    """
    import networkx as nx

    # Deep copy agents to avoid mutating original
    sim_agents = copy.deepcopy(agents)
    agent_map = {a.agent_id: a for a in sim_agents}

    # Find pairs
    pairs = find_active_nudge_pairs(sim_agents, num_pairs=num_pairs)
    logs = []

    for pair_idx, (agent_a, agent_b) in enumerate(pairs):
        logger.info(f"[Active Nudge] Pairing: {agent_a.name} ↔ {agent_b.name}")

        # Simulate conversation
        try:
            conversation = simulate_conversation(
                agent_a.get_persona_prompt(),
                agent_b.get_persona_prompt(),
                agent_a.name,
                agent_b.name,
                turns=interaction_turns,
            )
        except Exception as e:
            logger.error(f"Conversation failed: {e}")
            conversation = [
                (agent_a.name, f"[Auto] Hi {agent_b.name}, I saw your posts about the shape of Earth. Interesting perspective!"),
                (agent_b.name, f"[Auto] Thanks {agent_a.name}. I'm always happy to discuss this topic."),
                (agent_a.name, "[Auto] Let me think about what you've shared. It's given me something to consider."),
            ]

        # Build conversation text
        conv_text = "\n".join(f"{spk}: {msg}" for spk, msg in conversation)

        # Estimate stance shifts
        delta_a = estimate_stance_shift(
            agent_a.get_persona_prompt(),
            conv_text,
            agent_a.stance_score,
            nudge_type="active",
        )
        delta_b = estimate_stance_shift(
            agent_b.get_persona_prompt(),
            conv_text,
            agent_b.stance_score,
            nudge_type="active",
        )

        # Flat earthers shift toward centre (positive direction)
        # Round earthers may shift slightly or stay same
        if agent_a.stance == Stance.FLAT_EARTHER:
            agent_a.update_stance(abs(delta_a))
        elif agent_a.stance == Stance.ROUND_EARTHER:
            agent_a.update_stance(-abs(delta_a) * 0.2)  # minimal shift back
        else:
            agent_a.update_stance(delta_a * 0.5)

        if agent_b.stance == Stance.FLAT_EARTHER:
            agent_b.update_stance(abs(delta_b))
        elif agent_b.stance == Stance.ROUND_EARTHER:
            agent_b.update_stance(-abs(delta_b) * 0.2)
        else:
            agent_b.update_stance(delta_b * 0.5)

        agent_a.nudge_received = "active"
        agent_b.nudge_received = "active"

        log_entry = {
            "pair_index": pair_idx + 1,
            "agent_a": {
                "id": agent_a.agent_id,
                "name": agent_a.name,
                "initial_stance": agents[agent_a.agent_id - 1].stance_score,
                "final_stance": agent_a.stance_score,
                "label": agent_a.get_stance_label(),
            },
            "agent_b": {
                "id": agent_b.agent_id,
                "name": agent_b.name,
                "initial_stance": agents[agent_b.agent_id - 1].stance_score,
                "final_stance": agent_b.stance_score,
                "label": agent_b.get_stance_label(),
            },
            "compatibility_score": compute_compatibility_score(
                agents[agent_a.agent_id - 1], agents[agent_b.agent_id - 1]
            ),
            "conversation": conversation,
            "delta_a": delta_a,
            "delta_b": delta_b,
        }
        logs.append(log_entry)

    # Rebuild network with updated agents
    updated_graph = build_initial_network(sim_agents, echo_chamber_strength=0.4)
    return sim_agents, logs, updated_graph


# ──────────────────────────────────────────────────────────────────────────────
# PASSIVE NUDGE
# ──────────────────────────────────────────────────────────────────────────────

PRAMANA_NUDGE_TEMPLATES = {
    EpistemologicalBase.PRATYAKSA: (
        "You came across a post with a short video clip: a ship slowly disappearing "
        "hull-first over the horizon, filmed from a beach in Kerala. The caption reads: "
        "'Watch carefully — if the Earth were flat, the entire ship would just shrink. "
        "But the hull vanishes first. This is what curvature looks like from ground level.'"
    ),
    EpistemologicalBase.ANUMANA: (
        "You read an infographic that shows how during a lunar eclipse, "
        "Earth always casts a circular shadow on the Moon — from every angle, every time. "
        "The logical inference is: only a sphere casts a consistently circular shadow."
    ),
    EpistemologicalBase.SABDA: (
        "You see a post shared by ISRO (India's space agency), featuring an image of Earth "
        "taken from Chandrayaan-3. The post includes a statement from three independent "
        "senior scientists confirming the oblate spheroid shape of Earth."
    ),
    EpistemologicalBase.UPAMANA: (
        "A thoughtful post draws an analogy: 'Have you noticed that the Moon and Sun "
        "are clearly spherical? And that other planets, when observed through a telescope, "
        "are also spherical? By comparison and consistency, what shape would we expect Earth to be?'"
    ),
    EpistemologicalBase.ARTHAPATTI: (
        "You encounter an interactive explainer: 'What single shape of Earth would coherently "
        "explain all of the following at once? (1) Time zones, (2) Eclipses, (3) Different "
        "star constellations from different latitudes, (4) GPS satellite orbits, (5) Circumnavigation. "
        "The only coherent answer is: a spherical Earth.'"
    ),
    EpistemologicalBase.ANUPALABDHI: (
        "You read a careful post: 'Flat-Earth supporters often say \"I don't see curvature\" "
        "as proof. But this is not valid non-observation. At human scale, atmospheric refraction "
        "and the vast size of Earth make curvature imperceptible to the naked eye. "
        "Absence of visible curvature is NOT evidence of flatness.'"
    ),
}


def create_passive_nudge_prompt(agent: Agent) -> str:
    """Create a tailored passive nudge based on the agent's epistemological base."""
    return PRAMANA_NUDGE_TEMPLATES[agent.epistemological_base]


def run_passive_nudge(
    agents: List[Agent],
    target_flat_earthers: bool = True,
    target_neutrals: bool = True,
) -> Tuple[List[Agent], List[dict], "nx.DiGraph"]:
    """
    Run the Passive Nudge scenario.
    Returns: (updated agents, interaction logs, updated graph)
    """
    import networkx as nx

    sim_agents = copy.deepcopy(agents)
    logs = []

    for agent in sim_agents:
        should_nudge = (
            (target_flat_earthers and agent.stance == Stance.FLAT_EARTHER) or
            (target_neutrals and agent.stance == Stance.NEUTRAL)
        )

        if not should_nudge:
            continue

        nudge_content = create_passive_nudge_prompt(agent)

        # Agent reflects on the nudge
        reflection_prompt = (
            f"You just scrolled past this post on your social media feed:\n\n"
            f"\"{nudge_content}\"\n\n"
            f"What is your honest internal reaction to this? Do you find it credible, "
            f"suspicious, or thought-provoking? Respond in character. 2-3 sentences."
        )

        try:
            reflection = call_llm(
                agent.get_persona_prompt(),
                reflection_prompt,
                max_tokens=120,
                temperature=0.7,
            )
        except Exception as e:
            logger.error(f"Reflection failed for {agent.name}: {e}")
            reflection = "[Auto] This post made me stop and think for a moment. I'm not entirely sure what to make of it."

        # Estimate stance shift from passive nudge
        # Passive nudges are subtler — smaller effect
        initial_score = agents[agent.agent_id - 1].stance_score
        delta = estimate_stance_shift(
            agent.get_persona_prompt(),
            f"[Nudge seen]: {nudge_content}\n[Your reaction]: {reflection}",
            initial_score,
            nudge_type="passive",
        )

        # Flat earthers: shift toward center (less extreme)
        if agent.stance == Stance.FLAT_EARTHER:
            shift = abs(delta) * 0.6  # Passive is subtler
            agent.update_stance(shift)
        else:
            shift = abs(delta) * 0.3
            agent.update_stance(shift)

        agent.nudge_received = "passive"

        log_entry = {
            "agent": {
                "id": agent.agent_id,
                "name": agent.name,
                "age": agent.age,
                "occupation": agent.occupation,
                "ep_base": agent.epistemological_base.value,
                "initial_stance": initial_score,
                "final_stance": agent.stance_score,
                "initial_label": agents[agent.agent_id - 1].get_stance_label(),
                "final_label": agent.get_stance_label(),
            },
            "nudge_type": agent.epistemological_base.value,
            "nudge_content": nudge_content,
            "agent_reflection": reflection,
            "stance_shift": agent.stance_score - initial_score,
        }
        logs.append(log_entry)

    # Rebuild network with updated embeddings
    updated_graph = build_initial_network(sim_agents, echo_chamber_strength=0.4)
    return sim_agents, logs, updated_graph
