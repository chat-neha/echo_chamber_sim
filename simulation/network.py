"""
Social network graph construction for the Echo Chamber Simulation.
Builds a directed graph where edges represent information flow between agents.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple
from simulation.agents import Agent, Stance


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_initial_network(agents: List[Agent], echo_chamber_strength: float = 0.75) -> nx.DiGraph:
    """
    Build an initial directed social graph that reflects echo chamber tendencies.
    Agents are more likely to follow others with similar stances and interests.
    echo_chamber_strength: 0.0 = random, 1.0 = pure echo chamber
    """
    G = nx.DiGraph()

    # Add nodes
    for agent in agents:
        G.add_node(
            agent.agent_id,
            name=agent.name,
            age=agent.age,
            gender=agent.gender,
            occupation=agent.occupation,
            stance=agent.stance.value,
            stance_score=agent.stance_score,
            ep_base=agent.epistemological_base.value,
            interests=agent.interests,
            personality=agent.personality_traits,
            label=agent.get_stance_label(),
        )

    # Add edges based on similarity + echo chamber bias
    rng = np.random.default_rng(seed=42)

    for i, agent_a in enumerate(agents):
        # Each agent follows between 3-8 others
        num_follows = rng.integers(3, 9)
        candidates = [a for a in agents if a.agent_id != agent_a.agent_id]

        # Score each candidate
        scores = []
        for agent_b in candidates:
            sim = cosine_similarity(agent_a.embedding, agent_b.embedding)
            stance_diff = abs(agent_a.stance_score - agent_b.stance_score)

            # Echo chamber: prefer similar stance
            stance_similarity = 1.0 - stance_diff / 2.0

            # Combined score with echo chamber bias
            score = (echo_chamber_strength * stance_similarity +
                     (1 - echo_chamber_strength) * sim)

            # Add noise
            score += rng.uniform(-0.1, 0.1)
            scores.append((agent_b, score))

        # Sort and pick top candidates
        scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scores[:min(num_follows, len(scores))]

        for agent_b, _ in top_candidates:
            G.add_edge(
                agent_a.agent_id, agent_b.agent_id,
                weight=cosine_similarity(agent_a.embedding, agent_b.embedding)
            )

    return G


def compute_polarization_index(agents: List[Agent]) -> float:
    """
    Compute a polarization index based on variance in stance scores.
    Higher value = more polarized.
    """
    scores = [a.stance_score for a in agents]
    return float(np.std(scores))


def compute_echo_chamber_score(G: nx.DiGraph, agents: List[Agent]) -> float:
    """
    Estimate how echo-chamber-like the network is.
    Measure average stance similarity among connected agents.
    """
    agent_map = {a.agent_id: a for a in agents}
    similarities = []
    for u, v in G.edges():
        if u in agent_map and v in agent_map:
            diff = abs(agent_map[u].stance_score - agent_map[v].stance_score)
            similarities.append(1.0 - diff / 2.0)

    return float(np.mean(similarities)) if similarities else 0.0


def identify_echo_chambers(G: nx.DiGraph, agents: List[Agent]) -> List[List[int]]:
    """
    Identify clusters (echo chambers) in the graph.
    Returns list of node groups that are densely connected + share similar stances.
    """
    agent_map = {a.agent_id: a for a in agents}
    undirected = G.to_undirected()
    communities = list(nx.community.louvain_communities(undirected, seed=42))

    chambers = []
    for community in communities:
        node_list = list(community)
        if len(node_list) >= 3:
            # Check stance variance within community
            scores = [agent_map[n].stance_score for n in node_list if n in agent_map]
            if scores and np.std(scores) < 0.4:
                chambers.append(node_list)

    return chambers


def compute_network_stats(G: nx.DiGraph, agents: List[Agent]) -> dict:
    """Compute various network statistics."""
    agent_map = {a.agent_id: a for a in agents}

    flat_earthers = sum(1 for a in agents if a.stance == Stance.FLAT_EARTHER)
    neutrals = sum(1 for a in agents if a.stance == Stance.NEUTRAL)
    round_earthers = sum(1 for a in agents if a.stance == Stance.ROUND_EARTHER)

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "flat_earthers": flat_earthers,
        "neutrals": neutrals,
        "round_earthers": round_earthers,
        "polarization_index": compute_polarization_index(agents),
        "echo_chamber_score": compute_echo_chamber_score(G, agents),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "density": nx.density(G),
    }
