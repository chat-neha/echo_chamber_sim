"""
Agent definitions for the Echo Chamber Simulation.
Each agent is a social media user with a persona, stance, and epistemological base.
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class Stance(Enum):
    FLAT_EARTHER = "flat_earther"
    NEUTRAL = "neutral"
    ROUND_EARTHER = "round_earther"


class EpistemologicalBase(Enum):
    PRATYAKSA = "Pratyakṣa"        # Direct perception / sensory
    ANUMANA = "Anumāna"            # Inference / reasoning
    SABDA = "Śabda"                # Testimony / expert knowledge
    UPAMANA = "Upamāna"            # Analogy / comparison
    ARTHAPATTI = "Arthāpatti"      # Postulation / coherent explanation
    ANUPALABDHI = "Anupalabdhi"    # Non-perception / absence


PRAMANA_DESCRIPTIONS = {
    EpistemologicalBase.PRATYAKSA: "You rely heavily on what you can directly observe and experience with your senses.",
    EpistemologicalBase.ANUMANA: "You reason from observable signs and evidence to reach logical conclusions.",
    EpistemologicalBase.SABDA: "You trust expert testimony, scientific consensus, and authoritative sources.",
    EpistemologicalBase.UPAMANA: "You understand things through analogy and comparison with known phenomena.",
    EpistemologicalBase.ARTHAPATTI: "You seek the most coherent explanation that accounts for all observed facts.",
    EpistemologicalBase.ANUPALABDHI: "You reason from what is NOT observed or absent to draw conclusions.",
}


@dataclass
class Agent:
    agent_id: int
    name: str
    age: int
    gender: str
    occupation: str
    education: str
    social_media_usage: str  # "heavy", "moderate", "light"
    stance: Stance
    stance_score: float       # -1.0 (flat earth) to +1.0 (round earth)
    epistemological_base: EpistemologicalBase
    interests: List[str]
    personality_traits: List[str]
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(16))
    interaction_history: List[dict] = field(default_factory=list)
    nudge_received: Optional[str] = None  # "active" or "passive"

    def get_persona_prompt(self) -> str:
        pramana_desc = PRAMANA_DESCRIPTIONS[self.epistemological_base]
        stance_desc = {
            Stance.FLAT_EARTHER: "You firmly believe the Earth is flat and distrust mainstream science on this topic.",
            Stance.NEUTRAL: "You are genuinely uncertain about the Earth's shape and are open to arguments from both sides.",
            Stance.ROUND_EARTHER: "You are confident the Earth is a sphere based on scientific evidence.",
        }[self.stance]

        return f"""You are {self.name}, a {self.age}-year-old {self.gender} working as a {self.occupation}.
Education: {self.education}
Social media usage: {self.social_media_usage}
Interests: {', '.join(self.interests)}
Personality: {', '.join(self.personality_traits)}

Epistemological style: {self.epistemological_base.value} — {pramana_desc}

Your view on Earth's shape: {stance_desc}
Your internal conviction score: {self.stance_score:.2f} (scale: -1.0 = flat earth believer, +1.0 = round earth believer)

You are a real person on social media. Respond naturally, authentically, and in character.
Keep responses concise (2-4 sentences). Do NOT break character."""

    def get_stance_label(self) -> str:
        if self.stance_score <= -0.33:
            return "Flat Earther"
        elif self.stance_score >= 0.33:
            return "Round Earther"
        else:
            return "Neutral"

    def update_stance(self, delta: float):
        """Update stance score based on interaction delta."""
        self.stance_score = max(-1.0, min(1.0, self.stance_score + delta))
        if self.stance_score <= -0.33:
            self.stance = Stance.FLAT_EARTHER
        elif self.stance_score >= 0.33:
            self.stance = Stance.ROUND_EARTHER
        else:
            self.stance = Stance.NEUTRAL


def generate_embedding(agent: Agent) -> np.ndarray:
    """Generate a pseudo-embedding for an agent based on their attributes."""
    rng = np.random.default_rng(seed=agent.agent_id * 42)
    base = rng.standard_normal(16)

    # Encode stance into embedding dimensions 0-2
    if agent.stance == Stance.FLAT_EARTHER:
        base[0] = -1.5 + rng.uniform(-0.3, 0.3)
    elif agent.stance == Stance.ROUND_EARTHER:
        base[0] = 1.5 + rng.uniform(-0.3, 0.3)
    else:
        base[0] = 0.0 + rng.uniform(-0.3, 0.3)

    # Encode epistemological base into dimensions 3-6
    ep_map = {e: i for i, e in enumerate(EpistemologicalBase)}
    base[3] = ep_map[agent.epistemological_base] * 0.5

    # Encode age
    base[4] = (agent.age - 35) / 20.0

    # Social media usage
    usage_map = {"heavy": 1.0, "moderate": 0.0, "light": -1.0}
    base[5] = usage_map.get(agent.social_media_usage, 0.0)

    norm = np.linalg.norm(base)
    return base / norm if norm > 0 else base


def create_agent_pool() -> List[Agent]:
    """Create a diverse pool of 50 agents."""

    agent_specs = [
        # --- FLAT EARTHERS (15 agents) ---
        (1, "Rajesh Verma", 42, "Male", "Auto mechanic", "High school", "heavy",
         Stance.FLAT_EARTHER, -0.85, EpistemologicalBase.PRATYAKSA,
         ["DIY electronics", "conspiracy theories", "YouTube documentaries"],
         ["skeptical", "stubborn", "passionate"]),

        (2, "Meena Pillai", 34, "Female", "Homemaker", "Graduate", "moderate",
         Stance.FLAT_EARTHER, -0.72, EpistemologicalBase.ANUPALABDHI,
         ["astrology", "alternative health", "cooking"],
         ["curious", "trusting of peers", "emotional"]),

        (3, "Arjun Sharma", 27, "Male", "Social media influencer", "Some college", "heavy",
         Stance.FLAT_EARTHER, -0.90, EpistemologicalBase.PRATYAKSA,
         ["YouTube", "fitness", "street photography"],
         ["charismatic", "contrarian", "impulsive"]),

        (4, "Sunita Devi", 55, "Female", "Retired teacher", "Graduate", "light",
         Stance.FLAT_EARTHER, -0.65, EpistemologicalBase.SABDA,
         ["gardening", "religion", "local politics"],
         ["traditional", "community-oriented", "trusting"]),

        (5, "Ravi Kumar", 31, "Male", "Truck driver", "High school", "moderate",
         Stance.FLAT_EARTHER, -0.78, EpistemologicalBase.PRATYAKSA,
         ["cricket", "WhatsApp groups", "travel"],
         ["practical", "blunt", "loyal"]),

        (6, "Priya Nair", 23, "Female", "College student", "Undergraduate", "heavy",
         Stance.FLAT_EARTHER, -0.60, EpistemologicalBase.ANUPALABDHI,
         ["TikTok", "astrology", "fashion"],
         ["impressionable", "social", "creative"]),

        (7, "Dinesh Rao", 47, "Male", "Small business owner", "Graduate", "moderate",
         Stance.FLAT_EARTHER, -0.80, EpistemologicalBase.PRATYAKSA,
         ["business news", "WhatsApp", "regional politics"],
         ["assertive", "skeptical of elites", "entrepreneurial"]),

        (8, "Lalita Singh", 38, "Female", "Nurse", "Graduate", "moderate",
         Stance.FLAT_EARTHER, -0.55, EpistemologicalBase.SABDA,
         ["health and wellness", "spirituality", "family"],
         ["caring", "community-driven", "open-minded"]),

        (9, "Suresh Patil", 52, "Male", "Farmer", "Primary school", "light",
         Stance.FLAT_EARTHER, -0.88, EpistemologicalBase.PRATYAKSA,
         ["agriculture", "weather", "local news"],
         ["grounded", "tradition-bound", "observant"]),

        (10, "Kavitha Menon", 29, "Female", "Graphic designer", "Graduate", "heavy",
         Stance.FLAT_EARTHER, -0.68, EpistemologicalBase.UPAMANA,
         ["art", "alternative history", "social media"],
         ["creative", "questioning", "visual thinker"]),

        (11, "Mohit Gupta", 35, "Male", "Electrician", "Diploma", "moderate",
         Stance.FLAT_EARTHER, -0.75, EpistemologicalBase.ANUMANA,
         ["electronics", "conspiracy forums", "football"],
         ["analytical in small ways", "distrustful of media", "hands-on"]),

        (12, "Rekha Joshi", 44, "Female", "Shop owner", "High school", "light",
         Stance.FLAT_EARTHER, -0.62, EpistemologicalBase.ANUPALABDHI,
         ["local gossip", "devotional music", "cooking"],
         ["opinionated", "community anchor", "warm"]),

        (13, "Ajay Tiwari", 26, "Male", "Call center agent", "Graduate", "heavy",
         Stance.FLAT_EARTHER, -0.70, EpistemologicalBase.PRATYAKSA,
         ["gaming", "Reddit", "music"],
         ["online-native", "argumentative", "bored at work"]),

        (14, "Shalini Bhat", 33, "Female", "Yoga instructor", "Graduate", "moderate",
         Stance.FLAT_EARTHER, -0.58, EpistemologicalBase.PRATYAKSA,
         ["wellness", "spirituality", "nature"],
         ["holistic thinker", "non-confrontational", "intuitive"]),

        (15, "Prakash Reddy", 49, "Male", "Security guard", "High school", "light",
         Stance.FLAT_EARTHER, -0.82, EpistemologicalBase.SABDA,
         ["local religion", "cricket", "news channels"],
         ["deferential to authority he trusts", "patriotic", "stoic"]),

        # --- NEUTRAL AGENTS (20 agents) ---
        (16, "Ananya Krishnan", 24, "Female", "Marketing intern", "Undergraduate", "heavy",
         Stance.NEUTRAL, 0.05, EpistemologicalBase.SABDA,
         ["pop culture", "social media trends", "fitness"],
         ["pragmatic", "fence-sitter", "social"]),

        (17, "Vikram Shetty", 36, "Male", "IT professional", "Postgraduate", "moderate",
         Stance.NEUTRAL, -0.10, EpistemologicalBase.ANUMANA,
         ["tech news", "coding", "gaming"],
         ["logical", "apathetic to controversy", "busy"]),

        (18, "Deepa Iyer", 40, "Female", "Accountant", "Graduate", "light",
         Stance.NEUTRAL, 0.15, EpistemologicalBase.ARTHAPATTI,
         ["finance", "cooking", "family"],
         ["cautious", "fact-seeking", "risk-averse"]),

        (19, "Rohit Malhotra", 22, "Male", "Engineering student", "Undergraduate", "heavy",
         Stance.NEUTRAL, -0.05, EpistemologicalBase.ANUMANA,
         ["science", "memes", "cricket"],
         ["curious", "undecided", "easily distracted"]),

        (20, "Nandini Das", 31, "Female", "Journalist", "Postgraduate", "heavy",
         Stance.NEUTRAL, 0.20, EpistemologicalBase.SABDA,
         ["news", "politics", "literature"],
         ["balanced", "inquisitive", "professional"]),

        (21, "Arun Nambiar", 45, "Male", "Bank manager", "Postgraduate", "light",
         Stance.NEUTRAL, 0.10, EpistemologicalBase.ARTHAPATTI,
         ["economics", "cricket", "travel"],
         ["measured", "conservative", "deliberate"]),

        (22, "Pooja Shah", 28, "Female", "Pharmacist", "Graduate", "moderate",
         Stance.NEUTRAL, -0.08, EpistemologicalBase.SABDA,
         ["healthcare", "Bollywood", "reading"],
         ["detail-oriented", "skeptical", "empathetic"]),

        (23, "Girish Kulkarni", 53, "Male", "School principal", "Postgraduate", "light",
         Stance.NEUTRAL, 0.12, EpistemologicalBase.UPAMANA,
         ["education", "philosophy", "chess"],
         ["thoughtful", "mentor-like", "balanced"]),

        (24, "Swati Pandey", 25, "Female", "Fashion blogger", "Graduate", "heavy",
         Stance.NEUTRAL, -0.15, EpistemologicalBase.PRATYAKSA,
         ["fashion", "travel", "Instagram"],
         ["trend-driven", "visual", "social"]),

        (25, "Harish Nair", 39, "Male", "Civil engineer", "Graduate", "moderate",
         Stance.NEUTRAL, 0.18, EpistemologicalBase.ANUMANA,
         ["infrastructure", "football", "music"],
         ["methodical", "open-minded", "team player"]),

        (26, "Radha Murthy", 48, "Female", "NGO worker", "Postgraduate", "moderate",
         Stance.NEUTRAL, 0.05, EpistemologicalBase.ARTHAPATTI,
         ["social issues", "environment", "yoga"],
         ["compassionate", "pragmatic", "idealistic"]),

        (27, "Sanjay Mehta", 30, "Male", "Startup founder", "Postgraduate", "heavy",
         Stance.NEUTRAL, -0.12, EpistemologicalBase.ANUMANA,
         ["entrepreneurship", "tech", "podcasts"],
         ["ambitious", "open to data", "busy"]),

        (28, "Usha Rani", 57, "Female", "Housewife", "High school", "light",
         Stance.NEUTRAL, 0.08, EpistemologicalBase.UPAMANA,
         ["devotion", "cooking", "grandchildren"],
         ["gentle", "community-focused", "storyteller"]),

        (29, "Nikhil Jain", 21, "Male", "Commerce student", "Undergraduate", "heavy",
         Stance.NEUTRAL, -0.20, EpistemologicalBase.ANUPALABDHI,
         ["memes", "cryptocurrency", "gaming"],
         ["irreverent", "skeptical", "digitally native"]),

        (30, "Amrita Kaur", 35, "Female", "Teacher", "Postgraduate", "moderate",
         Stance.NEUTRAL, 0.22, EpistemologicalBase.SABDA,
         ["education", "books", "children's welfare"],
         ["nurturing", "evidence-leaning", "patient"]),

        (31, "Sunil Bose", 43, "Male", "Photographer", "Graduate", "moderate",
         Stance.NEUTRAL, -0.05, EpistemologicalBase.PRATYAKSA,
         ["travel", "visual arts", "nature"],
         ["observational", "introspective", "artistic"]),

        (32, "Geeta Pillai", 50, "Female", "Doctor", "Postgraduate", "light",
         Stance.NEUTRAL, 0.28, EpistemologicalBase.ARTHAPATTI,
         ["medicine", "evidence-based practice", "yoga"],
         ["rational", "evidence-driven", "calm"]),

        (33, "Varun Chopra", 27, "Male", "Digital marketer", "Graduate", "heavy",
         Stance.NEUTRAL, -0.18, EpistemologicalBase.ANUPALABDHI,
         ["SEO", "social media", "coffee"],
         ["analytical", "data-driven", "trend-conscious"]),

        (34, "Mamta Tiwari", 46, "Female", "Textile worker", "Primary school", "light",
         Stance.NEUTRAL, 0.02, EpistemologicalBase.PRATYAKSA,
         ["local fairs", "family", "Bollywood"],
         ["practical", "community-bound", "hardworking"]),

        (35, "Kiran Desai", 32, "Female", "Research analyst", "Postgraduate", "moderate",
         Stance.NEUTRAL, 0.25, EpistemologicalBase.ANUMANA,
         ["data science", "books", "travel"],
         ["methodical", "open-minded", "curious"]),

        # --- ROUND EARTHERS (15 agents) ---
        (36, "Dr. Priya Ramachandran", 41, "Female", "Astrophysicist", "PhD", "moderate",
         Stance.ROUND_EARTHER, 0.95, EpistemologicalBase.ARTHAPATTI,
         ["astronomy", "science communication", "hiking"],
         ["rational", "patient explainer", "passionate about science"]),

        (37, "Siddharth Rao", 28, "Male", "Software engineer", "Postgraduate", "heavy",
         Stance.ROUND_EARTHER, 0.88, EpistemologicalBase.ANUMANA,
         ["coding", "astronomy", "sci-fi"],
         ["logical", "data-driven", "skeptical of pseudoscience"]),

        (38, "Lakshmi Venkatesh", 37, "Female", "Science teacher", "Postgraduate", "moderate",
         Stance.ROUND_EARTHER, 0.80, EpistemologicalBase.SABDA,
         ["education", "popular science", "classical music"],
         ["methodical", "passionate", "articulate"]),

        (39, "Aditya Kumar", 24, "Male", "Physics student", "Undergraduate", "heavy",
         Stance.ROUND_EARTHER, 0.92, EpistemologicalBase.ANUMANA,
         ["physics", "Reddit", "debate"],
         ["confident", "fact-oriented", "sometimes impatient"]),

        (40, "Meghna Iyer", 33, "Female", "Science journalist", "Postgraduate", "heavy",
         Stance.ROUND_EARTHER, 0.85, EpistemologicalBase.SABDA,
         ["science communication", "writing", "travel"],
         ["clear communicator", "evidence-focused", "engaging"]),

        (41, "Rajiv Pillai", 50, "Male", "Retired scientist", "PhD", "light",
         Stance.ROUND_EARTHER, 0.90, EpistemologicalBase.ARTHAPATTI,
         ["research", "reading", "mentoring"],
         ["measured", "deeply knowledgeable", "patient"]),

        (42, "Chitra Nair", 29, "Female", "Aerospace engineer", "Postgraduate", "moderate",
         Stance.ROUND_EARTHER, 0.93, EpistemologicalBase.ANUMANA,
         ["aviation", "space exploration", "running"],
         ["precise", "technically minded", "enthusiastic"]),

        (43, "Nitin Sharma", 26, "Male", "Medical student", "Undergraduate", "heavy",
         Stance.ROUND_EARTHER, 0.75, EpistemologicalBase.SABDA,
         ["medicine", "science", "cricket"],
         ["studious", "trusting of evidence", "empathetic"]),

        (44, "Vandana Krishnan", 44, "Female", "College professor", "PhD", "light",
         Stance.ROUND_EARTHER, 0.87, EpistemologicalBase.UPAMANA,
         ["philosophy of science", "teaching", "literature"],
         ["thoughtful", "nuanced", "Socratic"]),

        (45, "Abhishek Das", 31, "Male", "Data scientist", "Postgraduate", "moderate",
         Stance.ROUND_EARTHER, 0.82, EpistemologicalBase.ANUMANA,
         ["machine learning", "statistics", "gaming"],
         ["analytical", "skeptical", "evidence-driven"]),

        (46, "Tara Menon", 38, "Female", "Environmental scientist", "Postgraduate", "moderate",
         Stance.ROUND_EARTHER, 0.78, EpistemologicalBase.ARTHAPATTI,
         ["climate science", "nature", "sustainability"],
         ["holistic thinker", "passionate", "collaborative"]),

        (47, "Mohan Lal", 55, "Male", "Retired navy officer", "Graduate", "light",
         Stance.ROUND_EARTHER, 0.91, EpistemologicalBase.PRATYAKSA,
         ["navigation", "history", "cricket"],
         ["practical", "experienced", "authoritative"]),

        (48, "Shruti Gupta", 22, "Female", "Science vlogger", "Undergraduate", "heavy",
         Stance.ROUND_EARTHER, 0.83, EpistemologicalBase.SABDA,
         ["YouTube science", "space", "dancing"],
         ["energetic", "communicator", "gen-Z"]),

        (49, "Prasad Rao", 47, "Male", "Civil servant", "Postgraduate", "light",
         Stance.ROUND_EARTHER, 0.70, EpistemologicalBase.ARTHAPATTI,
         ["policy", "current affairs", "badminton"],
         ["bureaucratic", "methodical", "fair-minded"]),

        (50, "Anita Bose", 36, "Female", "Librarian", "Postgraduate", "moderate",
         Stance.ROUND_EARTHER, 0.77, EpistemologicalBase.SABDA,
         ["books", "information literacy", "classical dance"],
         ["informed", "quiet", "deeply curious"]),
    ]

    agents = []
    for spec in agent_specs:
        (aid, name, age, gender, occ, edu, usage,
         stance, score, ep_base, interests, traits) = spec
        agent = Agent(
            agent_id=aid,
            name=name,
            age=age,
            gender=gender,
            occupation=occ,
            education=edu,
            social_media_usage=usage,
            stance=stance,
            stance_score=score,
            epistemological_base=ep_base,
            interests=interests,
            personality_traits=traits,
        )
        agent.embedding = generate_embedding(agent)
        agents.append(agent)

    return agents
