"""
LLM interaction engine.
Uses Groq's free API (llama3-8b-8192) for fast, free inference.
Falls back to a rule-based engine if Groq is unavailable.
"""

import os
import json
import time
import random
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Try importing groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not installed. Using rule-based fallback.")


GROQ_MODEL = "llama3-8b-8192"  # Free, fast model on Groq


def get_groq_client() -> Optional["Groq"]:
    """Get Groq client using API key from environment."""
    if not GROQ_AVAILABLE:
        return None
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set. Using fallback.")
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to create Groq client: {e}")
        return None


def call_llm(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 150,
    temperature: float = 0.7,
) -> str:
    """
    Call the LLM. Uses Groq if available, otherwise rule-based fallback.
    """
    client = get_groq_client()

    if client:
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return rule_based_response(system_prompt, user_message)
    else:
        return rule_based_response(system_prompt, user_message)


def rule_based_response(system_prompt: str, user_message: str) -> str:
    """
    Fallback rule-based response generator when LLM is unavailable.
    Generates plausible responses based on persona cues in system prompt.
    """
    is_flat = "firmly believe the Earth is flat" in system_prompt
    is_neutral = "genuinely uncertain" in system_prompt
    is_round = "confident the Earth is a sphere" in system_prompt

    flat_responses = [
        "I've seen enough videos and done my own research. The curvature just doesn't add up if you look at the horizon yourself.",
        "Why would you trust NASA? They've been lying to us for decades. I trust what I can see with my own eyes.",
        "Interesting what you're saying, but I've been in this community long enough to know that official science isn't always telling the truth.",
        "Look, when you stand by the ocean and look out, does it look curved to you? Exactly. My own eyes tell me the truth.",
        "I used to believe what they taught in school too. But once you start asking questions, you can't stop. The flat earth model makes more sense.",
    ]

    neutral_responses = [
        "Honestly I'm not sure what to believe. There are some compelling arguments on both sides.",
        "I haven't really looked into this deeply. I suppose the mainstream view makes sense but I'm open to hearing more.",
        "That's an interesting perspective. I'd want to see more evidence before making up my mind.",
        "I can see where you're coming from. I'm still doing my own research on this topic.",
        "Both sides seem to have some valid points. I think I need to read more before I decide.",
    ]

    round_responses = [
        "The evidence is pretty clear — from satellite images to the way ships disappear over the horizon. It's a sphere.",
        "I understand the appeal of questioning things, but this one has been settled for thousands of years. Ancient Greeks even calculated Earth's circumference.",
        "If you think about how eclipses work, or how time zones exist, the spherical model is the only one that makes consistent sense.",
        "I respect that you've done your own research, but peer-reviewed science and direct observation from space are pretty hard to argue with.",
        "Even without NASA, every space agency in the world — including India's ISRO — confirms the Earth is spherical. That's not a conspiracy.",
    ]

    if is_flat:
        return random.choice(flat_responses)
    elif is_neutral:
        return random.choice(neutral_responses)
    elif is_round:
        return random.choice(round_responses)
    else:
        return "That's an interesting point. I'll have to think about that more."


def simulate_conversation(
    agent_a_persona: str,
    agent_b_persona: str,
    agent_a_name: str,
    agent_b_name: str,
    topic: str = "flat earth theory",
    turns: int = 3,
) -> list:
    """
    Simulate a multi-turn conversation between two agents.
    Returns list of (speaker_name, message) tuples.
    """
    conversation = []

    # Agent A opens
    opening_prompt = (
        f"You are talking to {agent_b_name} on social media. "
        f"Start a brief, natural conversation about {topic}. "
        f"1-3 sentences only."
    )
    msg_a = call_llm(agent_a_persona, opening_prompt, max_tokens=100)
    conversation.append((agent_a_name, msg_a))
    time.sleep(0.3)  # Rate limit courtesy pause

    # Alternate turns
    history = f"{agent_a_name}: {msg_a}"

    for turn in range(turns - 1):
        # Agent B responds
        b_prompt = (
            f"You are responding to {agent_a_name} on social media. "
            f"Here is what they said:\n{history}\n\n"
            f"Respond naturally and in character. 1-3 sentences only."
        )
        msg_b = call_llm(agent_b_persona, b_prompt, max_tokens=100)
        conversation.append((agent_b_name, msg_b))
        history += f"\n{agent_b_name}: {msg_b}"
        time.sleep(0.3)

        if turn < turns - 2:
            # Agent A responds back
            a_prompt = (
                f"Continue the conversation with {agent_b_name}. "
                f"Here is the conversation so far:\n{history}\n\n"
                f"Respond naturally and in character. 1-3 sentences only."
            )
            msg_a = call_llm(agent_a_persona, a_prompt, max_tokens=100)
            conversation.append((agent_a_name, msg_a))
            history += f"\n{agent_a_name}: {msg_a}"
            time.sleep(0.3)

    return conversation


def estimate_stance_shift(
    agent_persona: str,
    conversation_history: str,
    current_score: float,
    nudge_type: str = "none",
) -> float:
    """
    Estimate how much an agent's stance shifts after an interaction.
    Uses LLM or heuristics to determine delta.
    """
    prompt = (
        f"Given this social media conversation:\n{conversation_history}\n\n"
        f"On a scale from -1 (very resistant) to +1 (very persuaded toward scientific consensus), "
        f"how much would your views shift after this conversation? "
        f"Reply with ONLY a decimal number between -0.3 and 0.3, nothing else."
    )

    raw = call_llm(agent_persona, prompt, max_tokens=10, temperature=0.3)

    # Parse the number
    try:
        delta = float(raw.strip())
        delta = max(-0.3, min(0.3, delta))
    except ValueError:
        # Heuristic fallback
        if nudge_type == "active":
            delta = random.uniform(0.02, 0.12)
        elif nudge_type == "passive":
            delta = random.uniform(0.01, 0.08)
        else:
            delta = random.uniform(-0.05, 0.05)

    return delta
