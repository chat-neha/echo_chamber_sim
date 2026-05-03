# 🌍 Echo Chamber Depolarization Simulator

A research simulation of **50 LLM-based social media agents** studying depolarization strategies for the **Flat Earth controversy**, grounded in **Indian epistemology (pramāṇa theory)**.

---

## 📁 Project Structure

```
echo_chamber_sim/
├── app.py                          # Main Streamlit page (network visualization)
├── pages/
│   └── 1_Interaction_Logs.py       # Second page (interaction logs)
├── simulation/
│   ├── __init__.py
│   ├── agents.py                   # 50 agent personas with full attributes
│   ├── network.py                  # NetworkX graph builder (directed, echo-chamber)
│   ├── llm_engine.py               # Groq LLM API wrapper + rule-based fallback
│   ├── nudges.py                   # Active Nudge + Passive Nudge strategies
│   ├── visualize.py                # Plotly network & chart builders
│   └── runner.py                   # Full simulation pipeline orchestrator
├── sim_data/                       # Auto-created: stores simulation_state.json
├── .streamlit/
│   └── config.toml                 # Dark theme config
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup on VSCode (Dell G15 5511)

### Prerequisites

Your Dell G15 5511 has an NVIDIA RTX 3060. The simulation uses cloud-based LLM inference
(Groq), so the GPU is not directly used for inference — but you can optionally run a
local model with Ollama if you prefer full offline operation.

### Step 1 — Clone / open the project

Open VSCode, then open the `echo_chamber_sim/` folder:

```
File → Open Folder → select echo_chamber_sim/
```

### Step 2 — Create a virtual environment

Open the VSCode Terminal (`Ctrl + ~`) and run:

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# If execution policy blocks this, run first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **If pip install fails for python-louvain:**
> ```bash
> pip install python-louvain
> # OR if that fails:
> pip install community
> ```

### Step 4 — Get a FREE Groq API Key

1. Go to **https://console.groq.com**
2. Sign up (free, no credit card needed)
3. Go to **API Keys** → **Create API Key**
4. Copy your key

### Step 5 — Set environment variable

**Option A: Create a `.env` file** (recommended)

```bash
# In the project root, create .env:
GROQ_API_KEY=gsk_your_actual_key_here
```

Then install python-dotenv and add this to the top of `app.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```

**Option B: Set directly in terminal** (simpler for testing)

```bash
# Windows PowerShell:
$env:GROQ_API_KEY = "gsk_your_key_here"

# Then run streamlit in the same terminal session
```

### Step 6 — Run the simulation

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 🚀 Using the App

### Page 1 — Network Visualization (`app.py`)

- **Sidebar → ▶ Run New Simulation** to start a fresh simulation
- **Tab 1: Initial Network** — Social graph before any intervention
- **Tab 2: After Active Nudge** — Graph after compatibility-based agent pairing
- **Tab 3: After Passive Nudge** — Graph after pramāṇa-tailored feed nudges
- Bottom section shows comparison charts and polarization gauges

### Page 2 — Interaction Logs (Sidebar → 📜 Interaction Logs)

- **Tab 1: Active Nudge Conversations** — Full conversation transcripts between pairs
- **Tab 2: Passive Nudge Reflections** — Each agent's reaction to their epistemological nudge

---

## 🔬 How the Simulation Works

### Agents
- 50 agents: 15 Flat Earthers, 20 Neutral, 15 Round Earthers
- Each has: name, age, gender, occupation, education, personality traits, interests
- Each has an **epistemological base** from Indian philosophy (pramāṇa)
- **Stance score**: −1.0 (hard flat earther) to +1.0 (confident round earther)

### Initial Network
- Directed social graph (follows/following)
- Echo chamber structure: agents preferentially follow similar-stance agents
- `echo_chamber_strength=0.75` creates realistic polarization

### Active Nudge
1. Compute **compatibility scores** for all cross-stance agent pairs
2. High compatibility = high embedding similarity + high stance difference
3. Top 15 pairs are introduced and engage in 3-turn LLM conversations
4. Stance scores updated based on LLM-estimated persuasion delta

### Passive Nudge
1. Each Flat Earther and Neutral agent receives a **pramāṇa-matched nudge** in their feed
2. Nudge content is tailored to their epistemological style (e.g., inference-based for Anumāna users)
3. Agent reflects on the nudge; stance shifts by a smaller amount (passive effect is subtler)

---

## 💡 LLM Usage Notes

- **Model**: `llama3-8b-8192` via Groq (fast, free tier, low latency)
- **Fallback**: If Groq is unavailable, a rule-based response generator activates
- **Token usage**: ~100 tokens per agent turn; full simulation ≈ 15,000–25,000 tokens
- Groq free tier: **14,400 requests/day** — more than sufficient

---

## 🖥️ GPU Note (Dell G15 5511 — RTX 3060)

The default setup uses **Groq cloud inference** (no local GPU needed).

**Optional: Run locally with Ollama + GPU**

If you want fully offline LLM inference using your RTX 3060:

```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3.1:8b

# In llm_engine.py, replace the call_llm function:
# Use requests to POST to http://localhost:11434/api/generate
```

Edit `simulation/llm_engine.py`, `call_llm()` to use:
```python
import requests
response = requests.post("http://localhost:11434/api/generate", json={
    "model": "llama3.1:8b",
    "prompt": f"System: {system_prompt}\n\nUser: {user_message}",
    "stream": False
})
return response.json()["response"]
```

Your RTX 3060 (6GB VRAM) can run llama3.1:8b with 4-bit quantization smoothly.

---

## 📊 Interpreting Results

| Metric | Description |
|--------|-------------|
| **Polarization Index** | Std deviation of stance scores. Lower = less polarized. |
| **Echo Chamber Score** | Avg stance similarity among connected agents. Lower = more diverse. |
| **Stance Shift** | Change in individual agent's stance score post-nudge. |

**Active Nudge** tends to produce larger individual shifts (direct conversation).
**Passive Nudge** tends to produce broader but smaller shifts (subtler, less resistance).

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: networkx.community` | `pip install networkx --upgrade` |
| `ModuleNotFoundError: groq` | `pip install groq` |
| Streamlit can't find `simulation/` | Make sure you run from the `echo_chamber_sim/` root |
| Graph looks empty | Simulation data may be corrupted; click ▶ Run New Simulation |
| Rate limit from Groq | Wait a few seconds; the rule-based fallback activates automatically |

---

## 📚 References

- Wang et al. — LLM-based social simulation
- Yoga Sūtras 1.7 — *Pratyakṣānumānāgamāḥ pramāṇāni* (Perception, Inference, Testimony as valid knowledge)
- Matilal, B.K. — *Perception: An Essay on Classical Indian Theories of Knowledge*
