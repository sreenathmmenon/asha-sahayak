---
title: ASHA Sahayak - Clinical Decision Support RL Environment
emoji: 🏥
colorFrom: green
colorTo: red
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - healthcare
  - india
  - asha-workers
  - imnci
  - maternal-health
---

# ASHA Sahayak — AI Clinical Decision Support for Frontline Health Workers

> *"There are 1.07 million ASHA workers in India. They are the healthcare infrastructure for 600 million people. They currently make life-or-death decisions with a printed booklet and no connectivity support."*

**ASHA Sahayak** is a multi-turn reinforcement learning environment that trains AI agents to assist ASHA (Accredited Social Health Activist) workers in rural India make correct clinical referral decisions — backed by the official Indian Government IMNCI protocol.

---

## How It Works

An ASHA worker presents a patient case in natural language (bilingual Hindi/English). The AI agent must:

1. Ask targeted clarifying questions to reveal symptoms incrementally
2. Identify key danger signs from the IMNCI protocol
3. Give a structured referral decision with urgency and action items

The grader is **100% deterministic** — no LLM judge, no randomness. Same inputs always produce the same score.

---

## Demo

```
ASHA Worker: "Didi, ek aurat hai 8 mahine pregnant. Bahut sar dard hai subah se."
             [Sister, a woman is 8 months pregnant. Very bad headache since morning.]

Agent asks: "Does she have any blurred vision or seeing spots?"

ASHA Worker: "Yes — vision is blurry. Hands and feet also swollen since yesterday."

Agent decides:
{
  "referral_decision": "REFER_IMMEDIATELY",
  "urgency": "immediate",
  "primary_concern": "pre_eclampsia_severe_features",
  "action_items": ["transport_to_fru_within_hours", "alert_phc_maternity_ward"]
}

Score: 0.92 / 1.00 ✅
```

---

## Setup

### Requirements

- Python 3.10+
- Docker (for containerized deployment)
- A HuggingFace account and API token

### Step 1 — Clone the repository

```bash
git clone https://github.com/sreenathmmenon/asha-sahayak.git
cd asha-sahayak
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -e asha_sahayak/
```

### Step 4 — Start the environment server

```bash
uvicorn asha_sahayak.server.app:app --host 0.0.0.0 --port 7860
```

The server will be running at `http://localhost:7860`. Visit `http://localhost:7860/ui` for the interactive Gradio interface.

### Step 5 — Set environment variables

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_huggingface_token_here"
export ENV_BASE_URL="http://localhost:7860"   # or your deployed Space URL
```

### Step 6 — Run inference

```bash
python asha_sahayak/inference.py
```

Expected output:
```
[START] task=easy env=asha_sahayak model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=Does the child have any chest indrawing... reward=0.12 done=false error=null
[STEP]  step=2 action=REFER_IMMEDIATELY|severe_pneumonia reward=1.00 done=true error=null
[END] success=true steps=2 score=1.000 rewards=0.12,1.00
...
[SUMMARY] tasks=3 successes=3 overall_score=0.895
```

---

## Docker

### Build and run locally

```bash
cd asha_sahayak
docker build -f Dockerfile -t asha-sahayak .
docker run -p 7860:7860 asha-sahayak
```

### Using the deployed Space

The environment is live at:
```
https://sreenathmmenon-asha-sahayak.hf.space
```

Set `ENV_BASE_URL` to the Space URL and run inference against it directly:
```bash
export ENV_BASE_URL="https://sreenathmmenon-asha-sahayak.hf.space"
python asha_sahayak/inference.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns `{"status": "healthy"}` |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Submit an action |
| `/state` | GET | Get current episode state |
| `/metadata` | GET | Environment metadata |
| `/schema` | GET | Action and observation schema |
| `/ui` | GET | Interactive Gradio interface |

### Example API calls

```bash
# Health check
curl http://localhost:7860/health

# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Ask a clarifying question
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "referral_decision": "PENDING",
    "urgency": "unknown",
    "primary_concern": "gathering_information",
    "question": "Does the child have chest indrawing when breathing?"
  }'

# Give final decision
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "referral_decision": "REFER_IMMEDIATELY",
    "urgency": "immediate",
    "primary_concern": "severe_pneumonia",
    "action_items": ["first_dose_antibiotic_before_transfer", "transport_to_hospital"]
  }'
```

---

## Tasks

### Easy — Clear Danger Signs
5 cases: severe pneumonia, eclampsia, mild diarrhea, neonatal sepsis, pneumonia (home treatment)
- Single dominant danger sign per case
- Max 4 turns

### Medium — Multi-symptom Cases
5 cases: severe pre-eclampsia, severe dehydration, TB suspect, some dehydration, postpartum haemorrhage
- Requires clarifying questions to distinguish severity
- Max 5 turns

### Hard — Complex Cross-cutting Signs
6 cases: neonatal hypothermia with sepsis, severe complicated SAM, antepartum haemorrhage in shock, meningitis signs, omphalitis with systemic spread, moderate malnutrition
- Multiple overlapping danger signs
- Includes cases where correct answer is **not** to refer (community-managed MAM)
- Max 6 turns

---

## Reward Function

```
R = 0.40 × R_referral
  + 0.25 × R_urgency
  + 0.20 × R_primary_concern
  + 0.15 × R_information_gathering
```

| Component | Description |
|-----------|-------------|
| R_referral | Exact match = 1.0. Per-case near-miss partial credit. Dangerous under-triage capped at 0.1 |
| R_urgency | Ordered distance scoring (off-by-one = 0.5, off-by-two = 0.2) |
| R_primary_concern | Keyword overlap between predicted and correct clinical concern |
| R_information_gathering | Agent must ask ≥1 clarifying question. No question = 0.2 cap |

Dense intermediate rewards are given when the agent asks about key danger signs (0.10 per new sign identified).

---

## Action Space

```python
{
  "referral_decision": "REFER_IMMEDIATELY",  # REFER_IMMEDIATELY | REFER_WITHIN_24H | TREAT_AT_HOME | MONITOR | PENDING
  "urgency": "immediate",                    # immediate | within_24h | routine | monitor | unknown
  "primary_concern": "severe_pneumonia",     # snake_case clinical identifier
  "action_items": ["transport_to_hospital"], # structured tags — no drug names or dosages
  "question": null,                          # clarifying question (set when referral_decision=PENDING)
  "confidence": 0.95                         # 0.0–1.0
}
```

## Observation Space

```python
{
  "conversation": [...],        # growing list of {role, text} turns
  "patient_context": {
    "age_description": "...",
    "gender": "...",
    "location": "...",
    "malaria_risk_area": false,
    "season": "..."
  },
  "task_id": "easy",
  "turn_number": 2,
  "max_turns": 4,
  "done": false,
  "reward": 0.12,               # intermediate reward per step; final score at done=True
  "feedback": null              # clinical explanation shown at done=True
}
```

---

## Ground Truth Sources

All cases derived from official published government protocols:

- **WHO/NHM IMNCI Chart Booklet** — nhm.gov.in
- **MOHFW India Maternal Health Guidelines** — JSSK scheme referral criteria
- **NTEP TB Referral Guidelines** — ntep.in
- **NHM Postpartum Haemorrhage Guidelines** — nhm.gov.in

Ground truth is not invented — it is the exact protocol ASHA workers are trained on.

---

## Infrastructure

- **Runtime:** Pure Python, no ML models loaded server-side
- **Memory:** < 200 MB
- **Latency:** < 50ms per step (pure dict lookups, no external calls)
- **Inference time:** ~3–4 minutes for all 3 tasks
- **Hardware:** Runs on 2 vCPU / 8 GB RAM

---

## Impact

- **1.07 million ASHA workers** serve as healthcare infrastructure for 600 million rural Indians
- Only **7.2% of ASHAs** know all key danger signs during pregnancy
- India's maternal mortality rate: **97/100,000** — most preventable with correct triage
- This environment trains agents on the same protocol ASHA workers are officially trained on

*Graders backed by official Government of India IMNCI protocol.*
