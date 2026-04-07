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

## What This Environment Does

**ASHA Sahayak** is a multi-turn RL environment that trains AI agents to assist ASHA (Accredited Social Health Activist) workers in rural India make correct clinical referral decisions.

An ASHA worker presents a patient case in natural language. The AI agent must:
1. Ask targeted clarifying questions to gather symptoms
2. Identify the key danger signs
3. Give a structured referral decision backed by official protocol

The grader is 100% deterministic, backed by the **Indian Government IMNCI (Integrated Management of Neonatal and Childhood Illness) protocol** and NHM maternal health guidelines.

---

## The Demo Case

```
ASHA Worker: "Didi, ek aurat hai 8 mahine pregnant. Bahut sar dard hai subah se."
             [Sister, a woman is 8 months pregnant. Very bad headache since morning.]

Agent: "Does she have any blurred vision or seeing spots?"

ASHA Worker: "Yes! Vision is blurry. Hands and feet also swollen since yesterday."

Agent: {
  "referral_decision": "REFER_IMMEDIATELY",
  "urgency": "immediate",
  "primary_concern": "pre_eclampsia_severe_features",
  "action_items": ["transport_to_fru_within_hours", "alert_phc_maternity_ward"]
}

Score: 0.92/1.00 ✅
```

Without this tool, the ASHA worker's protocol booklet lists headache under "common pregnancy complaints." The correct diagnosis (severe pre-eclampsia) requires integrating three symptoms from three different pages. This agent does it in seconds.

---

## Environment Specification

### Action Space

```python
AshaAction(
    referral_decision: str,   # REFER_IMMEDIATELY | REFER_WITHIN_24H | TREAT_AT_HOME | MONITOR | PENDING
    urgency: str,             # immediate | within_24h | routine | monitor
    primary_concern: str,     # snake_case clinical identifier
    action_items: List[str],  # structured tags — NO free-text drug names
    question: Optional[str],  # clarifying question (when referral_decision=PENDING)
    confidence: float,        # 0.0-1.0
)
```

### Observation Space

```python
AshaObservation(
    conversation: List[ConversationTurn],  # incrementally growing dialogue
    patient_context: PatientContext,       # age, gender, location, season, malaria risk
    task_id: str,                          # easy | medium | hard
    turn_number: int,
    max_turns: int,
    done: bool,
    reward: float,                         # 0.0 during episode; final score at done=True
    feedback: Optional[str],               # clinical explanation shown at episode end
)
```

---

## The 3 Tasks

### Task 1 — Easy: Clear Danger Signs
- 5 cases: severe pneumonia, eclampsia, mild diarrhea, neonatal sepsis, pneumonia (home treatment)
- Single dominant danger sign cluster
- Expected score for strong LLM: 0.80+

### Task 2 — Medium: Multi-symptom Cases
- 5 cases: severe pre-eclampsia, severe dehydration, TB suspect, some dehydration, postpartum haemorrhage  
- Requires clarifying questions to distinguish severity levels
- Expected score for strong LLM: 0.65+

### Task 3 — Hard: Complex Cross-cutting Signs
- 6 cases: neonatal hypothermia with sepsis, severe complicated SAM, antepartum haemorrhage in shock, meningitis signs, omphalitis with systemic spread, moderate malnutrition (community management)
- Multiple overlapping danger signs, ambiguous presentations
- Hard task includes cases where correct answer is **not** to refer (MAM case)
- Expected score for strong LLM: 0.50+

---

## Reward Function

```
R = 0.40 × R_referral
  + 0.25 × R_urgency
  + 0.20 × R_primary_concern
  + 0.15 × R_information_gathering
```

**R_referral:** Exact match = 1.0. Near-miss partial credit (e.g. REFER_WITHIN_24H when REFER_IMMEDIATELY correct = 0.4). Dangerous under-triage heavily penalized (sending REFER_IMMEDIATELY case home = 0.1).

**R_urgency:** Ordered distance scoring. Off-by-one = 0.5, off-by-two = 0.2.

**R_primary_concern:** Keyword overlap between predicted and correct clinical concern.

**R_information_gathering (LeCun requirement):** Agent must ask at least 1 clarifying question. No question = 0.2 cap on this component.

---

## Ground Truth Sources

All cases derived from official published protocols:
- **WHO/NHM IMNCI Chart Booklet** — nhm.gov.in
- **MOHFW India Maternal Health Guidelines** — JSSK scheme referral criteria
- **NTEP TB Referral Guidelines** — ntep.in
- **NHM Postpartum Haemorrhage Guidelines** — nhm.gov.in

Ground truth is **not invented** — it is the same protocol ASHA workers are trained on.

---

## Quick Start

### Run locally

```bash
# Start server
cd asha_sahayak
pip install -e .
uvicorn asha_sahayak.server.app:app --host 0.0.0.0 --port 7860

# Run inference
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"
python inference.py
```

### Docker

```bash
docker build -f server/Dockerfile -t asha-sahayak:latest .
docker run -p 7860:7860 asha-sahayak:latest
```

### Test endpoints

```bash
# Health check
curl http://localhost:7860/health

# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# Ask a question
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"referral_decision": "PENDING", "urgency": "unknown", "primary_concern": "gathering_information", "question": "Does the child have chest indrawing?"}'

# Give final decision
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"referral_decision": "REFER_IMMEDIATELY", "urgency": "immediate", "primary_concern": "severe_pneumonia", "action_items": ["first_dose_antibiotic_before_transfer"]}'
```

---

## Baseline Scores (seed=42/123/777)

| Task | Baseline (random) | Qwen2.5-72B (expected) |
|------|------------------|------------------------|
| Easy | ~0.25 | ~0.80 |
| Medium | ~0.20 | ~0.65 |
| Hard | ~0.15 | ~0.50 |

---

## Infrastructure

- **Runtime:** Pure Python, no external APIs in graders
- **Memory:** < 200MB (no ML models loaded server-side)
- **Latency:** < 50ms per step (grader is pure dict lookups)
- **Concurrency:** One episode per server instance (stateless reset on each call)
- **Inference runtime:** ~3-4 minutes for all 3 tasks on any LLM API

---

## Impact

- **1.07 million ASHA workers** in India make healthcare decisions for 600 million people
- Only **7.2% of ASHAs** know all key danger signs during pregnancy (published research)
- India's maternal mortality rate: **97/100,000** — most preventable with correct triage
- This environment trains agents on the same protocol ASHAs are supposed to use

*Graders backed by official Government of India IMNCI protocol.*
