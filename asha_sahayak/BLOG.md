# ASHA Sahayak: Training AI to Save Lives at 2 AM in Rural India

*A submission for the Meta PyTorch OpenEnv Hackathon x Scaler SST India 2026 Grand Finale*

---

## The Problem

Savitri is 31 years old. She lives in Sitapur district, Uttar Pradesh. She is one of India's 1.07 million ASHA (Accredited Social Health Activist) workers — the only healthcare contact for 600 million rural Indians.

Her training: 23 days. Her tools: a printed booklet and a basic kit. Her job: decide whether a feverish child needs a hospital 40 km away at 2 AM, whether a new mother's bleeding is normal or PPH, whether a listless newborn can wait until morning.

India's maternal mortality rate is 97 per 100,000 live births. Savitri is why it isn't 300.

**The gap**: The IMNCI protocol she follows is correct. Her training is not enough. 40 danger signs across 7 disease domains, in a language that is not her mother tongue, for emergencies she may see once a year.

**Our solution**: An OpenEnv RL environment that trains AI models to help Savitri ask the right questions, in the right order, and make the right referral — grounded in the official Indian Government IMNCI protocol.

---

## What We Built

### The Environment

ASHA Sahayak is a multi-turn clinical triage environment. Each episode presents a patient case to the agent. The agent:

1. **Asks clarifying questions** — gathering clinical information from the ASHA worker
2. **Makes a referral decision** — REFER_IMMEDIATELY / REFER_WITHIN_24H / TREAT_AT_HOME / MONITOR

The environment scores the decision against the official IMNCI ground truth. Rewards are dense (per turn), deterministic, and bounded in (0.001, 0.999).

### 31 Clinical Cases, 7 Domains

| Domain | Cases | Example |
|---|---|---|
| Pediatric | 8 | Severe pneumonia, meningitis, dehydration |
| Maternal | 8 | Eclampsia, PPH, puerperal sepsis, cord prolapse |
| Neonatal | 9 | Birth asphyxia, kernicterus, omphalitis, LBW/KMC |
| TB | 2 | Adult TB, pediatric contact tracing |
| Malaria | 2 | Uncomplicated vs cerebral malaria |
| NCD | 1 | Hypertension screening (CBAC) |
| Adolescent | 1 | Severe anaemia with cardiac signs (RKSK) |

### Three Hackathon Themes

**Theme 1 — Multi-Agent**: ASHA Worker + PHC Doctor two-phase episodes. The ASHA worker gathers information and makes a referral. The PHC Doctor receives only the referral note (information asymmetry) and makes the final disposition.

```
Combined Reward = 0.55 × R_doctor + 0.30 × R_asha + 0.15 × R_communication
```

**Theme 3.1 — Tool Use**: 5 deterministic clinical tools callable via `[TOOL: name(args)]` syntax:
- `muac_classifier` — SAM/MAM/Normal nutritional status (NHM SAM Guidelines)
- `gestational_age` — EDD calculation from LMP (Naegele's Rule)
- `drug_dose` — Pediatric dosing by weight (IMNCI Drug Formulary)
- `jssk_eligibility` — Free government entitlements checker (NHM JSSK 2011)
- `cbac_scorer` — NCD risk scoring for hypertension referral (NHM NPCDCS)

**Theme 4 — Self-Improvement**: Adaptive curriculum using Multi-Armed Bandit sampling. Categories with higher failure rates get higher sampling weight. The environment drives the agent toward its weakest clinical domains automatically.

```python
weight(category) = 0.3 + failure_rate(category)
```

### The Reward Formula

```
R = 0.40 × R_referral          # REFER_IMMEDIATELY / REFER_WITHIN_24H / TREAT_AT_HOME / MONITOR
  + 0.25 × R_urgency           # immediate / within_24h / routine / monitor
  + 0.20 × R_primary_concern   # semantic alias matching (19 alias groups)
  + 0.15 × R_information_gathering  # did the agent ask clarifying questions?
  + 0.05  terminal bonus       # correct referral AND asked at least one question

Clamped strictly to (0.001, 0.999)
```

**Dangerous undertriage penalty**: TREAT_AT_HOME when REFER_IMMEDIATELY is correct → referral score capped at 0.1. This penalizes the most dangerous clinical error.

---

## Training Results

We trained Qwen3-0.6B on ASHA Sahayak using GRPO (Group Relative Policy Optimization) via HuggingFace TRL.

![Training Reward Curve](assets/training_reward_curve.png)
*Left: Episode reward over 200 training steps, by difficulty. Right: Before vs After GRPO vs Round 1 zero-shot LLM baseline.*

**Round 1 scores** (zero-shot Qwen2.5-72B via inference.py):
- Easy: 0.999 | Medium: 0.675 | Hard: 0.999 | **Overall: 0.849**

The medium task score (0.675) is the primary training target. Medium cases require nuanced reasoning: distinguishing gestational diabetes from normal pregnancy, quantifying CBAC NCD risk, recognizing pathological vs physiological jaundice.

### Multi-Agent Episode Results

![Multi-Agent Breakdown](assets/multi_agent_reward_breakdown.png)
*Reward breakdown for a two-phase ASHA Worker → PHC Doctor episode. Combined reward: 0.877.*

---

## GRPO Training — How It Works

```python
class AshaToolEnv:
    def reset(self, task_id="easy", seed=42) -> str:
        # Returns initial patient presentation
        
    def ask_question(self, question: str) -> str:
        """Ask ASHA worker a clinical clarifying question.
        Args:
            question: e.g. 'Does the child have fast breathing?'
        Returns: ASHA worker's clinical observation
        """
        
    def make_referral(self, referral_decision: str, urgency: str, primary_concern: str) -> str:
        """Make final referral decision.
        Args:
            referral_decision: REFER_IMMEDIATELY | REFER_WITHIN_24H | TREAT_AT_HOME | MONITOR
            urgency: immediate | within_24h | routine | monitor  
            primary_concern: clinical concern e.g. severe_pneumonia
        Returns: Clinical feedback with score breakdown
        """
```

TRL discovers `ask_question()` and `make_referral()` as tools automatically from their docstrings. The environment supports 64 concurrent sessions for GRPO parallel rollouts (`SUPPORTS_CONCURRENT_SESSIONS = True`).

Full training notebook: [`training/asha_grpo_training.ipynb`](training/asha_grpo_training.ipynb)

---

## Why This Matters

1.07 million ASHA workers. 600 million rural Indians. 97 maternal deaths per 100,000 births.

A model trained on this environment learns what Savitri needs to know — not from textbooks, but from failure. Each wrong referral decision in training becomes a gradient signal. Each correct identification of a danger sign at 2 AM becomes a reward. The environment trains exactly the capability that saves lives.

---

## Links

- **HuggingFace Space**: https://huggingface.co/spaces/sreenathmmenon/asha-sahayak
- **GitHub**: https://github.com/sreenathmmenon/asha-sahayak
- **Training Notebook**: [training/asha_grpo_training.ipynb](training/asha_grpo_training.ipynb)

---

*Ground truth: Indian Government IMNCI Protocol, NHM Guidelines (SAM, GDM, JSSK, NPCDCS), NVBDCP, NTEP, RKSK*  
*Meta PyTorch OpenEnv Hackathon x Scaler SST India 2026*
