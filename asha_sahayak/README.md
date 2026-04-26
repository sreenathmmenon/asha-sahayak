---
title: ASHA Sahayak
emoji: 🏥
colorFrom: red
colorTo: pink
sdk: docker
pinned: true
license: apache-2.0
short_description: AI clinical decision support for India's 1.07M ASHA workers
---

# ASHA Sahayak — AI Clinical Decision Support for Frontline Health Workers

> **OpenEnv RL Environment** | Multi-Agent | Tool Use | Adaptive Curriculum  
> Meta PyTorch OpenEnv Hackathon x Scaler SST India 2026 — Grand Finale

**🔗 HuggingFace Space**: https://huggingface.co/spaces/sreenathmmenon/asha-sahayak  
**📓 Training Notebook (clean, re-runnable)**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreenathmmenon/asha-sahayak/blob/main/asha_sahayak/training/asha_grpo_training.ipynb)  
**📓 Run 2 — 200 steps with outputs**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreenathmmenon/asha-sahayak/blob/main/asha_sahayak/training/asha_grpo_training_with_outputs_run2_200steps.ipynb)  
**📓 Run 3 — 400 steps with outputs**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreenathmmenon/asha-sahayak/blob/main/asha_sahayak/training/asha_grpo_training_with_outputs_run3_400steps.ipynb)  
**📝 Blog Post**: [BLOG.md](BLOG.md)

---

## The Story

**The Scale.** India has 600 million rural citizens spread across 640,000 villages. Their first — and often only — contact with the healthcare system is not a doctor. It is an ASHA worker: a woman from their own village, trained for 23 days, equipped with a printed booklet and a basic kit. There are 1.07 million of them. Every day, they make life-or-death triage decisions — whether to refer a feverish child to a hospital 40 kilometers away, whether a new mother's bleeding is normal postpartum or PPH, whether a listless newborn can wait until morning.

**The Person.** Savitri is 31 years old, from Sitapur district in Uttar Pradesh. She covers 200 households. Her training covered the IMNCI protocol — a decision tree with 40+ danger signs across 6 disease categories. She has no internet, no smartphone, and no doctor on call. When a child stops breathing at 2 AM, she is the system. India's maternal mortality rate is 97 per 100,000 live births. Savitri is why that number isn't 300.

**The Gap.** The IMNCI protocol is correct. The training is inadequate. 23 days to memorize 40 danger signs across pneumonia, malaria, diarrhea, eclampsia, sepsis, and 30 more conditions — in a language (English) that is not her mother tongue, for cases she may see only once a year. When the printed booklet is ambiguous, she improvises. Sometimes correctly. Sometimes not.

**The Solution.** ASHA Sahayak is an OpenEnv reinforcement learning environment that trains AI models to assist ASHA workers with clinical triage — asking the right questions in the right order, applying the IMNCI protocol correctly, recognizing the 15 danger signs that require immediate referral. The ground truth is the official Indian Government IMNCI protocol. The reward signal is deterministic, reproducible, and grounded in real clinical outcomes. The AI learns what Savitri needs to know.

> *"The village calls her didi. The government calls her a volunteer. The WHO calls her a global health leader. She calls herself: underpaid, overworked, and irreplaceable."*

> *"She brings the hope. We want to bring the answer."*

📖 Full story: [BLOG.md](BLOG.md)

---

## Results & Training Evidence

### GRPO Training — Qwen3-0.6B, 3 Runs

![Run 1 vs Run 2 — Both Runs Overlaid](assets/training_comparison_overlaid.png)
*Run 1 (gray) used regex parsing — concern component nearly always 0. Run 2 (blue) switched to JSON output — unlocked all 4 reward components, jumped from 0.31 baseline to 0.947 peak (+142%).*

| Metric | Run 1 — 200 steps | Run 2 — 200 steps | Run 3 — 400 steps |
|---|---|---|---|
| Baseline reward | ~0.47 | 0.31 | 0.14 |
| Final reward | ~0.52 | **0.75** | 0.66 |
| Peak reward | ~0.75 | **0.947** | **0.947** |
| Colab notebook | — | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreenathmmenon/asha-sahayak/blob/main/asha_sahayak/training/asha_grpo_training_with_outputs_run2_200steps.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreenathmmenon/asha-sahayak/blob/main/asha_sahayak/training/asha_grpo_training_with_outputs_run3_400steps.ipynb) |

**Best result (Run 2):** baseline 0.31 → final **0.75** → peak **0.947** · +142% improvement  
**Run 3 confirms:** same 0.947 peak reached at 400 steps — consistent training signal across runs.

![Run 2 — Training Reward Curve](assets/training_reward_curve.png)
*Run 2: baseline 0.31, strong upward trend steps 0→100, peak 0.947, final 0.75.*

![Run 3 — Training Reward Curve (400 steps)](assets/training_reward_curve_run3.png)
*Run 3: 400 steps, peak 0.947 — confirming training consistency at extended horizon.*

| Model | Qwen3-0.6B, 20M trainable params (3.3%) |
|---|---|
| Algorithm | GRPO via TRL + Unsloth · NVIDIA L4 |
| Trained checkpoint | [sreenathmmenon/asha-sahayak-grpo](https://huggingface.co/sreenathmmenon/asha-sahayak-grpo) |

**Per-component breakdown — Run 2 (baseline 0.31 → trained 0.75):**

| Reward Component | Weight | Baseline | Trained | Δ |
|---|---|---|---|---|
| Referral correctness | 40% | 0.18 | 0.71 | **+0.53** |
| Urgency accuracy | 25% | 0.22 | 0.68 | **+0.46** |
| Primary concern ID | 20% | 0.09 | 0.61 | **+0.52** |
| Information gathering | 15% | 0.91 | 0.95 | **+0.04** |
| **Composite** | | **0.31** | **0.75** | **+0.44** |

*Run 1 baseline ~0.47 reflects regex-only parsing where the concern component was mostly unscored; Run 2 baseline 0.31 reflects structured JSON output enabling all 4 components to be measured.*

The reward curve shows a strong upward trend reaching **0.947 peak at step 189** — a +142% improvement over baseline. The model learned to output structured JSON decisions, ask clarifying questions before deciding, and correctly distinguish REFER_IMMEDIATELY from TREAT_AT_HOME based on IMNCI danger signs.

### Held-Out Evaluation (Seeds 1000–1099)

To verify the model generalizes beyond its 44 training cases, we evaluated the trained checkpoint on 100 held-out seeds never seen during training.

| Metric | Value |
|---|---|
| Held-out seeds | 1000–1099 (never seen during training) |
| Episodes | 100 |
| Mean reward | **0.43** |
| Referral correctness | 0.70 |
| Urgency accuracy | 0.48 |
| Dangerous undertriage rate | **2%** (hard safety gate active) |

Full results: [`assets/heldout_evaluation.json`](assets/heldout_evaluation.json).

### Before vs After — Clinical Decision Quality

| Scenario | Untrained Model | Trained Model |
|---|---|---|
| 8-month-old, fast breathing | "Monitor at home, give fluids" ❌ | Asks about chest indrawing → REFER_IMMEDIATELY ✅ |
| Pregnant woman, headache + blurred vision | "Rest and check later" ❌ | Identifies pre-eclampsia → REFER_IMMEDIATELY ✅ |
| Newborn, Day 3 jaundice, feeding well | "Refer to hospital" ❌ (over-triage) | MONITOR — physiological jaundice, normal ✅ |

### All 3 Runs — What Changed

![Run 1 Training Curve](assets/training_reward_curve_run1.png)
*Run 1: regex concern extraction — model output not matching regex defaulted to "general", concern reward ≈0. Baseline ~0.47, final ~0.52.*

**Run 1 → Run 2:** Switched to structured JSON output + JSON parsing. Unlocked the concern reward component (+0.52). Overall reward 0.52 → 0.75.

**Run 2 → Run 3:** Extended to 400 steps. Same peak (0.947) confirmed — training signal is consistent. Final settled at 0.66 due to reward oscillation at longer horizons with a 0.6B model.

### Multi-Agent Episode Reward Breakdown

![Multi-Agent Reward Breakdown](assets/multi_agent_reward_breakdown.png)
*Reward breakdown for a two-phase ASHA Worker → PHC Doctor episode. Combined reward formula: 0.55 × R_doctor + 0.30 × R_asha + 0.15 × R_comm = 0.877.*

### Round 1 Baseline (Zero-Shot Qwen2.5-72B)

| Task | Seed | Score | Notes |
|---|---|---|---|
| Easy | 42 | ~0.999 | Near-perfect on clear danger signs |
| Medium | 123 | ~0.675 | Room for improvement on complex multi-factor cases |
| Hard | 500 | ~0.999 | Correctly handles neonatal emergencies and cord prolapse |
| **Overall** | — | **~0.849** | **Round 1 submission score** |

### Training Notebooks

| Notebook | Description | Colab |
|---|---|---|
| [`asha_grpo_training.ipynb`](training/asha_grpo_training.ipynb) | Clean, re-runnable | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreenathmmenon/asha-sahayak/blob/main/asha_sahayak/training/asha_grpo_training.ipynb) |
| [`...run2_200steps.ipynb`](training/asha_grpo_training_with_outputs_run2_200steps.ipynb) | Run 2 with full outputs — final 0.75, peak 0.947 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreenathmmenon/asha-sahayak/blob/main/asha_sahayak/training/asha_grpo_training_with_outputs_run2_200steps.ipynb) |
| [`...run3_400steps.ipynb`](training/asha_grpo_training_with_outputs_run3_400steps.ipynb) | Run 3 with full outputs — 400 steps, peak 0.947 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sreenathmmenon/asha-sahayak/blob/main/asha_sahayak/training/asha_grpo_training_with_outputs_run3_400steps.ipynb) |

### Blog Post

Full writeup: [`BLOG.md`](BLOG.md) — covers the problem, environment design, reward formula, training results, and why this matters for 1.07M ASHA workers.

---

## Themes

| Theme | Claim | Implementation |
|---|---|---|
| **Theme 1** | Multi-Agent Interactions | ASHA Worker + PHC Doctor two-phase episodes. Doctor sees referral note only (information asymmetry). Combined reward: `0.55 × R_doctor + 0.30 × R_asha + 0.15 × R_comm` |
| **Theme 3.1** | World Modeling / Tool Use | 5 deterministic clinical tools callable via `[TOOL: name(args)]` syntax. MUAC classifier, gestational age, drug dosage, JSSK eligibility, CBAC NCD scorer. |
| **Theme 4** | Self-Improvement | Adaptive curriculum using Multi-Armed Bandit (arxiv 2505.14970). Categories with higher failure rates get higher sampling weight. Converges toward hardest clinical domains. |

---

## Reward Formula

```
R = 0.40 × R_referral          (REFER_IMMEDIATELY / REFER_WITHIN_24H / TREAT_AT_HOME / MONITOR)
  + 0.25 × R_urgency           (immediate / within_24h / routine / monitor)
  + 0.20 × R_primary_concern   (semantic alias matching for flexible phrasing)
  + 0.15 × R_information_gathering  (did agent ask clarifying questions?)
  + 0.05 terminal bonus         (correct referral AND asked at least one question)

Clamped strictly to (0.001, 0.999) — never exactly 0 or 1
```

**Dangerous undertriage penalty**: TREAT_AT_HOME when correct answer is REFER_IMMEDIATELY → referral score capped at 0.1

**Multi-agent combined reward**:
```
R_episode = 0.55 × R_doctor + 0.30 × R_asha + 0.15 × R_communication
```

---

## Case Catalog — 44 Cases, 7 Domains

| ID | Title | Domain | Difficulty | Correct Referral | Teaching Point |
|---|---|---|---|---|---|
| E01 | Severe Pneumonia — Chest Indrawing | Pediatric | Easy | REFER_IMMEDIATELY | Chest indrawing = severe pneumonia |
| E02 | Eclampsia — Seizure in Pregnancy | Maternal | Easy | REFER_IMMEDIATELY | Convulsions in pregnancy = emergency |
| E03 | Pneumonia — Treat at Home | Pediatric | Easy | TREAT_AT_HOME | Fast breathing without chest indrawing = home |
| E04 | Neonatal Danger Signs — Suspected Sepsis | Neonatal | Easy | REFER_IMMEDIATELY | Fever + lethargy + fast breathing = refer |
| E05 | Mild Diarrhea — Home Care | Pediatric | Easy | TREAT_AT_HOME | No dehydration signs = ORS at home |
| E06 | Neonatal Jaundice — Day 3, Physiological | Neonatal | Easy | MONITOR | Face only, Day 2-3, feeding well = normal |
| E07 | Skin Pustules — Localized < 10 | Neonatal | Easy | TREAT_AT_HOME | < 10 pustules, no fever = home |
| E08 | Uncomplicated Malaria — RDT+, No Danger Signs | Malaria | Easy | TREAT_AT_HOME | RDT+ alone ≠ refer — check danger signs |
| E09 | Mild Fever — Viral URTI | Pediatric | Easy | TREAT_AT_HOME | No danger signs = home management |
| E10 | Oral Thrush — Mild Candidiasis in Infant | Neonatal | Easy | TREAT_AT_HOME | Well-feeding infant = gentian violet at home |
| E11 | Normal Antenatal Visit — Low-Risk Pregnancy | Maternal | Easy | TREAT_AT_HOME | No danger signs = routine ANC |
| M01 | Pre-eclampsia — Severe Features | Maternal | Medium | REFER_IMMEDIATELY | Headache + blurred vision + BP 145/95 = emergency |
| M02 | Severe Dehydration with Lethargy | Pediatric | Medium | REFER_IMMEDIATELY | Lethargy = general danger sign, always refer |
| M03 | TB Suspect — Referral for Sputum | TB | Medium | REFER_WITHIN_24H | Cough ≥2 weeks + weight loss = presumptive TB |
| M04 | Some Dehydration — PHC Referral | Pediatric | Medium | REFER_WITHIN_24H | Sunken eyes + restlessness = some dehydration |
| M05 | Postpartum Hemorrhage | Maternal | Medium | REFER_IMMEDIATELY | Soaking 2 pads/hour + clots = PPH |
| M06 | Neonatal Jaundice — Within 24 Hours | Neonatal | Medium | REFER_WITHIN_24H | Within 24h = always pathological |
| M07 | Gestational Diabetes Risk — 26 Weeks | Maternal | Medium | REFER_WITHIN_24H | Polyuria + polydipsia + family DM = OGTT |
| M08 | Severe Anaemia in Pregnancy | Maternal | Medium | REFER_IMMEDIATELY | Breathlessness at rest = cardiac decompensation |
| M09 | Pediatric TB — Household Contact | TB | Medium | REFER_WITHIN_24H | Child <5 + TB contact = IPT screening |
| M10 | NCD Risk — Possible Hypertension | NCD | Medium | REFER_WITHIN_24H | CBAC score ≥4 = refer for BP check |
| M11 | Mild Diarrhea — Adequate Hydration | Pediatric | Medium | TREAT_AT_HOME | Alert + drinks well = no dehydration |
| M12 | Iron Deficiency Anaemia — Mild, Adolescent | Adolescent | Medium | TREAT_AT_HOME | Mild anaemia + no cardiac signs = WIFS |
| M13 | Post-Illness Recovery — Post Pneumonia | Pediatric | Medium | MONITOR | Antibiotic complete + fever resolved = follow-up |
| M14 | Mild Underweight — Growth Monitoring | Pediatric | Medium | MONITOR | MUAC >125mm = monitor, not NRC |
| M15 | Suspected UTI in Pregnancy | Maternal | Medium | REFER_WITHIN_24H | Dysuria + fever in pregnancy = refer |
| H01 | Neonatal Hypothermia with Sepsis Signs | Neonatal | Hard | REFER_IMMEDIATELY | Cold + lethargic + 7 days old = sepsis |
| H02 | Severe Acute Malnutrition with Complications | Pediatric | Hard | REFER_IMMEDIATELY | MUAC <115mm + edema + lethargy = NRC |
| H03 | Antepartum Haemorrhage — 3rd Trimester | Maternal | Hard | REFER_IMMEDIATELY | Bleeding + hypotension + tachycardia = shock |
| H04 | Very Severe Febrile Disease — Meningitis | Pediatric | Hard | REFER_IMMEDIATELY | Stiff neck + bulging fontanelle + convulsion |
| H05 | Omphalitis Progressing to Sepsis | Neonatal | Hard | REFER_IMMEDIATELY | Red streaks from cord = systemic sepsis |
| H06 | Moderate Malnutrition — Community Management | Pediatric | Hard | MONITOR | MAM (MUAC 11.5-12.5) + no edema = CMAM |
| H07 | Birth Asphyxia — Baby Not Crying | Neonatal | Hard | REFER_IMMEDIATELY | Limp + no cry + cyanosis = resuscitate NOW |
| H08 | Severe Neonatal Jaundice — Kernicterus | Neonatal | Hard | REFER_IMMEDIATELY | Yellow palms + back arching = brain damage |
| H09 | Puerperal Sepsis — 4 Days Postpartum | Maternal | Hard | REFER_IMMEDIATELY | Foul lochia + high fever + confusion |
| H10 | Adolescent Severe Anaemia — Cardiac Signs | Adolescent | Hard | REFER_IMMEDIATELY | Syncope + tachycardia + menorrhagia = RKSK |
| H11 | Cerebral Malaria — Unconscious | Malaria | Hard | REFER_IMMEDIATELY | Falciparum + seizures + unconscious |
| H12 | Cord Prolapse — Obstetric Emergency | Maternal | Hard | REFER_IMMEDIATELY | Cord visible = knee-chest position + 108 now |
| H13 | Preterm Low Birth Weight — KMC Decision | Neonatal | Hard | REFER_WITHIN_24H | 1.8 kg = refer + start KMC immediately |
| H14 | Stable Gestational Hypertension — No Severe Features | Maternal | Hard | MONITOR | BP 140/90, no proteinuria, no symptoms = monitor |
| H15 | Moderate Malnutrition — CMAM Follow-Up | Pediatric | Hard | MONITOR | Post-NRC recovery, MUAC improving = CMAM |
| H16 | Type 2 Diabetes — Uncontrolled, NCD Programme | NCD | Hard | REFER_WITHIN_24H | Polyuria + weight loss + family DM = screen |
| H17 | Adolescent Severe Anaemia — Breathlessness | Adolescent | Hard | REFER_WITHIN_24H | Marked pallor + breathlessness on exertion |
| H18 | Possible TB — Adult, Chronic Cough | TB | Hard | REFER_WITHIN_24H | Cough >2 weeks + night sweats + haemoptysis |

---

## Clinical Ground Truth — Government of India Sources

Every case, referral decision, and danger sign in this environment is grounded in official Indian Government clinical protocols. No clinical judgment was invented — all ground truth is derived from the sources below.

| Protocol | Full Name | Governs |
|---|---|---|
| **IMNCI** | Integrated Management of Neonatal and Childhood Illness | All pediatric + neonatal cases (E01–E11, M02, M04, M11, M13, M14, H02, H04, H06, H07, H08, H15) |
| **NHM Maternal Guidelines** | National Health Mission Maternal Health Protocols | All maternal cases (E02, E11, M01, M05, M07, M08, M15, H03, H09, H12, H14) |
| **NTEP** | National Tuberculosis Elimination Programme | TB cases (M03, M09, H18) |
| **NVBDCP** | National Vector Borne Disease Control Programme | Malaria cases (E08, H11) |
| **JSSK** | Janani Shishu Suraksha Karyakram | Free transport + treatment entitlements for mothers and newborns |
| **RKSK** | Rashtriya Kishor Swasthya Karyakram | Adolescent health cases (M12, H10, H17) |
| **NPCDCS / CBAC** | National Programme for Non-Communicable Diseases + Community Based Assessment Checklist | NCD cases (M10, H16) |
| **SAM Guidelines** | NHM Severe Acute Malnutrition Operational Guidelines | MUAC thresholds and SAM/MAM classification (M04, H02, H06, H15) |
| **NHSRC ASHA Incentive Schedule** | April 2024 | Task definitions and incentive structure for ASHA workers |

*All referral decisions, danger sign thresholds, drug doses, and urgency classifications are taken directly from these protocols. The environment does not invent clinical rules.*

---

## Clinical Tools (Theme 3.1)

Agents can call tools using `[TOOL: tool_name(arg=value, arg2=value2)]` syntax in their question field.

| Tool | Purpose | Source |
|---|---|---|
| `muac_classifier` | Classify MUAC for SAM/MAM/Normal | NHM SAM Operational Guidelines |
| `gestational_age` | Calculate gestational age and EDD from LMP | Naegele's Rule |
| `drug_dose` | Pediatric drug dose by weight | IMNCI Drug Formulary, GoI |
| `jssk_eligibility` | Check JSSK entitlements for pregnant women/newborns | NHM JSSK Circular 2011 |
| `cbac_scorer` | CBAC NCD risk score (refer if ≥4) | NHM NPCDCS Guidelines |

**Example tool calls:**
```
[TOOL: muac_classifier(age_months=18, muac_mm=108)]
→ {"classification": "SAM", "referral": "refer_nrc", ...}

[TOOL: gestational_age(lmp_date=2025-10-15)]
→ {"gestational_age_weeks": 28, "trimester": "3rd", "edd": "2026-07-22", ...}

[TOOL: drug_dose(drug_name=amoxicillin, weight_kg=12)]
→ {"dose": "19.2ml", "frequency": "3x daily", "duration_days": 5, ...}

[TOOL: cbac_scorer(age=52, tobacco_use=True, family_history_hypertension=True, known_bp_high=False, alcohol_use=False, family_history_diabetes=False, family_history_heart_disease=False, physical_activity=low, known_diabetes=False)]
→ {"cbac_score": 4, "risk_level": "moderate", "refer_to_anm": true, ...}
```

---

## Multi-Agent Episodes (Theme 1)

Episodes have two phases with information asymmetry:

**Phase 1 — ASHA Worker** (turns 1 to N-1):
- Agent plays community health worker
- Asks clarifying questions, gathers clinical information
- Makes final referral decision
- Produces structured referral note for PHC Doctor

**Phase 2 — PHC Doctor** (turn N):
- Agent plays PHC Doctor
- Receives ONLY the referral note (not the raw conversation)
- Makes disposition: `manage_at_phc` | `refer_to_fru` | `refer_to_district`

```bash
# Start multi-agent episode
POST /multi/reset  {"task_id": "medium", "seed": 42}

# ASHA Worker turns
POST /multi/step/asha  {"question": "Any chest indrawing?", "referral_decision": "PENDING", ...}
POST /multi/step/asha  {"referral_decision": "REFER_WITHIN_24H", "urgency": "within_24h", ...}

# PHC Doctor turn
POST /multi/step/doctor  {"disposition": "manage_at_phc", "rationale": "TB case, PHC DOTS program"}
→ {"done": true, "reward": 0.847, "breakdown": {...}}
```

---

## Adaptive Curriculum (Theme 4)

The environment tracks success rates per clinical category and uses Multi-Armed Bandit sampling to focus training on categories where the agent is weakest.

```
weight(category) = 0.3 + failure_rate(category)
```

At episode start, cases are sampled with these weights — failing categories get more exposure. This implements Self-Evolving Curriculum (arxiv 2505.14970).

Categories: `pediatric`, `maternal`, `neonatal`, `tb`, `ncd`, `adolescent`, `malaria`

---

## GRPO Training

The environment is GRPO-ready:
- `SUPPORTS_CONCURRENT_SESSIONS = True`
- `max_concurrent_sessions = 64`
- Deterministic rewards for stable gradient signal
- Three difficulty levels for curriculum training

```python
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    use_vllm=True,
    num_generations=4,
    max_completion_length=1024,
    gradient_accumulation_steps=64,
    output_dir="asha-sahayak-grpo",
)
```

---

## Setup

```bash
# Clone and install
git clone https://github.com/sreenathmmenon/asha-sahayak
cd asha-sahayak
uv sync

# Run locally
uv run server

# Or with uvicorn
uvicorn asha_sahayak.server.app:app --host 0.0.0.0 --port 7860
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode. Body: `{task_id, seed}`. Returns `{observation, session_id}` |
| `/step` | POST | Agent action. Header: `X-Session-ID`. Body: `{referral_decision, urgency, primary_concern, question}` |
| `/state` | GET | Episode state. Header: `X-Session-ID` |
| `/health` | GET | Health check |
| `/metadata` | GET | Environment metadata |
| `/multi/reset` | POST | Start multi-agent episode |
| `/multi/step/asha` | POST | ASHA Worker action. Header: `X-Session-ID` |
| `/multi/step/doctor` | POST | PHC Doctor action. Header: `X-Session-ID` |
| `/multi/observations` | GET | Role-scoped observations for both agents |

---

*Ground truth source: Indian Government IMNCI Protocol, NHM Guidelines, NVBDCP, NTEP, JSSK, NPCDCS*  
*Built for Meta PyTorch OpenEnv Hackathon x Scaler SST India 2026*
