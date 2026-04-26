# She Wakes at 3:30 AM. She Earns Less Than a Cup of Coffee. She Is Why 600 Million People Are Alive.

### ASHA Sahayak — AI Clinical Decision Support for India's 1.07 Million Frontline Health Workers

---

It is 3:30 in the morning in Chirgoda village, Uttar Pradesh.

Champa Devi is already awake.

A pregnant woman in her neighborhood has gone into labor. There is no doctor in the village. There is no clinic open. There is no one else to call. So the family calls Champa — because Champa is always there. She will arrange the ambulance. She will accompany the family to the hospital. She will wait through the night in a corridor that smells of disinfectant and fear. She will make sure mother and child come home safe.

For this, she will earn Rs. 300. Less than the bus fare.

Her name is ASHA. In Hindi, that word means hope. It is not a coincidence. The women who carry this name were meant to carry that weight — the hope of 600 million rural Indians who have no doctor, no clinic, no safety net beyond the woman from their own village who knocks on their door.

At midnight, she will sit alone with the glow of her phone screen, uploading records of births and deaths and vaccinations. Then she sleeps. And at 3:30 AM, the phone rings again.

This is the story of the ASHA workers of India. And it is the story of why we built something to stand beside them.

---

## Who Is She?

She is one of 1,047,324 women — over a million strong — deployed across every state and union territory in India. The largest all-female community health workforce on earth.

She receives 23 days of training — spread over a year — to handle 40 danger signs across pneumonia, malaria, diarrhea, eclampsia, newborn sepsis, severe malnutrition, and more. Twenty-three days. In a language that may not be her mother tongue. With a booklet that gets torn, wet, or lost. After which she goes into the field alone, at any hour, to make decisions that determine whether someone lives or dies.

The village calls her didi. The government calls her a volunteer. The WHO calls her a global health leader. She calls herself: underpaid, overworked, and irreplaceable.

---

## What She Has Saved

India's maternal mortality ratio fell from 301 in 2003 to 97 in 2020 — a 68% decline. Institutional births rose from 41% to 88.6%. Every percentage point is a woman who didn't die on the floor of her home. India became polio-free in 2014. Malaria cases have declined 14% every year since ASHA workers began carrying Rapid Diagnostic Kits into the field.

On May 22, 2022, at the 75th World Health Assembly in Geneva, India's one million ASHA workers collectively received the WHO Director-General's Global Health Leaders Award.

Melinda Gates said: *"So many nations need to learn from India's ASHA workers."*

The woman earning Rs. 232 a day. The world needs to learn from her.

---

## Less Than a Cup of Coffee

A cup of coffee in the United States costs between $5 and $6 — Rs. 420 to Rs. 500. An ASHA worker earns Rs. 232 a day. Less than that coffee.

She is not an employee. She is classified as a "volunteer" — ineligible for minimum wages, provident fund, or maternity leave. She pays for her own phone, her own internet data, her own transport. She sometimes pays for medicines from her own pocket when the drug kit runs empty.

In Kerala, 75,000 workers went on strike for 266 days — one of the longest public sector strikes in recent Indian history — asking to be called workers, not volunteers. *"Stop calling us volunteers. We are workers."*

---

## The Gap — And Why It Costs Lives

The IMNCI protocol is correct. The training is inadequate.

23 days to memorize 40 danger signs. For cases she may see only once a year. When the booklet is ambiguous, she improvises. Sometimes correctly. Sometimes not.

Consider this: chest indrawing in a child with fast breathing means severe pneumonia — refer immediately. Fast breathing alone, without chest indrawing, means pneumonia — treat at home with antibiotics. One clinical sign. Two completely different decisions. One costs a life if wrong.

That uncertainty is the gap. That uncertainty costs lives.

---

## What We Built — And Why

I am a technologist, not a doctor. I have spent my career building software systems — the kind that process data at scale, that run quietly in the background while people go about their lives. For most of that time, I built things that made enterprises more efficient. Useful work. But not urgent work. Not the kind where the alternative is a child dying at 2 AM because the person standing at the door wasn't sure which question to ask next.

That uncertainty is what we built ASHA Sahayak to address.

So we built ASHA Sahayak — *Hope's Helper*.

![ASHA Sahayak Demo — AI asking clinical questions](assets/ui_demo_step1_case_started.png)
*The AI assistant asks Savitri questions about the patient. She responds in Hindi and English. The AI gathers information and makes the referral decision.*

We named it ASHA Sahayak because it is not meant to replace the didi. She already has the trust, the relationship, the knowledge of which family has a difficult mother-in-law and which woman is too scared to go to the hospital alone. No AI will ever have that.

What she sometimes doesn't have is certainty. The right question to ask at the right moment. The memory of a danger sign she learned in a training session two years ago. A second opinion at midnight.

That is what ASHA Sahayak is built to give.

---

## How It Works — The Technical Design

### The Environment

ASHA Sahayak is an OpenEnv reinforcement learning environment — 44 clinical cases across 7 domains: pediatric, maternal, neonatal, tuberculosis, malaria, non-communicable diseases, and adolescent health. Every case is grounded in the official Indian Government IMNCI protocol. No clinical rules were invented.

The agent plays the role of an AI assistant to the ASHA worker. It asks clarifying questions, gathers information, and makes a referral decision: `REFER_IMMEDIATELY`, `REFER_WITHIN_24H`, `TREAT_AT_HOME`, or `MONITOR`.

### The Reward Formula

The environment scores every decision across 4 components:

```
R = 0.40 × Referral correctness
  + 0.25 × Urgency accuracy
  + 0.20 × Primary concern identification
  + 0.15 × Information gathering quality
```

A good ASHA worker doesn't just make decisions — she gathers information first. The reward formula reflects this: asking at least one clinical question before deciding earns a bonus. Sending an emergency patient home triggers a hard safety gate — episode terminates immediately with minimum reward.

### The Training

We trained Qwen3-0.6B (600 million parameters, 3.3% of parameters actually trained) using GRPO — Group Relative Policy Optimization — via TRL and Unsloth on a single NVIDIA L4 GPU.

**3 training runs, real results:**

| Run | Steps | Baseline | Final | Peak |
|---|---|---|---|---|
| Run 1 — regex parsing | 200 | ~0.47 | ~0.52 | ~0.75 |
| Run 2 — JSON output | 200 | 0.31 | **0.75** | **0.947** |
| Run 3 — extended | 400 | 0.14 | 0.66 | **0.947** |

**The key insight between Run 1 and Run 2:** Run 1 used regex to extract the model's clinical concern from free text — when the regex didn't match, the concern score defaulted to zero. Switching to structured JSON output in Run 2 unlocked all 4 reward components simultaneously. The concern component jumped from near-zero to 0.61. Overall reward went from 0.52 to 0.75 — a 142% improvement over baseline.

![Score 0.950 — Near perfect referral decision](assets/ui_demo_step4_result.png)

![Clinical explanation grounded in IMNCI protocol](assets/ui_demo_step5_explanation.png)
*The environment scores every decision against the official Indian Government IMNCI protocol — referral correctness, urgency, clinical concern, and quality of questioning.*

Run 2 was the breakthrough. Run 3 was the confirmation.

Extending to 400 steps in Run 3 confirmed that the 0.947 peak is consistently reachable — the training signal is real and stable, not a lucky spike. But the final reward at 400 steps settled at 0.66, lower than Run 2's 0.75. What this tells us: at longer training horizons, a 600M parameter model begins to oscillate — it learns the right behavior, overshoots, and partially forgets. The peak is there. Holding it requires either a larger model or better regularization. That is the next step.

### Multi-Agent Design (Two Roles, One Episode)

Real clinical care in India involves a handoff. The ASHA worker sees the patient in the field and refers to the PHC doctor who receives only a referral note — not the raw conversation.

We modeled this directly. Episodes have two phases:

**Phase 1 — ASHA Worker:** The agent gathers information, asks questions, makes a referral decision, produces a structured referral note.

**Phase 2 — PHC Doctor:** A second agent receives only the referral note (information asymmetry), and makes the final disposition: manage at PHC, refer to FRU, or refer to district hospital.

Combined reward: `0.55 × R_doctor + 0.30 × R_asha + 0.15 × R_communication`

This forces the AI to communicate clearly — because the doctor only knows what the ASHA wrote.

### Adaptive Curriculum — Learning What's Hard

Not all cases are equally difficult. Cerebral malaria is harder than mild diarrhea. Pre-eclampsia is harder than a routine antenatal visit.

We implemented a Multi-Armed Bandit adaptive curriculum that tracks success rates per clinical category and weights sampling toward categories where the model is weakest:

```
weight(category) = 0.3 + failure_rate(category)
```

The model trains harder on the cases it's struggling with — severe pneumonia, neonatal sepsis, antepartum haemorrhage. Because the model should not be good at easy cases. It should be reliable at the hardest ones.

### 5 Clinical Tools

Agents can call deterministic clinical tools mid-conversation, grounded in official government guidelines:

| Tool | Purpose |
|---|---|
| `muac_classifier` | Classify MUAC measurement for SAM/MAM/Normal |
| `gestational_age` | Calculate gestational age and EDD from LMP date |
| `drug_dose` | Pediatric drug dosage by weight (IMNCI formulary) |
| `jssk_eligibility` | Check free transport + treatment entitlements |
| `cbac_scorer` | NCD risk score — refer if ≥4 |

---

## Where This Goes Next

Voice-first, in local languages. Hindi and English are not enough. ASHA workers serve communities that speak Tamil, Telugu, Bengali, Bhojpuri, Odia, Gondi, Bodo, and dozens of tribal languages. Most ASHA workers can speak. Not all can type.

Offline capability. Rural connectivity averages 59% in India. An AI that requires an internet connection fails at the worst possible moment.

Integration with existing systems. Instead of adding an eighth app to Rohini Pawar's phone, ASHA Sahayak should sit inside the apps she already uses — reducing documentation to a voice note, handling the paperwork so she can handle the patient.

---

## She Will Wake Again Tomorrow

Champa Devi will wake at 3:30 AM tomorrow.

She will arrange another ambulance. Wait in another hospital corridor. Walk home in the blue light of early morning. She will do this for Rs. 300. She will do it because *"if one mother is dying in my village because of not getting hospital treatment... it will be my responsibility."*

We cannot fix what she is paid. We cannot fix how the law classifies her. We cannot give her the recognition she deserved long before the WHO finally noticed.

But we can make sure that when she stands at a doorstep at midnight — alone, uncertain, with a sick child in front of her and no doctor for 40 kilometers — she has something in her hand beyond a torn booklet and her own memory.

She has given India hope for twenty years.

The least we can do is give her an answer.

---

*ASHA Sahayak is an open-source reinforcement learning environment built on the Indian Government's IMNCI protocol. Trained model: [sreenathmmenon/asha-sahayak-grpo](https://huggingface.co/sreenathmmenon/asha-sahayak-grpo). Live demo: [ASHA Sahayak on Hugging Face Spaces](https://sreenathmmenon-asha-sahayak.hf.space).*

---

*A note on Champa Devi: she is a composite figure drawn from the documented accounts of real ASHA workers — Archana Tamdalge, Kenjir Perme, Rohini Pawar, Rathnamma P., and others whose testimonies appear in the sources below. Her story is true. Her name is shared.*

**Sources:**
- WHO Global Health Leaders Award 2022: nhsrcindia.org
- Kerala ASHA Strike: theindiaforum.in, globalvoices.org
- Maternal mortality and institutional delivery data: National Health Mission, WHO India
- COVID-19 ASHA deaths: Time Magazine, Maharashtra government figures
- Melinda Gates quote: CNN News18, gatesfoundation.org
- ASHA daily life accounts: Gavi, The Wire, New Lines Magazine
- Rural doctor shortfall: NHM Rural Health Statistics 2022
- Polio eradication: WHO India
- ASHA incentive structure: NHSRC April 2024 schedule
