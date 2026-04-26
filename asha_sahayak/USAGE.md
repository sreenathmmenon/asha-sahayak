# How to Use the ASHA Sahayak Demo

**You are the AI assistant helping an ASHA worker.**

Savitri is a frontline health worker in rural India. She is standing in front of a patient. She describes what she sees — in Hindi and English. You ask her questions, gather clinical information, and make the referral decision. She is waiting for your answer.

This is exactly what our trained AI model does — ask the right questions, apply the IMNCI protocol, and make the safe clinical decision. This demo lets you experience that role yourself, and see how the environment scores each decision against the official Indian Government IMNCI protocol.

---

## 3 Steps

**Step 1 — Start a case**
- Pick a difficulty (easy / medium / hard) and a seed number
- Click **Start New Case**
- The ASHA worker describes the patient in Hindi + English

**Step 2 — Ask questions**
- Click **"Ask Question Template"** to load the PENDING action
- Change the `question` field to what you want to ask
- Click **Submit Action** — the ASHA worker responds
- Repeat until you have enough information

**Step 3 — Make your decision**
- Click **"Final Decision Template"** to load the decision action
- Fill in `referral_decision`, `urgency`, `primary_concern`
- Click **Submit Action** — episode ends with score + clinical explanation

---

## JSON Field Reference

```json
{
  "referral_decision": "PENDING",         // PENDING while asking questions
  "urgency": "unknown",                   // unknown while asking questions
  "primary_concern": "gathering_information",
  "question": "Your question here",       // only needed when PENDING
  "confidence": 0.5
}
```

**referral_decision options:**
- `PENDING` — still gathering information
- `REFER_IMMEDIATELY` — emergency, refer now
- `REFER_WITHIN_24H` — non-emergency referral
- `TREAT_AT_HOME` — manage at home
- `MONITOR` — observe, follow up

**urgency options:** `immediate` | `within_24h` | `routine` | `monitor`

---

## Walkthrough 1 — Easy, Seed 42 (Pneumonia)

**Setup:** Difficulty `easy`, Seed `42`, click Start New Case

**ASHA Worker:**
> Bhaiya ka beta hai, 3 saal ka. 2 din se khansi hai, bukhar bhi hai.
> [The neighbour's son is 3 years old. Has had cough for 2 days, also fever.]

**Question asked:**
```json
{
  "referral_decision": "PENDING",
  "urgency": "unknown",
  "primary_concern": "gathering_information",
  "question": "Does the patient have any chest indrawing or difficulty breathing?",
  "confidence": 0.5
}
```

**ASHA Worker responds:**
> Breathing fast — I think faster than normal.

**Final decision made:**
```json
{
  "referral_decision": "REFER_IMMEDIATELY",
  "urgency": "immediate",
  "primary_concern": "severe_pneumonia",
  "action_items": ["first_dose_antibiotic_before_transfer", "transport_to_hospital"],
  "confidence": 0.9
}
```

**Score: 0.380** — Wrong decision. Here's why:

> IMNCI: Fast breathing (42/min) WITHOUT chest indrawing = PNEUMONIA. Treat at home with oral antibiotics. No chest indrawing = NOT severe pneumonia.

**What this teaches:** The IMNCI protocol distinguishes pneumonia (fast breathing only → treat at home) from severe pneumonia (chest indrawing → refer immediately). The correct answer was `TREAT_AT_HOME`.

---

## Walkthrough 2 — Medium, Seed 123 (Dehydration) ★ Score 0.95

**Setup:** Difficulty `medium`, Seed `123`, click Start New Case

**ASHA Worker:**
> 2 saal ka bacha. 2 din se dast hai. Bahut restless ho gaya hai.
> [2-year-old child. Has had diarrhea for 2 days. Became very restless.]

![Step 1 — Case started](assets/ui_demo_step1_case_started.png)

**Question 1:**
```json
{
  "referral_decision": "PENDING",
  "urgency": "unknown",
  "primary_concern": "gathering_information",
  "question": "Does the child have sunken eyes and is he able to drink?",
  "confidence": 0.5
}
```
**ASHA Worker:** Drinks eagerly — as soon as you offer water he grabs it.

![Step 2 — First question answered](assets/ui_demo_step2_question1.png)

**Question 2:**
```json
{
  "referral_decision": "PENDING",
  "urgency": "unknown",
  "primary_concern": "gathering_information",
  "question": "Are his eyes sunken and does his skin go back slowly when pinched?",
  "confidence": 0.5
}
```
**ASHA Worker:** Eyes look slightly sunken — not normal.

![Step 3 — Second question answered](assets/ui_demo_step3_question2.png)

**Final decision:**
```json
{
  "referral_decision": "REFER_WITHIN_24H",
  "urgency": "within_24h",
  "primary_concern": "dehydration",
  "action_items": ["oral_rehydration", "refer_to_phc"],
  "confidence": 0.8
}
```

**Score: 0.950** — Near perfect. Here's why:

![Step 4 — Score and breakdown](assets/ui_demo_step4_result.png)

![Step 5 — Clinical explanation](assets/ui_demo_step5_explanation.png)

> IMNCI: SOME DEHYDRATION — restlessness + sunken eyes + drinks eagerly (3 signs in 'some' category). No lethargy = NOT severe. Refer to PHC for supervised ORS.

**Score breakdown:**
| Component | Score | Note |
|---|---|---|
| Referral (40%) | 1.00 | Correct — REFER_WITHIN_24H ✅ |
| Urgency (25%) | 1.00 | Correct — within_24h ✅ |
| Concern (20%) | 0.50 | Partial — use `some_dehydration` not `dehydration` |
| Info gathering (15%) | 1.00 | Asked questions before deciding ✅ |

**What this teaches:** Restlessness + sunken eyes + drinks eagerly = some dehydration. Drinks eagerly rules out severe dehydration (which shows lethargy and inability to drink). Refer to PHC, not hospital.

---

## Tips

- **Always ask at least one question** before making a final decision — info gathering is 15% of your score
- **Change the question** each turn — asking the same question twice doesn't help
- **PENDING keeps the episode going** — only a final referral decision ends it
- **Read the clinical explanation** after each episode — it tells you exactly what IMNCI says about the case
- **Try the same seed again** after reading the explanation — you'll score much higher

---

## Real ASHA Worker Workflow

This is how Savitri, an ASHA worker in Sitapur, Uttar Pradesh, actually works:

1. A neighbour calls — child has been sick for 2 days
2. Savitri visits, observes, asks questions from her IMNCI checklist
3. She checks for danger signs: chest indrawing, inability to drink, lethargy, convulsions
4. She makes a decision: refer now, refer tomorrow, treat at home, or monitor
5. She documents and follows up

ASHA Sahayak trains an AI to do exactly this — ask the right questions in the right order, apply IMNCI correctly, and make the safe decision. The AI assists Savitri when the booklet is ambiguous or the case is rare.
