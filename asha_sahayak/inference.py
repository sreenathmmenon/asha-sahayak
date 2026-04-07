"""
ASHA Sahayak — Inference Script
===================================
Runs an LLM agent against all 3 task levels (easy, medium, hard).
Emits mandatory [START] / [STEP] / [END] log lines.

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   LLM API endpoint (default: HuggingFace router)
    MODEL_NAME     Model identifier
    HF_TOKEN       HuggingFace / API key
    ENV_BASE_URL   Environment server URL (default: http://localhost:7860)

STDOUT FORMAT (strict — do not change):
    [START] task=<task_id> env=asha_sahayak model=<model>
    [STEP]  step=<n> action=<summary> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from asha_sahayak.client import AshaClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS   = 6     # max steps per episode (matches hard task max_turns)
TEMPERATURE = 0.1
MAX_TOKENS  = 400
SUCCESS_THRESHOLD = 0.6  # score >= this = success

TASKS = [
    {"task_id": "easy",   "seed": 42,  "label": "Standard English — Clear Danger Signs"},
    {"task_id": "medium", "seed": 123, "label": "Multi-symptom — Requires Clarification"},
    {"task_id": "hard",   "seed": 500, "label": "Complex Case — Cross-cutting Signs"},
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI assistant supporting an ASHA (Accredited Social Health Activist) worker
in rural India. ASHA workers are frontline community health workers with limited
diagnostic equipment and no on-site doctor.

Your job is to help them make the correct clinical decision for their patient:
  REFER_IMMEDIATELY  → Emergency — transport to hospital now
  REFER_WITHIN_24H   → Urgent — go to PHC/facility today
  TREAT_AT_HOME      → Can be managed at home with medicines/ORS/counselling
  MONITOR            → Watch and wait, return if worsens

You must gather information before deciding. Ask ONE focused clarifying question
per turn until you have enough information.

KEY DANGER SIGNS to always ask about (based on IMNCI protocol):
  CHILD:    chest indrawing, fast breathing (count rate), lethargy/unconscious,
            unable to drink/breastfeed, convulsions, stridor, grunting
  NEONATE:  hypothermia (<35°C), umbilical redness/pus (omphalitis), jaundice,
            not feeding, bulging fontanelle
  MATERNAL: convulsions/fits in pregnancy (eclampsia), severe headache + blurred vision,
            heavy vaginal bleeding, BP > 140/90
  NUTRITION: MUAC < 11.5cm, bilateral pitting oedema (press feet for 3 sec),
             visible severe wasting, failed appetite test (RUTF refusal)
  FEVER:    stiff neck, bulging fontanelle, rash → meningitis signs

When you are ready to give your final decision, respond with ONLY a JSON object:
{
  "referral_decision": "REFER_IMMEDIATELY",
  "urgency": "immediate",
  "primary_concern": "severe_pneumonia",
  "action_items": ["first_dose_antibiotic_before_transfer", "transport_to_hospital"],
  "confidence": 0.95
}

When asking a clarifying question (not yet deciding), respond with ONLY:
{
  "referral_decision": "PENDING",
  "urgency": "unknown",
  "primary_concern": "gathering_information",
  "question": "Does the child have any chest indrawing — can you see the lower chest going in when breathing?",
  "confidence": 0.5
}

IMNCI PRIMARY CONCERN IDENTIFIERS (use these exact terms):
  severe_pneumonia              → chest indrawing present
  pneumonia_no_severe_signs     → fast breathing, NO chest indrawing
  very_severe_febrile_disease_possible_meningitis → fever + stiff neck / bulging fontanelle / petechial rash
  eclampsia                     → seizure in pregnancy
  pre_eclampsia_severe_features → severe headache + blurred vision + high BP (no seizure yet)
  postpartum_hemorrhage         → heavy bleeding after delivery
  antepartum_haemorrhage_shock  → bleeding in pregnancy + shock signs
  neonatal_sepsis               → fever/lethargy in neonate <28 days
  neonatal_hypothermia_with_sepsis_risk → cold baby <35°C + danger signs
  omphalitis_with_systemic_spread → cord pus + spreading redness + systemic signs
  severe_complicated_sam        → MUAC <11.5 + bilateral edema OR complications
  moderate_acute_malnutrition   → MUAC 11.5-12.5, no edema, passes appetite test
  severe_dehydration_with_lethargy → lethargy + sunken eyes + cannot drink
  some_dehydration              → restless + sunken eyes + drinks eagerly
  diarrhea_no_dehydration       → diarrhea, alert, drinking normally
  presumptive_tuberculosis      → cough ≥2 weeks + weight loss / hemoptysis

IMPORTANT:
- action_items must be structured tags (e.g. "transport_to_phc") — NO drug names or dosages
- Ask at least 1 clarifying question before giving a final decision
- primary_concern MUST be one of the IMNCI identifiers listed above — use exact snake_case
- For thin/not-eating children ALWAYS ask about MUAC measurement and bilateral foot oedema
- For fever cases ALWAYS ask about stiff neck and consciousness level
- Keep questions focused on key danger signs
""").strip()


def build_user_prompt(obs: Dict[str, Any]) -> str:
    conv = obs["conversation"]
    ctx  = obs["patient_context"]

    lines = [
        f"PATIENT: {ctx['age_description']}, {ctx['gender']}",
        f"Location: {ctx['location']} | Season: {ctx['season']} | Malaria risk: {ctx['malaria_risk_area']}",
        f"Turn: {obs['turn_number']} of {obs['max_turns']}",
        f"",
        f"CONVERSATION:",
    ]
    for turn in conv:
        prefix = "ASHA Worker" if turn["role"] == "asha_worker" else "You (Agent)"
        lines.append(f"  {prefix}: {turn['text']}")

    if obs["turn_number"] >= obs["max_turns"] - 1:
        lines.append("")
        lines.append("NOTE: This is your final turn. You MUST give a final decision now.")

    lines.append("")
    lines.append("Respond with JSON only.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Logging helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task_id: str, model: str) -> None:
    print(f"[START] task={task_id} env=asha_sahayak model={model}", flush=True)


def log_step(
    step: int,
    action_summary: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err_val  = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP]  step={step} action={action_summary} reward={reward:.2f}"
        f" done={done_val} error={err_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps}"
        f" score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(client: AshaClient, llm: OpenAI, task: Dict[str, Any]) -> Dict[str, Any]:
    task_id = task["task_id"]
    seed    = task["seed"]

    log_start(task_id, MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success     = False
    error_msg: Optional[str] = None

    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        obs = client.reset(task_id=task_id, seed=seed)

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            # Build prompt and call LLM
            user_msg = build_user_prompt(obs)
            conversation_history.append({"role": "user", "content": user_msg})

            try:
                completion = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=conversation_history,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                raw = (completion.choices[0].message.content or "").strip()
                conversation_history.append({"role": "assistant", "content": raw})
            except Exception as e:
                raw = ""
                error_msg = f"LLM error: {e}"

            # Parse JSON action
            try:
                # Extract JSON from response (handle markdown code blocks)
                json_str = raw
                if "```" in raw:
                    start = raw.find("{")
                    end   = raw.rfind("}") + 1
                    json_str = raw[start:end] if start != -1 else "{}"
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                parsed = {
                    "referral_decision": "PENDING",
                    "urgency":           "unknown",
                    "primary_concern":   "parse_error",
                    "question":          "Can you describe the main symptoms in more detail?",
                    "confidence":        0.5,
                }
                error_msg = f"JSON parse error on: {raw[:80]}"

            # Force final decision on last allowed step
            if step >= MAX_STEPS and parsed.get("referral_decision", "PENDING") == "PENDING":
                parsed["referral_decision"] = "MONITOR"
                parsed["urgency"] = "monitor"
                parsed.pop("question", None)
                error_msg = "forced_decision_max_steps"

            action_summary = (
                parsed.get("question", "")[:60]
                if parsed.get("referral_decision", "PENDING") == "PENDING"
                else f"{parsed.get('referral_decision')}|{parsed.get('primary_concern','')}"
            )

            # Call environment
            obs = client.step(
                referral_decision=parsed.get("referral_decision", "PENDING"),
                urgency=parsed.get("urgency", "monitor"),
                primary_concern=parsed.get("primary_concern", ""),
                action_items=parsed.get("action_items", []),
                question=parsed.get("question"),
                confidence=float(parsed.get("confidence", 0.5)),
            )

            reward = float(obs.get("reward", 0.0))
            done   = obs.get("done", False)
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action_summary=action_summary,
                reward=reward,
                done=done,
                error=error_msg,
            )
            error_msg = None  # reset after logging

            if done:
                final_score = reward
                success = final_score >= SUCCESS_THRESHOLD
                break

    except Exception as e:
        error_msg = str(e)
        if not rewards:
            rewards = [0.0]
        final_score = rewards[-1] if rewards else 0.0
    finally:
        if not rewards:
            rewards = [0.0]
        final_score = rewards[-1] if rewards else 0.0
        success = final_score >= SUCCESS_THRESHOLD
        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards,
        )

    return {
        "task_id":     task_id,
        "steps":       steps_taken,
        "score":       final_score,
        "success":     success,
        "rewards":     rewards,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY environment variable is not set. Exiting.")
        return

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    with AshaClient(base_url=ENV_BASE_URL) as client:
        # Verify server is up
        try:
            health = client.health()
            assert health == "healthy", f"Unexpected health response: {health}"
        except Exception as e:
            print(f"[ERROR] Environment server not reachable at {ENV_BASE_URL}: {e}")
            return

        for task in TASKS:
            result = run_task(client, llm, task)
            results.append(result)
            print("", flush=True)  # blank line between tasks

    # Final summary
    overall_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    total_successes = sum(1 for r in results if r["success"])

    print(f"[SUMMARY] tasks={len(results)} successes={total_successes} overall_score={overall_score:.3f}", flush=True)


if __name__ == "__main__":
    main()
