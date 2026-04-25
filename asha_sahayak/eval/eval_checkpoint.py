#!/usr/bin/env python3
"""
eval_checkpoint.py — Held-out evaluation of the trained ASHA Sahayak checkpoint.

Runs the trained model (sreenathmmenon/asha-sahayak-grpo) on seeds 1000-1099
(held-out, never seen during training). Reports mean reward, per-component
breakdown, and dangerous undertriage rate.

Usage:
    python eval/eval_checkpoint.py
    python eval/eval_checkpoint.py --seeds 1000-1099 --output assets/heldout_evaluation.json

Requirements:
    pip install transformers torch accelerate  (or use Colab/HF Spaces GPU)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from asha_sahayak.server.asha_environment import AshaEnvironment
from asha_sahayak.server.corpus.cases import ALL_CASES, CASES_BY_DIFFICULTY
from asha_sahayak.models import AshaAction

CHECKPOINT = "sreenathmmenon/asha-sahayak-grpo"
DEFAULT_OUTPUT = ROOT / "assets" / "heldout_evaluation.json"

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(f"Loading checkpoint: {checkpoint}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model.eval()
        print(f"Model loaded on: {next(model.parameters()).device}", flush=True)
        return model, tokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers torch accelerate")
        sys.exit(1)


def generate_action(model, tokenizer, system_prompt: str, obs_text: str) -> AshaAction:
    """Generate a structured JSON action from the model."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": obs_text},
    ]

    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        input_text = f"System: {system_prompt}\n\nUser: {obs_text}\n\nAssistant:"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    return _parse_action(text)


def _parse_action(text: str) -> AshaAction:
    """Extract JSON action from model output, with fallback."""
    # Try to find a JSON block
    json_match = re.search(r'\{[^{}]*"referral_decision"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return AshaAction(
                referral_decision=data.get("referral_decision", "TREAT_AT_HOME"),
                urgency=data.get("urgency", "routine"),
                primary_concern=data.get("primary_concern", "general"),
                action_items=data.get("action_items", []),
                question=data.get("question"),
                confidence=float(data.get("confidence", 0.5)),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Fallback: detect keywords in raw output
    text_lower = text.lower()
    referral = "TREAT_AT_HOME"
    if "refer_immediately" in text_lower or "refer immediately" in text_lower:
        referral = "REFER_IMMEDIATELY"
    elif "refer_within" in text_lower or "within 24" in text_lower:
        referral = "REFER_WITHIN_24H"
    elif "monitor" in text_lower:
        referral = "MONITOR"

    return AshaAction(
        referral_decision=referral,
        urgency="routine",
        primary_concern="general",
        action_items=[],
        question=None,
        confidence=0.5,
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an AI assistant helping an ASHA (Accredited Social Health Activist) worker in rural India. "
    "Given the patient presentation and conversation, make a clinical assessment. "
    "Output ONLY a JSON object with these fields:\n"
    '{"referral_decision": "REFER_IMMEDIATELY|REFER_WITHIN_24H|TREAT_AT_HOME|MONITOR|PENDING", '
    '"urgency": "immediate|within_24h|routine|monitor|unknown", '
    '"primary_concern": "<clinical_concern>", '
    '"action_items": [], '
    '"question": null_or_string, '
    '"confidence": 0.0_to_1.0}'
)


def run_evaluation(model, tokenizer, seed_start: int, seed_end: int) -> dict:
    all_case_ids = list(ALL_CASES.keys())
    difficulties = ["easy", "medium", "hard"]

    results = []
    total = 0
    dangerous_undertriage = 0

    component_sums = defaultdict(float)
    difficulty_rewards = defaultdict(list)

    print(f"\nRunning evaluation: seeds {seed_start}–{seed_end}", flush=True)
    print(f"Total episodes: {seed_end - seed_start}", flush=True)
    print("-" * 50, flush=True)

    for seed in range(seed_start, seed_end):
        # Cycle through all difficulties for coverage
        difficulty = difficulties[seed % len(difficulties)]

        env = AshaEnvironment()
        obs = env.reset(task_id=difficulty, seed=seed)
        case = env._case

        obs_text = (
            f"Patient: {obs.patient_context.age_description}, {obs.patient_context.gender}\n"
            f"Location: {obs.patient_context.location}\n"
            f"Season: {obs.patient_context.season}\n"
            f"Malaria risk area: {obs.patient_context.malaria_risk_area}\n\n"
            f"Presentation: {obs.conversation[0].text if obs.conversation else 'No presentation'}\n\n"
            "Make your clinical assessment and output JSON:"
        )

        action = generate_action(model, tokenizer, SYSTEM_PROMPT, obs_text)
        result_obs = env.step(action)

        rc = result_obs.reward_components or {}
        episode_reward = result_obs.reward
        total += 1

        # Track dangerous undertriage
        if (case.correct_referral == "REFER_IMMEDIATELY"
                and action.referral_decision in ("TREAT_AT_HOME", "MONITOR")):
            dangerous_undertriage += 1

        for comp in ("referral", "urgency", "primary_concern", "information_gathering"):
            component_sums[comp] += rc.get(comp, 0.0)

        difficulty_rewards[difficulty].append(episode_reward)

        results.append({
            "seed": seed,
            "difficulty": difficulty,
            "case_id": case.case_id,
            "correct_referral": case.correct_referral,
            "predicted_referral": action.referral_decision,
            "reward": round(episode_reward, 4),
            "components": {k: round(v, 4) for k, v in rc.items() if k != "weights"},
        })

        if (seed - seed_start + 1) % 10 == 0:
            running_mean = sum(r["reward"] for r in results) / len(results)
            print(f"  Seeds {seed_start}–{seed}: {len(results)} episodes, mean reward={running_mean:.3f}", flush=True)

    mean_reward = sum(r["reward"] for r in results) / total if total > 0 else 0.0

    return {
        "eval_date": datetime.now(tz=timezone.utc).isoformat(),
        "checkpoint": CHECKPOINT,
        "seeds": f"{seed_start}-{seed_end - 1}",
        "n_episodes": total,
        "mean_reward": round(mean_reward, 4),
        "components": {
            comp: round(component_sums[comp] / total, 4)
            for comp in ("referral", "urgency", "primary_concern", "information_gathering")
        },
        "dangerous_undertriage_rate": round(dangerous_undertriage / total, 4) if total > 0 else 0.0,
        "per_difficulty": {
            diff: round(sum(rewards) / len(rewards), 4) if rewards else 0.0
            for diff, rewards in difficulty_rewards.items()
        },
        "episodes": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Held-out evaluation of ASHA Sahayak checkpoint.")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--seeds", default="1000-1099",
                        help="Seed range as 'start-end' (inclusive)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    start_str, end_str = args.seeds.split("-")
    seed_start = int(start_str)
    seed_end = int(end_str) + 1  # inclusive → exclusive

    model, tokenizer = load_model(args.checkpoint)
    summary = run_evaluation(model, tokenizer, seed_start, seed_end)

    # Save full results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print("HELD-OUT EVALUATION RESULTS")
    print("=" * 50)
    print(f"Checkpoint:              {summary['checkpoint']}")
    print(f"Seeds:                   {summary['seeds']}")
    print(f"Episodes:                {summary['n_episodes']}")
    print(f"Mean reward:             {summary['mean_reward']:.4f}")
    print(f"Dangerous undertriage:   {summary['dangerous_undertriage_rate']:.1%}")
    print("\nPer-component breakdown:")
    for comp, val in summary["components"].items():
        print(f"  {comp:<25} {val:.4f}")
    print("\nPer-difficulty:")
    for diff, val in summary["per_difficulty"].items():
        print(f"  {diff:<10} {val:.4f}")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
