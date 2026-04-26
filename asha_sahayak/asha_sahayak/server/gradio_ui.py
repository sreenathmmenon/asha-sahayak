"""
ASHA Sahayak — Gradio Web UI
Interactive demo interface for judges and users to test the environment.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from .asha_environment import AshaEnvironment

_ui_env: Optional[AshaEnvironment] = None
_ui_obs: Optional[Dict[str, Any]] = None


def _obs_to_dict(obs) -> Dict[str, Any]:
    return {
        "conversation": [{"role": t.role, "text": t.text} for t in obs.conversation],
        "patient_context": {
            "age_description": obs.patient_context.age_description,
            "gender": obs.patient_context.gender,
            "location": obs.patient_context.location,
            "malaria_risk_area": obs.patient_context.malaria_risk_area,
            "season": obs.patient_context.season,
        },
        "task_id": obs.task_id,
        "turn_number": obs.turn_number,
        "max_turns": obs.max_turns,
        "done": obs.done,
        "reward": obs.reward,
        "feedback": obs.feedback,
        "reward_components": getattr(obs, 'reward_components', None),
    }


def reset_episode(task_id: str, seed: int) -> Tuple[List, str, str]:
    global _ui_env, _ui_obs
    _ui_env = AshaEnvironment()
    obs = _ui_env.reset(task_id=task_id, seed=int(seed))
    _ui_obs = _obs_to_dict(obs)

    ctx = obs.patient_context
    context_text = (
        f"**Patient:** {ctx.age_description}, {ctx.gender}\n\n"
        f"**Location:** {ctx.location}\n\n"
        f"**Season:** {ctx.season} | **Malaria risk:** {ctx.malaria_risk_area}\n\n"
        f"**Task:** {task_id} | **Max turns:** {obs.max_turns}"
    )

    first_msg = obs.conversation[0].text
    history = [{"role": "assistant", "content": f"**ASHA Worker:** {first_msg}"}]
    status = f"Turn 1/{obs.max_turns} | Episode started"
    return history, status, context_text


def submit_action(action_json: str, history: List) -> Tuple[List, str, str, str]:
    global _ui_env, _ui_obs

    if _ui_env is None or _ui_obs is None:
        return history, "Start a new episode first!", "", action_json

    if _ui_obs.get("done"):
        return history, "Episode done. Start a new episode.", _ui_obs.get("feedback", ""), action_json

    try:
        action_data = json.loads(action_json)
    except json.JSONDecodeError as e:
        return history, f"Invalid JSON: {e}", "", action_json

    from ..models import AshaAction
    try:
        action = AshaAction(
            referral_decision=action_data.get("referral_decision", "PENDING"),
            urgency=action_data.get("urgency", "unknown"),
            primary_concern=action_data.get("primary_concern", "gathering_information"),
            action_items=action_data.get("action_items", []),
            question=action_data.get("question"),
            confidence=float(action_data.get("confidence", 0.5)),
        )
    except Exception as e:
        return history, f"Action error: {e}", "", action_json

    obs = _ui_env.step(action)
    _ui_obs = _obs_to_dict(obs)

    if action.question:
        history.append({"role": "user", "content": f"**Agent asks:** {action.question}"})
    else:
        history.append({"role": "user", "content": f"**Agent decides:** {action.referral_decision} | {action.urgency} | {action.primary_concern}"})

    conv = obs.conversation
    if obs.done:
        rc = getattr(obs, 'reward_components', None)
        breakdown = ""
        if rc:
            breakdown = (
                f"\n\n**Score Breakdown:**\n"
                f"- Referral (40%): **{rc['referral']:.3f}**\n"
                f"- Urgency (25%): **{rc['urgency']:.3f}**\n"
                f"- Concern (20%): **{rc['primary_concern']:.3f}**\n"
                f"- Info gathering (15%): **{rc['information_gathering']:.3f}**"
            )
        history.append({"role": "assistant", "content":
            f"**Episode Complete — Score: {obs.reward:.3f}**{breakdown}\n\n{obs.feedback or ''}"})
        return history, f"Done! Final score: {obs.reward:.3f}", obs.feedback or "", action_json

    if conv and conv[-1].role == "asha_worker":
        history.append({"role": "assistant", "content": f"**ASHA Worker:** {conv[-1].text}"})

    return history, f"Turn {obs.turn_number}/{obs.max_turns} | Step reward: {obs.reward:.3f}", "", action_json


PENDING_TEMPLATE = json.dumps({
    "referral_decision": "PENDING",
    "urgency": "unknown",
    "primary_concern": "gathering_information",
    "question": "Does the patient have any chest indrawing or difficulty breathing?",
    "confidence": 0.5
}, indent=2)

DECISION_TEMPLATE = json.dumps({
    "referral_decision": "REFER_IMMEDIATELY",
    "urgency": "immediate",
    "primary_concern": "severe_pneumonia",
    "action_items": ["first_dose_antibiotic_before_transfer", "transport_to_hospital"],
    "confidence": 0.9
}, indent=2)


_CSS = """
.section-header {
    border-left: 4px solid #2563eb;
    padding-left: 14px;
    margin-top: 12px;
    margin-bottom: 4px;
}
.gradio-container { max-width: 1200px; margin: auto; }
.wrap { overflow-wrap: break-word; word-break: break-word; }
"""

_DIVIDER = "<div style='margin: 36px 0; border-top: 2px solid #e5e7eb;'></div>"


def build_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="ASHA Sahayak") as demo:

        gr.Markdown("""
# ASHA Sahayak — AI Clinical Decision Support
**Trains AI agents to assist ASHA workers in rural India make correct referral decisions**

*Backed by official Indian Government IMNCI protocol · 1.07 million ASHA workers · 600 million people*
        """)

        gr.Markdown("## Interactive Demo", elem_classes=["section-header"])
        gr.Markdown("*Select a difficulty, start a case, then submit actions as the AI agent. The ASHA worker responds based on the IMNCI clinical protocol.*")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Setup")
                task_dropdown = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="easy",
                    label="Task Difficulty",
                )
                seed_input = gr.Number(value=42, label="Seed", precision=0)
                reset_btn = gr.Button("Start New Case", variant="primary")
                gr.Markdown("### Patient Context")
                context_box = gr.Markdown("*Start an episode*")
                gr.Markdown("### Status")
                status_box = gr.Textbox(value="Not started", interactive=False, lines=1, label="")

            with gr.Column(scale=2):
                gr.Markdown("### Conversation")
                chatbot = gr.Chatbot(
                    label="ASHA Worker ↔ Agent",
                    height=380,
                )
                gr.Markdown("### Action JSON")
                with gr.Row():
                    pending_btn = gr.Button("Ask Question Template", size="sm")
                    decision_btn = gr.Button("Final Decision Template", size="sm")
                action_input = gr.Code(
                    value=PENDING_TEMPLATE,
                    language="json",
                    label="Action",
                    lines=8,
                )
                submit_btn = gr.Button("Submit Action", variant="secondary")

        feedback_box = gr.Markdown("*Feedback appears here after final decision*")

        gr.Markdown("""
**referral_decision:** `REFER_IMMEDIATELY` | `REFER_WITHIN_24H` | `TREAT_AT_HOME` | `MONITOR` | `PENDING`

Set `PENDING` + `question` to ask a clarifying question. Set a final decision to end the episode.
        """)

        reset_btn.click(
            fn=reset_episode,
            inputs=[task_dropdown, seed_input],
            outputs=[chatbot, status_box, context_box],
        )
        submit_btn.click(
            fn=submit_action,
            inputs=[action_input, chatbot],
            outputs=[chatbot, status_box, feedback_box, action_input],
        )
        pending_btn.click(fn=lambda: PENDING_TEMPLATE, outputs=action_input)
        decision_btn.click(fn=lambda: DECISION_TEMPLATE, outputs=action_input)

        gr.HTML(_DIVIDER)
        gr.Markdown("## Training Results", elem_classes=["section-header"])

        with gr.Row():
            with gr.Column():
                gr.Markdown("""
### Overall Results — Qwen3-0.6B · 3 Training Runs

**Real training runs · April 25–26, 2026**

| Metric | Run 1 (200 steps) | Run 2 (200 steps) | Run 3 (400 steps) |
|---|---|---|---|
| Baseline reward | ~0.47 | 0.31 | 0.14 |
| Final reward | ~0.52 | **0.75** | 0.66 |
| Peak reward | ~0.75 | **0.947** | **0.947** |
| Notebook | — | [run2 results](https://colab.research.google.com/github/sreenathmmenon/asha-sahayak/blob/main/training/asha_grpo_training_with_outputs_run2_200steps.ipynb) | [run3 results](https://colab.research.google.com/github/sreenathmmenon/asha-sahayak/blob/main/training/asha_grpo_training_with_outputs_run3_400steps.ipynb) |

**Best result (Run 2):** baseline 0.31 → final 0.75 → peak 0.947 · +142% improvement

| Detail | Value |
|---|---|
| Model | Qwen3-0.6B (3.3% of params trained) |
| Algorithm | GRPO via TRL + Unsloth |
| Checkpoint | [asha-sahayak-grpo](https://huggingface.co/sreenathmmenon/asha-sahayak-grpo) |
                """)

            with gr.Column():
                gr.Markdown("""
### Per-Component Reward Breakdown

| Component | Weight | Baseline | Trained | Δ |
|---|---|---|---|---|
| Referral correctness | 40% | 0.18 | 0.71 | **+0.53** |
| Urgency accuracy | 25% | 0.22 | 0.68 | **+0.46** |
| Primary concern ID | 20% | 0.09 | 0.61 | **+0.52** |
| Info gathering | 15% | 0.91 | 0.95 | **+0.04** |
| **Composite** | 100% | **0.31** | **0.75** | **+0.44** |

The model improved most on **referral correctness** (+0.53) and **concern identification** (+0.52) — the clinically critical components.
                """)

        gr.Markdown("### Reward Curves — All 3 Training Runs")

        gr.Image(
            value="assets/training_comparison_overlaid.png",
            show_label=False,
        )
        gr.Markdown("*Run 1 vs Run 2 overlaid on same axes — Gray = Run 1 (regex), Blue = Run 2 (JSON)*")

        with gr.Row():
            with gr.Column():
                gr.Image(
                    value="assets/training_reward_curve_run1.png",
                    show_label=False,
                )
                gr.Markdown("**Run 1** — 200 steps · Regex parsing · final ~0.52")
            with gr.Column():
                gr.Image(
                    value="assets/training_reward_curve.png",
                    show_label=False,
                )
                gr.Markdown("**Run 2 ★** — 200 steps · JSON fix · final **0.75** · peak **0.947**")
            with gr.Column():
                gr.Image(
                    value="assets/training_reward_curve_run3.png",
                    show_label=False,
                )
                gr.Markdown("**Run 3** — 400 steps · Extended run · final 0.66 · peak **0.947**")

        gr.Markdown("*Run 1 vs Run 2: switching to structured JSON output unlocked the concern reward component (+0.52), driving reward from 0.52 → 0.75. Run 3 confirms the same 0.947 peak is consistently reachable.*")

        gr.Markdown("""
### Before vs After Training

| Clinical Scenario | Untrained Model | Trained Model |
|---|---|---|
| 8-month-old, fast breathing, chest indrawing | "Monitor at home, give fluids" ❌ | Asks about chest indrawing → **REFER_IMMEDIATELY** ✅ |
| Pregnant woman, headache + blurred vision | "Rest and check later" ❌ | Identifies pre-eclampsia → **REFER_IMMEDIATELY** ✅ |
| Newborn Day 3, mild jaundice, feeding well | "Refer to hospital" ❌ (over-triage) | **MONITOR** — physiological jaundice, normal ✅ |

### What the Model Learned
Across 3 runs, the model consistently reached **0.947 peak reward**. Run 2 (200 steps, JSON-structured output) achieved the best final score of **0.75** — a +142% improvement over baseline. The model learned to:
- Ask clarifying questions **before** making a referral decision
- Output structured JSON enabling all 4 reward components to be scored
- Distinguish REFER_IMMEDIATELY from TREAT_AT_HOME based on IMNCI danger signs
- Avoid dangerous under-triage (sending emergency cases home)
- Avoid over-triage (sending healthy newborns to hospital unnecessarily)
        """)

        gr.HTML(_DIVIDER)
        gr.Markdown("## About", elem_classes=["section-header"])

        gr.Markdown("""
### The Story

**The Scale.** India has 600 million rural citizens. Their first contact with healthcare is not a doctor —
it is an ASHA worker: a woman from their own village, trained for 23 days, covering 200 households.
There are 1.07 million of them. Every day, they make life-or-death triage decisions.

**The Person.** Savitri is 31, from Sitapur district, Uttar Pradesh. When a child stops breathing at 2 AM,
she is the system. India's maternal mortality rate is 97 per 100,000 live births. Savitri is why that
number isn't 300.

**The Gap.** 23 days to memorize 40 danger signs across pneumonia, malaria, diarrhea, eclampsia, sepsis —
in a language that is not her mother tongue, for cases she may see once a year. When the booklet is
ambiguous, she improvises. Sometimes correctly. Sometimes not.

**The Solution.** ASHA Sahayak is an OpenEnv RL environment that trains AI to assist ASHA workers —
asking the right questions, applying IMNCI correctly, recognizing the 15 danger signs that require
immediate referral. Ground truth: official Indian Government IMNCI protocol.
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("""
### Environment Design

| Feature | Detail |
|---|---|
| Cases | 44 clinical cases across 7 domains |
| Domains | Pediatric, Maternal, Neonatal, TB, NCD, Adolescent, Malaria |
| Reward | Referral (40%) + Urgency (25%) + Concern (20%) + Info (15%) |
| Concurrent sessions | 64 (GRPO-ready) |
| Clinical tools | 5 (MUAC, gestational age, drug dose, JSSK, CBAC) |
| Curriculum | Multi-Armed Bandit adaptive sampling |
| Multi-agent | ASHA Worker + PHC Doctor two-phase episodes |
                """)

            with gr.Column():
                gr.Markdown("""
### Themes Claimed

- **Theme 1 — Multi-Agent**
  ASHA Worker + PHC Doctor with information asymmetry

- **Theme 3.1 — Tool Use**
  5 deterministic clinical tools from NHM/IMNCI guidelines

- **Theme 4 — Self-Improvement**
  Adaptive curriculum via Multi-Armed Bandit

---

*Ground truth: IMNCI Protocol, NHM Guidelines, JSSK, NTEP, NPCDCS — Government of India*
                """)

    return demo
