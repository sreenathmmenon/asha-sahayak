"""
ASHA Sahayak — Gradio Web UI
Interactive demo interface for judges and users to test the environment.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from .asha_environment import AshaEnvironment, MAX_TURNS

# Module-level environment instance for the UI (separate from API instance)
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
    }


def reset_episode(task_id: str, seed: int) -> Tuple[List, str, str]:
    """Reset environment and return (chat_history, status_text, context_text)."""
    global _ui_env, _ui_obs
    _ui_env = AshaEnvironment()
    obs = _ui_env.reset(task_id=task_id, seed=seed)
    _ui_obs = _obs_to_dict(obs)

    ctx = obs.patient_context
    context_text = (
        f"**Patient:** {ctx.age_description}, {ctx.gender}\n"
        f"**Location:** {ctx.location}\n"
        f"**Season:** {ctx.season}\n"
        f"**Malaria risk area:** {ctx.malaria_risk_area}\n"
        f"**Task:** {task_id} | **Max turns:** {obs.max_turns}"
    )

    # First message from ASHA worker
    chat_history = [{"role": "assistant", "content": f"🏥 **ASHA Worker:** {obs.conversation[0].text}"}]
    status = f"Turn 1/{obs.max_turns} | Reward: 0.00 | Episode active"
    return chat_history, status, context_text


def submit_action(action_json: str, chat_history: List) -> Tuple[List, str, str, str]:
    """Submit an action and return (chat_history, status, feedback, action_template)."""
    global _ui_env, _ui_obs

    if _ui_env is None or _ui_obs is None:
        return chat_history, "⚠️ Start a new episode first!", "", action_json

    if _ui_obs.get("done"):
        return chat_history, "✅ Episode done. Start a new episode.", _ui_obs.get("feedback", ""), action_json

    try:
        action_data = json.loads(action_json)
    except json.JSONDecodeError as e:
        return chat_history, f"❌ Invalid JSON: {e}", "", action_json

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
        return chat_history, f"❌ Action error: {e}", "", action_json

    obs = _ui_env.step(action)
    _ui_obs = _obs_to_dict(obs)

    # Add agent action to chat
    if action.question:
        chat_history.append({"role": "user", "content": f"🤖 **Agent asks:** {action.question}"})
    else:
        chat_history.append({
            "role": "user",
            "content": (
                f"🤖 **Agent decides:** {action.referral_decision}\n"
                f"Urgency: {action.urgency} | Concern: {action.primary_concern}"
            )
        })

    # Add ASHA worker response or feedback
    conv = obs.conversation
    if len(conv) > 0 and conv[-1].role == "asha_worker":
        chat_history.append({"role": "assistant", "content": f"👩 **ASHA Worker:** {conv[-1].text}"})

    if obs.done:
        feedback_md = obs.feedback or ""
        chat_history.append({
            "role": "assistant",
            "content": f"📋 **Episode Complete!**\n\n**Final Score: {obs.reward:.3f}**\n\n```\n{feedback_md}\n```"
        })
        status = f"✅ Done! Final score: {obs.reward:.3f}"
        return chat_history, status, feedback_md, action_json

    status = f"Turn {obs.turn_number}/{obs.max_turns} | Step reward: {obs.reward:.3f}"
    return chat_history, status, "", action_json


PENDING_TEMPLATE = json.dumps({
    "referral_decision": "PENDING",
    "urgency": "unknown",
    "primary_concern": "gathering_information",
    "question": "Does the patient have any danger signs like chest indrawing or convulsions?",
    "confidence": 0.5
}, indent=2)

DECISION_TEMPLATE = json.dumps({
    "referral_decision": "REFER_IMMEDIATELY",
    "urgency": "immediate",
    "primary_concern": "severe_pneumonia",
    "action_items": ["first_dose_antibiotic_before_transfer", "transport_to_hospital"],
    "confidence": 0.9
}, indent=2)


def build_gradio_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    with gr.Blocks(title="ASHA Sahayak — Clinical Decision Support") as demo:
        gr.Markdown("""
# 🏥 ASHA Sahayak — AI Clinical Decision Support
**Training AI agents to help ASHA workers in rural India make correct referral decisions**

*Backed by official Indian Government IMNCI protocol · 1.07 million ASHA workers · 600 million people served*
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Episode Setup")
                task_dropdown = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="easy",
                    label="Task Difficulty",
                    info="easy=clear signs, medium=multi-symptom, hard=complex overlapping"
                )
                seed_input = gr.Number(value=42, label="Random Seed", precision=0)
                reset_btn = gr.Button("▶ Start New Case", variant="primary", size="lg")

                gr.Markdown("### Patient Context")
                context_box = gr.Markdown("*Start an episode to see patient context*")

                gr.Markdown("### Episode Status")
                status_box = gr.Textbox(label="", value="Not started", interactive=False, lines=1)

            with gr.Column(scale=2):
                gr.Markdown("### Conversation")
                chatbot = gr.Chatbot(
                    label="ASHA Worker ↔ Agent",
                    height=400,
                    show_label=True,
                )

                gr.Markdown("### Agent Action (JSON)")
                gr.Markdown(
                    "Set `referral_decision: PENDING` + `question` to ask a clarifying question. "
                    "Set a final decision to end the episode."
                )

                with gr.Row():
                    pending_btn = gr.Button("Load: Ask Question", size="sm")
                    decision_btn = gr.Button("Load: Final Decision", size="sm")

                action_input = gr.Code(
                    value=PENDING_TEMPLATE,
                    language="json",
                    label="Action JSON",
                    lines=10,
                )
                submit_btn = gr.Button("Submit Action →", variant="secondary", size="lg")

        with gr.Row():
            feedback_box = gr.Markdown("*Feedback will appear here after the final decision*")

        gr.Markdown("""
---
### How to use
1. Select difficulty and click **Start New Case** — the ASHA worker presents a patient
2. Read the presentation, then submit a JSON action asking a clarifying question (PENDING)
3. Keep asking questions until you have enough information
4. Submit a final decision (REFER_IMMEDIATELY / REFER_WITHIN_24H / TREAT_AT_HOME / MONITOR)
5. See your score and the clinical explanation

### Action reference
| Field | Values |
|-------|--------|
| referral_decision | REFER_IMMEDIATELY, REFER_WITHIN_24H, TREAT_AT_HOME, MONITOR, PENDING |
| urgency | immediate, within_24h, routine, monitor, unknown |
| primary_concern | snake_case clinical identifier |
| action_items | list of structured tags (no drug names) |
| question | clarifying question (only when PENDING) |
| confidence | 0.0–1.0 |
        """)

        # Wire up events
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

    return demo
