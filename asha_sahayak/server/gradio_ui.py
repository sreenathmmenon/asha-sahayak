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
    history = [(None, f"**ASHA Worker:** {first_msg}")]
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
        agent_msg = f"**Agent asks:** {action.question}"
    else:
        agent_msg = f"**Agent decides:** {action.referral_decision} | {action.urgency} | {action.primary_concern}"

    conv = obs.conversation
    if obs.done:
        asha_reply = f"**Episode Complete — Score: {obs.reward:.3f}**\n\n{obs.feedback or ''}"
        history.append((agent_msg, asha_reply))
        return history, f"Done! Final score: {obs.reward:.3f}", obs.feedback or "", action_json

    if conv and conv[-1].role == "asha_worker":
        asha_reply = f"**ASHA Worker:** {conv[-1].text}"
    else:
        asha_reply = ""
    history.append((agent_msg, asha_reply))

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


def build_gradio_app() -> gr.Blocks:
    with gr.Blocks(title="ASHA Sahayak") as demo:

        gr.Markdown("""
# ASHA Sahayak — AI Clinical Decision Support
**Trains AI agents to assist ASHA workers in rural India make correct referral decisions**

*Backed by official Indian Government IMNCI protocol · 1.07 million ASHA workers · 600 million people*
        """)

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
---
**referral_decision:** REFER_IMMEDIATELY | REFER_WITHIN_24H | TREAT_AT_HOME | MONITOR | PENDING

Set PENDING + question to ask. Set a final decision to end the episode.
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

    return demo
