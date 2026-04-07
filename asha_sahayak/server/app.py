"""
ASHA Sahayak — FastAPI Server
OpenEnv-compatible HTTP server for the ASHA clinical decision support environment.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import traceback

from .asha_environment import AshaEnvironment
from ..models import AshaAction

# ---------------------------------------------------------------------------
# Pydantic request/response schemas (FastAPI validation layer)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42


class ActionRequest(BaseModel):
    referral_decision: str = "PENDING"
    urgency: str = "monitor"
    primary_concern: str = ""
    action_items: List[str] = []
    question: Optional[str] = None
    confidence: float = 0.8


class ConversationTurnOut(BaseModel):
    role: str
    text: str


class PatientContextOut(BaseModel):
    age_description: str
    gender: str
    location: str
    malaria_risk_area: bool
    season: str


class ObservationOut(BaseModel):
    conversation: List[ConversationTurnOut]
    patient_context: PatientContextOut
    task_id: str
    turn_number: int
    max_turns: int
    done: bool
    reward: float
    feedback: Optional[str]


class StateOut(BaseModel):
    episode_id: str
    step_count: int
    task_id: str
    case_id: str
    seed: int
    asked_at_least_one_question: bool
    final_score: float
    done: bool


# ---------------------------------------------------------------------------
# App + single environment instance per connection
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ASHA Sahayak",
    description=(
        "AI clinical decision support for ASHA workers in rural India. "
        "OpenEnv RL environment backed by official IMNCI protocol."
    ),
    version="0.1.0",
)

# Module-level environment (one instance per server process)
_env: AshaEnvironment = AshaEnvironment()


def _obs_to_dict(obs) -> Dict[str, Any]:
    return ObservationOut(
        conversation=[
            ConversationTurnOut(role=t.role, text=t.text)
            for t in obs.conversation
        ],
        patient_context=PatientContextOut(
            age_description=obs.patient_context.age_description,
            gender=obs.patient_context.gender,
            location=obs.patient_context.location,
            malaria_risk_area=obs.patient_context.malaria_risk_area,
            season=obs.patient_context.season,
        ),
        task_id=obs.task_id,
        turn_number=obs.turn_number,
        max_turns=obs.max_turns,
        done=obs.done,
        reward=obs.reward,
        feedback=obs.feedback,
    ).model_dump()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    return {
        "name": "asha_sahayak",
        "description": (
            "AI clinical decision support for ASHA workers in rural India. "
            "Multi-turn triage RL environment backed by official Indian Government IMNCI protocol."
        ),
        "version": "0.1.0",
        "supports_concurrent_sessions": False,
        "tasks": ["easy", "medium", "hard"],
        "num_cases": 16,
    }


@app.get("/schema")
def schema() -> Dict[str, Any]:
    return {
        "action": {
            "referral_decision": "REFER_IMMEDIATELY | REFER_WITHIN_24H | TREAT_AT_HOME | MONITOR | PENDING",
            "urgency": "immediate | within_24h | routine | monitor",
            "primary_concern": "string — clinical concern identified",
            "action_items": "list[string] — structured action items (no drug names)",
            "question": "string | null — clarifying question to ask ASHA worker",
            "confidence": "float 0.0-1.0",
        },
        "observation": {
            "conversation": "list of {role, text} turns",
            "patient_context": "age, gender, location, malaria_risk_area, season",
            "task_id": "easy | medium | hard",
            "turn_number": "int",
            "max_turns": "int",
            "done": "bool",
            "reward": "float 0.0-1.0 (intermediate reward on question steps, final score at done=True)",
            "feedback": "string | null — clinical explanation at episode end",
        },
        "state": {
            "episode_id": "string",
            "step_count": "int",
            "task_id": "string",
            "case_id": "string",
            "asked_at_least_one_question": "bool",
            "final_score": "float",
            "done": "bool",
        },
    }


@app.post("/reset")
def reset(request: ResetRequest = None) -> Dict[str, Any]:
    """Start a new episode. Returns initial observation."""
    global _env
    _env = AshaEnvironment()  # fresh instance

    if request is None:
        request = ResetRequest()

    try:
        obs = _env.reset(task_id=request.task_id, seed=request.seed)
        return {"observation": _obs_to_dict(obs), "metadata": {}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(action: ActionRequest) -> Dict[str, Any]:
    """Process one agent action. Returns new observation with reward."""
    try:
        asha_action = AshaAction(
            referral_decision=action.referral_decision,
            urgency=action.urgency,
            primary_concern=action.primary_concern,
            action_items=action.action_items,
            question=action.question,
            confidence=action.confidence,
        )
        obs = _env.step(asha_action)
        return {"observation": _obs_to_dict(obs), "metadata": {}}
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/state")
def state() -> Dict[str, Any]:
    """Return current internal state."""
    try:
        s = _env.state
        return {
            "state": StateOut(
                episode_id=s.episode_id,
                step_count=s.step_count,
                task_id=s.task_id,
                case_id=s.case_id,
                seed=s.seed,
                asked_at_least_one_question=s.asked_at_least_one_question,
                final_score=s.final_score,
                done=s.done,
            ).model_dump()
        }
    except AssertionError as e:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")


# Mount Gradio UI at /ui
try:
    import gradio as gr
    from .gradio_ui import build_gradio_app
    _gradio_app = build_gradio_app()
    app = gr.mount_gradio_app(app, _gradio_app, path="/ui")
except Exception as _gradio_err:
    pass  # Gradio optional — API still works without it


def main() -> None:
    """Entry point for the ASHA Sahayak server (used by openenv validate)."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
