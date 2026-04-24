"""
ASHA Sahayak — FastAPI Server
OpenEnv-compatible HTTP server for the ASHA clinical decision support environment.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional

import traceback
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

from .asha_environment import AshaEnvironment
from .multi_agent_env import MultiAgentAshaEnvironment
from ..models import AshaAction

# ---------------------------------------------------------------------------
# Pydantic request/response schemas (FastAPI validation layer)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42
    session_id: Optional[str] = None


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
# App + session-scoped environments
# ---------------------------------------------------------------------------

from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

class ForceHTTPSMiddleware(BaseHTTPMiddleware):
    """Rewrites http:// URLs to https:// in redirects when behind HF proxy."""
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        if response.status_code in (301, 302, 307, 308):
            location = response.headers.get("location", "")
            if location.startswith("http://"):
                response.headers["location"] = "https://" + location[len("http://"):]
        return response

app = FastAPI(
    title="ASHA Sahayak",
    description=(
        "AI clinical decision support for ASHA workers in rural India. "
        "OpenEnv RL environment backed by official IMNCI protocol."
    ),
    version="0.1.0",
)

app.add_middleware(ForceHTTPSMiddleware)

# Session registry — maps session_id -> AshaEnvironment
_sessions: dict[str, AshaEnvironment] = {}
_lock = threading.Lock()

# Multi-agent session registry
_multi_sessions: dict[str, MultiAgentAshaEnvironment] = {}

# Default session id for backward-compatible headerless /step calls.
# Updated on every /reset so the most-recently-reset session is the fallback.
_default_session_id: Optional[str] = None


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

@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")


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
        "supports_concurrent_sessions": True,
        "max_concurrent_sessions": 64,
        "tasks": ["easy", "medium", "hard"],
        "num_cases": 31,
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
def reset(req: ResetRequest = None) -> Dict[str, Any]:
    """Start a new episode. Returns initial observation and session_id."""
    global _default_session_id

    if req is None:
        req = ResetRequest()

    session_id = req.session_id or uuid.uuid4().hex[:12]

    try:
        env = AshaEnvironment()
        obs = env.reset(task_id=req.task_id, seed=req.seed)
        with _lock:
            _sessions[session_id] = env
            _default_session_id = session_id
        return {"observation": _obs_to_dict(obs), "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(
    action: ActionRequest,
    session_id: str = Header(default="", alias="X-Session-ID"),
) -> Dict[str, Any]:
    """Process one agent action. Returns new observation with reward."""
    global _default_session_id

    # Fall back to the most-recently-reset session if no header is provided.
    # This keeps backward compatibility with inference.py which does not send
    # an X-Session-ID header.
    effective_id = session_id or _default_session_id or ""

    with _lock:
        env = _sessions.get(effective_id)

    if env is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. POST /reset first.",
        )

    try:
        asha_action = AshaAction(
            referral_decision=action.referral_decision,
            urgency=action.urgency,
            primary_concern=action.primary_concern,
            action_items=action.action_items,
            question=action.question,
            confidence=action.confidence,
        )
        obs = env.step(asha_action)
        return {"observation": _obs_to_dict(obs), "metadata": {}}
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/state")
def state(
    session_id: str = Header(default="", alias="X-Session-ID"),
) -> Dict[str, Any]:
    """Return current internal state."""
    global _default_session_id

    effective_id = session_id or _default_session_id or ""

    with _lock:
        env = _sessions.get(effective_id)

    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. POST /reset first.")

    try:
        s = env._state
        if s is None:
            raise AssertionError("No active episode.")
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


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Remove a session from the registry."""
    with _lock:
        removed = _sessions.pop(session_id, None)
    if removed is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "deleted", "session_id": session_id}


# ---------------------------------------------------------------------------
# Multi-Agent Routes (Theme 1 — ASHA Worker + PHC Doctor)
# ---------------------------------------------------------------------------

class MultiResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42
    session_id: Optional[str] = None


class DoctorActionRequest(BaseModel):
    disposition: str = "manage_at_phc"
    investigations: List[str] = []
    treatment: str = ""
    rationale: str = ""


@app.post("/multi/reset")
def multi_reset(req: MultiResetRequest) -> Dict[str, Any]:
    """Start a new multi-agent episode (ASHA Worker + PHC Doctor)."""
    env = MultiAgentAshaEnvironment()
    result = env.reset(task_id=req.task_id, seed=req.seed, session_id=req.session_id)
    session_id = result["session_id"]
    with _lock:
        _multi_sessions[session_id] = env
    return result


@app.post("/multi/step/asha")
def multi_step_asha(
    action: ActionRequest,
    session_id: str = Header(default="", alias="X-Session-ID"),
) -> Dict[str, Any]:
    """ASHA Worker turn in a multi-agent episode."""
    with _lock:
        env = _multi_sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Multi-agent session not found. POST /multi/reset first.")
    try:
        asha_action = AshaAction(
            referral_decision=action.referral_decision,
            urgency=action.urgency,
            primary_concern=action.primary_concern,
            action_items=action.action_items,
            question=action.question,
            confidence=action.confidence,
        )
        return env.step_asha(asha_action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/multi/step/doctor")
def multi_step_doctor(
    action: DoctorActionRequest,
    session_id: str = Header(default="", alias="X-Session-ID"),
) -> Dict[str, Any]:
    """PHC Doctor turn in a multi-agent episode."""
    with _lock:
        env = _multi_sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Multi-agent session not found.")
    try:
        from ..models import PHCDoctorAction
        doctor_action = PHCDoctorAction(
            disposition=action.disposition,
            investigations=action.investigations,
            treatment=action.treatment,
            rationale=action.rationale,
        )
        result = env.step_doctor(doctor_action)
        # Clean up completed session
        if result.get("done"):
            with _lock:
                _multi_sessions.pop(session_id, None)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/multi/observations")
def multi_observations(
    session_id: str = Header(default="", alias="X-Session-ID"),
) -> Dict[str, Any]:
    """Return role-scoped observations for both agents in a multi-agent session."""
    with _lock:
        env = _multi_sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Multi-agent session not found.")
    return env.get_observations()


# Mount Gradio UI at /ui
try:
    import gradio as gr
    from .gradio_ui import build_gradio_app
    _gradio_app = build_gradio_app()
    app = gr.mount_gradio_app(app, _gradio_app, path="/ui")
    print("[INFO] Gradio UI mounted at /ui", flush=True)
except Exception as _gradio_err:
    print(f"[WARN] Gradio UI not available: {_gradio_err}", flush=True)


def main() -> None:
    """Entry point for the ASHA Sahayak server (used by openenv validate)."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
