"""
ASHA Sahayak — Data Models
All Action / Observation / State types for the OpenEnv RL environment.
Inherits from openenv-core base classes (Action, Observation, State).
"""

from __future__ import annotations

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Action — what the AI agent sends each turn
# ---------------------------------------------------------------------------

class AshaAction(Action):
    """
    The agent's response each turn.

    The agent may either:
      - ask a clarifying question (question field set, referral_decision = "PENDING")
      - give a final decision (referral_decision set, question = None)

    Structured output — no free-text drug names or dosages allowed.
    """

    model_config = {"extra": "allow", "validate_assignment": True, "arbitrary_types_allowed": True}

    # "PENDING" while still asking questions; final decision otherwise
    referral_decision: str  # REFER_IMMEDIATELY | REFER_WITHIN_24H | TREAT_AT_HOME | MONITOR | PENDING

    urgency: str  # immediate | within_24h | routine | monitor | unknown

    # The single most critical clinical finding the agent identified
    primary_concern: str  # e.g. "pre_eclampsia_risk", "neonatal_sepsis", "severe_pneumonia"

    # Structured action items — no free text drug names
    action_items: List[str] = []
    # e.g. ["transport_to_phc", "alert_district_officer", "keep_warm"]

    # Clarifying question the agent wants to ask (None = making final decision)
    question: Optional[str] = None

    # Internal confidence 0.0-1.0
    confidence: float = 0.8


# ---------------------------------------------------------------------------
# Nested types used inside Observation
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """A single turn in the ASHA worker conversation."""
    role: str          # "asha_worker" | "agent"
    text: str


class PatientContext(BaseModel):
    """Metadata about the patient — revealed progressively."""
    age_description: str        # "8 months", "pregnant 7 months", "newborn 3 days"
    gender: str                 # "male" | "female" | "unknown"
    location: str               # "rural_bihar", "tribal_jharkhand", etc.
    malaria_risk_area: bool     # affects fever classification
    season: str                 # "monsoon" | "winter" | "summer" | "post_monsoon"


# ---------------------------------------------------------------------------
# Observation — what the environment sends back each turn
# ---------------------------------------------------------------------------

class AshaObservation(Observation):
    """
    What the agent sees each turn.
    Symptoms are revealed INCREMENTALLY — the agent must ask clarifying questions
    to gather full information before making a final decision.
    """

    model_config = {"extra": "allow", "validate_assignment": True, "arbitrary_types_allowed": True}

    # Current conversation history
    conversation: List[ConversationTurn]

    # Patient context (available from turn 1)
    patient_context: PatientContext

    # Which task is being evaluated
    task_id: str        # "easy" | "medium" | "hard"

    # Turn number within this episode
    turn_number: int

    # Maximum turns allowed before forced decision
    max_turns: int

    # Feedback after final decision (None until done=True)
    feedback: Optional[str] = None


# ---------------------------------------------------------------------------
# State — internal episode metadata (not all visible to agent)
# ---------------------------------------------------------------------------

class AshaState(State):
    """Internal environment state."""

    model_config = {"extra": "allow", "validate_assignment": True, "arbitrary_types_allowed": True}

    task_id: str = ""
    case_id: str = ""
    seed: int = 0

    # Which symptom groups have been revealed so far
    revealed_symptom_groups: List[str] = Field(default_factory=list)

    # Whether agent has asked at least 1 clarifying question (LeCun requirement)
    asked_at_least_one_question: bool = False

    # Running score components
    score_referral: float = 0.0
    score_urgency: float = 0.0
    score_primary_concern: float = 0.0
    score_information_gathering: float = 0.0

    # Final episode outcome
    final_score: float = 0.0
    done: bool = False
