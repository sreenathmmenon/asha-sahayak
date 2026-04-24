"""
ASHA Sahayak — Multi-Agent Environment
Two-phase episode: ASHA Worker (community) → PHC Doctor (facility)

Phase 1 (ASHA Worker): Agent asks questions, gathers clinical information, makes referral decision.
Phase 2 (PHC Doctor): Agent receives ONLY the referral note (information asymmetry), makes disposition.

This implements OpenEnv Theme 1 (Multi-Agent Interactions).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .asha_environment import AshaEnvironment
from .grader import grade_action, grade_doctor_action
from ..models import AshaAction, AshaObservation, PHCDoctorAction
from .corpus.cases import ALL_CASES, CASES_BY_DIFFICULTY


@dataclass
class MultiAgentState:
    session_id: str
    phase: str  # "asha" | "doctor" | "done"
    case_id: str
    task_id: str
    seed: int

    # ASHA phase tracking
    asha_turn: int = 0
    asha_max_turns: int = 4
    asha_score: float = 0.0
    asha_done: bool = False
    asha_referral: str = ""
    asha_urgency: str = ""
    asha_primary_concern: str = ""
    asha_conversation: List[Dict[str, str]] = field(default_factory=list)

    # Doctor phase tracking
    doctor_turn: int = 0
    doctor_max_turns: int = 2
    doctor_score: float = 0.0
    doctor_done: bool = False

    # Combined
    episode_done: bool = False
    final_combined_reward: float = 0.0


class MultiAgentAshaEnvironment:
    """
    Two-phase multi-agent environment.

    ASHA Worker phase: Standard AshaEnvironment episode.
    PHC Doctor phase: Doctor sees referral note only (information asymmetry), makes disposition.

    Combined reward: 0.55 * R_doctor + 0.30 * R_asha + 0.15 * R_communication
    """

    def __init__(self):
        self._asha_env: Optional[AshaEnvironment] = None
        self._state: Optional[MultiAgentState] = None

    def reset(self, task_id: str = "easy", seed: int = 42, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Start a new multi-agent episode. Returns ASHA Worker's initial observation."""
        self._asha_env = AshaEnvironment()
        obs = self._asha_env.reset(task_id=task_id, seed=seed)
        case = self._asha_env._case

        max_turns_map = {"easy": 4, "medium": 5, "hard": 6}

        self._state = MultiAgentState(
            session_id=session_id or uuid.uuid4().hex[:12],
            phase="asha",
            case_id=case.case_id,
            task_id=task_id,
            seed=seed,
            asha_max_turns=max_turns_map.get(task_id, 4),
        )

        return {
            "phase": "asha",
            "session_id": self._state.session_id,
            "observation": self._obs_to_dict(obs),
            "role": "asha_worker",
            "instruction": (
                "You are an AI assistant helping an ASHA (Accredited Social Health Activist) worker "
                "in rural India assess a patient. Ask clarifying questions to gather clinical information, "
                "then make a referral decision. Your final referral note will be passed to a PHC Doctor."
            ),
        }

    def step_asha(self, action: AshaAction) -> Dict[str, Any]:
        """Process ASHA Worker's action. Returns observation and phase info."""
        if self._state is None or self._state.phase != "asha":
            raise ValueError("Not in ASHA phase. Call reset() first or doctor phase is active.")

        obs = self._asha_env.step(action)
        self._state.asha_turn += 1

        # Track conversation for doctor handoff
        if action.question:
            self._state.asha_conversation.append({
                "turn": self._state.asha_turn,
                "question": action.question,
            })

        if obs.done:
            # ASHA phase complete — record score and prepare doctor phase
            self._state.asha_done = True
            self._state.asha_score = obs.reward
            self._state.asha_referral = action.referral_decision
            self._state.asha_urgency = action.urgency
            self._state.asha_primary_concern = action.primary_concern
            self._state.phase = "doctor"

            # Build referral note for doctor (information asymmetry)
            case = self._asha_env._case
            referral_note = self._build_referral_note(action, case)

            return {
                "phase": "doctor",
                "session_id": self._state.session_id,
                "observation": self._obs_to_dict(obs),
                "asha_score": obs.reward,
                "role": "phc_doctor",
                "referral_note": referral_note,
                "instruction": (
                    "You are a PHC (Primary Health Centre) Doctor. You have received a referral note "
                    "from an ASHA worker. Based ONLY on this referral note, decide the patient disposition. "
                    "You do NOT have access to the original conversation — only the referral note."
                ),
                "available_dispositions": ["manage_at_phc", "refer_to_fru", "refer_to_district"],
                "message": "ASHA phase complete. Now in PHC Doctor phase.",
            }

        return {
            "phase": "asha",
            "session_id": self._state.session_id,
            "observation": self._obs_to_dict(obs),
            "role": "asha_worker",
        }

    def step_doctor(self, action: PHCDoctorAction) -> Dict[str, Any]:
        """Process PHC Doctor's disposition. Returns final episode result."""
        if self._state is None or self._state.phase != "doctor":
            raise ValueError("Not in Doctor phase. Complete ASHA phase first.")

        case = ALL_CASES[self._state.case_id]

        # Grade doctor's decision
        doctor_score = grade_doctor_action(
            disposition=action.disposition,
            case=case,
            asha_score=self._state.asha_score,
        )

        # Communication quality score: based on whether ASHA provided enough info
        # (more questions asked + correct concern identified = better handoff)
        questions_asked = len(self._state.asha_conversation)
        r_comm = min(1.0, 0.5 + 0.1 * questions_asked)

        # Combined reward formula
        combined_reward = (
            0.55 * doctor_score
            + 0.30 * self._state.asha_score
            + 0.15 * r_comm
        )
        combined_reward = max(0.001, min(0.999, combined_reward))

        self._state.doctor_score = doctor_score
        self._state.doctor_done = True
        self._state.episode_done = True
        self._state.final_combined_reward = combined_reward
        self._state.phase = "done"

        return {
            "phase": "done",
            "session_id": self._state.session_id,
            "done": True,
            "reward": combined_reward,
            "breakdown": {
                "doctor_score": round(doctor_score, 4),
                "asha_score": round(self._state.asha_score, 4),
                "communication_score": round(r_comm, 4),
                "combined_reward": round(combined_reward, 4),
            },
            "feedback": self._build_episode_feedback(case, action, doctor_score, combined_reward),
            "correct_doctor_decision": getattr(case, 'correct_doctor_decision', 'manage_at_phc'),
            "your_doctor_decision": action.disposition,
        }

    def get_observations(self) -> Dict[str, Any]:
        """Return role-scoped observations for both agents."""
        if self._state is None:
            return {"error": "No active episode"}

        case = ALL_CASES.get(self._state.case_id)
        asha_obs = None
        if self._asha_env and self._asha_env._state:
            asha_obs = {
                "turn": self._state.asha_turn,
                "max_turns": self._state.asha_max_turns,
                "done": self._state.asha_done,
                "score": self._state.asha_score,
            }

        doctor_obs = None
        if self._state.phase in ("doctor", "done"):
            doctor_obs = {
                "referral_decision": self._state.asha_referral,
                "urgency": self._state.asha_urgency,
                "primary_concern": self._state.asha_primary_concern,
                "questions_asked": len(self._state.asha_conversation),
                "done": self._state.doctor_done,
                "score": self._state.doctor_score,
            }

        return {
            "phase": self._state.phase,
            "session_id": self._state.session_id,
            "asha": asha_obs,
            "doctor": doctor_obs,
            "episode_done": self._state.episode_done,
            "combined_reward": self._state.final_combined_reward if self._state.episode_done else None,
        }

    def _build_referral_note(self, action: AshaAction, case) -> str:
        """Build a structured referral note from ASHA's decision — what the doctor sees."""
        questions_summary = ""
        if self._state.asha_conversation:
            q_list = [f"  - {q['question']}" for q in self._state.asha_conversation]
            questions_summary = "\nQuestions asked during assessment:\n" + "\n".join(q_list)

        return (
            f"ASHA WORKER REFERRAL NOTE\n"
            f"{'='*40}\n"
            f"Patient: {case.age_description}, {case.gender}\n"
            f"Location: {case.location}\n"
            f"Season: {case.season}\n"
            f"\nPresenting complaint:\n{case.initial_presentation}\n"
            f"{questions_summary}\n"
            f"\nASHA Worker Assessment:\n"
            f"  Referral decision: {action.referral_decision}\n"
            f"  Urgency: {action.urgency}\n"
            f"  Primary concern: {action.primary_concern}\n"
            f"  Confidence: {action.confidence}\n"
            f"\nReferred by ASHA Worker to PHC for further assessment."
        )

    def _build_episode_feedback(self, case, doctor_action: PHCDoctorAction,
                                 doctor_score: float, combined_reward: float) -> str:
        correct_decision = getattr(case, 'correct_doctor_decision', 'manage_at_phc')
        return (
            f"MULTI-AGENT EPISODE COMPLETE\n"
            f"{'='*40}\n"
            f"Case: {case.title}\n"
            f"\nASHA Worker Score: {self._state.asha_score:.3f}\n"
            f"PHC Doctor Score:  {doctor_score:.3f}\n"
            f"Combined Reward:   {combined_reward:.3f}\n"
            f"\nDoctor Decision:\n"
            f"  Your disposition: {doctor_action.disposition}\n"
            f"  Correct disposition: {correct_decision}\n"
            f"  Rationale provided: {doctor_action.rationale or '(none)'}\n"
            f"\nClinical explanation:\n{case.explanation}"
        )

    @staticmethod
    def _obs_to_dict(obs: AshaObservation) -> Dict[str, Any]:
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
