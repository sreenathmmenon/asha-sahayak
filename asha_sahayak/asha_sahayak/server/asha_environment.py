"""
ASHA Sahayak — Core Environment
Implements the OpenEnv reset() / step() / state interface.

Multi-turn episodic design:
  - Turn 1: ASHA worker presents the initial complaint
  - Turns 2-N: Agent asks clarifying questions, environment reveals more symptoms
  - Final turn: Agent submits structured referral decision
  - Grader scores the decision against IMNCI ground truth

LeCun requirement enforced: agent must ask at least 1 question before deciding.
"""

from __future__ import annotations

import random
import uuid
from typing import Optional

from ..models import (
    AshaAction,
    AshaObservation,
    AshaState,
    ConversationTurn,
    PatientContext,
)
from .corpus.cases import ALL_CASES, CASES_BY_DIFFICULTY, ClinicalCase
from .grader import StepGrade, grade_action

# Max turns per difficulty level
MAX_TURNS = {
    "easy":   4,
    "medium": 5,
    "hard":   6,
}


class AshaEnvironment:
    """
    OpenEnv-compatible environment for ASHA worker clinical decision support.
    Each episode = one patient case. Multi-turn conversation.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        self._state: Optional[AshaState] = None
        self._case: Optional[ClinicalCase] = None
        self._conversation: list[ConversationTurn] = []
        self._asked_any_question: bool = False

    # ------------------------------------------------------------------
    # reset() — start a new episode
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "easy", seed: int = 42) -> AshaObservation:
        """
        Start a new episode.

        Parameters
        ----------
        task_id : str
            Difficulty level: "easy" | "medium" | "hard"
        seed : int
            Random seed for reproducibility
        """
        rng = random.Random(seed)
        case_ids = CASES_BY_DIFFICULTY.get(task_id, CASES_BY_DIFFICULTY["easy"])
        case_id = rng.choice(case_ids)
        case = ALL_CASES[case_id]

        episode_id = str(uuid.uuid4())[:8]

        self._case = case
        self._asked_any_question = False
        self._conversation = [
            ConversationTurn(
                role="asha_worker",
                text=case.initial_presentation,
            )
        ]

        self._state = AshaState(
            episode_id=episode_id,
            step_count=0,
            task_id=task_id,
            case_id=case_id,
            seed=seed,
            revealed_symptom_groups=[],
            asked_at_least_one_question=False,
        )

        return AshaObservation(
            conversation=list(self._conversation),
            patient_context=PatientContext(
                age_description=case.age_description,
                gender=case.gender,
                location=case.location,
                malaria_risk_area=case.malaria_risk_area,
                season=case.season,
            ),
            task_id=task_id,
            turn_number=1,
            max_turns=MAX_TURNS[task_id],
            done=False,
            reward=0.0,
            feedback=None,
        )

    # ------------------------------------------------------------------
    # step() — process one agent action
    # ------------------------------------------------------------------

    def step(self, action: AshaAction) -> AshaObservation:
        """
        Process the agent's action for this turn.

        If action.question is set → agent is asking a clarifying question.
        If action.referral_decision != "PENDING" → agent is giving final decision.
        """
        assert self._state is not None, "Call reset() before step()"
        assert self._case is not None

        self._state.step_count += 1
        max_turns = MAX_TURNS[self._state.task_id]
        turn_number = self._state.step_count + 1  # +1 because turn 1 is the initial ASHA message

        # Determine if agent is asking a question or giving a final decision
        is_asking_question = (
            action.question is not None
            and len(action.question.strip()) > 0
            and action.referral_decision.upper() == "PENDING"
        )

        # Track if any question was ever asked
        if is_asking_question:
            self._asked_any_question = True
            self._state.asked_at_least_one_question = True

        # Record agent turn
        agent_text = (
            action.question
            if is_asking_question
            else f"Decision: {action.referral_decision} | Concern: {action.primary_concern}"
        )
        self._conversation.append(ConversationTurn(role="agent", text=agent_text))

        # Is this a final decision? (explicit non-PENDING decision OR max turns reached)
        is_final_decision = (
            not is_asking_question
            and action.referral_decision.upper() != "PENDING"
        )
        forced_end = (self._state.step_count >= max_turns - 1)

        if is_asking_question and not forced_end:
            # Agent asked a question — generate environment response
            response_text = self._generate_response(action.question)
            self._conversation.append(
                ConversationTurn(role="asha_worker", text=response_text)
            )

            # Dense/intermediate reward: reward agent for asking about key danger signs
            intermediate_reward = self._compute_intermediate_reward(action.question)

            return AshaObservation(
                conversation=list(self._conversation),
                patient_context=PatientContext(
                    age_description=self._case.age_description,
                    gender=self._case.gender,
                    location=self._case.location,
                    malaria_risk_area=self._case.malaria_risk_area,
                    season=self._case.season,
                ),
                task_id=self._state.task_id,
                turn_number=turn_number,
                max_turns=max_turns,
                done=False,
                reward=intermediate_reward,
                feedback=None,
            )

        # Final decision (or forced end) — grade the agent
        grade = grade_action(
            predicted_referral=action.referral_decision,
            predicted_urgency=action.urgency,
            predicted_concern=action.primary_concern,
            asked_question_this_turn=is_asking_question,
            asked_any_question=self._asked_any_question,
            case=self._case,
            turn_number=turn_number,
            max_turns=max_turns,
        )

        # Update state with scores
        self._state.score_referral = grade.referral_score
        self._state.score_urgency = grade.urgency_score
        self._state.score_primary_concern = grade.primary_concern_score
        self._state.score_information_gathering = grade.information_gathering_score
        self._state.final_score = grade.composite_reward
        self._state.done = True

        return AshaObservation(
            conversation=list(self._conversation),
            patient_context=PatientContext(
                age_description=self._case.age_description,
                gender=self._case.gender,
                location=self._case.location,
                malaria_risk_area=self._case.malaria_risk_area,
                season=self._case.season,
            ),
            task_id=self._state.task_id,
            turn_number=turn_number,
            max_turns=max_turns,
            done=True,
            reward=grade.composite_reward,
            feedback=grade.feedback_message,
        )

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> AshaState:
        assert self._state is not None, "Call reset() before accessing state"
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_intermediate_reward(self, question: str) -> float:
        """
        Dense reward signal for clarifying questions.

        Awards a small positive reward when the agent asks about a key danger sign
        that is relevant to this case and hasn't been asked about before.

        This gives RL agents a learning signal on every step, not just at episode end.

        Reward schedule:
          - Asks about a NEW key danger sign keyword: +0.10
          - Asks about a keyword already seen this episode: +0.02 (small, avoid repetition)
          - Asks an irrelevant question: 0.0
          - Small bonus if question is specific/long (avoids yes/no fishing): +0.02
        """
        assert self._case is not None
        question_lower = question.lower()

        # Check if the question touches any key danger sign
        matched_new = False
        matched_seen = False

        for sign in self._case.key_danger_signs:
            sign_lower = sign.lower()
            # Partial word match: any word from the sign appears in question
            sign_words = [w for w in sign_lower.replace("_", " ").split() if len(w) > 3]
            if any(w in question_lower for w in sign_words):
                if sign_lower not in self._state.revealed_symptom_groups:
                    matched_new = True
                    self._state.revealed_symptom_groups.append(sign_lower)
                else:
                    matched_seen = True

        if matched_new:
            reward = 0.10
        elif matched_seen:
            reward = 0.02
        else:
            reward = 0.0

        # Small bonus for specific/detailed questions (length proxy)
        if len(question.split()) >= 8:
            reward += 0.02

        return round(min(reward, 0.15), 3)

    def _generate_response(self, question: str) -> str:
        """
        Match the agent's question against the case's followup_responses dict.
        Uses keyword matching (case-insensitive).
        Returns the most relevant response, or a generic "no other signs" fallback.
        """
        assert self._case is not None
        question_lower = question.lower()

        best_key: Optional[str] = None
        best_score = 0

        for keyword, response in self._case.followup_responses.items():
            if keyword in question_lower:
                score = len(keyword)  # prefer longer/more specific keyword matches
                if score > best_score:
                    best_score = score
                    best_key = keyword

        if best_key:
            return self._case.followup_responses[best_key]

        # Fallback: no matching keyword — generic response
        return (
            "Didi, koi aur khaas baat nahi dikh rahi abhi. "
            "Jo bataya hai wahi hai."
            "\n[Sister, no other special signs visible right now. What I told you is all.]"
        )
