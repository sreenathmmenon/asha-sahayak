"""
Tests for ASHA Sahayak environment — correctness, reward bounds, interface compliance.
"""

import pytest
from openenv.core.env_server.types import Action, Observation, State

from asha_sahayak.models import AshaAction, AshaObservation, AshaState
from asha_sahayak.server.asha_environment import AshaEnvironment
from asha_sahayak.server.corpus.cases import ALL_CASES, CASES_BY_DIFFICULTY
from asha_sahayak.server.grader import grade_action


# ---------------------------------------------------------------------------
# Model inheritance
# ---------------------------------------------------------------------------

def test_action_inherits_openenv():
    assert issubclass(AshaAction, Action)

def test_observation_inherits_openenv():
    assert issubclass(AshaObservation, Observation)

def test_state_inherits_openenv():
    assert issubclass(AshaState, State)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def test_three_difficulty_levels():
    assert set(CASES_BY_DIFFICULTY.keys()) == {"easy", "medium", "hard"}

def test_minimum_cases_per_difficulty():
    for diff, cases in CASES_BY_DIFFICULTY.items():
        assert len(cases) >= 3, f"{diff} has fewer than 3 cases"

def test_all_cases_have_required_fields():
    for case_id, case in ALL_CASES.items():
        assert case.correct_referral in {"REFER_IMMEDIATELY", "REFER_WITHIN_24H", "TREAT_AT_HOME", "MONITOR"}
        assert case.correct_urgency in {"immediate", "within_24h", "routine", "monitor"}
        assert case.correct_primary_concern
        assert case.initial_presentation
        assert len(case.followup_responses) >= 3


# ---------------------------------------------------------------------------
# Environment lifecycle
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return AshaEnvironment()


def test_reset_returns_observation(env):
    obs = env.reset(task_id="easy", seed=42)
    assert isinstance(obs, AshaObservation)
    assert obs.turn_number == 1
    assert obs.done is False
    assert obs.reward == 0.0
    assert len(obs.conversation) == 1
    assert obs.conversation[0].role == "asha_worker"


def test_reset_is_deterministic(env):
    obs1 = env.reset(task_id="easy", seed=42)
    obs2 = env.reset(task_id="easy", seed=42)
    assert obs1.conversation[0].text == obs2.conversation[0].text


def test_step_question_returns_not_done(env):
    env.reset(task_id="easy", seed=42)
    action = AshaAction(
        referral_decision="PENDING",
        urgency="unknown",
        primary_concern="gathering_information",
        question="Does the child have chest indrawing?",
        confidence=0.5,
    )
    obs = env.step(action)
    assert obs.done is False
    assert 0.0 <= obs.reward <= 1.0


def test_step_final_decision_ends_episode(env):
    env.reset(task_id="easy", seed=42)
    # Ask one question first
    env.step(AshaAction(
        referral_decision="PENDING", urgency="unknown",
        primary_concern="info", question="Any danger signs?", confidence=0.5
    ))
    # Give final decision
    obs = env.step(AshaAction(
        referral_decision="REFER_IMMEDIATELY", urgency="immediate",
        primary_concern="severe_pneumonia", action_items=[], confidence=0.9
    ))
    assert obs.done is True
    assert 0.0 <= obs.reward <= 1.0
    assert obs.feedback is not None


def test_reward_bounds_all_tasks():
    for task_id, seed in [("easy", 42), ("medium", 123), ("hard", 500)]:
        env = AshaEnvironment()
        env.reset(task_id=task_id, seed=seed)
        env.step(AshaAction(
            referral_decision="PENDING", urgency="unknown",
            primary_concern="info", question="Any danger signs?", confidence=0.5
        ))
        obs = env.step(AshaAction(
            referral_decision="REFER_IMMEDIATELY", urgency="immediate",
            primary_concern="emergency", action_items=[], confidence=0.9
        ))
        assert 0.0 <= obs.reward <= 1.0, f"{task_id} reward out of bounds: {obs.reward}"


def test_perfect_score_on_correct_answer():
    """Agent that asks a question then gives the exact correct answer gets >= 0.9."""
    env = AshaEnvironment()
    env.reset(task_id="easy", seed=42)
    case = env._case  # use whatever case was selected for this seed
    env.step(AshaAction(
        referral_decision="PENDING", urgency="unknown",
        primary_concern="info", question="Does the child have chest indrawing?", confidence=0.5
    ))
    obs = env.step(AshaAction(
        referral_decision=case.correct_referral,
        urgency=case.correct_urgency,
        primary_concern=case.correct_primary_concern,
        action_items=[],
        confidence=0.95,
    ))
    assert obs.reward >= 0.9


def test_state_reflects_episode(env):
    env.reset(task_id="medium", seed=123)
    s = env.state
    assert s.task_id == "medium"
    assert s.step_count == 0
    assert s.done is False


def test_no_mutable_default_sharing():
    s1 = AshaState(task_id="easy", case_id="E01", seed=42)
    s2 = AshaState(task_id="hard", case_id="H01", seed=99)
    s1.revealed_symptom_groups.append("test")
    assert "test" not in s2.revealed_symptom_groups


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def test_grader_exact_match():
    case = ALL_CASES["E01"]
    grade = grade_action(
        predicted_referral=case.correct_referral,
        predicted_urgency=case.correct_urgency,
        predicted_concern=case.correct_primary_concern,
        asked_question_this_turn=False,
        asked_any_question=True,
        case=case,
        turn_number=2,
        max_turns=4,
    )
    assert grade.referral_score == 1.0
    assert grade.composite_reward >= 0.9


def test_grader_dangerous_undertriage_penalty():
    case = ALL_CASES["E01"]  # correct = REFER_IMMEDIATELY
    grade = grade_action(
        predicted_referral="TREAT_AT_HOME",
        predicted_urgency="routine",
        predicted_concern="mild_cough",
        asked_question_this_turn=False,
        asked_any_question=True,
        case=case,
        turn_number=2,
        max_turns=4,
    )
    assert grade.referral_score <= 0.1  # dangerous under-triage heavily penalized


def test_grader_no_question_penalty():
    case = ALL_CASES["E01"]
    grade = grade_action(
        predicted_referral=case.correct_referral,
        predicted_urgency=case.correct_urgency,
        predicted_concern=case.correct_primary_concern,
        asked_question_this_turn=False,
        asked_any_question=False,  # never asked anything
        case=case,
        turn_number=1,
        max_turns=4,
    )
    assert grade.information_gathering_score == 0.2  # capped at 0.2
