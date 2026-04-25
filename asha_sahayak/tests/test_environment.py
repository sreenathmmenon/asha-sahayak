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
    case = ALL_CASES["E01"]  # Severe pneumonia — chest indrawing → REFER_IMMEDIATELY
    env = AshaEnvironment()
    env.reset(task_id="easy", seed=42)
    env._case = case  # pin the case directly so adaptive curriculum doesn't interfere
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


# ---------------------------------------------------------------------------
# Hard safety gate (Gap 1)
# ---------------------------------------------------------------------------

def test_dangerous_undertriage_terminates_episode():
    """Dangerous undertriage (TREAT_AT_HOME for REFER_IMMEDIATELY case) must end episode instantly."""
    # Clear curriculum state so case selection is purely seed-based (no weighted sampling)
    AshaEnvironment._curriculum_attempts.clear()
    AshaEnvironment._curriculum_successes.clear()

    # seed=0 with no curriculum selects E01 (REFER_IMMEDIATELY) for easy difficulty
    env = AshaEnvironment()
    env.reset(task_id="easy", seed=0)
    assert env._case.correct_referral == "REFER_IMMEDIATELY", (
        f"Precondition failed: expected REFER_IMMEDIATELY, got {env._case.correct_referral} (case {env._case.case_id})"
    )

    obs = env.step(AshaAction(
        referral_decision="TREAT_AT_HOME",
        urgency="routine",
        primary_concern="mild_cough",
        action_items=[],
        question=None,
        confidence=0.5,
    ))

    assert obs.done is True, "Episode must terminate on dangerous undertriage"
    assert obs.reward <= 0.01, f"Reward must be minimal, got {obs.reward}"
    assert obs.feedback is not None and "CRITICAL" in obs.feedback, (
        f"Feedback must contain 'CRITICAL', got: {obs.feedback}"
    )


# ---------------------------------------------------------------------------
# Adaptive curriculum (Gap 2a)
# ---------------------------------------------------------------------------

def test_curriculum_weights_increase_after_failures():
    """After repeated failures in a category, its MAB sampling weight must exceed baseline 0.3."""
    # Clear shared curriculum state for isolation
    AshaEnvironment._curriculum_attempts.clear()
    AshaEnvironment._curriculum_successes.clear()

    # Run 12 episodes on E01 (REFER_IMMEDIATELY) sending wrong answer → triggers hard gate → low reward
    for _ in range(12):
        env = AshaEnvironment()
        env.reset(task_id="easy", seed=42)
        env.step(AshaAction(
            referral_decision="TREAT_AT_HOME",
            urgency="routine",
            primary_concern="mild_cough",
            action_items=[],
            question=None,
            confidence=0.5,
        ))

    curriculum = AshaEnvironment.get_curriculum_state()
    # At least one category should have weight > base 0.3 after repeated failures
    max_weight = max(v["weight"] for v in curriculum.values())
    assert max_weight > 0.3, (
        f"Curriculum weight should exceed baseline 0.3 after failures, got max={max_weight}"
    )


# ---------------------------------------------------------------------------
# Multi-agent episode path (Gap 2b)
# ---------------------------------------------------------------------------

def test_multi_agent_episode_path():
    """End-to-end ASHA Worker → PHC Doctor episode must complete with a valid reward."""
    try:
        from asha_sahayak.server.multi_agent_env import MultiAgentAshaEnvironment
        from asha_sahayak.models import PHCDoctorAction
    except ImportError as e:
        import pytest
        pytest.skip(f"Multi-agent env not importable: {e}")

    env = MultiAgentAshaEnvironment()
    result = env.reset(task_id="easy", seed=42)

    assert result["phase"] == "asha"
    assert result["role"] == "asha_worker"

    # Ask one clarifying question
    asha_q = AshaAction(
        referral_decision="PENDING",
        urgency="unknown",
        primary_concern="gathering_information",
        action_items=[],
        question="Does the child have any chest indrawing or fast breathing?",
        confidence=0.5,
    )
    step1 = env.step_asha(asha_q)
    assert step1["phase"] == "asha"

    # Make a final ASHA decision to trigger handoff to doctor
    asha_final = AshaAction(
        referral_decision="REFER_IMMEDIATELY",
        urgency="immediate",
        primary_concern="severe_pneumonia",
        action_items=["first_dose_antibiotic_before_transfer"],
        question=None,
        confidence=0.9,
    )
    step2 = env.step_asha(asha_final)
    assert step2["phase"] == "doctor", f"Expected doctor phase, got: {step2['phase']}"
    assert "referral_note" in step2

    # Doctor makes disposition
    doctor_action = PHCDoctorAction(
        disposition="refer_to_fru",
        rationale="Severe pneumonia requires FRU-level care.",
        treatment="First dose antibiotic given",
    )
    final = env.step_doctor(doctor_action)

    assert final["done"] is True
    assert 0.0 < final["reward"] <= 1.0, f"Reward out of range: {final['reward']}"
    assert "breakdown" in final
    assert "doctor_score" in final["breakdown"]
