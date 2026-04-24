"""
ASHA Sahayak — Deterministic Pure-Python Grader
Scores agent decisions against IMNCI ground truth.

All scoring is deterministic: same inputs always produce same outputs.
No LLM calls, no external APIs, no randomness.

Scoring formula:
  R = 0.40 * R_referral
    + 0.25 * R_urgency
    + 0.20 * R_primary_concern
    + 0.15 * R_information_gathering

LeCun requirement: agent must ask at least 1 clarifying question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .corpus.cases import ClinicalCase


# Ordered referral levels — used for partial-credit distance scoring
_REFERRAL_ORDER = {
    "REFER_IMMEDIATELY": 3,
    "REFER_WITHIN_24H":  2,
    "TREAT_AT_HOME":     1,
    "MONITOR":           0,
}

# Ordered urgency levels
_URGENCY_ORDER = {
    "immediate":  3,
    "within_24h": 2,
    "routine":    1,
    "monitor":    0,
}


@dataclass
class StepGrade:
    """Detailed grade breakdown for one agent decision."""

    referral_score:             float   # 0.0-1.0
    urgency_score:              float   # 0.0-1.0
    primary_concern_score:      float   # 0.0-1.0
    information_gathering_score: float  # 0.0-1.0

    composite_reward:           float   # weighted sum, clamped 0.0-1.0

    # Breakdown for feedback
    predicted_referral:    str
    correct_referral:      str
    predicted_urgency:     str
    correct_urgency:       str
    predicted_concern:     str
    correct_concern:       str
    asked_question:        bool
    danger_signs_surfaced: List[str]

    feedback_message:      str


def grade_action(
    predicted_referral: str,
    predicted_urgency: str,
    predicted_concern: str,
    asked_question_this_turn: bool,
    asked_any_question: bool,
    case: ClinicalCase,
    turn_number: int,
    max_turns: int,
) -> StepGrade:
    """
    Score an agent's final decision against the ground-truth case.

    Parameters
    ----------
    predicted_referral : str
        Agent's referral decision: REFER_IMMEDIATELY | REFER_WITHIN_24H | TREAT_AT_HOME | MONITOR
    predicted_urgency : str
        Agent's urgency: immediate | within_24h | routine | monitor
    predicted_concern : str
        Agent's primary clinical concern string
    asked_question_this_turn : bool
        Did the agent ask a clarifying question this specific turn?
    asked_any_question : bool
        Did the agent ask at least one question across the entire episode?
    case : ClinicalCase
        The ground-truth case from the corpus
    turn_number : int
        Current turn (1-indexed)
    max_turns : int
        Maximum turns allowed
    """

    # --- R_referral ---
    r_ref = _score_referral(predicted_referral, case)

    # --- R_urgency ---
    r_urg = _score_urgency(predicted_urgency, case.correct_urgency)

    # --- R_primary_concern ---
    r_concern = _score_primary_concern(predicted_concern, case.correct_primary_concern)

    # --- R_information_gathering (LeCun requirement) ---
    r_info = _score_information_gathering(
        asked_any_question=asked_any_question,
        turn_number=turn_number,
        max_turns=max_turns,
    )

    # --- Composite ---
    composite = (
        0.40 * r_ref
        + 0.25 * r_urg
        + 0.20 * r_concern
        + 0.15 * r_info
    )
    # Terminal bonus: perfect referral + asked a question = reward thoroughness
    if r_ref == 1.0 and asked_any_question:
        composite += 0.05
    composite = max(0.001, min(0.999, composite))

    # --- Feedback message ---
    fb = _build_feedback(
        predicted_referral, case.correct_referral,
        predicted_urgency, case.correct_urgency,
        predicted_concern, case.correct_primary_concern,
        r_ref, r_urg, r_concern, r_info,
        asked_any_question,
        case,
    )

    return StepGrade(
        referral_score=r_ref,
        urgency_score=r_urg,
        primary_concern_score=r_concern,
        information_gathering_score=r_info,
        composite_reward=composite,
        predicted_referral=predicted_referral,
        correct_referral=case.correct_referral,
        predicted_urgency=predicted_urgency,
        correct_urgency=case.correct_urgency,
        predicted_concern=predicted_concern,
        correct_concern=case.correct_primary_concern,
        asked_question=asked_any_question,
        danger_signs_surfaced=case.key_danger_signs,
        feedback_message=fb,
    )


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def _score_referral(predicted: str, case: ClinicalCase) -> float:
    """
    Score referral decision.
    - Exact match: 1.0
    - Near-miss (defined per case): partial credit from case.near_miss_referrals
    - Distance-based fallback: penalize by distance in referral order
    """
    predicted_norm = predicted.upper().strip()
    correct = case.correct_referral

    if predicted_norm == correct:
        return 1.0

    # Per-case near-miss partial credit
    if predicted_norm in case.near_miss_referrals:
        return case.near_miss_referrals[predicted_norm]

    # Distance fallback: further from correct = lower score
    pred_level = _REFERRAL_ORDER.get(predicted_norm, -1)
    corr_level = _REFERRAL_ORDER.get(correct, -1)

    if pred_level == -1 or corr_level == -1:
        return 0.0

    distance = abs(pred_level - corr_level)
    # distance 0=1.0, 1=0.5, 2=0.2, 3=0.0
    score_map = {0: 1.0, 1: 0.5, 2: 0.2, 3: 0.0}
    base_score = score_map.get(distance, 0.0)

    # Extra penalty for dangerous under-triage (treating immediately-urgent case as home care)
    if correct == "REFER_IMMEDIATELY" and predicted_norm in ("TREAT_AT_HOME", "MONITOR"):
        base_score = min(base_score, 0.1)

    return base_score


def _score_urgency(predicted: str, correct: str) -> float:
    """Score urgency level with ordered distance."""
    predicted_norm = predicted.lower().strip()
    correct_norm = correct.lower().strip()

    if predicted_norm == correct_norm:
        return 1.0

    pred_level = _URGENCY_ORDER.get(predicted_norm, -1)
    corr_level = _URGENCY_ORDER.get(correct_norm, -1)

    if pred_level == -1 or corr_level == -1:
        return 0.0

    distance = abs(pred_level - corr_level)
    score_map = {0: 1.0, 1: 0.5, 2: 0.2, 3: 0.0}
    return score_map.get(distance, 0.0)


def _score_primary_concern(predicted: str, correct: str) -> float:
    """
    Score primary clinical concern identification.
    Uses keyword matching for flexibility — the agent may use slightly different
    phrasing but if the key clinical words match, award partial credit.
    """
    pred_lower = predicted.lower().strip().replace("-", "_").replace(" ", "_")
    corr_lower = correct.lower().strip().replace("-", "_").replace(" ", "_")

    if pred_lower == corr_lower:
        return 1.0

    # Semantic aliases: common equivalent phrasings → score override
    # Keys are (predicted_norm_contains, correct_norm) → score
    _CONCERN_ALIASES: list[tuple[set[str], str, float]] = [
        # very_severe_febrile_disease_possible_meningitis aliases
        ({"meningitis", "severe_febrile_disease", "bacterial_meningitis", "viral_meningitis",
          "meningococcal_meningitis", "meningitis_signs", "possible_meningitis",
          "severe_febrile_illness_meningitis", "febrile_meningitis"},
         "very_severe_febrile_disease_possible_meningitis", 0.8),
        # pre_eclampsia_severe_features aliases
        ({"pre_eclampsia", "severe_pre_eclampsia", "preeclampsia", "eclampsia_risk",
          "pre_eclampsia_severe", "severe_preeclampsia"},
         "pre_eclampsia_severe_features", 0.8),
        # neonatal_hypothermia_with_sepsis_risk aliases
        ({"neonatal_hypothermia", "hypothermia_sepsis", "neonatal_cold_sepsis"},
         "neonatal_hypothermia_with_sepsis_risk", 0.8),
        # severe_complicated_sam aliases
        ({"severe_sam", "complicated_sam", "severe_malnutrition", "sam_complications",
          "severe_acute_malnutrition"},
         "severe_complicated_sam", 0.8),
        # omphalitis_with_systemic_spread aliases
        ({"omphalitis", "cord_infection_sepsis", "neonatal_omphalitis"},
         "omphalitis_with_systemic_spread", 0.8),
        # New cases (Phase 3)
        ({"physiological_jaundice", "neonatal_jaundice_physiological", "normal_newborn_jaundice"},
         "neonatal_physiological_jaundice", 0.8),
        ({"localized_pustules", "skin_infection_pustules", "neonatal_skin_pustules"},
         "localized_skin_pustules", 0.8),
        ({"malaria_uncomplicated", "simple_malaria", "rdt_positive_malaria", "plasmodium_malaria"},
         "uncomplicated_malaria", 0.8),
        ({"pathological_jaundice", "early_onset_jaundice", "hemolytic_jaundice", "jaundice_24h"},
         "neonatal_pathological_jaundice", 0.8),
        ({"gestational_diabetes", "gdm", "diabetes_pregnancy", "gdm_risk"},
         "gestational_diabetes_risk", 0.8),
        ({"severe_anemia_pregnancy", "pregnancy_anemia", "anaemia_pregnancy", "maternal_severe_anaemia"},
         "severe_anaemia_in_pregnancy", 0.8),
        ({"tb_contact", "child_tb", "pediatric_tuberculosis", "tb_exposure_child", "tuberculosis_child"},
         "pediatric_tb_contact", 0.8),
        ({"hypertension_ncd", "ncd_screening", "bp_risk", "cbac_referral", "hypertension_risk"},
         "ncd_hypertension_screening", 0.8),
        ({"birth_asphyxia", "neonatal_asphyxia", "perinatal_asphyxia", "baby_not_breathing"},
         "neonatal_birth_asphyxia", 0.8),
        ({"kernicterus", "bilirubin_encephalopathy", "severe_jaundice_neonate", "jaundice_brain"},
         "severe_neonatal_jaundice_kernicterus", 0.8),
        ({"postpartum_sepsis", "puerperal_infection", "postpartum_infection", "postpartum_fever_sepsis"},
         "puerperal_sepsis", 0.8),
        ({"adolescent_anaemia", "rksk_anaemia", "severe_anaemia_girl", "menorrhagia_anaemia"},
         "adolescent_severe_anaemia_with_cardiac_signs", 0.8),
        ({"cerebral_malaria", "severe_malaria_cns", "malaria_unconscious", "falciparum_cerebral"},
         "cerebral_malaria_falciparum", 0.8),
        ({"cord_prolapse", "umbilical_prolapse", "prolapsed_cord"},
         "cord_prolapse_obstetric_emergency", 0.8),
        ({"low_birth_weight", "lbw_preterm", "preterm_kmc", "kmc_lbw"},
         "preterm_low_birth_weight_kmc", 0.8),
    ]
    for alias_set, target_correct, alias_score in _CONCERN_ALIASES:
        if corr_lower == target_correct.replace("-", "_"):
            if pred_lower in alias_set or any(a in pred_lower for a in alias_set):
                return alias_score

    # Extract key clinical keywords from correct concern
    correct_keywords = set(corr_lower.split("_"))
    predicted_keywords = set(pred_lower.split("_"))

    # Remove stopwords
    stopwords = {"with", "and", "or", "of", "the", "in", "risk", "possible", "suspected"}
    correct_keywords -= stopwords
    predicted_keywords -= stopwords

    if not correct_keywords:
        return 0.0

    # Overlap ratio
    overlap = len(correct_keywords & predicted_keywords)
    ratio = overlap / len(correct_keywords)

    if ratio >= 0.8:
        return 0.9
    elif ratio >= 0.6:
        return 0.7
    elif ratio >= 0.4:
        return 0.5
    elif ratio >= 0.2:
        return 0.3
    else:
        return 0.0


def _score_information_gathering(
    asked_any_question: bool,
    turn_number: int,
    max_turns: int,
) -> float:
    """
    LeCun requirement: agent must ask at least one clarifying question.
    Rewards agents that gather information before deciding.
    """
    if not asked_any_question:
        # Agent never asked a question — significant penalty
        return 0.2

    # Reward proportional to how many turns were used for information gathering
    # (but don't penalize making a decision early if correct)
    turns_used_fraction = min(turn_number / max_turns, 1.0)

    if turns_used_fraction <= 0.3:
        # Decided very quickly — either very confident or did not explore enough
        return 0.7
    elif turns_used_fraction <= 0.6:
        return 0.9
    else:
        # Used most of the turns — thorough information gathering
        return 1.0


# ---------------------------------------------------------------------------
# Feedback builder
# ---------------------------------------------------------------------------

def _build_feedback(
    pred_ref: str, corr_ref: str,
    pred_urg: str, corr_urg: str,
    pred_concern: str, corr_concern: str,
    r_ref: float, r_urg: float, r_concern: float, r_info: float,
    asked_question: bool,
    case: ClinicalCase,
) -> str:
    lines = [
        f"CASE: {case.title}",
        f"",
        f"Your decision:    {pred_ref} ({pred_urg})",
        f"Correct decision: {corr_ref} ({corr_urg})",
        f"",
        f"Scores:",
        f"  Referral decision:      {r_ref:.2f}/1.00",
        f"  Urgency:                {r_urg:.2f}/1.00",
        f"  Primary concern:        {r_concern:.2f}/1.00",
        f"  Information gathering:  {r_info:.2f}/1.00",
        f"",
    ]

    if not asked_question:
        lines.append("WARNING: You did not ask any clarifying questions before deciding.")
        lines.append("         A good ASHA worker gathers information before referring.")
        lines.append("")

    lines.append(f"Clinical explanation:")
    lines.append(case.explanation)

    if case.key_danger_signs:
        lines.append(f"")
        lines.append(f"Key danger signs in this case: {', '.join(case.key_danger_signs)}")

    return "\n".join(lines)


def grade_doctor_action(
    disposition: str,
    case: ClinicalCase,
    asha_score: float,
) -> float:
    """
    Score the PHC Doctor's disposition decision.

    Rules:
    - REFER_IMMEDIATELY cases → correct doctor decision is "refer_to_fru"
    - REFER_WITHIN_24H cases → "manage_at_phc" or "refer_to_fru" both acceptable
    - TREAT_AT_HOME/MONITOR cases → "manage_at_phc" is correct

    Returns a score 0.001-0.999.
    """
    disposition_norm = disposition.lower().strip()
    correct = getattr(case, 'correct_doctor_decision', 'manage_at_phc')

    # Direct match
    if disposition_norm == correct:
        base = 1.0
    elif case.correct_referral == "REFER_IMMEDIATELY":
        # Must refer to FRU — staying at PHC is dangerous
        if disposition_norm == "refer_to_fru":
            base = 1.0
        elif disposition_norm == "refer_to_district":
            base = 0.8  # over-referring is safer than under-referring
        else:  # manage_at_phc
            base = 0.1  # dangerous undertriage
    elif case.correct_referral == "REFER_WITHIN_24H":
        # Either manage at PHC or refer to FRU is acceptable
        if disposition_norm in ("manage_at_phc", "refer_to_fru"):
            base = 1.0
        else:
            base = 0.5
    else:
        # TREAT_AT_HOME or MONITOR — PHC can manage
        if disposition_norm == "manage_at_phc":
            base = 1.0
        elif disposition_norm == "refer_to_fru":
            base = 0.5  # unnecessary referral
        else:
            base = 0.3

    # Doctor score is influenced by ASHA's score (information quality)
    # Poor ASHA handoff = harder for doctor to make good decision
    asha_weight = 0.85 + 0.15 * asha_score
    score = base * asha_weight

    return max(0.001, min(0.999, score))
