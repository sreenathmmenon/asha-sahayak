"""
ASHA Sahayak — Deterministic Clinical Tools
All ground truth from published NHM/IMNCI guidelines. No external APIs.
"""
from __future__ import annotations
import json
from datetime import date, datetime, timedelta


def muac_classifier(age_months: int, muac_mm: int, bilateral_edema: bool = False) -> dict:
    """
    Classify Mid-Upper Arm Circumference for nutritional status.
    Per NHM SAM Operational Guidelines and WHO standards.
    MUAC < 115mm = SAM, 115-125mm = MAM, > 125mm = Normal.
    Bilateral pitting edema always = Kwashiorkor SAM regardless of MUAC.
    """
    if bilateral_edema:
        return {
            "classification": "SAM",
            "subtype": "Kwashiorkor",
            "referral": "refer_nrc_immediately",
            "note": "Bilateral pitting edema = Kwashiorkor SAM regardless of MUAC measurement.",
            "source": "NHM SAM Operational Guidelines"
        }
    if muac_mm < 115:
        return {
            "classification": "SAM",
            "muac_mm": muac_mm,
            "threshold": "< 115mm",
            "referral": "refer_nrc",
            "management": "facility_or_community_based_depending_on_complications",
            "next_step": "Check for complications: fever, oedema, appetite test",
            "source": "NHM SAM Operational Guidelines"
        }
    elif muac_mm < 125:
        return {
            "classification": "MAM",
            "muac_mm": muac_mm,
            "threshold": "115-125mm",
            "referral": "community_management",
            "management": "enroll_supplementary_nutrition_programme",
            "next_step": "Monthly MUAC monitoring, ICDS anganwadi referral",
            "source": "NHM SAM Operational Guidelines"
        }
    else:
        return {
            "classification": "Normal",
            "muac_mm": muac_mm,
            "threshold": "> 125mm",
            "referral": "monitor_monthly",
            "management": "continue_routine_monitoring",
            "source": "NHM SAM Operational Guidelines"
        }


def gestational_age_calculator(lmp_date: str, current_date: str = None) -> dict:
    """
    Calculate gestational age and estimated due date from Last Menstrual Period.
    Uses Naegele's Rule: EDD = LMP + 280 days.
    Input: lmp_date in YYYY-MM-DD format.
    """
    try:
        lmp = datetime.strptime(lmp_date, "%Y-%m-%d").date()
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD. Example: 2025-08-15"}

    today = date.today() if current_date is None else datetime.strptime(current_date, "%Y-%m-%d").date()
    weeks = (today - lmp).days // 7
    days_extra = (today - lmp).days % 7
    edd = lmp + timedelta(days=280)
    trimester = "1st" if weeks < 13 else ("2nd" if weeks < 28 else "3rd")

    risk_notes = []
    if weeks < 0:
        risk_notes.append("ERROR: LMP date is in the future. Please verify.")
    elif weeks < 37:
        risk_notes.append(f"PRETERM — {37 - weeks} weeks before term. Monitor closely.")
    if weeks >= 40:
        risk_notes.append("POST-DATES — refer to PHC for assessment. Risk of stillbirth increases.")
    if weeks >= 28:
        risk_notes.append("3rd trimester: pre-eclampsia screening and fetal movement monitoring critical.")
    if weeks >= 20:
        anc_visits = []
        if weeks >= 20: anc_visits.append("20-week ANC")
        if weeks >= 26: anc_visits.append("26-week ANC")
        if weeks >= 30: anc_visits.append("30-week ANC")
        if weeks >= 34: anc_visits.append("34-week ANC")
        if anc_visits:
            risk_notes.append(f"ANC contacts due: {', '.join(anc_visits)}")

    return {
        "gestational_age_weeks": weeks,
        "gestational_age_days": days_extra,
        "trimester": trimester,
        "edd": str(edd),
        "risk_notes": risk_notes,
        "source": "Naegele's Rule / WHO 8-contact ANC Model"
    }


def drug_dosage_calculator(drug_name: str, weight_kg: float) -> dict:
    """
    Calculate safe pediatric drug dosage by patient weight.
    Per IMNCI Drug Formulary, Government of India.
    Drugs: amoxicillin, cotrimoxazole, paracetamol, zinc, ors_sachets.
    """
    drug_name = drug_name.lower().strip().replace("-", "_").replace(" ", "_")

    FORMULARY = {
        "amoxicillin": {
            "dose_mg_per_kg": 40,
            "frequency": "3x daily (every 8 hours)",
            "duration_days": 5,
            "formulation": "125mg/5ml suspension",
            "dose_ml": lambda w: round(w * 40 / 25, 1),
            "notes": "Give with food. For pneumonia — complete full 5-day course.",
            "concentration_mg_per_ml": 25,
        },
        "cotrimoxazole": {
            "dose_mg_per_kg": 4,
            "frequency": "2x daily (every 12 hours)",
            "duration_days": 5,
            "formulation": "240mg/5ml suspension (TMP+SMX combined)",
            "dose_ml": lambda w: round(w * 4 / 48, 1),
            "notes": "TMP component 4mg/kg. For severe infection with allergy to amoxicillin.",
            "concentration_mg_per_ml": 48,
        },
        "paracetamol": {
            "dose_mg_per_kg": 15,
            "frequency": "every 6 hours as needed (maximum 4 doses/day)",
            "duration_days": 3,
            "formulation": "120mg/5ml syrup",
            "dose_ml": lambda w: round(w * 15 / 24, 1),
            "notes": "Only for fever/pain relief. Do not use for more than 3 days without review.",
            "concentration_mg_per_ml": 24,
        },
        "zinc": {
            "dose_mg_per_kg": None,
            "frequency": "once daily",
            "duration_days": 14,
            "formulation": "20mg dispersible tablet (above 6 months) / 10mg (under 6 months)",
            "dose_ml": lambda w: "20mg tablet if above 6 months, 10mg if under 6 months",
            "notes": "Always give zinc with ORS in diarrhea. Dissolve in small amount of water.",
            "concentration_mg_per_ml": None,
        },
        "ors_sachets": {
            "dose_mg_per_kg": None,
            "frequency": "after every loose stool and as maintenance",
            "duration_days": 5,
            "formulation": "ORS sachet dissolved in 200ml clean water",
            "dose_ml": lambda w: f"{round(w * 75)}ml in first 4 hours, then {round(w * 10)}ml per loose stool",
            "notes": "Teach mother to make ORS. Use clean water only. Continue until diarrhea stops.",
            "concentration_mg_per_ml": None,
        },
    }

    if drug_name not in FORMULARY:
        return {
            "error": f"Drug '{drug_name}' not in ASHA formulary.",
            "available_drugs": list(FORMULARY.keys()),
            "note": "Contact PHC Medical Officer for drugs outside ASHA formulary."
        }

    entry = FORMULARY[drug_name]
    dose_result = entry["dose_ml"](weight_kg)

    return {
        "drug": drug_name,
        "weight_kg": weight_kg,
        "dose": dose_result,
        "frequency": entry["frequency"],
        "duration_days": entry["duration_days"],
        "formulation": entry["formulation"],
        "notes": entry["notes"],
        "source": "IMNCI Drug Formulary, Government of India"
    }


def jssk_eligibility_checker(patient_type: str) -> dict:
    """
    Check JSSK (Janani Shishu Suraksha Karyakram) government entitlements.
    All pregnant women delivering in government facility and their newborns are eligible.
    Source: NHM JSSK Circular 2011, extended 2014.
    """
    patient_type = patient_type.lower().strip().replace("-", "_").replace(" ", "_")

    ENTITLEMENTS = {
        "pregnant_woman": {
            "eligible": True,
            "free_services": [
                "normal_delivery",
                "caesarean_section",
                "medicines_and_consumables",
                "diagnostics_and_lab_tests",
                "blood_transfusion",
                "diet_3_days_after_normal_delivery",
                "diet_7_days_after_caesarean",
                "transport_home_to_government_facility",
                "drop_back_home_after_delivery",
                "free_referral_transport_between_government_facilities",
            ],
            "transport_number": "108",
            "transport_note": "Call 108 for FREE ambulance. Available 24/7. No payment required.",
            "coverage": "ALL pregnant women delivering in any government health facility. No BPL card needed.",
            "source": "NHM JSSK Circular, 2011 (extended 2014)"
        },
        "newborn": {
            "eligible": True,
            "free_services": [
                "treatment_up_to_30_days_in_government_facility",
                "medicines_and_consumables",
                "diagnostics",
                "blood_transfusion",
                "transport_to_sncu_or_nicu",
            ],
            "transport_number": "108",
            "transport_note": "Call 108 for sick newborn transport. Free of charge.",
            "coverage": "All sick newborns up to 30 days old in government facility.",
            "source": "NHM JSSK Circular, extended 2014"
        },
        "sick_infant": {
            "eligible": True,
            "free_services": [
                "inpatient_treatment",
                "medicines",
                "diagnostics",
                "blood",
                "transport",
            ],
            "transport_number": "108",
            "transport_note": "Call 108 for sick infant transport.",
            "coverage": "Sick infants up to 1 year in government facility. State-specific variations may apply.",
            "source": "NHM JSSK Circular, extended 2014"
        },
    }

    if patient_type not in ENTITLEMENTS:
        return {
            "eligible": "unknown",
            "error": f"Patient type '{patient_type}' not recognized.",
            "valid_types": list(ENTITLEMENTS.keys()),
            "note": "When in doubt, call 108 and the ambulance staff will assist."
        }

    return ENTITLEMENTS[patient_type]


def cbac_ncd_scorer(
    age: int,
    tobacco_use: bool = False,
    alcohol_use: bool = False,
    family_history_diabetes: bool = False,
    family_history_hypertension: bool = False,
    family_history_heart_disease: bool = False,
    physical_activity: str = "moderate",
    known_bp_high: bool = False,
    known_diabetes: bool = False,
    waist_cm: float = 0,
) -> dict:
    """
    Score Community Based Assessment Checklist (CBAC) for NCD risk screening.
    Score >= 4 means refer to ANM/Health and Wellness Centre for clinical screening.
    Source: NHM CBAC guidelines, Ministry of Health and Family Welfare.
    """
    score = 0
    breakdown = []

    if age >= 30:
        score += 1
        breakdown.append("Age >= 30 years: +1")
    if tobacco_use:
        score += 2
        breakdown.append("Tobacco use: +2")
    if alcohol_use:
        score += 1
        breakdown.append("Alcohol use: +1")
    if family_history_diabetes:
        score += 1
        breakdown.append("Family history of diabetes: +1")
    if family_history_hypertension:
        score += 1
        breakdown.append("Family history of hypertension: +1")
    if family_history_heart_disease:
        score += 1
        breakdown.append("Family history of heart disease: +1")
    if physical_activity.lower() == "low":
        score += 1
        breakdown.append("Low physical activity: +1")
    if known_bp_high:
        score += 2
        breakdown.append("Known high blood pressure: +2")
    if known_diabetes:
        score += 2
        breakdown.append("Known diabetes: +2")
    if waist_cm > 90:
        score += 1
        breakdown.append(f"Abdominal obesity (waist {waist_cm}cm > 90cm): +1")

    risk_level = "low" if score < 4 else ("moderate" if score < 7 else "high")
    refer = score >= 4

    return {
        "cbac_score": score,
        "max_score": 12,
        "risk_level": risk_level,
        "refer_to_anm": refer,
        "score_breakdown": breakdown,
        "recommended_action": (
            "REFER to ANM or Health and Wellness Centre for BP and glucose screening"
            if refer else
            "Reassess annually. Counsel on healthy lifestyle."
        ),
        "source": "NHM CBAC Form, Ministry of Health and Family Welfare"
    }
