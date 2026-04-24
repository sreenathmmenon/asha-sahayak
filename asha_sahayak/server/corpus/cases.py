"""
ASHA Sahayak — Clinical Case Library
21 cases derived from official Indian Government IMNCI protocol,
NHM maternal health guidelines, and JSSK scheme referral criteria.

Ground truth for all cases is deterministic and backed by published protocol.
Sources:
  - WHO/NHM IMNCI Chart Booklet (nhm.gov.in)
  - MOHFW Maternal Health Guidelines
  - NTEP TB Referral Guidelines

Each case has:
  - symptom_groups: dict of named groups revealed turn-by-turn
  - initial_presentation: what ASHA worker says on turn 1
  - followup_responses: responses keyed by question keywords
  - ground_truth: correct referral decision + urgency + primary_concern
  - difficulty: easy | medium | hard
  - explanation: why this is the correct answer (for feedback)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClinicalCase:
    case_id: str
    title: str
    difficulty: str          # easy | medium | hard
    category: str            # pediatric | neonatal | maternal | tb

    # Patient metadata
    age_description: str
    gender: str
    location: str
    malaria_risk_area: bool
    season: str

    # Conversation structure
    initial_presentation: str   # ASHA worker's opening message (Turn 1)

    # Keyword → response mapping for clarifying questions
    # Agent's question is matched against keywords (lowercase)
    followup_responses: Dict[str, str]

    # Ground truth (deterministic, from IMNCI protocol)
    correct_referral: str       # REFER_IMMEDIATELY | REFER_WITHIN_24H | TREAT_AT_HOME | MONITOR
    correct_urgency: str        # immediate | within_24h | routine | monitor
    correct_primary_concern: str
    correct_action_items: List[str]

    # Partial credit map — near-miss referral decisions and their scores
    # referral_decision -> score (0.0-1.0). Exact match always = 1.0.
    near_miss_referrals: Dict[str, float] = field(default_factory=dict)

    # Key danger signs that MUST be surfaced (agent must ask about or identify)
    key_danger_signs: List[str] = field(default_factory=list)

    # Explanation shown as feedback after episode
    explanation: str = ""

    # Optional: terminal message when done=True
    terminal_message: str = ""

    # Tool use and multi-agent context
    available_tools: List[str] = field(default_factory=list)
    tool_required_for_full_score: bool = False
    phc_manageable: bool = True
    correct_doctor_decision: str = "manage_at_phc"


# ---------------------------------------------------------------------------
# EASY CASES — Clear-cut decisions, one dominant danger sign cluster
# ---------------------------------------------------------------------------

CASE_E01 = ClinicalCase(
    case_id="E01",
    title="Severe Pneumonia — Chest Indrawing",
    difficulty="easy",
    category="pediatric",
    age_description="18 months old",
    gender="female",
    location="rural_bihar",
    malaria_risk_area=False,
    season="monsoon",
    initial_presentation=(
        "Didi, ek bachchi hai, 18 mahine ki. Teen din se khansi hai, bukhar bhi hai. "
        "Saansh lene mein takleef ho rahi hai."
        "\n[Sister, there is a girl, 18 months old. She has had cough for 3 days, also fever. "
        "She is having difficulty breathing.]"
    ),
    followup_responses={
        "breathing": "Breathing very fast. Lower chest going in when she breathes — I can see it clearly.",
        "chest": "Yes, lower chest wall going in with each breath. Chest indrawing present.",
        "fever": "Fever 39°C. Started 3 days ago.",
        "rate": "I counted — about 58 breaths per minute.",
        "respiratory": "Rate is 58 per minute. Chest indrawing visible.",
        "drink": "She is drinking, but with difficulty.",
        "danger": "She is alert but breathing very fast with chest indrawing.",
        "convulsion": "No fits or convulsions.",
        "conscious": "Child is conscious, not lethargic.",
        "vomit": "Some vomiting but not everything.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="severe_pneumonia",
    correct_action_items=["first_dose_antibiotic_before_transfer", "transport_to_hospital", "keep_warm"],
    near_miss_referrals={"REFER_WITHIN_24H": 0.4},
    key_danger_signs=["chest_indrawing", "fast_breathing"],
    explanation=(
        "IMNCI: Chest indrawing = SEVERE PNEUMONIA → REFER IMMEDIATELY. "
        "Respiratory rate 58/min (>50 for age) confirms fast breathing. "
        "Give first dose cotrimoxazole before transfer."
    ),
    terminal_message="Correct! Chest indrawing is a SEVERE PNEUMONIA sign. Refer immediately with first-dose antibiotic.",
)

CASE_E02 = ClinicalCase(
    case_id="E02",
    title="Eclampsia — Seizure in Pregnancy",
    difficulty="easy",
    category="maternal",
    age_description="8 months pregnant",
    gender="female",
    location="rural_up",
    malaria_risk_area=False,
    season="winter",
    initial_presentation=(
        "Didi, ek aurat hai, 8 mahine ki pregnant. Abhi unhe fits aa gayi — haath pair kaanp rahe the, "
        "2 minute tak. Ab hosh hai lekin confused hain. Pehle se BP high tha."
        "\n[Sister, a woman is 8 months pregnant. She just had a fit — hands and feet were shaking "
        "for 2 minutes. She is conscious now but confused. She had high BP before.]"
    ),
    followup_responses={
        "bp": "Blood pressure was 162/105 last week at PHC.",
        "headache": "Yes, she had very bad headache since morning — she said worst headache of her life.",
        "vision": "She said her vision was blurry before the fit. Seeing spots.",
        "conscious": "She is conscious but very confused after the fit.",
        "bleeding": "No vaginal bleeding.",
        "convulsion": "Yes, fit lasted about 2 minutes. Full body shaking.",
        "swelling": "Feet and hands are swollen since 2 days.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="eclampsia",
    correct_action_items=[
        "call_ambulance_108",
        "position_on_side_to_prevent_aspiration",
        "do_not_put_anything_in_mouth",
        "alert_phc_before_arrival",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.2},
    key_danger_signs=["convulsion_in_pregnancy", "hypertension"],
    explanation=(
        "ECLAMPSIA: Seizure in pregnancy = EMERGENCY. "
        "Call ambulance immediately. Position on side (recovery position). "
        "Do NOT put anything in mouth. Alert facility before arrival. "
        "Magnesium sulphate at facility."
    ),
    terminal_message="Correct! Eclampsia (seizure in pregnancy) is a life-threatening emergency — call 108 immediately.",
)

CASE_E03 = ClinicalCase(
    case_id="E03",
    title="Pneumonia — Treat at Home",
    difficulty="easy",
    category="pediatric",
    age_description="3 years old",
    gender="male",
    location="rural_mp",
    malaria_risk_area=False,
    season="winter",
    initial_presentation=(
        "Bhaiya ka beta hai, 3 saal ka. 2 din se khansi hai, bukhar bhi hai."
        "\n[The neighbour's son is 3 years old. Has had cough for 2 days, also fever.]"
    ),
    followup_responses={
        "breathing": "Breathing fast — I think faster than normal.",
        "rate": "I counted — about 42 breaths per minute.",
        "chest": "No, chest is not going in. Normal breathing except fast.",
        "indrawing": "No chest indrawing visible.",
        "danger": "Child is alert, playful, eating and drinking normally.",
        "drink": "Drinking fine, no vomiting.",
        "fever": "Temperature is 38.2°C.",
        "convulsion": "No fits.",
        "stridor": "No unusual sound when breathing.",
    },
    correct_referral="TREAT_AT_HOME",
    correct_urgency="routine",
    correct_primary_concern="pneumonia_no_severe_signs",
    correct_action_items=["oral_antibiotics_5_days", "follow_up_in_2_days", "counsel_danger_signs"],
    near_miss_referrals={"REFER_WITHIN_24H": 0.5, "MONITOR": 0.3},
    key_danger_signs=["fast_breathing"],
    explanation=(
        "IMNCI: Fast breathing (42/min ≥40 for age 12-59mo) WITHOUT chest indrawing = PNEUMONIA. "
        "Treat at home with oral antibiotics (amoxicillin/cotrimoxazole) for 5 days. "
        "Follow up in 2 days. No chest indrawing = NOT severe."
    ),
    terminal_message="Correct! Fast breathing without chest indrawing = Pneumonia treated at home with antibiotics.",
)

CASE_E04 = ClinicalCase(
    case_id="E04",
    title="Neonatal Danger Signs — Suspected Sepsis",
    difficulty="easy",
    category="neonatal",
    age_description="7 days old newborn",
    gender="male",
    location="rural_rajasthan",
    malaria_risk_area=False,
    season="summer",
    initial_presentation=(
        "Didi ek naya janam hua bacha hai, 7 din ka. Maa bol rahi hai bachcha dhang se doodh nahi pi raha, "
        "bahut kamzor lag raha hai."
        "\n[Sister, there is a newborn baby, 7 days old. Mother says baby is not feeding properly, looks very weak.]"
    ),
    followup_responses={
        "feed": "Very weak suck. Taking very little milk. Keeps falling asleep during feeding.",
        "fever": "Yes, I checked — temperature is 38.3°C (axillary).",
        "breathing": "Breathing fast — more than normal. Maybe 64 breaths per minute.",
        "color": "Lips look slightly bluish. Not fully normal color.",
        "cord": "Umbilical cord has slight redness around it. No pus yet.",
        "conscious": "Baby is difficult to rouse — very sleepy, not responding normally.",
        "cry": "Weak cry. Not the normal strong cry.",
        "movement": "Not moving much spontaneously.",
        "convulsion": "No fits seen.",
        "jaundice": "Skin looks slightly yellow.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="neonatal_sepsis",
    correct_action_items=[
        "transport_to_fru_or_district_hospital",
        "keep_baby_warm_skin_to_skin",
        "do_not_delay_for_any_home_treatment",
        "alert_sncu_before_arrival",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.3},
    key_danger_signs=["neonatal_fever", "poor_feeding", "lethargy", "fast_breathing"],
    explanation=(
        "NEONATAL DANGER SIGNS: Fever (38.3°C) + lethargy + fast breathing (64/min, >60) + poor feeding = "
        "SUSPECTED NEONATAL SEPSIS. All neonatal fever is serious. Refer immediately to SNCU. "
        "Keep warm (skin-to-skin). Do not delay."
    ),
    terminal_message="Correct! Multiple neonatal danger signs = immediate referral to SNCU. Neonatal fever is always serious.",
)

CASE_E05 = ClinicalCase(
    case_id="E05",
    title="Mild Diarrhea — Home Care",
    difficulty="easy",
    category="pediatric",
    age_description="4 years old",
    gender="male",
    location="rural_mp",
    malaria_risk_area=False,
    season="monsoon",
    initial_presentation=(
        "4 saal ke bachche ko 2 din se dast ho raha hai. Maa pareshan hai."
        "\n[A 4-year-old child has been having diarrhea for 2 days. Mother is worried.]"
    ),
    followup_responses={
        "drink": "Yes, drinking normally. Thirsty but not excessively.",
        "urine": "Yes, passing urine today — 2-3 times.",
        "eyes": "Eyes look normal, not sunken.",
        "skin": "Skin pinch goes back quickly, within 1 second.",
        "conscious": "Child is playful, running around. Alert.",
        "lethargy": "No — very active.",
        "blood": "No blood in stool. Watery but no blood.",
        "duration": "2 days only. Not more than 2 weeks.",
        "vomit": "Mild vomiting once, not everything.",
    },
    correct_referral="TREAT_AT_HOME",
    correct_urgency="routine",
    correct_primary_concern="diarrhea_no_dehydration",
    correct_action_items=["oral_rehydration_ors", "continue_breastfeeding", "continue_normal_foods", "counsel_danger_signs"],
    near_miss_referrals={"MONITOR": 0.5, "REFER_WITHIN_24H": 0.3},
    key_danger_signs=[],
    explanation=(
        "IMNCI: No dehydration signs (alert, normal eyes, drinks normally, fast skin pinch, urinating). "
        "Treat at home with ORS, continue feeds. "
        "Return if: lethargy, sunken eyes, unable to drink, blood in stool."
    ),
    terminal_message="Correct! No dehydration signs — treat at home with ORS and normal feeding.",
)


# ---------------------------------------------------------------------------
# MEDIUM CASES — Require clarifying questions, partial symptom revelation
# ---------------------------------------------------------------------------

CASE_M01 = ClinicalCase(
    case_id="M01",
    title="Pre-eclampsia — Severe Features",
    difficulty="medium",
    category="maternal",
    age_description="6 months pregnant",
    gender="female",
    location="tribal_jharkhand",
    malaria_risk_area=True,
    season="monsoon",
    initial_presentation=(
        "Didi, ek aurat hai 6 mahine pregnant. Bahut sar dard hai subah se. "
        "Aaram karne ke baad bhi theek nahi hui."
        "\n[Sister, a woman is 6 months pregnant. She has had very bad headache since morning. "
        "Did not go away even after rest.]"
    ),
    followup_responses={
        "vision": "Yes! She is saying her vision is blurry. Seeing spots in front of eyes.",
        "bp": "I checked with the BP machine — 145/95.",
        "swelling": "Yes, hands and face are swollen since yesterday. Feet also swollen.",
        "convulsion": "No fits yet. She is conscious.",
        "bleeding": "No vaginal bleeding.",
        "fever": "No fever. Temperature normal.",
        "vomit": "Some nausea and vomiting since morning.",
        "pain": "Some pain in upper right abdomen she said.",
        "urine": "Less urine since morning she says.",
        "fetal": "Baby is moving, she says.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="pre_eclampsia_severe_features",
    correct_action_items=[
        "transport_to_fru_within_hours",
        "do_not_leave_alone",
        "identify_blood_donors",
        "alert_phc_maternity_ward",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.5},
    key_danger_signs=["severe_headache", "blurred_vision", "hypertension", "edema"],
    explanation=(
        "SEVERE PRE-ECLAMPSIA: Headache + blurred vision + BP 145/95 + edema = high risk of eclampsia. "
        "Refer immediately (hours, not days). Magnesium sulphate prophylaxis at facility. "
        "Vision changes + headache = imminent seizure risk. Do not delay."
    ),
    terminal_message="Correct! Headache + blurred vision + high BP = severe pre-eclampsia. Refer immediately.",
)

CASE_M02 = ClinicalCase(
    case_id="M02",
    title="Severe Dehydration with Lethargy",
    difficulty="medium",
    category="pediatric",
    age_description="8 months old",
    gender="female",
    location="rural_up",
    malaria_risk_area=False,
    season="summer",
    initial_presentation=(
        "8 mahine ki bachchi ko 4 din se bahut dast ho raha hai. "
        "Maa bol rahi hai ab bahut kamzor ho gayi hai."
        "\n[An 8-month-old girl has had severe diarrhea for 4 days. "
        "Mother says she has become very weak now.]"
    ),
    followup_responses={
        "conscious": "Baby is very difficult to rouse — keeps falling asleep. Not responding normally when I call her.",
        "lethargy": "Yes, lethargic. Very difficult to wake up.",
        "eyes": "Eyes are sunken — clearly sunken.",
        "drink": "She is not drinking — when I give water she pushes it away, very weak.",
        "skin": "I pressed the skin on abdomen — it went back very slowly, more than 2 seconds.",
        "urine": "Mother says no urine since morning — 6 hours.",
        "fever": "Slight fever — 37.8°C.",
        "vomit": "Vomiting everything given.",
        "breathing": "Breathing fast also.",
        "cry": "Weak cry.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="severe_dehydration_with_lethargy",
    correct_action_items=[
        "iv_fluids_at_facility_needed",
        "transport_immediately",
        "do_not_give_oral_fluids_if_vomiting_everything",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.35},
    key_danger_signs=["lethargy", "sunken_eyes", "unable_to_drink", "slow_skin_pinch"],
    explanation=(
        "IMNCI: Lethargy = GENERAL DANGER SIGN → always refer immediately. "
        "Severe dehydration: lethargy + sunken eyes + unable to drink + slow skin pinch (4 signs). "
        "IV fluids needed at facility. Oral ORS not safe if vomiting everything."
    ),
    terminal_message="Correct! Lethargy is a general danger sign — always means immediate referral.",
)

CASE_M03 = ClinicalCase(
    case_id="M03",
    title="TB Suspect — Referral for Sputum",
    difficulty="medium",
    category="tb",
    age_description="35 years old adult",
    gender="male",
    location="urban_slum_delhi",
    malaria_risk_area=False,
    season="winter",
    initial_presentation=(
        "Ek aadmi hai, 35 saal ka. 3 hafte se khansi hai. "
        "Kaafi kamzor bhi ho gaya hai pichle kuch mahine mein."
        "\n[There is a man, 35 years old. He has had a cough for 3 weeks. "
        "He has also become quite weak over the past few months.]"
    ),
    followup_responses={
        "duration": "Cough for 3 weeks. Not getting better.",
        "blood": "Yes — twice he coughed blood. Small amount but there was blood.",
        "fever": "Mild fever in the evenings. Not very high but regular.",
        "night_sweat": "Yes, night sweats — wakes up sweating at night regularly.",
        "weight": "Lost weight — his clothes are loose now. Maybe 4-5 kg in 2 months.",
        "appetite": "No appetite. Not eating properly.",
        "contact": "His neighbor had TB last year and was on treatment.",
        "hiv": "Not known. Has not tested.",
        "breathless": "Some breathlessness on exertion.",
        "chest_pain": "Yes, chest pain sometimes.",
    },
    correct_referral="REFER_WITHIN_24H",
    correct_urgency="within_24h",
    correct_primary_concern="presumptive_tuberculosis",
    correct_action_items=[
        "refer_to_dm_tu_tb_unit",
        "do_not_start_antibiotics_before_testing",
        "sputum_sample_needed",
        "contact_tracing_household",
    ],
    near_miss_referrals={"REFER_IMMEDIATELY": 0.6, "MONITOR": 0.2},
    key_danger_signs=["cough_3_weeks", "hemoptysis", "weight_loss", "night_sweats"],
    explanation=(
        "NTEP: Presumptive TB — cough ≥2 weeks + weight loss + night sweats + hemoptysis + TB contact. "
        "Refer to TB unit for sputum test (CBNAAT/sputum smear). "
        "Do NOT start antibiotics before testing. Contact tracing required."
    ),
    terminal_message="Correct! Cough 3 weeks + hemoptysis + weight loss = Presumptive TB. Refer for sputum testing.",
)

CASE_M04 = ClinicalCase(
    case_id="M04",
    title="Some Dehydration — PHC Referral",
    difficulty="medium",
    category="pediatric",
    age_description="2 years old",
    gender="male",
    location="rural_maharashtra",
    malaria_risk_area=False,
    season="summer",
    initial_presentation=(
        "2 saal ka bacha. 2 din se dast hai. Bahut restless ho gaya hai."
        "\n[2-year-old child. Has had diarrhea for 2 days. Became very restless.]"
    ),
    followup_responses={
        "drink": "Drinks eagerly — as soon as you offer water he grabs it.",
        "eyes": "Eyes look slightly sunken — not normal.",
        "skin": "Skin pinch — goes back in about 1.5 seconds. Slower than normal.",
        "conscious": "Not lethargic — restless and irritable actually, crying frequently.",
        "lethargy": "No, opposite — very irritable and restless.",
        "urine": "Passed urine in last 4 hours — small amount.",
        "vomit": "Vomiting once, not everything.",
        "blood": "No blood in stool.",
        "fever": "Low grade — 37.5°C.",
    },
    correct_referral="REFER_WITHIN_24H",
    correct_urgency="within_24h",
    correct_primary_concern="some_dehydration",
    correct_action_items=[
        "oral_rehydration_at_phc",
        "monitor_hydration_status",
        "continue_breastfeeding",
        "return_if_worsens",
    ],
    near_miss_referrals={"TREAT_AT_HOME": 0.4, "REFER_IMMEDIATELY": 0.4},
    key_danger_signs=["restlessness", "sunken_eyes", "drinks_eagerly"],
    explanation=(
        "IMNCI: SOME DEHYDRATION — restlessness + sunken eyes + drinks eagerly (3 signs in 'some' category). "
        "No lethargy or unable-to-drink = NOT severe. "
        "Refer to PHC for supervised ORS. Distinguish from severe: no lethargy is key."
    ),
    terminal_message="Correct! Restlessness + sunken eyes + eager drinking = Some dehydration. Refer to PHC for ORS.",
)

CASE_M05 = ClinicalCase(
    case_id="M05",
    title="Postpartum Hemorrhage",
    difficulty="medium",
    category="maternal",
    age_description="delivered 5 hours ago",
    gender="female",
    location="rural_odisha",
    malaria_risk_area=False,
    season="post_monsoon",
    initial_presentation=(
        "Didi, ek aurat ne 5 ghante pehle ghar pe delivery ki. "
        "Ab bahut zyada khoon aa raha hai — pad bhiga hua hai."
        "\n[Sister, a woman delivered at home 5 hours ago. "
        "Now there is heavy bleeding — pad is soaked.]"
    ),
    followup_responses={
        "amount": "Soaking 2-3 pads per hour. Very heavy.",
        "clots": "Yes, large blood clots — bigger than a 50 rupee coin.",
        "conscious": "She is conscious but feeling dizzy when she tries to sit up.",
        "pulse": "Pulse is fast — I can feel it racing.",
        "uterus": "Uterus feels soft when I press on the lower abdomen. Not firm.",
        "placenta": "Placenta was delivered. She says it came out.",
        "fever": "No fever yet.",
        "pain": "She has lower abdominal pain.",
        "pale": "She looks pale — lips are pale.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="postpartum_hemorrhage",
    correct_action_items=[
        "uterine_massage_immediately",
        "call_108_ambulance",
        "iv_access_if_trained",
        "identify_blood_donors_now",
        "alert_fru_before_arrival",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.2},
    key_danger_signs=["heavy_vaginal_bleeding_postpartum", "large_clots", "dizziness", "tachycardia", "soft_uterus"],
    explanation=(
        "PPH: Soaking ≥2 pads/hour + large clots + soft uterus (uterine atony) + dizziness + tachycardia. "
        "Death can occur within 2 hours. Uterine massage immediately. Call 108. "
        "IV fluids. Blood donors. Alert FRU."
    ),
    terminal_message="Correct! Postpartum hemorrhage — uterine massage + call 108 immediately. Can be fatal within 2 hours.",
)


# ---------------------------------------------------------------------------
# HARD CASES — Multi-symptom, ambiguous, cross-cutting danger signs
# ---------------------------------------------------------------------------

CASE_H01 = ClinicalCase(
    case_id="H01",
    title="Neonatal Hypothermia with Sepsis Signs",
    difficulty="hard",
    category="neonatal",
    age_description="3 days old newborn",
    gender="male",
    location="rural_up_winter",
    malaria_risk_area=False,
    season="winter",
    initial_presentation=(
        "3 din ka nawajanat bacha. Maa bol rahi hai bachcha thanda lag raha hai, "
        "doodh nahi pi raha theek se."
        "\n[3-day-old newborn. Mother says baby feels cold, not feeding properly.]"
    ),
    followup_responses={
        "temperature": "I measured — axillary temperature is 35.1°C. Below normal.",
        "cold": "Yes, body is cold especially the hands and feet.",
        "feed": "Very weak sucking. Takes very little, keeps letting go.",
        "cry": "Weak, thin cry — not the strong cry of a healthy baby.",
        "conscious": "Difficult to rouse. Very sleepy.",
        "movement": "Not moving arms and legs much on their own.",
        "breathing": "I think breathing is a bit fast.",
        "cord": "Umbilical cord has slight smell — maybe mild infection starting.",
        "delivery": "Home delivery. Baby was not dried immediately — took some time.",
        "skin": "Skin looks slightly yellow (jaundice) today.",
        "color": "Slight blue around lips when feeding.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="neonatal_hypothermia_with_sepsis_risk",
    correct_action_items=[
        "warm_baby_skin_to_skin_kangaroo_care",
        "transport_to_sncu_immediately",
        "keep_warm_during_transport_wrap_in_cloth",
        "do_not_bathe_baby",
        "alert_sncu",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.3},
    key_danger_signs=["hypothermia", "poor_feeding", "lethargy", "cord_infection_risk"],
    explanation=(
        "NEONATAL HYPOTHERMIA (<35.5°C) + multiple danger signs (poor feeding, lethargy, weak cry, cord smell). "
        "Per 1°C drop in neonatal temp: sepsis risk +11%, death risk +28%. "
        "Skin-to-skin immediately + transport to SNCU. Do not delay. "
        "Cord smell = early omphalitis risk → makes referral more urgent."
    ),
    terminal_message="Correct! Hypothermia + poor feeding + lethargy = immediate SNCU referral. Skin-to-skin while transporting.",
)

CASE_H02 = ClinicalCase(
    case_id="H02",
    title="Severe Acute Malnutrition with Complications",
    difficulty="hard",
    category="pediatric",
    age_description="18 months old",
    gender="female",
    location="tribal_mp",
    malaria_risk_area=True,
    season="post_monsoon",
    initial_presentation=(
        "18 mahine ki bachchi. Maa bol rahi hai bahut patli ho gayi hai, "
        "kuch kha nahi rahi."
        "\n[18-month-old girl. Mother says she has become very thin and is not eating.]"
    ),
    followup_responses={
        "muac": "I measured MUAC — it is 10.8 cm.",
        "edema": "Yes! Both feet are puffy — when I press for 3 seconds, indent stays.",
        "conscious": "She is lethargic — not responding normally. Difficult to rouse.",
        "fever": "Fever — 38.2°C.",
        "appetite": "No appetite. When I gave her peanut paste for appetite test she refused.",
        "breathing": "Breathing seems fast.",
        "skin": "Skin on arms is loose, hanging. Visible muscle wasting.",
        "eyes": "Eyes look dull, no shine.",
        "infection": "She has a skin sore on left leg that looks infected.",
        "vomit": "Vomiting once today.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="severe_complicated_sam",
    correct_action_items=[
        "transport_to_nutritional_rehabilitation_centre",
        "do_not_give_high_calorie_food_at_home",
        "keep_warm",
        "alert_nrc",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.5},
    key_danger_signs=["muac_below_11_5", "bilateral_edema", "lethargy", "fever", "failed_appetite_test"],
    explanation=(
        "SEVERE COMPLICATED SAM: MUAC 10.8 (<11.5) + bilateral pitting edema + lethargy + fever + failed appetite test. "
        "Complications (fever, lethargy) = requires facility-based NRC. "
        "Uncomplicated SAM can do community management — complicated SAM cannot. "
        "Do NOT give high-calorie food at home (risk of refeeding syndrome)."
    ),
    terminal_message="Correct! MUAC <11.5 + bilateral edema + complications = Severe complicated SAM → NRC referral.",
)

CASE_H03 = ClinicalCase(
    case_id="H03",
    title="Antepartum Haemorrhage — 3rd Trimester",
    difficulty="hard",
    category="maternal",
    age_description="7 months pregnant",
    gender="female",
    location="remote_rural_chhattisgarh",
    malaria_risk_area=True,
    season="monsoon",
    initial_presentation=(
        "7 mahine pregnant aurat. Achanak khoon aa raya hai neeche se. "
        "Pet mein dard bhi hai."
        "\n[7-month pregnant woman. Suddenly there is bleeding from below. "
        "She also has abdominal pain.]"
    ),
    followup_responses={
        "amount": "Bleeding is heavy — bright red blood, soaking clothes.",
        "pain": "Moderate abdominal pain. Came suddenly.",
        "conscious": "Conscious but dizzy. Feeling faint when standing.",
        "bp": "Blood pressure — I checked — 90/60. Very low.",
        "pulse": "Pulse racing — very fast.",
        "trauma": "No fall or injury.",
        "fetal": "She says she cannot feel baby moving now.",
        "fever": "No fever.",
        "previous": "This is her 3rd pregnancy. Previous 2 were normal.",
        "placenta": "No, placenta not delivered — she is still pregnant.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="antepartum_haemorrhage_shock",
    correct_action_items=[
        "call_108_ambulance_immediately",
        "lay_flat_elevate_legs",
        "no_internal_examination",
        "identify_blood_donors_now",
        "alert_fru_for_emergency_obstetric_care",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.1},
    key_danger_signs=["vaginal_bleeding_pregnancy", "abdominal_pain", "hypotension", "tachycardia", "reduced_fetal_movement"],
    explanation=(
        "ANTEPARTUM HAEMORRHAGE with HAEMORRHAGIC SHOCK: Bleeding + pain + BP 90/60 (hypotension) + tachycardia + dizziness. "
        "Possible placental abruption. Fetal distress (no movement). "
        "DO NOT do internal examination. Emergency C-section likely needed. "
        "Can deteriorate to irreversible shock in minutes. Blood donors NOW."
    ),
    terminal_message="Correct! Antepartum haemorrhage with shock signs — call 108 immediately, no internal exam, blood donors.",
)

CASE_H04 = ClinicalCase(
    case_id="H04",
    title="Very Severe Febrile Disease — Meningitis Signs",
    difficulty="hard",
    category="pediatric",
    age_description="2 years old",
    gender="female",
    location="tribal_jharkhand",
    malaria_risk_area=True,
    season="monsoon",
    initial_presentation=(
        "2 saal ki bachchi. 1 din se tez bukhar hai. "
        "Maa bol rahi hai bahut zyada roe hai aaj."
        "\n[2-year-old girl. Has had high fever for 1 day. "
        "Mother says she has been crying a lot today.]"
    ),
    followup_responses={
        "fever": "Temperature is 39.5°C. Very high.",
        "neck": "When I try to bend the neck forward — she resists and cries. Neck seems stiff.",
        "conscious": "She is very difficult to rouse. Lethargic.",
        "convulsion": "About 1 hour ago she had shaking of both arms for 30 seconds.",
        "fontanelle": "I checked — the soft spot on top of head is bulging.",
        "rash": "Small red spots on arms and legs — appeared in last few hours.",
        "drink": "Not drinking — refusing all fluids.",
        "vomit": "Vomiting twice — projectile.",
        "malaria": "No malaria tablet given recently. Area has malaria.",
        "vaccination": "Vaccination card — not fully vaccinated. Missed some.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="very_severe_febrile_disease_possible_meningitis",
    correct_action_items=[
        "transport_to_hospital_immediately",
        "first_dose_antibiotic_before_transfer",
        "do_not_delay_for_malaria_test",
        "alert_paediatric_ward",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.2},
    key_danger_signs=["stiff_neck", "bulging_fontanelle", "convulsion", "lethargy", "petechial_rash"],
    explanation=(
        "VERY SEVERE FEBRILE DISEASE: Stiff neck + bulging fontanelle + convulsion + lethargy + petechial rash. "
        "IMNCI: Stiff neck = very severe febrile disease, refer immediately. "
        "Possible meningococcal meningitis (petechial rash + stiff neck). "
        "Give first-dose antibiotic before transfer. Even in malaria area — do not wait for malaria test."
    ),
    terminal_message="Correct! Stiff neck + bulging fontanelle + petechial rash = possible meningitis. Refer immediately with antibiotics.",
)

CASE_H05 = ClinicalCase(
    case_id="H05",
    title="Omphalitis Progressing to Sepsis",
    difficulty="hard",
    category="neonatal",
    age_description="5 days old newborn",
    gender="female",
    location="rural_up",
    malaria_risk_area=False,
    season="summer",
    initial_presentation=(
        "5 din ki bachi. Naabhi mein se badbu aa rahi hai. "
        "Maa ne dekha hai kuch pus bhi aa raha hai."
        "\n[5-day-old girl. Foul smell coming from umbilical cord. "
        "Mother noticed some pus also coming from it.]"
    ),
    followup_responses={
        "smell": "Yes, foul smell from cord. Seropurulent discharge — yellow-white with some blood.",
        "redness": "Redness around the cord — spreading to skin of abdomen. Getting bigger.",
        "fever": "Temperature 38.1°C — mild fever.",
        "feed": "Feeding reduced since yesterday — not latching as well.",
        "conscious": "Becoming slightly less alert than before. Not as responsive.",
        "skin": "Redness spreading on abdomen around cord — about 3 cm radius now.",
        "cry": "Cry is becoming weaker.",
        "movement": "Moving limbs but less than before.",
        "delivery": "Home delivery. Cord was cut with blade — not sure if clean.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="omphalitis_with_systemic_spread",
    correct_action_items=[
        "transport_to_sncu_immediately",
        "do_not_apply_anything_on_cord",
        "iv_antibiotics_needed_at_facility",
        "alert_paediatric_surgeon_if_available",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.4},
    key_danger_signs=["cord_pus_foul_smell", "spreading_erythema", "fever", "reduced_feeding", "decreased_alertness"],
    explanation=(
        "OMPHALITIS with SYSTEMIC SIGNS: Pus + foul smell + spreading erythema + fever + reduced feeding + decreased alertness. "
        "Spreading redness = cellulitis progressing. Systemic signs = sepsis developing. "
        "Can progress to necrotising fasciitis rapidly. "
        "IV antibiotics needed urgently. Refer immediately. Do not apply any home remedies on cord."
    ),
    terminal_message="Correct! Spreading omphalitis with systemic signs = immediate referral. Risk of necrotising fasciitis.",
)

CASE_H06 = ClinicalCase(
    case_id="H06",
    title="Moderate Malnutrition — Community Management",
    difficulty="hard",
    category="pediatric",
    age_description="2 years old",
    gender="male",
    location="tribal_odisha",
    malaria_risk_area=True,
    season="post_monsoon",
    initial_presentation=(
        "2 saal ka beta. Maa bol rahi hai kuch mahine se thak sa raha hai, "
        "khana bhi kam khata hai."
        "\n[2-year-old boy. Mother says for some months he seems tired, eating less.]"
    ),
    followup_responses={
        "muac": "MUAC is 11.9 cm.",
        "edema": "No swelling of feet — I pressed, no indent.",
        "appetite": "He ate the peanut paste I gave for the appetite test — took it.",
        "conscious": "Alert, active. Playing with toys.",
        "fever": "No fever. Temperature 36.8°C.",
        "infection": "No visible infection.",
        "weight": "Weight 8.2 kg. Below normal for age.",
        "height": "Height — he looks shorter than other children his age.",
        "vaccination": "Vaccinations up to date.",
        "diarrhea": "No current diarrhea.",
    },
    correct_referral="MONITOR",
    correct_urgency="monitor",
    correct_primary_concern="moderate_acute_malnutrition",
    correct_action_items=[
        "enrol_in_anganwadi_supplementary_feeding",
        "dietary_counselling_mother",
        "muac_monitoring_every_2_weeks",
        "micronutrient_supplementation",
        "refer_if_muac_drops_below_11_5",
    ],
    near_miss_referrals={"TREAT_AT_HOME": 0.7, "REFER_WITHIN_24H": 0.4},
    key_danger_signs=[],
    explanation=(
        "MODERATE ACUTE MALNUTRITION (MAM): MUAC 11.9 cm (11.5-12.5 range) + no edema + passes appetite test + alert, no complications. "
        "NOT severe — does not need NRC referral. "
        "Community management: Anganwadi supplementary feeding + dietary counselling + micronutrients + MUAC monitoring every 2 weeks. "
        "Refer to NRC if MUAC drops below 11.5 cm."
    ),
    terminal_message="Correct! MAM (MUAC 11.5-12.5, no complications) = community management, not hospital referral.",
)


# ---------------------------------------------------------------------------
# EASY CASES (Phase 3) — E06, E07, E08
# ---------------------------------------------------------------------------

CASE_E06 = ClinicalCase(
    case_id="E06",
    title="Neonatal Jaundice — Day 3, Physiological Pattern",
    difficulty="easy",
    category="neonatal",
    age_description="Newborn, 3 days old",
    gender="male",
    location="Rural home",
    malaria_risk_area=False,
    season="winter",
    initial_presentation=(
        "Didi, mere bete ko halki peeli skin ho gayi hai. Teen din ka hai. "
        "Dudh pee raha hai achhe se. Aktiv hai."
        "\n[Sister, my baby boy has mild yellow skin. He is 3 days old. Feeding well. He is active.]"
    ),
    followup_responses={
        "yellow": "Face and chest are yellow, but not the abdomen or legs.",
        "face": "Yes, only face and chest are yellow.",
        "fever": "No fever at all.",
        "stool": "Yes, stools are normal — yellow colour, normal consistency.",
        "soles": "No, only face and chest. Palms and soles are normal colour.",
        "palms": "No, palms look normal — not yellow.",
        "feed": "Feeding well every 2-3 hours. Good latch.",
        "active": "Yes, very active, cries normally.",
    },
    correct_referral="MONITOR",
    correct_urgency="monitor",
    correct_primary_concern="neonatal_physiological_jaundice",
    correct_action_items=[
        "Ensure adequate breastfeeding every 2-3 hours",
        "Monitor jaundice spread - check palms and soles",
        "Return immediately if soles/palms become yellow",
        "Ensure adequate sunlight exposure in mornings",
        "Follow up in 2 days",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.3},
    key_danger_signs=[
        "Jaundice within 24 hours is pathological",
        "Yellow palms/soles = danger sign",
        "Poor feeding with jaundice = refer immediately",
    ],
    explanation=(
        "Physiological jaundice appearing on Day 2-3 and affecting only face and chest in a well, "
        "feeding newborn is NORMAL and does not require referral. The IMNCI protocol requires referral "
        "only if: (1) jaundice within 24 hours, (2) yellow palms and soles, (3) poor feeding, or (4) "
        "unwell baby. Teaching point: NOT all jaundice = danger. Over-referral of physiological jaundice "
        "wastes precious PHC resources."
    ),
    terminal_message="Physiological jaundice on Day 3, face/chest only, feeding well = MONITOR with follow-up.",
    tool_required_for_full_score=False,
    phc_manageable=True,
    correct_doctor_decision="manage_at_phc",
)

CASE_E07 = ClinicalCase(
    case_id="E07",
    title="Skin Pustules — Localized, < 10 Pustules",
    difficulty="easy",
    category="neonatal",
    age_description="Newborn, 8 days old",
    gender="female",
    location="Rural home",
    malaria_risk_area=False,
    season="summer",
    initial_presentation=(
        "Didi, bachchi ke body pe chhote chhote daane aa gaye hain. Bukhaar nahi hai. Naabhi theek hai."
        "\n[Sister, baby has small pustules on body. No fever. Umbilicus is fine.]"
    ),
    followup_responses={
        "how many": "5-6 pustules on chest only.",
        "count": "5-6 pustules on chest only.",
        "fever": "No fever.",
        "cord": "Cord looks clean, no redness.",
        "umbilicus": "Cord looks clean, no redness.",
        "feed": "Feeding normally.",
        "feeding": "Feeding normally.",
        "spread": "Only on chest, not spreading.",
    },
    correct_referral="TREAT_AT_HOME",
    correct_urgency="routine",
    correct_primary_concern="localized_skin_pustules",
    correct_action_items=[
        "Clean pustules with gentian violet",
        "Keep skin clean and dry",
        "Monitor for spread or increase beyond 10 pustules",
        "Return if fever develops or cord becomes red",
        "Monitor daily for 3 days",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.4},
    key_danger_signs=[
        "> 10 pustules = REFER_WITHIN_24H",
        "Cord involvement = omphalitis = REFER_IMMEDIATELY",
        "Fever with pustules = REFER_IMMEDIATELY",
    ],
    explanation=(
        "Skin pustules < 10 without fever, cord involvement, or spread = TREAT AT HOME with gentian violet. "
        "The critical threshold is 10 pustules. > 10 pustules require referral. Cord involvement (omphalitis) "
        "is a separate emergency requiring REFER_IMMEDIATELY. Teaching point: quantitative thresholds matter "
        "in neonatal care."
    ),
    terminal_message="6 localized pustules, no fever, no cord involvement = TREAT_AT_HOME with gentian violet.",
    phc_manageable=True,
    correct_doctor_decision="manage_at_phc",
)

CASE_E08 = ClinicalCase(
    case_id="E08",
    title="Uncomplicated Malaria — RDT Positive, No Danger Signs",
    difficulty="easy",
    category="malaria",
    age_description="5 years old",
    gender="male",
    location="High malaria risk village",
    malaria_risk_area=True,
    season="monsoon",
    initial_presentation=(
        "Didi, bete ko bukhaar hai teen din se. RDT test kiya, positive aaya. "
        "Khana kha raha hai, koi aur taklif nahi."
        "\n[Sister, son has fever for 3 days. RDT test done, positive. He is eating, no other complaints.]"
    ),
    followup_responses={
        "conscious": "Alert and responsive.",
        "consciousness": "Alert and responsive.",
        "vomit": "No vomiting.",
        "vomiting": "No vomiting.",
        "fits": "No fits.",
        "seizure": "No fits.",
        "drink": "Yes drinking fine.",
        "pallor": "Mild pallor on nails only.",
        "pale": "Mild pallor on nails only.",
        "danger": "No danger signs — alert, feeding, no fits.",
    },
    correct_referral="TREAT_AT_HOME",
    correct_urgency="routine",
    correct_primary_concern="uncomplicated_malaria",
    correct_action_items=[
        "Give ACT (Artemisinin Combination Therapy) as per weight",
        "Ensure full 3-day course completion",
        "Paracetamol for fever",
        "Return if any danger sign develops",
        "Follow up in 2 days to check fever clearance",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.3, "REFER_IMMEDIATELY": 0.1},
    key_danger_signs=[
        "Unconscious or convulsions = REFER_IMMEDIATELY (cerebral malaria)",
        "Unable to drink/vomiting = REFER_IMMEDIATELY",
        "Severe pallor = REFER_IMMEDIATELY",
        "P.falciparum with any danger sign = REFER_IMMEDIATELY",
    ],
    explanation=(
        "RDT-positive malaria without any danger signs = uncomplicated malaria = TREAT AT HOME with ACT "
        "per NVBDCP guidelines. Positive RDT alone does NOT mean REFER_IMMEDIATELY. Danger signs "
        "(unconscious, convulsions, severe pallor, unable to drink) change it to REFER_IMMEDIATELY. "
        "Teaching point: the diagnosis does not determine referral — the SEVERITY does."
    ),
    terminal_message="RDT positive, alert, feeding well, no danger signs = uncomplicated malaria = TREAT_AT_HOME with ACT.",
    phc_manageable=True,
    correct_doctor_decision="manage_at_phc",
)


# ---------------------------------------------------------------------------
# MEDIUM CASES (Phase 3) — M06-M10
# ---------------------------------------------------------------------------

CASE_M06 = ClinicalCase(
    case_id="M06",
    title="Neonatal Jaundice — Onset Within 24 Hours (Pathological)",
    difficulty="medium",
    category="neonatal",
    age_description="Newborn, 18 hours old",
    gender="female",
    location="Rural home delivery",
    malaria_risk_area=False,
    season="winter",
    initial_presentation=(
        "Didi, bachchi ka rang kaafi peela ho gaya hai. Sirf 18 ghante ki hai. Raat ko normal thi."
        "\n[Sister, baby's skin has become very yellow. She is only 18 hours old. Was normal last night.]"
    ),
    followup_responses={
        "when": "Before 24 hours, from birth nearly.",
        "onset": "Before 24 hours, from birth nearly.",
        "feed": "Feeding but sleepy.",
        "feeding": "Feeding but sleepy.",
        "spread": "Face and chest, spreading to abdomen.",
        "spreading": "Face and chest, spreading to abdomen.",
        "palms": "Not yet yellow.",
        "soles": "Not yet yellow.",
        "sleepy": "Yes, more sleepy than a normal newborn.",
    },
    correct_referral="REFER_WITHIN_24H",
    correct_urgency="within_24h",
    correct_primary_concern="neonatal_pathological_jaundice",
    correct_action_items=[
        "Refer to PHC for bilirubin measurement",
        "Continue breastfeeding every 2 hours during transport",
        "Monitor for spread to palms and soles (emergency)",
        "Document time of onset in referral slip",
        "Phototherapy likely needed at PHC",
    ],
    near_miss_referrals={"REFER_IMMEDIATELY": 0.7},
    key_danger_signs=[
        "Jaundice WITHIN 24 hours = always pathological",
        "Spreading rapidly = urgent",
        "Sleepy with jaundice = danger sign",
    ],
    explanation=(
        "Jaundice appearing within 24 hours of birth is ALWAYS pathological (usually hemolytic — Rh or ABO "
        "incompatibility). This requires referral within 24h for bilirubin measurement and likely phototherapy. "
        "Compare with E06: Day 3 physiological jaundice = MONITOR. The TIMING is the critical differentiator. "
        "IMNCI: any newborn jaundice within 24 hours = REFER_WITHIN_24H."
    ),
    terminal_message="Jaundice onset < 24 hours = pathological = REFER_WITHIN_24H for bilirubin and phototherapy.",
    phc_manageable=True,
    correct_doctor_decision="manage_at_phc",
)

CASE_M07 = ClinicalCase(
    case_id="M07",
    title="Gestational Diabetes Risk — 32yr, 26 Weeks Pregnant",
    difficulty="medium",
    category="maternal",
    age_description="32 years old",
    gender="female",
    location="Rural village",
    malaria_risk_area=False,
    season="summer",
    initial_presentation=(
        "Didi, meri bahan ko peshab zyada aa rahi hai, bahut pyaas lagti hai. "
        "26 hafton ki pregnant hai. Maa ko diabetes hai."
        "\n[Sister, my sister is urinating a lot, very thirsty. 26 weeks pregnant. Mother has diabetes.]"
    ),
    followup_responses={
        "anc": "Only 2 ANC done, blood sugar not tested.",
        "visits": "Only 2 ANC done, blood sugar not tested.",
        "blood sugar": "Never tested.",
        "symptoms": "Excessive thirst, frequent urination for 2 weeks.",
        "thirst": "Excessive thirst, frequent urination for 2 weeks.",
        "weight": "More than expected, baby feels large.",
        "previous": "First baby was 3.8 kg.",
        "baby": "First baby was 3.8 kg.",
        "family": "Yes, mother has diabetes.",
        "history": "Yes, mother has diabetes.",
    },
    correct_referral="REFER_WITHIN_24H",
    correct_urgency="within_24h",
    correct_primary_concern="gestational_diabetes_risk",
    correct_action_items=[
        "Refer to PHC for OGTT (Oral Glucose Tolerance Test)",
        "Document LMP and gestational age on referral slip",
        "Register under JSSK for free testing",
        "Note family history of diabetes and previous macrosomic baby",
        "Advise to avoid sugary foods until tested",
    ],
    near_miss_referrals={"REFER_IMMEDIATELY": 0.5},
    key_danger_signs=[
        "Polydipsia + polyuria in pregnancy = screen for GDM",
        "Family history of diabetes = high risk",
        "Previous macrosomic baby (>3.5kg) = high risk",
        "All pregnant women need 75g OGTT at 24-28 weeks per NHM GDM guidelines 2018",
    ],
    explanation=(
        "Classic GDM presentation: excessive thirst (polydipsia), frequent urination (polyuria), family history "
        "of diabetes, previous macrosomic baby, inadequate ANC. NHM GDM Guidelines 2018 mandate universal 75g "
        "OGTT at 24-28 weeks for all pregnant women in India. This woman is at 26 weeks and has NEVER had blood "
        "sugar tested. REFER_WITHIN_24H for OGTT. GDM causes macrosomia, shoulder dystocia, IUGR, and neonatal "
        "hypoglycemia if uncontrolled."
    ),
    terminal_message="Multiple GDM risk factors at 26 weeks with no blood sugar testing = REFER_WITHIN_24H for OGTT.",
    available_tools=["gestational_age"],
    phc_manageable=True,
    correct_doctor_decision="manage_at_phc",
)

CASE_M08 = ClinicalCase(
    case_id="M08",
    title="Severe Anaemia in Pregnancy — Breathlessness at Rest",
    difficulty="medium",
    category="maternal",
    age_description="24 years old",
    gender="female",
    location="Remote village",
    malaria_risk_area=False,
    season="winter",
    initial_presentation=(
        "Didi, Kamla ko bahut kamzori hai, saansh lene mein bhi takleef hai. "
        "7 mahine pregnant hai. IFA nahi kha rahi thi."
        "\n[Sister, Kamla has extreme weakness and breathlessness. 7 months pregnant. Was not taking IFA tablets.]"
    ),
    followup_responses={
        "breathless": "Breathlessness even when sitting, can't walk.",
        "breathing": "Breathlessness even when sitting, can't walk.",
        "pallor": "Extremely pale — palms, tongue, and conjunctiva white.",
        "pale": "Extremely pale — palms, tongue, and conjunctiva white.",
        "pulse": "Fast and weak.",
        "heart rate": "Fast and weak pulse.",
        "haemoglobin": "Last checked at 4 months = 7.2 g/dL, never treated.",
        "ifa": "Stopped after 1 month, felt nauseous.",
        "iron": "Stopped after 1 month, felt nauseous.",
        "walk": "Cannot walk even a short distance, very breathless.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="severe_anaemia_in_pregnancy",
    correct_action_items=[
        "Call 108 immediately",
        "Ensure JSSK transport entitlement",
        "Keep patient lying down during transport",
        "Give 1 IFA tablet NOW if patient can swallow",
        "Alert PHC for blood transfusion preparation",
        "Do NOT delay for further assessment at home",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.2},
    key_danger_signs=[
        "Breathlessness at rest = cardiac decompensation from severe anaemia",
        "Haemoglobin < 7 g/dL in pregnancy = severe anaemia",
        "Pale palms + pale tongue + pale conjunctiva = severe pallor",
        "IFA non-compliance + ongoing pregnancy = cumulative risk",
    ],
    explanation=(
        "Severe anaemia in pregnancy (Hb < 7 g/dL) is the #1 cause of maternal death in India, contributing "
        "to 20-40% of maternal mortality. Breathlessness at rest indicates cardiac decompensation — the heart "
        "can no longer compensate for lack of oxygen-carrying capacity. This is a medical EMERGENCY requiring "
        "REFER_IMMEDIATELY for blood transfusion. The clinical triad: extreme pallor (palms, tongue, conjunctiva) "
        "+ breathlessness at rest + weak rapid pulse = severe anaemia with heart failure."
    ),
    terminal_message="Severe anaemia with breathlessness at rest and extreme pallor = cardiac decompensation = REFER_IMMEDIATELY.",
    phc_manageable=False,
    correct_doctor_decision="refer_to_fru",
)

CASE_M09 = ClinicalCase(
    case_id="M09",
    title="Pediatric TB — Household Contact of Confirmed TB Case",
    difficulty="medium",
    category="tb",
    age_description="3 years old",
    gender="male",
    location="Dense rural village",
    malaria_risk_area=False,
    season="autumn",
    initial_presentation=(
        "Didi, Raju ke pita ko TB hai, 3 hafte pehle pata chala. Ab Raju ko bhi khansi hai, "
        "teen hafte se. Weight nahi badh raha."
        "\n[Sister, Raju's father has TB, diagnosed 3 weeks ago. Now Raju also has cough for 3 weeks. "
        "Not gaining weight.]"
    ),
    followup_responses={
        "cough": "Persistent 3 weeks, does not improve.",
        "duration": "Persistent 3 weeks, does not improve.",
        "fever": "Low grade fever in evenings.",
        "temperature": "Low grade fever in evenings.",
        "weight": "Has lost 200g in past month, was 12 kg now 11.8 kg.",
        "bcg": "Yes, BCG scar present.",
        "vaccination": "Yes, BCG scar present.",
        "father": "Father started DOTS 2 weeks ago.",
        "treatment": "Father started DOTS 2 weeks ago.",
        "contact": "Yes, living in the same house as father with TB.",
    },
    correct_referral="REFER_WITHIN_24H",
    correct_urgency="within_24h",
    correct_primary_concern="pediatric_tb_contact",
    correct_action_items=[
        "Refer to ASHA/ANM/PHC for pediatric TB screening under NTEP",
        "Document household contact with confirmed TB case on referral slip",
        "Advise Isoniazid Preventive Therapy (IPT) assessment at PHC",
        "Note BCG vaccination status",
        "Advise father to cover mouth during coughing and improve ventilation",
    ],
    near_miss_referrals={"REFER_IMMEDIATELY": 0.5},
    key_danger_signs=[
        "Household contact with smear-positive TB = high exposure risk",
        "Cough > 2 weeks in child with TB contact = screen immediately",
        "Weight loss + prolonged cough + evening fever = TB triad",
        "Children < 5 in TB households need IPT per NTEP guidelines",
    ],
    explanation=(
        "Any child under 5 with a household contact of confirmed TB requires screening under the National TB "
        "Elimination Programme (NTEP). Classic pediatric TB triad: persistent cough > 2 weeks + evening "
        "low-grade fever + weight loss. Mantoux test and chest X-ray at PHC. Children under 5 qualify for "
        "Isoniazid Preventive Therapy (IPT) to prevent progression from TB infection to TB disease. "
        "REFER_WITHIN_24H — this is not a life emergency but requires timely evaluation."
    ),
    terminal_message="Household TB contact + 3-week cough + weight loss = pediatric TB screening = REFER_WITHIN_24H.",
    phc_manageable=True,
    correct_doctor_decision="manage_at_phc",
)

CASE_M10 = ClinicalCase(
    case_id="M10",
    title="NCD Risk — Headache, Blurred Vision, Possible Hypertension",
    difficulty="medium",
    category="ncd",
    age_description="52 years old",
    gender="male",
    location="Rural village",
    malaria_risk_area=False,
    season="winter",
    initial_presentation=(
        "Didi, Ramesh ko sar dard hai kai dino se, aankhen bhi dhundhli dikhti hain. "
        "Uske pita ko stroke hua tha."
        "\n[Sister, Ramesh has had headache for several days, also blurry vision. His father had a stroke.]"
    ),
    followup_responses={
        "bp": "ASHA does not have BP machine.",
        "blood pressure": "ASHA does not have BP machine.",
        "symptoms": "Severe persistent headache, blurred vision, occasional dizziness.",
        "headache": "Severe persistent headache, blurred vision, occasional dizziness.",
        "family": "Father died of stroke, maternal uncle has diabetes.",
        "history": "Father died of stroke, maternal uncle has diabetes.",
        "tobacco": "Yes, smokes 10 cigarettes daily.",
        "smoke": "Yes, smokes 10 cigarettes daily.",
        "known": "Never had BP checked in life, age 52.",
        "conditions": "Never had BP checked in life, age 52.",
        "cbac": "Age 52, tobacco user, family stroke history — CBAC score ≥4.",
    },
    correct_referral="REFER_WITHIN_24H",
    correct_urgency="within_24h",
    correct_primary_concern="ncd_hypertension_screening",
    correct_action_items=[
        "Refer to ANM or PHC for BP measurement and CBAC NCD screening",
        "Document CBAC risk factors on referral slip",
        "Advise stop smoking immediately",
        "Do not give aspirin without BP confirmation",
        "Register under NPCDCS programme at PHC",
    ],
    near_miss_referrals={"REFER_IMMEDIATELY": 0.6},
    key_danger_signs=[
        "Severe headache + blurred vision + family stroke history = possible hypertensive urgency",
        "Never screened at age 52 with multiple risk factors",
        "CBAC score: age 52 + tobacco + family stroke history = high risk (score >= 4 = refer)",
        "Hypertensive urgency without BP confirmation = refer promptly",
    ],
    explanation=(
        "Classic presentation of undiagnosed hypertension in middle-aged rural Indian male. CBAC (Community "
        "Based Assessment Checklist) risk factors present: age >30 (+1), tobacco use (+2), family history of "
        "stroke/CVD (+1) = score 4, which mandates referral per NHM NPCDCS guidelines. Blurred vision + severe "
        "headache = possible hypertensive urgency. ASHA should NEVER diagnose hypertension but MUST refer for "
        "BP measurement. Teaching point: this introduces the NCD screening domain entirely absent from previous cases."
    ),
    terminal_message="CBAC score >= 4 with headache, blurred vision, family stroke history = REFER_WITHIN_24H for BP screening.",
    available_tools=["cbac_scorer"],
    phc_manageable=True,
    correct_doctor_decision="manage_at_phc",
)


# ---------------------------------------------------------------------------
# HARD CASES (Phase 3) — H07-H13
# ---------------------------------------------------------------------------

CASE_H07 = ClinicalCase(
    case_id="H07",
    title="Birth Asphyxia — Baby Not Crying at Home Delivery",
    difficulty="hard",
    category="neonatal",
    age_description="Newborn, minutes old",
    gender="female",
    location="Home delivery in progress",
    malaria_risk_area=False,
    season="monsoon",
    initial_presentation=(
        "Didi jaldi aao! Ghar mein delivery ho gayi! Baby ro nahi rahi, haath-pair bhi nahi hila rahi!"
        "\n[Sister come quickly! Delivery happened at home! Baby is not crying, limbs not moving!]"
    ),
    followup_responses={
        "cry": "No cry at all.",
        "crying": "No cry at all.",
        "color": "Baby is pale/bluish.",
        "colour": "Baby is pale/bluish.",
        "breathing": "Not breathing or very gasping breaths.",
        "breathe": "Not breathing or very gasping breaths.",
        "limbs": "Limp, no movement.",
        "moving": "Limp, no movement.",
        "mother": "Mother is conscious, delivered placenta.",
        "mom": "Mother is conscious, delivered placenta.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="neonatal_birth_asphyxia",
    correct_action_items=[
        "Begin BASIC NEWBORN RESUSCITATION NOW — dry, stimulate, clear airway",
        "Call 108 for emergency transport simultaneously",
        "Do NOT wait to call 108 — begin resuscitation first",
        "Keep baby warm — skin-to-skin with mother during transport if breathing starts",
        "JSSK entitlement: free emergency transport",
        "Alert SNCU/NBSU at nearest FRU",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.05},
    key_danger_signs=[
        "Not crying + limp + not breathing = neonatal asphyxia = immediate resuscitation",
        "Every minute without oxygen = brain damage",
        "Call 108 WHILE beginning resuscitation — don't wait",
        "Golden Minute: resuscitation in first minute = best outcome",
    ],
    explanation=(
        "Birth asphyxia is a DIRE EMERGENCY. Not crying + limp + cyanosis/pallor = failure to establish "
        "breathing. The ASHA must begin basic resuscitation (dry, stimulate back, clear airway) while "
        "simultaneously calling 108. Every minute of oxygen deprivation causes irreversible brain damage. "
        "The 'Golden Minute' concept: resuscitation within 60 seconds of birth dramatically improves outcomes. "
        "This is the most time-critical case in the corpus — the agent must ask MINIMAL questions and act immediately."
    ),
    terminal_message="Not crying + limp + not breathing = neonatal asphyxia = RESUSCITATE NOW + REFER_IMMEDIATELY.",
    phc_manageable=False,
    correct_doctor_decision="refer_to_fru",
)

CASE_H08 = ClinicalCase(
    case_id="H08",
    title="Severe Neonatal Jaundice — Kernicterus Signs (Day 6)",
    difficulty="hard",
    category="neonatal",
    age_description="Newborn, 6 days old",
    gender="male",
    location="Rural home",
    malaria_risk_area=False,
    season="winter",
    initial_presentation=(
        "Didi, chhote bachche ka rang bahut peela hai, puri body. "
        "Kamar karti hai peeche ki taraf. Bahut zyada rota hai."
        "\n[Sister, the baby is completely yellow everywhere. His back arches. He cries very loudly/high pitch.]"
    ),
    followup_responses={
        "jaundice": "Yellow from head to soles of feet, palms also yellow.",
        "yellow": "Yellow from head to soles of feet, palms also yellow.",
        "soles": "Yes, soles are yellow.",
        "palms": "Yes, palms are yellow.",
        "back": "Yes, his back curves backward when he cries.",
        "arching": "Yes, his back curves backward when he cries.",
        "cry": "High-pitched, abnormal cry.",
        "feed": "Refusing to feed, very sleepy between crying bouts.",
        "feeding": "Refusing to feed, very sleepy between crying bouts.",
        "previous": "Was yellow on Day 2, parents thought it was normal jaundice.",
        "history": "Was yellow on Day 2, parents thought it was normal jaundice.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="severe_neonatal_jaundice_kernicterus",
    correct_action_items=[
        "Call 108 immediately — this is neurological emergency",
        "JSSK free emergency transport",
        "Do NOT delay for any reason",
        "Continue breastfeeding during transport if baby able",
        "Alert FRU/SNCU for intensive phototherapy/exchange transfusion",
        "Document age of onset and spread progression",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.15},
    key_danger_signs=[
        "Yellow palms and soles = severe jaundice = emergency",
        "Back arching (opisthotonus) = kernicterus/bilirubin encephalopathy = EMERGENCY",
        "High-pitched cry = neurological damage",
        "Refusing to feed + severe jaundice = REFER_IMMEDIATELY",
    ],
    explanation=(
        "Kernicterus (bilirubin encephalopathy) signs: yellow palms/soles + back arching (opisthotonus) + "
        "high-pitched cry + poor feeding = bilirubin has crossed the blood-brain barrier and is causing "
        "irreversible brain damage. This is a NEUROLOGICAL EMERGENCY. Exchange transfusion may be needed. "
        "Compare with E06 (Day 3 physiological = MONITOR) and M06 (< 24h = REFER_WITHIN_24H): this is the "
        "most severe end of the jaundice spectrum. Teaching point: same condition (jaundice) has 3 completely "
        "different management levels depending on timing, spread, and neurological signs."
    ),
    terminal_message="Yellow palms/soles + back arching + high-pitched cry = kernicterus = REFER_IMMEDIATELY for exchange transfusion.",
    phc_manageable=False,
    correct_doctor_decision="refer_to_fru",
)

CASE_H09 = ClinicalCase(
    case_id="H09",
    title="Puerperal Sepsis — 4 Days Postpartum",
    difficulty="hard",
    category="maternal",
    age_description="26 years old",
    gender="female",
    location="Rural home",
    malaria_risk_area=False,
    season="summer",
    initial_presentation=(
        "Didi, Priya ki delivery 4 din pehle ghar par hui. Ab usko tej bukhaar hai, "
        "pet mein dard hai, pahle jaisa nahi lag rahi."
        "\n[Sister, Priya delivered at home 4 days ago. Now she has high fever, abdominal pain, "
        "not feeling herself.]"
    ),
    followup_responses={
        "fever": "39.5 degrees temperature, started yesterday and worsening.",
        "temperature": "39.5 degrees temperature, started yesterday and worsening.",
        "discharge": "Foul-smelling discharge, greenish-yellow color.",
        "lochia": "Foul-smelling discharge, greenish-yellow color.",
        "vaginal": "Foul-smelling discharge, greenish-yellow color.",
        "pain": "Lower abdominal pain, tender to touch.",
        "abdomen": "Lower abdominal pain, tender to touch.",
        "general": "Looks very ill, confused at times.",
        "conscious": "Looks very ill, confused at times.",
        "breastfeed": "Has stopped breastfeeding baby due to illness.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="puerperal_sepsis",
    correct_action_items=[
        "Call 108 IMMEDIATELY — JSSK free emergency transport for mother",
        "Do NOT give home remedies or delay",
        "Keep patient lying down, ensure airway",
        "Alert PHC/FRU for IV antibiotics and sepsis management",
        "Ensure baby gets alternative feeding during maternal hospitalization",
        "Document 4 days postpartum + home delivery on referral slip",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.1},
    key_danger_signs=[
        "Fever > 38 degrees after delivery = puerperal sepsis until proven otherwise",
        "Foul-smelling lochia = uterine infection (endometritis)",
        "Abdominal tenderness + fever = pelvic sepsis",
        "Confusion/altered consciousness = septic shock developing",
        "Home delivery without aseptic technique = highest risk",
    ],
    explanation=(
        "Puerperal sepsis is a leading cause of maternal death in India: high fever + foul lochia + abdominal "
        "pain 4 days after home delivery = endometritis/puerperal sepsis. This can rapidly progress to septic "
        "shock and death within hours if untreated. Immediate IV antibiotics needed. The altered consciousness "
        "(confusion) indicates sepsis is already compromising organ function — this is NOT routine fever, this "
        "is a LIFE-THREATENING emergency. JSSK covers free emergency transport and all treatment at government facility."
    ),
    terminal_message="High fever + foul lochia + abdominal pain 4 days postpartum = puerperal sepsis = REFER_IMMEDIATELY.",
    phc_manageable=False,
    correct_doctor_decision="refer_to_fru",
)

CASE_H10 = ClinicalCase(
    case_id="H10",
    title="Adolescent Severe Anaemia — RKSK Case, Cardiac Signs",
    difficulty="hard",
    category="adolescent",
    age_description="15 years old",
    gender="female",
    location="Rural village",
    malaria_risk_area=False,
    season="winter",
    initial_presentation=(
        "Didi, Reema ko bahut chakkar aate hain, bohot kamzori hai. "
        "School mein behosh ho gayi. Haiz mein bahut zyada khoon aata hai."
        "\n[Sister, Reema has severe dizziness and weakness. Fainted at school. Heavy menstrual bleeding.]"
    ),
    followup_responses={
        "heart": "Heart is beating very fast, can see pulse in neck.",
        "pulse": "Heart is beating very fast, can see pulse in neck.",
        "pallor": "Extremely pale — white palms, white tongue, white inner eyelids.",
        "pale": "Extremely pale — white palms, white tongue, white inner eyelids.",
        "breathless": "Breathless even sitting, cannot climb one step.",
        "breathing": "Breathless even sitting, cannot climb one step.",
        "period": "Very heavy periods every month for 6 months, soaking 6-8 pads daily.",
        "menstrual": "Very heavy periods every month for 6 months, soaking 6-8 pads daily.",
        "nutrition": "Eats rice and dal, rarely vegetables.",
        "diet": "Eats rice and dal, rarely vegetables.",
        "faint": "Fainted once at school today.",
        "unconscious": "Fainted once at school today.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="adolescent_severe_anaemia_with_cardiac_signs",
    correct_action_items=[
        "REFER_IMMEDIATELY — cardiac decompensation from severe anaemia",
        "Call 108 — JSSK covers transport for adolescents under RKSK",
        "Keep patient lying down, legs slightly elevated",
        "Alert PHC for urgent haemoglobin test and possible blood transfusion",
        "Document menorrhagia as likely cause on referral slip",
        "Do NOT wait — tachycardia + breathlessness at rest = hemodynamic compromise",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.1},
    key_danger_signs=[
        "Tachycardia at rest + breathlessness + extreme pallor = hemodynamic compromise",
        "Syncope (fainting) = acute event from severe anaemia",
        "Heavy menstrual bleeding (menorrhagia) = major risk factor for adolescent anaemia",
        "RKSK (Rashtriya Kishor Swasthya Karyakram) covers adolescent health services",
    ],
    explanation=(
        "Adolescent severe anaemia with cardiac signs (tachycardia at rest, visible neck pulse, breathlessness "
        "at rest, syncope) indicates the cardiovascular system is compensating maximally. Menorrhagia (heavy "
        "periods) is the leading cause of iron deficiency anaemia in adolescent girls in India. RKSK programme "
        "covers adolescent health services. Extreme pallor (palms + tongue + inner eyelids all white) = "
        "haemoglobin likely < 5-6 g/dL. Syncope + tachycardia + orthopnea = HEMODYNAMIC EMERGENCY requiring "
        "immediate blood transfusion. Same pathophysiology as M08 (maternal anaemia) but in adolescent context."
    ),
    terminal_message="Fainting + tachycardia + white pallor from menorrhagia = cardiac compromise = REFER_IMMEDIATELY.",
    phc_manageable=False,
    correct_doctor_decision="refer_to_fru",
)

CASE_H11 = ClinicalCase(
    case_id="H11",
    title="Cerebral Malaria — Unconscious after Seizure, P. falciparum Area",
    difficulty="hard",
    category="malaria",
    age_description="3 years old",
    gender="male",
    location="High P. falciparum malaria area",
    malaria_risk_area=True,
    season="monsoon",
    initial_presentation=(
        "Didi jaldi aao! Rohit ko kal se bukhaar tha, abhi use dora padh gaya, 2 baar. Ab hosh nahi hai!"
        "\n[Sister come quickly! Rohit had fever since yesterday, now had seizures 2 times. Now unconscious!]"
    ),
    followup_responses={
        "conscious": "Completely unconscious, not responding.",
        "consciousness": "Completely unconscious, not responding.",
        "malaria": "Yes, this is P. falciparum area, last year many cases in village.",
        "zone": "Yes, this is P. falciparum area, last year many cases in village.",
        "seizure": "Two seizures in past hour, lasting 5-10 minutes each.",
        "fits": "Two seizures in past hour, lasting 5-10 minutes each.",
        "rdt": "Yes, RDT positive.",
        "test": "Yes, RDT positive.",
        "bleeding": "No visible bleeding.",
        "breathing": "Breathing but labored.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="cerebral_malaria_falciparum",
    correct_action_items=[
        "Call 108 IMMEDIATELY",
        "Give rectal artesunate (pre-referral treatment if available) per NVBDCP protocol",
        "Lateral position to prevent aspiration",
        "Do NOT give oral medications — patient unconscious",
        "Alert FRU for IV artesunate + ICU care",
        "JSSK free emergency transport",
        "Document time of last seizure and duration",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.05},
    key_danger_signs=[
        "Unconscious child in P. falciparum area = cerebral malaria until proven otherwise",
        "Seizures + fever + P. falciparum = classic cerebral malaria triad",
        "Pre-referral rectal artesunate can be given by ASHA per NVBDCP guidelines",
        "Mortality without treatment: > 20% even with treatment: 15%",
    ],
    explanation=(
        "Cerebral malaria (P. falciparum with CNS involvement) is one of the most dangerous malaria "
        "presentations, with 15-20% mortality even with treatment. Classic triad: fever + seizures + "
        "unconsciousness in a high P. falciparum area. Compare with E08 (uncomplicated malaria = TREAT_AT_HOME): "
        "same disease, entirely different severity and management. Pre-referral rectal artesunate is an "
        "ASHA-level intervention per NVBDCP severe malaria protocol that can be given before transport to "
        "start treatment."
    ),
    terminal_message="Unconscious + seizures + RDT positive in falciparum area = cerebral malaria = REFER_IMMEDIATELY + rectal artesunate.",
    phc_manageable=False,
    correct_doctor_decision="refer_to_fru",
)

CASE_H12 = ClinicalCase(
    case_id="H12",
    title="Cord Prolapse — Obstetric Emergency During Labor",
    difficulty="hard",
    category="maternal",
    age_description="28 years old",
    gender="female",
    location="Home labor in progress",
    malaria_risk_area=False,
    season="summer",
    initial_presentation=(
        "Didi, Sunita ko dard ho rahe hain, aur cord bahar dikh rahi hai! Fetal heartbeat bhi slow lag raha hai!"
        "\n[Sister, Sunita is in labor, and the cord is visible outside! Fetal heartbeat also seems slow!]"
    ),
    followup_responses={
        "cord": "Umbilical cord is visible and protruding outside vagina.",
        "umbilical": "Umbilical cord is visible and protruding outside vagina.",
        "fetal heart": "Very slow — was 160 now can barely hear it.",
        "heartbeat": "Very slow — was 160 now can barely hear it.",
        "mother": "Conscious, in labor, scared.",
        "conscious": "Conscious, in labor, scared.",
        "gestation": "40 weeks, normal pregnancy so far.",
        "weeks": "40 weeks, normal pregnancy so far.",
        "position": "Currently lying on back.",
    },
    correct_referral="REFER_IMMEDIATELY",
    correct_urgency="immediate",
    correct_primary_concern="cord_prolapse_obstetric_emergency",
    correct_action_items=[
        "Call 108 IMMEDIATELY — this is the #1 obstetric emergency",
        "Position: KNEE-CHEST position immediately to relieve cord compression",
        "Keep exposed cord MOIST with clean wet cloth — do NOT push it back",
        "Do NOT allow further labor pushing — stop bearing down",
        "Keep baby's presenting part off the cord with a gloved hand if possible",
        "Alert surgical FRU for emergency caesarean",
        "JSSK: free emergency CS at government facility",
    ],
    near_miss_referrals={"REFER_WITHIN_24H": 0.05},
    key_danger_signs=[
        "Cord visible outside vagina = cord prolapse = obstetric emergency",
        "Fetal heart rate dropping = cord compression = fetal distress",
        "Knee-chest position relieves cord pressure immediately",
        "Death of fetus without emergency CS within 30 minutes",
        "Do NOT push cord back — can cause cord spasm",
    ],
    explanation=(
        "Cord prolapse is a true obstetric emergency with < 30 minutes to delivery window before irreversible "
        "fetal hypoxia. Umbilical cord outside the vagina = cord compressed between presenting part and maternal "
        "pelvis = fetal oxygen cut off. ASHA critical interventions: (1) knee-chest position immediately, "
        "(2) keep cord moist, (3) call 108, (4) do NOT push cord back. Emergency CS is the only definitive "
        "treatment. JSSK covers free emergency caesarean. Without immediate action, fetal death is near-certain."
    ),
    terminal_message="Cord visible outside vagina = cord prolapse = knee-chest position + call 108 + REFER_IMMEDIATELY for emergency CS.",
    phc_manageable=False,
    correct_doctor_decision="refer_to_fru",
)

CASE_H13 = ClinicalCase(
    case_id="H13",
    title="Preterm Low Birth Weight — KMC Decision (1.8 kg)",
    difficulty="hard",
    category="neonatal",
    age_description="Newborn, 1 day old",
    gender="male",
    location="Home delivery, preterm",
    malaria_risk_area=False,
    season="winter",
    initial_presentation=(
        "Didi, Rani ne ghar par delivery kiya 7 mahine mein. Baby bahut chota hai. "
        "Kuch kha nahi raha, kaanp raha hai."
        "\n[Sister, Rani delivered at home at 7 months. Baby is very small. Not feeding well, shivering.]"
    ),
    followup_responses={
        "weight": "Baby looks about 1.8 kg, very small.",
        "breathing": "Breathing, no fast breathing or indrawing.",
        "breathe": "Breathing, no fast breathing or indrawing.",
        "temperature": "Baby feels cold, 35.8 degrees by axilla.",
        "temp": "Baby feels cold, 35.8 degrees by axilla.",
        "feed": "Weak suck, cannot latch properly.",
        "feeding": "Weak suck, cannot latch properly.",
        "color": "Pink, not cyanosed.",
        "colour": "Pink, not cyanosed.",
        "shiver": "Yes, shivering — feels cold to touch.",
    },
    correct_referral="REFER_WITHIN_24H",
    correct_urgency="within_24h",
    correct_primary_concern="preterm_low_birth_weight_kmc",
    correct_action_items=[
        "Start Kangaroo Mother Care (KMC) IMMEDIATELY — skin to skin with mother",
        "Refer to SNCU/NBU at PHC for assessment within 24 hours",
        "Assist with expressed breastmilk via cup/spoon if no suck reflex",
        "Keep baby warm with KMC during transport",
        "JSSK: free transport and SNCU care for sick newborns",
        "Document birth weight estimate and gestation on referral slip",
        "Do NOT give sugar water or formula",
    ],
    near_miss_referrals={"REFER_IMMEDIATELY": 0.6},
    key_danger_signs=[
        "Preterm < 37 weeks = high risk for hypothermia, infection, hypoglycemia",
        "< 1.5 kg = REFER_IMMEDIATELY; 1.5-2.5 kg = REFER_WITHIN_24H + KMC",
        "Axillary temp < 36 degrees = hypothermia = start KMC immediately",
        "Weak suck = risk of hypoglycemia — expressed breast milk every 2 hours",
    ],
    explanation=(
        "Weight-based decision: < 1.5 kg = REFER_IMMEDIATELY (very LBW), 1.5-2.5 kg = REFER_WITHIN_24H + KMC "
        "(low birth weight). At 1.8 kg, this baby needs PHC SNCU care but KMC can be started immediately at "
        "home during preparation for transport. KMC (skin-to-skin between mother and baby) maintains temperature, "
        "promotes breastfeeding, and reduces neonatal mortality by 40% in LBW infants per WHO evidence. The "
        "shivering indicates early hypothermia — KMC is both the treatment AND the transport method."
    ),
    terminal_message="Preterm 1.8 kg with hypothermia = REFER_WITHIN_24H + start KMC immediately during transport.",
    available_tools=["drug_dose"],
    phc_manageable=True,
    correct_doctor_decision="manage_at_phc",
)


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------

ALL_CASES: dict[str, ClinicalCase] = {
    case.case_id: case
    for case in [
        CASE_E01, CASE_E02, CASE_E03, CASE_E04, CASE_E05,
        CASE_E06, CASE_E07, CASE_E08,
        CASE_M01, CASE_M02, CASE_M03, CASE_M04, CASE_M05,
        CASE_M06, CASE_M07, CASE_M08, CASE_M09, CASE_M10,
        CASE_H01, CASE_H02, CASE_H03, CASE_H04, CASE_H05, CASE_H06,
        CASE_H07, CASE_H08, CASE_H09, CASE_H10, CASE_H11, CASE_H12, CASE_H13,
    ]
}

CASES_BY_DIFFICULTY: dict[str, list[str]] = {
    "easy":   ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08"],
    "medium": ["M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M10"],
    "hard":   ["H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08", "H09", "H10", "H11", "H12", "H13"],
}
