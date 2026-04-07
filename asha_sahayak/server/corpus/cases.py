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
# Case registry
# ---------------------------------------------------------------------------

ALL_CASES: dict[str, ClinicalCase] = {
    case.case_id: case
    for case in [
        CASE_E01, CASE_E02, CASE_E03, CASE_E04, CASE_E05,
        CASE_M01, CASE_M02, CASE_M03, CASE_M04, CASE_M05,
        CASE_H01, CASE_H02, CASE_H03, CASE_H04, CASE_H05, CASE_H06,
    ]
}

CASES_BY_DIFFICULTY: dict[str, list[str]] = {
    "easy":   ["E01", "E02", "E03", "E04", "E05"],
    "medium": ["M01", "M02", "M03", "M04", "M05"],
    "hard":   ["H01", "H02", "H03", "H04", "H05", "H06"],
}
