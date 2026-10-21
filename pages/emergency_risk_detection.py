
import re
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

GEMINI_API_KEY = "AQ.Ab8RN6LDq8jPgJwvOqXjtYwkyiWbDgzO1YhoMA-uwQXCQ-85sQ"
GEMINI_MODEL = "gemini-3.8-flash"

st.set_page_config(
    page_title="AI Emergency Risk Detection",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 AI Emergency Risk Detection System")
st.caption(
    "Educational decision-support prototype — not a diagnostic or "
    "clinically validated triage system."
)

st.warning(
    "If someone has severe difficulty breathing, persistent or severe "
    "chest pain, signs of stroke, a seizure, severe bleeding, or is "
    "unresponsive, seek emergency medical help immediately. Do not wait "
    "for this application to finish. In India, call 112 for emergencies."
)


# =========================================================
# SYMPTOM KNOWLEDGE BASE
# These flags are safety-oriented rules, not diagnoses.
# =========================================================

SYMPTOM_DATABASE = {
    "chest pain": {
        "severity": 35,
        "emergency": True,
        "aliases": ["chest pain", "chest pressure", "chest tightness"],
    },
    "shortness of breath": {
        "severity": 40,
        "emergency": True,
        "aliases": [
            "shortness of breath",
            "breathlessness",
            "short of breath",
        ],
    },
    "difficulty breathing": {
        "severity": 45,
        "emergency": True,
        "aliases": [
            "difficulty breathing",
            "trouble breathing",
            "cannot breathe",
            "can't breathe",
            "unable to breathe",
        ],
    },
    "seizure": {
        "severity": 50,
        "emergency": True,
        "aliases": ["seizure", "convulsion", "convulsions"],
    },
    "stroke warning signs": {
        "severity": 50,
        "emergency": True,
        "aliases": [
            "stroke",
            "face drooping",
            "facial droop",
            "slurred speech",
            "sudden trouble speaking",
            "sudden weakness on one side",
            "one sided weakness",
        ],
    },
    "paralysis": {
        "severity": 50,
        "emergency": True,
        "aliases": ["paralysis", "sudden paralysis"],
    },
    "loss of consciousness": {
        "severity": 50,
        "emergency": True,
        "aliases": [
            "loss of consciousness",
            "passed out",
            "unconscious",
            "fainted",
            "fainting",
            "unresponsive",
        ],
    },
    "severe bleeding": {
        "severity": 50,
        "emergency": True,
        "aliases": [
            "severe bleeding",
            "heavy bleeding",
            "bleeding heavily",
            "bleeding will not stop",
            "bleeding won't stop",
        ],
    },
    "coughing blood": {
        "severity": 45,
        "emergency": True,
        "aliases": ["coughing blood", "cough up blood", "coughing up blood"],
    },
    "blood in stool": {
        "severity": 35,
        "emergency": False,
        "aliases": ["blood in stool", "bloody stool", "bloody stools"],
    },
    "high fever": {
        "severity": 25,
        "emergency": False,
        "aliases": ["high fever", "very high fever"],
    },
    "fever": {
        "severity": 10,
        "emergency": False,
        "aliases": ["fever", "febrile"],
    },
    "headache": {
        "severity": 8,
        "emergency": False,
        "aliases": ["headache", "headaches"],
    },
    "vomiting": {
        "severity": 12,
        "emergency": False,
        "aliases": ["vomiting", "vomit", "throwing up"],
    },
    "dizziness": {
        "severity": 12,
        "emergency": False,
        "aliases": ["dizziness", "dizzy", "lightheaded", "light headed"],
    },
    "cough": {
        "severity": 8,
        "emergency": False,
        "aliases": ["cough", "coughing"],
    },
    "weakness": {
        "severity": 10,
        "emergency": False,
        "aliases": ["weakness", "weak"],
    },
    "fatigue": {
        "severity": 8,
        "emergency": False,
        "aliases": ["fatigue", "tiredness", "exhaustion"],
    },
    "abdominal pain": {
        "severity": 12,
        "emergency": False,
        "aliases": ["abdominal pain", "stomach pain", "belly pain"],
    },
    "diarrhea": {
        "severity": 10,
        "emergency": False,
        "aliases": ["diarrhea", "diarrhoea", "loose stools"],
    },
    "rash": {
        "severity": 8,
        "emergency": False,
        "aliases": ["rash", "skin rash"],
    },
}

BODY_PARTS = [
    "head", "chest", "heart", "lungs", "stomach",
    "abdomen", "arm", "leg", "back", "neck",
]

DISEASE_KEYWORDS = [
    "diabetes", "stroke", "heart attack", "asthma",
    "covid", "flu", "hypertension",
]

EMERGENCY_SYMPTOMS = {
    name for name, data in SYMPTOM_DATABASE.items()
    if data["emergency"]
}

# Symptom facts used by the inference engines.
SYMPTOM_FACTS = {
    "chest pain": "CHEST_PAIN",
    "shortness of breath": "BREATHING_DIFFICULTY",
    "difficulty breathing": "BREATHING_DIFFICULTY",
    "seizure": "SEIZURE",
    "stroke warning signs": "STROKE_WARNING_SIGNS",
    "paralysis": "PARALYSIS",
    "loss of consciousness": "LOSS_OF_CONSCIOUSNESS",
    "severe bleeding": "SEVERE_BLEEDING",
    "coughing blood": "COUGHING_BLOOD",
    "blood in stool": "BLOOD_IN_STOOL",
    "high fever": "HIGH_FEVER",
    "fever": "FEVER",
    "vomiting": "VOMITING",
    "cough": "COUGH",
    "fatigue": "FATIGUE",
}


# =========================================================
# NLP AND NEGATION-AWARE SYMPTOM EXTRACTION
# =========================================================

def normalize_text(text):
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_phrase_positions(text, phrase):
    """Find whole-word phrase occurrences."""
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return [m.start() for m in re.finditer(pattern, text)]


def is_negated(text, start, phrase):
    """
    Approximate negation detection.
    This is not a complete clinical NLP model.
    """
    prefix = text[max(0, start - 85):start]

    # Common explicit negations immediately before a symptom.
    negation_patterns = [
        r"\bno\b(?:\s+\w+){0,4}\s*$",
        r"\bnot\b(?:\s+\w+){0,4}\s*$",
        r"\bnever\b(?:\s+\w+){0,4}\s*$",
        r"\bwithout\b(?:\s+\w+){0,4}\s*$",
        r"\bdeny\b(?:\s+\w+){0,4}\s*$",
        r"\bdenies\b(?:\s+\w+){0,4}\s*$",
        r"\bdenied\b(?:\s+\w+){0,4}\s*$",
        r"\bdo not have\b(?:\s+\w+){0,3}\s*$",
        r"\bdon't have\b(?:\s+\w+){0,3}\s*$",
        r"\bdoes not have\b(?:\s+\w+){0,3}\s*$",
        r"\bdoesn't have\b(?:\s+\w+){0,3}\s*$",
        r"\bno evidence of\b(?:\s+\w+){0,3}\s*$",
    ]

    if any(re.search(p, prefix) for p in negation_patterns):
        return True

    # Past history is not necessarily a current symptom.
    if re.search(
        r"\b(history of|previously had|used to have)\b"
        r"(?:\s+\w+){0,4}\s*$",
        prefix,
    ):
        return True

    return False


def extract_medical_entities(text):
    processed = normalize_text(text)
    detected = []
    negated = []

    for symptom, data in SYMPTOM_DATABASE.items():
        matches = []

        for alias in data["aliases"]:
            matches.extend(
                (pos, alias)
                for pos in find_phrase_positions(processed, alias)
            )

        if not matches:
            continue

        # One symptom category is counted once.
        matches.sort(key=lambda item: item[0])
        current_found = False
        negated_found = False

        for position, alias in matches:
            if is_negated(processed, position, alias):
                negated_found = True
            else:
                current_found = True

        if current_found:
            detected.append(symptom)
        elif negated_found:
            negated.append(symptom)

    # Avoid double counting "high fever" and its broader "fever" match.
    if "high fever" in detected and "fever" in detected:
        detected.remove("fever")

    organs = [
        organ for organ in BODY_PARTS
        if find_phrase_positions(processed, organ)
    ]

    diseases = [
        disease for disease in DISEASE_KEYWORDS
        if find_phrase_positions(processed, disease)
    ]

    return {
        "symptoms": detected,
        "negated_symptoms": negated,
        "organs": organs,
        "diseases": diseases,
    }


# =========================================================
# BASELINE SEVERITY SCORE
# A transparent educational heuristic, NOT a medical score.
# =========================================================

def calculate_severity_score(symptoms):
    score = sum(
        SYMPTOM_DATABASE[s]["severity"]
        for s in set(symptoms)
        if s in SYMPTOM_DATABASE
    )
    return min(100, score)


def detect_emergency(symptoms):
    found = sorted(set(symptoms) & EMERGENCY_SYMPTOMS)
    return bool(found), found


def expert_system_analysis(symptoms):
    s = set(symptoms)
    findings = []

    if "chest pain" in s:
        findings.append(
            "Chest pain detected: urgent medical assessment may be needed."
        )

    if s & {"shortness of breath", "difficulty breathing"}:
        findings.append(
            "Breathing difficulty detected: urgent assessment is warranted."
        )

    if s & {
        "stroke warning signs", "paralysis",
        "loss of consciousness", "seizure",
    }:
        findings.append(
            "A neurological red flag was detected; seek emergency assessment."
        )

    if "severe bleeding" in s:
        findings.append(
            "Severe bleeding was reported; seek emergency assistance."
        )

    if "coughing blood" in s:
        findings.append(
            "Coughing blood requires prompt medical assessment; emergency "
            "care may be needed depending on severity."
        )

    if not findings:
        findings.append(
            "No listed red-flag pattern was identified by these rules. "
            "This does not rule out an emergency."
        )

    return findings


# =========================================================
# FUZZY MEMBERSHIP FUNCTIONS
# Used to demonstrate fuzzy classification of the heuristic score.
# =========================================================

def triangle(x, a, b, c):
    if x < a or x > c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


def fuzzy_risk_assessment(score):
    x = float(np.clip(score, 0, 100))

    memberships = {
        "Low (heuristic)": max(0.0, min(1.0, (50 - x) / 25))
        if x > 25 else 1.0,
        "Moderate (heuristic)": triangle(x, 25, 50, 75),
        "High (heuristic)": triangle(x, 50, 75, 90),
        "Critical (heuristic)": max(0.0, min(1.0, (x - 75) / 25)),
    }

    label = max(memberships, key=memberships.get)
    return memberships, label


# =========================================================
# FORWARD CHAINING
# Facts are strings. Rules are (premises, conclusion, explanation).
# =========================================================

RULES = [
    (("CHEST_PAIN",), "URGENT_ASSESSMENT_NEEDED",
     "Chest pain is a red-flag symptom."),
    (("BREATHING_DIFFICULTY",), "URGENT_ASSESSMENT_NEEDED",
     "Breathing difficulty is a red-flag symptom."),
    (("SEIZURE",), "URGENT_ASSESSMENT_NEEDED",
     "Seizure is a red-flag symptom."),
    (("STROKE_WARNING_SIGNS",), "URGENT_ASSESSMENT_NEEDED",
     "Possible stroke warning signs require urgent assessment."),
    (("PARALYSIS",), "URGENT_ASSESSMENT_NEEDED",
     "Sudden paralysis requires emergency assessment."),
    (("LOSS_OF_CONSCIOUSNESS",), "URGENT_ASSESSMENT_NEEDED",
     "Loss of consciousness is a red flag."),
    (("SEVERE_BLEEDING",), "URGENT_ASSESSMENT_NEEDED",
     "Severe bleeding requires emergency assessment."),
    (("COUGHING_BLOOD",), "PROMPT_MEDICAL_REVIEW",
     "Coughing blood requires medical review."),
    (("BLOOD_IN_STOOL",), "PROMPT_MEDICAL_REVIEW",
     "Blood in stool requires medical assessment."),
    (("HIGH_FEVER",), "PROMPT_MEDICAL_REVIEW",
     "High fever may require prompt medical review."),
    (("FEVER", "VOMITING"), "MEDICAL_REVIEW_MAY_BE_NEEDED",
     "Fever and vomiting together may warrant medical review."),
    (("URGENT_ASSESSMENT_NEEDED",), "URGENT_WARNING",
     "A red-flag rule has fired."),
]


def symptoms_to_facts(symptoms):
    facts = set()
    for symptom in symptoms:
        fact = SYMPTOM_FACTS.get(symptom)
        if fact:
            facts.add(fact)
    return facts


def forward_chaining(initial_facts):
    facts = set(initial_facts)
    trace = []
    changed = True

    while changed:
        changed = False

        for premises, conclusion, explanation in RULES:
            if all(p in facts for p in premises) and conclusion not in facts:
                facts.add(conclusion)
                trace.append({
                    "Rule premises": " AND ".join(premises),
                    "Derived fact": conclusion,
                    "Explanation": explanation,
                })
                changed = True

    return facts, trace


# =========================================================
# BACKWARD CHAINING
# =========================================================

def backward_chaining(goal, initial_facts):
    facts = set(initial_facts)
    trace = []
    active_goals = set()
    memo = {}

    def prove(target):
        if target in facts:
            trace.append(f"PROVED {target}: it is an initial fact.")
            return True

        if target in memo:
            return memo[target]

        if target in active_goals:
            trace.append(f"Cycle avoided while checking {target}.")
            return False

        active_goals.add(target)
        matching_rules = [
            rule for rule in RULES if rule[1] == target
        ]

        for premises, conclusion, explanation in matching_rules:
            trace.append(
                f"Trying rule: {' AND '.join(premises)} -> {conclusion}"
            )

            if all(prove(premise) for premise in premises):
                trace.append(f"PROVED {target}: {explanation}")
                active_goals.remove(target)
                memo[target] = True
                return True

        active_goals.remove(target)
        memo[target] = False
        trace.append(f"NOT PROVED: {target}.")
        return False

    result = prove(goal)
    return result, trace


# =========================================================
# PROPOSITIONAL RESOLUTION
# Literals are strings; e.g. "CHEST_PAIN" and "~CHEST_PAIN".
# A clause is a frozenset of literals.
# =========================================================

def negate_literal(literal):
    return literal[1:] if literal.startswith("~") else "~" + literal


def is_tautology(clause):
    return any(negate_literal(lit) in clause for lit in clause)


def rule_to_clause(premises, conclusion):
    # (A AND B) -> C is equivalent to (~A OR ~B OR C).
    return frozenset(
        [negate_literal(p) for p in premises] + [conclusion]
    )


def resolve_pair(clause_a, clause_b):
    resolvents = set()

    for literal in clause_a:
        opposite = negate_literal(literal)
        if opposite in clause_b:
            new_clause = frozenset(
                (set(clause_a) - {literal})
                | (set(clause_b) - {opposite})
            )

            if not is_tautology(new_clause):
                resolvents.add(new_clause)

    return resolvents


def resolution_entails(query, initial_facts, max_rounds=100):
    """
    Refutation: add the negation of the query and attempt to derive
    the empty clause. Returns an educational proof trace.
    """
    clauses = set()

    for fact in initial_facts:
        clauses.add(frozenset([fact]))

    for premises, conclusion, _ in RULES:
        clauses.add(rule_to_clause(premises, conclusion))

    clauses.add(frozenset([negate_literal(query)]))
    trace = []
    seen_pairs = set()

    for round_number in range(1, max_rounds + 1):
        current = list(clauses)
        new_clauses = set()

        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                a, b = current[i], current[j]
                pair_key = (a, b) if repr(a) <= repr(b) else (b, a)

                if pair_key in seen_pairs:
                    continue

                seen_pairs.add(pair_key)

                for resolvent in resolve_pair(a, b):
                    if not resolvent:
                        trace.append(
                            f"Round {round_number}: derived the empty clause. "
                            "The negated query contradicts the knowledge base."
                        )
                        return True, trace

                    if resolvent not in clauses:
                        new_clauses.add(resolvent)
                        trace.append(
                            f"Round {round_number}: derived "
                            f"{' OR '.join(sorted(resolvent))}."
                        )

        if not new_clauses:
            trace.append(
                "No new clauses could be derived. The query was not proven "
                "by this knowledge base."
            )
            return False, trace

        clauses.update(new_clauses)

        # Guard against runaway clause growth in this demonstration.
        if len(clauses) > 2500:
            trace.append(
                "Stopped at the clause limit. Result is inconclusive."
            )
            return None, trace

    trace.append("Reached the round limit. Result is inconclusive.")
    return None, trace


# =========================================================
# DECISION TREE CLASSIFICATION
# IMPORTANT: synthetic, rule-generated training examples.
# This demonstrates the algorithm, not a clinically validated model.
# =========================================================

FEATURES = [
    "chest_pain",
    "breathing_difficulty",
    "neurological_red_flag",
    "severe_bleeding",
    "coughing_blood",
    "high_fever",
    "fever",
    "vomiting",
    "dizziness",
    "cough",
    "fatigue",
    "age",
    "duration_days",
]


def create_synthetic_dataset(seed=42, n=3000):
    rng = np.random.default_rng(seed)

    data = pd.DataFrame({
        "chest_pain": rng.integers(0, 2, n),
        "breathing_difficulty": rng.integers(0, 2, n),
        "neurological_red_flag": rng.integers(0, 2, n),
        "severe_bleeding": rng.integers(0, 2, n),
        "coughing_blood": rng.integers(0, 2, n),
        "high_fever": rng.integers(0, 2, n),
        "fever": rng.integers(0, 2, n),
        "vomiting": rng.integers(0, 2, n),
        "dizziness": rng.integers(0, 2, n),
        "cough": rng.integers(0, 2, n),
        "fatigue": rng.integers(0, 2, n),
        "age": rng.integers(1, 101, n),
        "duration_days": rng.integers(1, 31, n),
    })

    red_flag = (
        (data["chest_pain"] == 1)
        | (data["breathing_difficulty"] == 1)
        | (data["neurological_red_flag"] == 1)
        | (data["severe_bleeding"] == 1)
        | (data["coughing_blood"] == 1)
    )

    prompt_review = (
        (data["high_fever"] == 1)
        | (data["fever"] == 1)
        & (data["vomiting"] == 1)
        | (data["duration_days"] >= 14)
    )

    data["target"] = np.select(
        [red_flag, prompt_review],
        ["Urgent red-flag pattern", "Medical review may be needed"],
        default="No listed red-flag pattern",
    )

    return data


@st.cache_resource
def train_decision_tree():
    dataset = create_synthetic_dataset()

    x = dataset[FEATURES]
    y = dataset["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, dataset, accuracy, x_test, y_test, predictions


def symptoms_to_tree_features(symptoms, age, duration):
    s = set(symptoms)

    return pd.DataFrame([{
        "chest_pain": int("chest pain" in s),
        "breathing_difficulty": int(
            bool(s & {"shortness of breath", "difficulty breathing"})
        ),
        "neurological_red_flag": int(
            bool(s & {
                "stroke warning signs", "paralysis",
                "seizure", "loss of consciousness",
            })
        ),
        "severe_bleeding": int("severe bleeding" in s),
        "coughing_blood": int("coughing blood" in s),
        "high_fever": int("high fever" in s),
        "fever": int("fever" in s or "high fever" in s),
        "vomiting": int("vomiting" in s),
        "dizziness": int("dizziness" in s),
        "cough": int("cough" in s),
        "fatigue": int("fatigue" in s),
        "age": int(age),
        "duration_days": int(duration),
    }], columns=FEATURES)


# =========================================================
# ILLUSTRATIVE RISK CATEGORY SUMMARY
# These are not validated probabilities or triage scores.
# =========================================================

def get_rule_summary(symptoms, derived_facts):
    emergency, emergency_list = detect_emergency(symptoms)

    if emergency or "URGENT_ASSESSMENT_NEEDED" in derived_facts:
        return (
            "Red flag detected",
            "Seek emergency medical assessment. Do not rely on the app "
            "to determine whether it is safe to wait.",
        )

    if "PROMPT_MEDICAL_REVIEW" in derived_facts:
        return (
            "Prompt medical review may be needed",
            "Contact a qualified healthcare professional for advice.",
        )

    if "MEDICAL_REVIEW_MAY_BE_NEEDED" in derived_facts:
        return (
            "Medical review may be needed",
            "Consider professional advice, particularly if symptoms "
            "persist, worsen, or concern you.",
        )

    return (
        "No listed red flag identified",
        "This result does not rule out an emergency or serious illness.",
    )


# =========================================================
# PATIENT INPUT
# =========================================================

st.subheader("🧑 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    patient_name = st.text_input("Patient name (optional)")

with col2:
    age = st.number_input("Age", min_value=1, max_value=120, value=25)

with col3:
    gender = st.selectbox("Gender (optional)", ["Prefer not to say", "Female", "Male", "Other"])

duration = st.number_input(
    "Duration of symptoms (days)",
    min_value=1,
    max_value=365,
    value=1,
)

symptom_text = st.text_area(
    "Describe current symptoms",
    height=160,
    placeholder=(
        "Example: I have a cough and fever. "
        "I do not have chest pain."
    ),
)

analyze = st.button(
    "🚨 Analyze Symptoms",
    use_container_width=True,
    type="primary",
)


# =========================================================
# ANALYSIS PIPELINE
# =========================================================

if analyze:
    if not symptom_text.strip():
        st.warning("Enter a symptom description first.")
        st.stop()

    entities = extract_medical_entities(symptom_text)
    symptoms = entities["symptoms"]
    negated_symptoms = entities["negated_symptoms"]

    severity_score = calculate_severity_score(symptoms)
    emergency_flag, emergency_list = detect_emergency(symptoms)
    expert_findings = expert_system_analysis(symptoms)

    initial_facts = symptoms_to_facts(symptoms)
    derived_facts, forward_trace = forward_chaining(initial_facts)

    urgent_proved, urgent_backward_trace = backward_chaining(
        "URGENT_ASSESSMENT_NEEDED", initial_facts
    )

    review_proved, review_backward_trace = backward_chaining(
        "PROMPT_MEDICAL_REVIEW", initial_facts
    )

    resolution_urgent, resolution_urgent_trace = resolution_entails(
        "URGENT_ASSESSMENT_NEEDED", initial_facts
    )

    resolution_review, resolution_review_trace = resolution_entails(
        "PROMPT_MEDICAL_REVIEW", initial_facts
    )

    memberships, fuzzy_label = fuzzy_risk_assessment(severity_score)

    rule_label, rule_advice = get_rule_summary(symptoms, derived_facts)

    tree_model, synthetic_data, tree_accuracy, x_test, y_test, predictions = (
        train_decision_tree()
    )

    patient_features = symptoms_to_tree_features(symptoms, age, duration)
    tree_prediction = tree_model.predict(patient_features)[0]
    tree_probabilities = tree_model.predict_proba(patient_features)[0]

    tree_probability_df = pd.DataFrame({
        "Model class": tree_model.classes_,
        "Model score (%)": np.round(tree_probabilities * 100, 2),
    }).sort_values("Model score (%)", ascending=False)

    # Save results so the user can change tabs without rerunning analysis.
    results = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_name": patient_name,
        "age": int(age),
        "gender": gender,
        "duration": int(duration),
        "symptom_text": symptom_text,
        "entities": entities,
        "symptoms": symptoms,
        "negated_symptoms": negated_symptoms,
        "severity_score": severity_score,
        "emergency_flag": emergency_flag,
        "emergency_list": emergency_list,
        "expert_findings": expert_findings,
        "initial_facts": sorted(initial_facts),
        "derived_facts": sorted(derived_facts),
        "forward_trace": forward_trace,
        "urgent_proved": urgent_proved,
        "urgent_backward_trace": urgent_backward_trace,
        "review_proved": review_proved,
        "review_backward_trace": review_backward_trace,
        "resolution_urgent": resolution_urgent,
        "resolution_urgent_trace": resolution_urgent_trace,
        "resolution_review": resolution_review,
        "resolution_review_trace": resolution_review_trace,
        "memberships": memberships,
        "fuzzy_label": fuzzy_label,
        "rule_label": rule_label,
        "rule_advice": rule_advice,
        "tree_prediction": tree_prediction,
        "tree_probabilities": tree_probability_df.to_dict("records"),
        "tree_accuracy": tree_accuracy,
        "tree_rules": export_text(tree_model, feature_names=FEATURES),
    }

    st.session_state["emergency_results"] = results


# =========================================================
# DISPLAY RESULTS
# =========================================================

if "emergency_results" in st.session_state:
    r = st.session_state["emergency_results"]

    st.markdown("---")
    st.subheader("📊 Emergency Risk Dashboard")

    if r["emergency_flag"]:
        st.error(
            "🚨 A listed red-flag symptom was identified: "
            + ", ".join(r["emergency_list"])
            + ". Seek emergency medical assessment."
        )
    else:
        st.info(
            "No listed emergency red flag was identified by the keyword "
            "rules. This does not rule out a serious condition."
        )

    st.write("**Rule-based assessment:**", r["rule_label"])
    st.write(r["rule_advice"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Educational severity score", f'{r["severity_score"]}/100')
    c2.metric("Decision tree class", r["tree_prediction"])
    c3.metric("Detected symptoms", len(r["symptoms"]))

    st.caption(
        "The severity score is a hand-authored heuristic. Decision-tree "
        "scores are outputs from synthetic, rule-generated training data. "
        "Neither is a calibrated medical risk estimate."
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Symptoms & Rules",
        "Decision Tree",
        "Forward Chaining",
        "Backward Chaining",
        "Resolution",
        "Report & Gemini",
    ])

    # -----------------------------------------------------
    # TAB 1: EXTRACTED ENTITIES AND RULES
    # -----------------------------------------------------

    with tab1:
        st.subheader("🧬 Extracted Medical Entities")

        a, b, c = st.columns(3)

        a.write("**Current symptom matches**")
        a.write(r["entities"]["symptoms"] or ["None identified"])

        b.write("**Negated or denied matches**")
        b.write(r["entities"]["negated_symptoms"] or ["None identified"])

        c.write("**Organs / body parts**")
        c.write(r["entities"]["organs"] or ["None identified"])

        st.write("**Disease terms mentioned:**", r["entities"]["diseases"] or ["None identified"])
        st.write("**Initial logical facts:**", r["initial_facts"])
        st.write("**Derived logical facts:**", r["derived_facts"])

        st.subheader("🧠 Fuzzy Membership Demonstration")
        fuzzy_df = pd.DataFrame({
            "Fuzzy category": list(r["memberships"].keys()),
            "Membership": list(r["memberships"].values()),
        })
        st.dataframe(fuzzy_df, use_container_width=True, hide_index=True)
        st.caption(
            "These memberships classify the heuristic score only. "
            "They are not clinical probabilities."
        )

        st.subheader("🩺 Expert Rule Findings")
        for finding in r["expert_findings"]:
            st.write(f"- {finding}")

    # -----------------------------------------------------
    # TAB 2: DECISION TREE
    # -----------------------------------------------------

    with tab2:
        st.subheader("🌳 Decision Tree Classification")

        st.write("**Prediction:**", r["tree_prediction"])
        st.write(
            f"**Held-out accuracy on synthetic test data:** "
            f'{r["tree_accuracy"] * 100:.1f}%'
        )

        st.warning(
            "This is a demonstration classifier trained on synthetic labels "
            "generated from predefined rules. Its accuracy measures agreement "
            "with those rules, not real-world medical accuracy."
        )

        st.dataframe(
            pd.DataFrame(r["tree_probabilities"]),
            use_container_width=True,
            hide_index=True,
        )

        st.write("**Decision path / learned rules**")
        st.code(r["tree_rules"], language="text")

        tree_model, _, _, _, _, _ = train_decision_tree()
        fig, ax = plt.subplots(figsize=(20, 9))
        plot_tree(
            tree_model,
            feature_names=FEATURES,
            class_names=tree_model.classes_,
            filled=False,
            rounded=True,
            max_depth=3,
            fontsize=7,
            ax=ax,
        )
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Feature importance")
        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": tree_model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))

    # -----------------------------------------------------
    # TAB 3: FORWARD CHAINING
    # -----------------------------------------------------

    with tab3:
        st.subheader("➡️ Forward Chaining")
        st.write(
            "Forward chaining starts with known symptom facts and applies "
            "rules until no additional facts can be derived."
        )

        if r["forward_trace"]:
            st.dataframe(
                pd.DataFrame(r["forward_trace"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No inference rules fired for the detected facts.")

        st.write("**Final fact set:**", r["derived_facts"])

    # -----------------------------------------------------
    # TAB 4: BACKWARD CHAINING
    # -----------------------------------------------------

    with tab4:
        st.subheader("⬅️ Backward Chaining")

        st.write("**Goal:** URGENT_ASSESSMENT_NEEDED")
        st.write("**Proved from current facts:**", r["urgent_proved"])

        for step in r["urgent_backward_trace"]:
            st.write("- " + step)

        st.divider()
        st.write("**Second goal:** PROMPT_MEDICAL_REVIEW")
        st.write("**Proved from current facts:**", r["review_proved"])

        for step in r["review_backward_trace"]:
            st.write("- " + step)

        st.caption(
            "A goal not proved by this limited knowledge base does not mean "
            "that the patient is medically safe."
        )

    # -----------------------------------------------------
    # TAB 5: RESOLUTION
    # -----------------------------------------------------

    with tab5:
        st.subheader("🔎 Propositional Resolution")

        st.write("**Query:** URGENT_ASSESSMENT_NEEDED")
        st.write("**Entailed by this knowledge base:**", r["resolution_urgent"])

        for step in r["resolution_urgent_trace"]:
            st.write("- " + step)

        st.divider()
        st.write("**Query:** PROMPT_MEDICAL_REVIEW")
        st.write("**Entailed by this knowledge base:**", r["resolution_review"])

        for step in r["resolution_review_trace"]:
            st.write("- " + step)

        st.caption(
            "Resolution proves consequences of the encoded logical rules. "
            "It cannot establish a medical diagnosis or prove that no emergency "
            "exists. An inconclusive result is not a negative medical finding."
        )

    # -----------------------------------------------------
    # TAB 6: REPORT AND OPTIONAL GEMINI
    # -----------------------------------------------------

    with tab6:
        st.subheader("🤖 Optional Gemini Explanation")

        ai_report = (
            "Gemini analysis was not requested or is unavailable. "
            "The rule-based results above are available without Gemini."
        )

        if st.button("Generate optional AI explanation"):
            if not GEMINI_API_KEY or GEMINI_API_KEY == "PASTE_YOUR_API_KEY_HERE":
                st.warning(
                    "Add your Gemini API key to GEMINI_API_KEY to enable "
                    "this optional feature."
                )
            else:
                try:
                    import google.generativeai as genai

                    genai.configure(api_key=GEMINI_API_KEY)
                    model = genai.GenerativeModel(GEMINI_MODEL)

                    prompt = f"""
Explain the following educational symptom-screening output in cautious,
plain language. Do not diagnose, calculate new probabilities, or claim
clinical certainty. Mention that urgent red flags require real medical
assessment and the software cannot rule out emergencies.

Age: {r['age']}
Duration in days: {r['duration']}
Current symptom matches: {r['symptoms']}
Negated symptom matches: {r['negated_symptoms']}
Rule-based assessment: {r['rule_label']}
Expert rule findings: {r['expert_findings']}
Decision tree output from synthetic training: {r['tree_prediction']}
Forward-chained facts: {r['derived_facts']}

Provide:
1. A brief description of what the software detected.
2. The limitations of keyword matching and synthetic decision-tree data.
3. General safety guidance.
4. A reminder to consult a qualified healthcare professional.
"""
                    with st.spinner("Generating explanation..."):
                        response = model.generate_content(prompt)
                        ai_report = getattr(response, "text", None) or (
                            "The model returned no text."
                        )

                    st.markdown(ai_report)

                except Exception as exc:
                    ai_report = "Gemini was unavailable: " + str(exc)
                    st.error(ai_report)

        st.subheader("📥 Download Report")

        report_text = f"""
AI EMERGENCY RISK DETECTION — EDUCATIONAL REPORT
Generated: {r['generated']}

IMPORTANT:
This report is not a diagnosis or a validated triage assessment.
A lack of detected red flags does not rule out an emergency.

Patient name: {r['patient_name'] or 'Not supplied'}
Age: {r['age']}
Gender: {r['gender']}
Symptom duration: {r['duration']} day(s)

Description:
{r['symptom_text']}

Detected symptom matches:
{r['symptoms']}

Negated symptom matches:
{r['negated_symptoms']}

Detected organs/body parts:
{r['entities']['organs']}

Disease terms mentioned:
{r['entities']['diseases']}

Educational heuristic severity score:
{r['severity_score']}/100

Emergency red flags matched:
{r['emergency_list']}

Rule-based summary:
{r['rule_label']}
{r['rule_advice']}

Decision Tree output (trained on synthetic, rule-generated examples):
{r['tree_prediction']}

Forward Chaining facts:
{r['derived_facts']}

Backward Chaining proved urgent goal:
{r['urgent_proved']}

Resolution result for urgent goal:
{r['resolution_urgent']}

Expert rule findings:
{r['expert_findings']}

Fuzzy memberships:
{json.dumps(r['memberships'], indent=2)}

AI explanation:
{ai_report}

The symptom extractor may miss symptoms or misunderstand context.
Seek professional medical advice when concerned.
"""
        st.download_button(
            "Download report (.txt)",
            data=report_text,
            file_name="Emergency_Risk_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption(
    "Educational prototype only. The keyword extractor, rule base, fuzzy "
    "membership functions and synthetic Decision Tree are not clinically "
    "validated. They must not be used to diagnose, rule out emergencies, "
    "or delay professional care."
)
