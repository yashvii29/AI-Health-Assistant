import pandas as pd
import re
from datetime import datetime

import streamlit as st


# ============================================================
# 1. OPTIONAL GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = "AQ.Ab8RN6LDq8jPgJwvOqXjtYwkyiWbDgzO1YhoMA-uwQXCQ-85sQ"
GEMINI_MODEL = "gemini-3.8-flash"

# Core functionality and all three algorithms work without Gemini.


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Symptom Checker",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 AI Symptom Checker")

st.write(
    "A rule-based symptom assessment demonstration using "
    "Forward Chaining, Backward Chaining, and Propositional Resolution."
)

st.warning(
    "This educational tool cannot diagnose illnesses, exclude emergencies, "
    "or replace professional medical assessment. If you may be experiencing "
    "a medical emergency, seek emergency help rather than relying on this app."
)


# ============================================================
# 3. SYMPTOM KNOWLEDGE BASE
# ============================================================

# These are symptom-matching patterns, not a comprehensive medical vocabulary.
# High-risk symptoms are intentionally handled conservatively.
SYMPTOM_PATTERNS = {
    "chest pain": [
        r"\bchest pain\b",
        r"\bchest pressure\b",
        r"\bchest tightness\b",
    ],
    "difficulty breathing": [
        r"\bdifficulty breathing\b",
        r"\btrouble breathing\b",
        r"\bshortness of breath\b",
        r"\bbreathlessness\b",
        r"\bstruggling to breathe\b",
    ],
    "loss of consciousness": [
        r"\bloss of consciousness\b",
        r"\bunconscious\b",
        r"\bpassed out\b",
        r"\bnot responding\b",
    ],
    "seizure": [
        r"\bseizure\b",
        r"\bconvulsion\b",
        r"\bconvulsions\b",
    ],
    "stroke warning signs": [
        r"\bsudden facial droop\b",
        r"\bface drooping\b",
        r"\bsudden weakness on one side\b",
        r"\bone-sided weakness\b",
        r"\bsudden trouble speaking\b",
        r"\bslurred speech\b",
        r"\bsudden vision loss\b",
    ],
    "severe bleeding": [
        r"\bsevere bleeding\b",
        r"\buncontrolled bleeding\b",
        r"\bbleeding that will not stop\b",
    ],
    "coughing blood": [
        r"\bcoughing blood\b",
        r"\bcoughed up blood\b",
        r"\bblood in my cough\b",
    ],
    "blood in stool": [
        r"\bblood in (?:my )?stool\b",
        r"\bbloody stool\b",
        r"\bblack tarry stool\b",
    ],
    "high fever": [
        r"\bhigh fever\b",
        r"\bfever of 103\b",
        r"\bfever of 104\b",
        r"\bfever of 105\b",
    ],
    "fever": [
        r"\bfever\b",
        r"\bfebrile\b",
        r"\btemperature\b",
    ],
    "cough": [
        r"\bcough\b",
        r"\bcoughing\b",
    ],
    "cold symptoms": [
        r"\brunny nose\b",
        r"\bnasal congestion\b",
        r"\bsneezing\b",
    ],
    "headache": [
        r"\bheadache\b",
        r"\bheadaches\b",
    ],
    "migraine": [
        r"\bmigraine\b",
        r"\bmigraines\b",
    ],
    "vomiting": [
        r"\bvomiting\b",
        r"\bvomited\b",
        r"\bthrowing up\b",
    ],
    "nausea": [
        r"\bnausea\b",
        r"\bnauseated\b",
        r"\bfeel sick to my stomach\b",
    ],
    "diarrhea": [
        r"\bdiarrhea\b",
        r"\bloose stools\b",
    ],
    "fatigue": [
        r"\bfatigue\b",
        r"\bvery tired\b",
        r"\bexhausted\b",
    ],
    "weakness": [
        r"\bweakness\b",
        r"\bweak\b",
    ],
    "sore throat": [
        r"\bsore throat\b",
        r"\bthroat pain\b",
    ],
    "dizziness": [
        r"\bdizziness\b",
        r"\bdizzy\b",
        r"\blightheaded\b",
    ],
    "abdominal pain": [
        r"\babdominal pain\b",
        r"\bstomach pain\b",
        r"\bbelly pain\b",
    ],
    "back pain": [
        r"\bback pain\b",
    ],
    "joint pain": [
        r"\bjoint pain\b",
        r"\bpainful joints\b",
    ],
    "rash": [
        r"\brash\b",
        r"\bskin rash\b",
    ],
    "itching": [
        r"\bitching\b",
        r"\bitchy\b",
    ],
    "anxiety": [
        r"\banxiety\b",
        r"\banxious\b",
        r"\bpanic\b",
    ],
    "sleep difficulty": [
        r"\binsomnia\b",
        r"\btrouble sleeping\b",
        r"\bdifficulty sleeping\b",
    ],
}

# Symptoms that warrant urgent assessment if actually present.
# This is not a complete emergency checklist.
EMERGENCY_SYMPTOMS = {
    "chest pain",
    "difficulty breathing",
    "loss of consciousness",
    "seizure",
    "stroke warning signs",
    "severe bleeding",
    "coughing blood",
}

SYMPTOM_WEIGHTS = {
    "chest pain": 30,
    "difficulty breathing": 40,
    "loss of consciousness": 50,
    "seizure": 50,
    "stroke warning signs": 50,
    "severe bleeding": 45,
    "coughing blood": 35,
    "blood in stool": 30,
    "high fever": 25,
    "fever": 10,
    "cough": 8,
    "cold symptoms": 5,
    "headache": 8,
    "migraine": 15,
    "vomiting": 12,
    "nausea": 8,
    "diarrhea": 12,
    "fatigue": 8,
    "weakness": 8,
    "sore throat": 7,
    "dizziness": 15,
    "abdominal pain": 20,
    "back pain": 10,
    "joint pain": 10,
    "rash": 8,
    "itching": 5,
    "anxiety": 8,
    "sleep difficulty": 5,
}


# ============================================================
# 4. TEXT PROCESSING AND NEGATION HANDLING
# ============================================================

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bnever\b",
    r"\bwithout\b",
    r"\bdenies\b",
    r"\bdeny\b",
    r"\bdo not have\b",
    r"\bdon't have\b",
    r"\bdoes not have\b",
    r"\bdoesn't have\b",
    r"\bdo not feel\b",
    r"\bdon't feel\b",
    r"\bno longer have\b",
]

NEGATION_RE = re.compile(
    "|".join(NEGATION_PATTERNS),
    flags=re.IGNORECASE,
)


def is_negated_mention(text, start_index):
    """
    Approximate negation handling by inspecting the short context preceding
    a symptom mention. This is not full clinical NLP and can make mistakes.
    """
    context_start = max(0, start_index - 55)
    preceding_text = text[context_start:start_index].lower()

    # Stop at sentence boundaries to reduce negation leaking into
    # a separate sentence.
    preceding_text = re.split(
        r"[.!?;\n]",
        preceding_text,
    )[-1]

    # Consider the final few words before the symptom.
    preceding_words = preceding_text.split()
    preceding_context = " ".join(preceding_words[-8:])

    # "No history of chest pain" is different from current chest pain.
    # History-specific language is not treated as a current symptom.
    if re.search(
        r"\b(history of|previously had|used to have)\b",
        preceding_context,
    ):
        return True

    # Basic scope check: a negation cue should be near the symptom mention.
    for pattern in NEGATION_PATTERNS:
        match = re.search(
            pattern + r"(?:\s+\w+){0,5}\s*$",
            preceding_context,
            flags=re.IGNORECASE,
        )
        if match:
            return True

    return False


def extract_symptoms(text):
    """
    Return present symptoms, negated mentions, and unmatched text status.
    A symptom is considered present if at least one detected mention is
    not judged to be negated by the approximate context rule.
    """
    normalised_text = text.lower()

    present = []
    negated = []

    for symptom, patterns in SYMPTOM_PATTERNS.items():
        found_present = False
        found_negated = False

        for pattern in patterns:
            for match in re.finditer(
                pattern,
                normalised_text,
                flags=re.IGNORECASE,
            ):
                if is_negated_mention(
                    normalised_text,
                    match.start(),
                ):
                    found_negated = True
                else:
                    found_present = True

        if found_present:
            present.append(symptom)
        elif found_negated:
            negated.append(symptom)

    return sorted(set(present)), sorted(set(negated))


def calculate_heuristic_score(detected_symptoms):
    """
    Produce a capped weighted indicator for demonstration only.
    This is NOT a validated severity or triage score.
    """
    total = sum(
        SYMPTOM_WEIGHTS.get(symptom, 0)
        for symptom in detected_symptoms
    )

    return min(100, total)


def get_heuristic_band(score):
    if score >= 75:
        return "Higher heuristic score"
    if score >= 45:
        return "Moderate-to-higher heuristic score"
    if score >= 20:
        return "Moderate heuristic score"
    return "Lower heuristic score"


# ============================================================
# 5. PROPOSITIONAL KNOWLEDGE BASE
# ============================================================
#
# Facts use uppercase symbolic names.
# Rules are Horn rules:
#     IF all premises are true, THEN the conclusion is true.
#
# The rule system is intentionally limited to transparent warning
# statements. It does not infer a specific disease.
# ============================================================

RULES = [
    {
        "name": "R1",
        "premises": ["CHEST_PAIN"],
        "conclusion": "URGENT_ASSESSMENT_NEEDED",
        "explanation": (
            "Chest pain can have serious causes and may require urgent assessment."
        ),
    },
    {
        "name": "R2",
        "premises": ["BREATHING_DIFFICULTY"],
        "conclusion": "URGENT_ASSESSMENT_NEEDED",
        "explanation": (
            "Difficulty breathing can be a medical emergency."
        ),
    },
    {
        "name": "R3",
        "premises": ["LOSS_OF_CONSCIOUSNESS"],
        "conclusion": "URGENT_ASSESSMENT_NEEDED",
        "explanation": (
            "Loss of consciousness requires urgent medical assessment."
        ),
    },
    {
        "name": "R4",
        "premises": ["SEIZURE"],
        "conclusion": "URGENT_ASSESSMENT_NEEDED",
        "explanation": (
            "A seizure may require urgent assessment, especially if ongoing, "
            "first-time, prolonged, or followed by difficulty waking."
        ),
    },
    {
        "name": "R5",
        "premises": ["STROKE_WARNING_SIGNS"],
        "conclusion": "URGENT_ASSESSMENT_NEEDED",
        "explanation": (
            "Sudden facial droop, one-sided weakness, or speech difficulty "
            "can be signs of stroke."
        ),
    },
    {
        "name": "R6",
        "premises": ["SEVERE_BLEEDING"],
        "conclusion": "URGENT_ASSESSMENT_NEEDED",
        "explanation": (
            "Severe or uncontrolled bleeding requires urgent medical help."
        ),
    },
    {
        "name": "R7",
        "premises": ["COUGHING_BLOOD"],
        "conclusion": "PROMPT_MEDICAL_REVIEW",
        "explanation": (
            "Coughing blood should be medically assessed; urgency depends "
            "on amount, breathing, and other symptoms."
        ),
    },
    {
        "name": "R8",
        "premises": ["BLOOD_IN_STOOL"],
        "conclusion": "PROMPT_MEDICAL_REVIEW",
        "explanation": (
            "Blood in stool warrants medical assessment, particularly if "
            "heavy, recurrent, or associated with weakness or faintness."
        ),
    },
    {
        "name": "R9",
        "premises": ["HIGH_FEVER"],
        "conclusion": "PROMPT_MEDICAL_REVIEW",
        "explanation": (
            "A very high fever may need medical assessment, depending on "
            "age, duration, and accompanying symptoms."
        ),
    },
    {
        "name": "R10",
        "premises": ["PERSISTENT_SYMPTOMS"],
        "conclusion": "MEDICAL_REVIEW_MAY_BE_NEEDED",
        "explanation": (
            "Persistent, worsening, or unexplained symptoms may need "
            "professional evaluation."
        ),
    },
    {
        "name": "R11",
        "premises": ["URGENT_ASSESSMENT_NEEDED"],
        "conclusion": "URGENT_WARNING",
        "explanation": (
            "A rule indicating urgent assessment has been triggered."
        ),
    },
]


SYMPTOM_TO_FACT = {
    "chest pain": "CHEST_PAIN",
    "difficulty breathing": "BREATHING_DIFFICULTY",
    "loss of consciousness": "LOSS_OF_CONSCIOUSNESS",
    "seizure": "SEIZURE",
    "stroke warning signs": "STROKE_WARNING_SIGNS",
    "severe bleeding": "SEVERE_BLEEDING",
    "coughing blood": "COUGHING_BLOOD",
    "blood in stool": "BLOOD_IN_STOOL",
    "high fever": "HIGH_FEVER",
}


# ============================================================
# 6. FORWARD CHAINING
# ============================================================

def forward_chaining(initial_facts, rules):
    """
    Data-driven inference:
    repeatedly fire every rule whose premises are all known facts.

    Returns the complete derived fact set and an inference trace.
    """
    facts = set(initial_facts)
    trace = []
    fired_rules = set()

    changed = True

    while changed:
        changed = False

        for rule in rules:
            rule_name = rule["name"]

            if rule_name in fired_rules:
                continue

            if all(
                premise in facts
                for premise in rule["premises"]
            ):
                conclusion = rule["conclusion"]

                if conclusion not in facts:
                    facts.add(conclusion)
                    changed = True

                fired_rules.add(rule_name)

                trace.append({
                    "rule": rule_name,
                    "premises": list(rule["premises"]),
                    "conclusion": conclusion,
                    "explanation": rule["explanation"],
                })

    return facts, trace


# ============================================================
# 7. BACKWARD CHAINING
# ============================================================

def backward_chaining(goal, facts, rules):
    """
    Goal-driven inference:
    start with a requested conclusion and recursively prove its premises.

    Returns whether the goal is provable and a trace of the attempted proof.
    """
    trace = []
    active_goals = set()
    memo = {}

    def prove(current_goal):
        if current_goal in facts:
            trace.append({
                "goal": current_goal,
                "result": "PROVED",
                "reason": "The fact is present in the initial or derived facts.",
            })
            return True

        if current_goal in memo:
            return memo[current_goal]

        if current_goal in active_goals:
            trace.append({
                "goal": current_goal,
                "result": "NOT PROVED",
                "reason": "A cycle was detected in the rule dependencies.",
            })
            return False

        active_goals.add(current_goal)

        candidate_rules = [
            rule for rule in rules
            if rule["conclusion"] == current_goal
        ]

        if not candidate_rules:
            trace.append({
                "goal": current_goal,
                "result": "NOT PROVED",
                "reason": "No rule concludes this goal and it is not a known fact.",
            })
            active_goals.remove(current_goal)
            memo[current_goal] = False
            return False

        for rule in candidate_rules:
            premise_results = []

            for premise in rule["premises"]:
                premise_results.append(prove(premise))

            if all(premise_results):
                trace.append({
                    "goal": current_goal,
                    "result": "PROVED",
                    "reason": (
                        f"Rule {rule['name']} succeeded because all premises "
                        "were proved."
                    ),
                })
                active_goals.remove(current_goal)
                memo[current_goal] = True
                return True

        trace.append({
            "goal": current_goal,
            "result": "NOT PROVED",
            "reason": "The premises of the available rules could not all be proved.",
        })

        active_goals.remove(current_goal)
        memo[current_goal] = False
        return False

    result = prove(goal)

    return result, trace


# ============================================================
# 8. RESOLUTION
# ============================================================
#
# Literals are represented as strings:
#     "CHEST_PAIN"    positive literal
#     "~CHEST_PAIN"   negative literal
#
# A clause is a frozenset of literals. For example:
#     {"~CHEST_PAIN", "URGENT_ASSESSMENT_NEEDED"}
# represents:
#     NOT CHEST_PAIN OR URGENT_ASSESSMENT_NEEDED
#
# To test whether KB entails query Q, resolution adds NOT Q and searches
# for the empty clause. This is refutation-based entailment.
# ============================================================

def negate_literal(literal):
    if literal.startswith("~"):
        return literal[1:]
    return "~" + literal


def rule_to_clause(rule):
    """
    Convert:
        A AND B -> C
    into:
        NOT A OR NOT B OR C
    """
    literals = {
        "~" + premise
        for premise in rule["premises"]
    }

    literals.add(rule["conclusion"])

    return frozenset(literals)


def resolution_pair(clause_a, clause_b):
    """
    Resolve two clauses on each complementary literal.
    Returns all non-tautological resolvents.
    """
    resolvents = set()

    for literal in clause_a:
        complement = negate_literal(literal)

        if complement in clause_b:
            new_clause = (
                set(clause_a)
                - {literal}
            ) | (
                set(clause_b)
                - {complement}
            )

            # A tautological clause contains both P and NOT P.
            tautology = any(
                negate_literal(item) in new_clause
                for item in new_clause
            )

            if not tautology:
                resolvents.add(frozenset(new_clause))

    return resolvents


def resolution_entails(initial_facts, rules, query, max_clauses=4000):
    """
    Check entailment through resolution refutation.

    Initial facts are treated as positive unit clauses.
    Rules are converted to CNF clauses.
    The negated query is added as a unit clause.

    If the empty clause is derived, the query is entailed by the encoded KB.
    If saturation or the safety limit is reached without contradiction,
    the query is not proven by this procedure.
    """
    clauses = set()

    for fact in initial_facts:
        clauses.add(frozenset([fact]))

    for rule in rules:
        clauses.add(rule_to_clause(rule))

    # Refutation: add NOT query.
    clauses.add(frozenset([negate_literal(query)]))

    known = set(clauses)
    agenda = list(clauses)
    resolution_steps = []
    step_count = 0

    while agenda and len(known) <= max_clauses:
        clause_a = agenda.pop(0)

        for clause_b in list(known):
            if clause_a == clause_b:
                continue

            for resolvent in resolution_pair(clause_a, clause_b):
                step_count += 1

                if len(resolvent) == 0:
                    resolution_steps.append({
                        "step": step_count,
                        "left": sorted(clause_a),
                        "right": sorted(clause_b),
                        "resolvent": [],
                        "result": "EMPTY CLAUSE — contradiction found",
                    })

                    return {
                        "entailed": True,
                        "steps": resolution_steps,
                        "clauses_examined": len(known),
                        "reason": (
                            "The empty clause was derived after adding the "
                            "negation of the query."
                        ),
                    }

                if resolvent not in known:
                    known.add(resolvent)
                    agenda.append(resolvent)

                    resolution_steps.append({
                        "step": step_count,
                        "left": sorted(clause_a),
                        "right": sorted(clause_b),
                        "resolvent": sorted(resolvent),
                        "result": "New clause derived",
                    })

                    if len(known) > max_clauses:
                        break

            if len(known) > max_clauses:
                break

    return {
        "entailed": False,
        "steps": resolution_steps,
        "clauses_examined": len(known),
        "reason": (
            "No contradiction was derived before the clause limit or "
            "saturation. The query is not proven by this run; this does "
            "not establish that the conclusion is medically false."
        ),
    }


# ============================================================
# 9. MAP SYMPTOMS TO KNOWLEDGE-BASE FACTS
# ============================================================

def build_initial_facts(detected_symptoms, persistent_symptoms):
    facts = set()

    for symptom in detected_symptoms:
        if symptom in SYMPTOM_TO_FACT:
            facts.add(SYMPTOM_TO_FACT[symptom])

    if persistent_symptoms:
        facts.add("PERSISTENT_SYMPTOMS")

    return facts


def humanise_fact(fact):
    return fact.replace("_", " ").title()


# ============================================================
# 10. USER INTERFACE
# ============================================================

st.subheader("Patient information")

with st.form("symptom_checker_form"):

    col1, col2 = st.columns(2)

    with col1:
        patient_name = st.text_input(
            "Name (optional)",
            max_chars=100,
        )

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=25,
            step=1,
        )

    with col2:
        gender = st.selectbox(
            "Gender (optional)",
            ["Prefer not to say", "Female", "Male", "Other"],
        )

        duration = st.selectbox(
            "How long have the symptoms been present?",
            [
                "Started today",
                "1–3 days",
                "4–7 days",
                "More than a week",
                "Not sure",
            ],
        )

    symptoms_text = st.text_area(
        "Describe the symptoms",
        height=160,
        placeholder=(
            "Example: I have had a headache and fever since yesterday. "
            "I do not have chest pain."
        ),
        help=(
            "Describe what is happening now, when it started, severity, "
            "and whether symptoms are getting worse."
        ),
    )

    persistent_symptoms = st.checkbox(
        "Symptoms are persistent, worsening, or unexplained",
        value=False,
    )

    submit = st.form_submit_button(
        "Analyse Symptoms",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 11. ANALYSIS PIPELINE
# ============================================================

if submit:

    if not symptoms_text.strip():
        st.error("Enter a description of your symptoms before analysing.")
        st.stop()

    detected_symptoms, negated_symptoms = extract_symptoms(
        symptoms_text
    )

    initial_facts = build_initial_facts(
        detected_symptoms,
        persistent_symptoms,
    )

    # Forward chaining: derive all conclusions from known facts.
    derived_facts, forward_trace = forward_chaining(
        initial_facts,
        RULES,
    )

    # Backward chaining: explicitly attempt to prove each key goal.
    urgent_proven, backward_urgent_trace = backward_chaining(
        "URGENT_ASSESSMENT_NEEDED",
        initial_facts,
        RULES,
    )

    review_proven, backward_review_trace = backward_chaining(
        "PROMPT_MEDICAL_REVIEW",
        initial_facts,
        RULES,
    )

    # Resolution: test whether urgent assessment is entailed by the
    # propositional knowledge base and the detected facts.
    resolution_result = resolution_entails(
        initial_facts,
        RULES,
        "URGENT_ASSESSMENT_NEEDED",
    )

    heuristic_score = calculate_heuristic_score(
        detected_symptoms
    )

    emergency_detected = any(
        symptom in EMERGENCY_SYMPTOMS
        for symptom in detected_symptoms
    )

    # A hard-coded conservative red-flag check supplements the rule engine.
    # The algorithms must never be treated as the only safety mechanism.
    urgent_flag = emergency_detected or urgent_proven

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_name": patient_name.strip() or "Not provided",
        "age": age,
        "gender": gender,
        "duration": duration,
        "symptoms_text": symptoms_text,
        "persistent_symptoms": persistent_symptoms,
        "detected_symptoms": detected_symptoms,
        "negated_symptoms": negated_symptoms,
        "initial_facts": sorted(initial_facts),
        "derived_facts": sorted(derived_facts),
        "forward_trace": forward_trace,
        "urgent_proven": urgent_proven,
        "review_proven": review_proven,
        "backward_urgent_trace": backward_urgent_trace,
        "backward_review_trace": backward_review_trace,
        "resolution_result": resolution_result,
        "heuristic_score": heuristic_score,
        "heuristic_band": get_heuristic_band(heuristic_score),
        "emergency_detected": emergency_detected,
        "urgent_flag": urgent_flag,
    }

    st.session_state["symptom_checker_result"] = result
    st.session_state.pop("symptom_checker_ai", None)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

if "symptom_checker_result" in st.session_state:

    result = st.session_state["symptom_checker_result"]

    st.divider()
    st.header("📊 Assessment overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Detected symptom terms",
        len(result["detected_symptoms"]),
    )

    c2.metric(
        "Heuristic indicator",
        f"{result['heuristic_score']}/100",
    )

    c3.metric(
        "Urgent rule flag",
        "YES" if result["urgent_flag"] else "Not triggered",
    )

    st.caption(
        "The indicator is an arbitrary weighted sum of matched terms, "
        "not a validated medical severity score. A low score does not "
        "mean that the symptoms are safe."
    )

    # --------------------------------------------------------
    # EMERGENCY GUIDANCE
    # --------------------------------------------------------

    st.subheader("🚨 Safety guidance")

    if result["urgent_flag"]:
        st.error(
            "One or more warning-sign rules were triggered. Seek urgent "
            "medical assessment. If there is severe chest pain, serious "
            "difficulty breathing, stroke warning signs, loss of "
            "consciousness, a seizure emergency, or uncontrolled bleeding, "
            "contact local emergency services now. Do not wait for AI output."
        )
    else:
        st.info(
            "The configured emergency rules were not triggered by the "
            "matched symptom terms. This does not rule out an emergency. "
            "Seek care if symptoms are severe, worsening, unusual, or concerning."
        )

    # --------------------------------------------------------
    # SYMPTOM EXTRACTION
    # --------------------------------------------------------

    st.divider()
    st.header("🔎 Symptom extraction")

    if result["detected_symptoms"]:
        st.write("**Terms treated as present:**")

        for symptom in result["detected_symptoms"]:
            st.write(f"- {symptom.title()}")
    else:
        st.warning(
            "No symptoms from the configured vocabulary were confidently "
            "matched. The checker may not recognise your wording."
        )

    if result["negated_symptoms"]:
        st.write("**Terms detected in a negated or historical context:**")

        for symptom in result["negated_symptoms"]:
            st.write(f"- {symptom.title()}")

        st.caption(
            "Negation detection is approximate. Review the extracted terms "
            "carefully, especially if a warning symptom is actually present."
        )

    st.write(f"**Reported duration:** {result['duration']}")

    # --------------------------------------------------------
    # FORWARD CHAINING
    # --------------------------------------------------------

    st.divider()
    st.header("➡️ Algorithm 1: Forward Chaining")

    st.write(
        "Forward chaining starts with the facts extracted from the symptom "
        "description, then repeatedly applies rules whose premises are true."
    )

    if result["initial_facts"]:
        st.write("**Initial facts:**")

        for fact in result["initial_facts"]:
            st.code(humanise_fact(fact))
    else:
        st.write(
            "No predefined rule facts were created from the matched symptoms."
        )

    if result["forward_trace"]:
        st.write("**Rules fired:**")

        forward_table = pd.DataFrame([
            {
                "Rule": item["rule"],
                "Premises": ", ".join(
                    humanise_fact(p)
                    for p in item["premises"]
                ),
                "Conclusion": humanise_fact(item["conclusion"]),
                "Explanation": item["explanation"],
            }
            for item in result["forward_trace"]
        ])

        st.dataframe(
            forward_table,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info(
            "No inference rules fired. This means only that no configured "
            "rule matched the available facts."
        )

    st.write("**Final fact set:**")

    st.write(
        ", ".join(
            humanise_fact(fact)
            for fact in result["derived_facts"]
        ) or "No facts"
    )

    # --------------------------------------------------------
    # BACKWARD CHAINING
    # --------------------------------------------------------

    st.divider()
    st.header("⬅️ Algorithm 2: Backward Chaining")

    st.write(
        "Backward chaining starts with a goal and works backward through "
        "rules to check whether its premises can be established."
    )

    b1, b2 = st.columns(2)

    with b1:
        st.subheader("Goal: urgent assessment")

        st.write(
            "**Result:** "
            + (
                "Supported by the configured rule base"
                if result["urgent_proven"]
                else "Not proven by the configured rule base"
            )
        )

        st.caption(
            "Not proven does not mean medically safe."
        )

    with b2:
        st.subheader("Goal: prompt medical review")

        st.write(
            "**Result:** "
            + (
                "Supported by the configured rule base"
                if result["review_proven"]
                else "Not proven by the configured rule base"
            )
        )

    with st.expander("View urgent-assessment proof trace"):
        if result["backward_urgent_trace"]:
            for item in result["backward_urgent_trace"]:
                st.write(
                    f"- **{humanise_fact(item['goal'])}:** "
                    f"{item['result']} — {item['reason']}"
                )
        else:
            st.write("No proof steps were generated.")

    with st.expander("View prompt-review proof trace"):
        if result["backward_review_trace"]:
            for item in result["backward_review_trace"]:
                st.write(
                    f"- **{humanise_fact(item['goal'])}:** "
                    f"{item['result']} — {item['reason']}"
                )
        else:
            st.write("No proof steps were generated.")

    # --------------------------------------------------------
    # RESOLUTION
    # --------------------------------------------------------

    st.divider()
    st.header("🧩 Algorithm 3: Propositional Resolution")

    st.write(
        "Resolution converts the rules into clauses in conjunctive normal "
        "form (CNF), adds the negation of the target conclusion, and tries "
        "to derive the empty clause by resolving complementary literals."
    )

    resolution_result = result["resolution_result"]

    if resolution_result["entailed"]:
        st.success(
            "The query URGENT_ASSESSMENT_NEEDED was entailed by the "
            "encoded facts and rules using refutation."
        )
    else:
        st.info(
            "Resolution did not prove URGENT_ASSESSMENT_NEEDED from the "
            "encoded facts and rules. This is not evidence that the person "
            "is medically safe."
        )

    st.write(f"**Resolution steps recorded:** {len(resolution_result['steps'])}")
    st.write(
        f"**Distinct clauses examined:** "
        f"{resolution_result['clauses_examined']}"
    )
    st.caption(resolution_result["reason"])

    with st.expander("View resolution derivation"):
        if resolution_result["steps"]:
            resolution_table = pd.DataFrame([
                {
                    "Step": item["step"],
                    "Clause A": " OR ".join(item["left"]) or "EMPTY",
                    "Clause B": " OR ".join(item["right"]) or "EMPTY",
                    "Resolvent": " OR ".join(item["resolvent"]) or "EMPTY CLAUSE",
                    "Result": item["result"],
                }
                for item in resolution_result["steps"]
            ])

            st.dataframe(
                resolution_table,
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.write(
                "No new resolvents were generated for this knowledge base "
                "and query."
            )

    # --------------------------------------------------------
    # RULE FINDINGS
    # --------------------------------------------------------

    st.divider()
    st.header("📋 Rule-based findings")

    if "URGENT_ASSESSMENT_NEEDED" in result["derived_facts"]:
        st.error(
            "The rule engine derived: urgent assessment needed."
        )

    if "PROMPT_MEDICAL_REVIEW" in result["derived_facts"]:
        st.warning(
            "The rule engine derived: prompt medical review."
        )

    if "MEDICAL_REVIEW_MAY_BE_NEEDED" in result["derived_facts"]:
        st.info(
            "The rule engine derived: medical review may be needed."
        )

    if not any(
        fact in result["derived_facts"]
        for fact in [
            "URGENT_ASSESSMENT_NEEDED",
            "PROMPT_MEDICAL_REVIEW",
            "MEDICAL_REVIEW_MAY_BE_NEEDED",
        ]
    ):
        st.info(
            "No configured medical-review conclusion was derived. "
            "Unrecognised or unmodelled symptoms may still need assessment."
        )

    # --------------------------------------------------------
    # OPTIONAL GEMINI EXPLANATION
    # --------------------------------------------------------

    st.divider()
    st.header("🤖 Optional AI explanation")

    if st.button("Generate general health guidance"):

        if not GEMINI_API_KEY.strip() or (
            GEMINI_API_KEY == "PASTE_YOUR_API_KEY_HERE"
        ):
            st.warning(
                "Add your Gemini API key to GEMINI_API_KEY near the top "
                "of the file to enable this optional feature."
            )
        else:
            try:
                import google.generativeai as genai

                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel(GEMINI_MODEL)

                prompt = f"""
You are a cautious health-information assistant, not a doctor.

Explain the information below in plain language. Do not diagnose the
patient, claim that a condition is ruled out, or invent a probability.
Do not contradict emergency guidance. If warning signs are present,
recommend urgent assessment before any general lifestyle advice.

Age: {result['age']}
Reported duration: {result['duration']}
Symptom description: {result['symptoms_text']}
Matched symptom terms: {', '.join(result['detected_symptoms'])}
Negated/historical terms: {', '.join(result['negated_symptoms'])}
Urgent rule flag: {result['urgent_flag']}
Derived facts: {', '.join(result['derived_facts'])}

Provide:
1. What the tool detected and what it cannot determine.
2. General information about why the symptoms may need attention.
3. What information to tell a healthcare professional.
4. Warning signs that should prompt emergency help.

Clearly state that this is not a medical diagnosis.
"""

                with st.spinner("Generating general health information..."):
                    response = model.generate_content(prompt)

                ai_text = getattr(response, "text", None)

                if ai_text:
                    st.session_state["symptom_checker_ai"] = ai_text
                else:
                    st.warning("The AI service returned no text.")

            except Exception as exc:
                st.error(
                    "AI guidance could not be generated. Check the API key, "
                    "model availability, package version, and connection. "
                    f"Details: {exc}"
                )

    if st.session_state.get("symptom_checker_ai"):
        st.markdown(st.session_state["symptom_checker_ai"])

    # --------------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------------

    st.divider()
    st.header("📄 Download assessment report")

    report_lines = [
        "AI SYMPTOM CHECKER REPORT",
        "=" * 45,
        f"Generated: {result['timestamp']}",
        "",
        "PATIENT INFORMATION",
        f"Name: {result['patient_name']}",
        f"Age: {result['age']}",
        f"Gender: {result['gender']}",
        f"Symptom duration: {result['duration']}",
        "",
        "ORIGINAL SYMPTOM DESCRIPTION",
        result["symptoms_text"],
        "",
        "DETECTED TERMS",
        ", ".join(result["detected_symptoms"]) or "None matched",
        "",
        "NEGATED OR HISTORICAL TERMS",
        ", ".join(result["negated_symptoms"]) or "None detected",
        "",
        "HEURISTIC INDICATOR (NOT CLINICALLY VALIDATED)",
        f"Score: {result['heuristic_score']}/100",
        f"Band: {result['heuristic_band']}",
        "",
        "INITIAL FACTS",
        ", ".join(result["initial_facts"]) or "None",
        "",
        "FORWARD CHAINING — DERIVED FACTS",
        ", ".join(result["derived_facts"]) or "None",
        "",
        "BACKWARD CHAINING",
        "Urgent assessment proven: " + str(result["urgent_proven"]),
        "Prompt review proven: " + str(result["review_proven"]),
        "",
        "RESOLUTION",
        "Urgent assessment entailed: "
        + str(result["resolution_result"]["entailed"]),
        "Reason: " + result["resolution_result"]["reason"],
        "",
        "SAFETY",
        "Urgent rule flag: " + str(result["urgent_flag"]),
        "",
        "DISCLAIMER",
        "This educational report is not a diagnosis, a validated triage "
        "assessment, or a substitute for a qualified healthcare professional.",
        "The symptom vocabulary and negation handling are incomplete.",
        "Failure to trigger a rule does not rule out a serious condition.",
    ]

    ai_text = st.session_state.get("symptom_checker_ai", "")

    if ai_text:
        report_lines.extend([
            "",
            "OPTIONAL AI EXPLANATION",
            ai_text,
        ])

    report = "\n".join(report_lines)

    st.download_button(
        "📥 Download symptom report (.txt)",
        data=report,
        file_name="Symptom_Checker_Report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    if st.button("🔄 Start a new assessment"):
        st.session_state.pop("symptom_checker_result", None)
        st.session_state.pop("symptom_checker_ai", None)
        st.rerun()


# ============================================================
# 13. ALGORITHM SUMMARY
# ============================================================

st.divider()
st.subheader("Algorithm implementation summary")

st.markdown(
    """
    **Forward chaining**
    - Begins with detected facts.
    - Fires rules when all premises are present.
    - Repeats until no new conclusions are added.
    - Displays each fired rule and its conclusion.

    **Backward chaining**
    - Begins with a target conclusion.
    - Recursively checks rules that could establish that goal.
    - Displays the proof attempt and whether the goal was established
      by the configured rule base.

    **Propositional resolution**
    - Converts Horn rules to CNF clauses.
    - Adds the negation of the target query.
    - Resolves complementary literals.
    - Checks whether the empty clause can be derived.

    These algorithms implement logical inference over a small, explicit
    knowledge base. They do not infer diseases from a complete medical
    knowledge system and are not clinically validated.
    """
)

st.info(
    "Emergency decisions should never depend solely on this application. "
    "Seek urgent professional help for severe, sudden, or rapidly worsening "
    "symptoms, even if no warning rule is triggered."
)
