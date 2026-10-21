
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


# ============================================================
# 1. OPTIONAL GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = "AQ.Ab8RN6LDq8jPgJwvOqXjtYwkyiWbDgzO1YhoMA-uwQXCQ-85sQ"
GEMINI_MODEL = "gemini-3.8-flash"

# The calculator and both algorithms work without Gemini.


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Water Intake Calculator",
    page_icon="💧",
    layout="wide",
)

st.title("💧 AI Daily Water Intake Calculator")

st.write(
    "Estimate daily water needs using a fuzzy rule-based system, "
    "a Decision Tree Classifier, and optional AI-generated guidance."
)

st.caption(
    "This tool provides an educational estimate for adults. It does not "
    "measure hydration status or diagnose dehydration."
)


# ============================================================
# 3. REFERENCE VALUES AND ACTIVITY ENCODING
# ============================================================

ACTIVITY_LEVELS = {
    "Sedentary": 0,
    "Lightly Active": 1,
    "Moderately Active": 2,
    "Very Active": 3,
    "Athlete": 4,
}

CLIMATE_LEVELS = {
    "Cold": 0,
    "Moderate": 1,
    "Hot": 2,
}

# Illustrative daily fluid starting estimate.
# This is not a universal clinical requirement.
BASE_FLUID_ML_PER_KG = 35.0

GLASS_SIZE_ML = 250


# ============================================================
# 4. FUZZY MEMBERSHIP FUNCTIONS
# ============================================================

def triangular_membership(x, a, b, c):
    """
    Triangular fuzzy membership in [0, 1].
    Requires a < b < c.
    """
    if x <= a or x >= c:
        return 0.0

    if x == b:
        return 1.0

    if x < b:
        return (x - a) / (b - a)

    return (c - x) / (c - b)


def left_shoulder_membership(x, full_until, zero_from):
    """Membership is 1 below full_until and declines to 0."""
    if x <= full_until:
        return 1.0

    if x >= zero_from:
        return 0.0

    return (zero_from - x) / (zero_from - full_until)


def right_shoulder_membership(x, zero_until, full_from):
    """Membership is 0 below zero_until and rises to 1."""
    if x <= zero_until:
        return 0.0

    if x >= full_from:
        return 1.0

    return (x - zero_until) / (full_from - zero_until)


def exercise_memberships(minutes):
    """
    Fuzzy sets for daily exercise duration.
    Overlap is intentional so boundary values can belong to more
    than one linguistic category.
    """
    return {
        "Low": left_shoulder_membership(
            minutes, full_until=15, zero_from=45
        ),
        "Moderate": triangular_membership(
            minutes, a=20, b=60, c=110
        ),
        "High": right_shoulder_membership(
            minutes, zero_until=75, full_from=150
        ),
    }


def activity_memberships(activity):
    """
    Convert the selected activity level into transparent linguistic
    membership values. These are modelling assumptions, not physiology.
    """
    level = ACTIVITY_LEVELS[activity]

    return {
        "Low": left_shoulder_membership(
            level, full_until=0, zero_from=2
        ),
        "Moderate": triangular_membership(
            level, a=0, b=2, c=4
        ),
        "High": right_shoulder_membership(
            level, zero_until=2, full_from=4
        ),
    }


def climate_memberships(climate):
    """
    Fuzzy encoding of the selected climate.
    The selected category has full membership; adjacent categories
    are not blended because no temperature measurement is provided.
    """
    return {
        "Cold": 1.0 if climate == "Cold" else 0.0,
        "Moderate": 1.0 if climate == "Moderate" else 0.0,
        "Hot": 1.0 if climate == "Hot" else 0.0,
    }


# ============================================================
# 5. FUZZY IF-THEN RULE BASE
# ============================================================
#
# Sugeno-style fuzzy inference:
# - Each rule has a firing strength in [0, 1].
# - AND is implemented using minimum.
# - Rule outputs are fixed fluid adjustments in litres.
# - The final adjustment is the weighted average of fired rules.
#
# The values below are illustrative planning assumptions, not
# clinically validated hydration recommendations.
# ============================================================

FUZZY_RULES = [
    {
        "name": "Low exercise + cold climate",
        "exercise": "Low",
        "activity": None,
        "climate": "Cold",
        "output_l": 0.00,
    },
    {
        "name": "Low exercise + moderate climate",
        "exercise": "Low",
        "activity": None,
        "climate": "Moderate",
        "output_l": 0.00,
    },
    {
        "name": "Low exercise + hot climate",
        "exercise": "Low",
        "activity": None,
        "climate": "Hot",
        "output_l": 0.30,
    },
    {
        "name": "Moderate exercise + cold climate",
        "exercise": "Moderate",
        "activity": None,
        "climate": "Cold",
        "output_l": 0.10,
    },
    {
        "name": "Moderate exercise + moderate climate",
        "exercise": "Moderate",
        "activity": None,
        "climate": "Moderate",
        "output_l": 0.25,
    },
    {
        "name": "Moderate exercise + hot climate",
        "exercise": "Moderate",
        "activity": None,
        "climate": "Hot",
        "output_l": 0.60,
    },
    {
        "name": "High exercise + cold climate",
        "exercise": "High",
        "activity": None,
        "climate": "Cold",
        "output_l": 0.30,
    },
    {
        "name": "High exercise + moderate climate",
        "exercise": "High",
        "activity": None,
        "climate": "Moderate",
        "output_l": 0.60,
    },
    {
        "name": "High exercise + hot climate",
        "exercise": "High",
        "activity": None,
        "climate": "Hot",
        "output_l": 1.00,
    },
    {
        "name": "High activity + moderate climate",
        "exercise": None,
        "activity": "High",
        "climate": "Moderate",
        "output_l": 0.30,
    },
    {
        "name": "High activity + hot climate",
        "exercise": None,
        "activity": "High",
        "climate": "Hot",
        "output_l": 0.60,
    },
    {
        "name": "Moderate activity + hot climate",
        "exercise": None,
        "activity": "Moderate",
        "climate": "Hot",
        "output_l": 0.30,
    },
]


def run_fuzzy_inference(exercise_minutes, activity, climate):
    """
    Apply the fuzzy rules and return the weighted fluid adjustment,
    rule firing strengths, and input membership values.
    """
    exercise_mu = exercise_memberships(exercise_minutes)
    activity_mu = activity_memberships(activity)
    climate_mu = climate_memberships(climate)

    fired_rules = []

    for rule in FUZZY_RULES:
        conditions = []

        if rule["exercise"] is not None:
            conditions.append(exercise_mu[rule["exercise"]])

        if rule["activity"] is not None:
            conditions.append(activity_mu[rule["activity"]])

        if rule["climate"] is not None:
            conditions.append(climate_mu[rule["climate"]])

        firing_strength = min(conditions) if conditions else 0.0

        fired_rules.append({
            "Rule": rule["name"],
            "Firing strength": firing_strength,
            "Output adjustment (L)": rule["output_l"],
            "Weighted output": firing_strength * rule["output_l"],
        })

    denominator = sum(
        row["Firing strength"] for row in fired_rules
    )

    if denominator > 0:
        adjustment_l = sum(
            row["Weighted output"] for row in fired_rules
        ) / denominator
    else:
        adjustment_l = 0.0

    # Additional activity adjustment is applied separately so the
    # chosen activity level is not ignored when exercise duration is low.
    activity_adjustments = {
        "Sedentary": 0.00,
        "Lightly Active": 0.10,
        "Moderately Active": 0.20,
        "Very Active": 0.35,
        "Athlete": 0.50,
    }

    activity_adjustment_l = activity_adjustments[activity]

    # Bound the illustrative adjustment to avoid unbounded outputs.
    total_adjustment_l = min(
        1.50,
        max(0.0, adjustment_l + activity_adjustment_l),
    )

    return {
        "adjustment_l": total_adjustment_l,
        "rule_adjustment_l": adjustment_l,
        "activity_adjustment_l": activity_adjustment_l,
        "exercise_memberships": exercise_mu,
        "activity_memberships": activity_mu,
        "climate_memberships": climate_mu,
        "rules": fired_rules,
    }


# ============================================================
# 6. DECISION TREE TRAINING DATA
# ============================================================
#
# The classifier predicts an educational "estimated adjustment band":
# Low, Moderate, or High.
#
# Synthetic examples are generated using the same transparent fuzzy
# rule system. This demonstrates supervised Decision Tree training,
# classification, and path tracing. It is NOT trained on clinical data.
# ============================================================

def adjustment_category(adjustment_l):
    if adjustment_l < 0.35:
        return "Low adjustment"

    if adjustment_l < 0.80:
        return "Moderate adjustment"

    return "High adjustment"


def generate_synthetic_training_data():
    """
    Generate reproducible synthetic examples from the rule system.
    Inputs: exercise duration, activity code, climate code.
    Label: fuzzy-system adjustment band.
    """
    rows = []

    activity_names = list(ACTIVITY_LEVELS.keys())
    climate_names = list(CLIMATE_LEVELS.keys())

    # Dense, deterministic grid of inputs.
    for exercise in range(0, 181, 5):
        for activity_name in activity_names:
            for climate_name in climate_names:
                fuzzy = run_fuzzy_inference(
                    exercise_minutes=exercise,
                    activity=activity_name,
                    climate=climate_name,
                )

                rows.append({
                    "Exercise minutes": exercise,
                    "Activity code": ACTIVITY_LEVELS[activity_name],
                    "Climate code": CLIMATE_LEVELS[climate_name],
                    "Label": adjustment_category(
                        fuzzy["adjustment_l"]
                    ),
                })

    return pd.DataFrame(rows)


@st.cache_resource
def train_decision_tree():
    training_df = generate_synthetic_training_data()

    feature_columns = [
        "Exercise minutes",
        "Activity code",
        "Climate code",
    ]

    X = training_df[feature_columns]
    y = training_df["Label"]

    classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
    )

    classifier.fit(X, y)

    return classifier, training_df


def trace_decision_path(classifier, feature_values):
    """
    Return readable comparisons for each node traversed by the sample.
    """
    tree = classifier.tree_
    path_matrix = classifier.decision_path(feature_values)
    leaf_node = classifier.apply(feature_values)[0]

    visited_nodes = path_matrix.indices[
        path_matrix.indptr[0]:path_matrix.indptr[1]
    ]

    feature_names = [
        "Exercise minutes",
        "Activity code",
        "Climate code",
    ]

    path = []

    for node_id in visited_nodes:
        if node_id == leaf_node:
            class_counts = tree.value[node_id][0]
            predicted_index = int(np.argmax(class_counts))
            predicted_label = classifier.classes_[predicted_index]

            path.append(
                f"Leaf node: predicted class = {predicted_label}"
            )
            continue

        feature_index = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_name = feature_names[feature_index]
        value = float(feature_values[0][feature_index])

        if value <= threshold:
            comparison = "<="
            branch = "left"
        else:
            comparison = ">"
            branch = "right"

        path.append(
            f"{feature_name} ({value:g}) {comparison} "
            f"{threshold:.2f} → follow {branch} branch."
        )

    return path


# ============================================================
# 7. GENERAL SAFETY MESSAGES
# ============================================================

def get_context_messages(age, climate, exercise_minutes):
    messages = []

    if age >= 65:
        messages.append(
            "Older adults may have a reduced sense of thirst or medical "
            "conditions affecting fluid balance. Individual advice may "
            "be appropriate."
        )

    if climate == "Hot":
        messages.append(
            "Hot conditions can increase fluid losses. Take breaks, "
            "drink regularly, and consider the duration and intensity "
            "of activity."
        )

    if exercise_minutes >= 60:
        messages.append(
            "Longer exercise sessions may require an individual hydration "
            "plan. For prolonged heavy sweating, electrolytes may matter "
            "as well as water."
        )

    messages.append(
        "People with heart, kidney, or liver conditions, or those advised "
        "to restrict fluids, should follow their healthcare professional's "
        "fluid guidance rather than a generic calculator."
    )

    return messages


# ============================================================
# 8. USER INPUTS
# ============================================================

st.subheader("Personal and activity details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=120,
        value=25,
        step=1,
    )

    weight_kg = st.number_input(
        "Weight (kg)",
        min_value=10.0,
        max_value=300.0,
        value=70.0,
        step=0.5,
    )

with col2:
    activity = st.selectbox(
        "Activity level",
        list(ACTIVITY_LEVELS.keys()),
        index=0,
    )

    climate = st.selectbox(
        "Climate",
        list(CLIMATE_LEVELS.keys()),
        index=1,
    )

with col3:
    exercise_minutes = st.slider(
        "Daily exercise duration (minutes)",
        min_value=0,
        max_value=180,
        value=30,
        step=5,
    )

    base_ml_per_kg = st.slider(
        "Starting estimate (mL/kg/day)",
        min_value=25,
        max_value=40,
        value=int(BASE_FLUID_ML_PER_KG),
        step=1,
        help=(
            "This is a configurable planning assumption, not a universal "
            "medical recommendation. Total fluid needs also depend on food, "
            "health, environment, and other factors."
        ),
    )


calculate_button = st.button(
    "💧 Calculate and Run Algorithms",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 9. RUN CALCULATION
# ============================================================

if calculate_button:

    base_intake_ml = weight_kg * base_ml_per_kg

    fuzzy_result = run_fuzzy_inference(
        exercise_minutes=exercise_minutes,
        activity=activity,
        climate=climate,
    )

    adjustment_ml = fuzzy_result["adjustment_l"] * 1000.0

    # This is a planning estimate only, not measured physiological need.
    estimated_intake_ml = base_intake_ml + adjustment_ml
    estimated_intake_l = estimated_intake_ml / 1000.0

    # Avoid claiming this is the amount of plain water a person must drink.
    glasses = round(estimated_intake_ml / GLASS_SIZE_ML)

    classifier, training_df = train_decision_tree()

    features = [[
        exercise_minutes,
        ACTIVITY_LEVELS[activity],
        CLIMATE_LEVELS[climate],
    ]]

    tree_category = classifier.predict(features)[0]
    decision_path = trace_decision_path(classifier, features)

    probabilities = classifier.predict_proba(features)[0]
    class_probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(
            classifier.classes_,
            probabilities,
        )
    }

    result_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "age": age,
        "weight_kg": weight_kg,
        "activity": activity,
        "climate": climate,
        "exercise_minutes": exercise_minutes,
        "base_ml_per_kg": base_ml_per_kg,
        "base_intake_ml": base_intake_ml,
        "fuzzy_result": fuzzy_result,
        "estimated_intake_ml": estimated_intake_ml,
        "estimated_intake_l": estimated_intake_l,
        "glasses": glasses,
        "tree_category": tree_category,
        "decision_path": decision_path,
        "class_probabilities": class_probabilities,
        "training_rows": len(training_df),
    }

    st.session_state["water_result"] = result_payload


# ============================================================
# 10. RESULTS DASHBOARD
# ============================================================

if "water_result" in st.session_state:

    result = st.session_state["water_result"]

    fuzzy_result = result["fuzzy_result"]
    estimated_intake_ml = result["estimated_intake_ml"]
    estimated_intake_l = result["estimated_intake_l"]
    tree_category = result["tree_category"]

    st.divider()
    st.header("📊 Estimated Daily Fluid Planning")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Estimated fluid target",
        f"{estimated_intake_l:.2f} L",
    )

    c2.metric(
        "Estimated amount",
        f"{estimated_intake_ml:,.0f} mL",
    )

    c3.metric(
        "250 mL equivalents",
        f"{result['glasses']}",
    )

    c4.metric(
        "Fuzzy adjustment",
        f"+{fuzzy_result['adjustment_l']:.2f} L",
    )

    st.caption(
        "The litre estimate is generated from a configurable weight-based "
        "starting value plus illustrative fuzzy adjustments. It is not a "
        "measurement of hydration status or a guarantee of individual need. "
        "Total fluid can come from beverages and food."
    )

    st.info(
        f"Starting estimate: {result['base_intake_ml'] / 1000:.2f} L/day "
        f"({result['base_ml_per_kg']} mL/kg) + fuzzy/activity adjustment: "
        f"{fuzzy_result['adjustment_l']:.2f} L/day."
    )

    # --------------------------------------------------------
    # FUZZY INPUT MEMBERSHIPS
    # --------------------------------------------------------

    st.divider()
    st.header("🧠 Fuzzy Rule-Based Analysis")

    st.write(
        "The system calculates overlapping memberships, evaluates "
        "IF–THEN rules, and combines their outputs using a weighted average. "
        "Membership values describe the model's inputs; they are not "
        "probabilities of dehydration."
    )

    fuzzy_col1, fuzzy_col2, fuzzy_col3 = st.columns(3)

    with fuzzy_col1:
        st.subheader("Exercise membership")
        exercise_df = pd.DataFrame({
            "Fuzzy set": list(
                fuzzy_result["exercise_memberships"].keys()
            ),
            "Membership": list(
                fuzzy_result["exercise_memberships"].values()
            ),
        })
        st.dataframe(
            exercise_df.round(3),
            hide_index=True,
            use_container_width=True,
        )

    with fuzzy_col2:
        st.subheader("Activity membership")
        activity_df = pd.DataFrame({
            "Fuzzy set": list(
                fuzzy_result["activity_memberships"].keys()
            ),
            "Membership": list(
                fuzzy_result["activity_memberships"].values()
            ),
        })
        st.dataframe(
            activity_df.round(3),
            hide_index=True,
            use_container_width=True,
        )

    with fuzzy_col3:
        st.subheader("Climate membership")
        climate_df = pd.DataFrame({
            "Fuzzy set": list(
                fuzzy_result["climate_memberships"].keys()
            ),
            "Membership": list(
                fuzzy_result["climate_memberships"].values()
            ),
        })
        st.dataframe(
            climate_df.round(3),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Fuzzy rule evaluation")

    rule_df = pd.DataFrame(fuzzy_result["rules"])
    active_rules = rule_df[rule_df["Firing strength"] > 0].copy()

    if active_rules.empty:
        st.warning(
            "No fuzzy rules fired for this combination. The adjustment "
            "defaults to zero before the separate activity adjustment."
        )
    else:
        st.dataframe(
            active_rules.round(3),
            hide_index=True,
            use_container_width=True,
        )

    st.metric(
        "Combined fuzzy rule adjustment",
        f"{fuzzy_result['rule_adjustment_l']:.2f} L/day",
    )

    st.metric(
        "Separate activity adjustment",
        f"{fuzzy_result['activity_adjustment_l']:.2f} L/day",
    )

    with st.expander("View the complete fuzzy rule base"):
        st.dataframe(
            pd.DataFrame(FUZZY_RULES),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("How fuzzy inference works"):
        st.markdown(
            """
            1. **Fuzzification:** exercise duration and activity are
               represented by linguistic membership values.
            2. **Rule firing:** each IF–THEN rule calculates its strength.
               The AND operator uses the minimum membership.
            3. **Rule output:** each fired rule contributes a predefined
               fluid adjustment.
            4. **Aggregation:** a weighted average combines the rule outputs.
            5. **Final estimate:** the rule adjustment and separate activity
               adjustment are added to the selected starting estimate.

            The rule outputs are explicit modelling assumptions. They have
            not been clinically validated as hydration prescriptions.
            """
        )

    # --------------------------------------------------------
    # DECISION TREE CLASSIFICATION
    # --------------------------------------------------------

    st.divider()
    st.header("🌳 Decision Tree Classification")

    st.metric(
        "Predicted adjustment category",
        tree_category,
    )

    st.caption(
        "This classifier predicts a fuzzy-adjustment band, not hydration "
        "status, dehydration, or a medical risk. It is trained on synthetic "
        "examples generated from this app's own rule system."
    )

    st.subheader("Decision path for your inputs")

    for index, step in enumerate(result["decision_path"], start=1):
        st.write(f"{index}. {step}")

    st.subheader("Class scores")

    score_df = pd.DataFrame({
        "Class": list(result["class_probabilities"].keys()),
        "Model score": list(result["class_probabilities"].values()),
    })

    st.bar_chart(
        score_df.set_index("Class"),
    )

    st.caption(
        "These are classifier probability estimates for synthetic rule labels, "
        "not probabilities of dehydration or medical outcomes."
    )

    classifier, training_df = train_decision_tree()

    with st.expander("View the learned Decision Tree rules"):
        st.code(
            export_text(
                classifier,
                feature_names=[
                    "Exercise minutes",
                    "Activity code",
                    "Climate code",
                ],
                decimals=2,
            ),
            language="text",
        )

    with st.expander("View the Decision Tree diagram"):
        fig, ax = plt.subplots(figsize=(18, 8))

        plot_tree(
            classifier,
            feature_names=[
                "Exercise minutes",
                "Activity code",
                "Climate code",
            ],
            class_names=classifier.classes_,
            filled=True,
            rounded=True,
            impurity=True,
            ax=ax,
        )

        ax.set_title(
            "Educational Decision Tree for Estimated Adjustment Band"
        )
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Training data and algorithm limitations"):
        st.write(
            f"The classifier was trained on {len(training_df)} synthetic "
            "examples generated from exercise duration, activity level, "
            "and climate."
        )

        st.write(
            "The labels are derived from the fuzzy system's own output. "
            "This demonstrates the Decision Tree training and prediction "
            "process but does not establish clinical accuracy or independent "
            "validation."
        )

        st.write(
            "The tree uses Gini impurity to choose splits, has a maximum "
            "depth of 5, and requires at least 5 training examples per leaf."
        )

    # --------------------------------------------------------
    # HYDRATION PLANNING SCHEDULE
    # --------------------------------------------------------

    st.divider()
    st.header("🕒 Example Fluid Distribution Schedule")

    schedule = [
        ("Morning", 0.20),
        ("Late morning", 0.15),
        ("Around lunch", 0.20),
        ("Afternoon", 0.15),
        ("Early evening", 0.20),
        ("Evening", 0.10),
    ]

    schedule_rows = []

    for period, proportion in schedule:
        amount_ml = estimated_intake_ml * proportion

        schedule_rows.append({
            "Period": period,
            "Approximate amount (mL)": round(amount_ml),
            "250 mL glass equivalents": round(
                amount_ml / GLASS_SIZE_ML, 1
            ),
        })

    schedule_df = pd.DataFrame(schedule_rows)

    st.dataframe(
        schedule_df,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "This is an illustrative distribution, not a mandatory schedule. "
        "Adjust to thirst, activity, access to fluids, and any medical advice. "
        "Avoid forcing large volumes of water quickly."
    )

    # --------------------------------------------------------
    # SAFETY INFORMATION
    # --------------------------------------------------------

    st.divider()
    st.header("⚠️ Hydration Safety")

    st.markdown(
        """
        **Possible signs of dehydration** can include thirst, dry mouth,
        reduced urination, darker urine, tiredness, and dizziness. These
        symptoms can also have other causes.

        Seek urgent medical care for severe confusion, fainting, inability
        to keep fluids down, or other severe or rapidly worsening symptoms.
        """
    )

    for message in get_context_messages(
        result["age"],
        result["climate"],
        result["exercise_minutes"],
    ):
        st.info(message)

    st.divider()
    st.header("🤖 Optional AI Hydration Guidance")

    if st.button("Generate AI Guidance"):

        if (
            not GEMINI_API_KEY
            or GEMINI_API_KEY == "PASTE_YOUR_API_KEY_HERE"
        ):
            st.warning(
                "Add your Gemini API key in GEMINI_API_KEY near the top "
                "of this file to enable AI guidance."
            )
        else:
            try:
                import google.generativeai as genai

                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel(GEMINI_MODEL)

                prompt = f"""
                Provide concise, general educational hydration guidance.

                Age: {result['age']}
                Weight: {result['weight_kg']} kg
                Activity level: {result['activity']}
                Climate: {result['climate']}
                Daily exercise: {result['exercise_minutes']} minutes

                Calculator planning estimate:
                {result['estimated_intake_l']:.2f} litres/day.

                This estimate uses a configurable weight-based starting
                value and illustrative fuzzy rule adjustments.

                Explain:
                1. Why water needs vary from person to person.
                2. Practical ways to spread fluid intake throughout the day.
                3. How heat and exercise can affect fluid losses.
                4. Why people with fluid restrictions should follow clinical
                   advice instead of a generic calculator.

                Do not diagnose dehydration, prescribe an exact medical
                fluid target, or describe this estimate as a clinical need.
                """

                with st.spinner("Generating general guidance..."):
                    response = model.generate_content(prompt)

                ai_text = getattr(response, "text", None)

                if ai_text:
                    st.session_state["water_ai_text"] = ai_text
                    st.markdown(ai_text)
                else:
                    st.warning("The AI service returned no text.")

            except Exception as exc:
                st.error(
                    "AI guidance could not be generated. Check the API key, "
                    "model availability, package version, and connection. "
                    f"Details: {exc}"
                )

    if st.session_state.get("water_ai_text"):
        st.markdown(st.session_state["water_ai_text"])

    # --------------------------------------------------------
    # DOWNLOADABLE REPORT
    # --------------------------------------------------------

    st.divider()
    st.header("📄 Download Water Intake Report")

    report_lines = [
        "DAILY WATER INTAKE PLANNING REPORT",
        "=" * 42,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "INPUTS",
        f"Age: {result['age']}",
        f"Weight: {result['weight_kg']} kg",
        f"Activity: {result['activity']}",
        f"Climate: {result['climate']}",
        f"Exercise: {result['exercise_minutes']} minutes/day",
        "",
        "PLANNING ESTIMATE",
        f"Starting estimate: {result['base_intake_ml']:.0f} mL/day",
        f"Fuzzy/activity adjustment: "
        f"{fuzzy_result['adjustment_l'] * 1000:.0f} mL/day",
        f"Estimated total: {estimated_intake_ml:.0f} mL/day",
        f"Estimated total: {estimated_intake_l:.2f} L/day",
        f"250 mL glass equivalents: {result['glasses']}",
        "",
        "FUZZY INFERENCE",
        f"Rule adjustment: {fuzzy_result['rule_adjustment_l']:.2f} L",
        f"Activity adjustment: "
        f"{fuzzy_result['activity_adjustment_l']:.2f} L",
        "",
        "DECISION TREE",
        f"Predicted adjustment category: {tree_category}",
        "This is a synthetic-label educational classification, "
        "not a dehydration prediction.",
        "",
        "DECISION PATH",
        *result["decision_path"],
        "",
        "LIMITATIONS",
        "This is an educational fluid-planning estimate.",
        "It does not measure hydration status or diagnose dehydration.",
        "Fuzzy rule outputs are illustrative assumptions.",
        "The Decision Tree is trained on synthetic labels generated "
        "from the fuzzy system.",
        "Food also contributes to total fluid intake.",
        "Follow clinician guidance if you have a medical fluid restriction.",
    ]

    ai_text = st.session_state.get("water_ai_text", "")

    if ai_text:
        report_lines.extend([
            "",
            "OPTIONAL AI GUIDANCE",
            ai_text,
        ])

    report_text = "\n".join(report_lines)

    st.download_button(
        label="📥 Download Report (.txt)",
        data=report_text,
        file_name="Water_Intake_Report.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ============================================================
# 11. ALGORITHM SUMMARY
# ============================================================

st.divider()
st.subheader("Algorithm Summary")

algorithm_summary = pd.DataFrame({
    "Component": [
        "Fuzzy rule-based system",
        "Decision Tree Classifier",
        "Fluid planning estimate",
    ],
    "Implementation": [
        "Membership functions, explicit IF–THEN rules, min-based rule firing, weighted output",
        "scikit-learn DecisionTreeClassifier using Gini impurity and a visible decision path",
        "Weight-based starting estimate plus bounded illustrative adjustments",
    ],
    "Limitation": [
        "Rule outputs are illustrative, not clinically validated",
        "Synthetic labels reproduce the fuzzy rule system; no clinical data is used",
        "Not a measurement of hydration status or a personalised medical prescription",
    ],
})

st.dataframe(
    algorithm_summary,
    hide_index=True,
    use_container_width=True,
)

st.info(
    "If you have kidney, heart, or liver disease, have been prescribed a "
    "fluid restriction, or have other medical concerns, use advice from "
    "your healthcare professional rather than this generic estimate."
)
