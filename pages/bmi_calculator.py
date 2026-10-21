
import io
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text


# ============================================================
# 1. OPTIONAL GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = "AQ.Ab8RN6LDq8jPgJwvOqXjtYwkyiWbDgzO1YhoMA-uwQXCQ-85sQ"
GEMINI_MODEL = "gemini-3.8-flash"

# AI is optional. The BMI calculator works without an API key.


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI BMI Calculator",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ AI BMI Calculator")
st.write(
    "Calculate BMI, explore a Decision Tree Classifier, "
    "inspect fuzzy membership scores, and generate a health report."
)

st.caption(
    "For adults aged 18 and over. BMI is a screening measure, "
    "not a diagnosis or a complete measure of health."
)


# ============================================================
# 3. STANDARD ADULT BMI CLASSIFICATION
# ============================================================

BMI_CLASSES = [
    "Underweight",
    "Normal weight",
    "Overweight",
    "Obesity",
]


def classify_adult_bmi(bmi):
    """Standard adult BMI categories; obesity is one broad category."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal weight"
    elif bmi < 30.0:
        return "Overweight"
    return "Obesity"


def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI from kilograms and centimetres."""
    height_m = height_cm / 100.0
    return weight_kg / (height_m ** 2)


def calculate_reference_weight_range(height_cm):
    """Weight corresponding to adult BMI values 18.5 and 24.9."""
    height_m = height_cm / 100.0
    return (
        18.5 * height_m**2,
        24.9 * height_m**2,
    )


# ============================================================
# 4. DECISION TREE CLASSIFICATION
# ============================================================
#
# Important:
# This is an educational, rule-derived classifier. Its training
# examples encode standard adult BMI boundaries; they are not
# patient records or a clinically validated training dataset.
#
# Using one feature (BMI) makes the learned decision path easy
# to inspect. It does not predict health outcomes or diseases.
# ============================================================

@st.cache_resource
def train_bmi_decision_tree():
    """
    Train a small DecisionTreeClassifier on synthetic boundary
    examples so the learned splits represent standard BMI bands.
    """

    training_bmi = np.array([
        [10.0],
        [16.0],
        [18.49],   # Below 18.5
        [18.51],   # Above 18.5
        [21.0],
        [24.99],   # Below 25
        [25.01],   # Above 25
        [27.5],
        [29.99],   # Below 30
        [30.01],   # Above 30
        [35.0],
        [40.0],
        [50.0],
    ])

    training_labels = np.array([
        "Underweight",
        "Underweight",
        "Underweight",
        "Normal weight",
        "Normal weight",
        "Normal weight",
        "Overweight",
        "Overweight",
        "Overweight",
        "Obesity",
        "Obesity",
        "Obesity",
        "Obesity",
    ])

    classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        random_state=42,
    )

    classifier.fit(training_bmi, training_labels)
    return classifier


def get_decision_path(classifier, bmi):
    """Return the rules traversed for an individual BMI prediction."""
    tree = classifier.tree_
    node_indicator = classifier.decision_path([[bmi]])
    leaf_id = classifier.apply([[bmi]])[0]

    visited_nodes = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]

    path = []

    for node_id in visited_nodes:
        if node_id == leaf_id:
            predicted_class = classifier.classes_[
                np.argmax(tree.value[node_id][0])
            ]
            path.append(
                f"Leaf node: predicted class = {predicted_class}"
            )
            continue

        feature_index = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        if bmi <= threshold:
            path.append(
                f"BMI ({bmi:.2f}) <= {threshold:.2f}: TRUE; "
                "follow left branch."
            )
        else:
            path.append(
                f"BMI ({bmi:.2f}) > {threshold:.2f}: TRUE; "
                "follow right branch."
            )

    return path


# ============================================================
# 5. FUZZY LOGIC MEMBERSHIP FUNCTIONS
# ============================================================

def triangular_membership(x, a, b, c):
    """Membership function for a triangular fuzzy set."""
    if x <= a or x >= c:
        return 0.0

    if x == b:
        return 1.0

    if x < b:
        return (x - a) / (b - a) if b != a else 1.0

    return (c - x) / (c - b) if c != b else 1.0


def left_shoulder_membership(x, full_until, zero_from):
    """Membership is 1 on the left, then declines to 0."""
    if x <= full_until:
        return 1.0
    if x >= zero_from:
        return 0.0

    return (zero_from - x) / (zero_from - full_until)


def right_shoulder_membership(x, zero_until, full_from):
    """Membership is 0 on the left, then increases to 1."""
    if x <= zero_until:
        return 0.0
    if x >= full_from:
        return 1.0

    return (x - zero_until) / (full_from - zero_until)


def calculate_fuzzy_memberships(bmi):
    """
    Calculate overlapping fuzzy memberships for interpretation.
    These are illustrative memberships, not medical probabilities.
    """

    underweight = left_shoulder_membership(
        bmi, full_until=17.0, zero_from=19.0
    )

    normal = triangular_membership(
        bmi, a=17.5, b=21.75, c=26.0
    )

    overweight = triangular_membership(
        bmi, a=23.0, b=27.5, c=32.0
    )

    obesity = right_shoulder_membership(
        bmi, zero_until=28.0, full_from=35.0
    )

    return {
        "Underweight": float(np.clip(underweight, 0, 1)),
        "Normal weight": float(np.clip(normal, 0, 1)),
        "Overweight": float(np.clip(overweight, 0, 1)),
        "Obesity": float(np.clip(obesity, 0, 1)),
    }


# ============================================================
# 6. NON-CLINICAL BMI REFERENCE INDICATOR
# ============================================================

def calculate_bmi_reference_indicator(bmi):
    """
    Retains the original project's BMI-proximity formula.
    This is not a validated health score.
    """
    score = 100 - abs(22 - bmi) * 5
    return int(np.clip(score, 0, 100))


# ============================================================
# 7. GENERAL INTERPRETATION
# ============================================================

def get_general_advice(category):
    advice = {
        "Underweight": (
            "BMI is below the standard adult reference range. "
            "If this is unexpected or you have lost weight without "
            "trying, discuss it with a healthcare professional."
        ),
        "Normal weight": (
            "BMI falls within the standard adult reference range. "
            "BMI alone cannot confirm nutritional status or overall health."
        ),
        "Overweight": (
            "BMI is above the standard adult reference range. "
            "Consider your waist measurement, activity, medical history, "
            "and other health indicators before drawing conclusions."
        ),
        "Obesity": (
            "BMI is in the adult obesity range. Consider discussing "
            "your overall health and individual risk factors with "
            "a qualified healthcare professional."
        ),
    }
    return advice[category]


# ============================================================
# 8. USER INPUTS
# ============================================================

st.subheader("Enter your measurements")

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    age = st.number_input(
        "Age (years)",
        min_value=2,
        max_value=120,
        value=25,
        step=1,
    )

with input_col2:
    height_cm = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=170.0,
        step=0.5,
    )

with input_col3:
    weight_kg = st.number_input(
        "Weight (kg)",
        min_value=10.0,
        max_value=300.0,
        value=70.0,
        step=0.5,
    )

calculate_button = st.button(
    "Calculate BMI and Run Decision Tree",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 9. CALCULATION AND RESULTS
# ============================================================

if calculate_button:

    if height_cm <= 0 or weight_kg <= 0:
        st.error("Height and weight must be greater than zero.")
        st.stop()

    bmi = calculate_bmi(weight_kg, height_cm)

    st.session_state["bmi_result"] = {
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


if "bmi_result" in st.session_state:

    result = st.session_state["bmi_result"]

    age = result["age"]
    height_cm = result["height_cm"]
    weight_kg = result["weight_kg"]
    bmi = result["bmi"]

    st.divider()
    st.header("📊 BMI Dashboard")

    bmi_col, category_col, indicator_col = st.columns(3)

    with bmi_col:
        st.metric("BMI", f"{bmi:.2f}")

    if age >= 18:

        official_category = classify_adult_bmi(bmi)
        reference_indicator = calculate_bmi_reference_indicator(bmi)

        with category_col:
            st.metric("Adult BMI category", official_category)

        with indicator_col:
            st.metric(
                "BMI reference indicator",
                f"{reference_indicator}/100",
            )

        st.caption(
            "The reference indicator measures proximity to BMI 22 "
            "using the original project's formula. It is not a "
            "medical or overall health score."
        )

        ideal_min, ideal_max = calculate_reference_weight_range(height_cm)

        st.info(
            f"**Adult reference weight range for this height:** "
            f"{ideal_min:.1f}–{ideal_max:.1f} kg. "
            "This range is based on BMI 18.5–24.9 and is not an "
            "individual weight-loss target."
        )

        # ----------------------------------------------------
        # 10. DECISION TREE MODEL PREDICTION
        # ----------------------------------------------------

        st.divider()
        st.header("🌳 Decision Tree Classification")

        classifier = train_bmi_decision_tree()
        tree_prediction = classifier.predict([[bmi]])[0]
        decision_path = get_decision_path(classifier, bmi)

        tree_col1, tree_col2 = st.columns(2)

        with tree_col1:
            st.subheader("Model prediction")
            st.metric("Decision Tree output", tree_prediction)

            st.write("**Decision path for your BMI**")
            for step_number, step in enumerate(decision_path, start=1):
                st.write(f"{step_number}. {step}")

        with tree_col2:
            st.subheader("Classification rules")
            st.code(
                export_text(
                    classifier,
                    feature_names=["BMI"],
                    decimals=2,
                ),
                language="text",
            )

        if tree_prediction != official_category:
            st.warning(
                "The educational tree's prediction differs from the "
                "standard adult BMI category at this value. The standard "
                "threshold-based category is shown as the reference result; "
                "the tree is trained only on synthetic boundary examples."
            )
        else:
            st.success(
                "The tree prediction matches the standard adult BMI category."
            )

        with st.expander("View the trained Decision Tree diagram"):
            fig, ax = plt.subplots(figsize=(14, 6))

            plot_tree(
                classifier,
                feature_names=["BMI"],
                class_names=classifier.classes_,
                filled=True,
                rounded=True,
                impurity=True,
                proportion=False,
                ax=ax,
            )

            ax.set_title(
                "Educational Decision Tree for Adult BMI Categories"
            )
            st.pyplot(fig)
            plt.close(fig)

        with st.expander("How the algorithm works"):
            st.markdown(
                """
                1. **Training data:** synthetic BMI examples are labelled
                   using standard adult BMI categories.
                2. **Gini impurity:** the classifier evaluates splits that
                   separate the labels into more homogeneous groups.
                3. **Tree construction:** the model learns threshold-based
                   branches from the labelled examples.
                4. **Prediction:** your BMI is passed to the trained model.
                5. **Decision path:** the app displays the branches used
                   to reach the predicted class.

                **Limitation:** this is a demonstration of supervised
                classification, not a model trained or validated on
                clinical patient data. It classifies BMI bands only.
                """
            )

        # ----------------------------------------------------
        # 11. FUZZY LOGIC ANALYSIS
        # ----------------------------------------------------

        st.divider()
        st.header("🧠 Fuzzy Logic Analysis")

        st.write(
            "Fuzzy membership scores express how strongly a BMI value "
            "belongs to overlapping illustrative categories. They are "
            "not probabilities and do not replace the standard BMI bands."
        )

        fuzzy_scores = calculate_fuzzy_memberships(bmi)

        fuzzy_df = pd.DataFrame({
            "Category": list(fuzzy_scores.keys()),
            "Membership": list(fuzzy_scores.values()),
        })

        st.bar_chart(
            fuzzy_df.set_index("Category"),
            y="Membership",
        )

        for label, membership in fuzzy_scores.items():
            st.write(f"**{label}:** {membership:.2f}")
            st.progress(membership)

        # ----------------------------------------------------
        # 12. INTERPRETATION AND LIMITATIONS
        # ----------------------------------------------------

        st.divider()
        st.header("🩺 BMI Interpretation")

        st.info(get_general_advice(official_category))

        st.markdown(
            """
            **BMI limitations**

            - BMI does not distinguish muscle from fat.
            - It does not measure fat distribution or directly assess
              blood pressure, blood glucose, cholesterol, or fitness.
            - Interpretation may require additional context, particularly
              for athletes, older adults, and people with certain medical
              conditions.
            - For children and teenagers, BMI must be interpreted using
              age- and sex-specific growth charts rather than adult cutoffs.
            """
        )

        # ----------------------------------------------------
        # 13. OPTIONAL GEMINI HEALTH INSIGHTS
        # ----------------------------------------------------

        st.divider()
        st.header("🤖 Optional AI Insights")

        st.caption(
            "AI-generated information is educational and may be inaccurate. "
            "It is not a diagnosis or a substitute for professional care."
        )

        if st.button("Generate AI Insights"):

            if (
                not GEMINI_API_KEY
                or GEMINI_API_KEY == "PASTE_YOUR_API_KEY_HERE"
            ):
                st.warning(
                    "Add your Gemini API key in the GEMINI_API_KEY "
                    "variable near the top of this file to enable AI insights."
                )

            else:
                try:
                    import google.generativeai as genai

                    genai.configure(api_key=GEMINI_API_KEY)
                    model = genai.GenerativeModel(GEMINI_MODEL)

                    prompt = f"""
                    Provide general educational information about these
                    adult BMI measurements. Do not diagnose disease,
                    prescribe treatment, or claim that BMI proves someone
                    is healthy or unhealthy.

                    Age: {age}
                    Height: {height_cm} cm
                    Weight: {weight_kg} kg
                    BMI: {bmi:.2f}
                    Standard adult BMI category: {official_category}

                    Include:
                    1. A short explanation of what BMI can and cannot tell us.
                    2. General balanced-nutrition suggestions.
                    3. General physical-activity and lifestyle suggestions.
                    4. When discussing concerns, recommend appropriate
                       advice from a qualified healthcare professional.

                    Keep the answer concise, practical, and non-judgmental.
                    """

                    with st.spinner("Generating educational insights..."):
                        response = model.generate_content(prompt)

                    ai_text = getattr(response, "text", None)

                    if ai_text:
                        st.session_state["bmi_ai_text"] = ai_text
                        st.markdown(ai_text)
                    else:
                        st.warning(
                            "The AI service returned no text. Check the "
                            "configured model and API response."
                        )

                except Exception as exc:
                    st.error(
                        "AI insights could not be generated. Check the API "
                        "key, model availability, package version, and "
                        f"network connection. Technical details: {exc}"
                    )

        if st.session_state.get("bmi_ai_text"):
            st.markdown(st.session_state["bmi_ai_text"])

        # ----------------------------------------------------
        # 14. DOWNLOADABLE REPORT
        # ----------------------------------------------------

        st.divider()
        st.header("📄 BMI Report")

        report_lines = [
            "AI BMI CALCULATOR REPORT",
            "=" * 40,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Age: {age} years",
            f"Height: {height_cm:.1f} cm",
            f"Weight: {weight_kg:.1f} kg",
            f"BMI: {bmi:.2f}",
        ]

        if age >= 18:
            report_lines.extend([
                f"Standard adult BMI category: {official_category}",
                f"Decision Tree prediction: {tree_prediction}",
                f"BMI reference indicator (not a clinical score): "
                f"{reference_indicator}/100",
                f"Adult reference weight range: "
                f"{ideal_min:.1f}–{ideal_max:.1f} kg",
                "",
                "DECISION TREE PATH",
                *decision_path,
                "",
                "FUZZY MEMBERSHIP SCORES (NOT PROBABILITIES)",
                *[
                    f"{name}: {score:.2f}"
                    for name, score in fuzzy_scores.items()
                ],
                "",
                "GENERAL INTERPRETATION",
                get_general_advice(official_category),
            ])

        report_lines.extend([
            "",
            "MODEL LIMITATIONS",
            "The decision tree uses synthetic boundary examples and has "
            "not been clinically validated.",
            "BMI is a screening measure and does not independently "
            "diagnose health conditions.",
        ])

        ai_report = st.session_state.get("bmi_ai_text", "")
        if ai_report:
            report_lines.extend(["", "OPTIONAL AI INSIGHTS", ai_report])

        report_text = "\n".join(report_lines)

        st.download_button(
            label="📥 Download BMI Report (.txt)",
            data=report_text,
            file_name="BMI_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    else:
        st.warning(
            "The BMI value has been calculated, but adult BMI categories "
            "and the adult Decision Tree are not applied to people under "
            "18. A clinician can interpret BMI using age- and sex-specific "
            "growth charts."
        )


# ============================================================
# 15. BMI REFERENCE TABLE
# ============================================================

st.divider()
st.subheader("Adult BMI Reference Categories")

reference_table = pd.DataFrame({
    "BMI range (kg/m²)": [
        "Below 18.5",
        "18.5 to below 25.0",
        "25.0 to below 30.0",
        "30.0 and above",
    ],
    "Category": BMI_CLASSES,
})

st.dataframe(
    reference_table,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Reference: commonly used adult BMI classification thresholds. "
    "This educational tool is not intended for emergency assessment, "
    "diagnosis, or individual treatment decisions."
)
