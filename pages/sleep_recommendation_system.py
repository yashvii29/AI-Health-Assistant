
import random
import math
from datetime import datetime, date, time, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


# ============================================================
# 1. OPTIONAL GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = "AQ.Ab8RN6LDq8jPgJwvOqXjtYwkyiWbDgzO1YhoMA-uwQXCQ-85sQ"
GEMINI_MODEL = "gemini-3.8-flash"

# The main application works without Gemini.


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Sleep Recommendation System",
    page_icon="😴",
    layout="wide",
)

st.title("😴 AI Sleep Recommendation System")

st.write(
    "Explore sleep-duration guidance, sleep-profile clustering, "
    "Decision Tree classification, and personalised schedule planning."
)

st.caption(
    "Educational tool only. It does not diagnose sleep disorders or "
    "measure sleep quality using clinical instruments."
)


# ============================================================
# 3. AGE-BASED SLEEP DURATION GUIDANCE
# ============================================================

def get_sleep_guidance(age):
    """
    Approximate age-based sleep-duration guidance.

    These ranges are general recommendations, not individual diagnoses.
    The adult categories distinguish the usual guidance for ages 18-60,
    61-64, and 65+.
    """
    if age <= 2:
        return {
            "minimum": 11.0,
            "maximum": 14.0,
            "label": "Young children (1–2 years)",
            "note": "This range includes naps.",
        }

    if age <= 5:
        return {
            "minimum": 10.0,
            "maximum": 13.0,
            "label": "Preschool children (3–5 years)",
            "note": "This range includes naps.",
        }

    if age <= 12:
        return {
            "minimum": 9.0,
            "maximum": 12.0,
            "label": "School-age children (6–12 years)",
            "note": "This range refers to total sleep per 24 hours.",
        }

    if age <= 17:
        return {
            "minimum": 8.0,
            "maximum": 10.0,
            "label": "Teenagers (13–17 years)",
            "note": "Individual needs can vary.",
        }

    if age <= 60:
        return {
            "minimum": 7.0,
            "maximum": 9.0,
            "label": "Adults (18–60 years)",
            "note": "At least 7 hours is generally recommended.",
        }

    if age <= 64:
        return {
            "minimum": 7.0,
            "maximum": 9.0,
            "label": "Adults (61–64 years)",
            "note": "Individual needs can vary.",
        }

    return {
        "minimum": 7.0,
        "maximum": 8.0,
        "label": "Adults (65+ years)",
        "note": "Individual needs can vary.",
    }


def get_sleep_category(age, sleep_hours):
    """
    Label a profile relative to the age-specific guidance range.
    This is an educational category, not a sleep disorder diagnosis.
    """
    guidance = get_sleep_guidance(age)

    if sleep_hours < guidance["minimum"]:
        return "Below recommended duration"

    if sleep_hours > guidance["maximum"]:
        return "Above typical duration range"

    return "Within recommended range"


def calculate_sleep_gap(age, sleep_hours):
    """
    Calculate the nightly shortfall relative to the lower boundary.
    A negative gap is not labelled as sleep debt.
    """
    guidance = get_sleep_guidance(age)
    return max(0.0, guidance["minimum"] - sleep_hours)


# ============================================================
# 4. DECISION TREE CLASSIFICATION
# ============================================================
#
# The classifier learns a sleep-duration category from age and sleep
# duration. Training labels are generated from the explicit age-based
# guidance rules, not from clinical patient records.
#
# The algorithm itself is a real supervised DecisionTreeClassifier.
# Its training labels encode the educational rules described above.
# ============================================================

def create_decision_tree_training_data():
    rows = []

    # Sample a grid of ages and sleep durations.
    # Age 1+ is supported by the age-guidance function.
    for age in range(1, 101):
        for sleep_hours in np.arange(0.0, 16.01, 0.25):
            rows.append({
                "Age": float(age),
                "Sleep hours": float(sleep_hours),
                "Category": get_sleep_category(
                    age, float(sleep_hours)
                ),
            })

    return pd.DataFrame(rows)


@st.cache_resource
def train_sleep_decision_tree():
    training_df = create_decision_tree_training_data()

    features = training_df[["Age", "Sleep hours"]]
    labels = training_df["Category"]

    classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
    )

    classifier.fit(features, labels)

    return classifier, training_df


def trace_decision_tree_path(classifier, age, sleep_hours):
    """Return readable branch decisions for one input profile."""
    input_values = np.array([[float(age), float(sleep_hours)]])

    tree = classifier.tree_
    node_indicator = classifier.decision_path(input_values)
    leaf_id = classifier.apply(input_values)[0]

    visited_nodes = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]

    feature_names = ["Age", "Sleep hours"]
    path = []

    for node_id in visited_nodes:

        if node_id == leaf_id:
            counts = tree.value[node_id][0]
            predicted_index = int(np.argmax(counts))
            predicted_label = classifier.classes_[predicted_index]

            path.append(
                f"Leaf node: predicted class = {predicted_label}"
            )
            continue

        feature_index = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_name = feature_names[feature_index]
        feature_value = input_values[0][feature_index]

        if feature_value <= threshold:
            path.append(
                f"{feature_name} ({feature_value:.2f}) <= "
                f"{threshold:.2f}: TRUE; follow left branch."
            )
        else:
            path.append(
                f"{feature_name} ({feature_value:.2f}) > "
                f"{threshold:.2f}: TRUE; follow right branch."
            )

    return path


# ============================================================
# 5. K-MEANS CLUSTERING
# ============================================================
#
# K-Means groups profiles using:
# - Age
# - Average sleep duration
# - Stress level
# - Bedtime represented as sine/cosine
#
# Circular encoding matters because 11:55 PM and 12:05 AM are close
# times, not 23 hours apart.
#
# Because a single user input is not enough to learn meaningful
# clusters, the current profile is compared against a generated
# synthetic reference dataset. These clusters are demonstrative,
# not validated population segments.
# ============================================================

STRESS_ENCODING = {
    "Low": 1.0,
    "Moderate": 2.0,
    "High": 3.0,
}


def bedtime_to_circular_features(bedtime):
    """
    Convert a time of day into cyclic features.

    The angle is based on minutes since midnight.
    """
    minutes = bedtime.hour * 60 + bedtime.minute
    angle = 2.0 * math.pi * minutes / 1440.0

    return math.sin(angle), math.cos(angle)


def make_synthetic_sleep_profiles(n_profiles=240, seed=42):
    """
    Generate reproducible synthetic reference profiles.

    The generator creates varied age, sleep duration, stress, and bedtime
    values for demonstrating clustering. These are not real patient data.
    """
    rng = random.Random(seed)
    rows = []

    for _ in range(n_profiles):
        age = rng.randint(18, 75)

        # Broad synthetic sleep-duration distribution.
        sleep_hours = max(
            3.0,
            min(12.0, rng.gauss(7.2, 1.5)),
        )

        stress = rng.choice(["Low", "Moderate", "High"])

        # Minutes after midnight; allow bedtimes from 7 PM to 3 AM.
        bedtime_minutes = rng.randint(19 * 60, 27 * 60)
        bedtime_minutes %= 1440

        angle = 2.0 * math.pi * bedtime_minutes / 1440.0

        rows.append({
            "Age": float(age),
            "Sleep hours": float(sleep_hours),
            "Stress code": STRESS_ENCODING[stress],
            "Bedtime sin": math.sin(angle),
            "Bedtime cos": math.cos(angle),
            "Stress": stress,
            "Bedtime minutes": bedtime_minutes,
        })

    return pd.DataFrame(rows)


def describe_cluster(cluster_df):
    """
    Produce a descriptive label based on the synthetic cluster's
    average sleep duration and stress level.
    """
    mean_sleep = cluster_df["Sleep hours"].mean()
    mean_stress = cluster_df["Stress code"].mean()

    if mean_sleep < 6.5 and mean_stress >= 2.2:
        return "Shorter sleep / higher stress"

    if mean_sleep < 6.5:
        return "Shorter sleep profile"

    if mean_sleep >= 7.5 and mean_stress < 2.0:
        return "Longer sleep / lower stress"

    if mean_sleep >= 7.5:
        return "Longer sleep profile"

    return "Intermediate sleep profile"


def run_kmeans_clustering(
    age,
    sleep_hours,
    stress,
    bedtime,
    n_clusters=3,
    seed=42,
):
    """
    Fit K-Means to synthetic reference profiles plus the current profile.
    Return the assigned cluster, centroids, and full clustered dataset.
    """
    synthetic_df = make_synthetic_sleep_profiles(
        n_profiles=240,
        seed=seed,
    )

    bedtime_sin, bedtime_cos = bedtime_to_circular_features(bedtime)

    user_profile = {
        "Age": float(age),
        "Sleep hours": float(sleep_hours),
        "Stress code": STRESS_ENCODING[stress],
        "Bedtime sin": bedtime_sin,
        "Bedtime cos": bedtime_cos,
        "Stress": stress,
        "Bedtime minutes": bedtime.hour * 60 + bedtime.minute,
    }

    user_df = pd.DataFrame([user_profile])

    combined_df = pd.concat(
        [synthetic_df, user_df],
        ignore_index=True,
    )

    feature_columns = [
        "Age",
        "Sleep hours",
        "Stress code",
        "Bedtime sin",
        "Bedtime cos",
    ]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        combined_df[feature_columns]
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=10,
        max_iter=300,
    )

    labels = kmeans.fit_predict(scaled_features)
    combined_df["Cluster"] = labels

    # The user's row was appended last.
    user_cluster = int(labels[-1])

    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    centroids_df = pd.DataFrame(
        centroids_original,
        columns=feature_columns,
    )
    centroids_df["Cluster"] = range(n_clusters)

    # Describe clusters using the unscaled original data.
    cluster_descriptions = {}

    for cluster_id in range(n_clusters):
        cluster_rows = combined_df[
            (combined_df["Cluster"] == cluster_id)
            & (combined_df.index < len(synthetic_df))
        ]

        cluster_descriptions[cluster_id] = (
            describe_cluster(cluster_rows)
            if not cluster_rows.empty
            else "Small synthetic cluster"
        )

    return {
        "cluster": user_cluster,
        "cluster_description": cluster_descriptions[user_cluster],
        "cluster_descriptions": cluster_descriptions,
        "clustered_data": combined_df,
        "centroids": centroids_df,
        "feature_columns": feature_columns,
        "scaler": scaler,
        "model": kmeans,
        "synthetic_count": len(synthetic_df),
    }


# ============================================================
# 6. SLEEP SCORE AND DESCRIPTIVE FUZZY MEMBERSHIPS
# ============================================================

def calculate_educational_sleep_score(
    age,
    sleep_hours,
    stress,
):
    """
    Retain a simple, non-clinical indicator for continuity with the
    original app. It is not a validated sleep-quality instrument.
    """
    guidance = get_sleep_guidance(age)
    midpoint = (guidance["minimum"] + guidance["maximum"]) / 2.0

    score = 100.0 - abs(midpoint - sleep_hours) * 10.0

    if stress == "Moderate":
        score -= 10.0
    elif stress == "High":
        score -= 20.0

    return int(max(0, min(100, score)))


def triangular_membership(x, a, b, c):
    if x <= a or x >= c:
        return 0.0

    if x == b:
        return 1.0

    if x < b:
        return (x - a) / (b - a)

    return (c - x) / (c - b)


def shoulder_left(x, full_until, zero_from):
    if x <= full_until:
        return 1.0
    if x >= zero_from:
        return 0.0
    return (zero_from - x) / (zero_from - full_until)


def shoulder_right(x, zero_until, full_from):
    if x <= zero_until:
        return 0.0
    if x >= full_from:
        return 1.0
    return (x - zero_until) / (full_from - zero_until)


def calculate_score_memberships(score):
    """
    Illustrative fuzzy sets over the educational score.
    These memberships are not probabilities or medical classifications.
    """
    return {
        "Poor": shoulder_left(
            score, full_until=30, zero_from=55
        ),
        "Fair": triangular_membership(
            score, a=35, b=60, c=80
        ),
        "Good": triangular_membership(
            score, a=60, b=80, c=100
        ),
        "Excellent": shoulder_right(
            score, zero_until=80, full_from=100
        ),
    }


# ============================================================
# 7. SCHEDULE HELPERS
# ============================================================

def calculate_wakeup_time(bedtime, sleep_hours):
    """
    Add sleep duration to bedtime, correctly crossing midnight.
    Return a time value.
    """
    anchor = datetime.combine(date.today(), bedtime)
    wakeup = anchor + timedelta(hours=float(sleep_hours))
    return wakeup.time()


def calculate_bedtime_for_wakeup(wakeup_time, sleep_hours):
    """
    Subtract desired sleep duration from wake-up time.
    Return a time value.
    """
    anchor = datetime.combine(date.today(), wakeup_time)
    bedtime_dt = anchor - timedelta(hours=float(sleep_hours))
    return bedtime_dt.time()


def format_time(time_value):
    return time_value.strftime("%I:%M %p").lstrip("0")


# ============================================================
# 8. USER INPUTS
# ============================================================

st.subheader("Your sleep profile")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=25,
        step=1,
    )

    sleep_hours = st.number_input(
        "Average sleep per 24 hours (hours)",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.25,
        help=(
            "Enter average total sleep, including naps where applicable."
        ),
    )

with col2:
    bedtime = st.time_input(
        "Usual bedtime",
        value=time(23, 0),
    )

    stress_level = st.selectbox(
        "Self-reported stress level",
        ["Low", "Moderate", "High"],
        index=1,
    )

wake_time_preference = st.time_input(
    "Preferred wake-up time",
    value=time(6, 30),
)

analyse_button = st.button(
    "😴 Analyse Sleep Profile",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 9. RUN ANALYSIS
# ============================================================

if analyse_button:

    guidance = get_sleep_guidance(age)
    sleep_category = get_sleep_category(age, sleep_hours)
    sleep_gap = calculate_sleep_gap(age, sleep_hours)

    score = calculate_educational_sleep_score(
        age,
        sleep_hours,
        stress_level,
    )

    fuzzy_memberships = calculate_score_memberships(score)

    decision_tree, training_df = train_sleep_decision_tree()

    tree_prediction = decision_tree.predict(
        [[float(age), float(sleep_hours)]]
    )[0]

    tree_probabilities = decision_tree.predict_proba(
        [[float(age), float(sleep_hours)]]
    )[0]

    tree_scores = {
        class_name: float(probability)
        for class_name, probability in zip(
            decision_tree.classes_,
            tree_probabilities,
        )
    }

    decision_path = trace_decision_tree_path(
        decision_tree,
        age,
        sleep_hours,
    )

    cluster_result = run_kmeans_clustering(
        age=age,
        sleep_hours=sleep_hours,
        stress=stress_level,
        bedtime=bedtime,
        n_clusters=3,
        seed=42,
    )

    # Schedule based on the midpoint of the recommended range.
    recommended_sleep_hours = (
        guidance["minimum"] + guidance["maximum"]
    ) / 2.0

    recommended_wakeup_from_bedtime = calculate_wakeup_time(
        bedtime,
        recommended_sleep_hours,
    )

    recommended_bedtime_for_wake = calculate_bedtime_for_wakeup(
        wake_time_preference,
        recommended_sleep_hours,
    )

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "age": age,
        "sleep_hours": sleep_hours,
        "bedtime": bedtime,
        "stress_level": stress_level,
        "wake_time_preference": wake_time_preference,
        "guidance": guidance,
        "sleep_category": sleep_category,
        "sleep_gap": sleep_gap,
        "score": score,
        "fuzzy_memberships": fuzzy_memberships,
        "tree_prediction": tree_prediction,
        "tree_scores": tree_scores,
        "decision_path": decision_path,
        "cluster_result": cluster_result,
        "recommended_sleep_hours": recommended_sleep_hours,
        "recommended_wakeup_from_bedtime": (
            recommended_wakeup_from_bedtime
        ),
        "recommended_bedtime_for_wake": (
            recommended_bedtime_for_wake
        ),
    }

    st.session_state["sleep_result"] = payload


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

if "sleep_result" in st.session_state:

    result = st.session_state["sleep_result"]

    guidance = result["guidance"]
    cluster_result = result["cluster_result"]

    st.divider()
    st.header("📊 Sleep Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Sleep duration",
        f"{result['sleep_hours']:.2f} hrs",
    )

    c2.metric(
        "Nightly shortfall",
        f"{result['sleep_gap']:.2f} hrs",
    )

    c3.metric(
        "Age-based range",
        f"{guidance['minimum']:.0f}–{guidance['maximum']:.0f} hrs",
    )

    c4.metric(
        "Educational score",
        f"{result['score']}/100",
    )

    st.caption(
        "The educational score is a simple project indicator, not a "
        "validated sleep-quality score. The nightly shortfall compares "
        "sleep duration with the lower boundary of the age-based range."
    )

    st.subheader("Sleep duration interpretation")

    st.write(f"**Age group:** {guidance['label']}")
    st.write(f"**Duration category:** {result['sleep_category']}")
    st.write(f"**Guidance note:** {guidance['note']}")

    if result["sleep_category"] == "Below recommended duration":
        st.warning(
            "Your reported sleep duration is below the selected age-based "
            "reference range. If this is persistent, consider reviewing "
            "your schedule and discussing ongoing sleep difficulties "
            "with a healthcare professional."
        )
    elif result["sleep_category"] == "Within recommended range":
        st.success(
            "Your reported sleep duration falls within the selected "
            "age-based reference range. Duration alone does not establish "
            "good sleep quality."
        )
    else:
        st.info(
            "Your reported sleep duration is above the selected reference "
            "range. Longer sleep can be normal in some circumstances; "
            "consider context, sleep quality, and daytime symptoms."
        )

    # --------------------------------------------------------
    # DECISION TREE
    # --------------------------------------------------------

    st.divider()
    st.header("🌳 Decision Tree Classification")

    st.write(
        "The classifier uses age and average sleep duration to predict "
        "the same educational categories defined by the age-based rules."
    )

    tree_col1, tree_col2 = st.columns(2)

    with tree_col1:
        st.metric(
            "Decision Tree prediction",
            result["tree_prediction"],
        )

        st.subheader("Decision path")

        for step_number, step in enumerate(
            result["decision_path"], start=1
        ):
            st.write(f"{step_number}. {step}")

    with tree_col2:
        st.subheader("Classifier scores")

        tree_score_df = pd.DataFrame({
            "Category": list(result["tree_scores"].keys()),
            "Model score": list(result["tree_scores"].values()),
        })

        st.bar_chart(
            tree_score_df.set_index("Category"),
        )

        st.caption(
            "These are classifier scores for rule-generated training labels, "
            "not probabilities of insomnia or another sleep disorder."
        )

    with st.expander("View the learned Decision Tree rules"):
        decision_tree, training_df = train_sleep_decision_tree()

        st.code(
            export_text(
                decision_tree,
                feature_names=["Age", "Sleep hours"],
                decimals=2,
            ),
            language="text",
        )

    with st.expander("View the Decision Tree diagram"):
        decision_tree, training_df = train_sleep_decision_tree()

        fig, ax = plt.subplots(figsize=(18, 8))

        plot_tree(
            decision_tree,
            feature_names=["Age", "Sleep hours"],
            class_names=decision_tree.classes_,
            filled=True,
            rounded=True,
            ax=ax,
        )

        ax.set_title(
            "Decision Tree: Educational Sleep-Duration Categories"
        )

        st.pyplot(fig)
        plt.close(fig)

    with st.expander("How Decision Tree Classification works"):
        st.markdown(
            """
            1. **Training data:** a grid of age and sleep-duration values is
               generated and labelled using the explicit sleep guidance rules.
            2. **Gini impurity:** the model evaluates splits that separate
               examples into more homogeneous classes.
            3. **Tree learning:** the classifier learns thresholds for age
               and sleep duration.
            4. **Prediction:** the current profile is passed through the tree.
            5. **Decision path:** each split used for the prediction is shown.

            **Limitation:** the labels come from the same rules that define
            the reference categories. This is a demonstration of supervised
            classification, not a clinically validated prediction model.
            """
        )

    # --------------------------------------------------------
    # K-MEANS CLUSTERING
    # --------------------------------------------------------

    st.divider()
    st.header("🔬 K-Means Sleep-Profile Clustering")

    st.write(
        "K-Means groups the current profile with synthetic reference "
        "profiles based on age, sleep duration, stress, and circular "
        "bedtime features."
    )

    cluster_col1, cluster_col2, cluster_col3 = st.columns(3)

    cluster_col1.metric(
        "Assigned cluster",
        f"Cluster {cluster_result['cluster'] + 1}",
    )

    cluster_col2.metric(
        "Cluster description",
        cluster_result["cluster_description"],
    )

    cluster_col3.metric(
        "Reference profiles",
        cluster_result["synthetic_count"],
    )

    st.caption(
        "The reference profiles are synthetic, not real users. Cluster "
        "numbers have no intrinsic ranking, and the descriptive labels "
        "are assigned from cluster averages."
    )

    clustered_df = cluster_result["clustered_data"]
    synthetic_count = cluster_result["synthetic_count"]

    synthetic_clustered = clustered_df.iloc[:synthetic_count].copy()

    st.subheader("Cluster summaries")

    cluster_summaries = []

    for cluster_id in sorted(
        synthetic_clustered["Cluster"].unique()
    ):
        cluster_df = synthetic_clustered[
            synthetic_clustered["Cluster"] == cluster_id
        ]

        cluster_summaries.append({
            "Cluster": f"Cluster {cluster_id + 1}",
            "Descriptive pattern": (
                cluster_result["cluster_descriptions"][cluster_id]
            ),
            "Synthetic profiles": len(cluster_df),
            "Mean age": cluster_df["Age"].mean(),
            "Mean sleep hours": cluster_df["Sleep hours"].mean(),
            "Mean stress code": cluster_df["Stress code"].mean(),
        })

    summary_df = pd.DataFrame(cluster_summaries)

    st.dataframe(
        summary_df.round(2),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Cluster visualisation")

    fig, ax = plt.subplots(figsize=(10, 6))

    for cluster_id in sorted(
        synthetic_clustered["Cluster"].unique()
    ):
        cluster_df = synthetic_clustered[
            synthetic_clustered["Cluster"] == cluster_id
        ]

        ax.scatter(
            cluster_df["Sleep hours"],
            cluster_df["Stress code"],
            label=f"Cluster {cluster_id + 1}",
            alpha=0.65,
        )

    ax.scatter(
        result["sleep_hours"],
        STRESS_ENCODING[result["stress_level"]],
        marker="*",
        s=250,
        edgecolors="black",
        label="Your profile",
    )

    ax.set_xlabel("Sleep duration (hours)")
    ax.set_ylabel("Stress code (1 = low, 3 = high)")
    ax.set_title("K-Means Clusters: Sleep Duration vs Stress")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Cluster centroids")

    centroids_df = cluster_result["centroids"].copy()
    centroids_df["Cluster"] = centroids_df["Cluster"].apply(
        lambda x: f"Cluster {int(x) + 1}"
    )

    st.dataframe(
        centroids_df.round(3),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("How K-Means works"):
        st.markdown(
            """
            1. **Feature preparation:** age, sleep duration, stress, and
               bedtime sine/cosine values are assembled.
            2. **Standardisation:** `StandardScaler` puts features on
               comparable scales.
            3. **Initialisation:** K-Means initialises three cluster centres.
            4. **Assignment:** each profile is assigned to its nearest centre.
            5. **Update:** cluster centres are recalculated as the mean of
               their assigned profiles.
            6. **Convergence:** assignment and update steps repeat until
               the algorithm converges or reaches its iteration limit.

            K-Means minimises within-cluster squared distances. It does not
            determine whether a profile is medically healthy.
            """
        )

    # --------------------------------------------------------
    # FUZZY SCORE MEMBERSHIPS
    # --------------------------------------------------------

    st.divider()
    st.header("🧠 Fuzzy Sleep-Score Interpretation")

    st.write(
        "These fuzzy memberships preserve the original module's educational "
        "score display. They are overlapping mathematical membership values, "
        "not probabilities or validated clinical categories."
    )

    membership_df = pd.DataFrame({
        "Category": list(result["fuzzy_memberships"].keys()),
        "Membership": list(result["fuzzy_memberships"].values()),
    })

    st.bar_chart(
        membership_df.set_index("Category"),
    )

    for category, membership in result["fuzzy_memberships"].items():
        st.write(f"**{category}:** {membership:.2f}")
        st.progress(float(membership))

    # --------------------------------------------------------
    # SLEEP SCHEDULE
    # --------------------------------------------------------

    st.divider()
    st.header("⏰ Recommended Sleep Schedule")

    schedule_col1, schedule_col2 = st.columns(2)

    with schedule_col1:
        st.info(
            "**Wake-up time based on your current bedtime:** "
            f"{format_time(result['recommended_wakeup_from_bedtime'])}"
        )

    with schedule_col2:
        st.info(
            "**Bedtime for your preferred wake-up time:** "
            f"{format_time(result['recommended_bedtime_for_wake'])}"
        )

    st.caption(
        f"Schedule calculations use the midpoint of your age-based range "
        f"({result['recommended_sleep_hours']:.1f} hours). This is a planning "
        "example, not a requirement to sleep exactly that duration."
    )

    # --------------------------------------------------------
    # GENERAL SLEEP HABITS
    # --------------------------------------------------------

    st.divider()
    st.header("🌙 Practical Sleep Habits")

    st.markdown(
        """
        - Keep a reasonably consistent bedtime and wake-up time.
        - Create a quiet, dark, comfortable sleep environment.
        - Reduce bright light and stimulating activities before bed.
        - Avoid caffeine close to bedtime if it affects your sleep.
        - Use a wind-down routine to manage stress.
        - If sleep difficulties persist or significantly affect daytime
          functioning, consider speaking with a healthcare professional.
        """
    )

    # --------------------------------------------------------
    # OPTIONAL GEMINI GUIDANCE
    # --------------------------------------------------------

    st.divider()
    st.header("🤖 Optional AI Sleep Guidance")

    if st.button("Generate AI Sleep Recommendations"):

        if (
            not GEMINI_API_KEY
            or GEMINI_API_KEY == "PASTE_YOUR_API_KEY_HERE"
        ):
            st.warning(
                "Add your Gemini API key in GEMINI_API_KEY near the top "
                "of this file to enable AI recommendations."
            )

        else:
            try:
                import google.generativeai as genai

                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel(GEMINI_MODEL)

                prompt = f"""
                Provide concise, general educational sleep guidance.

                Age: {result['age']}
                Sleep duration: {result['sleep_hours']} hours per 24 hours
                Stress: {result['stress_level']}
                Bedtime: {format_time(result['bedtime'])}
                Age-based sleep range: {guidance['minimum']}–{guidance['maximum']} hours
                Duration category: {result['sleep_category']}
                K-Means cluster description: {cluster_result['cluster_description']}

                Include:
                1. Practical sleep routine suggestions.
                2. General stress-management ideas.
                3. Ways to improve the sleep environment.
                4. When persistent sleep problems merit professional advice.

                Do not diagnose a sleep disorder or treat the educational
                score or synthetic cluster as clinical evidence.
                """

                with st.spinner("Generating sleep guidance..."):
                    response = model.generate_content(prompt)

                ai_text = getattr(response, "text", None)

                if ai_text:
                    st.session_state["sleep_ai_text"] = ai_text
                    st.markdown(ai_text)
                else:
                    st.warning("The AI service returned no text.")

            except Exception as exc:
                st.error(
                    "AI guidance could not be generated. Check the API key, "
                    "model availability, package version, and network "
                    f"connection. Details: {exc}"
                )

    if st.session_state.get("sleep_ai_text"):
        st.markdown(st.session_state["sleep_ai_text"])

    # --------------------------------------------------------
    # DOWNLOADABLE REPORT
    # --------------------------------------------------------

    st.divider()
    st.header("📄 Download Sleep Report")

    report_lines = [
        "AI SLEEP RECOMMENDATION REPORT",
        "=" * 42,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "USER INPUTS",
        f"Age: {result['age']}",
        f"Average sleep duration: {result['sleep_hours']:.2f} hours",
        f"Bedtime: {format_time(result['bedtime'])}",
        f"Preferred wake-up time: "
        f"{format_time(result['wake_time_preference'])}",
        f"Stress: {result['stress_level']}",
        "",
        "SLEEP DURATION GUIDANCE",
        f"Age group: {guidance['label']}",
        f"Reference range: {guidance['minimum']:.0f}–"
        f"{guidance['maximum']:.0f} hours",
        f"Duration category: {result['sleep_category']}",
        f"Nightly shortfall relative to lower bound: "
        f"{result['sleep_gap']:.2f} hours",
        f"Educational score (not clinically validated): {result['score']}/100",
        "",
        "DECISION TREE",
        f"Predicted category: {result['tree_prediction']}",
        "Decision path:",
        *result["decision_path"],
        "",
        "K-MEANS CLUSTERING",
        f"Assigned cluster: {cluster_result['cluster'] + 1}",
        f"Descriptive pattern: {cluster_result['cluster_description']}",
        "Clusters use synthetic reference profiles and are not clinical labels.",
        "",
        "SCHEDULE",
        f"Wake-up based on current bedtime: "
        f"{format_time(result['recommended_wakeup_from_bedtime'])}",
        f"Bedtime for preferred wake-up: "
        f"{format_time(result['recommended_bedtime_for_wake'])}",
        "",
        "LIMITATIONS",
        "The Decision Tree is trained on synthetic examples labelled using "
        "the age-based duration rules.",
        "K-Means compares the current profile with synthetic reference profiles.",
        "The score and fuzzy memberships are educational indicators.",
        "This report does not diagnose a sleep disorder.",
    ]

    ai_text = st.session_state.get("sleep_ai_text", "")

    if ai_text:
        report_lines.extend([
            "",
            "OPTIONAL AI GUIDANCE",
            ai_text,
        ])

    report_text = "\n".join(report_lines)

    st.download_button(
        label="📥 Download Sleep Report (.txt)",
        data=report_text,
        file_name="Sleep_Report.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ============================================================
# 11. ALGORITHM SUMMARY
# ============================================================

st.divider()
st.subheader("Algorithm Summary")

algorithm_df = pd.DataFrame({
    "Algorithm": [
        "K-Means Clustering",
        "Decision Tree Classification",
    ],
    "Implementation": [
        "StandardScaler, KMeans, three clusters, centroid display, profile assignment",
        "DecisionTreeClassifier, Gini impurity, prediction, decision path, tree visualisation",
    ],
    "Data used": [
        "Synthetic sleep profiles plus the current profile",
        "Synthetic age/sleep examples labelled using the sleep-duration rules",
    ],
    "Limitation": [
        "Clusters are illustrative and do not represent validated population groups",
        "Predictions reproduce educational rules; no clinical outcomes are predicted",
    ],
})

st.dataframe(
    algorithm_df,
    hide_index=True,
    use_container_width=True,
)

st.info(
    "The algorithms demonstrate clustering and supervised classification. "
    "Neither is a validated tool for diagnosing sleep disorders or predicting "
    "medical outcomes."
)
