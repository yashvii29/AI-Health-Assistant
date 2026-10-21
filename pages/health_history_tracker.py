
import sqlite3
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Health History Tracker",
    page_icon="📈",
    layout="wide"
)

# Optional Gemini integration. The module works without an API key.
GEMINI_API_KEY = "AQ.Ab8RN6LDq8jPgJwvOqXjtYwkyiWbDgzO1YhoMA-uwQXCQ-85sQ"
GEMINI_MODEL = "gemini-3.8-flash"

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "health_history.db"

st.title("Health History Tracker")
st.caption(
    "Maintain health records, explore historical trends, "
    "and examine patterns using graph traversal and clustering."
)

# ============================================================
# DATABASE: PERSISTENT SQLITE STORAGE
# ============================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS health_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_date TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_record_date
            ON health_records(record_date)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metric
            ON health_records(metric)
        """)



def load_records():
    columns = [
        "id", "record_date", "metric", "value",
        "unit", "notes", "created_at"
    ]

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, record_date, metric, value, unit, notes, created_at
            FROM health_records
            ORDER BY record_date ASC, id ASC
        """).fetchall()

    # Preserve the expected columns even when the database is empty.
    df = pd.DataFrame([dict(row) for row in rows], columns=columns)

    if not df.empty:
        df["record_date"] = pd.to_datetime(
            df["record_date"], errors="coerce"
        )
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["record_date", "value"])
        df["record_date"] = df["record_date"].dt.date

    return df



def add_record(record_date, metric, value, unit, notes):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO health_records
            (record_date, metric, value, unit, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            record_date.isoformat(),
            metric.strip(),
            float(value),
            unit.strip(),
            notes.strip(),
            datetime.now().isoformat(timespec="seconds")
        ))


def delete_record(record_id):
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM health_records WHERE id = ?",
            (int(record_id),)
        )


initialize_database()

# ============================================================
# HEALTH METRIC CATALOGUE
# Users can also enter custom metrics and units.
# ============================================================

METRIC_CATALOGUE = {
    "Weight": "kg",
    "Height": "cm",
    "BMI": "kg/m²",
    "Blood Pressure - Systolic": "mmHg",
    "Blood Pressure - Diastolic": "mmHg",
    "Heart Rate": "bpm",
    "Blood Glucose": "mg/dL",
    "Body Temperature": "°C",
    "Oxygen Saturation (SpO2)": "%",
    "Daily Water Intake": "L",
    "Sleep Duration": "hours",
    "Steps": "steps",
    "Calories Consumed": "kcal",
    "Custom Metric": ""
}

# ============================================================
# GRAPH CONSTRUCTION
#
# Each record is a node.
# Two records are connected when:
# 1. They measure the same metric.
# 2. Their dates are within the chosen maximum gap.
#
# Edges are undirected. BFS and DFS traverse this graph.
# This is a record-navigation graph, not a medical causal graph.
# ============================================================

def build_record_graph(df, max_gap_days=90):
    graph = {int(record_id): [] for record_id in df["id"].tolist()}

    if df.empty:
        return graph

    working = df.copy()
    working["record_date"] = pd.to_datetime(working["record_date"])

    for _, group in working.groupby("metric"):
        group = group.sort_values(["record_date", "id"])
        rows = list(group[["id", "record_date"]].itertuples(
            index=False, name=None
        ))

        # Only compare later records. This avoids duplicate edges.
        # Connecting all qualifying pairs makes the graph definition
        # independent of which record happens to be the starting node.
        for i in range(len(rows)):
            id_a, date_a = int(rows[i][0]), rows[i][1]

            for j in range(i + 1, len(rows)):
                id_b, date_b = int(rows[j][0]), rows[j][1]
                gap = (date_b - date_a).days

                if gap > max_gap_days:
                    # Rows are chronologically sorted, so later gaps
                    # cannot become smaller.
                    break

                graph[id_a].append(id_b)
                graph[id_b].append(id_a)

    for node in graph:
        graph[node] = sorted(set(graph[node]))

    return graph


def bfs_traversal(graph, start_node):
    """Breadth-First Search: visit nodes level by level."""
    if start_node not in graph:
        return []

    visited = {start_node}
    queue = deque([start_node])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


def dfs_traversal(graph, start_node):
    """Depth-First Search: explore a branch before backtracking."""
    if start_node not in graph:
        return []

    visited = set()
    stack = [start_node]
    order = []

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        # Reverse push order so the smallest ID is visited first.
        for neighbor in sorted(graph[node], reverse=True):
            if neighbor not in visited:
                stack.append(neighbor)

    return order


# ============================================================
# K-MEANS CLUSTERING
#
# Cluster records for ONE metric at a time.
# StandardScaler prevents scale differences from dominating.
# Clusters are statistical groupings, not medical risk classes.
# ============================================================

def run_kmeans(metric_df, requested_k, random_state=42):
    data = metric_df.copy()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["value"])

    if len(data) < 2:
        raise ValueError("At least two valid records are required.")

    unique_values = data["value"].nunique()
    effective_k = min(int(requested_k), len(data), unique_values)

    if effective_k < 2:
        raise ValueError(
            "At least two distinct values are required for clustering."
        )

    X = data[["value"]].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=effective_k,
        random_state=random_state,
        n_init=10
    )

    data["cluster"] = model.fit_predict(X_scaled)

    # Cluster IDs are arbitrary. Sort clusters by their original-unit
    # centroid to provide a consistent low-to-high display label.
    centroids_original = scaler.inverse_transform(
        model.cluster_centers_
    ).flatten()

    order = np.argsort(centroids_original)
    cluster_label_map = {
        int(old_label): f"Cluster {rank + 1}"
        for rank, old_label in enumerate(order)
    }

    data["cluster_label"] = data["cluster"].map(cluster_label_map)

    summary_rows = []
    for old_label, centroid in enumerate(centroids_original):
        subset = data[data["cluster"] == old_label]
        summary_rows.append({
            "Cluster": cluster_label_map[int(old_label)],
            "Centroid": float(centroid),
            "Records": int(len(subset)),
            "Minimum": float(subset["value"].min()),
            "Maximum": float(subset["value"].max())
        })

    summary = pd.DataFrame(summary_rows).sort_values("Centroid")

    return data, summary, effective_k


# ============================================================
# SIDEBAR: FILTERS AND GRAPH SETTINGS
# ============================================================

all_records = load_records()

st.sidebar.header("History Filters")

if all_records.empty:
    min_available_date = date.today() - timedelta(days=365)
    max_available_date = date.today()
else:
    min_available_date = min(all_records["record_date"])
    max_available_date = max(all_records["record_date"])

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_available_date, max_available_date),
    min_value=date(1900, 1, 1),
    max_value=date.today(),
    help="Choose the period used for history, graph traversal and analysis."
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
    start_date = end_date = date_range[0]
else:
    start_date = end_date = date_range

if start_date > end_date:
    st.sidebar.error("The start date must not be after the end date.")
    st.stop()

max_gap_days = st.sidebar.slider(
    "Maximum days between connected records",
    min_value=1,
    max_value=365,
    value=90,
    help=(
        "Graph edges connect records of the same metric when their "
        "dates are within this number of days."
    )
)

filtered_records = all_records[
    (all_records["record_date"] >= start_date)
    & (all_records["record_date"] <= end_date)
].copy()

# ============================================================
# OVERVIEW
# ============================================================

total_records = len(all_records)
filtered_count = len(filtered_records)
metric_count = (
    int(filtered_records["metric"].nunique())
    if not filtered_records.empty else 0
)

col1, col2, col3 = st.columns(3)
col1.metric("Total saved records", total_records)
col2.metric("Records in selected period", filtered_count)
col3.metric("Metrics tracked", metric_count)

st.divider()

# ============================================================
# ADD A RECORD
# ============================================================

st.subheader("Add Health Record")

with st.form("add_health_record_form", clear_on_submit=True):
    c1, c2 = st.columns(2)

    with c1:
        record_date_input = st.date_input(
            "Record date",
            value=date.today(),
            max_value=date.today()
        )

        metric_choice = st.selectbox(
            "Health metric",
            list(METRIC_CATALOGUE.keys())
        )

        if metric_choice == "Custom Metric":
            custom_metric = st.text_input(
                "Custom metric name",
                placeholder="e.g. Resting respiratory rate"
            )
            unit_input = st.text_input(
                "Unit",
                placeholder="e.g. breaths/min"
            )
        else:
            custom_metric = ""
            default_unit = METRIC_CATALOGUE[metric_choice]
            unit_input = st.text_input(
                "Unit",
                value=default_unit
            )

    with c2:
        value_input = st.number_input(
            "Measured value",
            value=0.0,
            step=0.1,
            format="%.4f"
        )

        notes_input = st.text_area(
            "Notes (optional)",
            placeholder="Context, measurement conditions, or source"
        )

    submitted = st.form_submit_button(
        "Save Health Record",
        type="primary",
        use_container_width=True
    )

    if submitted:
        metric_name = (
            custom_metric.strip()
            if metric_choice == "Custom Metric"
            else metric_choice
        )

        if not metric_name:
            st.error("Enter a name for the custom metric.")
        elif not unit_input.strip():
            st.error("Enter the measurement unit.")
        else:
            try:
                add_record(
                    record_date_input,
                    metric_name,
                    value_input,
                    unit_input,
                    notes_input
                )
                st.success("Health record saved permanently.")
                st.rerun()
            except sqlite3.Error as exc:
                st.error(f"Could not save the record: {exc}")

st.caption(
    "Use consistent units for the same metric. This tracker stores "
    "measurements as entered and does not independently validate "
    "their medical accuracy."
)

st.divider()

# ============================================================
# HISTORY TABLE AND DELETE
# ============================================================

st.subheader("Health Record History")

if filtered_records.empty:
    st.info(
        "No records exist in the selected date range. Add a record "
        "or expand the date range."
    )
else:
    display_df = filtered_records.copy()
    display_df["record_date"] = display_df["record_date"].astype(str)

    st.dataframe(
        display_df[
            ["id", "record_date", "metric", "value", "unit", "notes"]
        ].sort_values(["record_date", "id"], ascending=[False, False]),
        use_container_width=True,
        hide_index=True
    )

    csv_data = display_df[
        ["id", "record_date", "metric", "value", "unit", "notes"]
    ].to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download filtered history as CSV",
        data=csv_data,
        file_name="health_history.csv",
        mime="text/csv"
    )

    with st.expander("Delete a record"):
        delete_options = {
            (
                f"ID {int(row.id)} | {row.record_date} | "
                f"{row.metric}: {row.value} {row.unit}"
            ): int(row.id)
            for row in filtered_records.itertuples(index=False)
        }

        selected_delete_label = st.selectbox(
            "Select a record to delete",
            list(delete_options.keys())
        )

        confirm_delete = st.checkbox(
            "I confirm that I want to permanently delete this record."
        )

        if st.button("Delete selected record", type="secondary"):
            if not confirm_delete:
                st.warning("Confirm deletion before proceeding.")
            else:
                delete_record(delete_options[selected_delete_label])
                st.success("Record deleted.")
                st.rerun()

st.divider()

# ============================================================
# TREND VISUALIZATION
# ============================================================

st.subheader("Health Trends")

if filtered_records.empty:
    st.info("Add records to enable trend charts.")
else:
    available_metrics = sorted(filtered_records["metric"].unique())

    selected_metrics = st.multiselect(
        "Choose metrics to visualize",
        options=available_metrics,
        default=available_metrics[:1]
    )

    for metric in selected_metrics:
        metric_df = filtered_records[
            filtered_records["metric"] == metric
        ].copy()

        units = sorted(metric_df["unit"].dropna().unique())

        if len(units) > 1:
            st.warning(
                f"{metric} contains multiple units ({', '.join(units)}). "
                "Select or normalize a consistent unit before interpreting "
                "the trend."
            )
            continue

        metric_df["record_date"] = pd.to_datetime(
            metric_df["record_date"]
        )
        metric_df = metric_df.sort_values(["record_date", "id"])

        unit_label = units[0] if units else ""

        fig = px.line(
            metric_df,
            x="record_date",
            y="value",
            markers=True,
            title=f"{metric} over time",
            labels={
                "record_date": "Date",
                "value": f"Value ({unit_label})"
            },
            hover_data=["id", "notes"]
        )

        fig.update_layout(
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title=f"Value ({unit_label})"
        )

        st.plotly_chart(fig, use_container_width=True)

        if len(metric_df) >= 2:
            first_value = float(metric_df.iloc[0]["value"])
            last_value = float(metric_df.iloc[-1]["value"])
            change = last_value - first_value

            m1, m2, m3 = st.columns(3)
            m1.metric("First recorded value", f"{first_value:g} {unit_label}")
            m2.metric("Latest recorded value", f"{last_value:g} {unit_label}")
            m3.metric("Absolute change", f"{change:+g} {unit_label}")

            st.caption(
                "Change is the difference between the first and latest "
                "record in the selected period. It is not a diagnosis "
                "or an assessment of whether the change is beneficial."
            )

st.divider()

# ============================================================
# BFS AND DFS: GRAPH-BASED RECORD NAVIGATION
# ============================================================

st.subheader("Connected Health Record Exploration")
st.write(
    "BFS and DFS navigate a graph whose nodes are records. Two records "
    "are connected when they have the same metric and are within the "
    "selected date-gap threshold. The traversal shows connected records; "
    "it does not infer a medical relationship between them."
)

if filtered_records.empty:
    st.info("Add records to explore the record graph.")
else:
    graph = build_record_graph(
        filtered_records,
        max_gap_days=max_gap_days
    )

    record_options = {}
    for row in filtered_records.itertuples(index=False):
        label = (
            f"ID {int(row.id)} | {row.record_date} | "
            f"{row.metric}: {row.value} {row.unit}"
        )
        record_options[label] = int(row.id)

    start_label = st.selectbox(
        "Choose the starting record",
        options=list(record_options.keys()),
        key="graph_start_record"
    )
    start_node = record_options[start_label]

    g1, g2 = st.columns(2)

    with g1:
        if st.button("Run BFS", use_container_width=True):
            order = bfs_traversal(graph, start_node)
            st.session_state["bfs_result"] = order
            st.session_state["dfs_result"] = None

    with g2:
        if st.button("Run DFS", use_container_width=True):
            order = dfs_traversal(graph, start_node)
            st.session_state["dfs_result"] = order
            st.session_state["bfs_result"] = None

    result_order = None
    result_name = None

    if st.session_state.get("bfs_result") is not None:
        result_order = st.session_state["bfs_result"]
        result_name = "Breadth-First Search (BFS)"
    elif st.session_state.get("dfs_result") is not None:
        result_order = st.session_state["dfs_result"]
        result_name = "Depth-First Search (DFS)"

    if result_order is not None:
        st.markdown(f"**Traversal result: {result_name}**")
        st.write(f"Records visited: {len(result_order)}")

        record_lookup = filtered_records.set_index("id")
        traversal_rows = []

        for position, node_id in enumerate(result_order, start=1):
            row = record_lookup.loc[node_id]
            traversal_rows.append({
                "Visit order": position,
                "Record ID": int(node_id),
                "Date": str(row["record_date"]),
                "Metric": row["metric"],
                "Value": row["value"],
                "Unit": row["unit"],
                "Notes": row["notes"]
            })

        st.dataframe(
            pd.DataFrame(traversal_rows),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"The traversal reached {len(result_order)} records in the "
            "starting record's connected component. Other disconnected "
            "records are not included."
        )

    # Show graph structure statistics.
    total_nodes = len(graph)
    undirected_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
    isolated_nodes = sum(1 for neighbors in graph.values() if not neighbors)

    s1, s2, s3 = st.columns(3)
    s1.metric("Graph nodes", total_nodes)
    s2.metric("Graph edges", undirected_edges)
    s3.metric("Isolated records", isolated_nodes)

st.divider()

# ============================================================
# K-MEANS CLUSTERING
# ============================================================

st.subheader("K-Means: Group Similar Measurements")
st.write(
    "K-Means groups numerical records for one metric into clusters "
    "based on the measurement values. It requires multiple distinct "
    "values. Clusters are mathematical groupings, not medical risk "
    "categories or diagnoses."
)

if filtered_records.empty:
    st.info("Add records to use K-Means clustering.")
else:
    cluster_metric = st.selectbox(
        "Choose one metric to cluster",
        options=sorted(filtered_records["metric"].unique()),
        key="cluster_metric"
    )

    cluster_df = filtered_records[
        filtered_records["metric"] == cluster_metric
    ].copy()

    cluster_units = sorted(cluster_df["unit"].dropna().unique())

    if len(cluster_units) > 1:
        st.warning(
            "This metric has multiple units. K-Means is disabled for it "
            "because mixing units would make the numerical distances "
            "invalid. Use consistent units before clustering."
        )
    elif len(cluster_df) < 2:
        st.info("At least two records are required for clustering.")
    elif cluster_df["value"].nunique() < 2:
        st.info(
            "The selected metric needs at least two distinct values "
            "to form meaningful clusters."
        )
    else:
        max_k = min(8, len(cluster_df), cluster_df["value"].nunique())

        requested_k = st.slider(
            "Number of clusters (K)",
            min_value=2,
            max_value=max_k,
            value=min(3, max_k),
            help=(
                "Choose the number of groups. Different values of K "
                "can produce different partitions."
            )
        )

        if st.button("Run K-Means Clustering", type="primary"):
            try:
                clustered, cluster_summary, effective_k = run_kmeans(
                    cluster_df,
                    requested_k
                )

                st.session_state["kmeans_result"] = {
                    "metric": cluster_metric,
                    "data": clustered,
                    "summary": cluster_summary,
                    "k": effective_k
                }
            except ValueError as exc:
                st.error(str(exc))

    result = st.session_state.get("kmeans_result")

    if result is not None and result["metric"] == cluster_metric:
        clustered = result["data"].copy()
        cluster_summary = result["summary"]

        st.markdown(
            f"**Clustering completed:** {result['k']} clusters "
            f"for {cluster_metric}."
        )

        unit_label = cluster_units[0] if cluster_units else ""

        fig_cluster = px.scatter(
            clustered,
            x="record_date",
            y="value",
            color="cluster_label",
            hover_data=["id", "unit", "notes"],
            title=f"K-Means groups for {cluster_metric}",
            labels={
                "record_date": "Date",
                "value": f"Value ({unit_label})",
                "cluster_label": "Cluster"
            }
        )
        fig_cluster.update_layout(hovermode="closest")
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.markdown("**Cluster summary**")
        summary_display = cluster_summary.copy()
        summary_display["Centroid"] = summary_display["Centroid"].round(4)
        summary_display["Minimum"] = summary_display["Minimum"].round(4)
        summary_display["Maximum"] = summary_display["Maximum"].round(4)
        st.dataframe(
            summary_display,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("**Records with assigned clusters**")
        cluster_display = clustered.copy()
        cluster_display["record_date"] = (
            cluster_display["record_date"].astype(str)
        )
        st.dataframe(
            cluster_display[
                [
                    "id", "record_date", "metric", "value",
                    "unit", "cluster_label", "notes"
                ]
            ].sort_values(["cluster_label", "record_date"]),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "K-Means is sensitive to the selected K and the data "
            "distribution. Cluster labels are arbitrary and should not "
            "be interpreted as normal, abnormal, safe, or dangerous."
        )

st.divider()

# ============================================================
# OPTIONAL GEMINI SUMMARY
# ============================================================

st.subheader("Optional AI-Assisted History Summary")
st.caption(
    "This feature is optional. It sends selected health-history values "
    "to the configured Gemini service. Do not include identifying "
    "information in notes. AI output may be inaccurate and is not a "
    "medical diagnosis."
)

if st.button("Generate AI Summary"):
    if (
        not GEMINI_API_KEY
        or GEMINI_API_KEY == "PASTE_YOUR_API_KEY_HERE"
    ):
        st.warning(
            "Add your Gemini API key to GEMINI_API_KEY near the top "
            "of this file to enable this optional feature."
        )
    elif filtered_records.empty:
        st.info("There are no records in the selected period to summarize.")
    else:
        try:
            import google.generativeai as genai

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)

            summary_df = filtered_records[
                ["record_date", "metric", "value", "unit", "notes"]
            ].copy()
            summary_df["record_date"] = (
                summary_df["record_date"].astype(str)
            )

            # Send only the selected date range, and omit notes to
            # minimize unnecessary disclosure of personal information.
            safe_data = summary_df[
                ["record_date", "metric", "value", "unit"]
            ].to_dict(orient="records")

            prompt = f"""
You are helping a user understand a personal health-history log.

Analyze only the supplied measurements. Do not diagnose diseases,
estimate disease probabilities, or prescribe medication or treatment.
Do not assume reference ranges when age, sex, units, fasting status,
measurement conditions, or laboratory-specific ranges may matter.

Please provide:
1. A neutral summary of the recorded measurements.
2. Trends that can be directly observed from the data.
3. Missing context or limitations that affect interpretation.
4. Questions the user could discuss with a qualified clinician.

Do not describe a change as good or bad unless the provided data
supports that conclusion. Do not invent measurements or reference ranges.
If a potentially urgent symptom is not included, do not infer that it
is absent.

Health-history data:
{safe_data}
"""

            with st.spinner("Generating summary..."):
                response = model.generate_content(prompt)

            st.markdown("### AI Summary")
            st.write(response.text)

        except Exception as exc:
            st.error(
                "The AI summary could not be generated. Check your API "
                f"key, model availability, SDK configuration, and network. "
                f"Technical detail: {exc}"
            )

# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Privacy: records are stored in a local SQLite database file named "
    "'health_history.db' beside this page. Protect the project folder "
    "and its backups. This module is a record-keeping and educational "
    "tool, not a substitute for professional medical advice."
)
