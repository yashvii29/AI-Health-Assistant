
import io
import re
import sqlite3
from collections import deque
from datetime import date, datetime
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF Health Report Generator",
    page_icon="📄",
    layout="wide",
)

# Optional Gemini integration. The report generator works without it.
GEMINI_API_KEY = "AQ.Ab8RN6LDq8jPgJwvOqXjtYwkyiWbDgzO1YhoMA-uwQXCQ-85sQ"
GEMINI_MODEL = "gemini-3.8-flash"

# The Health History Tracker stores its database beside its page file.
DB_PATH = Path(__file__).resolve().parent / "health_history.db"

st.title("PDF Health Report Generator")
st.caption(
    "Generate downloadable reports from recorded health data, "
    "with explicit completeness checks and logical validation."
)

st.warning(
    "This application generates informational reports, not diagnoses. "
    "It does not verify the clinical accuracy of measurements or replace "
    "professional medical interpretation."
)

# ============================================================
# 1. HEALTH HISTORY DATABASE INTEGRATION
# ============================================================

EXPECTED_COLUMNS = [
    "id",
    "record_date",
    "metric",
    "value",
    "unit",
    "notes",
    "created_at",
]


def load_health_history():
    """Load the Health History Tracker's SQLite records if available."""
    if not DB_PATH.exists():
        return pd.DataFrame(columns=EXPECTED_COLUMNS), (
            "Health History database was not found."
        )

    try:
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            tables = pd.read_sql_query(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name = 'health_records'
                """,
                conn,
            )

            if tables.empty:
                return pd.DataFrame(columns=EXPECTED_COLUMNS), (
                    "The database exists, but health_records is missing."
                )

            df = pd.read_sql_query(
                """
                SELECT id, record_date, metric, value, unit, notes, created_at
                FROM health_records
                ORDER BY record_date ASC, id ASC
                """,
                conn,
            )

        for column in EXPECTED_COLUMNS:
            if column not in df.columns:
                df[column] = None

        df = df[EXPECTED_COLUMNS]

        if not df.empty:
            df["record_date"] = pd.to_datetime(
                df["record_date"], errors="coerce"
            )
            df["value"] = pd.to_numeric(
                df["value"], errors="coerce"
            )
            df = df.dropna(subset=["record_date", "value"])
            df["record_date"] = df["record_date"].dt.date

        return df, None

    except (sqlite3.Error, pd.errors.DatabaseError, ValueError) as exc:
        return pd.DataFrame(columns=EXPECTED_COLUMNS), (
            f"Could not read health history: {exc}"
        )


# ============================================================
# 2. ALGORITHM 1: BREADTH-FIRST SEARCH
#
# Traverse a directed graph of report-section dependencies.
# Nodes are report sections; edges mean "should be visited before".
# BFS uses a FIFO queue and visits nodes level by level.
# ============================================================

REPORT_GRAPH = {
    "Report Cover": ["Report Metadata", "Executive Summary"],
    "Report Metadata": ["Health Measurements", "Data Quality"],
    "Executive Summary": ["Health Measurements", "Limitations"],
    "Health Measurements": ["Trend Summary", "Data Quality"],
    "Trend Summary": ["Limitations"],
    "Data Quality": ["Limitations"],
    "Limitations": ["Final Disclaimer"],
    "Final Disclaimer": [],
}


def bfs_report_sections(graph, start):
    """Return BFS visit order and the reachable report sections."""
    if start not in graph:
        return []

    queue = deque([start])
    visited = {start}
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


def validate_report_graph(graph):
    """Validate all graph destinations before traversal."""
    for node, neighbors in graph.items():
        if not isinstance(neighbors, list):
            raise ValueError(f"Invalid edges for report section: {node}")

        for neighbor in neighbors:
            if neighbor not in graph:
                raise ValueError(
                    f"Unknown report section in graph: {neighbor}"
                )


validate_report_graph(REPORT_GRAPH)


# ============================================================
# 3. ALGORITHM 2: BACKWARD CHAINING
#
# Start from the goal REPORT_READY.
# Prove it by checking facts or recursively proving rule premises.
#
# Rules are represented as:
# conclusion <- one or more alternative sets of prerequisites.
#
# An empty prerequisite set means the rule is always provable.
# ============================================================

REPORT_RULES = {
    "REPORT_READY": [
        {
            "REPORT_TITLE_PRESENT",
            "DATE_RANGE_VALID",
            "DATA_AVAILABLE",
            "UNITS_VALID",
            "DISCLAIMER_INCLUDED",
        }
    ],
    "DATA_AVAILABLE": [
        {"HAS_HEALTH_RECORDS"},
        {"HAS_MANUAL_NOTES"},
    ],
}


def backward_chain(goal, facts, rules, trace=None, active=None):
    """
    Recursively attempt to prove a goal.

    Returns:
        (proved: bool, trace: list[str])

    Cycle protection prevents recursive rules from looping forever.
    """
    if trace is None:
        trace = []

    if active is None:
        active = set()

    if goal in facts:
        trace.append(f"PROVED: {goal} is an available fact.")
        return True, trace

    if goal in active:
        trace.append(f"FAILED: cycle detected while proving {goal}.")
        return False, trace

    alternatives = rules.get(goal, [])

    if not alternatives:
        trace.append(f"FAILED: {goal} has no supporting fact or rule.")
        return False, trace

    active = set(active)
    active.add(goal)
    trace.append(f"GOAL: attempt to prove {goal}.")

    for alternative_number, prerequisites in enumerate(
        alternatives, start=1
    ):
        trace.append(
            f"RULE {alternative_number}: {goal} requires "
            f"{', '.join(sorted(prerequisites)) or 'no prerequisites'}."
        )

        all_proved = True

        for prerequisite in sorted(prerequisites):
            proved, trace = backward_chain(
                prerequisite,
                facts,
                rules,
                trace,
                active,
            )

            if not proved:
                all_proved = False
                trace.append(
                    f"RULE {alternative_number} failed because "
                    f"{prerequisite} could not be proved."
                )
                break

        if all_proved:
            trace.append(f"PROVED: {goal} follows from RULE {alternative_number}.")
            return True, trace

    trace.append(f"FAILED: no rule could prove {goal}.")
    return False, trace


# ============================================================
# 4. ALGORITHM 3: PROPOSITIONAL RESOLUTION
#
# A clause is a set of literals, for example:
# {"HAS_RECORDS"} or {"~HAS_RECORDS", "~NO_RECORDS"}.
#
# Resolution rule:
# From (A OR P) and (B OR NOT P), infer (A OR B).
#
# Deriving the empty clause means contradiction.
# The implementation below performs resolution closure with a
# finite clause limit to avoid unbounded resource use.
# ============================================================

def negate_literal(literal):
    return literal[1:] if literal.startswith("~") else "~" + literal


def is_tautology(clause):
    """A clause containing P and NOT P is always true."""
    return any(
        negate_literal(literal) in clause
        for literal in clause
    )


def resolve_pair(clause_a, clause_b):
    """
    Return all non-tautological resolvents for a clause pair.
    Each resolvent removes one complementary literal pair.
    """
    resolvents = set()

    for literal in clause_a:
        opposite = negate_literal(literal)

        if opposite not in clause_b:
            continue

        combined = (
            (set(clause_a) - {literal})
            | (set(clause_b) - {opposite})
        )

        resolvent = frozenset(combined)

        if not is_tautology(resolvent):
            resolvents.add(resolvent)

    return resolvents


def resolution_refutation(initial_clauses, max_clauses=1000):
    """
    Attempt a resolution refutation.

    Returns:
        {
            "contradiction": bool,
            "derived_clauses": list,
            "steps": list[str],
            "complete": bool
        }

    A contradiction is established only if the empty clause is derived.
    If the clause limit is reached, the result is marked incomplete.
    """
    clauses = set()

    for clause in initial_clauses:
        normalized = frozenset(clause)

        # Tautologies add no constraint and can be omitted.
        if not is_tautology(normalized):
            clauses.add(normalized)

    if frozenset() in clauses:
        return {
            "contradiction": True,
            "derived_clauses": [frozenset()],
            "steps": ["The empty clause was present in the initial clauses."],
            "complete": True,
        }

    steps = []
    changed = True

    while changed:
        changed = False
        current = list(clauses)
        new_clauses = set()

        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                for resolvent in resolve_pair(current[i], current[j]):
                    if resolvent in clauses:
                        continue

                    if resolvent in new_clauses:
                        continue

                    new_clauses.add(resolvent)
                    steps.append(
                        f"Resolved {format_clause(current[i])} and "
                        f"{format_clause(current[j])} to "
                        f"{format_clause(resolvent)}."
                    )

                    if len(clauses) + len(new_clauses) > max_clauses:
                        return {
                            "contradiction": False,
                            "derived_clauses": list(clauses),
                            "steps": steps,
                            "complete": False,
                        }

                    if not resolvent:
                        return {
                            "contradiction": True,
                            "derived_clauses": (
                                list(clauses) + list(new_clauses)
                            ),
                            "steps": steps,
                            "complete": True,
                        }

        if new_clauses:
            clauses.update(new_clauses)
            changed = True

    return {
        "contradiction": False,
        "derived_clauses": list(clauses),
        "steps": steps,
        "complete": True,
    }


def format_clause(clause):
    if not clause:
        return "EMPTY CLAUSE (contradiction)"
    return "(" + " OR ".join(sorted(clause)) + ")"


# ============================================================
# 5. REPORT VALIDATION FACTS
#
# These facts describe document completeness and consistency.
# They do not make clinical claims about the user's health.
# ============================================================

def build_validation_facts(
    report_title,
    start_date,
    end_date,
    records,
    manual_notes,
):
    facts = {
        "DISCLAIMER_INCLUDED",
        "UNITS_VALID",
    }

    if report_title.strip():
        facts.add("REPORT_TITLE_PRESENT")

    if start_date <= end_date:
        facts.add("DATE_RANGE_VALID")
    else:
        facts.add("DATE_RANGE_INVALID")

    if not records.empty:
        facts.add("HAS_HEALTH_RECORDS")

        units_present = records["unit"].fillna("").astype(str).str.strip().ne("")

        if units_present.all():
            facts.add("ALL_UNITS_PRESENT")
        else:
            facts.add("MISSING_UNITS")
            facts.discard("UNITS_VALID")
    elif manual_notes.strip():
        facts.add("HAS_MANUAL_NOTES")
    else:
        facts.add("NO_DATA_AVAILABLE")
        facts.discard("UNITS_VALID")

    # Contradiction rules for mutually exclusive states.
    # If both states are asserted, resolution can derive the empty clause.
    clauses = [
        {fact} for fact in facts
    ]

    clauses.extend([
        {"~HAS_HEALTH_RECORDS", "~NO_DATA_AVAILABLE"},
        {"~HAS_MANUAL_NOTES", "~NO_DATA_AVAILABLE"},
        {"~ALL_UNITS_PRESENT", "~MISSING_UNITS"},
        {"~DATE_RANGE_VALID", "~DATE_RANGE_INVALID"},
    ])

    # Explicitly record missing-unit state when applicable.
    if "MISSING_UNITS" in facts:
        clauses.append({"MISSING_UNITS"})

    # Explicitly record that records exist, or that no data exists.
    # These facts are mutually exclusive by design.
    if "HAS_HEALTH_RECORDS" in facts:
        clauses.append({"HAS_HEALTH_RECORDS"})

    if "NO_DATA_AVAILABLE" in facts:
        clauses.append({"NO_DATA_AVAILABLE"})

    if "DATE_RANGE_VALID" in facts:
        clauses.append({"DATE_RANGE_VALID"})

    if "DATE_RANGE_INVALID" in facts:
        clauses.append({"DATE_RANGE_INVALID"})

    if "ALL_UNITS_PRESENT" in facts:
        clauses.append({"ALL_UNITS_PRESENT"})

    return facts, clauses


# ============================================================
# 6. PDF BUILDING HELPERS
# ============================================================

def safe_paragraph_text(value):
    """Escape user text before inserting it into ReportLab Paragraphs."""
    if value is None:
        return ""
    return escape(str(value)).replace("\n", "<br/>")


def make_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name="SectionCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=10,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        spaceAfter=5,
    ))

    styles.add(ParagraphStyle(
        name="SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#555555"),
    ))

    return styles


def make_table(rows, widths=None):
    table = Table(
        rows,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#F4F7FB"),
        ]),
    ]))

    return table


def build_pdf(
    report_title,
    subject_label,
    start_date,
    end_date,
    records,
    manual_notes,
    section_order,
    backward_proved,
    backward_trace,
    resolution_result,
):
    buffer = io.BytesIO()
    styles = make_styles()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=report_title,
        author="Health Report Generator",
    )

    story = []

    # Cover and metadata
    story.append(Paragraph(
        safe_paragraph_text(report_title),
        styles["ReportTitleCustom"],
    ))
    story.append(Paragraph(
        "Health Information Report",
        styles["BodyCustom"],
    ))
    story.append(Spacer(1, 5 * mm))

    metadata_rows = [
        ["Field", "Details"],
        ["Report subject", subject_label.strip() or "Not provided"],
        ["Report generated", datetime.now().strftime("%d %b %Y, %H:%M")],
        ["Period covered", f"{start_date} to {end_date}"],
        ["Data source", "User-entered Health History Tracker records"],
        ["Measurements included", str(len(records))],
    ]

    story.append(make_table(
        metadata_rows,
        widths=[42 * mm, 120 * mm],
    ))

    # Executive summary
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("Executive Summary", styles["SectionCustom"]))

    if records.empty:
        story.append(Paragraph(
            "No structured health measurements were found in the selected "
            "period. The report may contain the user's manually entered "
            "notes, but no measurement-based trend analysis is available.",
            styles["BodyCustom"],
        ))
    else:
        metric_count = records["metric"].nunique()
        story.append(Paragraph(
            f"The selected period contains {len(records)} records across "
            f"{metric_count} distinct metric(s). The report lists the "
            "measurements as stored and does not independently verify "
            "their clinical validity.",
            styles["BodyCustom"],
        ))

    if manual_notes.strip():
        story.append(Paragraph("User-Provided Notes", styles["SectionCustom"]))
        story.append(Paragraph(
            safe_paragraph_text(manual_notes),
            styles["BodyCustom"],
        ))

    # Measurements
    story.append(Paragraph("Health Measurements", styles["SectionCustom"]))

    if records.empty:
        story.append(Paragraph(
            "No structured measurements are available for this period.",
            styles["BodyCustom"],
        ))
    else:
        measurement_rows = [[
            "Date", "Metric", "Value", "Unit", "Notes"
        ]]

        for row in records.sort_values(
            ["record_date", "id"]
        ).itertuples(index=False):
            measurement_rows.append([
                str(row.record_date),
                Paragraph(safe_paragraph_text(row.metric), styles["SmallCustom"]),
                f"{row.value:g}",
                Paragraph(safe_paragraph_text(row.unit), styles["SmallCustom"]),
                Paragraph(
                    safe_paragraph_text(row.notes or ""),
                    styles["SmallCustom"],
                ),
            ])

        story.append(make_table(
            measurement_rows,
            widths=[24 * mm, 35 * mm, 22 * mm, 22 * mm, 59 * mm],
        ))

        # Metric-wise descriptive summary; no medical interpretation.
        story.append(Paragraph(
            "Descriptive Summary by Metric",
            styles["SectionCustom"],
        ))

        summary_rows = [[
            "Metric", "Unit", "Count", "Minimum", "Maximum", "Latest"
        ]]

        for metric, group in records.groupby("metric", sort=True):
            units = sorted(
                group["unit"].fillna("").astype(str).unique()
            )

            if len(units) != 1:
                unit_label = "Mixed units"
            else:
                unit_label = units[0]

            ordered = group.sort_values(["record_date", "id"])

            summary_rows.append([
                Paragraph(safe_paragraph_text(metric), styles["SmallCustom"]),
                Paragraph(safe_paragraph_text(unit_label), styles["SmallCustom"]),
                str(len(group)),
                f"{group['value'].min():g}",
                f"{group['value'].max():g}",
                f"{ordered.iloc[-1]['value']:g}",
            ])

        story.append(make_table(
            summary_rows,
            widths=[38 * mm, 27 * mm, 16 * mm, 24 * mm, 24 * mm, 28 * mm],
        ))

    # Logical validation section
    story.append(Paragraph(
        "Document Validation",
        styles["SectionCustom"],
    ))

    story.append(Paragraph(
        "Backward Chaining result: "
        + ("REPORT_READY was proved by the configured rules."
           if backward_proved
           else "REPORT_READY could not be proved by the configured rules."),
        styles["BodyCustom"],
    ))

    story.append(Paragraph(
        "Resolution result: "
        + (
            "a contradiction was derived."
            if resolution_result["contradiction"]
            else "no contradiction was derived from the checked clauses."
        ),
        styles["BodyCustom"],
    ))

    story.append(Paragraph(
        "These checks concern report completeness and logical consistency "
        "only. They do not validate a diagnosis, treatment, or the clinical "
        "meaning of any measurement.",
        styles["BodyCustom"],
    ))

    # BFS section order
    story.append(Paragraph(
        "Report Section Traversal (BFS)",
        styles["SectionCustom"],
    ))

    story.append(Paragraph(
        "The following sections were visited in breadth-first order:",
        styles["BodyCustom"],
    ))

    for index, section in enumerate(section_order, start=1):
        story.append(Paragraph(
            f"{index}. {safe_paragraph_text(section)}",
            styles["BodyCustom"],
        ))

    # Disclaimer
    story.append(Paragraph(
        "Important Limitations",
        styles["SectionCustom"],
    ))

    story.append(Paragraph(
        "This document is a summary of information entered into the "
        "application. Values, dates, units, and notes may be incomplete "
        "or inaccurate. Reference ranges may depend on the laboratory, "
        "measurement method, age, other context, and units. No diagnosis "
        "or treatment recommendation is provided. Consult a qualified "
        "healthcare professional for interpretation and medical decisions. "
        "Seek urgent medical care for severe or rapidly worsening symptoms.",
        styles["BodyCustom"],
    ))

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph(
        "Generated by the Health Report Generator.",
        styles["SmallCustom"],
    ))

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(
            18 * mm,
            10 * mm,
            "Health Information Report — Not a diagnosis",
        )
        canvas.drawRightString(
            A4[0] - 18 * mm,
            10 * mm,
            f"Page {document.page}",
        )
        canvas.restoreState()

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# 7. USER INTERFACE: REPORT DETAILS
# ============================================================

st.header("1. Report Details")

with st.form("report_details_form"):
    report_title = st.text_input(
        "Report title",
        value="Personal Health History Report",
        max_chars=120,
    )

    subject_label = st.text_input(
        "Subject label (optional)",
        placeholder="e.g. Personal record",
        max_chars=100,
    )

    c1, c2 = st.columns(2)

    with c1:
        start_date = st.date_input(
            "Start date",
            value=date.today().replace(day=1),
            max_value=date.today(),
        )

    with c2:
        end_date = st.date_input(
            "End date",
            value=date.today(),
            max_value=date.today(),
        )

    manual_notes = st.text_area(
        "Additional notes (optional)",
        placeholder=(
            "Enter contextual notes to include in the report. "
            "Avoid unnecessary identifying information."
        ),
        max_chars=5000,
    )

    submitted = st.form_submit_button(
        "Load Records and Validate",
        type="primary",
        use_container_width=True,
    )

if submitted:
    st.session_state["report_inputs"] = {
        "report_title": report_title.strip(),
        "subject_label": subject_label.strip(),
        "start_date": start_date,
        "end_date": end_date,
        "manual_notes": manual_notes.strip(),
    }
    st.session_state.pop("generated_pdf", None)

inputs = st.session_state.get("report_inputs")

if inputs:
    report_title = inputs["report_title"]
    subject_label = inputs["subject_label"]
    start_date = inputs["start_date"]
    end_date = inputs["end_date"]
    manual_notes = inputs["manual_notes"]

    st.divider()
    st.header("2. Data and Validation")

    all_records, db_warning = load_health_history()

    if db_warning:
        st.warning(db_warning)

    if start_date > end_date:
        st.error("The start date must not be after the end date.")
        st.stop()

    if all_records.empty:
        filtered_records = pd.DataFrame(columns=EXPECTED_COLUMNS)
    else:
        filtered_records = all_records[
            (all_records["record_date"] >= start_date)
            & (all_records["record_date"] <= end_date)
        ].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Records included", len(filtered_records))
    c2.metric(
        "Metrics included",
        int(filtered_records["metric"].nunique())
        if not filtered_records.empty else 0,
    )
    c3.metric(
        "Notes provided",
        "Yes" if manual_notes.strip() else "No",
    )

    if filtered_records.empty:
        st.info(
            "No structured records were found in this period. "
            "You can still generate a report with notes, but the "
            "measurement section will be empty."
        )
    else:
        st.dataframe(
            filtered_records[
                ["id", "record_date", "metric", "value", "unit", "notes"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # BFS: section traversal
    # --------------------------------------------------------

    st.subheader("Algorithm 1 — BFS Report Section Traversal")

    section_order = bfs_report_sections(REPORT_GRAPH, "Report Cover")

    st.write("Breadth-first traversal order:")
    st.write(" → ".join(section_order))

    with st.expander("View report dependency graph"):
        graph_rows = []

        for section, dependencies in REPORT_GRAPH.items():
            graph_rows.append({
                "Report section": section,
                "Next sections": ", ".join(dependencies) or "None",
            })

        st.dataframe(
            pd.DataFrame(graph_rows),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Backward chaining: prove REPORT_READY
    # --------------------------------------------------------

    st.subheader("Algorithm 2 — Backward Chaining")

    facts, validation_clauses = build_validation_facts(
        report_title,
        start_date,
        end_date,
        filtered_records,
        manual_notes,
    )

    backward_proved, backward_trace = backward_chain(
        "REPORT_READY",
        facts,
        REPORT_RULES,
    )

    if backward_proved:
        st.success(
            "Backward chaining proved REPORT_READY under the "
            "configured report-completeness rules."
        )
    else:
        st.warning(
            "Backward chaining could not prove REPORT_READY. "
            "Review the trace below for missing prerequisites."
        )

    with st.expander("View backward-chaining proof trace", expanded=True):
        for line in backward_trace:
            st.write("- " + line)

    st.caption(
        "The result reflects the configured rules only. It is not a "
        "certification of clinical accuracy or PDF correctness."
    )

    # --------------------------------------------------------
    # Resolution: consistency of validation clauses
    # --------------------------------------------------------

    st.subheader("Algorithm 3 — Resolution")

    resolution_result = resolution_refutation(validation_clauses)

    if resolution_result["contradiction"]:
        st.error(
            "Resolution derived the empty clause: the validation "
            "clauses are logically inconsistent."
        )
    elif resolution_result["complete"]:
        st.success(
            "Resolution completed without deriving the empty clause. "
            "No contradiction was found in the checked clauses."
        )
    else:
        st.warning(
            "The resolution clause limit was reached. The consistency "
            "check is incomplete; do not treat this as proof of consistency."
        )

    with st.expander("View resolution clauses and derivations"):
        st.markdown("**Initial clauses**")

        for clause in validation_clauses:
            st.code(format_clause(frozenset(clause)))

        st.markdown("**Resolution steps**")

        if resolution_result["steps"]:
            for step in resolution_result["steps"]:
                st.write("- " + step)
        else:
            st.write(
                "No new resolvents were required to reach the result."
            )

    st.caption(
        "Resolution checks the propositional validation facts supplied "
        "by this application. It does not determine whether a health "
        "measurement is medically normal or abnormal."
    )

    # --------------------------------------------------------
    # PDF generation
    # --------------------------------------------------------

    st.divider()
    st.header("3. Generate PDF")

    if not report_title:
        st.error("A report title is required.")
    else:
        if st.button(
            "Generate PDF Report",
            type="primary",
            use_container_width=True,
        ):
            try:
                pdf_bytes = build_pdf(
                    report_title=report_title,
                    subject_label=subject_label,
                    start_date=start_date,
                    end_date=end_date,
                    records=filtered_records,
                    manual_notes=manual_notes,
                    section_order=section_order,
                    backward_proved=backward_proved,
                    backward_trace=backward_trace,
                    resolution_result=resolution_result,
                )

                st.session_state["generated_pdf"] = pdf_bytes
                st.session_state["generated_pdf_name"] = (
                    re.sub(
                        r"[^A-Za-z0-9_-]+",
                        "_",
                        report_title,
                    ).strip("_") or "health_report"
                )

            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")

    generated_pdf = st.session_state.get("generated_pdf")

    if generated_pdf:
        filename = (
            st.session_state.get("generated_pdf_name", "health_report")
            + ".pdf"
        )

        st.download_button(
            "Download PDF Report",
            data=generated_pdf,
            file_name=filename,
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

        st.success(
            "PDF created. Download it using the button above."
        )

# ============================================================
# 8. OPTIONAL GEMINI API PLACEHOLDER
# ============================================================

st.divider()
st.header("4. Optional AI Explanation")

st.write(
    "This optional feature can create a plain-language description "
    "of the report contents. It does not alter the source records "
    "or replace the validation algorithms."
)

if st.button("Generate Optional AI Summary"):
    if (
        not GEMINI_API_KEY
        or GEMINI_API_KEY == "PASTE_YOUR_API_KEY_HERE"
    ):
        st.warning(
            "Add your Gemini API key to GEMINI_API_KEY near the top "
            "of this file to enable this feature."
        )
    elif not st.session_state.get("report_inputs"):
        st.info(
            "Complete the report details and click "
            "'Load Records and Validate' first."
        )
    else:
        try:
            import google.generativeai as genai

            current_inputs = st.session_state["report_inputs"]
            current_records, _ = load_health_history()

            current_records = current_records[
                (current_records["record_date"] >= current_inputs["start_date"])
                & (current_records["record_date"] <= current_inputs["end_date"])
            ].copy()

            safe_records = current_records[
                ["record_date", "metric", "value", "unit"]
            ].copy()

            if not safe_records.empty:
                safe_records["record_date"] = (
                    safe_records["record_date"].astype(str)
                )

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)

            prompt = f"""
Create a concise, neutral explanation of the following personal
health-history records for an informational report.

Report title: {current_inputs['report_title']}
Period: {current_inputs['start_date']} to {current_inputs['end_date']}
Measurements: {safe_records.to_dict(orient='records')}

Requirements:
- Do not diagnose or estimate disease probabilities.
- Do not prescribe medicines, doses, or treatment.
- Do not invent normal ranges or medical conclusions.
- Describe only trends directly supported by the measurements.
- State that units, context, and reference ranges may affect interpretation.
- Encourage consultation with a qualified healthcare professional.
"""

            with st.spinner("Generating AI summary..."):
                response = model.generate_content(prompt)

            st.markdown("### AI-Generated Summary")
            st.write(response.text)

        except Exception as exc:
            st.error(
                "AI summary generation failed. Check your API key, "
                "model availability, SDK configuration, and network. "
                f"Details: {exc}"
            )

# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Data source: local Health History Tracker SQLite database, when "
    "available. Report contents are based on saved measurements and "
    "user-entered notes. Protect downloaded reports because they may "
    "contain sensitive personal information."
)
