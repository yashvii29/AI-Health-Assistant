import re
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import PyPDF2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
GEMINI_API_KEY = "AQ.Ab8RN6LDq8jPgJwvOqXjtYwkyiWbDgzO1YhoMA-uwQXCQ-85sQ"
GEMINI_MODEL = "gemini-3.8-flash"

st.set_page_config(page_title="AI Medical Report Analyzer", page_icon="📄", layout="wide")
st.title("📄 AI Medical Report Analyzer")
st.caption("PDF extraction, lab-range comparison, Decision Tree, K-Means, and logical inference.")
st.warning(
    "Educational prototype only; not a diagnostic device. Verify every extracted value, unit, "
    "and reference interval against the original report. Seek professional medical advice."
)

# Alias list is used to identify report rows. No universal ranges are imposed:
# lab reference intervals vary by lab, units, age, sex, pregnancy status, and method.
ALIASES = {
    "Hemoglobin": ["hemoglobin", "haemoglobin", "hb"],
    "RBC": ["red blood cell count", "red blood cells", "rbc"],
    "WBC": ["white blood cell count", "total leukocyte count", "tlc", "wbc"],
    "Platelets": ["platelet count", "platelets", "platelet"],
    "Glucose": ["fasting blood glucose", "fasting glucose", "blood glucose", "blood sugar", "glucose", "fbs", "rbs"],
    "HbA1c": ["hba1c", "hemoglobin a1c", "haemoglobin a1c", "glycated hemoglobin"],
    "Total Cholesterol": ["total cholesterol", "cholesterol total", "serum cholesterol"],
    "LDL Cholesterol": ["ldl cholesterol", "ldl-c", "ldl"],
    "HDL Cholesterol": ["hdl cholesterol", "hdl-c", "hdl"],
    "Triglycerides": ["triglycerides", "triglyceride", "tg"],
    "Creatinine": ["serum creatinine", "creatinine"],
    "Urea": ["blood urea", "serum urea", "urea"],
    "TSH": ["thyroid stimulating hormone", "thyroid-stimulating hormone", "tsh"],
    "T3": ["triiodothyronine", "total t3", "t3"],
    "T4": ["thyroxine", "total t4", "t4"],
    "Free T3": ["free t3", "ft3"],
    "Free T4": ["free t4", "ft4"],
    "ALT / SGPT": ["alanine aminotransferase", "sgpt", "alt"],
    "AST / SGOT": ["aspartate aminotransferase", "sgot", "ast"],
    "Vitamin D": ["25-oh vitamin d", "25 hydroxy vitamin d", "vitamin d"],
    "Vitamin B12": ["vitamin b12", "vitamin b-12", "cobalamin"],
    "Ferritin": ["ferritin"],
    "Sodium": ["sodium", "na+"],
    "Potassium": ["potassium", "k+"],
    "Calcium": ["serum calcium", "calcium"],
    "CRP": ["c-reactive protein", "crp"],
    "eGFR": ["estimated glomerular filtration rate", "egfr"],
}

NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
RANGE_RE = re.compile(rf"(?P<lo>{NUMBER})\s*(?:-|to)\s*(?P<hi>{NUMBER})", re.I)


def extract_pdf_text(uploaded):
    reader = PyPDF2.PdfReader(uploaded)
    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            chunks.append("")
    return "\n".join(chunks).strip()


def clean_num(s):
    return float(s.replace(",", ""))


def find_parameter(line):
    low = line.lower()
    found = []
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            pattern = r"(?<![a-z0-9])" + re.escape(alias.lower()) + r"(?![a-z0-9])"
            m = re.search(pattern, low)
            if m:
                found.append((len(alias), m.start(), canonical, alias))
    if not found:
        return None
    found.sort(key=lambda x: (-x[0], x[1]))
    return found[0][2], found[0][3], found[0][1]


def parse_result_line(line):
    found = find_parameter(line)
    if not found:
        return None
    canonical, alias, pos = found
    tail = line[pos + len(alias):]
    tail = re.sub(r"^\s*[:=]?\s*", "", tail)
    vm = re.search(NUMBER, tail)
    if not vm:
        return None
    value = clean_num(vm.group())
    remainder = tail[vm.end():]

    # Reference range is captured only from text following the result.
    rm = RANGE_RE.search(remainder)
    lo = hi = np.nan
    if rm:
        lo, hi = clean_num(rm.group("lo")), clean_num(rm.group("hi"))
        if lo > hi:
            lo, hi = hi, lo

    # Capture a simple unit immediately after the result when present.
    um = re.match(r"\s*([a-zA-Zµμ%/³^0-9._-]{1,20})?", remainder)
    unit = (um.group(1) or "").strip() if um else ""
    fm = re.search(r"(?<![A-Za-z])([HL])(?=\s|$|[),;])", line.strip(), re.I)
    flag = ("High" if fm.group(1).upper() == "H" else "Low") if fm else ""

    return {
        "Parameter": canonical, "Value": value, "Unit": unit,
        "Reference Low": lo, "Reference High": hi, "Lab Flag": flag,
        "Source Line": line.strip(),
    }


def extract_results(text):
    rows = []
    first_seen = {}
    for line in text.replace("\u00a0", " ").replace("−", "-").splitlines():
        row = parse_result_line(line.strip())
        if not row:
            continue
        row["Review Note"] = ""
        if row["Parameter"] in first_seen:
            row["Review Note"] = "Duplicate label: verify current result manually."
            rows[first_seen[row["Parameter"]]]["Review Note"] = (
                "Duplicate label: verify current result manually."
            )
        else:
            first_seen[row["Parameter"]] = len(rows)
        rows.append(row)
    return rows


def interpret(row):
    lo, hi, value, flag = (
        row["Reference Low"], row["Reference High"], row["Value"], row["Lab Flag"]
    )
    if pd.notna(lo) and pd.notna(hi):
        if value < lo:
            return "Low"
        if value > hi:
            return "High"
        return "Within extracted lab range"
    if flag in ("Low", "High"):
        return f"Lab-flagged {flag}"
    return "Cannot classify: reference range not found"


def facts_from_rows(rows):
    facts = set()
    for row in rows:
        key = re.sub(r"[^A-Z0-9]+", "_", row["Parameter"].upper()).strip("_")
        status = row["Interpretation"]
        if status == "High" or status == "Lab-flagged High":
            facts.add(f"{key}_HIGH")
        elif status == "Low" or status == "Lab-flagged Low":
            facts.add(f"{key}_LOW")
        elif status == "Within extracted lab range":
            facts.add(f"{key}_WITHIN_LAB_RANGE")
    if any(x.endswith("_HIGH") or x.endswith("_LOW") for x in facts):
        facts.add("AT_LEAST_ONE_FLAGGED_ABNORMALITY")
    if not rows:
        facts.add("NO_PARAMETERS_EXTRACTED")
    return facts


# ============================================================
# FORWARD CHAINING / BACKWARD CHAINING / RESOLUTION
# Facts describe report flags only; no disease is inferred.
# ============================================================
RULES = [
    (("GLUCOSE_HIGH",), "GLUCOSE_REVIEW_SUGGESTED",
     "Glucose is above the report's extracted interval."),
    (("HBA1C_HIGH",), "HBA1C_REVIEW_SUGGESTED",
     "HbA1c is above the report's extracted interval."),
    (("CREATININE_HIGH",), "CREATININE_REVIEW_SUGGESTED",
     "Creatinine is above the report's extracted interval."),
    (("HEMOGLOBIN_LOW",), "HEMOGLOBIN_REVIEW_SUGGESTED",
     "Hemoglobin is below the report's extracted interval."),
    (("TSH_HIGH",), "THYROID_MARKER_REVIEW_SUGGESTED",
     "TSH is above the report's extracted interval."),
    (("TSH_LOW",), "THYROID_MARKER_REVIEW_SUGGESTED",
     "TSH is below the report's extracted interval."),
    (("AT_LEAST_ONE_FLAGGED_ABNORMALITY",), "CLINICIAN_REVIEW_MAY_BE_USEFUL",
     "At least one result is outside an extracted range or flagged by the lab."),
    (("NO_PARAMETERS_EXTRACTED",), "MANUAL_REVIEW_REQUIRED",
     "No supported laboratory result rows were extracted."),
]


def forward_chaining(initial):
    facts, trace, changed = set(initial), [], True
    while changed:
        changed = False
        for premises, conclusion, explanation in RULES:
            if all(p in facts for p in premises) and conclusion not in facts:
                facts.add(conclusion)
                trace.append({
                    "Premises": " AND ".join(premises),
                    "Derived fact": conclusion,
                    "Explanation": explanation,
                })
                changed = True
    return facts, trace


def backward_chaining(goal, initial):
    facts, active, memo, trace = set(initial), set(), {}, []

    def prove(target):
        if target in facts:
            trace.append(f"PROVED {target}: present in extracted facts.")
            return True
        if target in memo:
            return memo[target]
        if target in active:
            return False
        active.add(target)
        for premises, conclusion, explanation in RULES:
            if conclusion != target:
                continue
            trace.append(f"Trying rule: {' AND '.join(premises)} -> {conclusion}")
            outcomes = [prove(p) for p in premises]
            if all(outcomes):
                trace.append(f"PROVED {target}: {explanation}")
                active.remove(target)
                memo[target] = True
                return True
        active.remove(target)
        memo[target] = False
        trace.append(f"NOT PROVED: {target}")
        return False

    return prove(goal), trace


def negate(literal):
    return literal[1:] if literal.startswith("~") else "~" + literal


def rule_clause(premises, conclusion):
    return frozenset([negate(p) for p in premises] + [conclusion])


def tautology(clause):
    return any(negate(x) in clause for x in clause)


def resolve(a, b):
    out = set()
    for lit in a:
        opposite = negate(lit)
        if opposite in b:
            c = frozenset((set(a) - {lit}) | (set(b) - {opposite}))
            if not tautology(c):
                out.add(c)
    return out


def resolution_entails(query, initial, max_clauses=1200, max_rounds=60):
    clauses = {frozenset([x]) for x in initial}
    clauses.update(rule_clause(p, c) for p, c, _ in RULES)
    clauses.add(frozenset([negate(query)]))
    trace, seen = [], set()

    for round_no in range(1, max_rounds + 1):
        current, new = list(clauses), set()
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                a, b = current[i], current[j]
                pair = tuple(sorted((tuple(sorted(a)), tuple(sorted(b)))))
                if pair in seen:
                    continue
                seen.add(pair)
                for c in resolve(a, b):
                    if not c:
                        trace.append(f"Round {round_no}: empty clause derived; query is entailed.")
                        return True, trace
                    if c not in clauses:
                        new.add(c)
                        trace.append("Derived clause: " + " OR ".join(sorted(c)))
        if not new:
            trace.append("No new clauses; query was not proven by this knowledge base.")
            return False, trace
        clauses.update(new)
        if len(clauses) > max_clauses:
            trace.append("Clause limit reached; result inconclusive.")
            return None, trace
    trace.append("Round limit reached; result inconclusive.")
    return None, trace


# ============================================================
# DECISION TREE: SYNTHETIC DEMONSTRATION ONLY
# It classifies review workload from counts, not disease or severity.
# ============================================================
DT_FEATURES = ["abnormal_count", "high_count", "low_count", "missing_range_count"]


@st.cache_resource
def train_demo_tree():
    rng = np.random.default_rng(42)
    n = 1200
    data = pd.DataFrame({
        "abnormal_count": rng.integers(0, 9, n),
        "high_count": rng.integers(0, 6, n),
        "low_count": rng.integers(0, 6, n),
        "missing_range_count": rng.integers(0, 9, n),
    })
    score = data["abnormal_count"] + 0.25 * data["high_count"] + 0.25 * data["low_count"]
    data["target"] = np.select(
        [score >= 7, score >= 3],
        ["More rows to review", "Some rows to review"],
        default="Fewer rows to review",
    )
    xtr, xte, ytr, yte = train_test_split(
        data[DT_FEATURES], data["target"], test_size=0.2,
        random_state=42, stratify=data["target"]
    )
    model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
    model.fit(xtr, ytr)
    return model, accuracy_score(yte, model.predict(xte))


def tree_features(rows):
    high = sum(r["Interpretation"] in ("High", "Lab-flagged High") for r in rows)
    low = sum(r["Interpretation"] in ("Low", "Lab-flagged Low") for r in rows)
    missing = sum(
        not (pd.notna(r["Reference Low"]) and pd.notna(r["Reference High"]))
        for r in rows
    )
    abnormal = high + low
    return pd.DataFrame([{
        "abnormal_count": abnormal,
        "high_count": high,
        "low_count": low,
        "missing_range_count": missing,
    }], columns=DT_FEATURES)


# ============================================================
# K-MEANS: clusters relative-to-reference positions only
# Clusters are not clinical categories.
# ============================================================
def run_kmeans(rows, k):
    usable = []
    for row in rows:
        lo, hi = row["Reference Low"], row["Reference High"]
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            usable.append({
                "Parameter": row["Parameter"],
                "Value": row["Value"],
                "Unit": row["Unit"],
                "Relative interval position": (row["Value"] - lo) / (hi - lo),
                "Interpretation": row["Interpretation"],
            })
    if len(usable) < 3:
        return None
    df = pd.DataFrame(usable)
    # A 1-D clustering is an exploratory grouping of interval positions.
    x = StandardScaler().fit_transform(df[["Relative interval position"]])
    model = KMeans(n_clusters=min(k, len(df)), random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(x)
    return df


# ============================================================
# OPTIONAL GEMINI
# ============================================================
def gemini_explanation(rows, facts):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        raise ValueError("Add your Gemini API key in GEMINI_API_KEY first.")
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = f"""
Explain these extracted laboratory results cautiously in plain language.
Do not diagnose, invent ranges, prescribe medicine, or infer disease from one
abnormal value. State that extraction must be checked against the original PDF.
A lack of flags does not prove the report is normal.

Rows: {json.dumps(rows, indent=2, default=str)}
Logical facts: {sorted(facts)}

Provide a short summary, rows needing manual verification, limitations, and
general advice to discuss concerning results with a qualified clinician.
"""
    response = model.generate_content(prompt)
    return getattr(response, "text", None) or "No text returned by Gemini."


# ============================================================
# STREAMLIT UI
# ============================================================
uploaded = st.file_uploader("Upload medical/laboratory report (PDF)", type=["pdf"])

if uploaded:
    try:
        report_text = extract_pdf_text(uploaded)
    except Exception as exc:
        st.error(f"Could not read PDF: {exc}")
        st.stop()

    if not report_text.strip():
        st.error(
            "No selectable text was extracted. This may be a scanned PDF. "
            "OCR is not implemented; use a text-based PDF or manually verify "
            "and transcribe the values."
        )
        st.stop()

    with st.expander("View extracted PDF text"):
        st.text_area("PDF text", report_text, height=220)

    parsed = extract_results(report_text)
    rows = []
    for item in parsed:
        row = dict(item)
        row["Interpretation"] = interpret(row)
        lo, hi = row["Reference Low"], row["Reference High"]
        row["Reference Range"] = f"{lo:g} - {hi:g}" if pd.notna(lo) and pd.notna(hi) else "Not extracted"
        rows.append(row)

    st.subheader("🧪 Extracted results")
    if rows:
        cols = ["Parameter", "Value", "Unit", "Reference Range", "Lab Flag",
                "Interpretation", "Review Note", "Source Line"]
        st.dataframe(pd.DataFrame(rows)[cols], use_container_width=True, hide_index=True)
        st.caption(
            "Always compare the value, unit, reference interval, and source line "
            "with the original PDF. Table layouts can cause extraction errors."
        )
    else:
        st.error(
            "No supported result rows were extracted. Do not interpret this as "
            "a normal report. The parser may not support this PDF layout."
        )

    flagged = [
        r for r in rows
        if r["Interpretation"] in ("High", "Low")
        or r["Interpretation"].startswith("Lab-flagged")
    ]
    facts = facts_from_rows(rows)
    derived, ftrace = forward_chaining(facts)
    bgoal = "CLINICIAN_REVIEW_MAY_BE_USEFUL"
    bresult, btrace = backward_chaining(bgoal, facts)
    rresult, rtrace = resolution_entails(bgoal, facts)

    st.divider()
    st.subheader("📊 Report overview")
    a, b, c = st.columns(3)
    a.metric("Rows extracted", len(rows))
    b.metric("Flagged/out-of-range rows", len(flagged))
    c.metric("Rows without reference range", sum(
        r["Interpretation"] == "Cannot classify: reference range not found" for r in rows
    ))

    if flagged:
        st.subheader("⚠️ Rows requiring review")
        st.dataframe(pd.DataFrame([{
            "Parameter": r["Parameter"], "Value": r["Value"], "Unit": r["Unit"],
            "Reference Range": r["Reference Range"], "Interpretation": r["Interpretation"],
            "Source Line": r["Source Line"],
        } for r in flagged]), use_container_width=True, hide_index=True)
    else:
        st.info(
            "No out-of-range results were established from extracted reference "
            "intervals or explicit flags. This does not prove that the report is normal."
        )

    t1, t2, t3, t4, t5 = st.tabs([
        "Decision Tree", "K-Means", "Forward Chaining", "Backward Chaining", "Resolution"
    ])

    with t1:
        st.subheader("🌳 Decision Tree Classification")
        st.write(
            "This is an algorithm demonstration that classifies the number of "
            "rows requiring review. It is not trained on clinical outcomes and "
            "does not predict a disease."
        )
        tree, accuracy = train_demo_tree()
        prediction = tree.predict(tree_features(rows))[0]
        probs = tree.predict_proba(tree_features(rows))[0]
        st.metric("Demo class", prediction)
        st.metric("Held-out synthetic-label accuracy", f"{accuracy * 100:.1f}%")
        st.warning(
            "Accuracy only measures agreement with synthetic labels generated "
            "by predefined rules. It is not clinical accuracy."
        )
        st.dataframe(pd.DataFrame({
            "Class": tree.classes_, "Model score (%)": np.round(probs * 100, 2)
        }).sort_values("Model score (%)", ascending=False), use_container_width=True, hide_index=True)
        st.code(export_text(tree, feature_names=DT_FEATURES), language="text")
        fig, ax = plt.subplots(figsize=(14, 6))
        plot_tree(tree, feature_names=DT_FEATURES, class_names=tree.classes_,
                  max_depth=3, filled=False, rounded=True, fontsize=8, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with t2:
        st.subheader("🧩 K-Means Clustering")
        k = st.slider("Number of clusters", 2, 5, 2)
        clusters = run_kmeans(rows, k)
        st.caption(
            "K-Means groups results by their position relative to each result's "
            "extracted lab range. Clusters are not diagnoses or severity levels."
        )
        if clusters is None:
            st.info("At least three results with extracted numeric reference intervals are needed.")
        else:
            st.dataframe(clusters, use_container_width=True, hide_index=True)
            st.bar_chart(clusters.groupby("Cluster")["Relative interval position"].mean())

    with t3:
        st.subheader("➡️ Forward Chaining")
        st.write("Initial facts:", sorted(facts))
        if ftrace:
            st.dataframe(pd.DataFrame(ftrace), use_container_width=True, hide_index=True)
        else:
            st.info("No rules fired.")
        st.write("Derived facts:", sorted(derived))

    with t4:
        st.subheader("⬅️ Backward Chaining")
        st.write("Goal:", bgoal)
        st.write("Proved:", bresult)
        for line in btrace:
            st.write("- " + line)
        st.caption("A goal not proved by this limited rule base does not mean the report is normal.")

    with t5:
        st.subheader("🔎 Propositional Resolution")
        st.write("Query:", bgoal)
        st.write("Entailed by encoded facts/rules:", rresult)
        for line in rtrace:
            st.write("- " + line)
        st.caption("Resolution only tests the encoded rules; it cannot diagnose or exclude disease.")

    st.divider()
    st.subheader("🤖 Optional Gemini explanation")
    if st.button("Generate AI explanation"):
        try:
            with st.spinner("Generating explanation..."):
                ai = gemini_explanation(rows, derived)
            st.session_state["medical_ai_explanation"] = ai
        except Exception as exc:
            st.session_state["medical_ai_explanation"] = f"Gemini unavailable: {exc}"
    st.write(st.session_state.get(
        "medical_ai_explanation", "Not generated. The analyzer works without Gemini."
    ))

    report = f"""AI MEDICAL REPORT ANALYZER
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DISCLAIMER:
Educational aid only, not a diagnosis or clinically validated tool.
Verify every extracted result and range against the original report.

EXTRACTED PDF TEXT:
{report_text}

EXTRACTED RESULTS:
{json.dumps(rows, indent=2, default=str)}

ROWS REQUIRING REVIEW:
{json.dumps(flagged, indent=2, default=str)}

INITIAL FACTS:
{json.dumps(sorted(facts), indent=2)}

FORWARD-CHAINED FACTS:
{json.dumps(sorted(derived), indent=2)}

BACKWARD CHAINING:
Goal: {bgoal}
Proved: {bresult}
Trace: {json.dumps(btrace, indent=2)}

RESOLUTION:
Result: {rresult}
Trace: {json.dumps(rtrace, indent=2)}

GEMINI EXPLANATION:
{st.session_state.get('medical_ai_explanation', 'Not generated')}
"""
    st.download_button(
        "📥 Download full report (.txt)", report,
        file_name="AI_Medical_Report.txt", mime="text/plain",
        use_container_width=True
    )

st.divider()
st.info(
    "Known limitations: scanned-PDF OCR is not implemented; parsing is based on "
    "text lines and may misassociate values/ranges in complex tables; the "
    "Decision Tree uses synthetic labels; K-Means is exploratory. None of these "
    "algorithms guarantees medical accuracy."
)
