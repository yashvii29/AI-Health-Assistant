import streamlit as st

st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#00BFA6;
}

.sub-title{
    text-align:center;
    font-size:18px;
    color:gray;
}

.module-card{
    padding:15px;
    border-radius:12px;
    background-color:#f5f7fa;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown(
    '<p class="main-title">🩺 AI Health Assistant</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">AI Powered Healthcare & Wellness Platform</p>',
    unsafe_allow_html=True
)

st.divider()

# =========================
# SIDEBAR
# =========================

st.sidebar.title("🩺 Navigation")

st.sidebar.info(
    "Select any module from the sidebar."
)

# =========================
# DASHBOARD
# =========================

st.header("🏠 Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Modules", "12")

with col2:
    st.metric("AI Powered", "Yes")

with col3:
    st.metric("Version", "1.0")

st.write("")

st.subheader("Available Modules")

col1, col2 = st.columns(2)

with col1:

    st.success("🩺 Symptom Checker")

    st.success("⚖️ BMI Calculator")

    st.success("🔥 Calorie Calculator")

    st.success("🥗 Food Recommendation System")

    st.success("💧 Daily Water Intake Calculator")

    st.success("😴 Sleep Recommendation System")

with col2:

    st.success("📄 Medical Report Analyzer")

    st.success("📥 PDF Report Generator")

    st.success("💊 Medicine Information Lookup")

    st.success("🚨 Emergency Risk Detection")

    st.success("📈 Health History Tracker")

    st.success("🏠 Dashboard")

st.divider()

st.subheader("Project Overview")

st.write("""
This AI Health Assistant helps users analyze symptoms,
track wellness metrics, understand medical reports,
receive health recommendations, and generate health summaries.

Use the sidebar to navigate between modules.
""")