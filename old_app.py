import streamlit as st
import google.generativeai as genai

GEMINI_API_KEY = "AQ.Ab8RN6LDq8jPgJwvOqXjtYwkyiWbDgzO1YhoMA-uwQXCQ-85sQ"

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-3.6-flash")

st.set_page_config(page_title="AI Health Assistant")

st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Health Assistant")

# Session States
if "questions" not in st.session_state:
    st.session_state.questions = ""

if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False

if "answers_ready" not in st.session_state:
    st.session_state.answers_ready = False

if "final_ready" not in st.session_state:
    st.session_state.final_ready = False

if "final_assessment" not in st.session_state:
    st.session_state.final_assessment = ""

# User Inputs
name = st.text_input("Name")

age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    step=1
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

symptoms = st.text_area(
    "Describe your symptoms"
)

# STEP 1 - Analyze Symptoms
if not st.session_state.analysis_ready:

    if st.button("Analyze Symptoms"):

        prompt = f"""
You are a medical intake assistant.

Patient Name: {name}
Age: {age}
Gender: {gender}
Symptoms: {symptoms}

Your task:

Ask 5 important follow-up questions that would help a doctor better understand the condition.

Do NOT provide diagnosis.
Do NOT provide risk level.
Only ask questions.

Number them clearly.
"""

        response = model.generate_content(prompt)

        st.session_state.questions = response.text
        st.session_state.analysis_ready = True

        st.rerun()

# STEP 2 - Show Questions
if st.session_state.analysis_ready and not st.session_state.final_ready:

    st.write("## Follow-Up Questions")
    st.write(st.session_state.questions)

    followup_answers = st.text_area(
        "Answer the follow-up questions here"
    )

    if followup_answers.strip() != "":

        if st.button("Generate Final Assessment"):

            final_prompt = f"""
You are an AI health assistant.

Patient Name: {name}
Age: {age}
Gender: {gender}

Initial Symptoms:
{symptoms}

Follow-up Answers:
{followup_answers}

Based on all available information provide:

1. Possible causes
2. Risk level (Low/Medium/High)
3. Recommended specialist
4. General advice

Clearly mention that this is not a medical diagnosis.
"""

            final_response = model.generate_content(final_prompt)

            st.session_state.final_assessment = final_response.text
            st.session_state.final_ready = True

            st.rerun()

# STEP 4 - Show Final Assessment
if st.session_state.final_ready:

    st.write("## Final Health Assessment")
    st.write(st.session_state.final_assessment)