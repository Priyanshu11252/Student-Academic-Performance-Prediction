import streamlit as st
import joblib
import pandas as pd

st.title("Student Academic Performance Prediction System")

model = joblib.load("model.pkl")

attendance_rate = st.slider("Attendance Rate", 0.0, 1.0, 0.80)
avg_internal_score = st.slider("Average Internal Score", 0.0, 20.0, 10.0)
score_trend = st.slider("Score Trend", -10.0, 10.0, 0.0)
score_consistency = st.slider("Score Consistency", 0.0, 10.0, 1.0)

study_effort_index = st.slider("Study Effort Index", 0.0, 5.0, 2.0)
lifestyle_risk_index = st.slider("Lifestyle Risk Index", -5.0, 10.0, 2.0)
parent_edu_avg = st.slider("Parent Education Average", 0.0, 4.0, 2.0)
support_score = st.slider("Support Score", 0, 3, 1)

age = st.slider("Age", 15, 22, 17)
traveltime = st.slider("Travel Time", 1, 4, 1)
studytime = st.slider("Study Time", 1, 4, 2)
failures = st.slider("Failures", 0, 3, 0)

famrel = st.slider("Family Relationship", 1, 5, 4)
freetime = st.slider("Free Time", 1, 5, 3)
goout = st.slider("Go Out", 1, 5, 3)
Dalc = st.slider("Workday Alcohol", 1, 5, 1)
Walc = st.slider("Weekend Alcohol", 1, 5, 2)
health = st.slider("Health", 1, 5, 3)
absences = st.slider("Absences", 0, 30, 4)

sex = st.selectbox("Sex", ["F", "M"])
address = st.selectbox("Address", ["U", "R"])
Mjob = st.selectbox("Mother Job", ["teacher", "health", "services", "at_home", "other"])
Fjob = st.selectbox("Father Job", ["teacher", "health", "services", "at_home", "other"])
reason = st.selectbox("Reason", ["course", "home", "reputation", "other"])
guardian = st.selectbox("Guardian", ["mother", "father", "other"])
internet = st.selectbox("Internet", ["yes", "no"])
higher = st.selectbox("Higher Education", ["yes", "no"])
romantic = st.selectbox("Romantic Relationship", ["yes", "no"])

if st.button("Predict Final Grade"):

    data = pd.DataFrame([{
        "attendance_rate": attendance_rate,
        "avg_internal_score": avg_internal_score,
        "score_trend": score_trend,
        "score_consistency": score_consistency,
        "study_effort_index": study_effort_index,
        "lifestyle_risk_index": lifestyle_risk_index,
        "parent_edu_avg": parent_edu_avg,
        "support_score": support_score,
        "age": age,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "sex": sex,
        "address": address,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "internet": internet,
        "higher": higher,
        "romantic": romantic
    }])

    prediction = model.predict(data)[0]

    st.success(f"Predicted Final Grade (G3): {prediction:.2f}/20")
