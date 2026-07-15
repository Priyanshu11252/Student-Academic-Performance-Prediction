import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

st.title("Student Academic Performance Prediction System")
st.write("Enter student details below to predict the final grade (0-20 scale).")

model_data = joblib.load("model.pkl")
model = model_data["pipeline"]
FEATURES = model_data["features"]

avg_internal_score = st.slider("Average Internal Score (G1 & G2 avg)", 0.0, 20.0, 10.0)
score_trend = st.slider("Score Trend (G2 - G1)", -10.0, 10.0, 0.0)
attendance_rate = st.slider("Attendance Rate", 0.0, 1.0, 0.8)
absences = st.slider("Number of Absences", 0, 40, 4)
studytime = st.selectbox("Weekly Study Time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)", [1, 2, 3, 4], index=1)
failures = st.selectbox("Past Class Failures", [0, 1, 2, 3], index=0)
goout = st.slider("Going Out (1=low, 5=high)", 1, 5, 3)
health = st.slider("Health Status (1=bad, 5=good)", 1, 5, 4)
dalc = st.slider("Workday Alcohol Use (1=low, 5=high)", 1, 5, 1)
walc = st.slider("Weekend Alcohol Use (1=low, 5=high)", 1, 5, 2)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "avg_internal_score": avg_internal_score,
        "score_trend": score_trend,
        "attendance_rate": attendance_rate,
        "absences": absences,
        "studytime": studytime,
        "failures": failures,
        "goout": goout,
        "health": health,
        "Dalc": dalc,
        "Walc": walc,
    }])[FEATURES]

    prediction = model.predict(input_df)[0]
    prediction = float(np.clip(prediction, 0, 20))

    st.success(f"Predicted Final Grade: {prediction:.1f} / 20")
    st.progress(min(prediction / 20, 1.0))
