import streamlit as st
import joblib
import pandas as pd

st.title("Student Academic Performance Prediction System")

model = joblib.load("model.pkl")

attendance_rate = st.slider("Attendance Rate", 0.0, 1.0, 0.8)
avg_internal_score = st.slider("Average Internal Score", 0.0, 20.0, 10.0)
score_trend = st.slider("Score Trend", -10.0, 10.0, 0.0)
study_effort_index = st.slider("Study Effort Index", 0.0, 5.0, 2.0)

if st.button("Predict"):
    st.warning("Model loaded successfully!")
