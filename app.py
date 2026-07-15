import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

# Load model
model = joblib.load("model.pkl")

# Header
st.markdown("""
# 🎓 Student Academic Performance Prediction System
### Machine Learning Based Student Performance Analytics Dashboard
""")

# Sidebar
st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Graphs", "Prediction"]
)

# ================= DASHBOARD =================
if page == "Dashboard":

    col1, col2, col3 = st.columns(3)

    col1.metric("Best Model", "Random Forest")
    col2.metric("R² Score", "0.805")
    col3.metric("Dataset Size", "395 Students")

    st.markdown("---")

    st.subheader("📊 Model Comparison")
    st.image("eval_model_comparison.png")

    st.subheader("🔥 Feature Importance")
    st.image("eval_feature_importance.png")

# ================= GRAPHS =================
elif page == "Graphs":

    st.subheader("📈 Correlation Heatmap")
    st.image("eda_correlation_heatmap.png")

    st.subheader("📉 Attendance vs Grade")
    st.image("eda_attendance_vs_grade.png")

    st.subheader("🎯 Predicted vs Actual")
    st.image("eval_predicted_vs_actual.png")

    st.subheader("📋 Residual Analysis")
    st.image("eval_residuals.png")

# ================= PREDICTION =================
elif page == "Prediction":

    st.subheader("🎯 Student Performance Prediction")

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

        st.success(f"🎯 Predicted Final Grade (G3): {prediction:.2f}/20")

st.markdown("---")
st.caption("Developed by Priyanshu | Student Academic Performance Prediction System")
