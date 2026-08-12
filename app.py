import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Smartphone Addiction Predictor",
    page_icon="📱",
    layout="centered",
)

MODEL_PATH = r"F:\Smartphone_Addiction\Model\smartphone_addiction.joblib"

ADDICTED_GIF = "https://media.giphy.com/media/GOFg40jTXz47Nz7yTH/giphy.gif"
NOT_ADDICTED_GIF = "https://media.giphy.com/media/u6FP12KqNBydVRUj8G/giphy.gif"

st.title("📱 Smartphone Addiction Predictor")
st.caption("Answer a few quick questions and get your addiction risk score.")

st.divider()

model = joblib.load(MODEL_PATH)

with st.form("addiction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=80, value=22, step=1)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.subheader("Screen time")
    col3, col4 = st.columns(2)
    with col3:
        daily_screen_time_hours = st.slider("Daily screen time (hrs)", 0.0, 16.0, 5.0, 0.5)
    with col4:
        weekend_screen_time = st.slider("Weekend screen time (hrs)", 0.0, 16.0, 6.0, 0.5)

    col5, col6, col7 = st.columns(3)
    with col5:
        social_media_hours = st.slider("Social media (hrs)", 0.0, 12.0, 2.0, 0.5)
    with col6:
        gaming_hours = st.slider("Gaming (hrs)", 0.0, 12.0, 1.0, 0.5)
    with col7:
        work_study_hours = st.slider("Work/study on phone (hrs)", 0.0, 12.0, 1.0, 0.5)

    st.subheader("Habits & wellbeing")
    col8, col9 = st.columns(2)
    with col8:
        sleep_hours = st.slider("Sleep (hrs)", 0.0, 12.0, 7.0, 0.5)
    with col9:
        stress_level = st.slider("Stress level (1 = low, 10 = high)", 1, 10, 5)

    col10, col11 = st.columns(2)
    with col10:
        notifications_per_day = st.number_input("Notifications per day", min_value=0, max_value=500, value=60, step=5)
    with col11:
        app_opens_per_day = st.number_input("App opens per day", min_value=0, max_value=300, value=40, step=5)

    academic_work_impact = st.selectbox(
        "Has phone use affected your academic/work performance?",
        ["No", "Slightly", "Moderately", "Severely"],
    )

    predict_clicked = st.form_submit_button("Check my risk", use_container_width=True)

if predict_clicked:
    input_df = pd.DataFrame(
        [
            {
                "id": 1,
                "age": age,
                "daily_screen_time_hours": daily_screen_time_hours,
                "social_media_hours": social_media_hours,
                "gaming_hours": gaming_hours,
                "work_study_hours": work_study_hours,
                "sleep_hours": sleep_hours,
                "notifications_per_day": notifications_per_day,
                "app_opens_per_day": app_opens_per_day,
                "weekend_screen_time": weekend_screen_time,
                "gender": gender,
                "stress_level": stress_level,
                "academic_work_impact": academic_work_impact,
            }
        ]
    )

    with st.spinner("Analyzing your habits..."):
        proba = model.predict_proba(input_df)[0]
        addicted_prob = float(proba[-1])

    addicted_pct = round(addicted_prob * 100, 1)

    st.divider()
    st.subheader(f"There is a {addicted_pct}% chance you are addicted")

    if addicted_prob > 0.5:
        st.error("High risk — your phone habits suggest addiction.")
        st.image(ADDICTED_GIF, use_container_width=True)
    else:
        st.success("Low risk — your phone habits look healthy.")
        st.image(NOT_ADDICTED_GIF, use_container_width=True)