import streamlit as st
import pandas as pd
import joblib
import time

st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .title { color: #4CAF50; text-align: center; font-size: 40px; font-weight: bold; }
    .footer { text-align: center; color: grey; font-size: 12px; margin-top: 50px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>💰 Employee Salary Predictor 💰</div>", unsafe_allow_html=True)

with st.spinner('Loading model...'):
    model = joblib.load('models/salary_model.pkl')
    time.sleep(1)

st.success("Model Loaded Successfully!")

experience = st.slider("Years of Experience", 0, 30, 2)
education = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
job_title = st.selectbox("Job Title", ["Software Engineer", "Senior Software Engineer", "Tech Lead", "Manager"])
location = st.selectbox("Location", ["Hyderabad", "Bangalore", "Pune"])

if st.button("Predict Salary"):
    with st.spinner('Predicting Salary...'):
        time.sleep(1)
        input_df = pd.DataFrame([[experience, education, job_title, location]],
                                columns=['Experience', 'Education_Level', 'Job_Title', 'Location'])
        salary_pred = model.predict(input_df)[0]
        st.balloons()
        st.success(f"🎉 Estimated Salary: ₹{int(salary_pred):,}")

st.markdown("<div class='footer'>Built with ❤️ by  Sai Meghana - IBM SkillsBuild Learner</div>", unsafe_allow_html=True)

