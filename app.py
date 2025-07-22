import streamlit as st
import pandas as pd
import joblib
import time
import requests
import matplotlib.pyplot as plt
import seaborn as sns

from streamlit_lottie import st_lottie

# ------------------- LOTTIE ANIMATION LOADER ---------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_welcome = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_4kx2q32n.json")
lottie_chart = load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_editor_aozayv5o.json")

# ------------------- PAGE CONFIG ---------------------
st.set_page_config(page_title="Salary Prediction", page_icon="💼", layout="centered")

# Load model
model = joblib.load('models/salary_model.pkl')

# Load dataset for EDA charts
data = pd.read_csv('data/employee_data.csv')

# Clean columns
data.columns = data.columns.str.strip()

# ------------------- HEADER ---------------------
st_lottie(lottie_welcome, height=250, key="welcome")
st.title("💼 Employee Salary Prediction")

st.markdown("---")
st.subheader("📊 Enter Employee Details")

with st.form("salary_form"):
    experience = st.slider("Experience (Years)", 0, 20, 2)
    education = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
    job_title = st.selectbox("Job Title", ["Software Engineer", "Senior Software Engineer", "Tech Lead", "Manager"])
    location = st.selectbox("Location", ["Hyderabad", "Bangalore", "Pune", "Chennai"])
    submitted = st.form_submit_button("Predict Salary 💰")

if submitted:
    st.info("Predicting Salary... Please wait")
    time.sleep(1)

    input_df = pd.DataFrame([[experience, education, job_title, location]], columns=["Experience", "Education_Level", "Job_Title", "Location"])
    predicted_salary = model.predict(input_df)[0]
    time.sleep(0.5)

    st.success(f"🎉 Predicted Annual Salary: ₹ {int(predicted_salary):,}")
    st.balloons()

st.markdown("---")

# ------------------- EDA SECTION ---------------------
st.subheader("🔎 Exploratory Data Analysis")
st_lottie(lottie_chart, height=200, key="eda")

st.markdown("### 📄 Sample Data")
st.dataframe(data.head())

# Salary Distribution Plot
st.markdown("### 💰 Salary Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(data['Salary'], kde=True, color='skyblue', ax=ax1)
ax1.set_title('Salary Distribution')
st.pyplot(fig1)

# Experience vs Salary
st.markdown("### 🧑‍💻 Experience vs Salary")
fig2, ax2 = plt.subplots()
sns.scatterplot(x='Experience', y='Salary', data=data, hue='Education_Level', ax=ax2)
ax2.set_title('Experience vs Salary by Education Level')
st.pyplot(fig2)

# Salary by Job Title
st.markdown("### 🏷️ Salary by Job Title")
fig3, ax3 = plt.subplots()
sns.barplot(x='Job_Title', y='Salary', data=data, ax=ax3)
ax3.set_title('Average Salary by Job Title')
plt.xticks(rotation=45)
st.pyplot(fig3)

# Salary by Location
st.markdown("### 📍 Salary by Location")
fig4, ax4 = plt.subplots()
sns.boxplot(x='Location', y='Salary', data=data, ax=ax4)
ax4.set_title('Salary by Location')
st.pyplot(fig4)

st.markdown("<div class='footer'>Built with ❤️ by  Sai Meghana - IBM SkillsBuild Learner</div>", unsafe_allow_html=True)

