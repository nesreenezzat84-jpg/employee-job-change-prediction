import streamlit as st
import pandas as pd
import joblib

model = joblib.load("logistic_regression_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("🎓 Employee Job Change Prediction")

st.write("Predict whether an employee is likely to look for a new job.")

st.success("Model and preprocessor loaded successfully!")

st.subheader("Employee Information")

city = st.text_input("City", value="city_21")

city_development_index = st.number_input("City Development Index",min_value=0.448,max_value=0.949,value=0.700,step=0.001,format="%.3f")

enrolled_university = st.selectbox("Enrolled University",["no_enrollment","Full time course","Part time course","Unknown"])

gender = st.selectbox("Gender",["Unknown", "Male", "Female", "Other"])

relevant_experience = st.selectbox("Relevant Experience",["Has relevant experience", "No relevant experience"])

education_level = st.selectbox("Education Level",["Graduate", "Primary School", "Masters", "High School", "Unknown", "Phd"])

major_discipline = st.selectbox("Major Discipline", ["STEM", "Unknown", "Humanities", "Other", "Arts", "Business Degree", "No Major"])

experience = st.selectbox("Experience Level",["Entry (0-2)", "Junior (3-5)", "Mid (6-10)", "Senior (11-15)", "Expert (16+)"])

experience_map = {"Entry (0-2)": 1,"Junior (3-5)": 4,"Mid (6-10)": 8,"Senior (11-15)": 13,"Expert (16+)": 16}

experience_numeric = experience_map[experience]

company_size = st.selectbox("Company Size",["Unknown","<10","10/49","50-99","100-500","500-999","1000-4999","5000-9999","10000+"])

company_type = st.selectbox("Company Type",["Pvt Ltd","Unknown","NGO","Funded Startup","Public Sector","Early Stage Startup","Other"])

last_new_job = st.selectbox("Last New Job",["never","1","2","3","4",">4","Unknown"])

training_hours = st.number_input("Training Hours",min_value=1,max_value=336,value=50,step=1)

# Create derived features
if experience_numeric <= 2:
    experience_level = "Entry (0-2)"
elif experience_numeric <= 5:
    experience_level = "Junior (3-5)"
elif experience_numeric <= 10:
    experience_level = "Mid (6-10)"
elif experience_numeric <= 15:
    experience_level = "Senior (11-15)"
else:
    experience_level = "Expert (16+)"

if city_development_index <= 0.6:
    city_dev_tier = "Low"
elif city_development_index <= 0.75:
    city_dev_tier = "Medium"
elif city_development_index <= 0.9:
    city_dev_tier = "High"
else:
    city_dev_tier = "Very High"

if st.button("Predict Job Change"):

    input_data = pd.DataFrame({
        "city": [city],
        "city_development_index": [city_development_index],
        "enrolled_university": [enrolled_university],
        "gender": [gender],
        "relevent_experience": [relevant_experience],
        "education_level": [education_level],
        "major_discipline": [major_discipline],
        "experience": [experience_numeric],
        "experience_level": [experience_level],
        "city_dev_tier": [city_dev_tier],
        "company_size": [company_size],
        "company_type": [company_type],
        "last_new_job": [last_new_job],
        "training_hours": [training_hours]
    })

    input_transformed = preprocessor.transform(input_data)
    prediction = model.predict(input_transformed)[0]
    probability = model.predict_proba(input_transformed)[0][1]

    if prediction == 1:
        st.error(f"🔴 Employee is likely to look for a new job ({probability:.1%})")
    else:
        st.success(f"🟢 Employee is unlikely to look for a new job ({1-probability:.1%})")