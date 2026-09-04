import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Employee Job Change Prediction",page_icon="🎓",layout="wide")

# Load ML Model
model = joblib.load("logistic_regression_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.write(type(model))

st.markdown("""
<style>

/* =========================
   MAIN COLORS
   ========================= */

[data-testid="stAppViewContainer"] {
    background-color: #0F172A;
}

[data-testid="stSidebar"] {
    background-color: #1E293B;
}

/* Main title */
h1 {
    color: #FFFFFF !important;
}

/* Section headings */
h2, h3 {
    color: #2DD4BF !important;
}

/* Normal text */
p {
    color: #F8FAFC;
}


/* =========================
   ANIMATION
   ========================= */

/* Main title appears first */
h1 {
    animation: fadeSlideUp 0.8s ease-out both;
    animation-delay: 0.1s;
}

/* Subtitle */
h2 {
    animation: fadeSlideUp 0.8s ease-out both;
    animation-delay: 0.4s;
}

/* Paragraphs after subtitle */
p {
    animation: fadeSlideUp 0.7s ease-out both;
    animation-delay: 0.7s;
}


/* Smooth appearance */
@keyframes fadeSlideUp {

    from {
        opacity: 0;
        transform: translateY(18px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }

}


/* =========================
   COLUMNS
   ========================= */

[data-testid="column"] {
    animation: fadeSlideUp 0.8s ease-out both;
}

[data-testid="column"]:nth-child(1) {
    animation-delay: 1.0s;
}

[data-testid="column"]:nth-child(2) {
    animation-delay: 1.2s;
}

[data-testid="column"]:nth-child(3) {
    animation-delay: 1.4s;
}


/* =========================
   DIVIDER
   ========================= */

hr {
    border-color: #334155 !important;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.title("📌 Navigation")

df = pd.read_csv("aug_train.csv")

# =========================
# Create Derived Columns
# =========================

if "experience_level" not in df.columns:

    def get_experience_level(x):

        if pd.isna(x):
            return "Unknown"

        if x == "<1" or x == "1":
            return "Entry (0-2)"

        elif x in ["2", "3", "4", "5"]:
            return "Junior (3-5)"

        elif x in ["6", "7", "8", "9", "10"]:
            return "Mid (6-10)"

        elif x in ["11", "12", "13", "14", "15"]:
            return "Senior (11-15)"

        else:
            return "Expert (16+)"

    df["experience_level"] = df["experience"].apply(
        get_experience_level
    )

page = st.sidebar.radio("Go to:",["🏠 Home","🎯 Employee Prediction","📊 Analytics & Graphs","🧠 Model Information"])

if page == "🏠 Home":

    st.title("🎓 Employee Job Change Prediction")
    st.subheader("AI-Powered Employee Analytics")

    st.write(
        "A Machine Learning application designed to predict whether an employee "
        "may be interested in changing their current job."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎯 Prediction")
        st.write(
            "Identify employees who may be more likely to look for a new job."
        )

    with col2:
        st.markdown("### 📊 HR Analytics")
        st.write(
            "Explore employee factors that may influence job-change decisions."
        )

    with col3:
        st.markdown("### 🤖 Machine Learning")
        st.write(
            "Use a trained classification model to generate prediction results."
        )

    st.markdown("---")

    st.markdown("## 📌 About the Project")

    st.write(
        "This project applies Machine Learning techniques to HR Analytics. "
        "It uses employee information such as experience, education, company "
        "characteristics, training hours, and city development to predict "
        "the possibility of a job change."
    )

    st.markdown("## ⚙️ How It Works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("### 01 | Employee Data")
        st.write(
            "Enter the employee's professional and educational information."
        )

    with step2:
        st.markdown("### 02 | ML Processing")
        st.write(
            "The trained model processes the information using the same "
            "preprocessing pipeline used during training."
        )

    with step3:
        st.markdown("### 03 | Prediction")
        st.write(
            "The application provides a prediction together with its probability."
        )

    st.markdown("---")

    st.markdown("## 🎯 Project Goal")

    st.write(
        "The main goal is to demonstrate how Machine Learning can support "
        "HR decision-making by identifying employees who may have a higher "
        "probability of seeking a new job."
    )

    st.info(
        "💡 Start with **Employee Prediction** to enter employee information "
        "and generate a personalized prediction."
    )


elif page == "🎯 Employee Prediction":

    st.subheader("👤 Employee Information")

    st.write("Enter the employee details below to generate a prediction.")

    col1, col2 = st.columns(2)

    with col1:

        city = st.text_input(
            "City",
            value="city_21"
        )

        city_development_index = st.number_input(
            "City Development Index",
            min_value=0.448,
            max_value=0.949,
            value=0.700,
            step=0.001,
            format="%.3f"
        )

        enrolled_university = st.selectbox(
            "Enrolled University",
            [
                "no_enrollment",
                "Full time course",
                "Part time course",
                "Unknown"
            ]
        )

        gender = st.selectbox(
            "Gender",
            [
                "Unknown",
                "Male",
                "Female",
                "Other"
            ]
        )

        relevant_experience = st.selectbox(
            "Relevant Experience",
            [
                "Has relevant experience",
                "No relevant experience"
            ]
        )

        education_level = st.selectbox(
            "Education Level",
            [
                "Graduate",
                "Primary School",
                "Masters",
                "High School",
                "Unknown",
                "Phd"
            ]
        )

    with col2:

        major_discipline = st.selectbox(
            "Major Discipline",
            [
                "STEM",
                "Unknown",
                "Humanities",
                "Other",
                "Arts",
                "Business Degree",
                "No Major"
            ]
        )

        experience = st.selectbox(
            "Experience Level",
            [
                "Entry (0-2)",
                "Junior (3-5)",
                "Mid (6-10)",
                "Senior (11-15)",
                "Expert (16+)"
            ]
        )

        company_size = st.selectbox(
            "Company Size",
            [
                "Unknown",
                "<10",
                "10/49",
                "50-99",
                "100-500",
                "500-999",
                "1000-4999",
                "5000-9999",
                "10000+"
            ]
        )

        company_type = st.selectbox(
            "Company Type",
            [
                "Pvt Ltd",
                "Unknown",
                "NGO",
                "Funded Startup",
                "Public Sector",
                "Early Stage Startup",
                "Other"
            ]
        )

        last_new_job = st.selectbox(
            "Last New Job",
            [
                "never",
                "1",
                "2",
                "3",
                "4",
                ">4",
                "Unknown"
            ]
        )

        training_hours = st.number_input(
            "Training Hours",
            min_value=1,
            max_value=336,
            value=50,
            step=1
        )

    experience_map = {
        "Entry (0-2)": 1,
        "Junior (3-5)": 4,
        "Mid (6-10)": 8,
        "Senior (11-15)": 13,
        "Expert (16+)": 16
    }

    experience_numeric = experience_map[experience]

    experience_level = experience

    if city_development_index <= 0.6:
        city_dev_tier = "Low"

    elif city_development_index <= 0.75:
        city_dev_tier = "Medium"

    elif city_development_index <= 0.9:
        city_dev_tier = "High"

    else:
        city_dev_tier = "Very High"

    st.markdown(
        """
        <style>

        div.stButton > button {
            width: 280px;
            height: 55px;
            font-size: 18px;
            font-weight: bold;
            background-color: #20C7B5;
            color: white;
            border: none;
            border-radius: 10px;
        }

        div.stButton > button:hover {
            background-color: #19AFA0;
            color: white;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

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
            st.error(
                f"🔴 Employee is likely to look for a new job "
                f"({probability:.1%})"
            )

        else:
            st.success(
                f"🟢 Employee is unlikely to look for a new job "
                f"({1 - probability:.1%})"
            )

elif page == "📊 Analytics & Graphs":

    st.title("📊 Analytics & Graphs")

    st.write(
        "Explore employee characteristics and factors related to job change."
    )

    # =====================================================
    # COLOR PALETTE
    # =====================================================

    colors = [
        "#EF4444",  # Red
        "#2563EB",  # Dark Blue
        "#3B82F6",  # Blue
        "#4F7FF0",  # Blue
        "#5B6FF0",  # Light Blue
        "#60A5FA",  # Light Blue
        "#93C5FD"   # Very Light Blue
    ]

    # =====================================================
    # KPI CALCULATIONS
    # =====================================================

    total_employees = len(df)

    employees_looking = int(df["target"].sum())

    job_change_rate = (
        employees_looking / total_employees * 100
        if total_employees > 0 else 0
    )

    avg_training_hours = df["training_hours"].mean()

    # =====================================================
    # KPI CARDS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 Total Employees",
            f"{total_employees:,}"
        )

    with col2:
        st.metric(
            "🔄 Looking for New Job",
            f"{employees_looking:,}"
        )

    with col3:
        st.metric(
            "📈 Job Change Rate",
            f"{job_change_rate:.1f}%"
        )

    with col4:
        st.metric(
            "⏱️ Avg. Training Hours",
            f"{avg_training_hours:.1f}"
        )

    st.markdown("---")

    # =====================================================
    # CHART 1 - JOB CHANGE DISTRIBUTION
    # =====================================================

    st.subheader("🎯 Job Change Distribution")

    job_change = (
        df["target"]
        .value_counts()
        .rename(
            index={
                0: "Not Looking",
                1: "Looking for New Job"
            }
        )
        .reset_index()
    )

    job_change.columns = ["Status", "Employees"]

    fig1 = px.bar(
        job_change,
        x="Status",
        y="Employees",
        color="Status",
        text="Employees",
        color_discrete_sequence=[
            "#2563EB",
            "#EF4444"
        ]
    )

    fig1.update_layout(
        height=450,
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        xaxis=dict(
            title="Job Change Status",
            gridcolor="#1E293B"
        ),
        yaxis=dict(
            title="Number of Employees",
            gridcolor="#1E293B"
        ),
        showlegend=False
    )

    fig1.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================================
    # CHART 2 - JOB CHANGE BY EDUCATION
    # =====================================================

    st.subheader("🎓 Job Change by Education Level")

    education_analysis = (
        pd.crosstab(
            df["education_level"],
            df["target"]
        )
        .rename(
            columns={
                0: "Not Looking",
                1: "Looking for New Job"
            }
        )
        .reset_index()
    )

    education_long = education_analysis.melt(
        id_vars="education_level",
        var_name="Status",
        value_name="Employees"
    )

    fig2 = px.bar(
        education_long,
        x="education_level",
        y="Employees",
        color="Status",
        barmode="group",
        text="Employees",
        color_discrete_map={
            "Not Looking": "#2563EB",
            "Looking for New Job": "#EF4444"
        }
    )

    fig2.update_layout(
        height=500,
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        xaxis=dict(
            title="Education Level",
            gridcolor="#1E293B"
        ),
        yaxis=dict(
            title="Number of Employees",
            gridcolor="#1E293B"
        ),
        legend=dict(
            title=""
        )
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================================
    # CHART 3 - JOB CHANGE BY EXPERIENCE
    # =====================================================

    # =========================
    # CHART 3
    # =========================

    st.subheader("💼 Job Change by Experience Level")

    # Create a copy for analysis
    experience_data = df.copy()

    # Handle missing experience values
    experience_data["experience_level"] = (
        experience_data["experience_level"]
        .fillna("Not Specified")
        .astype(str)
    )

    # Count employees by experience level and job-change status
    experience_analysis = pd.crosstab(
        experience_data["experience_level"],
        experience_data["target"]
    )

    # Rename target columns
    experience_analysis = experience_analysis.rename(
        columns={
            0: "Not Looking",
            1: "Looking for New Job"
        }
    )

    # Make sure both columns always exist
    if "Not Looking" not in experience_analysis.columns:
        experience_analysis["Not Looking"] = 0

    if "Looking for New Job" not in experience_analysis.columns:
        experience_analysis["Looking for New Job"] = 0

    # Sort experience levels naturally
    experience_order = [
        "Entry (0-2)",
        "Junior (3-5)",
        "Mid (6-10)",
        "Senior (11-15)",
        "Expert (16+)"
    ]

    existing_levels = [
        level for level in experience_order
        if level in experience_analysis.index
    ]

    other_levels = [
        level for level in experience_analysis.index
        if level not in existing_levels
    ]

    final_order = existing_levels + sorted(other_levels)

    experience_analysis = experience_analysis.reindex(
        final_order
    )

    # Create chart
    fig3 = px.bar(
        experience_analysis,
        barmode="group",
        labels={
            "value": "Number of Employees",
            "experience_level": "Experience Level",
            "variable": "Job Change Status"
        },
        color_discrete_map={
            "Not Looking": "#2563EB",
            "Looking for New Job": "#EF4444"
        },
        text_auto=True
    )

    fig3.update_layout(
        height=550,
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(
            color="#F8FAFC"
        ),
        xaxis=dict(
            title="Experience Level",
            gridcolor="#1E293B"
        ),
        yaxis=dict(
            title="Number of Employees",
            gridcolor="#1E293B"
        ),
        legend=dict(
            title="",
            orientation="v"
        )
    )

    fig3.update_traces(
        textposition="inside"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.markdown("---")

    # =========================
# CHART 4
# =========================

    st.subheader("⏱️ Average Training Hours by Job Change")

    training_analysis = (
        df.groupby("target")["training_hours"]
        .mean()
        .rename(
            index={
                0: "Not Looking",
                1: "Looking for New Job"
            }
        )
        .reset_index()
    )

    training_analysis.columns = [
        "Job Change Status",
        "Average Training Hours"
    ]

    fig4 = px.bar(
        training_analysis,
        x="Job Change Status",
        y="Average Training Hours",
        color="Job Change Status",
        text="Average Training Hours",
        color_discrete_map={
            "Not Looking": "#2563EB",
            "Looking for New Job": "#EF4444"
        }
    )

    fig4.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside"
    )

    fig4.update_layout(
        height=450,
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        xaxis=dict(
            title="Job Change Status",
            gridcolor="#1E293B"
        ),
        yaxis=dict(
            title="Average Training Hours",
            gridcolor="#1E293B"
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    st.markdown("---")

elif page == "🧠 Model Information":

    st.title("🧠 Model Information")

    st.write(
        "This section provides an overview of the Machine Learning models "
        "used to predict whether an employee is likely to look for a new job."
    )

    st.markdown("---")

    # =========================
    # MODEL COMPARISON
    # =========================

    st.subheader("📊 Model Performance Comparison")

    model_results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "KNN"
        ],
        "Accuracy": [
            77.3,
            72.3,
            78.7,
            59.2
        ]
    })

    fig = px.bar(
        model_results,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        color="Model",
        color_discrete_map={
            "Logistic Regression": "#2563EB",
            "Decision Tree": "#2563EB",
            "Random Forest": "#EF4444",
            "KNN": "#2563EB"
        },
        labels={
            "Model": "Machine Learning Model",
            "Accuracy": "Accuracy (%)"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        height=500,
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#F8FAFC"),
        xaxis=dict(
            gridcolor="#1E293B"
        ),
        yaxis=dict(
            gridcolor="#1E293B",
            range=[0, 100]
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # =========================
    # BEST MODEL
    # =========================

    st.subheader("🏆 Best Performing Model")

    st.success(
        "Logistics Regression achieved the highest accuracy among the tested models "
        "with an F1 score of 62%."
    )

    st.write(
    "Logistics Regression was selected because it achieved the highest F1 Score of 62%"
    "among the evaluated Machine Learning models."
    )

    st.markdown("---")

    # =========================
    # MODEL DETAILS
    # =========================

    st.subheader("🧠 Models Used")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📈 Logistic Regression")

        st.write(
            "A classification model used as a baseline to predict whether "
            "an employee is likely to change jobs."
        )

        st.metric(
            "Accuracy",
            "77.3%"
        )

        st.markdown("### 🌳 Decision Tree")

        st.write(
            "A tree-based model that makes predictions using a sequence "
            "of decision rules based on employee characteristics."
        )

        st.metric(
            "Accuracy",
            "72.3%"
        )

    with col2:

        st.markdown("### 🌲 Random Forest")

        st.write(
            "An ensemble model that combines multiple decision trees "
            "to improve prediction performance."
        )

        st.metric(
            "Accuracy",
            "78.7%"
        )

        st.markdown("### 👥 KNN")

        st.write(
            "A classification algorithm that predicts an employee's "
            "job-change status based on similar observations."
        )

        st.metric(
            "Accuracy",
            "59.2%"
        )

    st.markdown("---")

    # =========================
    # HR INTERPRETATION
    # =========================

    st.subheader("💡 HR Interpretation")

    st.info(
        "The model comparison helps HR identify the algorithm that provides "
        "the strongest predictive performance. In this project, Random Forest "
        "achieved the highest accuracy and was therefore selected as the "
        "best-performing model among the tested algorithms."
    )


