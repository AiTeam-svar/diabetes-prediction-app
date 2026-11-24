
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩸",
    layout="centered"
)

# Simple blue colour theme
st.markdown("""
<style>
.stApp {
    background-color: #F4F8FC;
}

h1, h2, h3 {
    color: #173B63;
}

div.stButton > button {
    background-color: #287DB5;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #174D7A;
    color: white;
}

div[data-testid="stNumberInput"] input {
    border-color: #B8D2E8;
}
</style>
""", unsafe_allow_html=True)


# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()


# Main heading - original style
st.title("🩸Diabetes Prediction App🧠🩺")

st.write(
    "Welcome to our AI-based Diabetes Prediction Project! "
    "Enter the following health measurements to get "
    "**Smart Blood Sugar & Diabetes Forecast** 📈🩸"
)

st.divider()

# Health measurements
st.subheader("📝 Enter Your Health Measurements")

pregnancies = st.number_input(
    "Pregnancies", min_value=0, max_value=20, value=1
)

glucose = st.number_input(
    "Glucose", min_value=0, max_value=300, value=120
)

blood_pressure = st.number_input(
    "Blood Pressure", min_value=0, max_value=200, value=70
)

skin_thickness = st.number_input(
    "Skin Thickness", min_value=0, max_value=100, value=20
)

insulin = st.number_input(
    "Insulin", min_value=0, max_value=900, value=80
)

bmi = st.number_input(
    "BMI", min_value=0.0, max_value=70.0,
    value=25.0, step=0.1
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0, max_value=3.0,
    value=0.5, step=0.01
)

age = st.number_input(
    "Age", min_value=1, max_value=120, value=30
)

st.divider()

# Prediction
if st.button("🔍 Predict Diabetes", type="primary"):

    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    # Handle zero measurements consistently
    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    input_data[zero_columns] = input_data[
        zero_columns
    ].replace(0, float("nan"))

    prediction = model.predict(input_data)

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.warning(
            "The model predicts: Diabetes (Outcome 1)"
        )
    else:
        st.success(
            "The model predicts: Non-Diabetes (Outcome 0)"
        )

    st.caption(
        "This prediction is for educational purposes only "
        "and is not a medical diagnosis."
    )

# Model performance - expandable section
st.divider()

with st.expander("📈 Model Performance"):

    st.write("**Algorithm:** Logistic Regression")

    st.metric("Model Accuracy", "74.46%")

    st.write("**Classification Report:**")

    st.code("""
                 Precision  Recall  F1-score  Support

Non-Diabetic       0.77     0.86      0.81      150
Diabetic           0.67     0.53      0.59       81

Accuracy                              0.74      231
Macro Average      0.72     0.70      0.70      231
Weighted Average   0.74     0.74      0.74      231
""")

# Disclaimer
st.info(
    "⚕️ This application is an educational project. "
    "It cannot diagnose diabetes or replace professional "
    "medical advice."
)

# Small school credit
st.markdown("""
<div style="text-align:center; color:#8495A8;
            font-size:11px; padding:12px;">
    Developed by students of<br>
    Indian School Bahrain
</div>
""", unsafe_allow_html=True)
