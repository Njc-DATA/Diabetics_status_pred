import streamlit as st
import numpy as np
import joblib

# ------------------------------------------------------------
# PAGE SETTINGS
# ------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
model = joblib.load("Packages/diabetes.pkl")

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("🩺 Diabetes Prediction System")
st.markdown(
    """
This machine learning application predicts the **likelihood of diabetes**  
based on patient medical inputs.

**Algorithm:** Support Vector Machine (SVM)  
**Dataset:** Pima Indians Diabetes Dataset  
"""
)

st.divider()

# ------------------------------------------------------------
# INPUT FORM (Best for Mobile)
# ------------------------------------------------------------
st.subheader("Enter Patient Details")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
        Glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

    with col2:
        Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=85)
        BMI = st.number_input("BMI", min_value=0.0, max_value=80.0, value=30.1)
        DiabetesPedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        Age = st.number_input("Age", min_value=1, max_value=120, value=29)

    submitted = st.form_submit_button("Predict Diabetes")

# ------------------------------------------------------------
# PREDICTION OUTPUT
# ------------------------------------------------------------
if submitted:
    inputs = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                        Insulin, BMI, DiabetesPedigree, Age]])

    prediction = model.predict(inputs)[0]
    probability = model.predict_proba(inputs)[0][1]

    st.divider()

    if prediction == 1:
        st.error(
            f"""
            ### 🔴 High Diabetes Risk  
            **Probability: {probability:.2f}**

            The model predicts a **high likelihood of diabetes**.  
            Please consult a medical professional for further evaluation.
            """
        )
    else:
        st.success(
            f"""
            ### 🟢 Low Diabetes Risk  
            **Probability: {probability:.2f}**

            The model predicts a **low likelihood of diabetes**.  
            Maintain healthy lifestyle choices for long-term wellness.
            """
        )

st.divider()
st.markdown("Built by **Data Scientist Ngama Jude Chinedu** ⚡")
