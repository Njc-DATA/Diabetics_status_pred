import streamlit as st
import numpy as np
import joblib

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
model = joblib.load("Packages/diabetes.pkl")  # your classifier

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("🩺 Diabetes Prediction System")
st.markdown("""
This machine learning application predicts the likelihood of diabetes  
based on clinical input variables.

**Model:** Trained using a classification(SVM) algorithm  
**Dataset:** Pima Indians Diabetes Dataset  
""")

st.divider()

# ------------------------------------------------------------
# SIDEBAR (User Inputs)
# ------------------------------------------------------------
st.sidebar.header("User Input Features")

def user_inputs():
    Pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    Glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=120)
    BloodPressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    SkinThickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    Insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=85)
    BMI = st.sidebar.number_input("BMI", min_value=0.0, max_value=80.0, value=30.1)
    DiabetesPedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    Age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=29)

    data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigree, Age]])
    
    return data

inputs = user_inputs()

# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------
if st.button("Predict Diabetes"):
    prediction = model.predict(inputs)[0]
    probability = model.predict_proba(inputs)[0][1]

    if prediction == 1:
        st.error(f"🔴 **The person is diabetic: **{probability:.2f}**")
    else:
        st.success(f"🟢 **The person is not diabetic: **{probability:.2f}**")

st.divider()
st.markdown("Built by **Data Scientist Ngama Jude Chinedu** ⚡")