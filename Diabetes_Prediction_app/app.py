import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Diabetes Predictor", layout="centered")

# ---------------- CUSTOM CSS (Compact UI) ----------------
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
h1, h2, h3 {margin-bottom: 0.3rem;}
div[data-testid="stNumberInput"] {margin-bottom: -10px;}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🩺 Diabetes Predictor")
st.caption("⚡ Fast ML Prediction | Clean UI")

# ---------------- WARNING ----------------
st.warning("⚠️ Model trained on **female patients only**. Results may not be valid for males.")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()

# ---------------- MODEL TRAINING ----------------
X = df.drop("Outcome", axis=1)
Y = df["Outcome"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = svm.SVC(kernel="linear")
model.fit(X_train, Y_train)

accuracy = accuracy_score(Y_test, model.predict(X_test))

# ---------------- SIDEBAR ----------------
st.sidebar.metric("Model Accuracy", f"{accuracy*100:.2f}%")

# ---------------- INPUT SECTION ----------------
st.subheader("Patient Inputs")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 79)
    age = st.number_input("Age", 0, 120, 25)
    dpf = st.number_input("DPF", 0.0, 2.5, 0.5, format="%.3f")

# ---------------- BMI CALCULATOR ----------------
st.markdown("### BMI")

use_calc = st.checkbox("Calculate BMI")

if use_calc:
    c1, c2 = st.columns(2)
    with c1:
        height = st.number_input("Height (cm)", 100, 250, 170)
    with c2:
        weight = st.number_input("Weight (kg)", 20, 200, 65)

    bmi = weight / ((height / 100) ** 2)
    st.success(f"BMI: {bmi:.2f}")
else:
    bmi = st.number_input("BMI", 0.0, 70.0, 32.0)

# ---------------- PREDICTION ----------------
st.markdown("---")

if st.button("🔍 Predict", use_container_width=True):
    input_data = np.array([
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, dpf, age
    ])

    input_scaled = scaler.transform(input_data.reshape(1, -1))
    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        st.success("✅ Non-Diabetic")
    else:
        st.error("⚠️ Diabetic")

# ---------------- FOOTER ----------------
st.caption("Built with Streamlit | ML Model: SVM (Linear Kernel)")
