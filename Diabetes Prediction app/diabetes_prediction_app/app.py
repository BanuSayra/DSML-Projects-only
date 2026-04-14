import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Title
st.title("🩺 Diabetes Prediction App")

# Load dataset
df = pd.read_csv("diabetes.csv")

# Split data
X = df.drop(columns="Outcome", axis=1)
Y = df["Outcome"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Standardize
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model (linear kernel)
model = svm.SVC(kernel="linear")
model.fit(X_train, Y_train)

# Evaluate model
Y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)

st.sidebar.subheader("Model Accuracy")
st.sidebar.write(f"Accuracy: {acc*100:.2f}%")

# User inputs
st.header("Enter Patient Data")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
age = st.number_input("Age", min_value=0, max_value=120, value=25)

# Prediction button
if st.button("Predict"):
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
    input_data_reshaped = input_data.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = model.predict(std_data)

    if prediction[0] == 0:
        st.success("✅ The person is **Non-Diabetic**")
    else:
        st.error("⚠️ The person is **Diabetic**")
