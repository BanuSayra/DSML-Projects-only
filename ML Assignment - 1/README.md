
# 🚕 Sayra's RideEase Price Prediction App

🔗 **Live App:** https://dsml-projects-only-mlassignment1-rideeasedataset.streamlit.app/

---

## 📌 Project Overview

This project is a **Machine Learning-based Ride Price Prediction System** built using **Python, Scikit-learn, and Streamlit**.

It predicts the estimated ride price based on various factors such as distance, duration, weather, location, and driver rating.

---

## 🚀 Features

* 📊 Real-time price prediction
* 🔄 Model trains automatically on app startup (no saved model needed)
* 🧠 End-to-end ML pipeline (preprocessing + model)
* 🎛️ Interactive UI using Streamlit
* ⚡ Fast and user-friendly interface

---

## 🧠 Machine Learning Pipeline

The model is built using a structured pipeline:

* **Data Preprocessing**

  * Missing value handling (SimpleImputer)
  * Feature scaling (StandardScaler)
  * Categorical encoding (OneHotEncoder)

* **Model**

  * Linear Regression

* **Pipeline Includes**

  * ColumnTransformer for mixed data types
  * Fully automated training + prediction

---

## 📂 Dataset Features

The model uses the following input features:

* Distance (miles)
* Duration (minutes)
* Hour of the day
* Day of the week
* Weather conditions
* Temperature
* Pickup location
* Drop-off location
* Vehicle type
* Driver rating

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:**

  * Pandas, NumPy
  * Scikit-learn
* **Framework:** Streamlit

---

## 🖥️ How to Run Locally

1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app

```bash
streamlit run app.py
```

## 🎯 Learning Outcomes

* Built an end-to-end ML pipeline
* Learned feature engineering & preprocessing
* Deployed ML model using Streamlit
* Gained experience with real-world dataset handling

---

## 📌 Future Improvements

* Use advanced models (Random Forest, XGBoost)
* Add model evaluation metrics (R², MAE, RMSE)
* Improve UI with dropdowns for categorical inputs
* Deploy with Docker / cloud platforms

---

## 🤝 Feedback & Contribution

Feedback is always welcome! Feel free to fork, improve, or suggest ideas.

---

⭐ **If you like this project, give it a star!**
