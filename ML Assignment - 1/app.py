import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from pathlib import Path

# Path to your RideEase CSV dataset
DATA_PATH = Path(__file__).parent / "ride_sharing_dataset.csv"

st.set_page_config(page_title="RideEase Price Predictor", layout="centered")
st.title("🚕 RideEase — Price Prediction App")
st.write("The model trains on the dataset every time the app starts—no saved model required.")

@st.cache_resource
def load_and_train():
    df = pd.read_csv(DATA_PATH)

    target = "price"
    features = [
        'distance_miles', 'duration_minutes', 'hour', 'day_of_week',
        'weather', 'temperature', 'pickup_location',
        'dropoff_location', 'vehicle_type', 'driver_rating'
    ]
    # keep only columns present
    features = [c for c in features if c in df.columns]
    X = df[features]
    y = df[target]

    numeric_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_feats = [c for c in features if c not in numeric_feats]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) 
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_feats),
        ("cat", categorical_transformer, categorical_feats)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    model.fit(X, y)
    return model, features

st.info("Training model... (cached after first run)")
pipe, features = load_and_train()

# Default numeric values
defaults = {
    "distance_miles": 5.0,
    "duration_minutes": 12.0,
    "hour": 14,
    "day_of_week": 2,
    "temperature": 25.0,
    "driver_rating": 4.8
}

st.sidebar.header("Ride Details")
inputs = {}

for feat in features:
    if feat in ["distance_miles", "duration_minutes", "temperature",
                "driver_rating", "hour", "day_of_week"]:
        default = defaults.get(feat, 0.0)
        if feat == "hour":
            inputs[feat] = st.sidebar.slider("Hour of Day (0–23)", 0, 23, int(default))
        elif feat == "day_of_week":
            inputs[feat] = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, int(default))
        elif feat == "driver_rating":
            inputs[feat] = st.sidebar.slider("Driver Rating (1.0–5.0)", 1.0, 5.0,
                                             float(default), step=0.1)
        else:
            inputs[feat] = st.sidebar.number_input(feat.replace("_", " ").title(),
                                                   value=float(default), min_value=0.0)
    else:
        # Free text input for categorical features
        inputs[feat] = st.sidebar.text_input(feat.replace("_", " ").title(), value="")

st.sidebar.markdown("---")
if st.sidebar.button("Predict price"):
    input_df = pd.DataFrame([inputs])
    for c in ["distance_miles", "duration_minutes", "hour",
              "day_of_week", "temperature", "driver_rating"]:
        if c in input_df.columns:
            input_df[c] = pd.to_numeric(input_df[c], errors="coerce")
    try:
        pred = pipe.predict(input_df)[0]
        st.success(f"Estimated Price: **${pred:.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
