import joblib
import numpy as np
import os
import pandas as pd
import re

# -------------------------------
# Load data & model once
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "flattened_patient_visits.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "lifespan_model.pkl")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
print("Columns:", df.columns.tolist())

# -------------------------------
# Feature extractor
# -------------------------------
def extract_features_for_patient(patient_id: int) -> dict:
    visits = df[df["PatientID"] == patient_id]
    if visits.empty:
        raise ValueError(f"No data found for PatientID {patient_id}")

    # Sort by date
    visits = visits.sort_values("VisitDate")

    psa_values = visits["PSA"].dropna().tolist()
    weight_values = visits["Weight"].dropna().tolist()
    pirads_values = visits["PIRADS"].dropna().tolist()
    treatments = visits["Treatment"].dropna().astype(str).str.upper().tolist()

    features = {
        "PSA_Initial": psa_values[0] if psa_values else 0,
        "PSA_Latest": psa_values[-1] if psa_values else 0,
        "PSA_Change": (psa_values[-1] - psa_values[0]) if len(psa_values) > 1 else 0,
        "Max_PIRADS": max(pirads_values) if pirads_values else 0,
        "Weight_Initial": weight_values[0] if weight_values else 0,
        "Weight_Latest": weight_values[-1] if weight_values else 0,
        "Weight_Change": (weight_values[-1] - weight_values[0]) if len(weight_values) > 1 else 0,
        "Max_BonePainScore": 0,  # Set to 0 unless you add that column
        "Has_Surgery": int(any("SURGERY" in t for t in treatments)),
        "Has_Radiation": int(any("RADIATION" in t for t in treatments)),
        "Has_ADT": int(any("ADT" in t for t in treatments))
    }

    return features
# -------------------------------
# Final Prediction Function
# -------------------------------
def predict_lifespan(patient_id: str) -> float:
    features = extract_features_for_patient(patient_id)
    input_array = np.array([[
        features["PSA_Initial"],
        features["PSA_Latest"],
        features["PSA_Change"],
        features["Max_PIRADS"],
        features["Weight_Initial"],
        features["Weight_Latest"],
        features["Weight_Change"],
        features["Max_BonePainScore"],
        features["Has_Surgery"],
        features["Has_Radiation"],
        features["Has_ADT"]
    ]])
    return float(model.predict(input_array)[0])

