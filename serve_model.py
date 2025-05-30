from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()
model = joblib.load("xgboost_model.joblib")

class PatientInput(BaseModel):
    psa: float
    gleason: int
    metastasis: int

@app.post("/recommend-treatment")
def recommend(input: PatientInput):
    features = [[input.psa, input.gleason, input.metastasis]]
    prediction = model.predict(features)[0]
    return {"recommendation": prediction}
