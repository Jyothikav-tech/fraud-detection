from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib, pandas as pd

app = FastAPI(title="Credit Card Fraud Detection API")

model = joblib.load("models/xgb_model.joblib")
scaler = joblib.load("models/scaler.joblib")
columns = joblib.load("models/feature_columns.joblib")
threshold = joblib.load("models/threshold.joblib")


class Transaction(BaseModel):
    Time: float = Field(..., ge=0)
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(tx: Transaction):
    df = pd.DataFrame([tx.model_dump()])[columns]
    df[["Amount", "Time"]] = scaler.transform(df[["Amount", "Time"]])
    proba = float(model.predict_proba(df)[0, 1])
    return {
        "fraud_probability": round(proba, 4),
        "is_fraud": proba >= threshold,
        "threshold": round(threshold, 3),
    }