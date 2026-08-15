# app.py
# ---------------------------------------------------------------------------
# UDA7: Credit Card Fraud Detection - REST API
#
# This is a small FastAPI application that wraps the trained fraud detection
# model produced by UDA7_credit_card_fraud_detection_complete.py.
#
# Endpoints:
#   GET  /          -> simple health check / welcome message
#   GET  /sample    -> one random transaction from creditcard.csv (demo only)
#   POST /predict   -> fraud prediction for one transaction (JSON in, JSON out)
#
# The core ML logic (training, evaluation, threshold selection) lives in the
# original training script and is NOT modified here. This file only loads the
# saved model artifacts and serves predictions.
#
# Run with:
#   uvicorn app:app --reload --port 8000
# ---------------------------------------------------------------------------

import os
import random

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths (all files live next to this script inside the backend/ folder)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_detection_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "fraud_detection_metadata.joblib")
DATA_PATH = os.path.join(BASE_DIR, "creditcard.csv")

app = FastAPI(
    title="UDA7 Credit Card Fraud Detection API",
    description="Wraps the trained fraud detection model with a REST API.",
    version="1.0.0",
)

# Allow the frontend (served from a different port / opened as a local file)
# to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load the model and metadata ONCE at startup (faster than loading per request)
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(
        "Model artifacts not found. Run the training script first:\n"
        "  python UDA7_credit_card_fraud_detection_complete.py\n"
        "to generate fraud_detection_model.joblib and "
        "fraud_detection_metadata.joblib."
    )

MODEL = joblib.load(MODEL_PATH)
METADATA = joblib.load(METADATA_PATH)
FEATURES = METADATA["features"]  # e.g. ["Time", "V1", ..., "V28", "Amount"]
THRESHOLD = METADATA["threshold"]


# ---------------------------------------------------------------------------
# fraud_alert()
#
# Copied VERBATIM from section "17. FRAUD ALERT FUNCTION" of
# UDA7_credit_card_fraud_detection_complete.py so the original script stays
# unchanged. The only difference is that the default paths point to the files
# inside this backend/ folder.
# ---------------------------------------------------------------------------
def fraud_alert(transaction, model_path=MODEL_PATH,
                metadata_path=METADATA_PATH):
    """
    Predict one transaction.

    transaction:
        dict with the same feature names used during training.

    Returns:
        Dictionary containing fraud probability, decision and alert.
    """
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)

    required_features = metadata["features"]
    threshold = metadata["threshold"]

    missing = [f for f in required_features if f not in transaction]
    if missing:
        raise ValueError(
            "Transaction is missing required features: " + ", ".join(missing)
        )

    row = pd.DataFrame(
        [[transaction[f] for f in required_features]],
        columns=required_features
    )

    probability = float(model.predict_proba(row)[0, 1])
    is_fraud = probability >= threshold

    if is_fraud:
        alert = "FRAUD ALERT: Transaction requires investigation."
        decision = "FRAUD"
    else:
        alert = "No fraud alert: Transaction classified as legitimate."
        decision = "LEGITIMATE"

    return {
        "fraud_probability": probability,
        "threshold": threshold,
        "decision": decision,
        "alert": alert
    }


# ---------------------------------------------------------------------------
# Request model: one transaction with all 30 features.
# Using explicit fields keeps the API self-documenting at /docs.
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


@app.get("/")
def root():
    """Simple health check so you can confirm the backend is running."""
    return {
        "message": "UDA7 Credit Card Fraud Detection API is running.",
        "model": METADATA.get("model_name"),
        "threshold": THRESHOLD,
        "endpoints": {"predict": "POST /predict", "sample": "GET /sample"},
    }


@app.get("/sample")
def sample_transaction():
    """
    Return ONE random transaction from creditcard.csv for demo/testing.
    The full dataset is never exposed - only a single row at a time.
    """
    if not os.path.exists(DATA_PATH):
        raise HTTPException(
            status_code=500,
            detail="creditcard.csv not found in the backend folder.",
        )

    # Count data rows once, then read only the one row we need.
    # This avoids loading the entire large CSV into memory per request.
    with open(DATA_PATH) as f:
        total_rows = sum(1 for _ in f) - 1  # minus header row

    row_number = random.randint(0, total_rows - 1)
    row = pd.read_csv(DATA_PATH, skiprows=range(1, row_number + 1), nrows=1)

    record = row.iloc[0].to_dict()
    actual_class = int(record.pop("Class", -1))  # kept for demo comparison

    return {
        "transaction": {k: float(v) for k, v in record.items()},
        "actual_class": actual_class,  # 0 = legitimate, 1 = fraud (ground truth)
    }


@app.post("/predict")
def predict(transaction: Transaction):
    """
    Predict whether one transaction is fraudulent.
    Accepts JSON with Time, Amount and V1-V28; returns probability,
    decision and alert message using the saved model + tuned threshold.
    """
    try:
        result = fraud_alert(transaction.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return result
