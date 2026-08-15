# UDA7: Enhancing Credit Card Fraud Detection and Alert using AI-Powered Techniques

An MSc AI & Data Science project that detects fraudulent credit card
transactions using machine learning, and serves the trained model through a
small web application:

- **Backend** — a FastAPI REST API that wraps the trained model
  (`/predict` and `/sample` endpoints).
- **Frontend** — a clean HTML/CSS/JavaScript page where you can enter a
  transaction (or load a real sample) and instantly see whether it is
  flagged as **FRAUD** or **LEGITIMATE**, with the fraud probability shown
  on a gauge.

Everything runs locally with free, open-source tools only.

## Project structure

```
credit-fraud-project/
├── backend/
│   ├── app.py                                        # FastAPI REST API
│   ├── UDA7_credit_card_fraud_detection_complete.py  # ML training pipeline (unchanged)
│   ├── fraud_detection_model.joblib                  # trained model (generated)
│   ├── fraud_detection_metadata.joblib               # features + threshold (generated)
│   └── creditcard.csv                                # Kaggle dataset
├── frontend/
│   ├── index.html                                    # web page
│   ├── style.css                                     # styling
│   └── script.js                                     # calls the backend API
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Install dependencies

Requires **Python 3.10+**. From the project root:

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt
```

## 2. Train the model (first time only)

The API needs the two `.joblib` files. If they are not already present in
`backend/`, generate them by running the training script once (this can take
a few minutes):

```bash
cd backend
python UDA7_credit_card_fraud_detection_complete.py
cd ..
```

This creates `fraud_detection_model.joblib` and
`fraud_detection_metadata.joblib` (plus evaluation CSVs and plots for the
dissertation Results chapter).

## 3. Run the backend

```bash
cd backend
uvicorn app:app --port 8000
```

You should see `Uvicorn running on http://127.0.0.1:8000`.

Quick checks:

- Open <http://127.0.0.1:8000> — you should see a JSON welcome message with
  the model name and threshold.
- Open <http://127.0.0.1:8000/docs> — interactive API documentation where
  you can try `/sample` and `/predict` directly.

Keep this terminal open; the API must stay running for the frontend to work.

## 4. Run the frontend

In a **second terminal**, from the project root:

```bash
cd frontend
python3 -m http.server 3000
```

Then open <http://127.0.0.1:3000> in your browser.

(Simply double-clicking `index.html` also works in most browsers, but using
the small HTTP server above is the most reliable option.)

## 5. Confirm everything is connected

1. With both terminals running, open <http://127.0.0.1:3000>.
2. Click **"Load sample transaction"** — all fields (Amount, Time and the
   V1–V28 advanced fields) should fill automatically with a real transaction
   from the dataset. This proves the frontend can reach the backend.
3. Click **"Check transaction"** — you should see either **FRAUD ALERT** or
   **LEGITIMATE**, the fraud probability as a percentage, and a colored
   gauge with the alert threshold marked on it.
4. If the sample came from the dataset, the page also shows the sample's
   ground-truth label so you can compare the model's decision against it.

**Troubleshooting:** if the page shows *"Could not reach the backend"*,
make sure the backend terminal is still running on port 8000, and that
`API_URL` at the top of `frontend/script.js` matches its address.

## API reference

| Method | Endpoint   | Description                                                              |
| ------ | ---------- | ------------------------------------------------------------------------ |
| GET    | `/`        | Health check: model name, threshold, endpoint list                       |
| GET    | `/sample`  | One random transaction from the dataset (demo only, never the full file) |
| POST   | `/predict` | Body: JSON with `Time`, `Amount`, `V1`–`V28`. Returns probability, decision, alert |

Example `/predict` response:

```json
{
  "fraud_probability": 0.97,
  "threshold": 0.5,
  "decision": "FRAUD",
  "alert": "FRAUD ALERT: Transaction requires investigation."
}
```

## Notes

- The core ML logic (training, evaluation, threshold optimisation) lives in
  `UDA7_credit_card_fraud_detection_complete.py` and is **unchanged**; the
  API only loads the saved model and reuses the script's `fraud_alert()`
  logic.
- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  (anonymized PCA features V1–V28, plus Time, Amount and the Class label).
- `creditcard.csv` and the trained model artifacts are committed to the
  repository, so the app works immediately after cloning — step 2 is only
  needed if you want to retrain the model yourself.
