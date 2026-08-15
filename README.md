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

Everything runs locally with free, open-source tools only, and can also be
deployed for free to Render as a single web service (see
[Deploying to Render](#deploying-to-render-free) below).

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
├── render.yaml                                       # one-click Render deployment
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

## 3. Run the app (one terminal is enough)

The backend now also serves the frontend, so a single command runs the whole
app with **no cross-origin (CORS) requests at all**:

```bash
cd backend
uvicorn app:app --port 8000
```

You should see `Uvicorn running on http://127.0.0.1:8000`.

Quick checks:

- Open <http://127.0.0.1:8000> — the fraud detection web page loads.
- Open <http://127.0.0.1:8000/health> — JSON health check with the model
  name and threshold.
- Open <http://127.0.0.1:8000/docs> — interactive API documentation where
  you can try `/sample` and `/predict` directly.

## 4. (Optional) Run the frontend on its own server

Serving `frontend/` separately still works — the page automatically detects
that it is on another port and calls the API at port 8000 (CORS is enabled
for all origins). In a **second terminal**, from the project root:

```bash
cd frontend
python3 -m http.server 3000
```

Then open <http://127.0.0.1:3000> in your browser.

## 5. Confirm everything is connected

1. Open the app (<http://127.0.0.1:8000>, or <http://127.0.0.1:3000> if you
   used step 4).
2. Click **"Load sample transaction"** — all fields (Amount, Time and the
   V1–V28 advanced fields) should fill automatically with a real transaction
   from the dataset. This proves the frontend can reach the backend.
3. Click **"Check transaction"** — you should see either **FRAUD ALERT** or
   **LEGITIMATE**, the fraud probability as a percentage, and a colored
   gauge with the alert threshold marked on it.
4. If the sample came from the dataset, the page also shows the sample's
   ground-truth label so you can compare the model's decision against it.

**Troubleshooting:** if the page shows *"Could not reach the backend"*,
use <http://127.0.0.1:8000> directly (single-server mode) — the frontend and
API are then the same origin, so no cross-origin request is ever made. If
you insist on port 3000, make sure the backend terminal is still running on
port 8000 and check the API address shown in the page footer.

## API reference

| Method | Endpoint   | Description                                                              |
| ------ | ---------- | ------------------------------------------------------------------------ |
| GET    | `/`        | The web frontend (served by the backend)                                 |
| GET    | `/health`  | Health check: model name, threshold, endpoint list                       |
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

## Deploying to Render (free)

The repository includes a `render.yaml` blueprint that deploys the backend
and frontend together as **one free web service** (the FastAPI app serves
the frontend, so no separate static site or CORS setup is needed):

1. Create a free account at <https://render.com> (sign in with GitHub).
2. Go to <https://dashboard.render.com/blueprints> → **New Blueprint
   Instance**, select this repository, and click **Deploy**.
3. When the build finishes, open the service URL
   (e.g. `https://fraud-detection-ai.onrender.com`) — the full app works
   there on desktop and mobile.

Note: on the free plan the service sleeps after ~15 minutes of inactivity,
so the first request after a while can take up to a minute while it wakes up.

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
