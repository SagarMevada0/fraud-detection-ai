# generate_results.py
# ---------------------------------------------------------------------------
# UDA7: Credit Card Fraud Detection - Results/Evaluation artifact generator
#
# Regenerates the evaluation artifacts (metrics JSON + matplotlib plots) that
# the training script UDA7_credit_card_fraud_detection_complete.py produces
# during training, WITHOUT retraining anything:
#
#   - loads the saved model + metadata (fraud_detection_*.joblib)
#   - rebuilds the exact same train/validation/test split (same cleaning,
#     RANDOM_STATE and stratification as the training script, so the test
#     set is identical and the evaluation stays unbiased)
#   - saves, into backend/results/:
#       metrics.json            (Accuracy, Precision, Recall, F1, ROC-AUC,
#                                PR-AUC, confusion matrix counts, model name,
#                                threshold)
#       confusion_matrix.png    (section 12 of the training script)
#       roc_curve.png           (section 13)
#       pr_curve.png            (section 14)
#       threshold_analysis.png  (section 9, computed on the validation set)
#       feature_importance.png  (section 15, top 15 features)
#
# The plotting/metric code mirrors the training script sections named above;
# no ML logic is changed. Run once from the backend/ folder:
#   python generate_results.py
# ---------------------------------------------------------------------------

import json
import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_detection_model.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "fraud_detection_metadata.joblib")
DATA_PATH = os.path.join(BASE_DIR, "creditcard.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading model and metadata...")
model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)
threshold = metadata["threshold"]
target = metadata.get("target", "Class")
random_state = metadata.get("random_state", 42)

print("Loading dataset and rebuilding the exact train/val/test split...")
df = pd.read_csv(DATA_PATH)

# Same cleaning as the training script (section 2).
if df.duplicated().sum() > 0:
    df = df.drop_duplicates().reset_index(drop=True)
if df.isnull().sum().sum() > 0:
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

X = df.drop(columns=[target]).copy()
y = df[target].astype(int).copy()

# Same 60/20/20 split as the training script (section 4).
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=random_state, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25,
    random_state=random_state, stratify=y_train_full
)

# ---------------------------------------------------------------------------
# Final test evaluation (training script section 11)
# ---------------------------------------------------------------------------
print("Scoring the test set...")
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= threshold).astype(int)

final_cm = confusion_matrix(y_test, test_predictions)
tn, fp, fn, tp = final_cm.ravel()

test_metrics = {
    "Model": metadata["model_name"],
    "Threshold": float(threshold),
    "Accuracy": float(accuracy_score(y_test, test_predictions)),
    "Precision": float(precision_score(y_test, test_predictions, zero_division=0)),
    "Recall": float(recall_score(y_test, test_predictions, zero_division=0)),
    "F1": float(f1_score(y_test, test_predictions, zero_division=0)),
    "ROC_AUC": float(roc_auc_score(y_test, test_probabilities)),
    "PR_AUC": float(average_precision_score(y_test, test_probabilities)),
}

metrics_payload = {
    "model_name": metadata["model_name"],
    "threshold": float(threshold),
    "test_metrics": test_metrics,
    "confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    },
    "test_set_size": int(len(y_test)),
    "test_fraud_count": int(y_test.sum()),
}

with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_payload, f, indent=2)
print("Saved metrics.json")

# ---------------------------------------------------------------------------
# Confusion matrix plot (training script section 12)
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 5))
plt.imshow(final_cm)
plt.title("Final Model Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks([0, 1], ["Legitimate", "Fraud"])
plt.yticks([0, 1], ["Legitimate", "Fraud"])
for i in range(final_cm.shape[0]):
    for j in range(final_cm.shape[1]):
        plt.text(j, i, final_cm[i, j], ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("Saved confusion_matrix.png")

# ---------------------------------------------------------------------------
# ROC curve (training script section 13)
# ---------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, test_probabilities)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC-AUC = {test_metrics['ROC_AUC']:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Final Model")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
plt.close()
print("Saved roc_curve.png")

# ---------------------------------------------------------------------------
# Precision-Recall curve (training script section 14)
# ---------------------------------------------------------------------------
precision, recall, _ = precision_recall_curve(y_test, test_probabilities)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR-AUC = {test_metrics['PR_AUC']:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Final Model")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pr_curve.png"), dpi=150)
plt.close()
print("Saved pr_curve.png")

# ---------------------------------------------------------------------------
# Threshold analysis on the validation set (training script section 9)
# ---------------------------------------------------------------------------
print("Computing threshold analysis on the validation set...")
val_probabilities = model.predict_proba(X_val)[:, 1]

rows = []
for t in np.arange(0.05, 0.96, 0.05):
    pred = (val_probabilities >= t).astype(int)
    rows.append({
        "Threshold": round(float(t), 2),
        "Precision": precision_score(y_val, pred, zero_division=0),
        "Recall": recall_score(y_val, pred, zero_division=0),
        "F1": f1_score(y_val, pred, zero_division=0),
    })
threshold_df = pd.DataFrame(rows)

plt.figure(figsize=(8, 5))
plt.plot(threshold_df["Threshold"], threshold_df["Precision"], label="Precision")
plt.plot(threshold_df["Threshold"], threshold_df["Recall"], label="Recall")
plt.plot(threshold_df["Threshold"], threshold_df["F1"], label="F1")
plt.axvline(threshold, linestyle="--", color="gray",
            label=f"Selected threshold = {threshold:.2f}")
plt.xlabel("Fraud Alert Threshold")
plt.ylabel("Score")
plt.title("Threshold Analysis")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "threshold_analysis.png"), dpi=150)
plt.close()
print("Saved threshold_analysis.png")

# ---------------------------------------------------------------------------
# Feature importance plot (training script section 15)
# ---------------------------------------------------------------------------
fi_path = os.path.join(BASE_DIR, "feature_importance.csv")
if os.path.exists(fi_path):
    feature_importance = pd.read_csv(fi_path)
else:
    estimator = model
    if hasattr(estimator, "named_steps"):
        estimator = list(estimator.named_steps.values())[-1]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = None
    feature_importance = None
    if values is not None and len(values) == len(X.columns):
        feature_importance = pd.DataFrame({
            "Feature": X.columns.tolist(),
            "Importance": values,
        }).sort_values("Importance", ascending=False)

if feature_importance is not None:
    top_n = min(15, len(feature_importance))
    plot_df = feature_importance.head(top_n).sort_values("Importance")
    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["Feature"], plot_df["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Features Used by Final Model")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("Saved feature_importance.png")
else:
    print("Feature importance is not available for this model.")

print("\nAll results saved to:", RESULTS_DIR)
