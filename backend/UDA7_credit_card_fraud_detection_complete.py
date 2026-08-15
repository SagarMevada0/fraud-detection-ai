# UDA7: Enhancing Credit Card Fraud Detection and Alert using AI-Powered Techniques
# Complete MSc AI & Data Science project pipeline
#
# Run in Google Colab/Jupyter.
# Expected CSV: creditcard.csv
# Target column: Class (0 = legitimate, 1 = fraud)
#
# The script:
# 1. Loads and validates the dataset
# 2. Performs EDA
# 3. Splits data without leakage
# 4. Compares baseline Logistic Regression, Random Forest and XGBoost (if installed)
# 5. Handles imbalance using class weighting and SMOTE
# 6. Evaluates Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
# 7. Creates confusion matrices, ROC/PR curves and feature importance
# 8. Tunes the best candidate model
# 9. Optimises the fraud alert threshold on a validation set
# 10. Saves the final model and scaler
# 11. Provides a simple fraud-alert prediction function
#
# IMPORTANT:
# - Never apply SMOTE to the test set.
# - Threshold optimisation is performed on validation data only.
# - The final test set is used once for final unbiased evaluation.

# =========================
# 0. INSTALL / IMPORT
# =========================
# In Google Colab, uncomment:
# !pip -q install imbalanced-learn xgboost joblib

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    print("XGBoost is not available. The rest of the pipeline will still run.")

RANDOM_STATE = 42
DATA_PATH = "creditcard.csv"
TARGET = "Class"

# =========================
# 1. LOAD DATA
# =========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find '{DATA_PATH}'. Upload your dataset to the notebook/Colab "
        "working directory or change DATA_PATH."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found in the dataset.")

# =========================
# 2. DATA QUALITY CHECK
# =========================
print("\n" + "=" * 70)
print("DATA QUALITY")
print("=" * 70)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

print("\nDuplicate rows:", df.duplicated().sum())

print("\nTarget distribution:")
print(df[TARGET].value_counts())
print("\nTarget proportions:")
print(df[TARGET].value_counts(normalize=True))

# Remove exact duplicate rows if present.
# This is a methodological choice; report it in the dissertation.
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"\nRemoved {duplicate_count} duplicate rows.")

# Basic missing-value handling.
if df.isnull().sum().sum() > 0:
    # Median imputation for numeric columns.
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    print("Numeric missing values were median-imputed.")

# =========================
# 3. EDA
# =========================
print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Class distribution
plt.figure(figsize=(7, 5))
df[TARGET].value_counts().sort_index().plot(kind="bar")
plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Amount distribution if available
if "Amount" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df.loc[df[TARGET] == 0, "Amount"], bins=50, alpha=0.7, label="Legitimate")
    plt.hist(df.loc[df[TARGET] == 1, "Amount"], bins=50, alpha=0.7, label="Fraud")
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Log amount distribution can make skew easier to inspect.
if "Amount" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(np.log1p(df["Amount"]), bins=60)
    plt.title("Log-Transformed Transaction Amount Distribution")
    plt.xlabel("log(1 + Amount)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Correlation with target
numeric_df = df.select_dtypes(include=np.number)
if TARGET in numeric_df.columns:
    target_corr = (
        numeric_df.corr()[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    print("\nTop correlations with fraud target:")
    print(target_corr.head(15))

# =========================
# 4. PREPARE FEATURES
# =========================
X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].astype(int).copy()

# Drop non-numeric columns if any.
# Most standard creditcard.csv versions are entirely numeric.
non_numeric = X.select_dtypes(exclude=np.number).columns.tolist()
if non_numeric:
    print("\nNon-numeric columns found:", non_numeric)
    X = pd.get_dummies(X, columns=non_numeric, drop_first=True)

# Train / validation / test:
# 60% train, 20% validation, 20% final test.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.25,  # 25% of 80% = 20% overall
    random_state=RANDOM_STATE,
    stratify=y_train_full
)

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)
print("Training:", X_train.shape, "Fraud:", int(y_train.sum()))
print("Validation:", X_val.shape, "Fraud:", int(y_val.sum()))
print("Test:", X_test.shape, "Fraud:", int(y_test.sum()))

# =========================
# 5. EVALUATION FUNCTION
# =========================
def evaluate_model(name, model, X_eval, y_eval, threshold=0.5):
    """
    Evaluate a fitted probabilistic classifier.
    """
    probabilities = model.predict_proba(X_eval)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_eval, predictions),
        "Precision": precision_score(y_eval, predictions, zero_division=0),
        "Recall": recall_score(y_eval, predictions, zero_division=0),
        "F1": f1_score(y_eval, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_eval, probabilities),
        "PR_AUC": average_precision_score(y_eval, probabilities)
    }

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)
    for key, value in metrics.items():
        if key not in ["Model", "Threshold"]:
            print(f"{key:10s}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_eval, predictions))

    print("\nClassification report:")
    print(classification_report(y_eval, predictions, digits=4, zero_division=0))

    return metrics, probabilities, predictions

# =========================
# 6. BASELINE MODELS
# =========================
models = {}

# Logistic Regression with scaling.
models["Logistic Regression"] = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ))
])

# Random Forest with class weighting.
models["Random Forest (Balanced)"] = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

if XGBOOST_AVAILABLE:
    # scale_pos_weight is calculated from training data only.
    negative = max(int((y_train == 0).sum()), 1)
    positive = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = negative / positive

    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

# =========================
# 7. BASELINE TRAINING
# =========================
results = []
fitted_models = {}
validation_probabilities = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    metrics, probs, preds = evaluate_model(
        name, model, X_val, y_val, threshold=0.5
    )

    results.append(metrics)
    fitted_models[name] = model
    validation_probabilities[name] = probs

results_df = pd.DataFrame(results).sort_values(
    by=["PR_AUC", "F1"], ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("BASELINE MODEL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))

# =========================
# 8. SMOTE EXPERIMENT
# =========================
# SMOTE is applied ONLY to training data.
smote_logistic = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ))
])

smote_rf = ImbPipeline([
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

smote_models = {
    "Logistic Regression + SMOTE": smote_logistic,
    "Random Forest + SMOTE": smote_rf
}

if XGBOOST_AVAILABLE:
    smote_xgb = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    smote_models["XGBoost + SMOTE"] = smote_xgb

for name, model in smote_models.items():
    model.fit(X_train, y_train)

    metrics, probs, preds = evaluate_model(
        name, model, X_val, y_val, threshold=0.5
    )

    results.append(metrics)
    fitted_models[name] = model
    validation_probabilities[name] = probs

results_df = pd.DataFrame(results).sort_values(
    by=["PR_AUC", "F1"], ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("ALL MODEL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))

# Save comparison table.
results_df.to_csv("model_comparison.csv", index=False)

# =========================
# 9. THRESHOLD OPTIMISATION
# =========================
def threshold_table(y_true, probabilities):
    rows = []

    for threshold in np.arange(0.05, 0.96, 0.05):
        pred = (probabilities >= threshold).astype(int)
        cm = confusion_matrix(y_true, pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        rows.append({
            "Threshold": round(float(threshold), 2),
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred, zero_division=0),
            "F1": f1_score(y_true, pred, zero_division=0),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),
            "True_Negatives": int(tn)
        })

    return pd.DataFrame(rows)

# Choose model primarily by validation PR-AUC, then F1.
best_name = (
    results_df.sort_values(["PR_AUC", "F1"], ascending=False)
    .iloc[0]["Model"]
)

best_model = fitted_models[best_name]
best_val_probs = validation_probabilities[best_name]

threshold_df = threshold_table(y_val, best_val_probs)

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)
print(threshold_df.to_string(index=False))

# Default selection: maximum F1.
# For a real deployment, the threshold should be selected according to
# the institution's cost of false positives vs false negatives.
best_threshold = float(
    threshold_df.loc[threshold_df["F1"].idxmax(), "Threshold"]
)

print(f"\nSelected validation threshold by maximum F1: {best_threshold:.2f}")

# Plot precision/recall/F1 against threshold.
plt.figure(figsize=(8, 5))
plt.plot(threshold_df["Threshold"], threshold_df["Precision"], label="Precision")
plt.plot(threshold_df["Threshold"], threshold_df["Recall"], label="Recall")
plt.plot(threshold_df["Threshold"], threshold_df["F1"], label="F1")
plt.xlabel("Fraud Alert Threshold")
plt.ylabel("Score")
plt.title("Threshold Analysis")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# =========================
# 10. FINAL MODEL RETRAINING
# =========================
# Retrain the selected model on train + validation data.
X_train_final = pd.concat([X_train, X_val], axis=0)
y_train_final = pd.concat([y_train, y_val], axis=0)

# Recreate the selected estimator rather than refitting an already-used object
# to make the final training step explicit.
def make_model(name, y_fit):
    if name == "Logistic Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE
            ))
        ])

    if name == "Random Forest (Balanced)":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if name == "XGBoost":
        negative = max(int((y_fit == 0).sum()), 1)
        positive = max(int((y_fit == 1).sum()), 1)

        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negative / positive,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if name == "Logistic Regression + SMOTE":
        return ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE
            ))
        ])

    if name == "Random Forest + SMOTE":
        return ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

    if name == "XGBoost + SMOTE" and XGBOOST_AVAILABLE:
        return ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

    raise ValueError(f"Unknown model: {name}")

final_model = make_model(best_name, y_train_final)
final_model.fit(X_train_final, y_train_final)

# =========================
# 11. FINAL UNBIASED TEST EVALUATION
# =========================
test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

test_metrics = {
    "Model": best_name,
    "Threshold": best_threshold,
    "Accuracy": accuracy_score(y_test, test_predictions),
    "Precision": precision_score(y_test, test_predictions, zero_division=0),
    "Recall": recall_score(y_test, test_predictions, zero_division=0),
    "F1": f1_score(y_test, test_predictions, zero_division=0),
    "ROC_AUC": roc_auc_score(y_test, test_probabilities),
    "PR_AUC": average_precision_score(y_test, test_probabilities)
}

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

for key, value in test_metrics.items():
    if key in ["Model"]:
        print(f"{key:10s}: {value}")
    else:
        print(f"{key:10s}: {value:.4f}")

print("\nFinal confusion matrix:")
final_cm = confusion_matrix(y_test, test_predictions)
print(final_cm)

print("\nFinal classification report:")
print(classification_report(y_test, test_predictions, digits=4, zero_division=0))

# =========================
# 12. CONFUSION MATRIX PLOT
# =========================
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
plt.show()

# =========================
# 13. ROC CURVE
# =========================
fpr, tpr, roc_thresholds = roc_curve(y_test, test_probabilities)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC-AUC = {test_metrics['ROC_AUC']:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Final Model")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# =========================
# 14. PRECISION-RECALL CURVE
# =========================
precision, recall, pr_thresholds = precision_recall_curve(
    y_test, test_probabilities
)

plt.figure(figsize=(8, 6))
plt.plot(
    recall,
    precision,
    label=f"PR-AUC = {test_metrics['PR_AUC']:.4f}"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Final Model")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# =========================
# 15. FEATURE IMPORTANCE
# =========================
def get_feature_importance(model, feature_names):
    """
    Extract feature importance for tree models or absolute coefficients
    for logistic regression.
    """
    estimator = model

    # Handle imblearn pipeline.
    if hasattr(estimator, "named_steps"):
        last_step = list(estimator.named_steps.values())[-1]
        estimator = last_step

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        return None

    if len(values) != len(feature_names):
        return None

    fi = pd.DataFrame({
        "Feature": feature_names,
        "Importance": values
    }).sort_values("Importance", ascending=False)

    return fi

feature_names = X.columns.tolist()
feature_importance = get_feature_importance(final_model, feature_names)

if feature_importance is not None:
    print("\n" + "=" * 70)
    print("TOP FEATURE IMPORTANCE")
    print("=" * 70)
    print(feature_importance.head(20).to_string(index=False))

    feature_importance.to_csv("feature_importance.csv", index=False)

    top_n = min(15, len(feature_importance))
    plot_df = feature_importance.head(top_n).sort_values("Importance")

    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["Feature"], plot_df["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Features Used by Final Model")
    plt.tight_layout()
    plt.show()
else:
    print("\nFeature importance is not directly available for this final model.")

# =========================
# 16. SAVE FINAL MODEL
# =========================
joblib.dump(final_model, "fraud_detection_model.joblib")

metadata = {
    "model_name": best_name,
    "threshold": best_threshold,
    "features": feature_names,
    "target": TARGET,
    "random_state": RANDOM_STATE,
    "test_metrics": test_metrics
}

joblib.dump(metadata, "fraud_detection_metadata.joblib")

print("\nSaved:")
print(" - fraud_detection_model.joblib")
print(" - fraud_detection_metadata.joblib")
print(" - model_comparison.csv")
if feature_importance is not None:
    print(" - feature_importance.csv")

# =========================
# 17. FRAUD ALERT FUNCTION
# =========================
def fraud_alert(transaction, model_path="fraud_detection_model.joblib",
                metadata_path="fraud_detection_metadata.joblib"):
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

# =========================
# 18. EXAMPLE TRANSACTION TEST
# =========================
# Uses the first transaction from the test set purely as a demonstration.
example_transaction = X_test.iloc[0].to_dict()

example_result = fraud_alert(example_transaction)

print("\n" + "=" * 70)
print("EXAMPLE FRAUD ALERT")
print("=" * 70)
print(example_result)

# =========================
# 19. OPTIONAL: BATCH ALERT REPORT
# =========================
# Create a test-set alert report for analysis.
alert_report = X_test.copy()
alert_report["Actual_Class"] = y_test.values
alert_report["Fraud_Probability"] = test_probabilities
alert_report["Predicted_Class"] = test_predictions
alert_report["Alert"] = np.where(
    test_predictions == 1,
    "FRAUD ALERT",
    "LEGITIMATE"
)

alert_report.to_csv("fraud_alert_test_report.csv", index=False)

print("\nSaved batch alert report:")
print(" - fraud_alert_test_report.csv")

print("\n" + "=" * 70)
print("PROJECT PIPELINE COMPLETE")
print("=" * 70)
print("Use the generated CSV files and plots in your Results chapter.")
print("Do not report any metric until it has been produced by your actual run.")
