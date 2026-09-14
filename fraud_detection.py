"""
Credit Card Fraud Detection — XGBoost + SMOTE
======================================================================
Detects fraudulent transactions in a heavily imbalanced dataset
(fraud typically <1% of transactions, as in Kaggle's IEEE-CIS /
creditcard.csv datasets). Uses SMOTE oversampling on the training
fold only, XGBoost as the classifier, threshold tuning against a
precision/recall trade-off, and feature importance for interpretation.

NOTE: This script uses a SYNTHETIC dataset generator so it runs
out-of-the-box. To use real data, replace `generate_synthetic_data()`
with a `pd.read_csv(...)` call on the Kaggle "Credit Card Fraud
Detection" (V1-V28 PCA features) or "IEEE-CIS Fraud Detection" dataset,
and update FEATURE_COLUMNS accordingly.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report, f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------
def generate_synthetic_data(n_samples=50_000, fraud_rate=0.006, random_state=RANDOM_STATE):
    """Synthetic transaction data mimicking a heavily imbalanced fraud dataset."""
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    def make_block(n, fraud_flag):
        # Fraudulent transactions: higher amounts, odd hours, new devices,
        # more distance from home, higher velocity of recent transactions.
        if fraud_flag:
            amount = rng.gamma(3, 150, n)
            hour = rng.choice(np.arange(24), n, p=_night_weighted_hours())
            distance_from_home = rng.exponential(80, n)
            num_txn_last_hour = rng.poisson(4, n)
            is_new_device = rng.binomial(1, 0.6, n)
            merchant_risk_score = rng.beta(4, 2, n)
        else:
            amount = rng.gamma(2, 40, n)
            hour = rng.integers(0, 24, n)
            distance_from_home = rng.exponential(10, n)
            num_txn_last_hour = rng.poisson(0.5, n)
            is_new_device = rng.binomial(1, 0.05, n)
            merchant_risk_score = rng.beta(2, 5, n)

        return pd.DataFrame({
            "amount": amount,
            "hour": hour,
            "distance_from_home_km": distance_from_home,
            "num_txn_last_hour": num_txn_last_hour,
            "is_new_device": is_new_device,
            "merchant_risk_score": merchant_risk_score,
            "is_fraud": fraud_flag,
        })

    def _night_weighted_hours():
        w = np.ones(24)
        w[0:5] *= 3  # fraud skews toward late-night hours
        return w / w.sum()

    df = pd.concat([make_block(n_legit, 0), make_block(n_fraud, 1)], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


FEATURE_COLUMNS = [
    "amount", "hour", "distance_from_home_km",
    "num_txn_last_hour", "is_new_device", "merchant_risk_score",
]
TARGET_COLUMN = "is_fraud"


# ---------------------------------------------------------------------
# 2. Train with SMOTE (applied to training fold only!)
# ---------------------------------------------------------------------
def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply SMOTE only to training data, never to validation/test —
    # oversampling before the split leaks synthetic neighbors across folds.
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"Before SMOTE: {np.bincount(y_train)}")
    print(f"After  SMOTE: {np.bincount(y_res)}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_res, y_res)
    return model, scaler


# ---------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------
def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"\nTest ROC-AUC: {roc_auc:.4f}")
    print(f"Test PR-AUC : {pr_auc:.4f}  (more informative given ~{y_test.mean():.2%} fraud rate)")

    y_pred_default = (y_proba >= 0.5).astype(int)
    print("\n--- Classification report @ threshold = 0.5 ---")
    print(classification_report(y_test, y_pred_default, digits=3))

    return y_proba, roc_auc, pr_auc


# ---------------------------------------------------------------------
# 4. Threshold tuning
# ---------------------------------------------------------------------
def tune_threshold_by_f1(y_test, y_proba):
    """Find threshold maximizing F1 — a reasonable default when precision
    and recall are both important and no explicit cost model exists."""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])  # last point has no threshold
    best_threshold = thresholds[best_idx]

    print(f"\n--- F1-optimal threshold ---")
    print(f"Threshold = {best_threshold:.3f}  |  F1 = {f1_scores[best_idx]:.3f}  "
          f"|  Precision = {precisions[best_idx]:.3f}  |  Recall = {recalls[best_idx]:.3f}")

    y_pred = (y_proba >= best_threshold).astype(int)
    print("\n--- Classification report @ F1-optimal threshold ---")
    print(classification_report(y_test, y_pred, digits=3))
    return best_threshold


def cost_based_threshold(y_test, y_proba, cost_fn=50, cost_fp=1):
    """
    Alternative: pick threshold minimizing total business cost.
    cost_fn: cost of missing real fraud (chargeback, lost goods, fees)
    cost_fp: cost of a false alarm (blocked legit txn, customer friction,
             manual review cost)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        costs.append(fn * cost_fn + fp * cost_fp)

    best_idx = int(np.argmin(costs))
    best_threshold = thresholds[best_idx]
    print(f"\n--- Cost-based threshold (FN:FP = {cost_fn}:{cost_fp}) ---")
    print(f"Optimal threshold = {best_threshold:.2f}  |  Min expected cost = {costs[best_idx]:.0f}")

    y_pred_opt = (y_proba >= best_threshold).astype(int)
    print("\n--- Classification report @ cost-optimal threshold ---")
    print(classification_report(y_test, y_pred_opt, digits=3))
    return best_threshold


# ---------------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------------
def show_feature_importance(model):
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "gain_importance": model.feature_importances_,
    }).sort_values("gain_importance", ascending=False)
    print("\n--- XGBoost Feature Importance (gain) ---")
    print(importance.to_string(index=False))
    return importance


# ---------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------
def plot_curves(y_test, y_proba, importance_df, save_path="fraud_model_evaluation.png"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, y_proba):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
    axes[1].plot(recalls, precisions,
                 label=f"PR-AUC = {average_precision_score(y_test, y_proba):.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve (key metric for imbalanced fraud)")
    axes[1].legend()

    axes[2].barh(importance_df["feature"][::-1], importance_df["gain_importance"][::-1])
    axes[2].set_xlabel("Gain importance")
    axes[2].set_title("XGBoost Feature Importance")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved evaluation plots to {save_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_synthetic_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df[TARGET_COLUMN].mean():.3%}\n")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model, scaler = train_model(X_train, y_train)
    y_proba, roc_auc, pr_auc = evaluate_model(model, scaler, X_test, y_test)
    tune_threshold_by_f1(y_test, y_proba)
    cost_based_threshold(y_test, y_proba, cost_fn=50, cost_fp=1)
    importance_df = show_feature_importance(model)
    plot_curves(y_test, y_proba, importance_df)
