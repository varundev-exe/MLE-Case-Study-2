# Credit Card Fraud Detection — XGBoost + SMOTE

Detects fraudulent transactions in a heavily imbalanced dataset
(fraud is typically <1% of transactions), using XGBoost, SMOTE
oversampling, tuned decision thresholds, and feature importance
for interpretation.

## Why this approach

- **XGBoost**: handles nonlinear feature interactions (e.g. amount ×
  time-of-day × device) well, is fast on tabular data, and is the
  standard baseline for Kaggle-style fraud/credit datasets.
- **SMOTE (Synthetic Minority Oversampling)**: generates synthetic
  minority-class (fraud) examples by interpolating between nearest
  neighbors, so the model sees enough positive examples during
  training instead of just learning to always predict "legit."
  **Critical**: SMOTE is applied only to the training fold, after
  the train/test split — applying it before splitting leaks synthetic
  neighbors into the test set and inflates metrics.
- **Threshold tuning**: the default 0.5 cutoff is rarely optimal for
  imbalanced fraud data. This script offers two tuning strategies:
  1. **F1-optimal** — balances precision and recall when no explicit
     cost model is available.
  2. **Cost-based** — minimizes total business cost given a
     configurable false-negative (missed fraud) vs. false-positive
     (false alarm) cost ratio.

## Data

`generate_synthetic_data()` creates ~50,000 synthetic transactions at
a ~0.6% fraud rate, with features: amount, hour of day, distance from
home, transaction velocity (last hour), new-device flag, and a
merchant risk score.

To use real data, swap in:
- Kaggle's [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  (V1–V28 PCA-anonymized features + `Amount`, `Time`, `Class`), or
- [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)

and update `FEATURE_COLUMNS` to match the real schema.

## Pipeline

1. Train/test split (stratified — preserves the fraud rate in both sets)
2. Standardize features
3. **SMOTE** on the training set only, to balance classes before fitting
4. Train `XGBClassifier` (tuned for tabular imbalanced data: shallow-ish
   trees, subsampling, `eval_metric="aucpr"`)
5. Evaluate on the untouched, imbalanced test set — this is what matters,
   since real-world traffic is imbalanced too
6. Tune decision threshold two ways (F1-optimal, cost-optimal)
7. Extract and plot feature importance (gain-based)

## Running it

```bash
pip install -r requirements.txt
python fraud_detection.py
```

Outputs:
- Console: class balance before/after SMOTE, ROC-AUC/PR-AUC, classification
  reports at default / F1-optimal / cost-optimal thresholds, feature
  importance table
- `fraud_model_evaluation.png`: ROC curve, precision-recall curve, and
  feature importance bar chart

## Why PR-AUC matters more than ROC-AUC here

With ~99%+ of transactions legitimate, ROC-AUC can look excellent even
for a mediocre model, because the false positive **rate** stays low in
relative terms while absolute false positives pile up. **PR-AUC**
(precision vs. recall) is more sensitive to performance on the rare
positive (fraud) class and is the primary metric reported here.

## Cost framing

| Error type | Real-world consequence | Relative cost |
|---|---|---|
| **False Negative** | Missed fraud → chargeback, lost goods/funds, regulatory/reputational cost | High |
| **False Positive** | Legitimate transaction declined/flagged → customer friction, manual review cost | Moderate |

Unlike the readmission case study (FN >> FP), fraud false positives are
not negligible — blocking real customers' cards has real churn and
support costs. The default cost ratio here (50:1) is illustrative;
tune it against your actual chargeback and review costs.

## Next steps / extensions

- Try `scale_pos_weight` in XGBoost as an alternative/complement to SMOTE
  (avoids synthesizing data, often competitive)
- SHAP values for per-transaction, case-level explanations (vs. global
  gain importance)
- Time-based (not random) train/test split, since fraud patterns drift
- Precision@k / alerts-per-day framing for a fraud ops team's actual workflow
