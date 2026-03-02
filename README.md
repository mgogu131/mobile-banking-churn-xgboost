# Mobile Banking Churn (XGBoost)

Binary classification project to predict customer churn in mobile banking.  
This repo is structured from an original exploratory notebook into a reproducible training script with:
- time-based holdout (by `Calculation_Date`)
- feature filtering (target correlation + pairwise correlation de-duplication)
- XGBoost training + ROC-AUC / Gini reporting
- optional GridSearchCV + SHAP explainability

> **Note:** Dataset is not included (internal). Place it under `data/` as a CSV.

## Setup (pip)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train (default)
```bash
python -m churn.train \
  --data data/churn.csv \
  --target Target \
  --date_col Calculation_Date \
  --test_date 2023-09-30 \
  --out artifacts/model.joblib
```

Outputs:
- prints `val_auc`, `val_gini`, `test_auc`, `test_gini`
- saves model pipeline to `artifacts/model.joblib`
- saves selected feature list to `artifacts/features.json`

## Grid search (optional)
```bash
python -m churn.tune \
  --data data/churn.csv \
  --target Target \
  --date_col Calculation_Date \
  --test_date 2023-09-30 \
  --out artifacts/best_params.csv
```

## SHAP (optional)
```bash
python -m churn.shap_report \
  --model artifacts/model.joblib \
  --data data/churn.csv \
  --target Target \
  --date_col Calculation_Date \
  --test_date 2023-09-30 \
  --out artifacts/shap_top10.csv
```

## Reproducibility
Seeds are set in `src/churn/utils.py` (Python + NumPy) and via `random_state` in train/val split and XGBoost.
