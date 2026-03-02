#!/usr/bin/env bash
set -euo pipefail

python -m churn.train   --data data/churn.csv   --target Target   --date_col Calculation_Date   --test_date 2023-09-30   --out artifacts/model.joblib
