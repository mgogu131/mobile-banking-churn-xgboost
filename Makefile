.PHONY: venv install train tune shap test

install:
	pip install -r requirements.txt

train:
	python -m churn.train --data data/churn.csv --target Target --date_col Calculation_Date --test_date 2023-09-30 --out artifacts/model.joblib

tune:
	python -m churn.tune --data data/churn.csv --target Target --date_col Calculation_Date --test_date 2023-09-30 --out artifacts/best_params.csv

shap:
	python -m churn.shap_report --model artifacts/model.joblib --data data/churn.csv --target Target --date_col Calculation_Date --test_date 2023-09-30 --out artifacts/shap_top10.csv

test:
	pytest -q
