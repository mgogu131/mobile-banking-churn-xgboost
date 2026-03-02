import argparse
import json
import joblib
import numpy as np
import pandas as pd
import shap

from churn.data import load_csv, basic_clean

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--target", default="Target")
    p.add_argument("--date_col", default="Calculation_Date")
    p.add_argument("--test_date", required=True)
    p.add_argument("--features", default="artifacts/features.json")
    p.add_argument("--out", default="artifacts/shap_top10.csv")
    args = p.parse_args()

    model = joblib.load(args.model)
    df = basic_clean(load_csv(args.data))
    df[args.date_col] = df[args.date_col].astype(str)

    # use non-test part as background/analysis set
    train_df = df[df[args.date_col] != str(args.test_date)].copy()

    with open(args.features, "r", encoding="utf-8") as f:
        feats = json.load(f)["features"]

    X = train_df[feats].copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    top = pd.Series(shap_abs_mean, index=feats).sort_values(ascending=False).head(10)
    top.to_csv(args.out, header=["mean_abs_shap"])
    print("saved", args.out)

if __name__ == "__main__":
    main()
