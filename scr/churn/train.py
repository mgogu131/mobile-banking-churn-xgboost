import argparse
import json
import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from churn.config import SEED
from churn.utils import set_seeds
from churn.data import load_csv, basic_clean
from churn.feature_selection import select_features_by_corr

def gini_from_auc(auc: float) -> float:
    return 2.0 * auc - 1.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to CSV")
    p.add_argument("--target", default="Target")
    p.add_argument("--date_col", default="Calculation_Date")
    p.add_argument("--test_date", required=True, help="Holdout date for final test, e.g. 2023-09-30")
    p.add_argument("--out", default="artifacts/model.joblib")
    p.add_argument("--features_out", default="artifacts/features.json")
    p.add_argument("--corr_target_threshold", type=float, default=0.01)
    p.add_argument("--pair_corr_threshold", type=float, default=0.8)
    p.add_argument("--snapshot_date", default=None, help="Optional snapshot date for pairwise corr filtering")
    args = p.parse_args()

    set_seeds(SEED)

    df = basic_clean(load_csv(args.data))

    # split: time-based final test
    df[args.date_col] = df[args.date_col].astype(str)
    test_df = df[df[args.date_col] == str(args.test_date)].copy()
    train_df = df[df[args.date_col] != str(args.test_date)].copy()

    if len(test_df) == 0:
        raise ValueError(f"No rows found where {args.date_col} == {args.test_date}")

    y_test = test_df[args.target].astype(int)

    # feature selection (from notebook logic)
    model_vars = select_features_by_corr(
        train_df,
        target_col=args.target,
        date_col=args.date_col,
        corr_target_threshold=args.corr_target_threshold,
        pair_corr_threshold=args.pair_corr_threshold,
        snapshot_date=args.snapshot_date,
        non_model_vars=[args.target, "ClientID", "ClientName", "PhoneMobile"]
    )

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        train_df, train_df[args.target].astype(int),
        test_size=0.2, random_state=SEED, stratify=train_df[args.target].astype(int)
    )

    # model params aligned with the notebook baseline
    model = xgb.XGBClassifier(
        base_score=float(y_train.mean()),
        reg_alpha=0.1,
        reg_lambda=150,
        min_child_weight=20,
        max_depth=5,
        gamma=3,
        learning_rate=0.1,
        n_estimators=200,
        colsample_bynode=0.9,
        colsample_bylevel=0.9,
        subsample=0.9,
        tree_method="exact",
        sampling_method="uniform",
        n_jobs=-1,
        random_state=0,
        use_label_encoder=False,
        disable_default_eval_metric=True
    )

    model.fit(X_train[model_vars], y_train)

    val_pred = model.predict_proba(X_val[model_vars])[:, 1]
    test_pred = model.predict_proba(test_df[model_vars])[:, 1]

    val_auc = roc_auc_score(y_val, val_pred)
    test_auc = roc_auc_score(y_test, test_pred)

    print(f"val_auc={val_auc:.6f} val_gini={gini_from_auc(val_auc):.6f}")
    print(f"test_auc={test_auc:.6f} test_gini={gini_from_auc(test_auc):.6f}")
    print(f"n_features={len(model_vars)}")

    joblib.dump(model, args.out)
    with open(args.features_out, "w", encoding="utf-8") as f:
        json.dump({"features": model_vars}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
