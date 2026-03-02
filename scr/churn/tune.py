import argparse
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from churn.config import SEED
from churn.utils import set_seeds
from churn.data import load_csv, basic_clean
from churn.feature_selection import select_features_by_corr

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--target", default="Target")
    p.add_argument("--date_col", default="Calculation_Date")
    p.add_argument("--test_date", required=True)
    p.add_argument("--out", default="artifacts/best_params.csv")
    args = p.parse_args()

    set_seeds(SEED)

    df = basic_clean(load_csv(args.data))
    df[args.date_col] = df[args.date_col].astype(str)

    test_df = df[df[args.date_col] == str(args.test_date)].copy()
    train_df = df[df[args.date_col] != str(args.test_date)].copy()

    # create a validation split (similar to notebook)
    X_train, X_val, y_train, y_val = train_test_split(
        train_df, train_df[args.target].astype(int),
        test_size=0.2, random_state=SEED, stratify=train_df[args.target].astype(int)
    )

    model_vars = select_features_by_corr(
        train_df,
        target_col=args.target,
        date_col=args.date_col,
        corr_target_threshold=0.01,
        pair_corr_threshold=0.8,
        snapshot_date=None,
        non_model_vars=[args.target, "ClientID", "ClientName", "PhoneMobile"]
    )

    # notebook-like grid (trimmed slightly to keep runtime reasonable)
    parameters = {
        "base_score": [float(y_val.mean())],
        "colsample_bynode": [0.8, 1.0],
        "colsample_bylevel": [0.8, 1.0],
        "reg_alpha": [1.5, 5.0],
        "reg_lambda": [10.0, 100.0],
        "min_child_weight": [0.1, 20.0],
        "gamma": [5.0, 20.0],
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.5],
        "n_estimators": [200, 500],
        "subsample": [0.8, 1.0],
        "tree_method": ["exact"],
        "sampling_method": ["uniform"],
        "n_jobs": [-1],
        "use_label_encoder": [False],
        "disable_default_eval_metric": [True],
        "random_state": [0],
    }

    base = xgb.XGBClassifier(seed=123)
    gs = GridSearchCV(base, param_grid=parameters, scoring="roc_auc", cv=3, verbose=1, n_jobs=-1, return_train_score=True)
    gs.fit(X_val[model_vars].head(50000), y_val.head(50000))

    best = pd.DataFrame(gs.best_params_.items(), columns=["arg", "value"])
    best.to_csv(args.out, index=False)
    print("best_params_saved_to", args.out)
    print(gs.best_params_)

if __name__ == "__main__":
    main()
