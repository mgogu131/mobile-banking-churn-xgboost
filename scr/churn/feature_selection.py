import numpy as np
import pandas as pd

def select_features_by_corr(
    df: pd.DataFrame,
    target_col: str,
    date_col: str | None = None,
    corr_target_threshold: float = 0.01,
    pair_corr_threshold: float = 0.8,
    snapshot_date: str | None = None,
    non_model_vars: list[str] | None = None
) -> list[str]:
    """Feature filtering adapted from the original notebook:
    1) keep columns with |corr(feature, target)| above a threshold
    2) drop one from each highly-correlated pair (keep the one with higher |corr with target|)
    """
    if non_model_vars is None:
        non_model_vars = [target_col]

    # numeric-only correlation with target
    num_df = df.select_dtypes(include=["number"]).copy()
    if target_col not in num_df.columns:
        raise ValueError(f"Target '{target_col}' must be numeric (0/1) for correlation filtering.")

    corrs = num_df.corr(numeric_only=True)[target_col].drop(labels=[target_col])
    vs = pd.DataFrame({"name": corrs.index, "corr_target": corrs.values})
    vs["corr_target_abs"] = vs["corr_target"].abs()

    candidate = vs[(vs.corr_target_abs > corr_target_threshold) & (~vs.name.isin(non_model_vars))]["name"].tolist()

    # pairwise correlation snapshot (optional)
    working = df
    if date_col and snapshot_date and date_col in df.columns:
        working = df[df[date_col].astype(str) == str(snapshot_date)]
        if len(working) == 0:
            working = df

    corr_matrix = working[candidate].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    pairs = upper.unstack().dropna()
    pairs = pairs[pairs > pair_corr_threshold].sort_values(ascending=False)

    to_drop = set()
    vs_map = vs.set_index("name")["corr_target_abs"].to_dict()
    while len(pairs) > 0:
        a, b = pairs.index[0]
        drop = b if vs_map.get(a, 0) >= vs_map.get(b, 0) else a
        to_drop.add(drop)
        pairs = pairs[pairs.index.map(lambda x: drop not in x)]

    final = sorted(list(set(candidate) - to_drop))
    return final
