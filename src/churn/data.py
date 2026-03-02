import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal, real cleaning step (deduplicate + trim strings)."""
    out = df.copy()
    out = out.drop_duplicates()
    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].astype(str).str.strip()
    return out
