import pandas as pd

def test_target_exists():
    df = pd.DataFrame({"Target": [0, 1, 0]})
    assert "Target" in df.columns
