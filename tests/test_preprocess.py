import pandas as pd
from src.data.preprocess import engineer_time_features

def test_engineer_time_features_adds_columns():
    df = pd.DataFrame({"Time": [0, 3600, 7200]})
    out = engineer_time_features(df)
    for c in ["hour_of_day", "day_of_week", "is_weekend"]:
        assert c in out.columns
