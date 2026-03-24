import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

NUMERIC = [
    "amount",
    "lat",
    "lon",
    "hours_since_last_txn",
    "num_txn_24h",
    "velocity_1h",
    "velocity_10m",
    "avg_amount_30d",
    "amount_vs_avg",
    "distance_from_home_km",
    "distance_from_last_txn_km",
    "account_age_days",
    "merchant_risk_score",
]
CATEGORICAL = [
    "channel",
    "country",
    "merchant_category",
    "entry_mode",
    "card_present",
    "is_foreign",
    "high_risk_country",
    "is_night",
    "new_device",
    "travel_mismatch",
]

def make_preprocessor():
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num = StandardScaler()
    return ColumnTransformer(
        transformers=[
            ("num", num, NUMERIC),
            ("cat", cat, CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

def build_pipeline(estimator):
    pre = make_preprocessor()
    pipe = Pipeline(steps=[("prep", pre), ("clf", estimator)])
    return pipe

def ensure_dataframe(x):
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame([x]) if isinstance(x, dict) else pd.DataFrame(x)
