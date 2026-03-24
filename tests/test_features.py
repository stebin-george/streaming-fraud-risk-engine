from fraud.data import synth_transactions
from fraud.features import make_preprocessor
import numpy as np


def test_preprocessor_shapes():
    df = synth_transactions(n=50, with_labels=True)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].values
    pre = make_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
    assert np.isfinite(Xt).all()
    assert set(np.unique(y)).issubset({0,1})


def test_synth_transactions_include_richer_behavior_features():
    df = synth_transactions(n=200, with_labels=True)
    expected = {
        "velocity_1h",
        "velocity_10m",
        "avg_amount_30d",
        "amount_vs_avg",
        "distance_from_home_km",
        "distance_from_last_txn_km",
        "account_age_days",
        "merchant_risk_score",
        "is_night",
        "new_device",
        "travel_mismatch",
        "is_fraud",
    }
    assert expected.issubset(df.columns)
    assert df["amount_vs_avg"].gt(0).all()
    assert df["merchant_risk_score"].between(0, 1).all()
    assert 0.01 < df["is_fraud"].mean() < 0.35
