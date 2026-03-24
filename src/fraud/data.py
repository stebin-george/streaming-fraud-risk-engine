from datetime import UTC, datetime
from typing import Optional
import uuid

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class Transaction(BaseModel):
    transaction_id: str
    user_id: int
    merchant_id: int
    amount: float
    timestamp: float  # epoch seconds
    device_id: int
    channel: str
    lat: float
    lon: float
    country: str
    merchant_category: str
    entry_mode: str
    card_present: bool
    hours_since_last_txn: float
    num_txn_24h: int
    velocity_1h: int
    velocity_10m: int
    avg_amount_30d: float
    amount_vs_avg: float
    distance_from_home_km: float
    distance_from_last_txn_km: float
    account_age_days: int
    merchant_risk_score: float
    is_foreign: bool
    high_risk_country: bool
    is_night: bool
    new_device: bool
    travel_mismatch: bool
    # for training only
    is_fraud: Optional[int] = Field(default=None)


CHANNELS = ["POS", "WEB", "APP"]
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "IN", "CN", "NG", "BR", "MX"]
MCCS = ["grocery", "electronics", "fashion", "fuel", "travel"]
ENTRY = ["chip", "swipe", "manual", "online"]
HIGH_RISK = {"NG", "CN", "BR"}
COUNTRY_PROBS = [0.44, 0.18, 0.08, 0.06, 0.05, 0.06, 0.04, 0.03, 0.03, 0.03]
HOME_COUNTRY_PROBS = [0.62, 0.12, 0.07, 0.05, 0.04, 0.04, 0.02, 0.01, 0.01, 0.02]
MCC_PROBS = [0.34, 0.19, 0.16, 0.18, 0.13]
CHANNEL_PROBS = [0.55, 0.27, 0.18]
ENTRY_PROBS = [0.48, 0.24, 0.08, 0.20]
MERCHANT_RISK_BASE = {
    "grocery": 0.06,
    "electronics": 0.65,
    "fashion": 0.28,
    "fuel": 0.14,
    "travel": 0.72,
}
RNG = np.random.default_rng(7)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _fraud_logit(df: pd.DataFrame) -> np.ndarray:
    logit = np.full(len(df), -4.35)
    logit += 0.85 * (df["amount_vs_avg"] > 3.2)
    logit += 0.55 * (df["amount"] > 850)
    logit += 0.70 * (df["velocity_10m"] >= 3)
    logit += 0.45 * (df["velocity_1h"] >= 7)
    logit += 0.62 * (df["distance_from_last_txn_km"] > 500)
    logit += 0.35 * (df["distance_from_home_km"] > 250)
    logit += 0.40 * df["is_foreign"]
    logit += 0.55 * df["high_risk_country"]
    logit += 0.60 * df["travel_mismatch"]
    logit += 0.52 * df["new_device"]
    logit += 0.32 * df["is_night"]
    logit += 0.38 * (df["account_age_days"] < 30)
    logit += 0.48 * (df["merchant_risk_score"] > 0.55)
    logit += 0.30 * df["entry_mode"].isin(["manual", "online"])
    logit += 0.28 * (~df["card_present"])
    logit += 0.16 * (df["merchant_category"].isin(["electronics", "travel"]))
    logit += 0.20 * ((df["channel"] == "APP") & df["new_device"])
    logit -= 0.18 * ((df["channel"] == "POS") & df["card_present"])
    logit -= 0.12 * (df["account_age_days"] > 365)
    return logit


def synth_transactions(n=10000, with_labels=True, start_ts=None) -> pd.DataFrame:
    now = datetime.now(UTC).timestamp() if start_ts is None else start_ts
    user_id = RNG.integers(1, 5000, size=n)
    merchant_category = RNG.choice(MCCS, size=n, p=MCC_PROBS)
    timestamp = now + RNG.uniform(0, 3600 * 24, size=n)
    hour_of_day = ((timestamp // 3600) % 24).astype(int)
    avg_amount_30d = RNG.gamma(shape=2.5, scale=55.0, size=n).clip(20, 900)
    amount = (avg_amount_30d * RNG.lognormal(mean=0.0, sigma=0.7, size=n)).clip(2, 5000)
    home_country = RNG.choice(COUNTRIES, size=n, p=HOME_COUNTRY_PROBS)
    is_foreign = RNG.random(n) < 0.18
    country = np.where(
        is_foreign,
        RNG.choice(COUNTRIES, size=n, p=COUNTRY_PROBS),
        home_country,
    )
    same_country = country == home_country
    while same_country[is_foreign].any():
        country[is_foreign & same_country] = RNG.choice(
            COUNTRIES,
            size=int((is_foreign & same_country).sum()),
            p=COUNTRY_PROBS,
        )
        same_country = country == home_country

    high_risk_country = pd.Series(country).isin(HIGH_RISK).to_numpy()
    num_txn_24h = RNG.poisson(3.5, size=n).clip(0, 45)
    velocity_1h = np.minimum(
        num_txn_24h,
        RNG.poisson(1.5 + 0.18 * num_txn_24h, size=n),
    ).clip(0, 12)
    velocity_10m = np.minimum(
        velocity_1h,
        RNG.binomial(np.maximum(velocity_1h, 1), 0.28, size=n),
    )
    hours_since_last_txn = RNG.exponential(scale=10.0, size=n).clip(0.01, 96)
    account_age_days = RNG.integers(1, 3650, size=n)
    new_device = RNG.random(n) < np.where(account_age_days < 45, 0.22, 0.07)
    is_night = (hour_of_day <= 5) | (hour_of_day >= 23)
    travel_mismatch = is_foreign & (hours_since_last_txn < 2.5) & (RNG.random(n) < 0.35)
    distance_from_home_km = np.where(
        is_foreign,
        RNG.normal(1600, 900, size=n),
        RNG.normal(35, 40, size=n),
    ).clip(0, 6000)
    distance_from_last_txn_km = np.where(
        travel_mismatch,
        RNG.normal(1800, 950, size=n),
        np.where(new_device, RNG.normal(180, 130, size=n), RNG.normal(28, 35, size=n)),
    ).clip(0, 8000)

    merchant_risk_score = np.array(
        [MERCHANT_RISK_BASE[mcc] for mcc in merchant_category]
    ) + RNG.normal(0.0, 0.08, size=n)
    merchant_risk_score = merchant_risk_score.clip(0.01, 0.99)

    lat = RNG.uniform(-55, 55, size=n)
    lon = RNG.uniform(-120, 120, size=n)
    df = pd.DataFrame(
        {
            "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
            "user_id": user_id,
            "merchant_id": RNG.integers(1, 1500, size=n),
            "amount": amount.round(2),
            "timestamp": timestamp,
            "device_id": RNG.integers(1, 6000, size=n),
            "channel": RNG.choice(CHANNELS, size=n, p=CHANNEL_PROBS),
            "lat": lat,
            "lon": lon,
            "country": country,
            "merchant_category": merchant_category,
            "entry_mode": RNG.choice(ENTRY, size=n, p=ENTRY_PROBS),
            "card_present": RNG.random(n) < 0.66,
            "hours_since_last_txn": hours_since_last_txn.round(3),
            "num_txn_24h": num_txn_24h,
            "velocity_1h": velocity_1h,
            "velocity_10m": velocity_10m,
            "avg_amount_30d": avg_amount_30d.round(2),
            "amount_vs_avg": (amount / avg_amount_30d).round(3),
            "distance_from_home_km": distance_from_home_km.round(1),
            "distance_from_last_txn_km": distance_from_last_txn_km.round(1),
            "account_age_days": account_age_days,
            "merchant_risk_score": merchant_risk_score.round(3),
            "is_foreign": is_foreign,
            "high_risk_country": high_risk_country,
            "is_night": is_night,
            "new_device": new_device,
            "travel_mismatch": travel_mismatch,
        }
    )
    if with_labels:
        prob = _sigmoid(_fraud_logit(df))
        df["is_fraud"] = RNG.binomial(1, prob)
    return df
