from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os
from fraud.features import ensure_dataframe
from fraud.config import FRAUD_THRESHOLD, MODEL_LOCAL_PATH

_bundle = None  # cached model bundle


class TxIn(BaseModel):
    transaction_id: str
    user_id: int
    merchant_id: int
    amount: float
    timestamp: float
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


def _model_path() -> str:
    # env var set by the test; fall back to config
    return os.getenv("MODEL_LOCAL_PATH", MODEL_LOCAL_PATH)


def _default_metadata(model):
    clf = model.named_steps.get("clf") if hasattr(model, "named_steps") else model
    return {
        "selected_model": clf.__class__.__name__,
        "threshold": float(FRAUD_THRESHOLD),
        "trained_at": None,
        "metrics": {},
        "leaderboard": [],
    }


def get_model_bundle():
    """Lazy-load the bundle so requests work even if startup didn’t run."""
    global _bundle
    if _bundle is None:
        path = _model_path()
        loaded = joblib.load(path)
        if isinstance(loaded, dict) and "model" in loaded:
            metadata = loaded.get("metadata", {})
            metadata.setdefault("threshold", float(FRAUD_THRESHOLD))
            _bundle = {"model": loaded["model"], "metadata": metadata}
        else:
            _bundle = {"model": loaded, "metadata": _default_metadata(loaded)}
    return _bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the model if possible; don't crash if MODEL_LOCAL_PATH isn't set
    try:
        get_model_bundle()
    except Exception:
        pass
    yield


app = FastAPI(title="Fraud Scoring API", lifespan=lifespan)


@app.get("/health")
def health():
    try:
        bundle = get_model_bundle()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    return {
        "status": "ok",
        "model_loaded": bundle["model"] is not None,
        "model_path": _model_path(),
    }


@app.get("/model-info")
def model_info():
    try:
        bundle = get_model_bundle()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    metadata = bundle["metadata"]
    return {
        "model_path": _model_path(),
        "selected_model": metadata.get("selected_model"),
        "threshold": float(metadata.get("threshold", FRAUD_THRESHOLD)),
        "trained_at": metadata.get("trained_at"),
        "metrics": metadata.get("metrics", {}),
        "leaderboard": metadata.get("leaderboard", []),
    }


@app.post("/predict")
def predict(tx: TxIn):
    df = ensure_dataframe(tx.model_dump())
    try:
        bundle = get_model_bundle()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model not available: {e}")
    model = bundle["model"]
    metadata = bundle["metadata"]
    threshold = float(metadata.get("threshold", FRAUD_THRESHOLD))
    p = float(model.predict_proba(df)[0, 1])
    if p >= 0.85:
        risk_band = "high"
    elif p >= threshold:
        risk_band = "medium"
    else:
        risk_band = "low"
    return {
        "transaction_id": tx.transaction_id,
        "fraud_probability": p,
        "is_fraud": p >= threshold,
        "risk_band": risk_band,
        "threshold": threshold,
        "selected_model": metadata.get("selected_model"),
        "trained_at": metadata.get("trained_at"),
    }
