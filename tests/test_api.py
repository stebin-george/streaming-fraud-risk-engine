import joblib
from fastapi.testclient import TestClient
from fraud.api import main as api_main
from fraud.data import synth_transactions


def test_api_predict_and_model_metadata(tmp_path, monkeypatch):
    from sklearn.ensemble import RandomForestClassifier
    from fraud.features import build_pipeline

    df = synth_transactions(n=200, with_labels=True)
    X, y = df.drop(columns=["is_fraud"]), df["is_fraud"]
    pipe = build_pipeline(RandomForestClassifier(n_estimators=50, random_state=0))
    pipe.fit(X, y)
    path = tmp_path/"model.joblib"
    joblib.dump(
        {
            "model": pipe,
            "metadata": {
                "selected_model": "random_forest",
                "threshold": 0.55,
                "trained_at": "2026-03-23T00:00:00+00:00",
                "metrics": {"avg_precision": 0.7, "roc_auc": 0.8},
                "leaderboard": [{"candidate_name": "random_forest", "metrics": {"avg_precision": 0.7}}],
            },
        },
        path,
    )
    monkeypatch.setenv("MODEL_LOCAL_PATH", str(path))
    api_main._bundle = None

    client = TestClient(api_main.app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["selected_model"] == "random_forest"

    tx = synth_transactions(n=1, with_labels=False).to_dict(orient="records")[0]
    resp = client.post("/predict", json=tx)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["selected_model"] == "random_forest"
    assert body["threshold"] == 0.55
    assert body["risk_band"] in {"low", "medium", "high"}
