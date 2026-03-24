from fraud import streaming


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_score_transaction_retries_then_succeeds(monkeypatch):
    calls = {"count": 0}

    def flaky_post(url, json, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary failure")
        return DummyResponse({"transaction_id": json["transaction_id"], "fraud_probability": 0.91})

    monkeypatch.setattr(streaming, "STREAM_SCORE_RETRIES", 2)
    monkeypatch.setattr(streaming, "STREAM_RETRY_BACKOFF_S", 0.0)
    scored = streaming.score_transaction_with_retries({"transaction_id": "tx-1"}, post_fn=flaky_post)
    assert scored["transaction_id"] == "tx-1"
    assert scored["attempts_used"] == 2


def test_build_failure_event_includes_original_payload():
    failure = streaming.build_failure_event({"transaction_id": "tx-2", "amount": 120.0}, RuntimeError("boom"))
    assert failure["transaction_id"] == "tx-2"
    assert failure["payload"]["amount"] == 120.0
    assert "boom" in failure["failure_reason"]
