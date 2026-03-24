import json
import time

import requests
from kafka import KafkaConsumer, KafkaProducer

from .config import (
    API_URL,
    KAFKA_BOOTSTRAP,
    KAFKA_TOPIC_ALERTS,
    KAFKA_TOPIC_DLQ,
    KAFKA_TOPIC_SCORED,
    KAFKA_TOPIC_TX,
    STREAM_ALERT_THRESHOLD,
    STREAM_RETRY_BACKOFF_S,
    STREAM_SCORE_RETRIES,
    STREAM_SCORE_TIMEOUT_S,
)


def score_transaction_with_retries(tx, post_fn=requests.post):
    last_error = None
    for attempt in range(1, STREAM_SCORE_RETRIES + 1):
        try:
            response = post_fn(API_URL, json=tx, timeout=STREAM_SCORE_TIMEOUT_S)
            response.raise_for_status()
            scored = response.json()
            scored["attempts_used"] = attempt
            return scored
        except Exception as exc:
            last_error = exc
            if attempt < STREAM_SCORE_RETRIES:
                time.sleep(STREAM_RETRY_BACKOFF_S * attempt)
    raise last_error


def build_failure_event(tx, error):
    return {
        "transaction_id": tx.get("transaction_id"),
        "failure_reason": str(error),
        "failed_at": time.time(),
        "source_topic": KAFKA_TOPIC_TX,
        "api_url": API_URL,
        "payload": tx,
    }


def build_alert_event(scored):
    return {
        **scored,
        "alert_threshold": STREAM_ALERT_THRESHOLD,
        "alert_reason": "fraud_probability_above_threshold",
    }


def start_stream():
    consumer = KafkaConsumer(
        KAFKA_TOPIC_TX,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="fraud-consumer-1",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    for msg in consumer:
        tx = msg.value
        try:
            scored = score_transaction_with_retries(tx)
            scored["processed_ts"] = time.time()
            scored["source_topic"] = KAFKA_TOPIC_TX
            producer.send(KAFKA_TOPIC_SCORED, scored)
            if scored["fraud_probability"] >= STREAM_ALERT_THRESHOLD:
                alert_event = build_alert_event(scored)
                producer.send(KAFKA_TOPIC_ALERTS, alert_event)
                print("[ALERT] High-risk txn:", alert_event)
        except Exception as e:
            failure_event = build_failure_event(tx, e)
            producer.send(KAFKA_TOPIC_DLQ, failure_event)
            print("scoring failed:", failure_event)
