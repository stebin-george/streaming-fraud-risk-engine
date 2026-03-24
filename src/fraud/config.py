import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_model")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "/app/artifacts/model.joblib")
MODEL_SELECTION_METRIC = os.getenv("MODEL_SELECTION_METRIC", "avg_precision")
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.55"))

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "redpanda:9092")
KAFKA_TOPIC_TX = os.getenv("KAFKA_TOPIC_TX", "transactions")
KAFKA_TOPIC_SCORED = os.getenv("KAFKA_TOPIC_SCORED", "transactions_scored")
KAFKA_TOPIC_ALERTS = os.getenv("KAFKA_TOPIC_ALERTS", "transactions_alerts")
KAFKA_TOPIC_DLQ = os.getenv("KAFKA_TOPIC_DLQ", "transactions_dlq")
API_URL = os.getenv("API_URL", "http://api:8000/predict")
STREAM_SCORE_TIMEOUT_S = float(os.getenv("STREAM_SCORE_TIMEOUT_S", "1.5"))
STREAM_SCORE_RETRIES = int(os.getenv("STREAM_SCORE_RETRIES", "3"))
STREAM_RETRY_BACKOFF_S = float(os.getenv("STREAM_RETRY_BACKOFF_S", "0.25"))
STREAM_ALERT_THRESHOLD = float(os.getenv("STREAM_ALERT_THRESHOLD", "0.8"))
