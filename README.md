# Real-time Fraud Detection System

An end-to-end, containerized fraud detection project that now goes beyond a single baseline model. It simulates richer transaction behavior, benchmarks multiple sklearn models, serves the selected model through FastAPI, and scores a live Kafka-compatible stream with alerting and dead-letter handling.

## What the system does

- Generates synthetic card transactions with behavior-driven fraud signals
- Trains and benchmarks multiple sklearn classifiers with MLflow tracking
- Saves the best model locally for online inference
- Serves predictions through a FastAPI microservice
- Consumes raw events from Redpanda, scores them, emits alerts, and routes failures to a DLQ
- Includes tests and CI for core data, API, and streaming behavior

## Architecture

```text
generator -> transactions topic -> stream worker -> FastAPI /predict
                                           |-> transactions_scored
                                           |-> transactions_alerts
                                           |-> transactions_dlq

trainer -> synthetic labeled data -> benchmark models -> MLflow + model artifact
```

## What makes this version stronger

- Richer fraud simulation with velocity, device novelty, travel mismatch, account age, merchant risk, and amount-vs-history features
- Multi-model benchmark path using Logistic Regression, Random Forest, and Extra Trees
- Operational API endpoints for `/health` and `/model-info`
- More resilient stream scoring with retries, alert routing, and dead-letter events

## Quickstart

```bash
docker compose up --build
# MLflow UI: http://localhost:5000
# API docs:  http://localhost:8000/docs
```

To retrain and re-run the benchmark:

```bash
docker compose run --rm trainer
```

## API

- `GET /health` -> confirms the model artifact is available
- `GET /model-info` -> returns the loaded model name, threshold, metrics, and leaderboard
- `POST /predict` -> returns `transaction_id`, `fraud_probability`, `is_fraud`, `risk_band`, and model metadata

## Design choices

- Redpanda provides a lightweight Kafka-compatible local broker
- Sklearn pipelines keep preprocessing and inference behavior aligned
- MLflow records candidate runs and the final selected model
- The API serves a local artifact for simple local deployment while exposing model metadata

## Testing

```bash
pytest
```

## Environment variables

See `src/fraud/config.py` for defaults, including fraud threshold, alert threshold, retry counts, and topic names.
