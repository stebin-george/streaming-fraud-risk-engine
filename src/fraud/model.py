from datetime import UTC, datetime

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import FRAUD_THRESHOLD, MODEL_LOCAL_PATH, MODEL_SELECTION_METRIC
from .features import build_pipeline


def _candidate_estimators():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=7,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=350,
            random_state=7,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }


def _evaluate_candidate(y_true, proba):
    pred = (proba >= FRAUD_THRESHOLD).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "avg_precision": float(average_precision_score(y_true, proba)),
        "positive_rate": float(pred.mean()),
        "threshold": float(FRAUD_THRESHOLD),
    }


def train_and_log(train_df: pd.DataFrame, mlflow_tracking_uri: str, model_name: str):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    X = train_df.drop(columns=["is_fraud"])
    y = train_df["is_fraud"]
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=7,
    )

    leaderboard = []
    best = None

    for candidate_name, estimator in _candidate_estimators().items():
        pipe = build_pipeline(estimator)
        with mlflow.start_run(run_name=f"benchmark_{candidate_name}") as run:
            pipe.fit(Xtr, ytr)
            proba = pipe.predict_proba(Xte)[:, 1]
            metrics = _evaluate_candidate(yte, proba)
            mlflow.log_param("candidate_name", candidate_name)
            mlflow.log_param("selection_metric", MODEL_SELECTION_METRIC)
            mlflow.log_param("n_train_rows", int(len(Xtr)))
            mlflow.log_param("n_test_rows", int(len(Xte)))
            mlflow.log_params(
                {
                    f"estimator__{key}": value
                    for key, value in estimator.get_params().items()
                    if isinstance(value, (str, int, float, bool))
                }
            )
            mlflow.log_metrics(metrics)

            entry = {
                "candidate_name": candidate_name,
                "run_id": run.info.run_id,
                "metrics": metrics,
                "pipeline": pipe,
            }
            leaderboard.append(entry)

            score = metrics.get(MODEL_SELECTION_METRIC, metrics["avg_precision"])
            best_score = None if best is None else best["metrics"].get(
                MODEL_SELECTION_METRIC,
                best["metrics"]["avg_precision"],
            )
            if (
                best is None
                or score > best_score
                or (
                    score == best_score
                    and metrics["roc_auc"] > best["metrics"]["roc_auc"]
                )
            ):
                best = entry

    best_pipe = best["pipeline"]
    best_metrics = best["metrics"]
    trained_at = datetime.now(UTC).isoformat()
    metadata = {
        "selected_model": best["candidate_name"],
        "selection_metric": MODEL_SELECTION_METRIC,
        "threshold": float(FRAUD_THRESHOLD),
        "trained_at": trained_at,
        "feature_count": int(X.shape[1]),
        "metrics": best_metrics,
        "leaderboard": [
            {
                "candidate_name": row["candidate_name"],
                "run_id": row["run_id"],
                "metrics": row["metrics"],
            }
            for row in sorted(
                leaderboard,
                key=lambda row: (
                    row["metrics"].get(MODEL_SELECTION_METRIC, row["metrics"]["avg_precision"]),
                    row["metrics"]["roc_auc"],
                ),
                reverse=True,
            )
        ],
    }

    with mlflow.start_run(run_name="selected_best_model") as run:
        mlflow.log_param("selected_model", best["candidate_name"])
        mlflow.log_param("selection_metric", MODEL_SELECTION_METRIC)
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(
            best_pipe,
            "model",
            registered_model_name=model_name,
        )
        info = {
            "run_id": run.info.run_id,
            "model_uri": f"runs:/{run.info.run_id}/model",
            "selected_model": best["candidate_name"],
            "metrics": best_metrics,
            "leaderboard": metadata["leaderboard"],
        }

    bundle = {
        "model": best_pipe,
        "metadata": metadata,
    }
    joblib.dump(bundle, MODEL_LOCAL_PATH)
    return info
