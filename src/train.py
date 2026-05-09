from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    LATEST_EXPERIMENT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PR_CURVE_PATH,
    RANDOM_STATE,
    SCORED_DATA_PATH,
    TARGET_COLUMN,
    TEMPORAL_METRICS_PATH,
    THRESHOLD_PATH,
    ensure_project_dirs,
)
from .data_prep import build_feature_frame, load_transactions, split_train_validation_test
from .evaluate import classification_metrics, find_best_threshold, precision_recall_frame
from .features import add_risk_bucket


def try_build_xgboost(scale_pos_weight: float):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    return XGBClassifier(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def supervised_models(scale_pos_weight: float) -> dict[str, Pipeline]:
    models = {
        "logistic_regression_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1500,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=180,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    xgb_model = try_build_xgboost(scale_pos_weight)
    if xgb_model is not None:
        models["xgboost_weighted"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", xgb_model),
            ]
        )

    return models


def normalize_from_validation(valid_scores, target_scores) -> np.ndarray:
    min_score = float(np.min(valid_scores))
    max_score = float(np.max(valid_scores))
    if max_score == min_score:
        return np.zeros_like(target_scores, dtype=float)
    return np.clip((target_scores - min_score) / (max_score - min_score), 0, 1)


def evaluate_isolation_forest(x_train, x_valid, x_test, y_train, y_valid, y_test) -> dict:
    normal_train = x_train[y_train == 0]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    contamination=max(float(y_train.mean()), 0.001),
                    n_estimators=220,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(normal_train)

    valid_raw = -model.decision_function(x_valid)
    test_raw = -model.decision_function(x_test)
    valid_scores = normalize_from_validation(valid_raw, valid_raw)
    test_scores = normalize_from_validation(valid_raw, test_raw)
    best_threshold, _ = find_best_threshold(y_valid, valid_scores)

    return {
        "name": "isolation_forest_anomaly",
        "model": model,
        "best_threshold": best_threshold,
        "validation": classification_metrics(y_valid, valid_scores, best_threshold),
        "test": classification_metrics(y_test, test_scores, best_threshold),
    }


def temporal_split(df: pd.DataFrame, train_size: float = 0.60, validation_size: float = 0.20):
    sort_column = "Time" if "Time" in df.columns else None
    ordered = df.sort_values(sort_column).reset_index(drop=True) if sort_column else df.reset_index(drop=True)

    train_end = int(len(ordered) * train_size)
    valid_end = int(len(ordered) * (train_size + validation_size))

    train_df = ordered.iloc[:train_end].copy()
    valid_df = ordered.iloc[train_end:valid_end].copy()
    test_df = ordered.iloc[valid_end:].copy()

    feature_names = list(build_feature_frame(train_df).columns)
    x_train = build_feature_frame(train_df, expected_columns=feature_names)
    x_valid = build_feature_frame(valid_df, expected_columns=feature_names)
    x_test = build_feature_frame(test_df, expected_columns=feature_names)

    return (
        x_train,
        x_valid,
        x_test,
        train_df[TARGET_COLUMN],
        valid_df[TARGET_COLUMN],
        test_df[TARGET_COLUMN],
    )


def evaluate_temporal_models(df: pd.DataFrame, source: str) -> pd.DataFrame:
    x_train, x_valid, x_test, y_train, y_valid, y_test = temporal_split(df)
    positive_count = max(int(y_train.sum()), 1)
    negative_count = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = negative_count / positive_count

    rows = []
    for name, model in supervised_models(scale_pos_weight).items():
        model.fit(x_train, y_train)
        valid_scores = model.predict_proba(x_valid)[:, 1]
        test_scores = model.predict_proba(x_test)[:, 1]
        best_threshold, _ = find_best_threshold(y_valid, valid_scores)
        validation_metrics = classification_metrics(y_valid, valid_scores, best_threshold)
        test_metrics = classification_metrics(y_test, test_scores, best_threshold)

        rows.append(
            {
                "model": name,
                "dataset_source": source,
                "split": "temporal",
                "validation_f2": validation_metrics["f2"],
                "validation_pr_auc": validation_metrics["pr_auc"],
                "test_f2": test_metrics["f2"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_recall": test_metrics["recall"],
                "test_precision": test_metrics["precision"],
                "threshold": best_threshold,
                "alerts": test_metrics["alerts"],
                "false_negatives": test_metrics["false_negatives"],
                "false_positives": test_metrics["false_positives"],
            }
        )

    temporal_metrics = pd.DataFrame(rows).sort_values(
        ["validation_f2", "validation_pr_auc"],
        ascending=False,
    )
    temporal_metrics.to_csv(TEMPORAL_METRICS_PATH, index=False)
    return temporal_metrics


def write_experiment_log(
    df: pd.DataFrame,
    source: str,
    metrics: pd.DataFrame,
    temporal_metrics: pd.DataFrame,
    threshold_config: dict,
    sample_size: int | None,
) -> None:
    experiment = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_source": source,
        "sample_size": sample_size,
        "rows": int(len(df)),
        "frauds": int(df[TARGET_COLUMN].sum()),
        "fraud_rate": float(df[TARGET_COLUMN].mean()),
        "best_model": threshold_config["model_name"],
        "selected_threshold": float(threshold_config["threshold"]),
        "holdout_test": threshold_config["test"],
        "metrics": metrics.to_dict(orient="records"),
        "temporal_metrics": temporal_metrics.to_dict(orient="records"),
        "artifacts": {
            "model": str(MODEL_PATH),
            "scored_data": str(SCORED_DATA_PATH),
            "holdout_metrics": str(METRICS_PATH),
            "temporal_metrics": str(TEMPORAL_METRICS_PATH),
        },
    }
    LATEST_EXPERIMENT_PATH.write_text(json.dumps(experiment, indent=2), encoding="utf-8")


def train_all(sample_size: int | None = None) -> dict:
    ensure_project_dirs()
    df, source = load_transactions()

    if sample_size and len(df) > sample_size:
        df = (
            df.groupby(TARGET_COLUMN, group_keys=False)
            .apply(
                lambda group: group.sample(
                    min(len(group), max(1, int(sample_size * len(group) / len(df)))),
                    random_state=RANDOM_STATE,
                )
            )
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

    x_train, x_valid, x_test, y_train, y_valid, y_test = split_train_validation_test(df)
    feature_names = list(x_train.columns)
    positive_count = max(int(y_train.sum()), 1)
    negative_count = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = negative_count / positive_count

    rows = []
    fitted_models = []
    for name, model in supervised_models(scale_pos_weight).items():
        model.fit(x_train, y_train)
        valid_scores = model.predict_proba(x_valid)[:, 1]
        test_scores = model.predict_proba(x_test)[:, 1]
        best_threshold, threshold_curve = find_best_threshold(y_valid, valid_scores)

        validation_metrics = classification_metrics(y_valid, valid_scores, best_threshold)
        test_metrics = classification_metrics(y_test, test_scores, best_threshold)

        rows.append(
            {
                "model": name,
                "dataset_source": source,
                "validation_f2": validation_metrics["f2"],
                "validation_pr_auc": validation_metrics["pr_auc"],
                "test_f2": test_metrics["f2"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_recall": test_metrics["recall"],
                "test_precision": test_metrics["precision"],
                "threshold": best_threshold,
                "alerts": test_metrics["alerts"],
                "false_negatives": test_metrics["false_negatives"],
                "false_positives": test_metrics["false_positives"],
            }
        )
        fitted_models.append((name, model, best_threshold, validation_metrics, test_metrics))

    anomaly_result = evaluate_isolation_forest(x_train, x_valid, x_test, y_train, y_valid, y_test)
    rows.append(
        {
            "model": anomaly_result["name"],
            "dataset_source": source,
            "validation_f2": anomaly_result["validation"]["f2"],
            "validation_pr_auc": anomaly_result["validation"]["pr_auc"],
            "test_f2": anomaly_result["test"]["f2"],
            "test_pr_auc": anomaly_result["test"]["pr_auc"],
            "test_recall": anomaly_result["test"]["recall"],
            "test_precision": anomaly_result["test"]["precision"],
            "threshold": anomaly_result["best_threshold"],
            "alerts": anomaly_result["test"]["alerts"],
            "false_negatives": anomaly_result["test"]["false_negatives"],
            "false_positives": anomaly_result["test"]["false_positives"],
        }
    )

    metrics = pd.DataFrame(rows).sort_values(
        ["validation_f2", "validation_pr_auc"],
        ascending=False,
    )
    best_name = metrics.iloc[0]["model"]

    best_tuple = next(item for item in fitted_models if item[0] == best_name)
    best_model_name, best_model, best_threshold, validation_metrics, test_metrics = best_tuple

    full_features = build_feature_frame(df, expected_columns=feature_names)
    full_scores = best_model.predict_proba(full_features)[:, 1]
    scored = df.copy()
    scored["risk_score"] = full_scores
    scored["prediction"] = (scored["risk_score"] >= best_threshold).astype(int)
    scored = add_risk_bucket(scored)

    metrics.to_csv(METRICS_PATH, index=False)
    scored.to_csv(SCORED_DATA_PATH, index=False)

    pr_curve = precision_recall_frame(y_test, best_model.predict_proba(x_test)[:, 1])
    pr_curve.to_csv(PR_CURVE_PATH, index=False)

    artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "threshold": best_threshold,
        "feature_names": feature_names,
        "dataset_source": source,
    }
    joblib.dump(artifact, MODEL_PATH)

    threshold_config = {
        "model_name": best_model_name,
        "threshold": best_threshold,
        "dataset_source": source,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    THRESHOLD_PATH.write_text(json.dumps(threshold_config, indent=2), encoding="utf-8")

    temporal_metrics = evaluate_temporal_models(df, source)
    write_experiment_log(
        df=df,
        source=source,
        metrics=metrics,
        temporal_metrics=temporal_metrics,
        threshold_config=threshold_config,
        sample_size=sample_size,
    )

    return threshold_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina modelos de deteccao de fraude.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Opcional: limita a quantidade de linhas para um treino rapido.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_all(sample_size=args.sample_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
