from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from .config import (
    EXPLANATION_SUMMARY_PATH,
    FEATURE_IMPORTANCE_PATH,
    LOCAL_EXPLANATIONS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    SCORED_DATA_PATH,
    ensure_project_dirs,
)
from .data_prep import build_feature_frame, load_transactions


def model_steps(artifact: dict):
    pipeline = artifact["model"]
    estimator = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
    transformer = pipeline[:-1] if hasattr(pipeline, "steps") and len(pipeline.steps) > 1 else None
    return pipeline, transformer, estimator


def transformed_features(artifact: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    features = build_feature_frame(df, expected_columns=artifact["feature_names"])
    _, transformer, _ = model_steps(artifact)
    transformed = transformer.transform(features) if transformer is not None else features.to_numpy()
    return features, transformed


def extract_positive_shap_values(raw_values: object) -> np.ndarray:
    if isinstance(raw_values, list):
        return np.asarray(raw_values[1])

    values = np.asarray(raw_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def compute_shap_importance(artifact: dict, sample_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, str]:
    try:
        import shap

        features, transformed = transformed_features(artifact, sample_df)
        _, _, estimator = model_steps(artifact)
        explainer = shap.TreeExplainer(estimator)
        shap_values = extract_positive_shap_values(explainer.shap_values(transformed))
        importance = pd.DataFrame(
            {
                "feature": features.columns,
                "importance": np.abs(shap_values).mean(axis=0),
                "method": "mean_abs_shap",
            }
        ).sort_values("importance", ascending=False)
        return importance, shap_values, "shap"
    except Exception:
        _, _, estimator = model_steps(artifact)
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_)
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.asarray(estimator.coef_)).ravel()
        else:
            values = np.zeros(len(artifact["feature_names"]))

        importance = pd.DataFrame(
            {
                "feature": artifact["feature_names"],
                "importance": values,
                "method": "model_importance_fallback",
            }
        ).sort_values("importance", ascending=False)
        return importance, np.empty((0, len(artifact["feature_names"]))), "fallback"


def local_explanations(
    artifact: dict,
    raw_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    top_n_transactions: int,
    top_n_features: int,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    top_indexes = scored_df.sort_values("risk_score", ascending=False).head(top_n_transactions).index
    top_raw = raw_df.iloc[top_indexes].copy()
    features, _ = transformed_features(artifact, top_raw)

    if shap_values.shape[0] != len(top_raw):
        _, _, estimator = model_steps(artifact)
        if hasattr(estimator, "feature_importances_"):
            local_values = np.tile(np.asarray(estimator.feature_importances_), (len(top_raw), 1))
        else:
            local_values = np.zeros((len(top_raw), len(artifact["feature_names"])))
    else:
        local_values = shap_values

    rows = []
    for position, row_index in enumerate(top_indexes):
        ranking = np.argsort(np.abs(local_values[position]))[::-1][:top_n_features]
        for feature_position in ranking:
            feature = artifact["feature_names"][feature_position]
            rows.append(
                {
                    "row_index": int(row_index),
                    "risk_score": float(scored_df.loc[row_index, "risk_score"]),
                    "class": int(scored_df.loc[row_index, "Class"]),
                    "feature": feature,
                    "feature_value": float(features.iloc[position][feature]),
                    "contribution": float(local_values[position][feature_position]),
                }
            )
    return pd.DataFrame(rows)


def generate_explanations(sample_size: int = 1200, top_n_transactions: int = 8, top_n_features: int = 8) -> dict:
    ensure_project_dirs()
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modelo não encontrado. Execute `python -m src.train` antes.")
    if not SCORED_DATA_PATH.exists():
        raise FileNotFoundError("Dados pontuados não encontrados. Execute `python -m src.train` antes.")

    artifact = joblib.load(MODEL_PATH)
    raw_df, source = load_transactions()
    scored_df = pd.read_csv(SCORED_DATA_PATH)

    sample_df = raw_df.sample(min(len(raw_df), sample_size), random_state=RANDOM_STATE)
    importance, _, method = compute_shap_importance(artifact, sample_df)

    top_raw = raw_df.iloc[scored_df.sort_values("risk_score", ascending=False).head(top_n_transactions).index]
    _, top_transformed = transformed_features(artifact, top_raw)
    try:
        import shap

        _, _, estimator = model_steps(artifact)
        explainer = shap.TreeExplainer(estimator)
        top_shap_values = extract_positive_shap_values(explainer.shap_values(top_transformed))
        local_method = "shap"
    except Exception:
        top_shap_values = np.empty((0, len(artifact["feature_names"])))
        local_method = "fallback"

    local = local_explanations(
        artifact,
        raw_df=raw_df,
        scored_df=scored_df,
        top_n_transactions=top_n_transactions,
        top_n_features=top_n_features,
        shap_values=top_shap_values,
    )

    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    local.to_csv(LOCAL_EXPLANATIONS_PATH, index=False)

    summary = {
        "dataset_source": source,
        "model_name": artifact["model_name"],
        "global_method": method,
        "local_method": local_method,
        "sample_size": min(len(raw_df), sample_size),
        "top_n_transactions": top_n_transactions,
        "top_n_features": top_n_features,
        "feature_importance_path": str(FEATURE_IMPORTANCE_PATH),
        "local_explanations_path": str(LOCAL_EXPLANATIONS_PATH),
    }
    EXPLANATION_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera explicabilidade global e local do modelo.")
    parser.add_argument("--sample-size", type=int, default=1200)
    parser.add_argument("--top-n-transactions", type=int, default=8)
    parser.add_argument("--top-n-features", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_explanations(
        sample_size=args.sample_size,
        top_n_transactions=args.top_n_transactions,
        top_n_features=args.top_n_features,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
