from __future__ import annotations

import joblib
import pandas as pd

from .config import MODEL_PATH
from .data_prep import build_feature_frame
from .features import add_risk_bucket


def load_model_artifact(path=MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError("Modelo nao encontrado. Execute: python -m src.train")
    return joblib.load(path)


def score_transactions(df: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    artifact = artifact or load_model_artifact()
    features = build_feature_frame(df, expected_columns=artifact["feature_names"])
    scores = artifact["model"].predict_proba(features)[:, 1]

    scored = df.copy()
    scored["risk_score"] = scores
    scored["prediction"] = (scores >= artifact["threshold"]).astype(int)
    return add_risk_bucket(scored)

