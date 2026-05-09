import numpy as np
import pandas as pd

from .config import TARGET_COLUMN


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stable features for the Kaggle dataset and the demo dataset."""
    data = df.copy()

    if "Amount" in data.columns:
        amount = pd.to_numeric(data["Amount"], errors="coerce").fillna(0)
        data["amount_log"] = np.log1p(amount.clip(lower=0))
        amount_std = amount.std()
        if amount_std and amount_std > 0:
            data["amount_zscore"] = (amount - amount.mean()) / amount_std
        else:
            data["amount_zscore"] = 0.0

    if "Time" in data.columns:
        time_seconds = pd.to_numeric(data["Time"], errors="coerce").fillna(0)
        hour = ((time_seconds // 3600) % 24).astype(int)
        data["hour"] = hour
        data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        data["is_night"] = hour.isin([0, 1, 2, 3, 4, 5]).astype(int)

    return data


def feature_columns(df: pd.DataFrame) -> list[str]:
    ignored = {TARGET_COLUMN, "risk_score", "risk_bucket", "prediction"}
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns
    return [column for column in numeric_columns if column not in ignored]


def add_risk_bucket(df: pd.DataFrame, score_column: str = "risk_score") -> pd.DataFrame:
    data = df.copy()
    score = data[score_column].fillna(0)
    data["risk_bucket"] = pd.cut(
        score,
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["baixo", "moderado", "alto", "critico"],
    ).astype(str)
    return data

