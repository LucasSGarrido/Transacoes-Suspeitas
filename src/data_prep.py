from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, RAW_DATA_PATH, TARGET_COLUMN
from .features import engineer_features, feature_columns


def generate_demo_transactions(
    n_samples: int = 12000,
    fraud_rate: float = 0.018,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a local demo dataset when Kaggle data is not available."""
    x, y = make_classification(
        n_samples=n_samples,
        n_features=28,
        n_informative=10,
        n_redundant=6,
        n_clusters_per_class=3,
        weights=[1 - fraud_rate, fraud_rate],
        flip_y=0.003,
        class_sep=1.6,
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)
    data = pd.DataFrame(x, columns=[f"V{i}" for i in range(1, 29)])
    data["Time"] = np.sort(rng.integers(0, 172800, size=n_samples))

    regular_amount = rng.lognormal(mean=3.2, sigma=0.9, size=n_samples)
    fraud_amount = rng.lognormal(mean=4.0, sigma=1.0, size=n_samples)
    data["Amount"] = np.where(y == 1, fraud_amount, regular_amount).round(2)
    data[TARGET_COLUMN] = y.astype(int)
    return data[["Time", *[f"V{i}" for i in range(1, 29)], "Amount", TARGET_COLUMN]]


def load_transactions(path=RAW_DATA_PATH) -> tuple[pd.DataFrame, str]:
    if path.exists():
        data = pd.read_csv(path)
        source = "kaggle"
    else:
        data = generate_demo_transactions()
        source = "demo_sintetico"

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Coluna alvo '{TARGET_COLUMN}' nao encontrada.")

    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    return data, source


def build_feature_frame(df: pd.DataFrame, expected_columns: list[str] | None = None) -> pd.DataFrame:
    data = engineer_features(df)

    if expected_columns is None:
        columns = feature_columns(data)
    else:
        columns = expected_columns
        for column in columns:
            if column not in data.columns:
                data[column] = 0

    return data[columns].copy()


def split_train_validation_test(
    df: pd.DataFrame,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = RANDOM_STATE,
):
    x = build_feature_frame(df)
    y = df[TARGET_COLUMN]

    x_train_valid, x_test, y_train_valid, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    validation_relative_size = validation_size / (1 - test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_valid,
        y_train_valid,
        test_size=validation_relative_size,
        stratify=y_train_valid,
        random_state=random_state,
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test

