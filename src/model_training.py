from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_processing import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    PROJECT_ROOT,
    TARGET,
    run_data_pipeline,
)
from src.monitoring import save_baseline


MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "diamond_price_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
BASELINE_PATH = PROJECT_ROOT / "reports" / "monitoring" / "baseline.json"

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES


def build_model(random_state: int = 42) -> Pipeline:
    """Собрать preprocessing и regression pipeline для обучения."""
    numeric_features = NUMERIC_FEATURES + ENGINEERED_FEATURES
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    regressor = RandomForestRegressor(
        n_estimators=120,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", regressor)])


def calculate_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Рассчитать regression metrics по реальным предсказаниям."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    denominator = np.where(np.asarray(y_true) == 0, 1e-8, np.asarray(y_true))
    mape = float(np.mean(np.abs((np.asarray(y_true) - y_pred) / denominator)) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def load_processed_data(processed_dir: Path | str = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузить processed train/test data, создав их при необходимости."""
    processed_dir = Path(processed_dir)
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        return run_data_pipeline(output_dir=processed_dir)
    return pd.read_csv(train_path), pd.read_csv(test_path)


def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_path: Path | str = MODEL_PATH,
    metrics_path: Path | str = METRICS_PATH,
    baseline_path: Path | str = BASELINE_PATH,
) -> tuple[Pipeline, dict[str, float]]:
    model = build_model()
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET]

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = calculate_metrics(y_test, predictions)

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_baseline(train_df[FEATURE_COLUMNS], metrics, baseline_path)
    return model, metrics


def load_model(path: Path | str = MODEL_PATH) -> Any | None:
    """Загрузить trained model artifact, если он существует."""
    path = Path(path)
    if not path.exists():
        return None
    return joblib.load(path)


def main() -> None:
    train_df, test_df = load_processed_data()
    _, metrics = train_model(train_df, test_df)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
