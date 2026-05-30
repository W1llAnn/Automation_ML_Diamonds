from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diamonds.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NUMERIC_FEATURES = ["carat", "depth", "table", "x", "y", "z"]
CATEGORICAL_FEATURES = ["cut", "color", "clarity"]
ENGINEERED_FEATURES = ["volume", "density", "depth_to_width"]
TARGET = "price"
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def generate_sample_data(rows: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Создать небольшой diamonds-like датасет для демо и тестов."""
    rng = np.random.default_rng(random_state)
    carat = rng.uniform(0.2, 2.5, rows).round(2)
    x = (4.0 + carat * 2.2 + rng.normal(0, 0.15, rows)).clip(3.0, None)
    y = (4.0 + carat * 2.1 + rng.normal(0, 0.15, rows)).clip(3.0, None)
    z = (2.4 + carat * 1.3 + rng.normal(0, 0.1, rows)).clip(1.5, None)
    depth = rng.normal(61.5, 2.0, rows).clip(55.0, 70.0)
    table = rng.normal(57.5, 2.5, rows).clip(50.0, 68.0)

    cut_values = np.array(["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color_values = np.array(["J", "I", "H", "G", "F", "E", "D"])
    clarity_values = np.array(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

    cut = rng.choice(cut_values, rows, p=[0.05, 0.15, 0.25, 0.25, 0.30])
    color = rng.choice(color_values, rows)
    clarity = rng.choice(clarity_values, rows)

    cut_score = pd.Series(cut).map({name: idx for idx, name in enumerate(cut_values, start=1)}).to_numpy()
    color_score = pd.Series(color).map({name: idx for idx, name in enumerate(color_values, start=1)}).to_numpy()
    clarity_score = pd.Series(clarity).map({name: idx for idx, name in enumerate(clarity_values, start=1)}).to_numpy()
    price = (
        500
        + carat * 4200
        + cut_score * 180
        + color_score * 130
        + clarity_score * 160
        + rng.normal(0, 350, rows)
    ).clip(300, None)

    return pd.DataFrame(
        {
            "carat": carat,
            "cut": cut,
            "color": color,
            "clarity": clarity,
            "depth": depth.round(2),
            "table": table.round(2),
            "price": price.round(2),
            "x": x.round(2),
            "y": y.round(2),
            "z": z.round(2),
        }
    )


def load_raw_data(path: Path | str = RAW_DATA_PATH, create_if_missing: bool = True) -> pd.DataFrame:
    """Загрузить raw diamonds data или создать детерминированный sample dataset."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    if not create_if_missing:
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_sample_data()
    data.to_csv(path, index=False)
    return data


def validate_data(data: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Проверить обязательные колонки, непустые строки и числовые типы данных."""
    missing = set(required_columns) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if data.empty:
        raise ValueError("Dataset is empty")

    for column in NUMERIC_FEATURES + [TARGET]:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column {column} must be numeric")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Удалить дубликаты, пропуски и некорректные физические измерения."""
    cleaned = data.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)

    positive_columns = ["carat", "price", "x", "y", "z", "depth", "table"]
    for column in positive_columns:
        cleaned = cleaned[cleaned[column] > 0]

    cleaned = cleaned[(cleaned["depth"] >= 40) & (cleaned["depth"] <= 80)]
    cleaned = cleaned[(cleaned["table"] >= 40) & (cleaned["table"] <= 90)]
    return cleaned.reset_index(drop=True)


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Добавить domain features, используемые при обучении и inference."""
    featured = data.copy()
    featured["volume"] = featured["x"] * featured["y"] * featured["z"]
    featured["density"] = featured["carat"] / (featured["volume"] + 0.001)
    featured["depth_to_width"] = featured["depth"] / (featured["x"] + 0.001)
    return featured


def split_and_save(
    data: pd.DataFrame,
    output_dir: Path | str = PROCESSED_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    data.to_csv(output_dir / "diamonds_processed.csv", index=False)
    return train_df, test_df


def run_data_pipeline(raw_path: Path | str = RAW_DATA_PATH, output_dir: Path | str = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_data = load_raw_data(raw_path)
    validate_data(raw_data)
    cleaned = clean_data(raw_data)
    featured = add_features(cleaned)
    return split_and_save(featured, output_dir)


def main() -> None:
    train_df, test_df = run_data_pipeline()
    print(f"Saved processed train/test data: train={len(train_df)}, test={len(test_df)}")


if __name__ == "__main__":
    main()
