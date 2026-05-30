from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_processing import PROCESSED_DIR, PROJECT_ROOT, TARGET
from src.model_training import FEATURE_COLUMNS, MODEL_PATH, load_processed_data, train_model


FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def _load_data_and_model() -> tuple[pd.DataFrame, pd.DataFrame, object]:
    train_df, test_df = load_processed_data(PROCESSED_DIR)
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model, _ = train_model(train_df, test_df)
    return train_df, test_df, model


def plot_price_distribution(data: pd.DataFrame, output_path: Path) -> None:
    """Построить распределение цены бриллиантов."""
    plt.figure(figsize=(9, 5))
    plt.hist(data[TARGET], bins=30, color="#2f80ed", edgecolor="white")
    plt.title("Diamond Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_predicted_vs_actual(test_df: pd.DataFrame, model: object, output_path: Path) -> None:
    """Построить график сравнения фактической и предсказанной цены."""
    actual = test_df[TARGET].to_numpy()
    predicted = model.predict(test_df[FEATURE_COLUMNS])
    min_value = float(min(actual.min(), predicted.min()))
    max_value = float(max(actual.max(), predicted.max()))

    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.75, color="#27ae60", edgecolor="white", linewidth=0.4)
    plt.plot([min_value, max_value], [min_value, max_value], color="#eb5757", linestyle="--", label="Ideal prediction")
    plt.title("Predicted vs Actual Price")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(model: object, output_path: Path, top_n: int = 12) -> None:
    """Построить важность признаков RandomForest."""
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    order = np.argsort(importances)[-top_n:]
    names = [feature_names[index].replace("numeric__", "").replace("categorical__", "") for index in order]
    values = importances[order]

    plt.figure(figsize=(9, 5))
    plt.barh(names, values, color="#9b51e0")
    plt.title("RandomForest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_report_figures(output_dir: Path | str = FIGURES_DIR) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df, model = _load_data_and_model()

    processed_data = pd.concat([train_df, test_df], ignore_index=True)
    outputs = [
        output_dir / "price_distribution.png",
        output_dir / "predicted_vs_actual.png",
        output_dir / "feature_importance.png",
    ]

    plot_price_distribution(processed_data, outputs[0])
    plot_predicted_vs_actual(test_df, model, outputs[1])
    plot_feature_importance(model, outputs[2])
    return outputs


def main() -> None:
    for path in generate_report_figures():
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
