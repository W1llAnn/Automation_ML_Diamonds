from __future__ import annotations

from src.data_processing import run_data_pipeline
from src.model_training import train_model


def main() -> None:
    train_df, test_df = run_data_pipeline()
    _, metrics = train_model(train_df, test_df)
    print("Pipeline completed")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
