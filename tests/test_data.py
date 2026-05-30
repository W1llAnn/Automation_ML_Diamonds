import pandas as pd
import pytest

from src.data_processing import add_features, clean_data, load_raw_data, validate_data


def test_validate_data_requires_expected_columns():
    data = pd.DataFrame({"carat": [1.0]})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(data)


def test_validate_data_rejects_empty_dataset():
    data = pd.DataFrame(
        columns=["carat", "cut", "color", "clarity", "depth", "table", "price", "x", "y", "z"]
    )

    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_data(data)


def test_clean_data_removes_invalid_rows():
    data = pd.DataFrame(
        {
            "carat": [1.0, 0.0, 1.1],
            "cut": ["Ideal", "Good", "Ideal"],
            "color": ["E", "F", "E"],
            "clarity": ["VS1", "SI1", "VS1"],
            "depth": [61.0, 62.0, 100.0],
            "table": [57.0, 58.0, 57.0],
            "price": [5000, 4000, 5100],
            "x": [6.0, 0.0, 6.1],
            "y": [6.1, 6.0, 6.1],
            "z": [3.8, 3.7, 3.8],
        }
    )

    cleaned = clean_data(data)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["carat"] == 1.0


def test_clean_data_can_return_empty_dataframe_for_invalid_input():
    data = pd.DataFrame(
        {
            "carat": [0.0],
            "cut": ["Ideal"],
            "color": ["E"],
            "clarity": ["VS1"],
            "depth": [61.0],
            "table": [57.0],
            "price": [5000],
            "x": [0.0],
            "y": [6.1],
            "z": [3.8],
        }
    )

    assert clean_data(data).empty


def test_add_features_creates_expected_columns():
    data = pd.DataFrame(
        {
            "carat": [1.0],
            "cut": ["Ideal"],
            "color": ["E"],
            "clarity": ["VS1"],
            "depth": [60.0],
            "table": [57.0],
            "price": [5000],
            "x": [5.0],
            "y": [4.0],
            "z": [3.0],
        }
    )

    result = add_features(data)

    assert result.loc[0, "volume"] == 60.0
    assert result.loc[0, "density"] == pytest.approx(1.0 / 60.001)
    assert result.loc[0, "depth_to_width"] == pytest.approx(60.0 / 5.001)


def test_load_raw_data_can_fail_explicitly_when_missing(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_raw_data(missing_path, create_if_missing=False)
