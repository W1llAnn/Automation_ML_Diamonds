from fastapi.testclient import TestClient

from src import app as app_module
from src.app import app
from src.data_processing import add_features, generate_sample_data
from src.model_training import train_model


client = TestClient(app)


def test_health_endpoint_does_not_require_model():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "infrastructure" in response.json()


def test_predict_endpoint_returns_prediction(tmp_path, monkeypatch):
    data = add_features(generate_sample_data(rows=80))
    train_df = data.iloc[:60].reset_index(drop=True)
    test_df = data.iloc[60:].reset_index(drop=True)
    trained_model, _ = train_model(
        train_df,
        test_df,
        model_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
        baseline_path=tmp_path / "baseline.json",
    )
    monkeypatch.setattr(app_module, "model", trained_model)

    payload = {
        "carat": 1.0,
        "cut": "Ideal",
        "color": "E",
        "clarity": "VS1",
        "depth": 61.0,
        "table": 57.0,
        "x": 6.0,
        "y": 6.1,
        "z": 3.8,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0
    assert response.json()["message"] == "Prediction completed successfully."


def test_predict_endpoint_returns_503_without_model(monkeypatch):
    monkeypatch.setattr(app_module, "model", None)
    monkeypatch.setattr(app_module, "load_model", lambda: None)

    response = client.post(
        "/predict",
        json={
            "carat": 1.0,
            "cut": "Ideal",
            "color": "E",
            "clarity": "VS1",
            "depth": 61.0,
            "table": 57.0,
            "x": 6.0,
            "y": 6.1,
            "z": 3.8,
        },
    )

    assert response.status_code == 503
    assert "Model is not trained yet" in response.json()["detail"]


def test_predict_endpoint_validates_positive_numeric_fields():
    response = client.post(
        "/predict",
        json={
            "carat": -1.0,
            "cut": "Ideal",
            "color": "E",
            "clarity": "VS1",
            "depth": 61.0,
            "table": 57.0,
            "x": 6.0,
            "y": 6.1,
            "z": 3.8,
        },
    )

    assert response.status_code == 422


def test_model_info_endpoint():
    response = client.get("/model/info")

    assert response.status_code == 200
    assert "features" in response.json()
    assert "message" in response.json()
