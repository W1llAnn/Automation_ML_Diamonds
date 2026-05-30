from src.data_processing import add_features, generate_sample_data
from src.monitoring import detect_data_drift, detect_degradation, get_infrastructure_metrics, save_baseline


def test_drift_detection_flags_large_mean_shift(tmp_path):
    baseline_data = add_features(generate_sample_data(rows=50))
    current_data = baseline_data.copy()
    current_data["carat"] = current_data["carat"] * 2
    baseline = save_baseline(baseline_data, {"rmse": 100.0, "r2": 0.9}, tmp_path / "baseline.json")

    result = detect_data_drift(current_data, baseline, relative_threshold=0.1)

    assert result["drift_detected"] is True
    assert "carat" in result["drifted_features"]


def test_drift_detection_returns_false_for_similar_data(tmp_path):
    baseline_data = add_features(generate_sample_data(rows=50))
    baseline = save_baseline(baseline_data, {"rmse": 100.0, "r2": 0.9}, tmp_path / "baseline.json")

    result = detect_data_drift(baseline_data.copy(), baseline, relative_threshold=0.1)

    assert result["drift_detected"] is False
    assert result["drifted_features"] == {}


def test_degradation_detection_uses_real_metrics():
    result = detect_degradation(
        {"rmse": 130.0, "r2": 0.75},
        {"rmse": 100.0, "r2": 0.9},
        rmse_threshold=0.1,
        r2_drop_threshold=0.05,
    )

    assert result["degradation_detected"] is True


def test_infrastructure_metrics_have_expected_keys():
    metrics = get_infrastructure_metrics()

    assert {"cpu_percent", "ram_percent", "disk_percent"} <= set(metrics)
