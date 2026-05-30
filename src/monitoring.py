from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import psutil


def _numeric_profile(data: pd.DataFrame) -> dict[str, dict[str, float]]:
    profile: dict[str, dict[str, float]] = {}
    numeric_data = data.select_dtypes(include="number")
    for column in numeric_data.columns:
        series = numeric_data[column]
        profile[column] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return profile


def save_baseline(data: pd.DataFrame, metrics: dict[str, float], path: Path | str) -> dict[str, Any]:
    """Сохранить model metrics и numeric feature profile как monitoring baseline."""
    baseline = {
        "metrics": metrics,
        "numeric_profile": _numeric_profile(data),
        "rows": int(len(data)),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return baseline


def load_baseline(path: Path | str) -> dict[str, Any] | None:
    """Загрузить сохраненный monitoring baseline, если он доступен."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def detect_data_drift(
    current_data: pd.DataFrame,
    baseline: dict[str, Any],
    relative_threshold: float = 0.25,
) -> dict[str, Any]:
    current_profile = _numeric_profile(current_data)
    baseline_profile = baseline.get("numeric_profile", {})
    drifted_features: dict[str, dict[str, float]] = {}

    for column, current_stats in current_profile.items():
        base_stats = baseline_profile.get(column)
        if not base_stats:
            continue
        base_mean = float(base_stats["mean"])
        current_mean = float(current_stats["mean"])
        denominator = abs(base_mean) if abs(base_mean) > 1e-8 else 1.0
        relative_change = abs(current_mean - base_mean) / denominator
        if relative_change > relative_threshold:
            drifted_features[column] = {
                "baseline_mean": base_mean,
                "current_mean": current_mean,
                "relative_change": float(relative_change),
            }

    return {
        "drift_detected": bool(drifted_features),
        "drifted_features": drifted_features,
        "checked_features": sorted(current_profile.keys()),
    }


def detect_degradation(
    current_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    rmse_threshold: float = 0.15,
    r2_drop_threshold: float = 0.10,
) -> dict[str, Any]:
    base_rmse = baseline_metrics.get("rmse")
    current_rmse = current_metrics.get("rmse")
    base_r2 = baseline_metrics.get("r2")
    current_r2 = current_metrics.get("r2")

    rmse_degraded = False
    r2_degraded = False
    if base_rmse and current_rmse is not None:
        rmse_degraded = (current_rmse - base_rmse) / base_rmse > rmse_threshold
    if base_r2 is not None and current_r2 is not None:
        r2_degraded = base_r2 - current_r2 > r2_drop_threshold

    return {
        "degradation_detected": bool(rmse_degraded or r2_degraded),
        "rmse_degraded": bool(rmse_degraded),
        "r2_degraded": bool(r2_degraded),
    }


def get_infrastructure_metrics() -> dict[str, float]:
    """Вернуть текущую загрузку CPU, memory и disk в процентах."""
    disk = psutil.disk_usage("/")
    memory = psutil.virtual_memory()
    return {
        "cpu_percent": float(psutil.cpu_percent(interval=0.1)),
        "ram_percent": float(memory.percent),
        "disk_percent": float(disk.percent),
    }
