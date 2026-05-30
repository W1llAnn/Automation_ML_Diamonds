from __future__ import annotations

from dataclasses import dataclass, asdict

import psutil


@dataclass(slots=True)
class InfrastructureMetrics:
    cpu_percent: float
    ram_percent: float
    disk_percent: float


def get_infrastructure_metrics() -> dict[str, float]:
    """
    Return current host resource utilization percentages.

    psutil.cpu_percent uses a short interval to avoid returning stale 0.0 values.
    """
    metrics = InfrastructureMetrics(
        cpu_percent=psutil.cpu_percent(interval=0.1),
        ram_percent=psutil.virtual_memory().percent,
        disk_percent=psutil.disk_usage("/").percent,
    )
    return asdict(metrics)


def main() -> None:
    metrics = get_infrastructure_metrics()
    print(
        "Infrastructure metrics: "
        f"CPU={metrics['cpu_percent']:.1f}% "
        f"RAM={metrics['ram_percent']:.1f}% "
        f"DISK={metrics['disk_percent']:.1f}%"
    )


if __name__ == "__main__":
    main()
