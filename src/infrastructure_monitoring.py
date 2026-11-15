"""CLI-точка входа для проверки infrastructure-метрик (CPU, RAM, disk)."""

from __future__ import annotations

import json

from src.monitoring import get_infrastructure_metrics


def main() -> None:
    """Напечатать текущие infrastructure-метрики в формате JSON."""
    print(json.dumps(get_infrastructure_metrics(), indent=2))


if __name__ == "__main__":
    main()
