# Changelog

All notable changes to this MLOps project are documented here.

## 1.0.0 - 2026-05-28

### Added

- ETL pipeline for the Diamonds regression dataset.
- Feature engineering: `volume`, `density`, `depth_to_width`.
- RandomForestRegressor training pipeline with RMSE, MAE, R2, and MAPE metrics.
- FastAPI application with health, prediction, and model info endpoints.
- Lightweight monitoring for drift, degradation, CPU, RAM, and disk usage.
- Pytest suite for data processing, model training, API behavior, and monitoring.
- Dockerfile and Docker Compose configuration.
- GitHub Actions workflow for dependency installation, compile check, tests, and Docker build.
- README, technical review report, and project structure documentation.
