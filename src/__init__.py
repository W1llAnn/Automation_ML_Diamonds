"""Пакет обучения модели и FastAPI для Diamonds MLOps project."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("automation-ml-diamonds")
except PackageNotFoundError:  # проект запущен без установки пакета
    __version__ = "1.0.0"
