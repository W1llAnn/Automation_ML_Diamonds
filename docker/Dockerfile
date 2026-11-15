FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONPATH=/app \
    POETRY_VERSION=2.5.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock /app/
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --retries 10 --timeout 120 --progress-bar off "poetry==$POETRY_VERSION" && \
    poetry install --only main --no-root --no-ansi

COPY src /app/src
COPY data /app/data
COPY models /app/models
COPY reports /app/reports

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
