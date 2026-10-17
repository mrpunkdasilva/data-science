FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf-cache \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY tests/fixtures ./tests/fixtures

RUN pip install --upgrade pip setuptools \
    && pip install -e . \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Build a small, self-contained dataset from the offline fixture so the
# dashboard starts with data out of the box (no network needed at runtime).
RUN python -m cli.main pipeline -i tests/fixtures/tst-sample.pdf

EXPOSE 8501

CMD ["streamlit", "run", "src/app/dashboard.py"]
