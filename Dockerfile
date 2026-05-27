FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

WORKDIR /build

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY semcode ./semcode
COPY scripts ./scripts

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install --no-deps .


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    APP_HOME=/app \
    DATA_DIR=/app/data \
    FAISS_INDEX_PATH=/app/data/index.faiss \
    METADATA_PATH=/app/data/metadata.parquet \
    RERANKER_MODEL_PATH=/app/data/reranker \
    HF_HOME=/home/semcode/.cache/huggingface \
    TORCH_HOME=/home/semcode/.cache/torch \
    TRANSFORMERS_CACHE=/home/semcode/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system semcode \
    && adduser --system --ingroup semcode --home /home/semcode semcode \
    && mkdir -p /app/data /home/semcode/.cache/huggingface /home/semcode/.cache/torch \
    && chown -R semcode:semcode /app /home/semcode

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER semcode

EXPOSE 8000

CMD ["python", "-m", "semcode", "serve", "--host", "0.0.0.0", "--port", "8000"]
