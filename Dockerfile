# syntax=docker/dockerfile:1.6

ARG PYTORCH_IMAGE=pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# --- Build Stage ---
FROM ${PYTORCH_IMAGE} AS builder
ENV PATH="/opt/conda/bin:${PATH}"
ARG JOBS=1
ENV MAKEFLAGS="-j${JOBS}"

# Install system dependencies required for compiling Python packages and
# interacting with external LLM tooling (Git, curl, etc.).
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libffi-dev \
        libgl1 \
        libjpeg-dev \
        libpq-dev \
        libssl-dev \
        libxml2-dev \
        libxslt1-dev \
        pkg-config \
        python3-venv \
        unzip \
        wget \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system

WORKDIR /app
COPY requirements.txt .
COPY package.json package-lock.json ./
COPY vendor/llm2vec_monGARS ./vendor/llm2vec_monGARS
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download fr_core_news_sm
RUN npm ci
COPY . /app
RUN NODE_ENV=production npm run build \
    && mkdir -p /app/static \
    && cp -R webapp/static/. /app/static/ \
    && rm -rf node_modules

# --- Final Stage ---
ARG PYTORCH_IMAGE
FROM ${PYTORCH_IMAGE} AS runtime
ENV PATH="/opt/conda/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime dependencies for multimedia processing, SSL/TLS, and Git-based model
# downloads used by LLM tooling.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libffi-dev \
        libgl1 \
        libjpeg-dev \
        libpq5 \
        libssl-dev \
        libxml2 \
        libxslt1.1 \
        python3-venv \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system

RUN groupadd --system --gid 10001 mongars \
    && useradd --system --uid 10001 --gid mongars --create-home --home-dir /home/mongars mongars
WORKDIR /app
COPY --from=builder --chown=mongars:mongars /app /app
COPY --from=builder --chown=mongars:mongars /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
USER mongars

EXPOSE 8000
CMD ["uvicorn", "monGARS.api.web_api:app", "--host", "0.0.0.0", "--port", "8000"]
