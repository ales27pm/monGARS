# --- Build Stage ---
FROM nvcr.io/nvidia/pytorch:23.10-py3 AS builder
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
        python3-venv \
        libffi-dev \
        libgl1 \
        libjpeg-dev \
        libpq-dev \
        libssl-dev \
        libxml2-dev \
        libxslt1-dev \
        pkg-config \
        unzip \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system

WORKDIR /app
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download fr_core_news_sm
COPY . /app

# --- Final Stage ---
FROM nvcr.io/nvidia/pytorch:23.10-py3 AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
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
        python3-venv \
        libffi8 \
        libgl1 \
        libjpeg-turbo8 \
        libpq5 \
        libssl3 \
        libxml2 \
        libxslt1.1 \
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
