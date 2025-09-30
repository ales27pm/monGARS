# --- Build Stage ---
FROM nvcr.io/nvidia/pytorch:23.10-py3 AS builder
ARG JOBS=1
ENV MAKEFLAGS="-j${JOBS}"
WORKDIR /app
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --no-cache-dir -r requirements.txt
COPY . /app

# --- Final Stage ---
FROM nvcr.io/nvidia/pytorch:23.10-py3-runtime
RUN groupadd --system --gid 10001 mongars \
    && useradd --system --uid 10001 --gid mongars --create-home --home-dir /home/mongars mongars
WORKDIR /app
COPY --from=builder --chown=mongars:mongars /app /app
COPY --from=builder --chown=mongars:mongars /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
USER mongars

EXPOSE 8000
CMD ["uvicorn", "monGARS.main:app", "--host", "0.0.0.0", "--port", "8000"]
