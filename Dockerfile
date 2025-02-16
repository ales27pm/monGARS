# --- Build Stage ---
FROM nvcr.io/nvidia/pytorch:23.10-py3 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

# --- Final Stage ---
FROM nvcr.io/nvidia/pytorch:23.10-py3-runtime
WORKDIR /app
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
EXPOSE 8000
CMD ["uvicorn", "monGARS.main:app", "--host", "0.0.0.0", "--port", "8000"]