# Lean image for the aggregation server. It needs only torch (CPU — the server does NO model
# inference, just int64 sums + safetensors I/O), safetensors, cryptography, fastapi, uvicorn, and
# boto3 (S3 object-storage backend for the scale-to-zero deploy). We copy the package and set
# PYTHONPATH rather than `pip install .` to keep the image minimal.
#
# Build from the REPO ROOT:   docker build -t roger-agg .
# Run (scale-to-zero, S3):    see DEPLOY.md — the durable global lives in object storage, not on disk.
FROM python:3.13-slim

RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir safetensors cryptography fastapi "uvicorn[standard]" boto3

WORKDIR /app
COPY roger_server/ /app/roger_server/
ENV PYTHONPATH=/app
ENV ROGER_SERVER_DATA=/data
EXPOSE 8000

CMD ["python", "-m", "roger_server"]
