# SkinSense AI - web app image.   Build:  docker build -t skinsense .
# CPU-only PyTorch keeps the image small. For a GPU host use the default CUDA wheels instead:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 -t skinsense:gpu .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch torchvision --index-url ${TORCH_INDEX}

COPY requirements.txt requirements-deploy.txt ./
RUN pip install -r requirements-deploy.txt

COPY backend ./backend
COPY frontend ./frontend
COPY data ./data

# Run as an unprivileged user. Weights are mounted at /models, uploads live on a volume at /data.
RUN useradd --create-home app && mkdir -p /data/uploads /models && chown -R app:app /app /data
USER app

ENV SKINSENSE_UPLOADS=/data/uploads \
    SKINSENSE_WEIGHTS=/models/skin_sense_ensemble_best.pth \
    SKINSENSE_CALIBRATION=/app/backend/calibration.json

WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
