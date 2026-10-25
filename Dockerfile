FROM node:22-alpine AS frontend-build

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api.py ./src/api.py
COPY src/dashboard.html ./src/dashboard.html
COPY --from=frontend-build /frontend/dist/ ./frontend/dist/
COPY results/full_pipeline/models/ ./models/
COPY results/full_pipeline/tables/ ./results/full_pipeline/tables/
COPY results/full_pipeline/figures/ ./results/full_pipeline/figures/

ENV MODEL_DIR=/app/models
EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]