# Multi-stage: frontend static assets + Python backend with ONNX Runtime.
FROM node:20-bookworm-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
ARG VITE_API_URL=http://127.0.0.1:8000
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

FROM python:3.11-slim-bookworm AS backend
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=frontend-build /app/frontend/dist ./frontend/dist
ENV SIGN_LANGUAGE_DATA_ROOT=/data
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# Models: mount /data or /app/models/served; on empty start, restore from HF if HF_TOKEN set.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
