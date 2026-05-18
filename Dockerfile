# ── Imagen base Alpine (ligera y segura) ──────────────────────────────────────
FROM python:3.10-alpine

# Metadatos
LABEL maintainer="Proyecto IA UPV"
LABEL description="YouTube Spam Detector — Red Neuronal Superficial"
LABEL version="1.0"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── Dependencias del sistema (Alpine usa apk, no apt) ─────────────────────────
RUN apk add --no-cache \
      gcc \
      g++ \
      musl-dev \
      linux-headers \
      curl \
      libffi-dev \
      openssl-dev

# ── Directorio de trabajo ─────────────────────────────────────────────────────
WORKDIR /app

# ── Dependencias Python ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Código fuente ─────────────────────────────────────────────────────────────
COPY app.py .
COPY .streamlit/ ./.streamlit/

# ── Datos ─────────────────────────────────────────────────────────────────────
COPY Youtube-Spam-Dataset.csv .
COPY Youtube-Spam-Dataset\ equilibrado.csv.csv .

# Crear directorio model (para montar modelo entrenado)
RUN mkdir -p model

# ── Puerto Streamlit ──────────────────────────────────────────────────────────
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Comando de inicio ─────────────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
