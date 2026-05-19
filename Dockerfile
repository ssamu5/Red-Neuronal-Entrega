# ════════════════════════════════════════════════════════════════════════════
# YouTube Spam Detector — Dockerfile
# Proyecto I: Introducción a la IA — UPV 2026
# ════════════════════════════════════════════════════════════════════════════

# Imagen base slim (Python 3.11) — equilibrio entre peso y compatibilidad
FROM python:3.11-slim

# Metadatos
LABEL maintainer="Proyecto IA UPV — Ángel, Samuel, Artur, Pablo"
LABEL description="YouTube Spam Detector - Red Neuronal Superficial con Streamlit"
LABEL version="1.0"

# Variables de entorno para Python optimizado
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── Dependencias del sistema ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      libffi-dev \
      openssl-dev

# ── Directorio de trabajo ────────────────────────────────────────────────────
WORKDIR /app

# ── Instalar dependencias Python (capa de caché separada) ───────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Copiar código fuente y datos ─────────────────────────────────────────────
COPY app.py                          .
COPY Youtube-Spam-Dataset.csv        .
COPY .streamlit/                     ./.streamlit/

# Copiar modelo pre-entrenado si existe (opcional — se entrena en primer arranque si no)
COPY model/                          ./model/

# Crear directorios necesarios
RUN mkdir -p model .streamlit

# ── Healthcheck — verifica que Streamlit responde ────────────────────────────
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Puerto de Streamlit ──────────────────────────────────────────────────────
EXPOSE 8501

# ── Arranque de la aplicación ────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
