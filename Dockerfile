# ============================================================
# NeuroVision AI — FastAPI Backend
# ============================================================
FROM python:3.12-slim

# System dependencies for OpenCV, TensorFlow, ReportLab
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxrender1 libxext6 \
        libgl1-mesa-glx libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /neurovision

# Install Python dependencies first (layer-cached if requirements unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir \
        fastapi==0.104.1 \
        "uvicorn[standard]==0.24.0" \
        python-multipart \
        pydantic==2.13.4 \
        tensorflow-cpu \
        opencv-python-headless \
        Pillow \
        numpy \
        matplotlib \
        loguru \
        python-dotenv \
        PyYAML \
        tqdm \
        reportlab \
        fpdf2 \
        albumentations \
        scikit-learn

# Copy source code (ordered from least- to most-frequently changed)
COPY configs/   ./configs/
COPY src/        ./src/
COPY app/        ./app/
COPY scripts/    ./scripts/

# Models volume (mount at runtime: -v ./models:/neurovision/models)
RUN mkdir -p models outputs/plots outputs/logs outputs/reports outputs/pseudo_masks

# Non-root user for security
RUN useradd -m -u 1000 neurovision && chown -R neurovision:neurovision /neurovision
USER neurovision

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PROJECT_ROOT=/neurovision

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
