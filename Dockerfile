# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system deps needed for python-docx, PyPDF2, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy slim requirements first (for Docker layer caching)
COPY requirements-deploy.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy source code
COPY src/ ./src/
COPY frontend/ ./frontend/

# Copy ONNX model + tokenizer files only (no heavy PyTorch weights)
COPY model/onnx/ ./model/onnx/

# Create logs directory
RUN mkdir -p logs

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Start FastAPI on port 7860
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "7860"]
