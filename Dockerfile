# Use lightweight Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY Scripts/ /app/
COPY requirements.txt /app/

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip & install dependencies (excluding PyTorch)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install PyTorch manually (CUDA 11.8)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Expose FastAPI & Gradio ports
EXPOSE 8000 7860

# Run FastAPI & Gradio together
CMD ["bash", "-c", "uvicorn apiapp:app --host 0.0.0.0 --port 8000 & python gradio_apiapp.py"]
