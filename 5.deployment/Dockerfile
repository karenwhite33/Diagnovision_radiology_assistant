# Use lightweight Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY Scripts/ /app/
COPY requirements.txt /app/
COPY download_models.py /app/

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

# Install gdown and download models from Google Drive
RUN pip install gdown
RUN python /app/download_models.py

# Install PyTorch manually (CUDA 11.8)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Expose FastAPI & Gradio ports
EXPOSE 8000 7860

# Start FastAPI + Gradio together
CMD ["sh", "-c", "python download_models.py && python gradio_apiapp.py"]
