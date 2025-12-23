# Dockerfile for Sherpa-ONNX ASR Demo (Step 9: Docker Compose Porting)
# Builds ten-vad from source with ONNX Runtime (ONNX model, not JIT)

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Note: FFmpeg required by mp3_writer.py for MP3 recording
RUN apt-get update && apt-get install -y \
    build-essential \
    libasound2-dev \
    portaudio19-dev \
    git \
    cmake \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Download and install ONNX Runtime (to /root/ for ten-vad build script)
RUN ONNX_VER=1.22.0 && \
    ARCH=$(uname -m) && if [ "$ARCH" = "x86_64" ]; then ARCH="x64"; fi && \
    curl -L -o onnxruntime.tgz \
        https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VER}/onnxruntime-linux-${ARCH}-${ONNX_VER}.tgz && \
    tar -xzf onnxruntime.tgz && \
    mv onnxruntime-linux-${ARCH}-${ONNX_VER} ~/onnxruntime-linux-${ARCH}-${ONNX_VER} && \
    rm onnxruntime.tgz && \
    echo "ONNX Runtime installed to ~/onnxruntime-linux-${ARCH}-${ONNX_VER}"

# Clone ten-vad repository
RUN git clone --depth 1 https://github.com/TEN-framework/ten-vad.git /tmp/ten-vad

# Build ten-vad Python module with ONNX backend
# Note: .so file is in build directory, not build/lib/ subdirectory
RUN cd /tmp/ten-vad/examples_onnx/python && \
    pip install pybind11 && \
    mkdir -p build && \
    cd build && \
    cmake .. -DORT_PATH=~/onnxruntime-linux-x64-1.22.0 && \
    make -j$(nproc) && \
    cp ten_vad_python.cpython-*.so /usr/local/lib/python3.12/site-packages/ && \
    echo "ten-vad Python module built and installed"

# Copy Python requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py /app/

# Copy ONNX model directory (required by ten-vad .so)
# Note: onnx_model is NOT in ten-vad repo, copy from host context
COPY onnx_model /app/onnx_model/

# Create symlink to match hardcoded path in asr_config.py
RUN mkdir -p /home/luigi/sherpa && \
    ln -s /app/models /home/luigi/sherpa/models && \
    echo "Created symlink: /home/luigi/sherpa/models → /app/models"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LD_LIBRARY_PATH=~/onnxruntime-linux-x64-1.22.0/lib:/usr/local/lib
ENV PYTHONPATH=/usr/local/lib/python3.12/site-packages

# Create directory for ASR models
RUN mkdir -p /app/models

# Volume for ASR models (downloaded outside Docker)
VOLUME /app/models

# Default command (can be overridden by docker-compose)
CMD ["python3", "demo_full_integration.py"]
