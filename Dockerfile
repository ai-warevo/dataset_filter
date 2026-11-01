# Use the pre-compiled image with GPU/CUDA support baked in
FROM animcogn/face_recognition:gpu

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

# Remove broken NVIDIA repositories to avoid GPG errors during build
RUN rm -f /etc/apt/sources.list.d/cuda*.list && \
    rm -f /etc/apt/sources.list.d/nvidia*.list && \
    apt-get update && apt-get install -y \
    libgl1 \
    git \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md .
COPY main.py .
COPY src/ ./src/

RUN /opt/venv/bin/python3 -m pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/python3 -m pip install --no-cache-dir .

ENTRYPOINT ["/opt/venv/bin/python3"]
