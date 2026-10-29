# Use the pre-compiled image with GPU/CUDA support baked in
FROM animcogn/face_recognition:gpu

WORKDIR /app

# Upgrade pip and install tqdm
RUN /opt/venv/bin/python3 -m pip install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir tqdm

# Copy application layers
COPY main.py .
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

ENTRYPOINT ["python3", "main.py"]
