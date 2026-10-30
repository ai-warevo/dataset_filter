# Use the pre-compiled image with GPU/CUDA support baked in
FROM animcogn/face_recognition:gpu

WORKDIR /app

RUN /opt/venv/bin/python3 -m pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README.md .

RUN pip3 install --no-cache-dir .

COPY main.py .
COPY README.md .
COPY src/ ./src/

ENTRYPOINT ["python3", "main.py"]
