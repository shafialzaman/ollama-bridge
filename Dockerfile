# Dockerfile for the Ollama Bridge API server
# Used by docker-compose.yml on the Linux/CUDA workstation

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py server.py ./

EXPOSE 8000

CMD ["python", "server.py"]
