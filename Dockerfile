# Hugging Face Spaces (Docker SDK) expects the app to listen on port 7860.
# Free tier gives 16GB RAM / 2 vCPU - comfortably fits our stack, unlike
# Render's free tier (512MB RAM / 0.1 CPU) which OOM-crashed under load.
FROM python:3.12-slim

WORKDIR /app

# lightgbm needs libgomp1 at runtime (OpenMP) - not included in slim by default
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
