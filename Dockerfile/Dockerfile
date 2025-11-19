FROM python:3.10-slim

WORKDIR /app

# System dependencies for audio/image processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API source, models, schemas, data
COPY api/ ./api/
COPY models/ ./models/
COPY schemas/ ./schemas/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
