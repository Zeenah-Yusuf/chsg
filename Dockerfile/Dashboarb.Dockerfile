FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dashboard source and data
COPY dashboard/ ./dashboard/
COPY data/ ./data/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.headless=true"]
