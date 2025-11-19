FROM python:3.10-slim

WORKDIR /app

# Install cron + system deps
RUN apt-get update && apt-get install -y \
    cron \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy jobs folder
COPY jobs/ ./jobs/
COPY data/ ./data/

# Copy crontab definition
COPY jobs/crontab /etc/cron.d/chsg-cron
RUN chmod 0644 /etc/cron.d/chsg-cron && crontab /etc/cron.d/chsg-cron

# Log file
RUN touch /var/log/cron.log

CMD ["cron", "-f"]
