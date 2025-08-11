# Build stage to have ffmpeg and python with onvif-zeep
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy app
WORKDIR /app
COPY app/ /app/

# Default envs
ENV CLIP_SECONDS=10
ENV COOLDOWN_SECONDS=20

CMD ["python", "-u", "main.py"]