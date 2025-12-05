# Base Image
FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Requirements (Local now)
COPY requirements.txt .

# Install Dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy Application Code (Local now)
COPY src/ ./src/
COPY app/ ./app/

# Models & Config
RUN mkdir -p models

# Expose Port
EXPOSE 8000

# Env
ENV PYTHONUNBUFFERED=1

# Run (Ensure we are in /app so imports work)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]