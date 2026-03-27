FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]