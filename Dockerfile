FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend and build React app
COPY frontend/package*.json ./frontend/
WORKDIR /app/frontend
RUN npm ci --production=false

COPY frontend/ ./
RUN chmod -R +x node_modules/.bin && npm run build

# Back to app root
WORKDIR /app

# Copy remaining app files
COPY api.py .
COPY model/ ./model/

# Create user for HuggingFace
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["python", "api.py"]
