FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model/ ./model/
COPY test_images/ ./test_images/
COPY histo.ico .

RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

EXPOSE 7860

CMD ["python", "app.py"]
