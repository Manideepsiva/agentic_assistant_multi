FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ .
RUN npm run build          


FROM python:3.12-slim

# System dependencies:
#  - tesseract-ocr : image + scanned-PDF OCR (with confidence scores)
#  - poppler-utils : pdf2image page rasterisation for scanned PDFs
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY --from=frontend-build /static ./static

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
