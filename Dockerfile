# Utiliza una imagen base oficial de Python
FROM python:3.13.5-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de requerimientos y el código fuente
COPY requirements.txt ./
COPY ingest.py ./
COPY main.py ./
COPY documents/ ./documents/

# Instala las dependencias del sistema necesarias para procesar PDFs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        poppler-utils \
        gcc \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        tesseract-ocr \
        && rm -rf /var/lib/apt/lists/*

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Comando por defecto para ejecutar el script
CMD ["python", "ingest.py"]