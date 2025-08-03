"""
Configuración del proyecto para descarga de imágenes de MercadoLibre.
"""

import os
from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"

# URLs y configuración de descarga
BASE_IMAGE_URL = "https://http2.mlstatic.com/D_{picture_id}-F.jpg"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 10
BATCH_SIZE = 100  # Número de imágenes a procesar en cada batch

# Configuración de paralelización
MAX_WORKERS = 4  # Número máximo de requests concurrentes
CHUNK_SIZE = 20  # Tamaño de chunk para procesamiento paralelo

# Configuración de logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Extensión de archivo para imágenes
IMAGE_EXTENSION = ".jpg"

# Crear directorios si no existen
IMAGES_DIR.mkdir(parents=True, exist_ok=True)