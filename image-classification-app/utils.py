"""
Utilidades para el procesamiento de imágenes y datos.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image
import pandas as pd

try:
    from .config import (
        BASE_IMAGE_URL, 
        MAX_RETRIES, 
        TIMEOUT_SECONDS, 
        IMAGE_EXTENSION,
        LOG_FORMAT,
        LOG_LEVEL
    )
except ImportError:
    # Fallback para ejecución directa
    import config
    
    BASE_IMAGE_URL = config.BASE_IMAGE_URL
    MAX_RETRIES = config.MAX_RETRIES
    TIMEOUT_SECONDS = config.TIMEOUT_SECONDS
    IMAGE_EXTENSION = config.IMAGE_EXTENSION
    LOG_FORMAT = config.LOG_FORMAT
    LOG_LEVEL = config.LOG_LEVEL


def setup_logging() -> logging.Logger:
    """
    Configura el sistema de logging para el proyecto.
    
    Returns:
        logging.Logger: Logger configurado
    """
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('image_download.log')
        ]
    )
    return logging.getLogger(__name__)


def build_image_url(picture_id: str) -> str:
    """
    Construye la URL de la imagen a partir del picture_id.
    
    Args:
        picture_id (str): ID de la imagen
        
    Returns:
        str: URL completa de la imagen
    """
    return BASE_IMAGE_URL.format(picture_id=picture_id)


def generate_filename(item_id: str, picture_id: str) -> str:
    """
    Genera un nombre de archivo único para la imagen.
    
    Args:
        item_id (str): ID del item
        picture_id (str): ID de la imagen
        
    Returns:
        str: Nombre del archivo
    """
    # Limpiamos caracteres especiales que pueden causar problemas
    safe_item_id = "".join(c for c in item_id if c.isalnum() or c in ('-', '_'))
    safe_picture_id = "".join(c for c in picture_id if c.isalnum() or c in ('-', '_'))
    
    return f"{safe_item_id}_{safe_picture_id}{IMAGE_EXTENSION}"


def download_image(url: str, filepath: Path, logger: logging.Logger) -> Tuple[bool, Optional[str]]:
    """
    Descarga una imagen desde una URL y la guarda en el path especificado.
    
    Args:
        url (str): URL de la imagen
        filepath (Path): Ruta donde guardar la imagen
        logger (logging.Logger): Logger para registrar eventos
        
    Returns:
        Tuple[bool, Optional[str]]: (éxito, mensaje de error si aplica)
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=TIMEOUT_SECONDS, stream=True)
            response.raise_for_status()
            
            # Verificar que es una imagen válida
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return False, f"Contenido no es una imagen: {content_type}"
            
            # Guardar la imagen
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verificar que la imagen se puede abrir (validación adicional)
            try:
                with Image.open(filepath) as img:
                    img.verify()
                logger.debug(f"Imagen descargada exitosamente: {filepath}")
                return True, None
                
            except Exception as img_error:
                filepath.unlink(missing_ok=True)  # Eliminar archivo corrupto
                return False, f"Imagen corrupta: {str(img_error)}"
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Intento {attempt + 1}/{MAX_RETRIES} falló para {url}: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Backoff exponencial
            else:
                return False, f"Error después de {MAX_RETRIES} intentos: {str(e)}"
        
        except Exception as e:
            return False, f"Error inesperado: {str(e)}"
    
    return False, "Error desconocido"


def validate_csv_structure(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Valida que el DataFrame tenga la estructura esperada.
    
    Args:
        df (pd.DataFrame): DataFrame a validar
        
    Returns:
        Tuple[bool, str]: (es_válido, mensaje)
    """
    required_columns = ['item_id', 'site_id', 'domain_id', 'picture_id', 'correct_background?']
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        return False, f"Columnas faltantes: {missing_columns}"
    
    if df.empty:
        return False, "El DataFrame está vacío"
    
    # Verificar que no hay valores nulos en columnas críticas
    critical_columns = ['item_id', 'picture_id']
    for col in critical_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            return False, f"La columna '{col}' tiene {null_count} valores nulos"
    
    return True, "Estructura válida"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame eliminando registros problemáticos.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame limpio
    """
    # Hacer una copia para no modificar el original
    df_clean = df.copy()
    
    # Eliminar filas con picture_id nulo o vacío
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['item_id', 'picture_id'])
    df_clean = df_clean[df_clean['picture_id'].str.strip() != '']
    
    final_count = len(df_clean)
    if initial_count != final_count:
        print(f"Se eliminaron {initial_count - final_count} registros con datos faltantes")
    
    return df_clean


def save_results_dataframe(download_results: list, output_path: Path) -> pd.DataFrame:
    """
    Crea y guarda un DataFrame con los resultados de la descarga.
    
    Args:
        download_results (list): Lista de resultados de descarga
        output_path (Path): Ruta donde guardar el CSV
        
    Returns:
        pd.DataFrame: DataFrame con los resultados
    """
    results_df = pd.DataFrame(download_results)
    results_df.to_csv(output_path, index=False)
    
    return results_df