"""
Script principal para descargar imágenes del dataset de MercadoLibre.

Este script lee el archivo training_data.csv, descarga las imágenes correspondientes
y genera un nuevo dataset con las rutas de las imágenes descargadas.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Añadir el directorio scripts al path para importaciones
sys.path.append(str(Path(__file__).parent))

try:
    from .config import (
        TRAINING_DATA_PATH,
        IMAGES_DIR,
        DATA_DIR,
        BATCH_SIZE,
        MAX_WORKERS,
        CHUNK_SIZE
    )
    from .utils import (
        setup_logging,
        build_image_url,
        generate_filename,
        download_image,
        validate_csv_structure,
        clean_dataframe,
        save_results_dataframe
    )
except ImportError:
    # Fallback para ejecución directa
    import config
    import utils
    
    TRAINING_DATA_PATH = config.TRAINING_DATA_PATH
    IMAGES_DIR = config.IMAGES_DIR
    DATA_DIR = config.DATA_DIR
    BATCH_SIZE = config.BATCH_SIZE
    MAX_WORKERS = config.MAX_WORKERS
    CHUNK_SIZE = config.CHUNK_SIZE
    
    setup_logging = utils.setup_logging
    build_image_url = utils.build_image_url
    generate_filename = utils.generate_filename
    download_image = utils.download_image
    validate_csv_structure = utils.validate_csv_structure
    clean_dataframe = utils.clean_dataframe
    save_results_dataframe = utils.save_results_dataframe


class ThreadSafeProgressBar:
    """
    Barra de progreso thread-safe para descargas paralelas.
    """
    
    def __init__(self, total: int, desc: str = "Descargando"):
        self.pbar = tqdm(total=total, desc=desc)
        self.lock = threading.Lock()
    
    def update(self, n: int = 1):
        """Actualiza la barra de progreso de forma thread-safe."""
        with self.lock:
            self.pbar.update(n)
    
    def close(self):
        """Cierra la barra de progreso."""
        self.pbar.close()


class ImageDownloader:
    """
    Clase principal para manejar la descarga de imágenes.
    """
    
    def __init__(self):
        """Inicializa el descargador de imágenes."""
        self.logger = setup_logging()
        self.download_results: List[Dict[str, Any]] = []
        self.results_lock = threading.Lock()  # Lock para resultados thread-safe
        
    def load_training_data(self) -> pd.DataFrame:
        """
        Carga y valida el archivo de datos de entrenamiento.
        
        Returns:
            pd.DataFrame: DataFrame con los datos de entrenamiento
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si la estructura del archivo es inválida
        """
        if not TRAINING_DATA_PATH.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {TRAINING_DATA_PATH}")
        
        self.logger.info(f"Cargando datos desde: {TRAINING_DATA_PATH}")
        df = pd.read_csv(TRAINING_DATA_PATH)
        
        # Validar estructura
        is_valid, message = validate_csv_structure(df)
        if not is_valid:
            raise ValueError(f"Estructura de CSV inválida: {message}")
        
        # Limpiar datos
        df_clean = clean_dataframe(df)
        
        self.logger.info(f"Datos cargados exitosamente. Registros: {len(df_clean)}")
        return df_clean
    
    def download_single_image(self, row: pd.Series) -> Dict[str, Any]:
        """
        Descarga una sola imagen.
        
        Args:
            row (pd.Series): Fila del DataFrame con los datos del item
            
        Returns:
            Dict[str, Any]: Resultado de la descarga
        """
        item_id = row['item_id']
        picture_id = row['picture_id']
        
        # Construir URL y nombre de archivo
        image_url = build_image_url(picture_id)
        filename = generate_filename(item_id, picture_id)
        filepath = IMAGES_DIR / filename
        
        # Información del resultado
        result = {
            'item_id': item_id,
            'site_id': row['site_id'],
            'domain_id': row['domain_id'],
            'picture_id': picture_id,
            'correct_background': row['correct_background?'],
            'image_url': image_url,
            'local_path': str(filepath.relative_to(DATA_DIR.parent)),
            'filename': filename,
            'download_success': False,
            'error_message': None,
            'file_exists': False
        }
        
        # Verificar si ya existe el archivo
        if filepath.exists():
            result['download_success'] = True
            result['file_exists'] = True
            self.logger.debug(f"Archivo ya existe: {filename}")
            return result
        
        # Descargar imagen
        success, error_message = download_image(image_url, filepath, self.logger)
        
        result['download_success'] = success
        result['error_message'] = error_message
        
        if success:
            self.logger.debug(f"Descarga exitosa: {filename}")
        else:
            self.logger.warning(f"Error descargando {filename}: {error_message}")
        
        return result
    
    def download_single_image_threaded(self, row: pd.Series, progress_bar: ThreadSafeProgressBar) -> Dict[str, Any]:
        """
        Descarga una sola imagen en un hilo separado.
        
        Args:
            row (pd.Series): Fila del DataFrame con los datos del item
            progress_bar (ThreadSafeProgressBar): Barra de progreso thread-safe
            
        Returns:
            Dict[str, Any]: Resultado de la descarga
        """
        result = self.download_single_image(row)
        progress_bar.update(1)
        
        # Añadir resultado de forma thread-safe
        with self.results_lock:
            self.download_results.append(result)
        
        return result
    
    def download_images_chunk_parallel(self, df_chunk: pd.DataFrame, chunk_num: int) -> List[Dict[str, Any]]:
        """
        Descarga un chunk de imágenes usando paralelización.
        
        Args:
            df_chunk (pd.DataFrame): Chunk del DataFrame
            chunk_num (int): Número del chunk
            
        Returns:
            List[Dict[str, Any]]: Lista de resultados de descarga
        """
        chunk_size = len(df_chunk)
        
        self.logger.info(f"Procesando chunk {chunk_num} ({chunk_size} imágenes) con {MAX_WORKERS} workers")
        
        # Crear barra de progreso para este chunk
        progress_bar = ThreadSafeProgressBar(
            total=chunk_size, 
            desc=f"Chunk {chunk_num} ({MAX_WORKERS} workers)"
        )
        
        chunk_results = []
        
        # Usar ThreadPoolExecutor para paralelización
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Enviar todas las tareas al pool
            future_to_row = {
                executor.submit(self.download_single_image_threaded, row, progress_bar): row
                for _, row in df_chunk.iterrows()
            }
            
            # Recopilar resultados conforme se completan
            for future in as_completed(future_to_row):
                try:
                    result = future.result()
                    chunk_results.append(result)
                except Exception as exc:
                    row = future_to_row[future]
                    self.logger.error(f"Error descargando imagen {row['item_id']}: {exc}")
                    
                    # Crear resultado de error
                    error_result = {
                        'item_id': row['item_id'],
                        'site_id': row['site_id'],
                        'domain_id': row['domain_id'],
                        'picture_id': row['picture_id'],
                        'correct_background': row['correct_background?'],
                        'image_url': build_image_url(row['picture_id']),
                        'local_path': '',
                        'filename': '',
                        'download_success': False,
                        'error_message': f"Error en hilo: {str(exc)}",
                        'file_exists': False
                    }
                    chunk_results.append(error_result)
                    progress_bar.update(1)
        
        progress_bar.close()
        return chunk_results
    
    def download_images_batch(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """
        Descarga un lote de imágenes usando paralelización.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            start_idx (int): Índice de inicio
            end_idx (int): Índice de fin
            
        Returns:
            List[Dict[str, Any]]: Lista de resultados de descarga
        """
        batch_df = df.iloc[start_idx:end_idx]
        batch_size = len(batch_df)
        
        self.logger.info(f"Procesando lote {start_idx}-{end_idx} ({batch_size} imágenes)")
        
        # Dividir el lote en chunks más pequeños para procesamiento paralelo
        all_results = []
        
        for chunk_start in range(0, batch_size, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, batch_size)
            chunk_df = batch_df.iloc[chunk_start:chunk_end]
            chunk_num = (start_idx // BATCH_SIZE) + 1 + (chunk_start // CHUNK_SIZE)
            
            chunk_results = self.download_images_chunk_parallel(chunk_df, chunk_num)
            all_results.extend(chunk_results)
        
        return all_results
    
    def download_all_images(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Descarga todas las imágenes del DataFrame usando paralelización.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos
            
        Returns:
            List[Dict[str, Any]]: Lista de todos los resultados
        """
        total_images = len(df)
        self.logger.info(f"Iniciando descarga paralela de {total_images} imágenes con {MAX_WORKERS} workers")
        
        # Limpiar resultados previos
        self.download_results = []
        all_results = []
        
        # Procesar en lotes
        for start_idx in range(0, total_images, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_images)
            
            batch_results = self.download_images_batch(df, start_idx, end_idx)
            all_results.extend(batch_results)
            
            # Estadísticas del lote
            successful = sum(1 for r in batch_results if r['download_success'])
            self.logger.info(f"Lote completado. Exitosas: {successful}/{len(batch_results)}")
        
        return all_results
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> None:
        """
        Genera un reporte resumen de la descarga.
        
        Args:
            results (List[Dict[str, Any]]): Resultados de la descarga
        """
        total = len(results)
        successful = sum(1 for r in results if r['download_success'])
        already_existed = sum(1 for r in results if r['file_exists'])
        failed = total - successful
        
        self.logger.info("="*50)
        self.logger.info("REPORTE DE DESCARGA PARALELA")
        self.logger.info("="*50)
        self.logger.info(f"Total de imágenes: {total}")
        self.logger.info(f"Descargas exitosas: {successful}")
        self.logger.info(f"Ya existían: {already_existed}")
        self.logger.info(f"Fallos: {failed}")
        self.logger.info(f"Tasa de éxito: {(successful/total)*100:.2f}%")
        self.logger.info(f"Workers concurrentes utilizados: {MAX_WORKERS}")
        
        if failed > 0:
            # Mostrar errores más comunes
            error_counts = {}
            for result in results:
                if not result['download_success'] and result['error_message']:
                    error = result['error_message']
                    error_counts[error] = error_counts.get(error, 0) + 1
            
            self.logger.info("\nErrores más comunes:")
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                self.logger.info(f"  {error}: {count} veces")
    
    def run(self, limit: int = None) -> str:
        """
        Ejecuta el proceso completo de descarga.
        
        Args:
            limit (int, optional): Límite de imágenes a procesar (para testing)
            
        Returns:
            str: Path del archivo CSV con resultados
        """
        try:
            # Cargar datos
            df = self.load_training_data()
            
            # Aplicar límite si se especifica (útil para testing)
            if limit:
                df = df.head(limit)
                self.logger.info(f"Limitando a {limit} imágenes para testing")
            
            # Descargar imágenes
            results = self.download_all_images(df)
            
            # Generar reporte
            self.generate_summary_report(results)
            
            # Guardar resultados
            output_path = DATA_DIR / "downloaded_images_dataset.csv"
            results_df = save_results_dataframe(results, output_path)
            
            self.logger.info(f"Resultados guardados en: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error durante la ejecución: {str(e)}")
            raise


def main():
    """Función principal del script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Descarga imágenes del dataset de MercadoLibre")
    parser.add_argument("--limit", type=int, help="Límite de imágenes a procesar (para testing)")
    parser.add_argument("--test", action="store_true", help="Ejecutar en modo test con 10 imágenes")
    
    args = parser.parse_args()
    
    # Configurar límite
    limit = None
    if args.test:
        limit = 10
    elif args.limit:
        limit = args.limit
    
    # Ejecutar descarga
    downloader = ImageDownloader()
    output_file = downloader.run(limit=limit)
    
    print(f"\n✅ Proceso completado!")
    print(f"📁 Imágenes guardadas en: {IMAGES_DIR}")
    print(f"📊 Dataset con rutas guardado en: {output_file}")


if __name__ == "__main__":
    main()