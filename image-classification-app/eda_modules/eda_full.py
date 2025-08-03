"""
Módulo EDAFull para análisis exploratorio completo del dataset.

Orquesta la carga de datasets, aplicación de análisis y generación
de estadísticas descriptivas.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')

# Importar módulos locales
import sys
sys.path.append(str(Path(__file__).parent))

from image_analyzer import ImageAnalyzer
from segmentation import ImageSegmentation


class EDAFull:
    """
    Clase para análisis exploratorio completo del dataset de imágenes.
    """
    
    def __init__(self, data_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Inicializa el analizador EDA completo.
        
        Args:
            data_dir: Directorio con los datos
            logger: Logger opcional
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.logger = logger or logging.getLogger(__name__)
        
        # Inicializar analizadores
        self.image_analyzer = ImageAnalyzer(self.logger)
        self.segmentation = ImageSegmentation(self.logger)
        
        # DataFrames
        self.dataset_df = None
        self.analysis_results = None
        
        # Configuración
        self.output_dir = self.data_dir / "eda_results"
        self.output_dir.mkdir(exist_ok=True)
    
    def load_datasets(self) -> bool:
        """
        Carga los datasets necesarios para el análisis.
        
        Returns:
            True si la carga fue exitosa
        """
        try:
            # Cargar dataset principal
            training_data_path = self.data_dir / "training_data.csv"
            if not training_data_path.exists():
                self.logger.error(f"No se encontró el archivo: {training_data_path}")
                return False
            
            self.dataset_df = pd.read_csv(training_data_path)
            self.logger.info(f"Dataset cargado: {len(self.dataset_df)} registros")
            
            # Cargar dataset con rutas de imágenes si existe
            downloaded_data_path = self.data_dir / "downloaded_images_dataset.csv"
            if downloaded_data_path.exists():
                self.downloaded_df = pd.read_csv(downloaded_data_path)
                self.logger.info(f"Dataset de imágenes descargadas cargado: {len(self.downloaded_df)} registros")
            else:
                self.downloaded_df = None
                self.logger.warning("No se encontró dataset de imágenes descargadas")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando datasets: {e}")
            return False
    
    def analyze_image_quality(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Analiza la calidad de las imágenes disponibles.
        
        Args:
            sample_size: Tamaño de muestra (None para todas las disponibles)
            
        Returns:
            DataFrame con métricas de calidad
        """
        try:
            # Obtener lista de imágenes disponibles
            if self.downloaded_df is not None:
                # Usar dataset de imágenes descargadas
                available_images = self.downloaded_df[
                    self.downloaded_df['download_success'] == True
                ].copy()
            else:
                # Usar dataset original y verificar qué imágenes existen
                available_images = self.dataset_df.copy()
                available_images['image_path'] = available_images['picture_id'].apply(
                    lambda x: self.images_dir / f"{x}.jpg"
                )
                available_images['exists'] = available_images['image_path'].apply(lambda x: x.exists())
                available_images = available_images[available_images['exists'] == True]
            
            # Aplicar sample si se especifica
            if sample_size and len(available_images) > sample_size:
                available_images = available_images.sample(n=sample_size, random_state=42)
            
            self.logger.info(f"Analizando calidad de {len(available_images)} imágenes")
            
            # Analizar cada imagen
            quality_results = []
            
            for _, row in tqdm(available_images.iterrows(), total=len(available_images), 
                              desc="Analizando calidad"):
                try:
                    # Determinar ruta de imagen
                    if 'local_path' in row:
                        image_path = self.data_dir.parent / row['local_path']
                    else:
                        image_path = row['image_path']
                    
                    # Analizar imagen
                    metrics = self.image_analyzer.analyze_single_image(image_path)
                    
                    if metrics:
                        # Añadir información del dataset
                        metrics['item_id'] = row['item_id']
                        metrics['correct_background'] = row['correct_background?']
                        metrics['site_id'] = row['site_id']
                        metrics['domain_id'] = row['domain_id']
                        quality_results.append(metrics)
                        
                except Exception as e:
                    self.logger.warning(f"Error analizando imagen {row.get('item_id', 'unknown')}: {e}")
                    continue
            
            # Crear DataFrame con resultados
            if quality_results:
                self.quality_df = pd.DataFrame(quality_results)
                self.logger.info(f"Análisis de calidad completado: {len(self.quality_df)} imágenes")
                return self.quality_df
            else:
                self.logger.error("No se pudieron analizar imágenes")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error en análisis de calidad: {e}")
            return pd.DataFrame()
    
    def analyze_target_balance(self) -> Dict[str, any]:
        """
        Analiza el balance de la variable objetivo "correct_background?".
        
        Returns:
            Diccionario con estadísticas del balance
        """
        try:
            if self.dataset_df is None:
                self.logger.error("Dataset no cargado")
                return {}
            
            # Análisis básico
            target_counts = self.dataset_df['correct_background?'].value_counts()
            target_balance = target_counts / len(self.dataset_df)
            
            # Análisis por sitio
            site_balance = self.dataset_df.groupby('site_id')['correct_background?'].value_counts(normalize=True)
            
            # Análisis por dominio
            domain_balance = self.dataset_df.groupby('domain_id')['correct_background?'].value_counts(normalize=True)
            
            balance_stats = {
                'total_samples': len(self.dataset_df),
                'target_distribution': target_counts.to_dict(),
                'target_balance': target_balance.to_dict(),
                'site_balance': site_balance.to_dict(),
                'domain_balance': domain_balance.to_dict(),
                'imbalance_ratio': target_counts.max() / target_counts.min() if len(target_counts) > 1 else 1
            }
            
            self.logger.info(f"Balance de target analizado: {balance_stats['imbalance_ratio']:.2f} ratio de desbalance")
            return balance_stats
            
        except Exception as e:
            self.logger.error(f"Error analizando balance del target: {e}")
            return {}
    
    def analyze_image_sizes(self) -> Dict[str, any]:
        """
        Analiza las dimensiones y tamaños de las imágenes.
        
        Returns:
            Diccionario con estadísticas de tamaños
        """
        try:
            if not hasattr(self, 'quality_df') or self.quality_df.empty:
                self.logger.warning("No hay datos de calidad disponibles. Ejecutando análisis...")
                self.analyze_image_quality()
            
            if self.quality_df.empty:
                return {}
            
            # Estadísticas de dimensiones
            size_stats = {
                'width_stats': self.quality_df['width'].describe(),
                'height_stats': self.quality_df['height'].describe(),
                'aspect_ratio_stats': self.quality_df['aspect_ratio'].describe(),
                'total_pixels_stats': self.quality_df['total_pixels'].describe(),
                
                # Categorías de tamaño
                'size_categories': {
                    'small': len(self.quality_df[self.quality_df['total_pixels'] < 100000]),
                    'medium': len(self.quality_df[(self.quality_df['total_pixels'] >= 100000) & 
                                                 (self.quality_df['total_pixels'] < 500000)]),
                    'large': len(self.quality_df[self.quality_df['total_pixels'] >= 500000])
                },
                
                # Aspect ratios comunes
                'aspect_ratio_categories': {
                    'square': len(self.quality_df[abs(self.quality_df['aspect_ratio'] - 1) < 0.1]),
                    'portrait': len(self.quality_df[self.quality_df['aspect_ratio'] < 0.9]),
                    'landscape': len(self.quality_df[self.quality_df['aspect_ratio'] > 1.1])
                }
            }
            
            self.logger.info(f"Análisis de tamaños completado: {len(self.quality_df)} imágenes")
            return size_stats
            
        except Exception as e:
            self.logger.error(f"Error analizando tamaños de imagen: {e}")
            return {}
    
    def generate_descriptive_statistics(self) -> pd.DataFrame:
        """
        Genera estadísticas descriptivas completas del dataset.
        
        Returns:
            DataFrame con estadísticas descriptivas
        """
        try:
            if not hasattr(self, 'quality_df') or self.quality_df.empty:
                self.logger.warning("No hay datos de calidad disponibles. Ejecutando análisis...")
                self.analyze_image_quality()
            
            if self.quality_df.empty:
                return pd.DataFrame()
            
            # Seleccionar columnas numéricas para estadísticas
            numeric_columns = self.quality_df.select_dtypes(include=[np.number]).columns
            
            # Generar estadísticas descriptivas
            descriptive_stats = self.quality_df[numeric_columns].describe()
            
            # Añadir estadísticas adicionales
            additional_stats = pd.DataFrame({
                'skewness': self.quality_df[numeric_columns].skew(),
                'kurtosis': self.quality_df[numeric_columns].kurtosis(),
                'missing_values': self.quality_df[numeric_columns].isnull().sum()
            })
            
            # Combinar estadísticas
            complete_stats = pd.concat([descriptive_stats, additional_stats])
            
            self.logger.info(f"Estadísticas descriptivas generadas para {len(numeric_columns)} variables")
            return complete_stats
            
        except Exception as e:
            self.logger.error(f"Error generando estadísticas descriptivas: {e}")
            return pd.DataFrame()
    
    def create_visualizations(self) -> None:
        """
        Crea visualizaciones del análisis exploratorio.
        """
        try:
            if not hasattr(self, 'quality_df') or self.quality_df.empty:
                self.logger.warning("No hay datos de calidad disponibles para visualizaciones")
                return
            
            # Configurar estilo
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Distribución de la variable objetivo
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Target distribution
            target_counts = self.quality_df['correct_background'].value_counts()
            axes[0, 0].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Distribución de correct_background')
            
            # Image sizes
            axes[0, 1].scatter(self.quality_df['width'], self.quality_df['height'], 
                              alpha=0.6, c=self.quality_df['correct_background'])
            axes[0, 1].set_xlabel('Ancho')
            axes[0, 1].set_ylabel('Alto')
            axes[0, 1].set_title('Dimensiones de Imágenes')
            
            # Aspect ratio distribution
            axes[0, 2].hist(self.quality_df['aspect_ratio'], bins=30, alpha=0.7)
            axes[0, 2].set_xlabel('Aspect Ratio')
            axes[0, 2].set_ylabel('Frecuencia')
            axes[0, 2].set_title('Distribución de Aspect Ratio')
            
            # Brightness vs Contrast
            axes[1, 0].scatter(self.quality_df['brightness'], self.quality_df['contrast'], 
                              alpha=0.6, c=self.quality_df['correct_background'])
            axes[1, 0].set_xlabel('Brillo')
            axes[1, 0].set_ylabel('Contraste')
            axes[1, 0].set_title('Brillo vs Contraste')
            
            # Background ratio
            axes[1, 1].hist(self.quality_df['background_ratio'], bins=30, alpha=0.7)
            axes[1, 1].set_xlabel('Proporción de Fondo')
            axes[1, 1].set_ylabel('Frecuencia')
            axes[1, 1].set_title('Distribución de Proporción de Fondo')
            
            # Sharpness distribution
            axes[1, 2].hist(self.quality_df['sharpness'], bins=30, alpha=0.7)
            axes[1, 2].set_xlabel('Nitidez')
            axes[1, 2].set_ylabel('Frecuencia')
            axes[1, 2].set_title('Distribución de Nitidez')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'eda_overview.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 2. Análisis por sitio y dominio
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Balance por sitio
            site_balance = self.quality_df.groupby('site_id')['correct_background'].value_counts(normalize=True)
            site_balance.unstack().plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Balance por Sitio')
            axes[0, 0].set_ylabel('Proporción')
            
            # Balance por dominio (top 10)
            domain_balance = self.quality_df.groupby('domain_id')['correct_background'].value_counts(normalize=True)
            domain_balance.unstack().head(10).plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Balance por Dominio (Top 10)')
            axes[0, 1].set_ylabel('Proporción')
            
            # Calidad por sitio
            quality_by_site = self.quality_df.groupby('site_id')[['brightness', 'contrast', 'sharpness']].mean()
            quality_by_site.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Calidad Promedio por Sitio')
            axes[1, 0].set_ylabel('Valor Promedio')
            
            # Correlación con target
            numeric_cols = ['brightness', 'contrast', 'sharpness', 'entropy', 'background_ratio']
            correlations = self.quality_df[numeric_cols + ['correct_background']].corr()['correct_background'].drop('correct_background')
            correlations.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Correlación con correct_background')
            axes[1, 1].set_ylabel('Correlación')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'eda_by_categories.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            self.logger.info(f"Visualizaciones guardadas en: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creando visualizaciones: {e}")
    
    def export_results(self) -> None:
        """
        Exporta todos los resultados del análisis a archivos CSV.
        """
        try:
            # Exportar dataset con métricas
            if hasattr(self, 'quality_df') and not self.quality_df.empty:
                output_path = self.output_dir / "image_quality_analysis.csv"
                self.quality_df.to_csv(output_path, index=False)
                self.logger.info(f"Análisis de calidad exportado a: {output_path}")
            
            # Exportar estadísticas descriptivas
            descriptive_stats = self.generate_descriptive_statistics()
            if not descriptive_stats.empty:
                output_path = self.output_dir / "descriptive_statistics.csv"
                descriptive_stats.to_csv(output_path)
                self.logger.info(f"Estadísticas descriptivas exportadas a: {output_path}")
            
            # Exportar análisis de balance
            balance_stats = self.analyze_target_balance()
            if balance_stats:
                balance_df = pd.DataFrame([balance_stats])
                output_path = self.output_dir / "target_balance_analysis.csv"
                balance_df.to_csv(output_path, index=False)
                self.logger.info(f"Análisis de balance exportado a: {output_path}")
            
            # Exportar análisis de tamaños
            size_stats = self.analyze_image_sizes()
            if size_stats:
                # Convertir estadísticas a DataFrame
                size_data = {}
                for key, value in size_stats.items():
                    if isinstance(value, pd.Series):
                        size_data[key] = value.to_dict()
                    else:
                        size_data[key] = value
                
                size_df = pd.DataFrame([size_data])
                output_path = self.output_dir / "image_size_analysis.csv"
                size_df.to_csv(output_path, index=False)
                self.logger.info(f"Análisis de tamaños exportado a: {output_path}")
            
            self.logger.info(f"Todos los resultados exportados a: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error exportando resultados: {e}")
    
    def run_full_analysis(self, sample_size: Optional[int] = None) -> Dict[str, any]:
        """
        Ejecuta el análisis exploratorio completo.
        
        Args:
            sample_size: Tamaño de muestra para análisis de calidad
            
        Returns:
            Diccionario con resumen de resultados
        """
        try:
            self.logger.info("Iniciando análisis exploratorio completo...")
            
            # 1. Cargar datasets
            if not self.load_datasets():
                return {}
            
            # 2. Analizar calidad de imágenes
            quality_df = self.analyze_image_quality(sample_size)
            
            # 3. Analizar balance del target
            balance_stats = self.analyze_target_balance()
            
            # 4. Analizar tamaños de imagen
            size_stats = self.analyze_image_sizes()
            
            # 5. Generar estadísticas descriptivas
            descriptive_stats = self.generate_descriptive_statistics()
            
            # 6. Crear visualizaciones
            self.create_visualizations()
            
            # 7. Exportar resultados
            self.export_results()
            
            # Resumen de resultados
            summary = {
                'total_images_analyzed': len(quality_df) if not quality_df.empty else 0,
                'target_balance_ratio': balance_stats.get('imbalance_ratio', 0),
                'size_categories': size_stats.get('size_categories', {}),
                'output_directory': str(self.output_dir),
                'analysis_completed': True
            }
            
            self.logger.info("Análisis exploratorio completo finalizado")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error en análisis completo: {e}")
            return {} 