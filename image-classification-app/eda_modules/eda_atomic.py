"""
Módulo EDAAtomic para análisis individual de imágenes.

Permite visualizar muestras aleatorias, aplicar métodos de segmentación
y extraer características del fondo.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import random
import logging
import warnings
warnings.filterwarnings('ignore')

# Importar módulos locales
import sys
sys.path.append(str(Path(__file__).parent))

from image_analyzer import ImageAnalyzer
from segmentation import ImageSegmentation


class EDAAtomic:
    """
    Clase para análisis atómico (individual) de imágenes.
    """
    
    def __init__(self, data_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Inicializa el analizador EDA atómico.
        
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
        self.downloaded_df = None
        
        # Configuración
        self.output_dir = self.data_dir / "eda_atomic_results"
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
    
    def get_random_sample(self, n_samples: int = 3, balance_by_target: bool = True) -> pd.DataFrame:
        """
        Obtiene una muestra aleatoria de imágenes.
        
        Args:
            n_samples: Número de imágenes a muestrear
            balance_by_target: Si balancear por la variable objetivo
            
        Returns:
            DataFrame con la muestra seleccionada
        """
        try:
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
            
            if balance_by_target and 'correct_background?' in available_images.columns:
                # Balancear por target
                target_values = available_images['correct_background?'].unique()
                samples_per_target = n_samples // len(target_values)
                
                balanced_sample = []
                for target_value in target_values:
                    target_images = available_images[available_images['correct_background?'] == target_value]
                    if len(target_images) >= samples_per_target:
                        sample = target_images.sample(n=samples_per_target, random_state=42)
                    else:
                        sample = target_images  # Usar todas las disponibles
                    balanced_sample.append(sample)
                
                sample_df = pd.concat(balanced_sample, ignore_index=True)
                
                # Si no tenemos suficientes muestras, completar aleatoriamente
                if len(sample_df) < n_samples:
                    remaining = available_images[~available_images.index.isin(sample_df.index)]
                    additional = remaining.sample(n=n_samples - len(sample_df), random_state=42)
                    sample_df = pd.concat([sample_df, additional], ignore_index=True)
                
            else:
                # Muestra aleatoria simple
                sample_df = available_images.sample(n=min(n_samples, len(available_images)), random_state=42)
            
            self.logger.info(f"Muestra aleatoria seleccionada: {len(sample_df)} imágenes")
            return sample_df
            
        except Exception as e:
            self.logger.error(f"Error obteniendo muestra aleatoria: {e}")
            return pd.DataFrame()
    
    def visualize_random_sample(self, n_samples: int = 3, balance_by_target: bool = True, 
                               save_path: Optional[Path] = None) -> None:
        """
        Visualiza una muestra aleatoria de imágenes con sus etiquetas.
        
        Args:
            n_samples: Número de imágenes a mostrar
            balance_by_target: Si balancear por la variable objetivo
            save_path: Ruta opcional para guardar la visualización
        """
        try:
            # Obtener muestra aleatoria
            sample_df = self.get_random_sample(n_samples, balance_by_target)
            
            if sample_df.empty:
                self.logger.error("No se pudo obtener muestra de imágenes")
                return
            
            # Crear figura
            fig, axes = plt.subplots(1, n_samples, figsize=(5*n_samples, 5))
            if n_samples == 1:
                axes = [axes]
            
            for i, (_, row) in enumerate(sample_df.iterrows()):
                try:
                    # Determinar ruta de imagen
                    if 'local_path' in row:
                        image_path = self.data_dir.parent / row['local_path']
                    else:
                        image_path = row['image_path']
                    
                    # Cargar y mostrar imagen
                    image = self.image_analyzer.load_image(image_path)
                    if image is not None:
                        axes[i].imshow(image)
                        
                        # Título con información
                        target_value = row.get('correct_background?', 'N/A')
                        item_id = row.get('item_id', 'N/A')
                        title = f"Item: {item_id}\nTarget: {target_value}"
                        axes[i].set_title(title, fontsize=10)
                        
                        # Añadir borde de color según target
                        if target_value == 1:
                            color = 'green'  # Fondo correcto
                        elif target_value == 0:
                            color = 'red'    # Fondo incorrecto
                        else:
                            color = 'gray'   # Desconocido
                        
                        # Añadir borde
                        for spine in axes[i].spines.values():
                            spine.set_color(color)
                            spine.set_linewidth(3)
                        
                        axes[i].axis('off')
                    else:
                        axes[i].text(0.5, 0.5, 'Error cargando imagen', 
                                   ha='center', va='center', transform=axes[i].transAxes)
                        axes[i].set_title(f"Error: {row.get('item_id', 'N/A')}")
                        axes[i].axis('off')
                        
                except Exception as e:
                    self.logger.warning(f"Error procesando imagen {i}: {e}")
                    axes[i].text(0.5, 0.5, 'Error', ha='center', va='center', 
                               transform=axes[i].transAxes)
                    axes[i].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Visualización de muestra guardada en: {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error creando visualización de muestra: {e}")
    
    def analyze_segmentation_methods(self, image_path: Path) -> Dict[str, Dict]:
        """
        Analiza una imagen usando los tres métodos de segmentación.
        
        Args:
            image_path: Ruta de la imagen a analizar
            
        Returns:
            Diccionario con resultados de los tres métodos
        """
        try:
            # Cargar imagen
            image = self.image_analyzer.load_image(image_path)
            if image is None:
                return {}
            
            # Aplicar los tres métodos de segmentación
            segmentation_results = self.segmentation.compare_segmentation_methods(image)
            
            # Añadir métricas de calidad de la imagen
            quality_metrics = self.image_analyzer.analyze_single_image(image_path)
            
            # Combinar resultados
            complete_results = {
                'image_path': str(image_path),
                'quality_metrics': quality_metrics,
                'segmentation_results': segmentation_results
            }
            
            self.logger.info(f"Análisis de segmentación completado para: {image_path.name}")
            return complete_results
            
        except Exception as e:
            self.logger.error(f"Error analizando segmentación: {e}")
            return {}
    
    def visualize_segmentation_analysis(self, image_path: Path, 
                                      save_path: Optional[Path] = None) -> None:
        """
        Visualiza el análisis de segmentación de una imagen.
        
        Args:
            image_path: Ruta de la imagen
            save_path: Ruta opcional para guardar la visualización
        """
        try:
            # Cargar imagen
            image = self.image_analyzer.load_image(image_path)
            if image is None:
                return
            
            # Obtener resultados de segmentación
            segmentation_results = self.segmentation.compare_segmentation_methods(image)
            
            # Obtener métricas de calidad
            quality_metrics = self.image_analyzer.analyze_single_image(image_path)
            
            # Crear visualización
            self.segmentation.visualize_segmentation_comparison(image, segmentation_results, save_path)
            
            # Mostrar métricas de calidad
            if quality_metrics:
                self.image_analyzer.plot_image_with_metrics(image_path, quality_metrics)
            
        except Exception as e:
            self.logger.error(f"Error visualizando análisis de segmentación: {e}")
    
    def extract_background_features_sample(self, n_samples: int = 3) -> pd.DataFrame:
        """
        Extrae características del fondo de una muestra de imágenes.
        
        Args:
            n_samples: Número de imágenes a analizar
            
        Returns:
            DataFrame con características del fondo
        """
        try:
            # Obtener muestra aleatoria
            sample_df = self.get_random_sample(n_samples, balance_by_target=True)
            
            if sample_df.empty:
                return pd.DataFrame()
            
            background_features = []
            
            for _, row in sample_df.iterrows():
                try:
                    # Determinar ruta de imagen
                    if 'local_path' in row:
                        image_path = self.data_dir.parent / row['local_path']
                    else:
                        image_path = row['image_path']
                    
                    # Cargar imagen
                    image = self.image_analyzer.load_image(image_path)
                    if image is None:
                        continue
                    
                    # Aplicar métodos de segmentación
                    segmentation_results = self.segmentation.compare_segmentation_methods(image)
                    
                    # Extraer características del fondo para cada método
                    for method_name, method_result in segmentation_results.items():
                        if 'background_mask' in method_result:
                            bg_features = self.segmentation.extract_background_features(
                                image, method_result['background_mask']
                            )
                            
                            if bg_features:
                                # Añadir información de la imagen
                                bg_features.update({
                                    'item_id': row.get('item_id', 'N/A'),
                                    'correct_background': row.get('correct_background?', 'N/A'),
                                    'site_id': row.get('site_id', 'N/A'),
                                    'domain_id': row.get('domain_id', 'N/A'),
                                    'segmentation_method': method_name,
                                    'image_path': str(image_path)
                                })
                                
                                background_features.append(bg_features)
                    
                except Exception as e:
                    self.logger.warning(f"Error analizando imagen {row.get('item_id', 'unknown')}: {e}")
                    continue
            
            # Crear DataFrame
            if background_features:
                features_df = pd.DataFrame(background_features)
                self.logger.info(f"Características del fondo extraídas: {len(features_df)} registros")
                return features_df
            else:
                self.logger.warning("No se pudieron extraer características del fondo")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error extrayendo características del fondo: {e}")
            return pd.DataFrame()
    
    def compare_background_features(self, features_df: pd.DataFrame, 
                                  save_path: Optional[Path] = None) -> None:
        """
        Compara las características del fondo entre métodos de segmentación.
        
        Args:
            features_df: DataFrame con características del fondo
            save_path: Ruta opcional para guardar la visualización
        """
        try:
            if features_df.empty:
                self.logger.warning("No hay datos de características para comparar")
                return
            
            # Configurar estilo
            plt.style.use('seaborn-v0_8')
            
            # Crear visualizaciones
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Comparación de uniformidad por método
            if 'bg_uniformity' in features_df.columns:
                features_df.boxplot(column='bg_uniformity', by='segmentation_method', ax=axes[0, 0])
                axes[0, 0].set_title('Uniformidad del Fondo por Método')
                axes[0, 0].set_xlabel('Método de Segmentación')
                axes[0, 0].set_ylabel('Uniformidad')
            
            # 2. Comparación de brillo del fondo
            if 'bg_brightness' in features_df.columns:
                features_df.boxplot(column='bg_brightness', by='segmentation_method', ax=axes[0, 1])
                axes[0, 1].set_title('Brillo del Fondo por Método')
                axes[0, 1].set_xlabel('Método de Segmentación')
                axes[0, 1].set_ylabel('Brillo')
            
            # 3. Comparación de contraste del fondo
            if 'bg_contrast' in features_df.columns:
                features_df.boxplot(column='bg_contrast', by='segmentation_method', ax=axes[0, 2])
                axes[0, 2].set_title('Contraste del Fondo por Método')
                axes[0, 2].set_xlabel('Método de Segmentación')
                axes[0, 2].set_ylabel('Contraste')
            
            # 4. Distribución de saturación del fondo
            if 'bg_saturation_mean' in features_df.columns:
                for method in features_df['segmentation_method'].unique():
                    method_data = features_df[features_df['segmentation_method'] == method]['bg_saturation_mean']
                    axes[1, 0].hist(method_data, alpha=0.7, label=method, bins=10)
                axes[1, 0].set_title('Distribución de Saturación del Fondo')
                axes[1, 0].set_xlabel('Saturación')
                axes[1, 0].set_ylabel('Frecuencia')
                axes[1, 0].legend()
            
            # 5. Distribución de valor (brightness HSV) del fondo
            if 'bg_value_mean' in features_df.columns:
                for method in features_df['segmentation_method'].unique():
                    method_data = features_df[features_df['segmentation_method'] == method]['bg_value_mean']
                    axes[1, 1].hist(method_data, alpha=0.7, label=method, bins=10)
                axes[1, 1].set_title('Distribución de Valor del Fondo')
                axes[1, 1].set_xlabel('Valor (HSV)')
                axes[1, 1].set_ylabel('Frecuencia')
                axes[1, 1].legend()
            
            # 6. Proporción de área de fondo
            if 'bg_area_ratio' in features_df.columns:
                features_df.boxplot(column='bg_area_ratio', by='segmentation_method', ax=axes[1, 2])
                axes[1, 2].set_title('Proporción de Área de Fondo por Método')
                axes[1, 2].set_xlabel('Método de Segmentación')
                axes[1, 2].set_ylabel('Proporción de Área')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Comparación de características guardada en: {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error comparando características del fondo: {e}")
    
    def run_atomic_analysis(self, n_samples: int = 3) -> Dict[str, any]:
        """
        Ejecuta el análisis atómico completo.
        
        Args:
            n_samples: Número de imágenes para el análisis
            
        Returns:
            Diccionario con resumen de resultados
        """
        try:
            self.logger.info("Iniciando análisis atómico...")
            
            # 1. Cargar datasets
            if not self.load_datasets():
                return {}
            
            # 2. Visualizar muestra aleatoria
            self.visualize_random_sample(n_samples, balance_by_target=True)
            
            # 3. Extraer características del fondo
            background_features = self.extract_background_features_sample(n_samples)
            
            # 4. Comparar características del fondo
            if not background_features.empty:
                self.compare_background_features(background_features)
                
                # Guardar resultados
                output_path = self.output_dir / "background_features_analysis.csv"
                background_features.to_csv(output_path, index=False)
                self.logger.info(f"Características del fondo guardadas en: {output_path}")
            
            # Resumen de resultados
            summary = {
                'samples_analyzed': n_samples,
                'background_features_extracted': len(background_features) if not background_features.empty else 0,
                'segmentation_methods_tested': 3,
                'output_directory': str(self.output_dir),
                'analysis_completed': True
            }
            
            self.logger.info("Análisis atómico completado")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error en análisis atómico: {e}")
            return {} 