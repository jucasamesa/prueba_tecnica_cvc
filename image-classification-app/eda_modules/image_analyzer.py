"""
Módulo base para análisis de imágenes.

Contiene funciones para análisis de calidad, métricas y procesamiento
básico de imágenes.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import measure, filters, morphology
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops
import logging


class ImageAnalyzer:
    """
    Clase base para análisis de imágenes.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Inicializa el analizador de imágenes.
        
        Args:
            logger: Logger opcional para logging
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Carga una imagen desde el path especificado.
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Imagen como array numpy o None si hay error
        """
        try:
            # Usar OpenCV para cargar la imagen
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"No se pudo cargar la imagen: {image_path}")
                return None
            
            # Convertir de BGR a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_rgb
            
        except Exception as e:
            self.logger.error(f"Error cargando imagen {image_path}: {e}")
            return None
        
    def clean_gray_image(self, image: np.ndarray, num_pixels: int=100) -> Dict[str, float]:
        """
        Utilizar la librería skimage.morphology.remove_small_objects para eliminar aquellos objetos cuya área sea menor a 300 píxeles
        Más información en https://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.remove_small_objects
        
        Args:
            image: Imagen como array numpy
            
        Returns:
            clean image
        """
        try:
            img_clean = morphology.remove_small_objects(image.astype(np.bool), num_pixels).astype('uint8')

            return img_clean
        
        except Exception as e:
            self.error(f"Error limpiando imagen: {e}")
            return None
    
    def get_image_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de calidad de imagen.
        
        Args:
            image: Imagen como array numpy
            
        Returns:
            Diccionario con métricas de calidad
        """
        try:
            # Convertir a escala de grises para algunos cálculos
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Métricas básicas
            height, width = image.shape[:2]
            aspect_ratio = width / height
            total_pixels = width * height
            
            # Brillo promedio
            brightness = np.mean(gray)
            
            # Contraste (desviación estándar)
            contrast = np.std(gray)
            
            # Entropía (medida de complejidad)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / hist.sum()  # Normalizar
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            
            # Sharpness (nitidez) usando Laplaciano
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Ruido estimado (diferencia entre imagen y versión suavizada)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
            
            # Saturaciones de color
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:, :, 1])
            
            # Valor (brightness en HSV)
            value = np.mean(hsv[:, :, 2])
            
            return {
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'total_pixels': total_pixels,
                'brightness': brightness,
                'contrast': contrast,
                'entropy': entropy,
                'sharpness': laplacian_var,
                'noise_level': noise,
                'saturation': saturation,
                'value': value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando métricas de calidad: {e}")
            return {}
    
    def get_background_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analiza características del fondo de la imagen.
        
        Args:
            image: Imagen como array numpy
            
        Returns:
            Diccionario con métricas del fondo
        """
        try:
            # Convertir a diferentes espacios de color
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Limpiar imagen gris
            gray = self.clean_gray_image(gray)

            # Detectar bordes para identificar objetos
            edges = cv2.Canny(gray, 50, 150)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calcular área total de contornos (objetos)
            contour_area = sum(cv2.contourArea(c) for c in contours)
            total_area = image.shape[0] * image.shape[1]
            
            # Proporción de fondo vs objetos
            background_ratio = 1 - (contour_area / total_area)
            
            # Análisis de color del fondo (asumiendo que el fondo es el color más común)
            # Usar k-means para encontrar colores dominantes
            pixels = image.reshape(-1, 3)
            pixels = np.float32(pixels)
            
            # Aplicar k-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            k = 3
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Encontrar el color más común (probablemente el fondo)
            unique, counts = np.unique(labels, return_counts=True)
            dominant_color_idx = unique[np.argmax(counts)]
            dominant_color = centers[dominant_color_idx]
            
            # Convertir a HSV para análisis
            dominant_color_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_RGB2HSV)[0][0]
            
            return {
                'background_ratio': background_ratio,
                'object_ratio': 1 - background_ratio,
                'dominant_color_r': dominant_color[0],
                'dominant_color_g': dominant_color[1],
                'dominant_color_b': dominant_color[2],
                'dominant_color_h': dominant_color_hsv[0],
                'dominant_color_s': dominant_color_hsv[1],
                'dominant_color_v': dominant_color_hsv[2],
                'edge_density': np.sum(edges > 0) / total_area,
                'contour_count': len(contours)
            }
            
        except Exception as e:
            self.logger.error(f"Error analizando fondo: {e}")
            return {}
    
    def get_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae características de textura usando GLCM (Gray-Level Co-occurrence Matrix).
        
        Args:
            image: Imagen como array numpy
            
        Returns:
            Diccionario con características de textura
        """
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Redimensionar para consistencia
            gray = cv2.resize(gray, (128, 128))
            
            # Calcular GLCM
            distances = [1, 2, 3]
            angles = [0, 45, 90, 135]
            
            glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
            
            # Extraer propiedades de textura
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            
            return {
                'texture_contrast': contrast,
                'texture_dissimilarity': dissimilarity,
                'texture_homogeneity': homogeneity,
                'texture_energy': energy,
                'texture_correlation': correlation
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando características de textura: {e}")
            return {}
    
    def analyze_single_image(self, image_path: Path) -> Dict[str, any]:
        """
        Analiza una imagen completa y retorna todas las métricas.
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Diccionario con todas las métricas calculadas
        """
        # Cargar imagen
        image = self.load_image(image_path)
        if image is None:
            return {}
        
        # Calcular todas las métricas
        quality_metrics = self.get_image_quality_metrics(image)
        background_metrics = self.get_background_analysis(image)
        texture_metrics = self.get_texture_features(image)
        
        # Combinar todas las métricas
        all_metrics = {
            'image_path': str(image_path),
            'image_name': image_path.name,
            **quality_metrics,
            **background_metrics,
            **texture_metrics
        }
        
        return all_metrics
    
    def plot_image_with_metrics(self, image_path: Path, metrics: Dict[str, any], 
                               save_path: Optional[Path] = None) -> None:
        """
        Visualiza una imagen con sus métricas principales.
        
        Args:
            image_path: Ruta de la imagen
            metrics: Métricas calculadas
            save_path: Ruta opcional para guardar la visualización
        """
        try:
            image = self.load_image(image_path)
            if image is None:
                return
            
            # Crear figura
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Mostrar imagen
            ax1.imshow(image)
            ax1.set_title(f'Imagen: {image_path.name}')
            ax1.axis('off')
            
            # Mostrar métricas principales
            metrics_text = f"""
            Dimensiones: {metrics.get('width', 'N/A')}x{metrics.get('height', 'N/A')}
            Brillo: {metrics.get('brightness', 'N/A'):.1f}
            Contraste: {metrics.get('contrast', 'N/A'):.1f}
            Nitidez: {metrics.get('sharpness', 'N/A'):.1f}
            Entropía: {metrics.get('entropy', 'N/A'):.2f}
            Proporción fondo: {metrics.get('background_ratio', 'N/A'):.2f}
            """
            
            ax2.text(0.1, 0.5, metrics_text, transform=ax2.transAxes, 
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax2.set_title('Métricas Principales')
            ax2.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Visualización guardada en: {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error creando visualización: {e}") 