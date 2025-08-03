"""
Módulo de segmentación de imágenes.

Implementa diferentes metodologías de segmentación para análisis
de fondos y objetos en imágenes.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure
from skimage.color import rgb2gray, rgb2hsv
from skimage.segmentation import slic, mark_boundaries
# from skimage.future import graph  # No disponible en esta versión
import logging


class ImageSegmentation:
    """
    Clase para segmentación de imágenes usando diferentes metodologías.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Inicializa el segmentador de imágenes.
        
        Args:
            logger: Logger opcional para logging
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def method1_threshold_based(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Método 1: Segmentación basada en umbrales (Otsu + Morfología).
        
        Args:
            image: Imagen como array numpy
            
        Returns:
            Diccionario con máscaras y resultados
        """
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Aplicar filtro Gaussiano para reducir ruido
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Aplicar umbral de Otsu
            otsu_thresh, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Operaciones morfológicas para limpiar la máscara
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Encontrar el componente más grande (objeto principal)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
            
            # Crear máscara del objeto principal
            if num_labels > 1:
                # Encontrar el componente más grande (excluyendo el fondo)
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                object_mask = (labels == largest_label).astype(np.uint8) * 255
            else:
                object_mask = np.zeros_like(cleaned)
            
            # Máscara del fondo (inversa)
            background_mask = 255 - object_mask
            
            return {
                'method': 'threshold_based',
                'binary': binary,
                'cleaned': cleaned,
                'object_mask': object_mask,
                'background_mask': background_mask,
                'otsu_threshold': otsu_thresh
            }
            
        except Exception as e:
            self.logger.error(f"Error en segmentación por umbrales: {e}")
            return {}
    
    def method2_watershed(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Método 2: Segmentación Watershed.
        
        Args:
            image: Imagen como array numpy
            
        Returns:
            Diccionario con máscaras y resultados
        """
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Aplicar filtro Gaussiano
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detectar bordes
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilatar los bordes para crear marcadores
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Encontrar contornos para crear marcadores
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Crear marcadores para watershed
            markers = np.zeros_like(gray)
            
            # Marcar objetos (contornos internos)
            for i, contour in enumerate(contours):
                cv2.drawContours(markers, [contour], -1, i + 1, -1)
            
            # Marcar fondo
            markers[dilated > 0] = 255
            
            # Aplicar watershed
            markers = markers.astype(np.int32)
            cv2.watershed(image, markers)
            
            # Crear máscaras
            object_mask = np.zeros_like(gray)
            object_mask[markers > 0] = 255
            
            background_mask = 255 - object_mask
            
            return {
                'method': 'watershed',
                'edges': edges,
                'dilated': dilated,
                'markers': markers,
                'object_mask': object_mask,
                'background_mask': background_mask
            }
            
        except Exception as e:
            self.logger.error(f"Error en segmentación watershed: {e}")
            return {}
    
    def method3_slic_superpixels(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Método 3: Segmentación usando SLIC Superpixels.
        
        Args:
            image: Imagen como array numpy
            
        Returns:
            Diccionario con máscaras y resultados
        """
        try:
            # Convertir a escala de grises para análisis
            gray = rgb2gray(image)
            
            # Aplicar SLIC para crear superpixels
            segments = slic(image, n_segments=50, compactness=10, sigma=1)
            
            # Calcular características de cada superpixel
            unique_segments = np.unique(segments)
            
            # Analizar cada superpixel para determinar si es fondo u objeto
            object_segments = []
            
            for segment_id in unique_segments:
                # Crear máscara para este superpixel
                segment_mask = (segments == segment_id)
                
                # Calcular características del superpixel
                segment_pixels = image[segment_mask]
                
                if len(segment_pixels) > 0:
                    # Calcular varianza de color (objetos suelen tener más variación)
                    color_variance = np.var(segment_pixels, axis=0).mean()
                    
                    # Calcular posición (fondo suele estar en los bordes)
                    segment_coords = np.where(segment_mask)
                    center_y, center_x = np.mean(segment_coords[0]), np.mean(segment_coords[1])
                    
                    # Normalizar posición
                    height, width = image.shape[:2]
                    normalized_y = center_y / height
                    normalized_x = center_x / width
                    
                    # Distancia al centro (fondo suele estar más lejos del centro)
                    distance_to_center = np.sqrt((normalized_x - 0.5)**2 + (normalized_y - 0.5)**2)
                    
                    # Criterio simple: si tiene alta varianza y está cerca del centro, es objeto
                    if color_variance > 1000 and distance_to_center < 0.4:
                        object_segments.append(segment_id)
            
            # Crear máscaras
            object_mask = np.zeros_like(gray, dtype=np.uint8)
            for segment_id in object_segments:
                object_mask[segments == segment_id] = 255
            
            background_mask = 255 - object_mask
            
            return {
                'method': 'slic_superpixels',
                'segments': segments,
                'object_segments': object_segments,
                'object_mask': object_mask,
                'background_mask': background_mask
            }
            
        except Exception as e:
            self.logger.error(f"Error en segmentación SLIC: {e}")
            return {}
    
    def extract_background_features(self, image: np.ndarray, background_mask: np.ndarray) -> Dict[str, float]:
        """
        Extrae características del fondo usando la máscara invertida.
        
        Args:
            image: Imagen original
            background_mask: Máscara del fondo (255 = fondo, 0 = objeto)
            
        Returns:
            Diccionario con características del fondo
        """
        try:
            # Aplicar máscara para obtener solo el fondo
            background_pixels = image[background_mask > 0]
            
            if len(background_pixels) == 0:
                return {}
            
            # Convertir a diferentes espacios de color
            background_rgb = background_pixels
            background_hsv = cv2.cvtColor(background_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
            background_gray = cv2.cvtColor(background_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).reshape(-1)
            
            # Características de color del fondo
            bg_color_mean = np.mean(background_rgb, axis=0)
            bg_color_std = np.std(background_rgb, axis=0)
            
            # Características HSV del fondo
            bg_hsv_mean = np.mean(background_hsv, axis=0)
            bg_hsv_std = np.std(background_hsv, axis=0)
            
            # Características de textura del fondo
            bg_brightness = np.mean(background_gray)
            bg_contrast = np.std(background_gray)
            
            # Uniformidad del fondo (baja varianza = más uniforme)
            bg_uniformity = 1 / (1 + np.var(background_gray))
            
            # Proporción de área de fondo
            total_pixels = image.shape[0] * image.shape[1]
            bg_area_ratio = np.sum(background_mask > 0) / total_pixels
            
            return {
                'bg_color_r_mean': bg_color_mean[0],
                'bg_color_g_mean': bg_color_mean[1],
                'bg_color_b_mean': bg_color_mean[2],
                'bg_color_r_std': bg_color_std[0],
                'bg_color_g_std': bg_color_std[1],
                'bg_color_b_std': bg_color_std[2],
                'bg_hue_mean': bg_hsv_mean[0],
                'bg_saturation_mean': bg_hsv_mean[1],
                'bg_value_mean': bg_hsv_mean[2],
                'bg_hue_std': bg_hsv_std[0],
                'bg_saturation_std': bg_hsv_std[1],
                'bg_value_std': bg_hsv_std[2],
                'bg_brightness': bg_brightness,
                'bg_contrast': bg_contrast,
                'bg_uniformity': bg_uniformity,
                'bg_area_ratio': bg_area_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Error extrayendo características del fondo: {e}")
            return {}
    
    def compare_segmentation_methods(self, image: np.ndarray) -> Dict[str, Dict]:
        """
        Compara los tres métodos de segmentación en una imagen.
        
        Args:
            image: Imagen como array numpy
            
        Returns:
            Diccionario con resultados de los tres métodos
        """
        results = {}
        
        # Aplicar los tres métodos
        results['threshold'] = self.method1_threshold_based(image)
        results['watershed'] = self.method2_watershed(image)
        results['slic'] = self.method3_slic_superpixels(image)
        
        # Extraer características del fondo para cada método
        for method_name, method_result in results.items():
            if 'background_mask' in method_result:
                bg_features = self.extract_background_features(image, method_result['background_mask'])
                method_result['background_features'] = bg_features
        
        return results
    
    def visualize_segmentation_comparison(self, image: np.ndarray, results: Dict[str, Dict], 
                                        save_path: Optional[Path] = None) -> None:
        """
        Visualiza la comparación de los tres métodos de segmentación.
        
        Args:
            image: Imagen original
            results: Resultados de los métodos de segmentación
            save_path: Ruta opcional para guardar la visualización
        """
        try:
            # Crear figura con subplots
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            
            # Imagen original
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Imagen Original')
            axes[0, 0].axis('off')
            
            # Método 1: Threshold
            if 'threshold' in results and 'object_mask' in results['threshold']:
                axes[0, 1].imshow(results['threshold']['object_mask'], cmap='gray')
                axes[0, 1].set_title('Método 1: Threshold')
                axes[0, 1].axis('off')
                
                axes[1, 1].imshow(results['threshold']['background_mask'], cmap='gray')
                axes[1, 1].set_title('Fondo - Threshold')
                axes[1, 1].axis('off')
            
            # Método 2: Watershed
            if 'watershed' in results and 'object_mask' in results['watershed']:
                axes[0, 2].imshow(results['watershed']['object_mask'], cmap='gray')
                axes[0, 2].set_title('Método 2: Watershed')
                axes[0, 2].axis('off')
                
                axes[1, 2].imshow(results['watershed']['background_mask'], cmap='gray')
                axes[1, 2].set_title('Fondo - Watershed')
                axes[1, 2].axis('off')
            
            # Método 3: SLIC
            if 'slic' in results and 'object_mask' in results['slic']:
                axes[0, 3].imshow(results['slic']['object_mask'], cmap='gray')
                axes[0, 3].set_title('Método 3: SLIC Superpixels')
                axes[0, 3].axis('off')
                
                axes[1, 3].imshow(results['slic']['background_mask'], cmap='gray')
                axes[1, 3].set_title('Fondo - SLIC')
                axes[1, 3].axis('off')
            
            # Ocultar el subplot vacío
            axes[1, 0].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Comparación de segmentación guardada en: {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error creando visualización de comparación: {e}") 