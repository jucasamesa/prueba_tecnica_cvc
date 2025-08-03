# Módulos EDA - Análisis Exploratorio de Datos de Imágenes

Este directorio contiene los módulos para análisis exploratorio de datos (EDA) de imágenes del dataset de MercadoLibre. Los módulos están diseñados para ser utilizados tanto desde scripts como desde Jupyter notebooks.

## Estructura de Módulos

```
eda_modules/
├── __init__.py           # Inicialización del paquete
├── image_analyzer.py     # Análisis de calidad y métricas de imágenes
├── segmentation.py       # Métodos de segmentación (3 algoritmos)
├── eda_full.py          # Clase EDAFull para análisis completo del dataset
├── eda_atomic.py        # Clase EDAAtomic para análisis individual
└── README.md            # Este archivo
```

## Clases Principales

### 1. EDAFull - Análisis Completo del Dataset

La clase `EDAFull` está diseñada para realizar análisis exploratorio completo de todo el dataset de imágenes, incluyendo estadísticas descriptivas, análisis de balance y visualizaciones.

#### Características Principales:
- ✅ Carga automática de datasets (CSV + imágenes descargadas)
- ✅ Análisis de calidad de imágenes (métricas avanzadas)
- ✅ Análisis de balance de la variable objetivo
- ✅ Análisis de tamaños y dimensiones de imágenes
- ✅ Generación de estadísticas descriptivas completas
- ✅ Visualizaciones automáticas
- ✅ Exportación de resultados a CSV

#### Ejemplo de Uso en Jupyter Notebook:

```python
# Importar la clase
import sys
sys.path.append('../eda_modules')
from eda_full import EDAFull
from pathlib import Path

# Inicializar el analizador
data_dir = Path('../data')  # Ajustar según tu estructura
eda_full = EDAFull(data_dir)

# 1. Cargar datasets
eda_full.load_datasets()

# 2. Analizar calidad de imágenes (con muestra limitada para demo)
quality_df = eda_full.analyze_image_quality(sample_size=50)
print(f"Imágenes analizadas: {len(quality_df)}")

# 3. Analizar balance del target
balance_stats = eda_full.analyze_target_balance()
print(f"Ratio de desbalance: {balance_stats['imbalance_ratio']:.2f}")

# 4. Analizar tamaños de imagen
size_stats = eda_full.analyze_image_sizes()
print(f"Categorías de tamaño: {size_stats['size_categories']}")

# 5. Generar estadísticas descriptivas
descriptive_stats = eda_full.generate_descriptive_statistics()
print(descriptive_stats.head())

# 6. Crear visualizaciones
eda_full.create_visualizations()

# 7. Exportar todos los resultados
eda_full.export_results()

# 8. Ejecutar análisis completo de una vez
summary = eda_full.run_full_analysis(sample_size=100)
print(f"Resumen: {summary}")
```

#### Métodos Principales de EDAFull:

```python
# Análisis de calidad
quality_df = eda_full.analyze_image_quality(sample_size=None)  # None = todas las imágenes

# Análisis de balance
balance_stats = eda_full.analyze_target_balance()

# Análisis de tamaños
size_stats = eda_full.analyze_image_sizes()

# Estadísticas descriptivas
descriptive_stats = eda_full.generate_descriptive_statistics()

# Visualizaciones
eda_full.create_visualizations()

# Exportación
eda_full.export_results()

# Análisis completo
summary = eda_full.run_full_analysis(sample_size=100)
```

### 2. EDAAtomic - Análisis Individual de Imágenes

La clase `EDAAtomic` está diseñada para análisis detallado de imágenes individuales o muestras pequeñas, incluyendo visualización, segmentación y extracción de características del fondo.

#### Características Principales:
- ✅ Visualización de muestras aleatorias con etiquetas
- ✅ 3 métodos de segmentación (Threshold, Watershed, SLIC)
- ✅ Extracción de características del fondo
- ✅ Comparación de métodos de segmentación
- ✅ Análisis de características por método

#### Ejemplo de Uso en Jupyter Notebook:

```python
# Importar la clase
import sys
sys.path.append('scripts/eda_modules')
from eda_atomic import EDAAtomic
from pathlib import Path

# Inicializar el analizador
data_dir = Path('../data')  # Ajustar según tu estructura
eda_atomic = EDAAtomic(data_dir)

# 1. Cargar datasets
eda_atomic.load_datasets()

# 2. Visualizar muestra aleatoria de 3 imágenes
eda_atomic.visualize_random_sample(n_samples=3, balance_by_target=True)

# 3. Obtener muestra aleatoria para análisis
sample_df = eda_atomic.get_random_sample(n_samples=5, balance_by_target=True)
print(f"Muestra seleccionada: {len(sample_df)} imágenes")

# 4. Analizar segmentación de una imagen específica
if not sample_df.empty:
    row = sample_df.iloc[0]
    image_path = Path(row['local_path']) if 'local_path' in row else Path(row['image_path'])
    
    # Analizar segmentación
    segmentation_results = eda_atomic.analyze_segmentation_methods(image_path)
    print(f"Métodos aplicados: {list(segmentation_results.keys())}")
    
    # Visualizar comparación de segmentación
    eda_atomic.visualize_segmentation_analysis(image_path)

# 5. Extraer características del fondo de una muestra
background_features = eda_atomic.extract_background_features_sample(n_samples=3)
print(f"Características extraídas: {len(background_features)} registros")

# 6. Comparar características del fondo entre métodos
if not background_features.empty:
    eda_atomic.compare_background_features(background_features)

# 7. Ejecutar análisis atómico completo
summary = eda_atomic.run_atomic_analysis(n_samples=5)
print(f"Resumen: {summary}")
```

#### Métodos Principales de EDAAtomic:

```python
# Obtener muestra aleatoria
sample_df = eda_atomic.get_random_sample(n_samples=3, balance_by_target=True)

# Visualizar muestra
eda_atomic.visualize_random_sample(n_samples=3, balance_by_target=True)

# Analizar segmentación de imagen específica
results = eda_atomic.analyze_segmentation_methods(image_path)

# Visualizar análisis de segmentación
eda_atomic.visualize_segmentation_analysis(image_path)

# Extraer características del fondo
features_df = eda_atomic.extract_background_features_sample(n_samples=3)

# Comparar características
eda_atomic.compare_background_features(features_df)

# Análisis completo
summary = eda_atomic.run_atomic_analysis(n_samples=5)
```

## Módulos de Soporte

### ImageAnalyzer

Clase base para análisis de imágenes individuales:

```python
from image_analyzer import ImageAnalyzer

analyzer = ImageAnalyzer()

# Cargar imagen
image = analyzer.load_image(image_path)

# Obtener métricas de calidad
quality_metrics = analyzer.get_image_quality_metrics(image)

# Análisis de fondo
background_metrics = analyzer.get_background_analysis(image)

# Características de textura
texture_metrics = analyzer.get_texture_features(image)

# Análisis completo de una imagen
all_metrics = analyzer.analyze_single_image(image_path)

# Visualizar imagen con métricas
analyzer.plot_image_with_metrics(image_path, all_metrics)
```

### ImageSegmentation

Clase para métodos de segmentación:

```python
from segmentation import ImageSegmentation

segmenter = ImageSegmentation()

# Método 1: Threshold-based
threshold_results = segmenter.method1_threshold_based(image)

# Método 2: Watershed
watershed_results = segmenter.method2_watershed(image)

# Método 3: SLIC Superpixels
slic_results = segmenter.method3_slic_superpixels(image)

# Comparar todos los métodos
all_results = segmenter.compare_segmentation_methods(image)

# Extraer características del fondo
bg_features = segmenter.extract_background_features(image, background_mask)

# Visualizar comparación
segmenter.visualize_segmentation_comparison(image, all_results)
```

## Ejemplo Completo en Jupyter Notebook

```python
# Configuración inicial
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Añadir módulos al path
sys.path.append('scripts/eda_modules')

# Importar clases
from eda_full import EDAFull
from eda_atomic import EDAAtomic

# Configurar directorio de datos
data_dir = Path('../data')
eda_full = EDAFull(data_dir)
eda_atomic = EDAAtomic(data_dir)

# ===== ANÁLISIS COMPLETO =====
print("=== ANÁLISIS COMPLETO DEL DATASET ===")

# Cargar y analizar
eda_full.load_datasets()
quality_df = eda_full.analyze_image_quality(sample_size=50)
balance_stats = eda_full.analyze_target_balance()

# Mostrar resultados
print(f"Imágenes analizadas: {len(quality_df)}")
print(f"Balance del target: {balance_stats['imbalance_ratio']:.2f}")

# Crear visualizaciones
eda_full.create_visualizations()

# ===== ANÁLISIS ATÓMICO =====
print("\n=== ANÁLISIS ATÓMICO ===")

# Cargar datasets
eda_atomic.load_datasets()

# Visualizar muestra aleatoria
eda_atomic.visualize_random_sample(n_samples=3, balance_by_target=True)

# Extraer características del fondo
bg_features = eda_atomic.extract_background_features_sample(n_samples=3)

# Comparar métodos de segmentación
if not bg_features.empty:
    eda_atomic.compare_background_features(bg_features)

# Mostrar resumen
print(f"Características extraídas: {len(bg_features)}")
print(f"Columnas disponibles: {list(bg_features.columns)}")

# ===== ANÁLISIS DE SEGMENTACIÓN ESPECÍFICO =====
print("\n=== ANÁLISIS DE SEGMENTACIÓN ===")

# Obtener una imagen para análisis detallado
sample_df = eda_atomic.get_random_sample(n_samples=1)
if not sample_df.empty:
    row = sample_df.iloc[0]
    image_path = Path(row['local_path']) if 'local_path' in row else Path(row['image_path'])
    
    # Analizar segmentación
    eda_atomic.visualize_segmentation_analysis(image_path)
```

## Métricas Generadas

### Métricas de Calidad de Imagen:
- **Básicas:** width, height, aspect_ratio, total_pixels
- **Luminosidad:** brightness, contrast, saturation, value
- **Calidad:** sharpness, noise_level, entropy
- **Textura:** texture_contrast, texture_homogeneity, texture_energy, etc.

### Características del Fondo:
- **Color:** bg_color_r_mean, bg_color_g_mean, bg_color_b_mean
- **HSV:** bg_hue_mean, bg_saturation_mean, bg_value_mean
- **Textura:** bg_brightness, bg_contrast, bg_uniformity
- **Área:** bg_area_ratio

### Métodos de Segmentación:
1. **Threshold-based:** Umbral de Otsu + operaciones morfológicas
2. **Watershed:** Detección de bordes + algoritmo watershed
3. **SLIC Superpixels:** Segmentación por superpíxeles

## Archivos de Salida

### EDAFull:
- `data/eda_results/image_quality_analysis.csv`
- `data/eda_results/descriptive_statistics.csv`
- `data/eda_results/target_balance_analysis.csv`
- `data/eda_results/image_size_analysis.csv`
- `data/eda_results/eda_overview.png`
- `data/eda_results/eda_by_categories.png`

### EDAAtomic:
- `data/eda_atomic_results/background_features_analysis.csv`

## Configuración y Dependencias

Asegúrate de tener instaladas las dependencias:

```bash
# Con UV
uv sync

# O con pip
pip install opencv-python matplotlib seaborn scikit-image scipy
```

## Troubleshooting

### Error de importación:
```python
# Asegúrate de añadir el path correcto
import sys
sys.path.append('ruta/a/scripts/eda_modules')
```

### Error de memoria:
```python
# Usa sample_size más pequeño
quality_df = eda_full.analyze_image_quality(sample_size=10)
```

### Error de visualización:
```python
# Configura matplotlib para notebooks
%matplotlib inline
plt.style.use('seaborn-v0_8')
```

## Notas Importantes

1. **Rendimiento:** El análisis completo puede tomar tiempo con datasets grandes
2. **Memoria:** Las imágenes se cargan en memoria, considera usar muestras pequeñas
3. **Visualizaciones:** Se abren automáticamente, usa `%matplotlib inline` en notebooks
4. **Logs:** Los logs se guardan en archivos separados para seguimiento
5. **Exportación:** Todos los resultados se guardan automáticamente en CSV 

## Estructura Recomendada para Notebooks

notebooks/
├── 01_eda_setup.ipynb          # Configuración inicial
├── 02_eda_full_analysis.ipynb  # Análisis completo
├── 03_eda_atomic_analysis.ipynb # Análisis individual
├── 04_segmentation_analysis.ipynb # Segmentación
└── 05_custom_analysis.ipynb    # Análisis personalizados