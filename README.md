# Aplicación de Clasificación de Imágenes - Predicción de Calidad de Fondo

Este proyecto es una aplicación integral de procesamiento de imágenes y aprendizaje automático enfocada en la **predicción de calidad de fondo** para imágenes de productos. El sistema descarga automáticamente imágenes, las procesa para eliminar objetos en primer plano, analiza la calidad del fondo y entrena múltiples modelos de aprendizaje automático para predecir si un fondo cumple con los estándares de calidad.

## 🎯 Resumen del Proyecto

El objetivo es predecir si el fondo de una imagen de producto es de calidad aceptable (fondos crema/blancos que "cumple" vs. otros fondos que "no cumple"). El sistema procesa imágenes a través de múltiples etapas: descarga → preprocesamiento → análisis → modelado → evaluación.

## 📁 Estructura del Proyecto

```
image-classification-app/
├── data/
│   ├── images/                                    # Imágenes originales descargadas
│   ├── train/                                     # Imágenes de entrenamiento
│   │   ├── processed/                            # Imágenes de entrenamiento procesadas (redimensionadas + rembg)
│   │   └── background_masks_data_with_labels.csv # Datos de entrenamiento con etiquetas
│   ├── val/                                      # Imágenes de validación
│   │   ├── processed/                            # Imágenes de validación procesadas
│   │   └── background_masks_data_with_labels.csv # Datos de validación con etiquetas
│   ├── train_processed/                          # Datos de entrenamiento procesados
│   │   ├── background_masks_arrays_filtered.npz  # Arrays de máscaras filtradas
│   │   ├── mask_arrays_mapping_filtered.csv      # Mapeo para arrays filtrados
│   │   └── background_masks_data_with_labels.csv # Datos de entrenamiento con métricas de calidad
│   └── val_processed/                            # Datos de validación procesados
│       ├── background_masks_arrays_filtered.npz  # Arrays de máscaras filtradas
│       ├── mask_arrays_mapping_filtered.csv      # Mapeo para arrays filtrados
│       └── background_masks_data_with_labels.csv # Datos de validación con métricas de calidad
├── models/                                       # Modelos entrenados
│   ├── background_logistic_regression_classifier_cv.pkl
│   ├── background_svc_classifier_cv.pkl
│   ├── background_random_forest_classifier_cv.pkl
│   └── background_cnn_classifier_cv.pkl
├── logs/                                         # Registros de entrenamiento y evaluación
│   ├── logistic_regression_training_*.log
│   ├── svc_training_*.log
│   ├── random_forest_training_*.log
│   ├── cnn_training_*.log
│   └── model_evaluation_*.log
├── scripts/                                      # Scripts de procesamiento y análisis
│   ├── image_downloader.py                      # Utilidades de descarga de imágenes
│   ├── image_bg_extraction.py                   # Eliminación y procesamiento de fondos
│   ├── preprocess_labels.py                     # Filtrado y preprocesamiento de etiquetas
│   ├── apply_image_analyzer.py                  # Aplicación de métricas de calidad a datos
│   ├── simple_svc_classifier.py                 # Clasificador SVC rápido
│   ├── svc_classifier.py                        # Clasificador SVC completo
│   ├── logistic_regression_classifier.py        # Clasificador de regresión logística
│   ├── random_forest_classifier.py              # Clasificador de bosque aleatorio
│   ├── cnn_classifier.py                        # Clasificador CNN
│   └── evaluate_model.py                        # Evaluador general de modelos
├── eda_modules/                                  # Módulos de análisis exploratorio de datos
│   ├── __init__.py
│   ├── eda_atomic.py                            # Análisis a nivel atómico
│   ├── eda_full.py                              # Flujos de trabajo de análisis completo
│   ├── image_analyzer.py                        # Análisis de calidad de imágenes
│   ├── segmentation.py                          # Segmentación de imágenes
│   └── example.ipynb
├── notebooks/
│   └── prueba_tecnica_cvc.ipynb                # Notebook principal del proyecto con análisis
├── utils.py                                      # Funciones de utilidad general
├── config.py                                     # Configuración de ajustes
├── requirements.txt                              # Dependencias del proyecto
├── setup.py                                      # Configuración de configuración del paquete
├── MODEL_EVALUATOR_README.md                    # Documentación del evaluador de modelos
├── CNN_README.md                                # Documentación del clasificador CNN
├── SVC_README.md                                # Documentación del clasificador SVC
└── README.md                                    # Este archivo
```

**Nota**: Todos los scripts de procesamiento y análisis se encuentran en el directorio `scripts/`. Al ejecutar scripts, asegúrate de usar la ruta correcta: `python scripts/script_name.py`.

## 🔄 Flujo del Proceso

### 1. **Descarga de Imágenes** 📥
```bash
python scripts/image_downloader.py
```
- Descarga imágenes de entrenamiento y validación de fuentes externas
- Organiza imágenes en directorios `data/train/` y `data/val/`
- Maneja URLs de imágenes y gestión de archivos locales

### 2. **Procesamiento de Imágenes** 🖼️
```bash
python scripts/image_bg_extraction.py
```
- **Redimensiona** imágenes a dimensiones estándar (512x512)
- **Elimina objetos en primer plano** usando `rembg` (eliminación de fondo con IA)
- **Guarda imágenes solo de fondo** con primeros planos transparentes (RGBA)
- **Genera arrays de máscaras** y los almacena en archivos NPZ
- **Crea archivos CSV** con estadísticas de fondo y metadatos
- **Analiza colores de fondo** usando espacio de color CIELAB (1976) para detección de crema

**Características Clave:**
- Preserva colores de fondo originales (sin relleno blanco/negro)
- Usa primeros planos transparentes para evitar interferencia de color
- Detección de color crema basada en CIELAB con precisión perceptual
- Seguimiento completo de metadatos

### 3. **Filtrado de Datos** 🔍
```bash
python scripts/preprocess_labels.py
```
- **Filtra datos inciertos** (elimina filas con `correct_background? = ?`)
- **Crea conjuntos de datos filtrados** para entrenamiento y validación
- **Genera archivos NPZ filtrados** con solo etiquetas ciertas
- **Actualiza archivos de mapeo** para datos filtrados

### 4. **Análisis de Calidad** 📊
```bash
python scripts/apply_image_analyzer.py
```
- **Aplica ImageAnalyzer** a imágenes procesadas
- **Calcula métricas de calidad**: brillo, contraste, nitidez, etc.
- **Actualiza archivos CSV** con características de calidad adicionales
- **Enriquece datos de entrenamiento** para mejor rendimiento del modelo

### 5. **Entrenamiento de Modelos** 🤖

#### 5.1 Regresión Logística
```bash
python scripts/logistic_regression_classifier.py
```
- Modelo lineal rápido e interpretable
- Validación cruzada con k-fold estratificado
- Ajuste automático de hiperparámetros
- Registro completo

#### 5.2 Clasificador de Vectores de Soporte (SVC)
```bash
# Modo rápido para pruebas rápidas
python scripts/simple_svc_classifier.py

# Modo completo con ajuste integral
python scripts/svc_classifier.py
```
- Clasificación no lineal con métodos de kernel
- Búsqueda en cuadrícula para optimización de hiperparámetros
- Procesamiento eficiente en memoria
- Modo rápido para validación rápida

#### 5.3 Bosque Aleatorio
```bash
python scripts/random_forest_classifier.py
```
- Método de conjunto con múltiples árboles de decisión
- Análisis de importancia de características
- Robusto al sobreajuste
- Bueno para conjuntos de datos desequilibrados

#### 5.4 Red Neuronal Convolucional (CNN)
```bash
python scripts/cnn_classifier.py
```
- Enfoque de aprendizaje profundo usando TensorFlow/Keras
- Aprendizaje basado en imágenes (no solo arrays aplanados)
- Aumentación de datos para mejor generalización
- Parada temprana y reducción de tasa de aprendizaje
- Soporte para aceleración GPU

### 6. **Evaluación de Modelos** 🎯
```bash
# Listar modelos disponibles
python scripts/evaluate_model.py --list

# Evaluar modelo específico
python scripts/evaluate_model.py -m models/background_logistic_regression_classifier_cv.pkl

# Evaluar sin registro
python scripts/evaluate_model.py -m models/background_svc_classifier_cv.pkl --no-log
```
- **Evaluador universal** para todos los modelos entrenados
- **Métricas integrales**: Precisión, F1-Score, Precisión, Recuperación
- **Análisis por clase** para predicción de calidad de fondo
- **Matriz de confusión** y reportes de clasificación detallados
- **Registro automático** a archivos con marca de tiempo

### 7. **Análisis del Proyecto** 📈
```bash
# Abrir notebook Jupyter
jupyter notebook notebooks/prueba_tecnica_cvc.ipynb
```
- **Documentación completa del proyecto** y análisis
- **Respuestas a tareas técnicas** y requisitos
- **Exploración de datos** y visualización
- **Comparación de modelos** y análisis de resultados
- **Análisis del espacio de color CIELAB** para detección de crema

## 🚀 Inicio Rápido

### 1. Configurar Entorno
```bash
# Instalar dependencias
pip install -r requirements.txt

# Activar entorno virtual (si se usa)
source meli/bin/activate  # Linux/Mac
# o
meli\Scripts\activate.bat  # Windows
```

### 2. Descargar Imágenes
```bash
python scripts/image_downloader.py
```

### 3. Procesar Imágenes
```bash
python scripts/image_bg_extraction.py
```

### 4. Filtrar Datos
```bash
python scripts/preprocess_labels.py
```

### 5. Aplicar Análisis de Calidad
```bash
python scripts/apply_image_analyzer.py
```

### 6. Entrenar Modelos
```bash
# Comenzar con regresión logística (más rápido)
python scripts/logistic_regression_classifier.py

# Luego probar otros modelos
python scripts/random_forest_classifier.py
python scripts/simple_svc_classifier.py
python scripts/cnn_classifier.py
```

### 7. Evaluar Modelos
```bash
# Listar modelos disponibles
python scripts/evaluate_model.py --list

# Evaluar un modelo específico
python scripts/evaluate_model.py -m models/background_logistic_regression_classifier_cv.pkl
```

### 8. Analizar Resultados
```bash
jupyter notebook notebooks/prueba_tecnica_cvc.ipynb
```

## 📊 Características Clave

### Procesamiento de Imágenes
- **Eliminación de fondo con IA** usando `rembg`
- **Primeros planos transparentes** para preservar colores de fondo
- **Espacio de color CIELAB** para detección precisa de crema
- **Dimensiones de imagen estandarizadas** (512x512)
- **Seguimiento completo de metadatos**

### Aprendizaje Automático
- **Múltiples tipos de modelos**: Regresión Logística, SVC, Bosque Aleatorio, CNN
- **Validación cruzada** para evaluación robusta
- **Ajuste de hiperparámetros** con búsqueda en cuadrícula
- **Ingeniería de características** con métricas de calidad
- **Procesamiento eficiente en memoria**

### Evaluación y Análisis
- **Evaluador universal de modelos** para todos los modelos entrenados
- **Métricas integrales** y análisis por clase
- **Registro automático** a archivos con marca de tiempo
- **Reportes de clasificación detallados**
- **Análisis de confianza**

### Documentación
- **Documentación completa del proceso** en notebooks
- **READMEs específicos de modelos** para cada clasificador
- **Ejemplos de uso** y guías de solución de problemas
- **Comparaciones de rendimiento** e insights

## 🔧 Configuración

### Variables de Entorno
- `CUDA_VISIBLE_DEVICES`: Para aceleración GPU (CNN)
- `OMP_NUM_THREADS`: Para procesamiento paralelo

### Parámetros del Modelo
- **Modo rápido**: Búsqueda reducida de hiperparámetros para pruebas rápidas
- **Modo completo**: Optimización integral de hiperparámetros
- **Optimización de memoria**: Tipos de datos `float32`, `n_jobs=1`

## 📈 Rendimiento

### Comparación de Modelos
| Modelo | Precisión | F1-Score | Tiempo de Entrenamiento | Uso de Memoria |
|--------|-----------|----------|-------------------------|----------------|
| Regresión Logística | ~67% | ~0.67 | Rápido | Bajo |
| Bosque Aleatorio | ~88% | ~0.89 | Medio | Medio |
| SVC | ~94% | ~94% | MuyLento | Alto |
| CNN | ~91% | ~0.90 | Medio | Alto |

### Procesamiento de Datos
- **Procesamiento de imágenes**: ~2-3 segundos por imagen
- **Eliminación de fondo**: ~1-2 segundos por imagen
- **Análisis de calidad**: ~0.5 segundos por imagen
- **Entrenamiento de modelos**: 5-30 minutos dependiendo del tipo de modelo

## 🚨 Solución de Problemas

### Problemas Comunes
1. **Errores de memoria**: Usar `simple_svc_classifier.py` o reducir tamaño de datos
2. **Errores de importación**: Verificar activación del entorno virtual
3. **Modelo no encontrado**: Verificar ruta del modelo con `python scripts/evaluate_model.py --list`
4. **Datos no encontrados**: Verificar si se completaron los pasos de preprocesamiento
5. **Errores de ruta de script**: Todos los scripts ahora están en el directorio `scripts/` y usan rutas de raíz del proyecto

### Optimización de Rendimiento
- Usar `float32` en lugar de `float64` para conjuntos de datos grandes
- Establecer `n_jobs=1` para entornos con restricciones de memoria
- Usar modo rápido para pruebas rápidas
- Habilitar aceleración GPU para entrenamiento CNN

### Actualizaciones de Ruta de Scripts
Todos los scripts han sido movidos al directorio `scripts/` y actualizados para usar rutas de raíz del proyecto. Esto significa:
- Los scripts se pueden ejecutar desde cualquier directorio usando `python scripts/script_name.py`
- Todas las rutas de datos se resuelven automáticamente relativas a la raíz del proyecto
- No es necesario cambiar el directorio de trabajo antes de ejecutar scripts

## 📝 Registros y Documentación

Todos los procesos generan registros integrales:
- **Registros de entrenamiento**: `logs/*_training_*.log`
- **Registros de evaluación**: `logs/model_evaluation_*.log`
- **Registros de procesamiento**: Salida de consola con seguimiento de progreso

## 🎯 Objetivos del Proyecto

1. **Predicción automatizada de calidad de fondo** para imágenes de productos
2. **Pipeline robusto de aprendizaje automático** con múltiples tipos de modelos
3. **Marco de evaluación integral** para comparación de modelos
4. **Procesamiento de imágenes escalable** con eliminación de fondo con IA
5. **Documentación completa** y análisis para requisitos técnicos

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🤝 Contribución

1. Hacer fork del repositorio
2. Crear una rama de características
3. Hacer tus cambios
4. Agregar pruebas si aplica
5. Enviar una solicitud de pull

## 📞 Soporte

Para preguntas o problemas:
1. Revisar la sección de solución de problemas
2. Revisar los registros en el directorio `logs/`
3. Consultar los READMEs específicos de modelos
4. Abrir un issue en GitHub
