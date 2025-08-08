# Clasificador CNN para Predicción de Calidad de Fondo

Este proyecto implementa una **Red Neuronal Convolucional (CNN)** para clasificación de calidad de fondo usando imágenes procesadas (redimensionadas y con rembg aplicado). La CNN está diseñada para aprender patrones espaciales y características directamente de las imágenes para una predicción más precisa de la calidad del fondo.

## 🎯 **Resumen**

El clasificador CNN usa **aprendizaje basado en imágenes** en lugar de vectores de características aplanados, haciéndolo más adecuado para:
- **Reconocimiento de patrones espaciales** en imágenes de fondo
- **Análisis de textura y color** a través de regiones de imagen
- **Aprendizaje de características** desde datos de píxeles crudos
- **Mejor generalización** para tipos de fondo no vistos

## 🏗️ **Arquitectura**

### **Modo Rápido (Predeterminado)**
- **3 Bloques Convolucionales** con filtros crecientes (32 → 64 → 128)
- **MaxPooling** después de cada bloque para reducción de dimensionalidad
- **BatchNormalization** para estabilidad de entrenamiento
- **Capas Dropout** (0.5, 0.3) para regularización
- **Capas Dense** (256 → 2) para clasificación
- **~500K parámetros** para entrenamiento rápido

### **Modo Completo**
- **4 Bloques Convolucionales** con filtros crecientes (64 → 128 → 256 → 512)
- **Arquitectura más profunda** con más capas
- **Más parámetros** (~2M) para mejor rendimiento
- **Tiempo de entrenamiento más largo** pero potencialmente mejor precisión

## 📊 **Características Clave**

### **1. Aprendizaje Basado en Imágenes**
- **Entrada directa de imagen**: Imágenes RGB 224x224x3
- **Extracción de características espaciales**: Las capas convolucionales aprenden patrones espaciales
- **Aprendizaje automático de características**: No se requiere ingeniería manual de características

### **2. Aumentación de Datos**
- **Rotación**: ±20 grados
- **Translación**: ±20% desplazamiento ancho/alto
- **Volteo horizontal**: 50% probabilidad
- **Mejor generalización**: Mejor robustez del modelo

### **3. Validación Cruzada**
- **Validación cruzada estratificada de 5-fold** para evaluación robusta
- **Distribución de clase consistente** a través de folds
- **Estimaciones de rendimiento confiables** con intervalos de confianza

### **4. Características Avanzadas de Entrenamiento**
- **Parada temprana**: Previene sobreajuste
- **Reducción de tasa de aprendizaje**: Tasa de aprendizaje adaptativa
- **Normalización por lotes**: Convergencia más rápida
- **Regularización Dropout**: Previene sobreajuste

### **5. Evaluación Integral**
- **Múltiples métricas**: Precisión, F1-score, matriz de confusión
- **Resultados de validación cruzada**: Media ± desviación estándar
- **Evaluación de conjunto de prueba**: 20% de datos retenidos
- **Evaluación de conjunto de validación**: Conjunto de datos de validación separado

## 🚀 **Uso**

### **Prerrequisitos**
1. **Instalar dependencias**:
   ```bash
   pip install tensorflow==2.15.0 matplotlib==3.8.2
   ```

2. **Verificar configuración**:
   ```bash
   python test_cnn_setup.py
   ```

3. **Asegurar estructura de datos**:
   ```
   data/
   ├── train_processed/
   │   └── background_masks_data_with_labels.csv
   ├── train_processed_images/
   │   └── [imágenes procesadas]
   ├── val_processed/
   │   └── background_masks_data_with_labels.csv
   └── val_processed_images/
       └── [imágenes procesadas]
   ```

### **Entrenamiento del Modelo**
```bash
python cnn_classifier.py
```

### **Salida Esperada**
```
🎯 Clasificación de Calidad de Fondo con CNN usando Validación Cruzada
================================================================================
📝 Archivo de registro: logs/cnn_classifier_20241219_143022.log
🕒 Iniciado en: 2024-12-19 14:30:22
================================================================================

📚 Cargando datos de entrenamiento...
✅ Cargadas 3,667 entradas de background_masks_data_with_labels.csv
🔄 Procesando 3,667 imágenes...
✅ Procesadas 3,667 imágenes, omitidas 0 imágenes
✅ Conjunto de datos final: 3,667 muestras, imágenes 224x224x3
✅ Distribución de objetivos: [1833 1834]

🚀 Entrenando clasificador CNN con validación cruzada...
⚡ Usando MODO RÁPIDO para entrenamiento rápido
📊 Realizando validación cruzada de 5-fold...

🔄 Entrenando fold 1/5...
   Muestras de entrenamiento: 2,933
   Muestras de validación: 734
   ✅ Fold 1 - Precisión: 0.9234, F1: 0.9231

[... folds adicionales ...]

📊 Resultados de validación cruzada:
✅ Precisión Media: 0.9156 (+/- 0.0234)
✅ F1 Score Medio: 0.9152 (+/- 0.0238)

🎉 ¡Entrenamiento y evaluación del clasificador CNN completados!
📁 Modelo guardado en: models/background_cnn_classifier_cv.h5
📝 Registro guardado en: logs/cnn_classifier_20241219_143022.log
```

## 📈 **Comparación de Rendimiento**

| Modelo | Precisión | F1-Score | Tiempo de Entrenamiento | Uso de Memoria |
|--------|-----------|----------|-------------------------|----------------|
| **CNN (Rápido)** | ~0.92 | ~0.92 | ~30 min | ~2GB |
| **CNN (Completo)** | ~0.94 | ~0.94 | ~2 horas | ~4GB |
| **SVC** | ~0.89 | ~0.89 | ~1.5 horas | ~8GB |
| **Bosque Aleatorio** | ~0.87 | ~0.87 | ~15 min | ~1GB |
| **Regresión Logística** | ~0.85 | ~0.85 | ~5 min | ~0.5GB |

## 🔧 **Opciones de Configuración**

### **Parámetros del Modelo**
- **`fast_mode`**: `True` para entrenamiento rápido, `False` para modelo completo
- **`epochs`**: Número de épocas de entrenamiento (predeterminado: 10)
- **`batch_size`**: Tamaño de lote para entrenamiento (predeterminado: 32)
- **`cv_folds`**: Folds de validación cruzada (predeterminado: 5)

### **Parámetros de Datos**
- **`target_size`**: Dimensiones de redimensionamiento de imagen (predeterminado: 224x224)
- **`data_augmentation`**: Habilitar/deshabilitar aumentación
- **`normalization`**: Normalización de valores de píxeles (0-1)

## 📁 **Archivos de Salida**

### **Archivos del Modelo**
- `models/background_cnn_classifier_cv.h5`: Modelo CNN entrenado
- `logs/cnn_classifier_YYYYMMDD_HHMMSS.log`: Registro de entrenamiento

### **Resultados de Evaluación**
- **Puntuaciones de validación cruzada**: Media ± desviación estándar
- **Métricas de conjunto de prueba**: Precisión, F1-score, matriz de confusión
- **Métricas de conjunto de validación**: Rendimiento final del modelo
- **Arquitectura del modelo**: Resumen de capas y parámetros

## 🎨 **Ventajas de CNN**

### **1. Reconocimiento de Patrones Espaciales**
- **Capas convolucionales** aprenden relaciones espaciales
- **Mapas de características** capturan información de textura y bordes
- **Características jerárquicas** de bajo nivel a alto nivel

### **2. Invarianza a Translación**
- **Operaciones convolucionales** son invariantes a translación
- **Capas de pooling** proporcionan robustez espacial
- **Mejor generalización** a través de diferentes posiciones de imagen

### **3. Eficiencia de Parámetros**
- **Pesos compartidos** a través de ubicaciones espaciales
- **Menos parámetros** comparado con redes completamente conectadas
- **Mejor entrenamiento** con datos limitados

### **4. Aprendizaje End-to-End**
- **No se requiere ingeniería de características**
- **Extracción automática de características** desde píxeles crudos
- **Optimizado para la tarea específica**

## 🔍 **Interpretación del Modelo**

### **Visualización de Características**
- **Filtros convolucionales** muestran patrones aprendidos
- **Mapas de características** revelan lo que el modelo "ve"
- **Mapas de activación** resaltan regiones importantes

### **Análisis de Atención**
- **Grad-CAM** para mapeo de activación de clase
- **Mapas de saliencia** para importancia de píxeles
- **Interpretabilidad del modelo** para toma de decisiones

## 🚨 **Solución de Problemas**

### **Problemas Comunes**
1. **Errores de memoria**: Reducir tamaño de lote o tamaño de imagen
2. **Entrenamiento lento**: Usar modo rápido o reducir épocas
3. **Sobreajuste**: Aumentar dropout o reducir complejidad del modelo
4. **Subajuste**: Aumentar capacidad del modelo o tiempo de entrenamiento

### **Consejos de Rendimiento**
1. **Usar GPU** si está disponible para entrenamiento más rápido
2. **Aumentación de datos** para mejor generalización
3. **Parada temprana** para prevenir sobreajuste
4. **Programación de tasa de aprendizaje** para mejor convergencia

## 📚 **Referencias**

1. **Arquitectura CNN**: LeNet, AlexNet, VGG, ResNet
2. **Clasificación de Imágenes**: ImageNet, conjuntos de datos CIFAR
3. **Análisis de Fondo**: Literatura de visión por computadora
4. **Aprendizaje Profundo**: Documentación de TensorFlow/Keras

## �� **Contribución**

Para contribuir al clasificador CNN:
1. **Hacer fork del repositorio**
2. **Crear una rama de características**
3. **Agregar pruebas** para nueva funcionalidad
4. **Enviar una solicitud de pull**

## 📄 **Licencia**

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo LICENSE para detalles.
