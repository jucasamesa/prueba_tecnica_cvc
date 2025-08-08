# General Model Evaluator

## 🎯 Overview

El `evaluate_model.py` es un evaluador general que puede trabajar con **cualquier modelo entrenado** para evaluar su rendimiento en los datos de validación.

## 🚀 Características

- **Universal**: Funciona con cualquier modelo entrenado (Logistic Regression, SVC, Random Forest, CNN, etc.)
- **Logging automático**: Guarda todos los logs de evaluación en archivos timestamped
- **Métricas completas**: Accuracy, F1-Score, Precision, Recall, Confusion Matrix
- **Análisis por clase**: Métricas específicas para cada clase
- **Fácil de usar**: Interfaz de línea de comandos simple

## 📁 Estructura de Archivos

```
models/
├── background_logistic_regression_classifier_cv.pkl
├── background_svc_classifier_cv.pkl
├── background_random_forest_classifier_cv.pkl
└── ...

logs/
├── model_evaluation_20241201_143022.log
├── model_evaluation_20241201_150145.log
└── ...
```

## 🎮 Uso

### 1. Listar modelos disponibles

```bash
python evaluate_model.py --list
# o
python evaluate_model.py -l
```

**Salida:**
```
📁 Available trained models:
   1. background_logistic_regression_classifier_cv.pkl
   2. background_svc_classifier_cv.pkl
   3. background_random_forest_classifier_cv.pkl
```

### 2. Evaluar un modelo específico

```bash
# Especificar el modelo
python evaluate_model.py --model models/background_logistic_regression_classifier_cv.pkl

# O usar la ruta corta
python evaluate_model.py -m models/background_logistic_regression_classifier_cv.pkl
```

### 3. Evaluar el primer modelo disponible (automático)

```bash
python evaluate_model.py
```

### 4. Evaluar sin guardar logs

```bash
python evaluate_model.py --model models/background_logistic_regression_classifier_cv.pkl --no-log
```

## 📊 Métricas de Evaluación

El evaluador calcula y muestra:

### 🎯 Métricas Generales
- **Accuracy**: Precisión general del modelo
- **F1-Score**: Media armónica de precisión y recall
- **Precision**: Proporción de predicciones positivas correctas
- **Recall**: Proporción de casos positivos reales identificados

### 📈 Métricas por Clase
- **Background No Cumple (0)**: Métricas específicas para fondos que no cumplen
- **Background Cumple (1)**: Métricas específicas para fondos que cumplen

### 📊 Matriz de Confusión
```
   Predicted:
             0    1
   Actual 0: 245   12
         1:  23  156
```

### 📋 Reporte Detallado
- Clasificación detallada por clase
- Métricas específicas por clase
- Análisis de confianza del modelo

## 🔍 Ejemplos de Uso

### Ejemplo 1: Evaluar modelo de Regresión Logística

```bash
python evaluate_model.py -m models/background_logistic_regression_classifier_cv.pkl
```

**Salida esperada:**
```
🎯 General Model Evaluator for Validation Data
==============================================================
📝 Log file: logs/model_evaluation_20241201_143022.log
🕒 Started at: 2024-12-01 14:30:22
==============================================================

🔍 Loading validation data...
✅ Loaded 1380 validation mask arrays
✅ Loaded validation mapping with 1380 entries
✅ Loaded validation labels with 1380 entries
🔄 Processing validation mask arrays...
✅ Processed 1380 validation samples
📊 Validation data shape: (1380, 262144)
📊 Validation labels shape: (1380,)

🔍 Loading trained model from models/background_logistic_regression_classifier_cv.pkl...
✅ Model loaded successfully
📋 Model type: Pipeline(LogisticRegression)
🚀 Making predictions on validation data...
✅ Predictions completed successfully

==============================================================
📊 MODEL EVALUATION ON VALIDATION DATA
==============================================================

🤖 MODEL INFORMATION:
   Model type: Pipeline(LogisticRegression)
   Model path: models/background_logistic_regression_classifier_cv.pkl

🎯 OVERALL PERFORMANCE:
   Accuracy:  0.9232 (92.32%)
   F1 Score:  0.9231
   Precision: 0.9234
   Recall:    0.9232

📈 PER-CLASS PERFORMANCE:
   Background No Cumple (0):
     - F1 Score:  0.9234
     - Precision: 0.9232
     - Recall:    0.9236
   Background Cumple (1):
     - F1 Score:  0.9228
     - Precision: 0.9236
     - Recall:    0.9228

📊 CONFUSION MATRIX:
   Predicted:
             0    1
   Actual 0: 245   12
         1:  23  156

📋 DETAILED CLASSIFICATION REPORT:
              precision    recall  f1-score   support

Background No Cumple       0.92      0.92      0.92       257
   Background Cumple       0.92      0.92      0.92       179

    accuracy                           0.92       436
   macro avg       0.92      0.92      0.92       436
weighted avg       0.92      0.92      0.92       436

💡 ADDITIONAL INSIGHTS:
   Total validation samples: 436
   Background No Cumple (0): 257 (58.9%)
   Background Cumple (1): 179 (41.1%)
   Average prediction confidence: 0.9234

==============================================================

✅ Evaluation completed successfully!
📁 Model evaluated: models/background_logistic_regression_classifier_cv.pkl
📊 Validation samples: 436
📝 Log saved to: logs/model_evaluation_20241201_143022.log

🕒 Completed at: 2024-12-01 14:30:25
```

### Ejemplo 2: Evaluar modelo SVC

```bash
python evaluate_model.py -m models/background_svc_classifier_cv.pkl
```

### Ejemplo 3: Evaluar modelo Random Forest

```bash
python evaluate_model.py -m models/background_random_forest_classifier_cv.pkl
```

## 📝 Logs

Todos los logs se guardan automáticamente en el directorio `logs/` con el formato:
```
logs/model_evaluation_YYYYMMDD_HHMMSS.log
```

### Contenido del Log
- Información completa de la evaluación
- Métricas detalladas
- Matriz de confusión
- Reporte de clasificación
- Errores y advertencias
- Timestamps de inicio y fin

## 🔧 Opciones Avanzadas

### Argumentos Disponibles

| Argumento | Descripción | Ejemplo |
|-----------|-------------|---------|
| `--model`, `-m` | Ruta al modelo a evaluar | `-m models/my_model.pkl` |
| `--list`, `-l` | Listar modelos disponibles | `-l` |
| `--no-log` | No guardar logs en archivo | `--no-log` |

### Combinaciones Útiles

```bash
# Listar modelos y luego evaluar uno específico
python evaluate_model.py -l
python evaluate_model.py -m models/background_logistic_regression_classifier_cv.pkl

# Evaluar sin logs (solo terminal)
python evaluate_model.py -m models/background_svc_classifier_cv.pkl --no-log

# Evaluar el primer modelo disponible
python evaluate_model.py
```

## 🎯 Ventajas del Evaluador General

1. **Reutilizable**: Un solo script para todos los modelos
2. **Consistente**: Mismas métricas y formato para todos los modelos
3. **Mantenible**: Un solo lugar para actualizaciones
4. **Flexible**: Fácil de extender para nuevos tipos de modelos
5. **Documentado**: Logs completos para análisis posterior

## 🚨 Solución de Problemas

### Error: "No trained models found"
```bash
# Verificar que existen modelos en el directorio models/
ls models/
```

### Error: "Model not found"
```bash
# Verificar la ruta exacta del modelo
python evaluate_model.py -l
```

### Error: "Validation data not found"
```bash
# Verificar que existen los datos de validación procesados
ls data/val_processed/
```

## 📈 Comparación de Modelos

Para comparar múltiples modelos, ejecuta el evaluador para cada uno y revisa los logs:

```bash
# Evaluar todos los modelos disponibles
for model in models/*.pkl; do
    echo "Evaluating $model..."
    python evaluate_model.py -m "$model"
done
```

Luego compara los resultados en los logs generados.
