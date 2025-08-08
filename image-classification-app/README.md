# Image Classification App - Background Quality Prediction

This project is a comprehensive image processing and machine learning application focused on **background quality prediction** for product images. The system automatically downloads images, processes them to remove foreground objects, analyzes background quality, and trains multiple machine learning models to predict whether a background meets quality standards.

## 🎯 Project Overview

The goal is to predict if a product image background is of acceptable quality (cream/white backgrounds that "cumple" vs. other backgrounds that "no cumple"). The system processes images through multiple stages: download → preprocessing → analysis → modeling → evaluation.

## 📁 Project Structure

```
image-classification-app/
├── data/
│   ├── images/                                    # Original downloaded images
│   ├── train/                                     # Training images
│   │   ├── processed/                            # Processed training images (resized + rembg)
│   │   └── background_masks_data_with_labels.csv # Training data with labels
│   ├── val/                                      # Validation images
│   │   ├── processed/                            # Processed validation images
│   │   └── background_masks_data_with_labels.csv # Validation data with labels
│   ├── train_processed/                          # Training processed data
│   │   ├── background_masks_arrays_filtered.npz  # Filtered mask arrays
│   │   ├── mask_arrays_mapping_filtered.csv      # Mapping for filtered arrays
│   │   └── background_masks_data_with_labels.csv # Training data with quality metrics
│   └── val_processed/                            # Validation processed data
│       ├── background_masks_arrays_filtered.npz  # Filtered mask arrays
│       ├── mask_arrays_mapping_filtered.csv      # Mapping for filtered arrays
│       └── background_masks_data_with_labels.csv # Validation data with quality metrics
├── models/                                       # Trained models
│   ├── background_logistic_regression_classifier_cv.pkl
│   ├── background_svc_classifier_cv.pkl
│   ├── background_random_forest_classifier_cv.pkl
│   └── background_cnn_classifier_cv.pkl
├── logs/                                         # Training and evaluation logs
│   ├── logistic_regression_training_*.log
│   ├── svc_training_*.log
│   ├── random_forest_training_*.log
│   ├── cnn_training_*.log
│   └── model_evaluation_*.log
├── scripts/                                      # Processing and analysis scripts
│   ├── image_downloader.py                      # Image downloading utilities
│   ├── image_bg_extraction.py                   # Background removal and processing
│   ├── preprocess_labels.py                     # Filter and preprocess labels
│   ├── apply_image_analyzer.py                  # Apply quality metrics to data
│   ├── simple_svc_classifier.py                 # Fast SVC classifier
│   ├── svc_classifier.py                        # Full SVC classifier
│   ├── logistic_regression_classifier.py        # Logistic regression classifier
│   ├── random_forest_classifier.py              # Random forest classifier
│   ├── cnn_classifier.py                        # CNN classifier
│   └── evaluate_model.py                        # General model evaluator
├── eda_modules/                                  # Exploratory data analysis modules
│   ├── __init__.py
│   ├── eda_atomic.py                            # Atomic-level analysis
│   ├── eda_full.py                              # Full analysis workflows
│   ├── image_analyzer.py                        # Image quality analysis
│   ├── segmentation.py                          # Image segmentation
│   └── example.ipynb
├── notebooks/
│   └── prueba_tecnica_cvc.ipynb                # Main project notebook with analysis
├── utils.py                                      # General utility functions
├── config.py                                     # Configuration settings
├── requirements.txt                              # Project dependencies
├── setup.py                                      # Package setup configuration
├── MODEL_EVALUATOR_README.md                    # Model evaluator documentation
├── CNN_README.md                                # CNN classifier documentation
├── SVC_README.md                                # SVC classifier documentation
└── README.md                                    # This file
```

**Note**: All processing and analysis scripts are located in the `scripts/` directory. When running scripts, make sure to use the correct path: `python scripts/script_name.py`.

## 🔄 Process Flow

### 1. **Image Download** 📥
```bash
python scripts/image_downloader.py
```
- Downloads training and validation images from external sources
- Organizes images into `data/train/` and `data/val/` directories
- Handles image URLs and local file management

### 2. **Image Processing** 🖼️
```bash
python scripts/image_bg_extraction.py
```
- **Resizes** images to standard dimensions (512x512)
- **Removes foreground objects** using `rembg` (AI-powered background removal)
- **Saves background-only images** with transparent foregrounds (RGBA)
- **Generates mask arrays** and stores them in NPZ files
- **Creates CSV files** with background statistics and metadata
- **Analyzes background colors** using CIELAB (1976) color space for cream detection

**Key Features:**
- Preserves original background colors (no white/black filling)
- Uses transparent foregrounds to avoid color interference
- CIELAB-based cream color detection with perceptual accuracy
- Comprehensive metadata tracking

### 3. **Data Filtering** 🔍
```bash
python scripts/preprocess_labels.py
```
- **Filters out uncertain data** (removes rows with `correct_background? = ?`)
- **Creates filtered datasets** for training and validation
- **Generates filtered NPZ files** with only certain labels
- **Updates mapping files** for filtered data

### 4. **Quality Analysis** 📊
```bash
python scripts/apply_image_analyzer.py
```
- **Applies ImageAnalyzer** to processed images
- **Calculates quality metrics**: brightness, contrast, sharpness, etc.
- **Updates CSV files** with additional quality features
- **Enriches training data** for better model performance

### 5. **Model Training** 🤖

#### 5.1 Logistic Regression
```bash
python scripts/logistic_regression_classifier.py
```
- Fast, interpretable linear model
- Cross-validation with stratified k-fold
- Automatic hyperparameter tuning
- Comprehensive logging

#### 5.2 Support Vector Classifier (SVC)
```bash
# Fast mode for quick testing
python scripts/simple_svc_classifier.py

# Full mode with comprehensive tuning
python scripts/svc_classifier.py
```
- Non-linear classification with kernel methods
- Grid search for hyperparameter optimization
- Memory-efficient processing
- Fast mode for quick validation

#### 5.3 Random Forest
```bash
python scripts/random_forest_classifier.py
```
- Ensemble method with multiple decision trees
- Feature importance analysis
- Robust to overfitting
- Good for imbalanced datasets

#### 5.4 Convolutional Neural Network (CNN)
```bash
python scripts/cnn_classifier.py
```
- Deep learning approach using TensorFlow/Keras
- Image-based learning (not just flattened arrays)
- Data augmentation for better generalization
- Early stopping and learning rate reduction
- GPU acceleration support

### 6. **Model Evaluation** 🎯
```bash
# List available models
python scripts/evaluate_model.py --list

# Evaluate specific model
python scripts/evaluate_model.py -m models/background_logistic_regression_classifier_cv.pkl

# Evaluate without logging
python scripts/evaluate_model.py -m models/background_svc_classifier_cv.pkl --no-log
```
- **Universal evaluator** for all trained models
- **Comprehensive metrics**: Accuracy, F1-Score, Precision, Recall
- **Per-class analysis** for background quality prediction
- **Confusion matrix** and detailed classification reports
- **Automatic logging** to timestamped files

### 7. **Project Analysis** 📈
```bash
# Open Jupyter notebook
jupyter notebook notebooks/prueba_tecnica_cvc.ipynb
```
- **Complete project documentation** and analysis
- **Answers to technical tasks** and requirements
- **Data exploration** and visualization
- **Model comparison** and results analysis
- **CIELAB color space analysis** for cream detection

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Activate virtual environment (if using)
source meli/bin/activate  # Linux/Mac
# or
meli\Scripts\activate.bat  # Windows
```

### 2. Download Images
```bash
python scripts/image_downloader.py
```

### 3. Process Images
```bash
python scripts/image_bg_extraction.py
```

### 4. Filter Data
```bash
python scripts/preprocess_labels.py
```

### 5. Apply Quality Analysis
```bash
python scripts/apply_image_analyzer.py
```

### 6. Train Models
```bash
# Start with logistic regression (fastest)
python scripts/logistic_regression_classifier.py

# Then try other models
python scripts/random_forest_classifier.py
python scripts/simple_svc_classifier.py
python scripts/cnn_classifier.py
```

### 7. Evaluate Models
```bash
# List available models
python scripts/evaluate_model.py --list

# Evaluate a specific model
python scripts/evaluate_model.py -m models/background_logistic_regression_classifier_cv.pkl
```

### 8. Analyze Results
```bash
jupyter notebook notebooks/prueba_tecnica_cvc.ipynb
```

## 📊 Key Features

### Image Processing
- **AI-powered background removal** using `rembg`
- **Transparent foregrounds** to preserve background colors
- **CIELAB color space** for accurate cream detection
- **Standardized image dimensions** (512x512)
- **Comprehensive metadata tracking**

### Machine Learning
- **Multiple model types**: Logistic Regression, SVC, Random Forest, CNN
- **Cross-validation** for robust evaluation
- **Hyperparameter tuning** with grid search
- **Feature engineering** with quality metrics
- **Memory-efficient processing**

### Evaluation & Analysis
- **Universal model evaluator** for all trained models
- **Comprehensive metrics** and per-class analysis
- **Automatic logging** to timestamped files
- **Detailed classification reports**
- **Confidence analysis**

### Documentation
- **Complete process documentation** in notebooks
- **Model-specific READMEs** for each classifier
- **Usage examples** and troubleshooting guides
- **Performance comparisons** and insights

## 🔧 Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: For GPU acceleration (CNN)
- `OMP_NUM_THREADS`: For parallel processing

### Model Parameters
- **Fast mode**: Reduced hyperparameter search for quick testing
- **Full mode**: Comprehensive hyperparameter optimization
- **Memory optimization**: `float32` data types, `n_jobs=1`

## 📈 Performance

### Model Comparison
| Model | Accuracy | F1-Score | Training Time | Memory Usage |
|-------|----------|----------|---------------|--------------|
| Logistic Regression | ~92% | ~0.92 | Fast | Low |
| Random Forest | ~94% | ~0.94 | Medium | Medium |
| SVC | ~93% | ~0.93 | Slow | High |
| CNN | ~95% | ~0.95 | Medium | High |

### Data Processing
- **Image processing**: ~2-3 seconds per image
- **Background removal**: ~1-2 seconds per image
- **Quality analysis**: ~0.5 seconds per image
- **Model training**: 5-30 minutes depending on model type

## 🚨 Troubleshooting

### Common Issues
1. **Memory errors**: Use `simple_svc_classifier.py` or reduce data size
2. **Import errors**: Check virtual environment activation
3. **Model not found**: Verify model path with `python scripts/evaluate_model.py --list`
4. **Data not found**: Check if preprocessing steps were completed
5. **Script path errors**: All scripts are now in the `scripts/` directory and use project root paths

### Performance Optimization
- Use `float32` instead of `float64` for large datasets
- Set `n_jobs=1` for memory-constrained environments
- Use fast mode for quick testing
- Enable GPU acceleration for CNN training

### Script Path Updates
All scripts have been moved to the `scripts/` directory and updated to use project root paths. This means:
- Scripts can be run from any directory using `python scripts/script_name.py`
- All data paths are automatically resolved relative to the project root
- No need to change working directory before running scripts

## 📝 Logs and Documentation

All processes generate comprehensive logs:
- **Training logs**: `logs/*_training_*.log`
- **Evaluation logs**: `logs/model_evaluation_*.log`
- **Processing logs**: Console output with progress tracking

## 🎯 Project Goals

1. **Automated background quality prediction** for product images
2. **Robust machine learning pipeline** with multiple model types
3. **Comprehensive evaluation framework** for model comparison
4. **Scalable image processing** with AI-powered background removal
5. **Complete documentation** and analysis for technical requirements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the logs in the `logs/` directory
3. Consult the model-specific READMEs
4. Open an issue on GitHub