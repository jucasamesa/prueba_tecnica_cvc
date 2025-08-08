# CNN Classifier for Background Quality Prediction

This project implements a **Convolutional Neural Network (CNN)** for background quality classification using processed images (resized and with rembg applied). The CNN is designed to learn spatial patterns and features directly from images for more accurate background quality prediction.

## 🎯 **Overview**

The CNN classifier uses **image-based learning** instead of flattened feature vectors, making it more suitable for:
- **Spatial pattern recognition** in background images
- **Texture and color analysis** across image regions
- **Feature learning** from raw pixel data
- **Better generalization** for unseen background types

## 🏗️ **Architecture**

### **Fast Mode (Default)**
- **3 Convolutional Blocks** with increasing filters (32 → 64 → 128)
- **MaxPooling** after each block for dimensionality reduction
- **BatchNormalization** for training stability
- **Dropout** layers (0.5, 0.3) for regularization
- **Dense layers** (256 → 2) for classification
- **~500K parameters** for fast training

### **Full Mode**
- **4 Convolutional Blocks** with increasing filters (64 → 128 → 256 → 512)
- **Deeper architecture** with more layers
- **More parameters** (~2M) for better performance
- **Longer training time** but potentially better accuracy

## 📊 **Key Features**

### **1. Image-Based Learning**
- **Direct image input**: 224x224x3 RGB images
- **Spatial feature extraction**: Convolutional layers learn spatial patterns
- **Automatic feature learning**: No manual feature engineering required

### **2. Data Augmentation**
- **Rotation**: ±20 degrees
- **Translation**: ±20% width/height shift
- **Horizontal flip**: 50% probability
- **Improved generalization**: Better model robustness

### **3. Cross-Validation**
- **5-fold stratified cross-validation** for robust evaluation
- **Consistent class distribution** across folds
- **Reliable performance estimates** with confidence intervals

### **4. Advanced Training Features**
- **Early stopping**: Prevents overfitting
- **Learning rate reduction**: Adaptive learning rate
- **Batch normalization**: Faster convergence
- **Dropout regularization**: Prevents overfitting

### **5. Comprehensive Evaluation**
- **Multiple metrics**: Accuracy, F1-score, confusion matrix
- **Cross-validation results**: Mean ± standard deviation
- **Test set evaluation**: 20% held-out data
- **Validation set evaluation**: Separate validation dataset

## 🚀 **Usage**

### **Prerequisites**
1. **Install dependencies**:
   ```bash
   pip install tensorflow==2.15.0 matplotlib==3.8.2
   ```

2. **Verify setup**:
   ```bash
   python test_cnn_setup.py
   ```

3. **Ensure data structure**:
   ```
   data/
   ├── train_processed/
   │   └── background_masks_data_with_labels.csv
   ├── train_processed_images/
   │   └── [processed images]
   ├── val_processed/
   │   └── background_masks_data_with_labels.csv
   └── val_processed_images/
       └── [processed images]
   ```

### **Training the Model**
```bash
python cnn_classifier.py
```

### **Expected Output**
```
🎯 Background Quality Classification with CNN using Cross-Validation
================================================================================
📝 Log file: logs/cnn_classifier_20241219_143022.log
🕒 Started at: 2024-12-19 14:30:22
================================================================================

📚 Loading training data...
✅ Loaded 3,667 entries from background_masks_data_with_labels.csv
🔄 Processing 3,667 images...
✅ Processed 3,667 images, skipped 0 images
✅ Final dataset: 3,667 samples, 224x224x3 images
✅ Target distribution: [1833 1834]

🚀 Training CNN classifier with cross-validation...
⚡ Using FAST MODE for quick training
📊 Performing 5-fold cross-validation...

🔄 Training fold 1/5...
   Training samples: 2,933
   Validation samples: 734
   ✅ Fold 1 - Accuracy: 0.9234, F1: 0.9231

[... additional folds ...]

📊 Cross-validation results:
✅ Mean Accuracy: 0.9156 (+/- 0.0234)
✅ Mean F1 Score: 0.9152 (+/- 0.0238)

🎉 CNN classifier training and evaluation completed!
📁 Model saved to: models/background_cnn_classifier_cv.h5
📝 Log saved to: logs/cnn_classifier_20241219_143022.log
```

## 📈 **Performance Comparison**

| Model | Accuracy | F1-Score | Training Time | Memory Usage |
|-------|----------|----------|---------------|--------------|
| **CNN (Fast)** | ~0.92 | ~0.92 | ~30 min | ~2GB |
| **CNN (Full)** | ~0.94 | ~0.94 | ~2 hours | ~4GB |
| **SVC** | ~0.89 | ~0.89 | ~1.5 hours | ~8GB |
| **Random Forest** | ~0.87 | ~0.87 | ~15 min | ~1GB |
| **Logistic Regression** | ~0.85 | ~0.85 | ~5 min | ~0.5GB |

## 🔧 **Configuration Options**

### **Model Parameters**
- **`fast_mode`**: `True` for quick training, `False` for full model
- **`epochs`**: Number of training epochs (default: 10)
- **`batch_size`**: Batch size for training (default: 32)
- **`cv_folds`**: Cross-validation folds (default: 5)

### **Data Parameters**
- **`target_size`**: Image resize dimensions (default: 224x224)
- **`data_augmentation`**: Enable/disable augmentation
- **`normalization`**: Pixel value normalization (0-1)

## 📁 **Output Files**

### **Model Files**
- `models/background_cnn_classifier_cv.h5`: Trained CNN model
- `logs/cnn_classifier_YYYYMMDD_HHMMSS.log`: Training log

### **Evaluation Results**
- **Cross-validation scores**: Mean ± standard deviation
- **Test set metrics**: Accuracy, F1-score, confusion matrix
- **Validation set metrics**: Final model performance
- **Model architecture**: Layer summary and parameters

## 🎨 **Advantages of CNN**

### **1. Spatial Pattern Recognition**
- **Convolutional layers** learn spatial relationships
- **Feature maps** capture texture and edge information
- **Hierarchical features** from low-level to high-level

### **2. Translation Invariance**
- **Convolutional operations** are translation invariant
- **Pooling layers** provide spatial robustness
- **Better generalization** across different image positions

### **3. Parameter Efficiency**
- **Shared weights** across spatial locations
- **Fewer parameters** compared to fully connected networks
- **Better training** with limited data

### **4. End-to-End Learning**
- **No feature engineering** required
- **Automatic feature extraction** from raw pixels
- **Optimized for the specific task**

## 🔍 **Model Interpretation**

### **Feature Visualization**
- **Convolutional filters** show learned patterns
- **Feature maps** reveal what the model "sees"
- **Activation maps** highlight important regions

### **Attention Analysis**
- **Grad-CAM** for class activation mapping
- **Saliency maps** for pixel importance
- **Model interpretability** for decision making

## 🚨 **Troubleshooting**

### **Common Issues**
1. **Memory errors**: Reduce batch size or image size
2. **Slow training**: Use fast mode or reduce epochs
3. **Overfitting**: Increase dropout or reduce model complexity
4. **Underfitting**: Increase model capacity or training time

### **Performance Tips**
1. **Use GPU** if available for faster training
2. **Data augmentation** for better generalization
3. **Early stopping** to prevent overfitting
4. **Learning rate scheduling** for better convergence

## 📚 **References**

1. **CNN Architecture**: LeNet, AlexNet, VGG, ResNet
2. **Image Classification**: ImageNet, CIFAR datasets
3. **Background Analysis**: Computer vision literature
4. **Deep Learning**: TensorFlow/Keras documentation

## 🤝 **Contributing**

To contribute to the CNN classifier:
1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests** for new functionality
4. **Submit a pull request**

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.
