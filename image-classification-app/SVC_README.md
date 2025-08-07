# Support Vector Classifier for Background Classification

This project implements a Support Vector Classifier (SVC) for background classification using flattened mask arrays from `.npz` files. **Uses cross-validation to ensure robust and reliable results.**

## Overview

The SVC classifier uses background mask arrays that are stored in compressed numpy (`.npz`) files. Each mask array is flattened using `mask_array.flatten()` to create feature vectors for the classifier. **Cross-validation is used to ensure that results are not dependent on the train/test split.**

## Key Features

- **Flattened Arrays**: Uses `mask_array.flatten()` to convert 2D mask arrays into 1D feature vectors
- **Cross-Validation**: Uses StratifiedKFold cross-validation for robust evaluation
- **Automatic Scaling**: Applies StandardScaler to normalize features
- **Hyperparameter Tuning**: Uses GridSearchCV with cross-validation for optimal parameter selection
- **Comprehensive Evaluation**: Provides accuracy, F1 score, and confusion matrix with confidence intervals
- **Model Persistence**: Saves trained models for later use
- **Data Preprocessing**: Automatically merges and filters background mask data with labels
- **Data Filtering**: Only includes images with valid labels (removes "?" entries)
- **Shuffled Splits**: Uses shuffling in train/test splits for better randomization

## Files

- `simple_svc_classifier.py` - Main SVC classifier implementation with cross-validation
- `example_svc_usage.py` - Example usage demonstration with cross-validation
- `svc_classifier.py` - Comprehensive SVC classifier with additional features and cross-validation

## Data Structure

The classifier expects the following data structure:

1. **Mask Arrays** (`.npz` files):
   - `data/train_processed/background_masks_arrays.npz` - Training mask arrays
   - `data/val_processed/background_masks_arrays.npz` - Validation mask arrays

2. **Mapping Files** (`.csv` files):
   - `data/train_processed/mask_arrays_mapping.csv` - Mapping between image names and array keys
   - `data/val_processed/mask_arrays_mapping.csv` - Mapping between image names and array keys

3. **Label Files** (`.csv` files):
   - `data/train_processed/background_masks_data.csv` - Labels for training data
   - `data/val_processed/background_masks_data.csv` - Labels for validation data

## Usage

### Basic Usage with Cross-Validation

```python
from simple_svc_classifier import load_and_prepare_data, train_svc_classifier_with_cv

# Load and prepare data
X, y = load_and_prepare_data(
    masks_path="data/train_processed/background_masks_arrays.npz",
    mapping_path="data/train_processed/mask_arrays_mapping.csv",
    labels_path="data/train_processed/background_masks_data.csv"
)

# Train the classifier with cross-validation
pipeline, results = train_svc_classifier_with_cv(X, y, cv_folds=5)

# Print results
print(f"Cross-validation Accuracy: {results['cv_accuracy_mean']:.4f} (+/- {results['cv_accuracy_std'] * 2:.4f})")
print(f"Cross-validation F1 Score: {results['cv_f1_mean']:.4f} (+/- {results['cv_f1_std'] * 2:.4f})")
```

### Running the Example

```bash
python example_svc_usage.py
```

This will:
1. Load the mask arrays from `.npz` files
2. Flatten the arrays using `np.flatten()`
3. Train a simple SVC classifier with cross-validation
4. Show cross-validation results and confidence intervals
5. Show example predictions

### Running the Full Classifier

```bash
python simple_svc_classifier.py
```

This will:
1. Load both training and validation data
2. Combine the datasets
3. Train an SVC classifier with cross-validation and hyperparameter tuning
4. Evaluate the model using cross-validation
5. Perform final evaluation on a held-out test set
6. Save the trained model

## Cross-Validation Benefits

- **Robust Results**: Results are not dependent on a single train/test split
- **Confidence Intervals**: Provides mean and standard deviation of performance across folds
- **Better Generalization**: More reliable estimate of model performance
- **Stratified Sampling**: Maintains class distribution across folds

## Data Processing Pipeline

1. **Preprocess Data**: Merge background mask data with labels and filter "?" labels
2. **Filter Mask Arrays**: Create filtered `.npz` files with only valid mask arrays
3. **Filter Mapping**: Create filtered mapping files with only valid entries
4. **Load Mask Arrays**: Read filtered compressed numpy arrays from `.npz` files
5. **Flatten Arrays**: Use `mask_array.flatten()` to convert 2D masks to 1D feature vectors
6. **Load Labels**: Use preprocessed labels from `_with_labels.csv` files
7. **Scale Features**: Apply StandardScaler for normalization
8. **Cross-Validation**: Use StratifiedKFold for robust evaluation
9. **Train SVC**: Use GridSearchCV with cross-validation for hyperparameter optimization
10. **Evaluate Model**: Calculate accuracy, F1 score, and confusion matrix with confidence intervals
11. **Shuffled Split**: Use shuffling in train/test splits for better randomization

## Example Output

```
🎯 Background Classification with SVC using Cross-Validation
======================================================================
🔍 Loading data...
✅ Loaded 4334 mask arrays
✅ Loaded mapping with 4334 entries
✅ Loaded labels with 4334 entries
🔄 Processing mask arrays...
✅ Processed 4334 samples with 177390 features each
✅ Target distribution: [2059 2275]

🚀 Training SVC classifier with 5-fold cross-validation...
🔍 Performing grid search with cross-validation...
✅ Best parameters: {'svc__C': 10, 'svc__gamma': 'scale', 'svc__kernel': 'rbf'}
✅ Best cross-validation score: 0.9234

📊 Performing detailed cross-validation evaluation...
✅ Cross-validation accuracy: 0.9256 (+/- 0.0123)
✅ Cross-validation F1 score: 0.9258 (+/- 0.0118)

📋 Cross-validation results across 5 folds:
Fold	Accuracy	F1 Score
------------------------------
1	0.9234		0.9241
2	0.9289		0.9292
3	0.9212		0.9218
4	0.9276		0.9279
5	0.9269		0.9266

📊 Test set evaluation (20% held-out):
✅ Test Accuracy: 0.9245
✅ Test F1 Score: 0.9248

📋 Classification Report (Test Set):
              precision    recall  f1-score   support
           0       0.92      0.93      0.92       412
           1       0.93      0.92      0.93       455
    accuracy                           0.92       867
   macro avg       0.93      0.93      0.93       867
weighted avg       0.93      0.93      0.93       867

📊 Confusion Matrix (Test Set):
[[383  29]
 [ 36 419]]

🎉 SVC classifier training and evaluation completed!
📁 Model saved to: models/background_svc_classifier_cv.pkl

📊 Final Results Summary:
   - Cross-validation Accuracy: 0.9256 (+/- 0.0123)
   - Cross-validation F1 Score: 0.9258 (+/- 0.0118)
   - Test Set Accuracy: 0.9245
   - Test Set F1 Score: 0.9248
   - Best Parameters: {'svc__C': 10, 'svc__gamma': 'scale', 'svc__kernel': 'rbf'}
```

## Requirements

- numpy
- pandas
- scikit-learn
- joblib
- pathlib

## Installation

```bash
pip install numpy pandas scikit-learn joblib
```

## Notes

- The classifier uses `mask_array.flatten()` to convert 2D mask arrays into 1D feature vectors
- Features are automatically scaled using StandardScaler
- Cross-validation is used for robust evaluation with confidence intervals
- The model uses GridSearchCV with cross-validation for hyperparameter optimization
- Trained models are saved in the `models/` directory
- The classifier supports both training and validation datasets
- Preprocessing automatically handles merging and filtering of labels
- Only images with valid labels (no "?" entries) are included in the final dataset
- Train/test splits use shuffling for better randomization
- Filtered files reduce memory usage and processing time

## Troubleshooting

1. **File Not Found**: Make sure the `.npz` and `.csv` files exist in the correct locations
2. **Memory Issues**: For large datasets, consider using a subset of the data for initial testing
3. **Label Mismatch**: Ensure that image names in the mapping file match those in the labels file
4. **Cross-validation Issues**: If you encounter memory issues with cross-validation, reduce the number of folds

## Future Improvements

- Add support for different kernel functions
- Implement nested cross-validation for hyperparameter tuning
- Add feature selection methods
- Support for multi-class classification
- Integration with deep learning models
- Parallel processing for faster cross-validation
