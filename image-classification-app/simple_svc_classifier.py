#!/usr/bin/env python3
"""
Simple Support Vector Classifier for Background Classification
This script implements an SVC model using flattened background mask arrays from .npz files.
Uses cross-validation to ensure robust results.
Uses preprocessed background_masks_data_with_labels.csv files and filtered mask arrays.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(masks_path, mapping_path, labels_path):
    """
    Load the mask arrays and prepare them for training.
    
    Args:
        masks_path (str or Path): Path to the .npz file containing mask arrays (filtered)
        mapping_path (str or Path): Path to the CSV mapping file (filtered)
        labels_path (str or Path): Path to the CSV file with labels (preprocessed with _with_labels.csv)
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    print("🔍 Loading data...")
    
    # Load the compressed numpy arrays (filtered)
    masks_data = np.load(masks_path)
    print(f"✅ Loaded {len(masks_data.files)} mask arrays (filtered)")
    
    # Load the mapping (filtered)
    mapping_df = pd.read_csv(mapping_path)
    print(f"✅ Loaded mapping with {len(mapping_df)} entries (filtered)")
    
    # Load the preprocessed labels (already filtered and merged)
    labels_df = pd.read_csv(labels_path)
    print(f"✅ Loaded preprocessed labels with {len(labels_df)} entries")
    
    # Check if the required columns exist
    required_columns = ['original_image_name', 'correct_background?']
    missing_columns = [col for col in required_columns if col not in labels_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in labels file: {missing_columns}")
    
    # Prepare data
    X_list = []
    y_list = []
    
    print("🔄 Processing mask arrays...")
    
    # Track the first flattened mask length to ensure consistency
    first_flattened_length = None
    processed_count = 0
    skipped_count = 0
    
    for idx, row in mapping_df.iterrows():
        image_name = row['original_image_name']
        array_key = row['numpy_array_key']
        
        # Check if we have the mask array
        if array_key not in masks_data:
            print(f"⚠️  Warning: Array key {array_key} not found in mask data")
            skipped_count += 1
            continue
            
        # Get the mask array and flatten it using the correct method
        mask_array = masks_data[array_key]
        flattened_mask = mask_array.flatten()  # Use the correct ndarray.flatten() method
        
        # Check if this is the first mask array
        if first_flattened_length is None:
            first_flattened_length = len(flattened_mask)
            print(f"✅ First mask array length: {first_flattened_length}")
        else:
            # Check if the flattened mask has the same length as the first one
            if len(flattened_mask) != first_flattened_length:
                print(f"⚠️  Warning: Inconsistent mask length for {array_key}")
                print(f"   Expected: {first_flattened_length}, Got: {len(flattened_mask)}")
                print(f"   Original shape: {mask_array.shape}")
                skipped_count += 1
                continue
        
        # Check if we have a label for this image in the preprocessed data
        label_row = labels_df[labels_df['original_image_name'] == image_name]
        if label_row.empty:
            print(f"⚠️  Warning: No label found for {image_name} in preprocessed data")
            skipped_count += 1
            continue
            
        # Get the label (should already be cleaned in preprocessed data)
        label = label_row.iloc[0]['correct_background?']
        
        # Convert label to numeric if needed
        if isinstance(label, str):
            if label == '1':
                label = 1
            elif label == '0':
                label = 0
            else:
                print(f"⚠️  Warning: Unknown label value {label} for {image_name}")
                skipped_count += 1
                continue
        
        X_list.append(flattened_mask)
        y_list.append(label)
        processed_count += 1
    
    print(f"✅ Processed {processed_count} samples, skipped {skipped_count} samples")
    
    if not X_list:
        raise ValueError("No valid data found!")
    
    # Verify all arrays have the same length before converting to numpy array
    lengths = [len(x) for x in X_list]
    unique_lengths = set(lengths)
    
    if len(unique_lengths) > 1:
        raise ValueError(f"Inconsistent mask lengths detected! Found lengths: {sorted(unique_lengths)}")
    
    # Convert to numpy arrays
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"✅ Processed {len(X)} samples with {X.shape[1]} features each")
    print(f"✅ Target distribution: {np.bincount(y)}")
    
    return X, y

def train_svc_classifier_with_cv(X, y, cv_folds=5, random_state=42):
    """
    Train an SVC classifier using cross-validation for robust evaluation.
    
    Args:
        X (np.array): Feature matrix (flattened mask arrays)
        y (np.array): Target vector
        cv_folds (int): Number of cross-validation folds
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (trained_pipeline, evaluation_results)
    """
    print(f"🚀 Training SVC classifier with {cv_folds}-fold cross-validation...")
    
    # Create pipeline with scaling and SVC
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(random_state=random_state, probability=True))
    ])
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'svc__kernel': ['rbf', 'linear']
    }
    
    # Create cross-validation strategy
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Perform grid search with cross-validation
    print("🔍 Performing grid search with cross-validation...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    # Get the best model
    best_pipeline = grid_search.best_estimator_
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Perform additional cross-validation on the best model for detailed metrics
    print("📊 Performing detailed cross-validation evaluation...")
    
    # Cross-validation scores for different metrics
    cv_accuracy = cross_val_score(best_pipeline, X, y, cv=cv, scoring='accuracy')
    cv_f1 = cross_val_score(best_pipeline, X, y, cv=cv, scoring='f1_weighted')
    
    print(f"✅ Cross-validation accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
    print(f"✅ Cross-validation F1 score: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
    
    # Detailed cross-validation results
    print(f"\n📋 Cross-validation results across {cv_folds} folds:")
    print("Fold\tAccuracy\tF1 Score")
    print("-" * 30)
    for i, (acc, f1) in enumerate(zip(cv_accuracy, cv_f1)):
        print(f"{i+1}\t{acc:.4f}\t\t{f1:.4f}")
    
    # Perform final evaluation on a held-out test set (20% of data) with shuffling
    print("\n🔄 Creating train/test split with shuffling...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y, shuffle=True
    )
    
    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    
    # Fit the best model on the training data
    best_pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_pipeline.predict(X_test)
    y_pred_proba = best_pipeline.predict_proba(X_test)
    
    # Calculate test metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n📊 Test set evaluation (20% held-out, shuffled):")
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    print(f"✅ Test F1 Score: {test_f1:.4f}")
    
    # Print detailed classification report
    print("\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\n📊 Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    evaluation_results = {
        'cv_accuracy_mean': cv_accuracy.mean(),
        'cv_accuracy_std': cv_accuracy.std(),
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'test_accuracy': test_accuracy,
        'test_f1_score': test_f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm,
        'best_params': grid_search.best_params_,
        'cv_scores_accuracy': cv_accuracy,
        'cv_scores_f1': cv_f1,
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape
    }
    
    return best_pipeline, evaluation_results

def save_model(pipeline, model_path):
    """
    Save the trained model.
    
    Args:
        pipeline: Trained pipeline
        model_path (str or Path): Path to save the model
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(pipeline, model_path)
    print(f"✅ Model saved to {model_path}")

def main():
    """
    Main function to demonstrate the SVC classifier with cross-validation.
    """
    print("🎯 Background Classification with SVC using Cross-Validation")
    print("=" * 70)
    
    # Define paths for preprocessed and filtered data
    masks_train_path = Path("data/train_processed/background_masks_arrays_filtered.npz")
    mapping_train_path = Path("data/train_processed/mask_arrays_mapping_filtered.csv")
    labels_train_path = Path("data/train_processed/background_masks_data_with_labels.csv")
    
    masks_val_path = Path("data/val_processed/background_masks_arrays_filtered.npz")
    mapping_val_path = Path("data/val_processed/mask_arrays_mapping_filtered.csv")
    labels_val_path = Path("data/val_processed/background_masks_data_with_labels.csv")
    
    # Check if preprocessed files exist
    if not labels_train_path.exists():
        print(f"❌ Preprocessed training labels not found at {labels_train_path}")
        print("Please run preprocess_labels.py first to create the preprocessed files.")
        return
    
    if not labels_val_path.exists():
        print(f"❌ Preprocessed validation labels not found at {labels_val_path}")
        print("Please run preprocess_labels.py first to create the preprocessed files.")
        return
    
    # Check if filtered mask files exist
    if not masks_train_path.exists():
        print(f"❌ Filtered training masks not found at {masks_train_path}")
        print("Please run preprocess_labels.py first to create the filtered files.")
        return
    
    if not masks_val_path.exists():
        print(f"❌ Filtered validation masks not found at {masks_val_path}")
        print("Please run preprocess_labels.py first to create the filtered files.")
        return
    
    # Check if filtered mapping files exist
    if not mapping_train_path.exists():
        print(f"❌ Filtered training mapping not found at {mapping_train_path}")
        print("Please run preprocess_labels.py first to create the filtered files.")
        return
    
    if not mapping_val_path.exists():
        print(f"❌ Filtered validation mapping not found at {mapping_val_path}")
        print("Please run preprocess_labels.py first to create the filtered files.")
        return
    
    try:
        # Load training data
        print("\n📚 Loading training data...")
        X_train_full, y_train_full = load_and_prepare_data(
            masks_train_path, mapping_train_path, labels_train_path
        )
        
        # Load validation data
        print("\n📚 Loading validation data...")
        X_val_full, y_val_full = load_and_prepare_data(
            masks_val_path, mapping_val_path, labels_val_path
        )
        
        # Combine data for cross-validation
        X_combined = np.vstack([X_train_full, X_val_full])
        y_combined = np.hstack([y_train_full, y_val_full])
        
        print(f"\n📊 Combined dataset: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
        
        # Train the SVC classifier with cross-validation
        trained_pipeline, results = train_svc_classifier_with_cv(X_combined, y_combined, cv_folds=5)
        
        # Save the model
        model_path = Path("models/background_svc_classifier_cv.pkl")
        save_model(trained_pipeline, model_path)
        
        print(f"\n🎉 SVC classifier training and evaluation completed!")
        print(f"📁 Model saved to: {model_path}")
        print(f"\n📊 Final Results Summary:")
        print(f"   - Cross-validation Accuracy: {results['cv_accuracy_mean']:.4f} (+/- {results['cv_accuracy_std'] * 2:.4f})")
        print(f"   - Cross-validation F1 Score: {results['cv_f1_mean']:.4f} (+/- {results['cv_f1_std'] * 2:.4f})")
        print(f"   - Test Set Accuracy: {results['test_accuracy']:.4f}")
        print(f"   - Test Set F1 Score: {results['test_f1_score']:.4f}")
        print(f"   - Best Parameters: {results['best_params']}")
        print(f"   - Training Set Size: {results['X_train_shape'][0]} samples")
        print(f"   - Test Set Size: {results['X_test_shape'][0]} samples")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
