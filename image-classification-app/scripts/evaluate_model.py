#!/usr/bin/env python3
"""
General Model Evaluator for Validation Data
This script loads any trained model and evaluates it on validation data.
Supports all trained models: Logistic Regression, SVC, Random Forest, CNN, etc.

Usage:
    python scripts/evaluate_model.py --list                    # List available models
    python scripts/evaluate_model.py --model model_name.pkl    # Evaluate specific model
    python scripts/evaluate_model.py                           # Evaluate first available model
    python scripts/evaluate_model.py --no-log                  # Don't save logs to file

eg. scripts/python evaluate_model.py --model "background_logistic_regression_classifier_cv.pkl"
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import joblib
import warnings
import sys
import datetime
import argparse
import os
import cv2
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class Logger:
    """Custom logger to capture all output and save to file."""
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.terminal = sys.stdout
        self.log = open(log_file_path, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

def load_validation_data(model_type=None):
    """
    Load validation data for evaluation based on model type.
    
    Args:
        model_type (str): Type of model to determine data loading strategy
    
    Returns:
        tuple: (X_val, y_val) validation features and labels
    """
    print("🔍 Loading validation data...")
    
    # Define paths for validation data (relative to project root)
    project_root = Path(__file__).parent.parent
    
    # Check if we need to load specific features (for Random Forest with features)
    if model_type and "random_forest" in model_type.lower() and "features" in model_type.lower():
        print("📊 Loading validation data for Random Forest with ImageAnalyzer features...")
        return load_validation_data_features(project_root)
    
    # Check if we need to load images (for CNN)
    elif model_type and "cnn" in model_type.lower():
        print("🖼️  Loading validation data for CNN...")
        return load_validation_data_images(project_root)
    
    # Default: load flattened arrays (for Logistic Regression, SVC, etc.)
    else:
        print("📊 Loading validation data for flattened arrays...")
        return load_validation_data_arrays(project_root)

def load_validation_data_arrays(project_root):
    """
    Load validation data as flattened arrays (for Logistic Regression, SVC, etc.).
    
    Returns:
        tuple: (X_val, y_val) validation features and labels
    """
    masks_val_path = project_root / "data/val_processed/background_masks_arrays_filtered.npz"
    mapping_val_path = project_root / "data/val_processed/mask_arrays_mapping_filtered.csv"
    labels_val_path = project_root / "data/val_processed/background_masks_data_with_labels.csv"
    
    # Check if files exist
    if not masks_val_path.exists():
        raise FileNotFoundError(f"Validation masks not found at {masks_val_path}")
    if not mapping_val_path.exists():
        raise FileNotFoundError(f"Validation mapping not found at {mapping_val_path}")
    if not labels_val_path.exists():
        raise FileNotFoundError(f"Validation labels not found at {labels_val_path}")
    
    # Load the compressed numpy arrays
    masks_data = np.load(masks_val_path)
    print(f"✅ Loaded {len(masks_data.files)} validation mask arrays")
    
    # Load the mapping
    mapping_df = pd.read_csv(mapping_val_path)
    print(f"✅ Loaded validation mapping with {len(mapping_df)} entries")
    
    # Load the labels
    labels_df = pd.read_csv(labels_val_path)
    print(f"✅ Loaded validation labels with {len(labels_df)} entries")
    
    # Check required columns
    required_columns = ['original_image_name', 'correct_background?']
    missing_columns = [col for col in required_columns if col not in labels_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in labels file: {missing_columns}")
    
    # Prepare data
    X_list = []
    y_list = []
    processed_count = 0
    skipped_count = 0
    
    print("🔄 Processing validation mask arrays...")
    
    for idx, row in mapping_df.iterrows():
        image_name = row['original_image_name']
        array_key = row['numpy_array_key']
        
        # Check if we have the mask array
        if array_key not in masks_data:
            print(f"⚠️  Warning: Array key {array_key} not found in mask data")
            skipped_count += 1
            continue
        
        # Get the mask array and flatten it
        mask_array = masks_data[array_key]
        flattened_mask = mask_array.flatten()
        
        # Find corresponding label
        label_row = labels_df[labels_df['original_image_name'] == image_name]
        if label_row.empty:
            print(f"⚠️  Warning: No label found for {image_name}")
            skipped_count += 1
            continue
        
        label = label_row['correct_background?'].iloc[0]
        
        # Store data
        X_list.append(flattened_mask)
        y_list.append(label)
        processed_count += 1
    
    print(f"✅ Processed {processed_count} validation samples")
    if skipped_count > 0:
        print(f"⚠️  Skipped {skipped_count} samples")
    
    # Convert to numpy arrays
    X_val = np.array(X_list, dtype=np.float32)
    y_val = np.array(y_list, dtype=np.int8)
    
    print(f"📊 Validation data shape: {X_val.shape}")
    print(f"📊 Validation labels shape: {y_val.shape}")
    
    return X_val, y_val

def resize_validation_data(X_val, expected_features):
    """
    Resize validation data to match the expected number of features.
    
    Args:
        X_val (np.array): Validation features
        expected_features (int): Expected number of features
    
    Returns:
        np.array: Resized validation features
    """
    current_features = X_val.shape[1]
    
    if current_features == expected_features:
        print(f"✅ Validation data already has the correct size: {current_features} features")
        return X_val
    
    print(f"🔄 Resizing validation data from {current_features} to {expected_features} features...")
    
    if current_features > expected_features:
        # Need to downsample - take first expected_features
        print(f"   Downsampling: taking first {expected_features} features")
        X_val_resized = X_val[:, :expected_features]
    else:
        # Need to upsample - pad with zeros
        print(f"   Upsampling: padding with zeros to {expected_features} features")
        X_val_resized = np.zeros((X_val.shape[0], expected_features), dtype=np.float32)
        X_val_resized[:, :current_features] = X_val
    
    print(f"✅ Resized validation data shape: {X_val_resized.shape}")
    return X_val_resized

def load_validation_data_features(project_root):
    """
    Load validation data with specific ImageAnalyzer features (for Random Forest with features).
    
    Returns:
        tuple: (X_val, y_val) validation features and labels
    """
    csv_path = project_root / "data/val_processed/background_masks_data_with_labels.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found at {csv_path}")
    
    # Load the CSV data
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded CSV with {len(df)} entries")
    
    # Check if the required columns exist
    required_features = [
        'brightness', 'contrast', 'entropy', 'noise_level', 'saturation', 'value',
        'dominant_color_r', 'dominant_color_g', 'dominant_color_b', 
        'dominant_color_h', 'dominant_color_s', 'dominant_color_v'
    ]
    
    missing_features = [col for col in required_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Check if target column exists
    if 'correct_background?' not in df.columns:
        raise ValueError("Missing target column 'correct_background?'")
    
    # Select only the required features
    X = df[required_features].copy()
    y = df['correct_background?'].copy()
    
    # Handle missing values
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️  Found {missing_count} missing values. Filling with median...")
        X = X.fillna(X.median())
    
    # Convert to numpy arrays
    X_val = X.values.astype(np.float32)
    y_val = y.values.astype(np.int8)
    
    print(f"✅ Prepared {len(X_val)} samples with {X_val.shape[1]} features")
    print(f"✅ Target distribution: {np.bincount(y_val)}")
    
    return X_val, y_val

def load_validation_data_images(project_root):
    """
    Load validation data as images (for CNN).
    
    Returns:
        tuple: (X_val, y_val) validation features and labels
    """
    csv_path = project_root / "data/val_processed/background_masks_data_with_labels.csv"
    images_dir = project_root / "data/val_processed_images"
    target_size = (224, 224)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found at {csv_path}")
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Validation images directory not found at {images_dir}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} entries from {csv_path.name}")
    
    # Check if required columns exist
    required_columns = ['processed_image_name', 'correct_background?']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV file: {missing_columns}")
    
    # Prepare data
    X_list = []
    y_list = []
    processed_count = 0
    skipped_count = 0
    
    print(f"🔄 Processing {len(df)} images...")
    
    for idx, row in df.iterrows():
        try:
            # Get the processed image path
            processed_image_name = row['processed_image_name']
            image_path = images_dir / processed_image_name
            
            # Check if image exists
            if not image_path.exists():
                print(f"⚠️  Warning: Image not found: {image_path}")
                skipped_count += 1
                continue
            
            # Load and preprocess the image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"⚠️  Warning: Could not load image: {image_path}")
                skipped_count += 1
                continue
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image to target size
            image_resized = cv2.resize(image_rgb, target_size)
            
            # Normalize pixel values to [0, 1]
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Get the label
            label = row['correct_background?']
            
            # Convert label to numeric if needed
            if isinstance(label, str):
                if label == '1':
                    label = 1
                elif label == '0':
                    label = 0
                else:
                    print(f"⚠️  Warning: Unknown label value {label} for {processed_image_name}")
                    skipped_count += 1
                    continue
            
            X_list.append(image_normalized)
            y_list.append(label)
            processed_count += 1
            
        except Exception as e:
            print(f"⚠️  Warning: Error processing image {row.get('processed_image_name', 'unknown')}: {e}")
            skipped_count += 1
            continue
    
    print(f"✅ Processed {processed_count} images, skipped {skipped_count} images")
    
    if not X_list:
        raise ValueError("No valid images found!")
    
    # Convert to numpy arrays
    X_val = np.array(X_list, dtype=np.float32)
    y_val = np.array(y_list, dtype=np.int8)
    
    print(f"✅ Final dataset: {X_val.shape[0]} samples, {X_val.shape[1]}x{X_val.shape[2]}x{X_val.shape[3]} images")
    print(f"✅ Target distribution: {np.bincount(y_val)}")
    
    return X_val, y_val

def evaluate_model(model_path, X_val, y_val):
    """
    Evaluate the trained model on validation data.
    
    Args:
        model_path (Path): Path to the trained model
        X_val (np.array): Validation features
        y_val (np.array): Validation labels
    
    Returns:
        dict: Evaluation results
    """
    print(f"\n🔍 Loading trained model from {model_path}...")
    
    # Load the model
    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        
        # Get model type/name for logging
        model_type = type(model).__name__
        if hasattr(model, 'steps') and len(model.steps) > 0:
            # It's a pipeline, get the final estimator
            final_estimator = model.steps[-1][1]
            model_type = f"Pipeline({type(final_estimator).__name__})"
        
        print(f"📋 Model type: {model_type}")
        
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")
    
    # Check if we need to resize the validation data
    if hasattr(model, 'n_features_in_'):
        expected_features = model.n_features_in_
        print(f"📊 Model expects {expected_features} features")
        
        if X_val.shape[1] != expected_features:
            X_val = resize_validation_data(X_val, expected_features)
    elif hasattr(model, 'steps') and len(model.steps) > 0:
        # It's a pipeline, check the first step (usually scaler)
        first_step = model.steps[0][1]
        if hasattr(first_step, 'n_features_in_'):
            expected_features = first_step.n_features_in_
            print(f"📊 Model pipeline expects {expected_features} features")
            
            if X_val.shape[1] != expected_features:
                X_val = resize_validation_data(X_val, expected_features)
    
    # Make predictions
    print("🚀 Making predictions on validation data...")
    try:
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)
        print("✅ Predictions completed successfully")
    except Exception as e:
        raise Exception(f"Failed to make predictions: {e}")
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    
    # Calculate per-class metrics
    f1_per_class = f1_score(y_val, y_pred, average=None)
    precision_per_class = precision_score(y_val, y_pred, average=None)
    recall_per_class = recall_score(y_val, y_pred, average=None)
    
    # Create results dictionary
    results = {
        'model_type': model_type,
        'model_path': str(model_path),
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'f1_per_class': f1_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'y_true': y_val,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return results

def print_evaluation_results(results):
    """
    Print comprehensive evaluation results.
    
    Args:
        results (dict): Evaluation results
    """
    print("\n" + "="*70)
    print(f"📊 MODEL EVALUATION ON VALIDATION DATA")
    print("="*70)
    
    # Model information
    print(f"\n🤖 MODEL INFORMATION:")
    print(f"   Model type: {results['model_type']}")
    print(f"   Model path: {results['model_path']}")
    
    # Overall metrics
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   F1 Score:  {results['f1_score']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    
    # Per-class metrics
    print(f"\n📈 PER-CLASS PERFORMANCE:")
    classes = ['Background No Cumple (0)', 'Background Cumple (1)']
    for i, class_name in enumerate(classes):
        print(f"   {class_name}:")
        print(f"     - F1 Score:  {results['f1_per_class'][i]:.4f}")
        print(f"     - Precision: {results['precision_per_class'][i]:.4f}")
        print(f"     - Recall:    {results['recall_per_class'][i]:.4f}")
    
    # Confusion matrix
    print(f"\n📊 CONFUSION MATRIX:")
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    print("   Predicted:")
    print("             0    1")
    print(f"   Actual 0: {cm[0][0]:4d} {cm[0][1]:4d}")
    print(f"         1: {cm[1][0]:4d} {cm[1][1]:4d}")
    
    # Classification report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(results['y_true'], results['y_pred'], 
                              target_names=['Background No Cumple', 'Background Cumple']))
    
    # Additional insights
    print(f"\n💡 ADDITIONAL INSIGHTS:")
    total_samples = len(results['y_true'])
    class_0_samples = np.sum(results['y_true'] == 0)
    class_1_samples = np.sum(results['y_true'] == 1)
    
    print(f"   Total validation samples: {total_samples}")
    print(f"   Background No Cumple (0): {class_0_samples} ({class_0_samples/total_samples*100:.1f}%)")
    print(f"   Background Cumple (1):    {class_1_samples} ({class_1_samples/total_samples*100:.1f}%)")
    
    # Model confidence analysis
    max_proba = np.max(results['y_pred_proba'], axis=1)
    avg_confidence = np.mean(max_proba)
    print(f"   Average prediction confidence: {avg_confidence:.4f}")
    
    print("\n" + "="*70)

def get_available_models():
    """
    Get list of available trained models.
    
    Returns:
        list: List of available model files
    """
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    if not models_dir.exists():
        return []
    
    model_files = list(models_dir.glob("*.pkl"))
    return sorted(model_files)  # Sort for consistent ordering

def main():
    """
    Main function to evaluate any trained model.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate any trained model on validation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/evaluate_model.py --list                    # List available models
    python scripts/evaluate_model.py --model model_name.pkl    # Evaluate specific model
    python scripts/evaluate_model.py                           # Evaluate first available model
    python scripts/evaluate_model.py --no-log                  # Don't save logs to file
        """
    )
    
    parser.add_argument(
        "--model", "-m", 
        type=str, 
        help="Name or path of the model file to evaluate (e.g., 'background_random_forest_classifier_features_fast.pkl')"
    )
    parser.add_argument(
        "--list", "-l", 
        action="store_true", 
        help="List all available trained models in the models/ directory"
    )
    parser.add_argument(
        "--no-log", 
        action="store_true", 
        help="Don't save evaluation logs to file (output only to terminal)"
    )
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list:
        print("📁 Available trained models:")
        print("="*50)
        available_models = get_available_models()
        if available_models:
            for i, model_path in enumerate(available_models, 1):
                # Get file size
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"   {i:2d}. {model_path.name} ({size_mb:.1f} MB)")
            print(f"\nTotal: {len(available_models)} model(s) found")
            print("\nTo evaluate a specific model, use:")
            print("   python scripts/evaluate_model.py --model model_name.pkl")
        else:
            print("   No trained models found in models/ directory")
            print("\nPlease train a model first using one of the classifier scripts:")
            print("   python scripts/random_forest_classifier.py --mode fast")
            print("   python scripts/logistic_regression_classifier.py")
            print("   python scripts/simple_svc_classifier.py --mode fast")
        return
    
    # Set up logging
    if not args.no_log:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = Path(__file__).parent.parent
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file_path = log_dir / f"model_evaluation_{timestamp}.log"
        
        # Redirect stdout to both terminal and log file
        original_stdout = sys.stdout
        logger = Logger(log_file_path)
        sys.stdout = logger
    
    try:
        print("🎯 General Model Evaluator for Validation Data")
        print("="*70)
        if not args.no_log:
            print(f"📝 Log file: {log_file_path}")
        print(f"🕒 Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Determine model path
        if args.model:
            # Check if it's a full path or just a filename
            model_path = Path(args.model)
            if not model_path.is_absolute():
                # It's a filename, look in models directory
                project_root = Path(__file__).parent.parent
                model_path = project_root / "models" / args.model
        else:
            # Try to find the most recent model
            available_models = get_available_models()
            if not available_models:
                print("❌ No trained models found in models/ directory")
                print("\nPlease train a model first or specify a model path with --model")
                print("\nAvailable options:")
                print("   python scripts/evaluate_model.py --list                    # List available models")
                print("   python scripts/evaluate_model.py --model model_name.pkl    # Evaluate specific model")
                return
            
            # Use the first available model
            model_path = available_models[0]
            print(f"🔍 No model specified, using: {model_path.name}")
        
        if not model_path.exists():
            print(f"❌ Model not found at {model_path}")
            print("\nAvailable models:")
            available_models = get_available_models()
            for model_file in available_models:
                print(f"   - {model_file.name}")
            print(f"\nUse --list to see all available models")
            return
        
        # Detect model type from filename
        model_name = model_path.name.lower()
        model_type = None
        
        if "random_forest" in model_name and "features" in model_name:
            model_type = "random_forest_features"
        elif "cnn" in model_name:
            model_type = "cnn"
        elif "logistic_regression" in model_name:
            model_type = "logistic_regression"
        elif "svc" in model_name:
            model_type = "svc"
        else:
            # Default to arrays for unknown models
            model_type = "arrays"
        
        print(f"🔍 Detected model type: {model_type}")
        
        # Load validation data based on model type
        X_val, y_val = load_validation_data(model_type)
        
        # Evaluate model
        results = evaluate_model(model_path, X_val, y_val)
        
        # Print results
        print_evaluation_results(results)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Model evaluated: {model_path.name}")
        print(f"📊 Validation samples: {len(y_val)}")
        if not args.no_log:
            print(f"📝 Log saved to: {log_file_path}")
        print(f"\n🕒 Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore stdout and close log file
        if not args.no_log:
            sys.stdout = original_stdout
            logger.close()
            print(f"📝 Log saved to: {log_file_path}")

if __name__ == "__main__":
    main()
