#!/usr/bin/env python3
"""
General Model Evaluator for Validation Data
This script loads any trained model and evaluates it on validation data.
Supports all trained models: Logistic Regression, SVC, Random Forest, CNN, etc.
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

def load_validation_data():
    """
    Load validation data for evaluation.
    
    Returns:
        tuple: (X_val, y_val) validation features and labels
    """
    print("🔍 Loading validation data...")
    
    # Define paths for validation data (relative to project root)
    project_root = Path(__file__).parent.parent
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
    return model_files

def main():
    """
    Main function to evaluate any trained model.
    """
    parser = argparse.ArgumentParser(description="Evaluate any trained model on validation data")
    parser.add_argument("--model", "-m", type=str, help="Path to the model file to evaluate")
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    parser.add_argument("--no-log", action="store_true", help="Don't save logs to file")
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list:
        print("📁 Available trained models:")
        available_models = get_available_models()
        if available_models:
            for i, model_path in enumerate(available_models, 1):
                print(f"   {i}. {model_path.name}")
        else:
            print("   No trained models found in models/ directory")
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
            model_path = Path(args.model)
        else:
            # Try to find the most recent model
            available_models = get_available_models()
            if not available_models:
                print("❌ No trained models found in models/ directory")
                print("Please train a model first or specify a model path with --model")
                return
            
            # Use the first available model
            model_path = available_models[0]
            print(f"🔍 No model specified, using: {model_path.name}")
        
        if not model_path.exists():
            print(f"❌ Model not found at {model_path}")
            print("Available models:")
            available_models = get_available_models()
            for model_file in available_models:
                print(f"   - {model_file}")
            return
        
        # Load validation data
        X_val, y_val = load_validation_data()
        
        # Evaluate model
        results = evaluate_model(model_path, X_val, y_val)
        
        # Print results
        print_evaluation_results(results)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Model evaluated: {model_path}")
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
