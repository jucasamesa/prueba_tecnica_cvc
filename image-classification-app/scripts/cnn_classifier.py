#!/usr/bin/env python3
"""
CNN Classifier for Background Classification
This script implements a Convolutional Neural Network (CNN) model for background quality prediction.
Uses image-based learning (not just flattened arrays) for better feature extraction.
Includes data augmentation, early stopping, and learning rate reduction.

Usage:
    python scripts/cnn_classifier.py --mode fast    # Fast mode (smaller model, faster training)
    python scripts/cnn_classifier.py --mode full    # Full mode (larger model, better performance)
    python scripts/cnn_classifier.py                # Default: fast mode
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import warnings
import sys
import datetime
import cv2
import matplotlib.pyplot as plt
import argparse
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

def load_and_prepare_data(csv_path: Path, images_dir: Path, target_size=(224, 224)):
    """
    Load images and labels for CNN training.
    
    Args:
        csv_path (Path): Path to the CSV file with image data and labels
        images_dir (Path): Directory containing the processed images
        target_size (tuple): Target size for image resizing (height, width)
        
    Returns:
        tuple: (X, y) where X is the image array and y is the target vector
    """
    print("🔍 Loading data for CNN...")
    
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
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)
    
    print(f"✅ Final dataset: {X.shape[0]} samples, {X.shape[1]}x{X.shape[2]}x{X.shape[3]} images")
    print(f"✅ Target distribution: {np.bincount(y)}")
    print(f"✅ Memory usage: X={X.nbytes / 1024**2:.2f} MB, y={y.nbytes / 1024**2:.2f} MB")
    
    return X, y

def create_cnn_model(input_shape=(224, 224, 3), num_classes=2, fast_mode=True):
    """
    Create a CNN model for background quality classification.
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels)
        num_classes (int): Number of classes (2 for binary classification)
        fast_mode (bool): If True, use a smaller model for faster training
        
    Returns:
        keras.Model: Compiled CNN model
    """
    if fast_mode:
        print("⚡ Creating FAST MODE CNN model (smaller architecture)")
        
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
    else:
        print("🔍 Creating FULL MODE CNN model (larger architecture)")
        
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Second convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Third convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Fourth convolutional block
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ CNN model created with {model.count_params():,} parameters")
    return model

def train_cnn_with_cv(X, y, cv_folds=5, random_state=42, fast_mode=True, epochs=10, batch_size=32):
    """
    Train a CNN model using cross-validation for robust evaluation.
    
    Args:
        X (np.array): Image array (samples, height, width, channels)
        y (np.array): Target vector
        cv_folds (int): Number of cross-validation folds
        random_state (int): Random state for reproducibility
        fast_mode (bool): If True, use smaller model for faster training
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (trained_model, evaluation_results)
    """
    print(f"🚀 Training CNN model with {cv_folds}-fold cross-validation...")
    
    # Create cross-validation strategy
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Store cross-validation results
    cv_accuracy_scores = []
    cv_f1_scores = []
    cv_models = []
    
    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    print(f"📊 Performing {cv_folds}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        print(f"\n🔄 Training fold {fold}/{cv_folds}...")
        
        # Split data for this fold
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        print(f"   Training samples: {len(X_train_fold)}")
        print(f"   Validation samples: {len(X_val_fold)}")
        
        # Create model for this fold
        model = create_cnn_model(input_shape=X.shape[1:], fast_mode=fast_mode)
        
        # Early stopping callback
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Reduce learning rate callback
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train the model
        history = model.fit(
            datagen.flow(X_train_fold, y_train_fold, batch_size=batch_size),
            steps_per_epoch=len(X_train_fold) // batch_size,
            epochs=epochs,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val_fold)
        y_val_pred_classes = np.argmax(y_val_pred, axis=1)
        
        # Calculate metrics
        val_accuracy = accuracy_score(y_val_fold, y_val_pred_classes)
        val_f1 = f1_score(y_val_fold, y_val_pred_classes, average='weighted')
        
        cv_accuracy_scores.append(val_accuracy)
        cv_f1_scores.append(val_f1)
        cv_models.append(model)
        
        print(f"   ✅ Fold {fold} - Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
    
    # Calculate cross-validation statistics
    cv_accuracy_mean = np.mean(cv_accuracy_scores)
    cv_accuracy_std = np.std(cv_accuracy_scores)
    cv_f1_mean = np.mean(cv_f1_scores)
    cv_f1_std = np.std(cv_f1_scores)
    
    print(f"\n📊 Cross-validation results:")
    print(f"✅ Mean Accuracy: {cv_accuracy_mean:.4f} (+/- {cv_accuracy_std * 2:.4f})")
    print(f"✅ Mean F1 Score: {cv_f1_mean:.4f} (+/- {cv_f1_std * 2:.4f})")
    
    # Detailed cross-validation results
    print(f"\n📋 Cross-validation results across {cv_folds} folds:")
    print("Fold\tAccuracy\tF1 Score")
    print("-" * 30)
    for i, (acc, f1) in enumerate(zip(cv_accuracy_scores, cv_f1_scores)):
        print(f"{i+1}\t{acc:.4f}\t\t{f1:.4f}")
    
    # Use the best model (highest accuracy) for final evaluation
    best_model_idx = np.argmax(cv_accuracy_scores)
    best_model = cv_models[best_model_idx]
    print(f"\n🏆 Best model: Fold {best_model_idx + 1} (Accuracy: {cv_accuracy_scores[best_model_idx]:.4f})")
    
    # Perform final evaluation on a held-out test set
    print("\n🔄 Creating train/test split for final evaluation...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y, shuffle=True
    )
    
    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    
    # Train final model on full training data
    final_model = create_cnn_model(input_shape=X.shape[1:], fast_mode=fast_mode)
    
    # Early stopping for final training
    early_stopping = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train final model
    final_history = final_model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on test set
    y_test_pred = final_model.predict(X_test)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    
    # Calculate test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred_classes)
    test_f1 = f1_score(y_test, y_test_pred_classes, average='weighted')
    
    print(f"\n📊 Test set evaluation (20% held-out):")
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    print(f"✅ Test F1 Score: {test_f1:.4f}")
    
    # Print detailed classification report
    print("\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred_classes))
    
    # Print confusion matrix
    print("\n📊 Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred_classes)
    print(cm)
    
    evaluation_results = {
        'cv_accuracy_mean': cv_accuracy_mean,
        'cv_accuracy_std': cv_accuracy_std,
        'cv_f1_mean': cv_f1_mean,
        'cv_f1_std': cv_f1_std,
        'test_accuracy': test_accuracy,
        'test_f1_score': test_f1,
        'predictions': y_test_pred_classes,
        'probabilities': y_test_pred,
        'confusion_matrix': cm,
        'cv_scores_accuracy': cv_accuracy_scores,
        'cv_scores_f1': cv_f1_scores,
        'fast_mode': fast_mode,
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'model_architecture': final_model.summary()
    }
    
    return final_model, evaluation_results

def save_model(model, model_path):
    """
    Save the trained CNN model.
    
    Args:
        model: Trained Keras model
        model_path (str or Path): Path to save the model
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    model.save(model_path)
    print(f"✅ CNN model saved to {model_path}")

def main():
    """
    Main function to demonstrate the CNN classifier with cross-validation.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='CNN Classifier for Background Classification using Cross-Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/cnn_classifier.py --mode fast    # Fast mode (smaller model, faster training)
    python scripts/cnn_classifier.py --mode full    # Full mode (larger model, better performance)
    python scripts/cnn_classifier.py                # Default: fast mode
        """
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['fast', 'full'], 
        default='fast',
        help='Training mode: fast (smaller model, faster training) or full (larger model, better performance)'
    )
    
    args = parser.parse_args()
    
    # Convert mode to boolean
    fast_mode = args.mode == 'fast'
    
    # Set up logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / f"cnn_classifier_{args.mode}_{timestamp}.log"
    
    original_stdout = sys.stdout
    logger = Logger(log_file_path)
    sys.stdout = logger
    
    try:
        print("🎯 Background Quality Classification with CNN using Cross-Validation")
        print("=" * 80)
        print(f"📝 Log file: {log_file_path}")
        print(f"🕒 Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎛️  Mode: {args.mode.upper()}")
        print("=" * 80)
        
        # Define paths for processed data
        train_csv_path = project_root / "data/train_processed/background_masks_data_with_labels.csv"
        train_images_dir = project_root / "data/train_processed_images"
        
        val_csv_path = project_root / "data/val_processed/background_masks_data_with_labels.csv"
        val_images_dir = project_root / "data/val_processed_images"
        
        # Check if files exist
        if not train_csv_path.exists():
            print(f"❌ Training CSV not found: {train_csv_path}")
            print("Please run preprocess_labels.py first to create the processed files.")
            return
        
        if not val_csv_path.exists():
            print(f"❌ Validation CSV not found: {val_csv_path}")
            print("Please run preprocess_labels.py first to create the processed files.")
            return
        
        if not train_images_dir.exists():
            print(f"❌ Training images directory not found: {train_images_dir}")
            print("Please run image_bg_extraction.py first to create the processed images.")
            return
        
        if not val_images_dir.exists():
            print(f"❌ Validation images directory not found: {val_images_dir}")
            print("Please run image_bg_extraction.py first to create the processed images.")
            return
        
        try:
            # Load training data
            print("\n📚 Loading training data...")
            X_train, y_train = load_and_prepare_data(
                csv_path=train_csv_path,
                images_dir=train_images_dir,
                target_size=(224, 224)
            )
            
            print(f"\n📊 Training dataset: {X_train.shape[0]} samples, {X_train.shape[1]}x{X_train.shape[2]}x{X_train.shape[3]} images")
            
            # Train the CNN classifier with cross-validation
            print(f"\n🚀 Training CNN classifier with cross-validation...")
            if fast_mode:
                print("⚡ Using FAST MODE - smaller model for faster training")
            else:
                print("🔍 Using FULL MODE - larger model for better performance")
            
            trained_model, results = train_cnn_with_cv(
                X_train, y_train, 
                cv_folds=5, 
                fast_mode=fast_mode, 
                epochs=10, 
                batch_size=32
            )
            
            # Save the model
            models_dir = project_root / "models"
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / f"background_cnn_classifier_{args.mode}.h5"
            save_model(trained_model, model_path)
            
            # Now load validation data for final evaluation
            print("\n📚 Loading validation data for final evaluation...")
            X_val, y_val = load_and_prepare_data(
                csv_path=val_csv_path,
                images_dir=val_images_dir,
                target_size=(224, 224)
            )
            
            print(f"\n📊 Validation dataset: {X_val.shape[0]} samples, {X_val.shape[1]}x{X_val.shape[2]}x{X_val.shape[3]} images")
            
            # Evaluate the trained model on validation data
            print("\n🔍 Evaluating trained model on validation data...")
            y_val_pred = trained_model.predict(X_val)
            y_val_pred_classes = np.argmax(y_val_pred, axis=1)
            
            # Calculate validation metrics
            val_accuracy = accuracy_score(y_val, y_val_pred_classes)
            val_f1 = f1_score(y_val, y_val_pred_classes, average='weighted')
            
            print(f"\n📊 Validation set evaluation:")
            print(f"✅ Validation Accuracy: {val_accuracy:.4f}")
            print(f"✅ Validation F1 Score: {val_f1:.4f}")
            
            # Print detailed classification report for validation
            print("\n📋 Classification Report (Validation Set):")
            print(classification_report(y_val, y_val_pred_classes))
            
            # Print confusion matrix for validation
            print("\n📊 Confusion Matrix (Validation Set):")
            val_cm = confusion_matrix(y_val, y_val_pred_classes)
            print(val_cm)
            
            print(f"\n🎉 CNN classifier training and evaluation completed!")
            print(f"📁 Model saved to: {model_path}")
            print(f"📝 Log saved to: {log_file_path}")
            print(f"\n📊 Final Results Summary:")
            print(f"   - Mode Used: {args.mode.upper()}")
            print(f"   - Cross-validation Accuracy: {results['cv_accuracy_mean']:.4f} (+/- {results['cv_accuracy_std'] * 2:.4f})")
            print(f"   - Cross-validation F1 Score: {results['cv_f1_mean']:.4f} (+/- {results['cv_f1_std'] * 2:.4f})")
            print(f"   - Test Set Accuracy: {results['test_accuracy']:.4f}")
            print(f"   - Test Set F1 Score: {results['test_f1_score']:.4f}")
            print(f"   - Validation Set Accuracy: {val_accuracy:.4f}")
            print(f"   - Validation Set F1 Score: {val_f1:.4f}")
            print(f"   - Training Set Size: {results['X_train_shape'][0]} samples")
            print(f"   - Test Set Size: {results['X_test_shape'][0]} samples")
            print(f"   - Validation Set Size: {X_val.shape[0]} samples")
            print(f"   - Model Architecture: {'Fast Mode' if results['fast_mode'] else 'Full Mode'}")
            print(f"\n🕒 Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"📝 Log saved to: {log_file_path}")

if __name__ == "__main__":
    main()
