#!/usr/bin/env python3
"""
SVC Classifier for Background Classification
This script implements a Support Vector Classifier (SVC) model using flattened background mask arrays from .npz files.
Uses cross-validation to ensure robust results.
Uses preprocessed background_masks_data_with_labels.csv files and filtered mask arrays.
Full hyperparameter optimization with comprehensive grid search.
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
import sys
import datetime
from io import StringIO
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class BackgroundSVCClassifier:
    """
    Support Vector Classifier for background classification using flattened mask arrays.
    Uses cross-validation for robust evaluation.
    Uses preprocessed background_masks_data_with_labels.csv files and filtered mask arrays.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the SVC classifier.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.svc = SVC(random_state=random_state, probability=True)
        self.pipeline = None
        self.is_trained = False
        self.feature_names = None
        
    def load_data(self, masks_path, mapping_path, labels_path):
        """
        Load the mask arrays and corresponding labels from preprocessed data.
        
        Args:
            masks_path (str or Path): Path to the .npz file containing mask arrays (filtered)
            mapping_path (str or Path): Path to the CSV mapping file (filtered)
            labels_path (str or Path): Path to the CSV file with preprocessed labels
            
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
        valid_indices = []
        
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
            valid_indices.append(idx)
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
    
    def preprocess_data(self, X, y, test_size=0.2):
        """
        Preprocess the data and split into train/test sets with shuffling.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target vector
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("🔄 Preprocessing data with shuffling...")
        
        # Split the data with shuffling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y, shuffle=True
        )
        
        print(f"✅ Training set: {X_train.shape[0]} samples")
        print(f"✅ Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def create_pipeline(self):
        """
        Create a pipeline with scaling and SVC.
        """
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(random_state=self.random_state, probability=True))
        ])
        print("✅ Pipeline created with StandardScaler and SVC")
    
    def train_with_cv(self, X, y, cv_folds=5, use_grid_search=True):
        """
        Train the SVC model using cross-validation for robust evaluation.
        
        Args:
            X (np.array): Training features
            y (np.array): Training labels
            cv_folds (int): Number of cross-validation folds
            use_grid_search (bool): Whether to use GridSearchCV for hyperparameter tuning
        """
        print(f"🚀 Training SVC model with {cv_folds}-fold cross-validation...")
        
        if self.pipeline is None:
            self.create_pipeline()
        
        # Create cross-validation strategy
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        if use_grid_search:
            # Define parameter grid for GridSearchCV
            param_grid = {
                'svc__C': [0.1, 1, 10, 100],
                'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'svc__kernel': ['rbf', 'linear']
            }
            
            # Perform grid search with cross-validation
            print("🔍 Performing grid search with cross-validation...")
            grid_search = GridSearchCV(
                self.pipeline, 
                param_grid, 
                cv=cv,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            # Get the best model
            self.pipeline = grid_search.best_estimator_
            print(f"✅ Best parameters: {grid_search.best_params_}")
            print(f"✅ Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            # Train with default parameters
            self.pipeline.fit(X, y)
            print("✅ Model trained with default parameters")
        
        # Perform additional cross-validation on the best model for detailed metrics
        print("📊 Performing detailed cross-validation evaluation...")
        
        # Cross-validation scores for different metrics
        cv_accuracy = cross_val_score(self.pipeline, X, y, cv=cv, scoring='accuracy')
        cv_f1 = cross_val_score(self.pipeline, X, y, cv=cv, scoring='f1_weighted')
        
        print(f"✅ Cross-validation accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
        print(f"✅ Cross-validation F1 score: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
        
        # Detailed cross-validation results
        print(f"\n📋 Cross-validation results across {cv_folds} folds:")
        print("Fold\tAccuracy\tF1 Score")
        print("-" * 30)
        for i, (acc, f1) in enumerate(zip(cv_accuracy, cv_f1)):
            print(f"{i+1}\t{acc:.4f}\t\t{f1:.4f}")
        
        self.is_trained = True
        print("🎉 Training completed!")
        
        return {
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std(),
            'cv_scores_accuracy': cv_accuracy,
            'cv_scores_f1': cv_f1
        }
    
    def train(self, X_train, y_train, use_grid_search=True):
        """
        Train the SVC model (legacy method - use train_with_cv for cross-validation).
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
            use_grid_search (bool): Whether to use GridSearchCV for hyperparameter tuning
        """
        print("⚠️  Using legacy train method. Consider using train_with_cv for cross-validation.")
        return self.train_with_cv(X_train, y_train, use_grid_search=use_grid_search)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")
        
        print("📊 Evaluating model...")
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"✅ Test Accuracy: {accuracy:.4f}")
        print(f"✅ Test F1 Score: {f1:.4f}")
        
        # Print detailed classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print confusion matrix
        print("\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def save_model(self, model_path):
        """
        Save the trained model.
        
        Args:
            model_path (str or Path): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving!")
        
        joblib.dump(self.pipeline, model_path)
        print(f"✅ Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str or Path): Path to the saved model
        """
        self.pipeline = joblib.load(model_path)
        self.is_trained = True
        print(f"✅ Model loaded from {model_path}")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (np.array): Feature matrix
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        predictions = self.pipeline.predict(X)
        probabilities = self.pipeline.predict_proba(X)
        
        return predictions, probabilities

def main():
    """
    Main function to demonstrate the SVC classifier with cross-validation.
    """
    print("🎯 Background Classification with SVC using Cross-Validation")
    print("=" * 50)
    
    # Initialize classifier
    classifier = BackgroundSVCClassifier(random_state=42)
    
    # Define paths for preprocessed and filtered data
    project_root = Path(__file__).parent.parent
    masks_train_path = project_root / "data/train_processed/background_masks_arrays_filtered.npz"
    mapping_train_path = project_root / "data/train_processed/mask_arrays_mapping_filtered.csv"
    labels_train_path = project_root / "data/train_processed/background_masks_data_with_labels.csv"
    
    masks_val_path = project_root / "data/val_processed/background_masks_arrays_filtered.npz"
    mapping_val_path = project_root / "data/val_processed/mask_arrays_mapping_filtered.csv"
    labels_val_path = project_root / "data/val_processed/background_masks_data_with_labels.csv"
    
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
        X_train_full, y_train_full = classifier.load_data(
            masks_train_path, mapping_train_path, labels_train_path
        )
        
        # Load validation data
        print("\n📚 Loading validation data...")
        X_val_full, y_val_full = classifier.load_data(
            masks_val_path, mapping_val_path, labels_val_path
        )
        
        # Combine data for cross-validation
        X_combined = np.vstack([X_train_full, X_val_full])
        y_combined = np.hstack([y_train_full, y_val_full])
        
        print(f"\n📊 Combined dataset: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
        
        # Train the model with cross-validation
        cv_results = classifier.train_with_cv(X_combined, y_combined, cv_folds=5)
        
        # Perform final evaluation on a held-out test set with shuffling
        X_train, X_test, y_train, y_test = classifier.preprocess_data(X_combined, y_combined, test_size=0.2)
        
        # Fit the best model on the training data
        classifier.pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        test_results = classifier.evaluate(X_test, y_test)
        
        # Save the model
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "background_svc_classifier_cv.pkl"
        classifier.save_model(model_path)
        
        print(f"\n🎉 SVC classifier training and evaluation completed!")
        print(f"📁 Model saved to: {model_path}")
        print(f"\n📊 Final Results Summary:")
        print(f"   - Cross-validation Accuracy: {cv_results['cv_accuracy_mean']:.4f} (+/- {cv_results['cv_accuracy_std'] * 2:.4f})")
        print(f"   - Cross-validation F1 Score: {cv_results['cv_f1_mean']:.4f} (+/- {cv_results['cv_f1_std'] * 2:.4f})")
        print(f"   - Test Set Accuracy: {test_results['accuracy']:.4f}")
        print(f"   - Test Set F1 Score: {test_results['f1_score']:.4f}")
        print(f"   - Training Set Size: {X_train.shape[0]} samples")
        print(f"   - Test Set Size: {X_test.shape[0]} samples")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
