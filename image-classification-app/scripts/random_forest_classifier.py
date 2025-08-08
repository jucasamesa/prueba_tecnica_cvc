#!/usr/bin/env python3
"""
Random Forest Classifier for Background Classification using ImageAnalyzer Features
This script implements a Random Forest model using specific features from ImageAnalyzer:
- brightness, contrast, entropy, noise_level, saturation, value
- dominant_color_r, dominant_color_g, dominant_color_b, dominant_color_h, dominant_color_s, dominant_color_v

Uses cross-validation to ensure robust results.
Applies proper scaling to ensure no variable weighs more than another.
Generates feature importance plot to determine which characteristics are most linked to classification.

Usage:
    python scripts/random_forest_classifier.py --mode fast    # Fast mode (1 combination × 5 folds = 5 total fits)
    python scripts/random_forest_classifier.py --mode full    # Full mode (108 combinations × 5 folds = 540 total fits)
    python scripts/random_forest_classifier.py                # Default: fast mode
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
import sys
import datetime
import argparse
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("⚠️  Seaborn not available, using matplotlib for plotting")
from io import StringIO
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class Logger:
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

def load_and_prepare_data(csv_path):
    """
    Load the CSV data and prepare specific ImageAnalyzer features for training.
    
    Args:
        csv_path (str or Path): Path to the CSV file with ImageAnalyzer features
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    print("🔍 Loading data from CSV...")
    
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
        print(f"❌ Missing required features: {missing_features}")
        print("Please run apply_image_analyzer.py first to add ImageAnalyzer features to the CSV.")
        return None, None
    
    # Check if target column exists
    if 'correct_background?' not in df.columns:
        print("❌ Missing target column 'correct_background?'")
        return None, None
    
    # Select only the required features
    X = df[required_features].copy()
    y = df['correct_background?'].copy()
    
    # Handle missing values
    missing_count = X.isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️  Found {missing_count} missing values. Filling with median...")
        X = X.fillna(X.median())
    
    # Convert to numpy arrays
    X = X.values.astype(np.float32)
    y = y.values.astype(np.int8)
    
    print(f"✅ Prepared {len(X)} samples with {X.shape[1]} features")
    print(f"✅ Target distribution: {np.bincount(y)}")
    print(f"✅ Memory usage: X={X.nbytes / 1024**2:.2f} MB, y={y.nbytes / 1024**2:.2f} MB")
    
    # Print feature statistics
    print(f"\n📊 Feature Statistics:")
    feature_stats = pd.DataFrame(X, columns=required_features).describe()
    print(feature_stats)
    
    return X, y, required_features

def train_random_forest_classifier_with_cv(X, y, feature_names, cv_folds=5, random_state=42, fast_mode=True):
    """
    Train a Random Forest classifier using cross-validation for robust evaluation.
    
    Args:
        X (np.array): Feature matrix (ImageAnalyzer features)
        y (np.array): Target vector
        feature_names (list): List of feature names
        cv_folds (int): Number of cross-validation folds
        random_state (int): Random state for reproducibility
        fast_mode (bool): If True, use minimal parameter grid for speed
        
    Returns:
        tuple: (trained_pipeline, evaluation_results)
    """
    print(f"🚀 Training Random Forest classifier with {cv_folds}-fold cross-validation...")
    
    # Create pipeline with scaling and Random Forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=random_state, n_jobs=-1))
    ])
    
    # Define parameter grid based on mode
    if fast_mode:
        print("⚡ Using FAST MODE - minimal parameter grid for speed")
        param_grid = {
            'rf__n_estimators': [100],  # Single value
            'rf__max_depth': [10],      # Single value
            'rf__min_samples_split': [2]  # Single value
        }
        # This results in only 1 combination × 5 folds = 5 total fits
    else:
        print("🔍 Using FULL MODE - comprehensive parameter grid")
        param_grid = {
            'rf__n_estimators': [50, 100, 200],
            'rf__max_depth': [5, 10, 15, None],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4]
        }
        # This results in 3 × 4 × 3 × 3 = 108 combinations × 5 folds = 540 total fits
    
    # Create cross-validation strategy
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Perform grid search with cross-validation
    print(f"🔍 Performing grid search with cross-validation ({len(param_grid)} combinations × {cv_folds} folds = {len(param_grid) * cv_folds} total fits)...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv,
        scoring='f1_weighted',
        n_jobs=1,  # Single job to avoid memory issues
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
    cv_accuracy = cross_val_score(best_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    cv_f1 = cross_val_score(best_pipeline, X, y, cv=cv, scoring='f1_weighted', n_jobs=1)
    
    print(f"✅ Cross-validation accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
    print(f"✅ Cross-validation F1 score: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
    
    # Detailed cross-validation results
    print(f"\n📋 Cross-validation results across {cv_folds} folds:")
    print("Fold\tAccuracy\tF1 Score")
    print("-" * 30)
    for i, (acc, f1) in enumerate(zip(cv_accuracy, cv_f1)):
        print(f"{i+1}\t{acc:.4f}\t\t{f1:.4f}")
    
    # Perform final evaluation on a held-out test set (20% of data)
    print("\n🔄 Creating train/test split...")
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
    
    print(f"\n📊 Test set evaluation (20% held-out):")
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    print(f"✅ Test F1 Score: {test_f1:.4f}")
    
    # Print detailed classification report
    print("\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\n📊 Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance analysis (Random Forest specific)
    rf_model = best_pipeline.named_steps['rf']
    feature_importance = rf_model.feature_importances_
    print(f"\n🌳 Feature Importance Analysis:")
    print(f"✅ Number of features: {len(feature_importance)}")
    print(f"✅ Mean feature importance: {feature_importance.mean():.6f}")
    print(f"✅ Max feature importance: {feature_importance.max():.6f}")
    print(f"✅ Min feature importance: {feature_importance.min():.6f}")
    
    # Create feature importance DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 Feature Importance Ranking:")
    for i, (_, row) in enumerate(feature_importance_df.iterrows()):
        print(f"   {i+1:2d}. {row['feature']:20s}: {row['importance']:.6f}")
    
    # Generate feature importance plot
    print("\n📈 Generating feature importance plot...")
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    if SEABORN_AVAILABLE:
        sns.barplot(data=feature_importance_df, x='importance', y='feature', palette='viridis')
    else:
        # Use matplotlib if seaborn is not available
        y_pos = range(len(feature_importance_df))
        plt.barh(y_pos, feature_importance_df['importance'], color='skyblue', edgecolor='navy')
        plt.yticks(y_pos, feature_importance_df['feature'])
    
    plt.title('Random Forest Feature Importance', fontsize=16, fontweight='bold')
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (_, row) in enumerate(feature_importance_df.iterrows()):
        plt.text(row['importance'] + 0.001, i, f'{row["importance"]:.4f}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    project_root = Path(__file__).parent.parent
    plots_dir = project_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / "random_forest_feature_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Feature importance plot saved to: {plot_path}")
    
    # Show the plot
    plt.show()
    
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
        'feature_importance': feature_importance,
        'feature_importance_df': feature_importance_df,
        'feature_names': feature_names,
        'fast_mode': fast_mode,
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
    Main function to demonstrate the Random Forest classifier with ImageAnalyzer features.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Random Forest Classifier for Background Classification using ImageAnalyzer Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/random_forest_classifier.py --mode fast    # Fast mode (1 combination × 5 folds = 5 total fits)
    python scripts/random_forest_classifier.py --mode full    # Full mode (108 combinations × 5 folds = 540 total fits)
    python scripts/random_forest_classifier.py                # Default: fast mode
        """
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['fast', 'full'], 
        default='fast',
        help='Training mode: fast (1 combination × 5 folds = 5 total fits) or full (108 combinations × 5 folds = 540 total fits)'
    )
    
    args = parser.parse_args()
    
    # Convert mode to boolean
    fast_mode = args.mode == 'fast'
    
    # Set up logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / f"random_forest_classifier_features_{args.mode}_{timestamp}.log"
    
    original_stdout = sys.stdout
    logger = Logger(log_file_path)
    sys.stdout = logger
    
    try:
        print("🎯 Background Classification with Random Forest using ImageAnalyzer Features")
        print("=" * 80)
        print(f"📝 Log file: {log_file_path}")
        print(f"🕒 Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎛️  Mode: {args.mode.upper()}")
        print("=" * 80)
        
        # Define paths for data with ImageAnalyzer features
        train_csv_path = project_root / "data/train_processed/background_masks_data_with_labels.csv"
        val_csv_path = project_root / "data/val_processed/background_masks_data_with_labels.csv"
        
        # Check if files exist
        if not train_csv_path.exists():
            print(f"❌ Training CSV not found at {train_csv_path}")
            print("Please run apply_image_analyzer.py first to add ImageAnalyzer features to the CSV.")
            return
        
        if not val_csv_path.exists():
            print(f"❌ Validation CSV not found at {val_csv_path}")
            print("Please run apply_image_analyzer.py first to add ImageAnalyzer features to the CSV.")
            return
        
        try:
            # Load training data only for model training and cross-validation
            print("\n📚 Loading training data with ImageAnalyzer features...")
            X_train, y_train, feature_names = load_and_prepare_data(train_csv_path)
            
            if X_train is None or y_train is None:
                print("❌ Failed to load training data")
                return
            
            print(f"\n📊 Training dataset: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"📋 Features used: {feature_names}")
            
            # Train the Random Forest classifier with cross-validation using only training data
            print(f"\n🚀 Training Random Forest classifier with cross-validation (training data only)...")
            if fast_mode:
                print("⚡ Using FAST MODE - minimal parameter grid for speed (1 combination × 5 folds = 5 total fits)")
            else:
                print("🔍 Using FULL MODE - comprehensive parameter grid (108 combinations × 5 folds = 540 total fits)")
            
            trained_pipeline, results = train_random_forest_classifier_with_cv(
                X_train, y_train, feature_names, cv_folds=5, fast_mode=fast_mode
            )
            
            # Save the model
            models_dir = project_root / "models"
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / f"background_random_forest_classifier_features_{args.mode}.pkl"
            save_model(trained_pipeline, model_path)
            
            # Now load validation data for final evaluation
            print("\n📚 Loading validation data for final evaluation...")
            X_val, y_val, _ = load_and_prepare_data(val_csv_path)
            
            if X_val is None or y_val is None:
                print("❌ Failed to load validation data")
                return
            
            print(f"\n📊 Validation dataset: {X_val.shape[0]} samples, {X_val.shape[1]} features")
            
            # Evaluate the trained model on validation data
            print("\n🔍 Evaluating trained model on validation data...")
            y_val_pred = trained_pipeline.predict(X_val)
            y_val_pred_proba = trained_pipeline.predict_proba(X_val)
            
            # Calculate validation metrics
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted')
            
            print(f"\n📊 Validation set evaluation:")
            print(f"✅ Validation Accuracy: {val_accuracy:.4f}")
            print(f"✅ Validation F1 Score: {val_f1:.4f}")
            
            # Print detailed classification report for validation
            print("\n📋 Classification Report (Validation Set):")
            print(classification_report(y_val, y_val_pred))
            
            # Print confusion matrix for validation
            print("\n📊 Confusion Matrix (Validation Set):")
            val_cm = confusion_matrix(y_val, y_val_pred)
            print(val_cm)
            
            print(f"\n🎉 Random Forest classifier training and evaluation completed!")
            print(f"📁 Model saved to: {model_path}")
            print(f"📝 Log saved to: {log_file_path}")
            print(f"📈 Feature importance plot saved to: {project_root}/plots/random_forest_feature_importance.png")
            print(f"\n📊 Final Results Summary:")
            print(f"   - Mode Used: {args.mode.upper()}")
            print(f"   - Cross-validation Accuracy: {results['cv_accuracy_mean']:.4f} (+/- {results['cv_accuracy_std'] * 2:.4f})")
            print(f"   - Cross-validation F1 Score: {results['cv_f1_mean']:.4f} (+/- {results['cv_f1_std'] * 2:.4f})")
            print(f"   - Test Set Accuracy: {results['test_accuracy']:.4f}")
            print(f"   - Test Set F1 Score: {results['test_f1_score']:.4f}")
            print(f"   - Validation Set Accuracy: {val_accuracy:.4f}")
            print(f"   - Validation Set F1 Score: {val_f1:.4f}")
            print(f"   - Best Parameters: {results['best_params']}")
            print(f"   - Training Set Size: {results['X_train_shape'][0]} samples")
            print(f"   - Test Set Size: {results['X_test_shape'][0]} samples")
            print(f"   - Validation Set Size: {X_val.shape[0]} samples")
            print(f"   - Features Used: {len(feature_names)}")
            print(f"\n🏆 Top 5 Most Important Features:")
            top_features = results['feature_importance_df'].head(5)
            for i, (_, row) in enumerate(top_features.iterrows()):
                print(f"   {i+1}. {row['feature']:20s}: {row['importance']:.4f}")
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

