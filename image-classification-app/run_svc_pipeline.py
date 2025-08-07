#!/usr/bin/env python3
"""
Complete SVC Pipeline: Preprocessing + Training
This script runs the complete pipeline from preprocessing the background mask data
to training the SVC classifier with cross-validation.
"""

import subprocess
import sys
from pathlib import Path

def run_preprocessing():
    """
    Run the preprocessing step to create background_masks_data_with_labels.csv files.
    """
    print("🔧 Step 1: Running preprocessing...")
    print("=" * 50)
    
    try:
        # Run the preprocessing script
        result = subprocess.run([sys.executable, "preprocess_labels.py"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Preprocessing failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ preprocess_labels.py not found. Please make sure the file exists.")
        return False

def run_svc_training():
    """
    Run the SVC training with cross-validation.
    """
    print("\n🚀 Step 2: Running SVC training...")
    print("=" * 50)
    
    try:
        # Run the SVC training script
        result = subprocess.run([sys.executable, "simple_svc_classifier.py"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ SVC training failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ simple_svc_classifier.py not found. Please make sure the file exists.")
        return False

def check_prerequisites():
    """
    Check if all required files exist before running the pipeline.
    """
    print("🔍 Checking prerequisites...")
    
    required_files = [
        "data/train_processed/background_masks_data.csv",
        "data/val_processed/background_masks_data.csv",
        "data/train_images_dataset.csv",
        "data/train_processed/background_masks_arrays.npz",
        "data/val_processed/background_masks_arrays.npz",
        "data/train_processed/mask_arrays_mapping.csv",
        "data/val_processed/mask_arrays_mapping.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files found!")
    return True

def check_output_files():
    """
    Check if the preprocessing step created the expected output files.
    """
    print("🔍 Checking output files...")
    
    expected_output_files = [
        "data/train_processed/background_masks_data_with_labels.csv",
        "data/val_processed/background_masks_data_with_labels.csv",
        "data/train_processed/background_masks_arrays_filtered.npz",
        "data/val_processed/background_masks_arrays_filtered.npz",
        "data/train_processed/mask_arrays_mapping_filtered.csv",
        "data/val_processed/mask_arrays_mapping_filtered.csv"
    ]
    
    missing_output_files = []
    for file_path in expected_output_files:
        if not Path(file_path).exists():
            missing_output_files.append(file_path)
    
    if missing_output_files:
        print("❌ Missing output files after preprocessing:")
        for file_path in missing_output_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All expected output files found!")
    return True

def main():
    """
    Main function to run the complete SVC pipeline.
    """
    print("🎯 Complete SVC Pipeline: Preprocessing + Training")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please ensure all required files exist.")
        return
    
    # Step 1: Run preprocessing
    if not run_preprocessing():
        print("\n❌ Pipeline failed at preprocessing step.")
        return
    
    # Check if preprocessing created the expected output files
    if not check_output_files():
        print("\n❌ Preprocessing did not create all expected output files.")
        return
    
    # Step 2: Run SVC training
    if not run_svc_training():
        print("\n❌ Pipeline failed at SVC training step.")
        return
    
    print("\n🎉 Pipeline completed successfully!")
    print("=" * 60)
    print("📁 Output files created:")
    print("   - data/train_processed/background_masks_data_with_labels.csv")
    print("   - data/val_processed/background_masks_data_with_labels.csv")
    print("   - data/train_processed/background_masks_arrays_filtered.npz")
    print("   - data/val_processed/background_masks_arrays_filtered.npz")
    print("   - data/train_processed/mask_arrays_mapping_filtered.csv")
    print("   - data/val_processed/mask_arrays_mapping_filtered.csv")
    print("   - models/background_svc_classifier_cv.pkl")
    print("\n📊 Results:")
    print("   - Preprocessed background mask data with labels")
    print("   - Filtered mask arrays and mapping files")
    print("   - Trained SVC classifier with cross-validation")
    print("   - Model saved for future use")
    print("\n📝 Key improvements:")
    print("   - Only images with valid labels are included")
    print("   - Train/test split uses shuffling for better randomization")
    print("   - Cross-validation ensures robust results")
    print("   - Filtered files reduce memory usage and processing time")
    print("\n📝 Next steps:")
    print("   - Use the trained model for predictions")
    print("   - Analyze the cross-validation results")
    print("   - Fine-tune hyperparameters if needed")

if __name__ == "__main__":
    main()
