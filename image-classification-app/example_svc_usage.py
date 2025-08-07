#!/usr/bin/env python3
"""
Example usage of SVC classifier with flattened mask arrays
This script demonstrates how to use the SVC classifier with the specific data structure.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib

def main():
    """
    Example of how to use SVC with flattened mask arrays.
    """
    print("🎯 Example: SVC Classifier with Flattened Mask Arrays")
    print("=" * 60)
    
    # Define paths (adjust these to match your actual file structure)
    masks_train_path = Path("data/train_processed/background_masks_arrays.npz")
    masks_val_path = Path("data/val_processed/background_masks_arrays.npz")
    
    # Check if files exist
    if not masks_train_path.exists():
        print(f"❌ Training masks not found at {masks_train_path}")
        print("Please make sure the .npz files exist in the correct location.")
        return
    
    if not masks_val_path.exists():
        print(f"❌ Validation masks not found at {masks_val_path}")
        print("Please make sure the .npz files exist in the correct location.")
        return
    
    try:
        # Load the compressed numpy arrays
        print("📁 Loading mask arrays...")
        masks_train_data = np.load(masks_train_path)
        masks_val_data = np.load(masks_val_path)
        
        print(f"✅ Loaded {len(masks_train_data.files)} training mask arrays")
        print(f"✅ Loaded {len(masks_val_data.files)} validation mask arrays")
        
        # Example: Process a few mask arrays to demonstrate the approach
        print("\n🔄 Processing mask arrays with mask_array.flatten()...")
        
        # Get the first few mask arrays as examples
        train_keys = list(masks_train_data.files)[:5]  # First 5 for demonstration
        val_keys = list(masks_val_data.files)[:5]      # First 5 for demonstration
        
        X_train_list = []
        X_val_list = []
        
        # Process training masks
        for key in train_keys:
            mask_array = masks_train_data[key]
            # This is the key step: flatten the mask array using the correct method
            flattened_mask = mask_array.flatten()
            X_train_list.append(flattened_mask)
            print(f"  Training: {key} -> shape: {mask_array.shape} -> flattened: {flattened_mask.shape}")
        
        # Process validation masks
        for key in val_keys:
            mask_array = masks_val_data[key]
            # This is the key step: flatten the mask array using the correct method
            flattened_mask = mask_array.flatten()
            X_val_list.append(flattened_mask)
            print(f"  Validation: {key} -> shape: {mask_array.shape} -> flattened: {flattened_mask.shape}")
        
        # Convert to numpy arrays
        X_train = np.array(X_train_list)
        X_val = np.array(X_val_list)
        
        print(f"\n📊 Feature matrix shapes:")
        print(f"  Training: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        
        # For demonstration, create dummy labels (you would use real labels from your CSV)
        # In practice, you would load these from your labels CSV file
        y_train = np.array([0, 1, 0, 1, 0])  # Dummy labels
        y_val = np.array([1, 0, 1, 0, 1])    # Dummy labels
        
        print(f"  Training labels: {y_train}")
        print(f"  Validation labels: {y_val}")
        
        # Create and train a simple SVC classifier
        print("\n🚀 Training SVC classifier...")
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create and train SVC
        svc = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svc.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = svc.predict(X_train_scaled)
        y_val_pred = svc.predict(X_val_scaled)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f"\n📊 Results (with dummy labels):")
        print(f"  Training Accuracy: {train_accuracy:.4f}")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Training F1 Score: {train_f1:.4f}")
        print(f"  Validation F1 Score: {val_f1:.4f}")
        
        # Show how to use the model for new predictions
        print(f"\n🔮 Example prediction for new data:")
        if len(X_val_list) > 0:
            # Use the first validation sample as an example
            new_sample = X_val_scaled[0:1]  # Reshape for single prediction
            prediction = svc.predict(new_sample)
            probability = svc.predict_proba(new_sample)
            print(f"  New sample prediction: {prediction[0]}")
            print(f"  Prediction probabilities: {probability[0]}")
        
        print(f"\n✅ Example completed successfully!")
        print(f"\n📝 Key points:")
        print(f"  1. Mask arrays are loaded from .npz files")
        print(f"  2. Each mask array is flattened using np.flatten()")
        print(f"  3. Flattened arrays become feature vectors for SVC")
        print(f"  4. Features are scaled before training")
        print(f"  5. SVC classifier is trained on the flattened features")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
