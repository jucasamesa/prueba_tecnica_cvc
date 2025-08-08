#!/usr/bin/env python3
"""
Test script to verify CNN setup and dependencies.
This script tests if TensorFlow/Keras is properly installed and can create a simple CNN model.
"""

import sys
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        from tensorflow import keras
        print(f"✅ Keras version: {keras.__version__}")
    except ImportError as e:
        print(f"❌ Keras import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas version: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    return True

def test_cnn_creation():
    """Test if we can create a simple CNN model."""
    print("\n🔍 Testing CNN model creation...")
    
    try:
        from tensorflow import keras
        from tensorflow.keras import layers, models
        
        # Create a simple CNN model
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ CNN model created successfully with {model.count_params():,} parameters")
        return True
        
    except Exception as e:
        print(f"❌ CNN model creation failed: {e}")
        return False

def test_data_loading():
    """Test if we can load and process sample data."""
    print("\n🔍 Testing data loading...")
    
    try:
        # Create a small sample dataset
        sample_images = np.random.rand(10, 224, 224, 3).astype(np.float32)
        sample_labels = np.random.randint(0, 2, 10).astype(np.int8)
        
        print(f"✅ Sample data created: {sample_images.shape} images, {sample_labels.shape} labels")
        print(f"✅ Data types: images={sample_images.dtype}, labels={sample_labels.dtype}")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_gpu_availability():
    """Test if GPU is available for TensorFlow."""
    print("\n🔍 Testing GPU availability...")
    
    try:
        import tensorflow as tf
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU found: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   - {gpu.name}")
        else:
            print("ℹ️  No GPU found, will use CPU")
        
        # Check TensorFlow device placement
        print(f"✅ TensorFlow device placement: {tf.config.list_physical_devices()}")
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Testing CNN setup and dependencies")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing dependencies.")
        return False
    
    # Test CNN creation
    if not test_cnn_creation():
        print("\n❌ CNN creation test failed.")
        return False
    
    # Test data loading
    if not test_data_loading():
        print("\n❌ Data loading test failed.")
        return False
    
    # Test GPU availability
    test_gpu_availability()
    
    print("\n🎉 All tests passed! CNN setup is ready.")
    print("\n📝 Next steps:")
    print("   1. Run 'python cnn_classifier.py' to train the CNN model")
    print("   2. Make sure you have processed images in data/train_processed_images/")
    print("   3. Make sure you have processed images in data/val_processed_images/")
    print("   4. Make sure you have CSV files with labels in data/train_processed/ and data/val_processed/")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
