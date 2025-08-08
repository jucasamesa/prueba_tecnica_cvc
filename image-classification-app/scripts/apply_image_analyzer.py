#!/usr/bin/env python3
"""
Script para Aplicar ImageAnalyzer a Imágenes Procesadas / Se puede usar para las descargadas al cambiar los path

Objetivo:
Este script analiza imágenes procesadas (redimensionadas y con fondo eliminado mediante rembg) y guarda los resultados en archivos CSV.
Script para Aplicar ImageAnalyzer a Imágenes Procesadas
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from eda_modules.image_analyzer import ImageAnalyzer

def apply_image_analyzer_to_dataset(csv_path: Path, images_dir: Path, output_path: Path = None):
    """
    Apply ImageAnalyzer to a dataset of processed images.
    
    Args:
        csv_path (Path): Path to the CSV file with image data
        images_dir (Path): Directory containing the processed images
        output_path (Path): Path to save the updated CSV (if None, overwrites original)
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"🎯 Processing dataset: {csv_path.name}")
    print(f"📁 Images directory: {images_dir}")
    print("=" * 60)
    
    # Load the dataset
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows from {csv_path.name}")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return False
    
    # Check if processed_image_name column exists
    if 'processed_image_name' not in df.columns:
        print("❌ 'processed_image_name' column not found in CSV")
        return False
    
    # Initialize the ImageAnalyzer
    analyzer = ImageAnalyzer()
    
    # Create list to store analysis results
    analysis_results = []
    
    # Process each image
    print(f"🔄 Analyzing {len(df)} images...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing images"):
        try:
            # Get the processed image path
            processed_image_name = row['processed_image_name']
            image_path = images_dir / processed_image_name
            
            # Check if image exists
            if not image_path.exists():
                print(f"⚠️  Warning: Image not found: {image_path}")
                analysis_results.append({})
                continue
            
            # Analyze the image
            analysis = analyzer.analyze_single_image(image_path)
            
            # Add the analysis results
            analysis_results.append(analysis)
            
        except Exception as e:
            print(f"⚠️  Warning: Error analyzing image {row.get('processed_image_name', 'unknown')}: {e}")
            analysis_results.append({})
    
    # Convert analysis results to DataFrame
    print("🔄 Converting analysis results to DataFrame...")
    analysis_df = pd.DataFrame(analysis_results)
    
    # Remove duplicate columns that might already exist
    existing_columns = set(df.columns)
    new_columns = [col for col in analysis_df.columns if col not in existing_columns]
    
    if new_columns:
        print(f"✅ Adding {len(new_columns)} new columns: {new_columns}")
        
        # Add new columns to the original DataFrame
        for col in new_columns:
            df[col] = analysis_df[col]
    else:
        print("ℹ️  No new columns to add")
    
    # Save the updated DataFrame
    output_path = output_path or csv_path
    try:
        df.to_csv(output_path, index=False)
        print(f"✅ Updated dataset saved to: {output_path}")
        print(f"📊 Final dataset shape: {df.shape}")
        return True
    except Exception as e:
        print(f"❌ Error saving updated dataset: {e}")
        return False

def main():
    """Main function to process both training and validation datasets."""
    print("🚀 Starting ImageAnalyzer application to processed datasets")
    print("=" * 70)
    
    # Define paths
    project_root = Path(__file__).parent.parent
    base_dir = project_root / "data"
    
    # Training dataset
    train_csv_path = base_dir / "train_processed" / "background_masks_data_with_labels.csv"
    train_images_dir = base_dir / "train_processed_images"
    
    # Validation dataset
    val_csv_path = base_dir / "val_processed" / "background_masks_data_with_labels.csv"
    val_images_dir = base_dir / "val_processed_images"
    
    # Check if files exist
    if not train_csv_path.exists():
        print(f"❌ Training CSV not found: {train_csv_path}")
        return False
    
    if not val_csv_path.exists():
        print(f"❌ Validation CSV not found: {val_csv_path}")
        return False
    
    if not train_images_dir.exists():
        print(f"❌ Training images directory not found: {train_images_dir}")
        return False
    
    if not val_images_dir.exists():
        print(f"❌ Validation images directory not found: {val_images_dir}")
        return False
    
    # Process training dataset
    print("\n📚 Processing TRAINING dataset...")
    train_success = apply_image_analyzer_to_dataset(
        csv_path=train_csv_path,
        images_dir=train_images_dir
    )
    
    if not train_success:
        print("❌ Failed to process training dataset")
        return False
    
    # Process validation dataset
    print("\n📚 Processing VALIDATION dataset...")
    val_success = apply_image_analyzer_to_dataset(
        csv_path=val_csv_path,
        images_dir=val_images_dir
    )
    
    if not val_success:
        print("❌ Failed to process validation dataset")
        return False
    
    print("\n🎉 SUCCESS! Both datasets have been processed and updated.")
    print("=" * 70)
    print("📊 Summary:")
    print(f"   ✅ Training dataset: {train_csv_path}")
    print(f"   ✅ Validation dataset: {val_csv_path}")
    print("\n📝 The CSV files now contain additional image analysis features")
    print("   including quality metrics, background analysis, and texture features.")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
