#!/usr/bin/env python3
"""
Script to update the is_cream_background column in CSV files using CIELAB (1976) color space.
This script updates the files in-place without running the full image extraction process.

Ya que la anterior heuristica para clasificar si era color crema o no, no estaba bien sustentada bajo una norma se modifico el cálculo
en `image_bg_extraction.py` según CIELAB (1976) generando los siguientes resultados:

The script has successfully updated both CSV files with the new CIELAB-based cream detection. Here are the results:
For training data:
Original cream backgrounds: 987 (26.9%)
New cream backgrounds: 30 (0.8%)
Change: -957 images
For validation data:
Original cream backgrounds: 305 (22.1%)
New cream backgrounds: 13 (0.9%)
Change: -292 images

CIELAB (1976) is more perceptually uniform and better suited for defining colors like cream
because it separates lightness (L*) from color (a*, b*) components, making it easier to define:
- L* (Lightness): 85-95 (very light, like cream)
- a* (Red-Green): 0 to +5 (slightly warm/yellowish)
- b* (Yellow-Blue): +10 to +20 (noticeable yellow tint)

The files have been updated in-place, so you can continue using them with your existing workflow. No need to run the full image extraction process again.
"""

import pandas as pd
import numpy as np
import colour
from pathlib import Path

def is_cream(rgb, L_range=(85, 95), a_range=(0, 5), b_range=(10, 20)):
    """
    Check if an RGB color is cream based on CIELAB (1976) color space.
    
    CIELAB is more perceptually uniform and better suited for defining colors like cream
    because it separates lightness (L*) from color (a*, b*) components.

    Parameters:
        rgb (tuple): RGB values in [0, 1] or [0, 255] range.
        L_range (tuple): Min/max lightness (L*) for cream (default: 85-95).
        a_range (tuple): Min/max red-green (a*) for cream (default: 0-5, slightly warm).
        b_range (tuple): Min/max yellow-blue (b*) for cream (default: 10-20, noticeable yellow).

    Returns:
        bool: True if the color is cream, False otherwise.
    """
    # Convert RGB to [0, 1] if in [0, 255]
    if max(rgb) > 1:
        rgb = np.array(rgb) / 255.0

    # Convert RGB to CIELAB (L*a*b*)
    XYZ = colour.sRGB_to_XYZ(rgb)
    L, a, b = colour.XYZ_to_Lab(XYZ)
    
    # Check thresholds
    is_in_L = L_range[0] <= L <= L_range[1]
    is_in_a = a_range[0] <= a <= a_range[1]
    is_in_b = b_range[0] <= b <= b_range[1]
    
    return is_in_L and is_in_a and is_in_b

def update_cream_detection(file_path):
    """
    Update the is_cream_background column in a CSV file using CIE 1931 color space.
    
    Args:
        file_path (str or Path): Path to the CSV file to update
    """
    print(f"\n🔄 Processing {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} rows")
    
    # Store original cream detection results
    original_cream_count = df['is_cream_background'].sum()
    print(f"Original cream backgrounds: {original_cream_count} ({original_cream_count/len(df)*100:.1f}%)")
    
    # Update is_cream_background using CIE 1931 color space
    print("🎨 Applying CIELAB (1976) cream detection...")
    df['is_cream_background'] = df.apply(
        lambda row: is_cream((row['avg_background_r'], 
                            row['avg_background_g'], 
                            row['avg_background_b'])), 
        axis=1
    )
    
    # Show results
    new_cream_count = df['is_cream_background'].sum()
    print(f"New cream backgrounds: {new_cream_count} ({new_cream_count/len(df)*100:.1f}%)")
    print(f"Change: {new_cream_count - original_cream_count:+d} images")
    
    # Save the updated file
    df.to_csv(file_path, index=False)
    print(f"✅ Updated file saved: {file_path}")

def main():
    """Update cream detection in both training and validation files."""
    print("🎯 Updating cream background detection using CIELAB (1976) color space")
    print("=" * 70)
    
    # Define file paths
    train_path = Path("data/train_processed/background_masks_data_with_labels.csv")
    val_path = Path("data/val_processed/background_masks_data_with_labels.csv")
    
    # Update training data
    if train_path.exists():
        update_cream_detection(train_path)
    else:
        print(f"❌ Training file not found: {train_path}")
    
    # Update validation data
    if val_path.exists():
        update_cream_detection(val_path)
    else:
        print(f"❌ Validation file not found: {val_path}")
    
    print("\n✨ Processing completed!")

if __name__ == "__main__":
    main()
