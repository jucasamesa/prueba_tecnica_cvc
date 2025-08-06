#!/usr/bin/env python3
"""
Background Color Analysis Helper Script
This script demonstrates how to load and analyze the background mask arrays
to check for white or cream colored backgrounds.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def load_mask_arrays():
    """Load the background mask arrays from the numpy file."""
    masks_path = Path("data/processed/background_masks_arrays.npz")
    mapping_path = Path("data/processed/mask_arrays_mapping.csv")
    
    if not masks_path.exists() or not mapping_path.exists():
        print("❌ Mask arrays not found. Run image_bg_extraction.py first.")
        return None, None
    
    # Load the compressed numpy arrays
    masks_data = np.load(masks_path)
    
    # Load the mapping
    mapping_df = pd.read_csv(mapping_path)
    
    print(f"✅ Loaded {len(masks_data.files)} mask arrays")
    return masks_data, mapping_df

def analyze_background_colors():
    """Analyze the background color data from the CSV."""
    csv_path = Path("data/processed/background_masks_data.csv")
    
    if not csv_path.exists():
        print("❌ Background mask data CSV not found. Run image_bg_extraction.py first.")
        return
    
    df = pd.read_csv(csv_path)
    
    print("\n🎨 Background Color Analysis:")
    print(f"Total images processed: {len(df)}")
    
    # Count white and cream backgrounds
    white_count = df['is_white_background'].sum()
    cream_count = df['is_cream_background'].sum()
    other_count = len(df) - white_count - cream_count
    
    print(f"White backgrounds: {white_count} ({white_count/len(df)*100:.1f}%)")
    print(f"Cream backgrounds: {cream_count} ({cream_count/len(df)*100:.1f}%)")
    print(f"Other backgrounds: {other_count} ({other_count/len(df)*100:.1f}%)")
    
    # Show average RGB values
    print(f"\nAverage background colors:")
    print(f"Average R: {df['avg_background_r'].mean():.1f}")
    print(f"Average G: {df['avg_background_g'].mean():.1f}")
    print(f"Average B: {df['avg_background_b'].mean():.1f}")
    
    # Show some examples
    print(f"\n📋 Sample white backgrounds:")
    white_samples = df[df['is_white_background']].head(5)
    for _, row in white_samples.iterrows():
        print(f"  {row['original_image_name']}: RGB({row['avg_background_r']:.0f}, {row['avg_background_g']:.0f}, {row['avg_background_b']:.0f})")
    
    print(f"\n📋 Sample cream backgrounds:")
    cream_samples = df[df['is_cream_background']].head(5)
    for _, row in cream_samples.iterrows():
        print(f"  {row['original_image_name']}: RGB({row['avg_background_r']:.0f}, {row['avg_background_g']:.0f}, {row['avg_background_b']:.0f})")
    
    return df

def get_mask_for_image(image_name, masks_data, mapping_df):
    """Get the background mask array for a specific image."""
    # Find the numpy array key for this image
    mapping_row = mapping_df[mapping_df['original_image_name'] == image_name]
    if mapping_row.empty:
        print(f"❌ Image {image_name} not found in mapping.")
        return None
    
    array_key = mapping_row.iloc[0]['numpy_array_key']
    
    if array_key not in masks_data:
        print(f"❌ Array key {array_key} not found in mask data.")
        return None
    
    return masks_data[array_key]

def main():
    """Main analysis function."""
    print("🔍 Background Color Analysis Tool")
    print("=" * 50)
    
    # Analyze color statistics from CSV
    df = analyze_background_colors()
    
    # Load mask arrays
    masks_data, mapping_df = load_mask_arrays()
    
    if masks_data is not None and mapping_df is not None:
        print(f"\n📊 Mask Array Information:")
        print(f"Available mask arrays: {len(masks_data.files)}")
        
        # Show example of how to access individual masks
        if len(df) > 0:
            example_image = df.iloc[0]['original_image_name']
            example_mask = get_mask_for_image(example_image, masks_data, mapping_df)
            if example_mask is not None:
                print(f"\n📋 Example mask for {example_image}:")
                print(f"  Shape: {example_mask.shape}")
                print(f"  Background pixels (>127): {np.sum(example_mask > 127)}")
                print(f"  Total pixels: {example_mask.size}")
    
    print("\n" + "=" * 50)
    print("✅ Analysis completed!")
    print("\nTo access individual mask arrays programmatically:")
    print("  masks_data, mapping_df = load_mask_arrays()")
    print("  mask = get_mask_for_image('your_image.jpg', masks_data, mapping_df)")

if __name__ == "__main__":
    main()