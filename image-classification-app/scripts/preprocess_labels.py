#!/usr/bin/env python3
"""
Preprocess background mask data by merging with labels and filtering out images with "?" labels.
This script creates background_masks_data_with_labels.csv files for both train and validation datasets.
Also filters mask arrays and mapping files to only include images with valid labels.
"""

import pandas as pd
from pathlib import Path
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def preprocess_background_masks_data(data_type='train'):
    """
    Preprocess background mask data by merging with labels and filtering out "?" labels.
    Also filters mask arrays and mapping files to only include images with valid labels.
    
    Args:
        data_type (str): Either 'train' or 'val' to specify which dataset to process
    """
    print(f"🔍 Preprocessing {data_type} background mask data...")
    
    # Define paths based on data type
    project_root = Path(__file__).parent.parent
    if data_type == 'train':
        background_masks_path = project_root / "data/train_processed/background_masks_data.csv"
        train_images_path = project_root / "data/train_images_dataset.csv"
        output_path = project_root / "data/train_processed/background_masks_data_with_labels.csv"
        masks_arrays_path = project_root / "data/train_processed/background_masks_arrays.npz"
        mapping_path = project_root / "data/train_processed/mask_arrays_mapping.csv"
        filtered_masks_arrays_path = project_root / "data/train_processed/background_masks_arrays_filtered.npz"
        filtered_mapping_path = project_root / "data/train_processed/mask_arrays_mapping_filtered.csv"
    elif data_type == 'val':
        background_masks_path = project_root / "data/val_processed/background_masks_data.csv"
        train_images_path = project_root / "data/productive_images_dataset.csv"
        output_path = project_root / "data/val_processed/background_masks_data_with_labels.csv"
        masks_arrays_path = project_root / "data/val_processed/background_masks_arrays.npz"
        mapping_path = project_root / "data/val_processed/mask_arrays_mapping.csv"
        filtered_masks_arrays_path = project_root / "data/val_processed/background_masks_arrays_filtered.npz"
        filtered_mapping_path = project_root / "data/val_processed/mask_arrays_mapping_filtered.csv"
    else:
        raise ValueError("data_type must be either 'train' or 'val'")
    
    # Check if files exist
    if not background_masks_path.exists():
        print(f"❌ Background masks data not found at {background_masks_path}")
        return False
    
    if not train_images_path.exists():
        print(f"❌ Train images dataset not found at {train_images_path}")
        return False
    
    if not masks_arrays_path.exists():
        print(f"❌ Mask arrays not found at {masks_arrays_path}")
        return False
    
    if not mapping_path.exists():
        print(f"❌ Mapping file not found at {mapping_path}")
        return False
    
    try:
        # Load background masks data
        print(f"📁 Loading background masks data from {background_masks_path}")
        background_masks_df = pd.read_csv(background_masks_path)
        print(f"✅ Loaded {len(background_masks_df)} background mask records")
        
        # Load train images dataset (contains labels)
        print(f"📁 Loading train images dataset from {train_images_path}")
        train_images_df = pd.read_csv(train_images_path)
        print(f"✅ Loaded {len(train_images_df)} train image records")
        
        # Check if 'correct_background?' column exists
        if 'correct_background?' not in train_images_df.columns:
            print("❌ 'correct_background?' column not found in train_images_dataset.csv")
            return False
        
        # Rename columns for merging
        train_images_renamed = train_images_df.rename(columns={'filename': 'original_image_name'})
        
        # Merge the datasets
        print("🔄 Merging background masks data with labels...")
        merged_df = background_masks_df.merge(
            train_images_renamed[['original_image_name', 'correct_background?']], 
            how='left', 
            on='original_image_name'
        )
        
        print(f"✅ Merged dataset has {len(merged_df)} records")
        
        # Check for missing labels
        missing_labels = merged_df['correct_background?'].isna().sum()
        if missing_labels > 0:
            print(f"⚠️  Warning: {missing_labels} records have missing labels")
        
        # Filter out records with "?" labels
        print("🔍 Filtering out records with '?' labels...")
        filtered_df = merged_df[merged_df['correct_background?'] != "?"].reset_index(drop=True)
        
        print(f"✅ Filtered dataset has {len(filtered_df)} records (removed {len(merged_df) - len(filtered_df)} records with '?' labels)")
        
        # Check the distribution of labels
        label_distribution = filtered_df['correct_background?'].value_counts()
        print(f"\n📊 Label distribution:")
        for label, count in label_distribution.items():
            percentage = (count / len(filtered_df)) * 100
            print(f"  Label {label}: {count} records ({percentage:.1f}%)")
        
        # Save the processed data
        print(f"💾 Saving processed data to {output_path}")
        filtered_df.to_csv(output_path, index=False)
        print(f"✅ Successfully saved {len(filtered_df)} records to {output_path}")
        
        # Now filter the mask arrays and mapping to only include images with valid labels
        print("\n🔍 Filtering mask arrays and mapping to only include images with valid labels...")
        
        # Load the original mapping
        mapping_df = pd.read_csv(mapping_path)
        print(f"📁 Loaded mapping with {len(mapping_df)} entries")
        
        # Get the list of valid image names (those with valid labels)
        valid_image_names = set(filtered_df['original_image_name'].tolist())
        print(f"📊 Found {len(valid_image_names)} valid image names")
        
        # Filter the mapping to only include valid images
        filtered_mapping_df = mapping_df[mapping_df['original_image_name'].isin(valid_image_names)].reset_index(drop=True)
        print(f"✅ Filtered mapping has {len(filtered_mapping_df)} entries (removed {len(mapping_df) - len(filtered_mapping_df)} entries)")
        
        # Save the filtered mapping
        print(f"💾 Saving filtered mapping to {filtered_mapping_path}")
        filtered_mapping_df.to_csv(filtered_mapping_path, index=False)
        print(f"✅ Successfully saved filtered mapping to {filtered_mapping_path}")
        
        # Load the original mask arrays
        print(f"📁 Loading mask arrays from {masks_arrays_path}")
        masks_data = np.load(masks_arrays_path)
        print(f"✅ Loaded {len(masks_data.files)} mask arrays")
        
        # Create a new .npz file with only the valid mask arrays
        print("🔄 Creating filtered mask arrays file...")
        filtered_masks_data = {}
        
        # Get the valid array keys from the filtered mapping
        valid_array_keys = set(filtered_mapping_df['numpy_array_key'].tolist())
        
        # Copy only the valid arrays
        for key in valid_array_keys:
            if key in masks_data:
                filtered_masks_data[key] = masks_data[key]
            else:
                print(f"⚠️  Warning: Array key {key} not found in mask data")
        
        print(f"✅ Filtered mask arrays has {len(filtered_masks_data)} arrays (removed {len(masks_data.files) - len(filtered_masks_data)} arrays)")
        
        # Save the filtered mask arrays
        print(f"💾 Saving filtered mask arrays to {filtered_masks_arrays_path}")
        np.savez_compressed(filtered_masks_arrays_path, **filtered_masks_data)
        print(f"✅ Successfully saved filtered mask arrays to {filtered_masks_arrays_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {data_type} data: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to preprocess both train and validation datasets.
    """
    print("🎯 Background Mask Data Preprocessing")
    print("=" * 50)
    
    # Process training data
    print("\n📚 Processing training data...")
    train_success = preprocess_background_masks_data('train')
    
    # Process validation data
    print("\n📚 Processing validation data...")
    val_success = preprocess_background_masks_data('val')
    
    if train_success and val_success:
        print(f"\n🎉 Preprocessing completed successfully!")
        print(f"📁 Output files created:")
        project_root = Path(__file__).parent.parent
        print(f"   - {project_root}/data/train_processed/background_masks_data_with_labels.csv")
        print(f"   - {project_root}/data/val_processed/background_masks_data_with_labels.csv")
        print(f"   - {project_root}/data/train_processed/background_masks_arrays_filtered.npz")
        print(f"   - {project_root}/data/val_processed/background_masks_arrays_filtered.npz")
        print(f"   - {project_root}/data/train_processed/mask_arrays_mapping_filtered.csv")
        print(f"   - {project_root}/data/val_processed/mask_arrays_mapping_filtered.csv")
        print(f"\n📝 Next steps:")
        print(f"   1. Update the SVC classifier to use the new filtered files")
        print(f"   2. Run the SVC classifier with the preprocessed data")
    else:
        print(f"\n❌ Preprocessing failed!")
        if not train_success:
            print(f"   - Training data preprocessing failed")
        if not val_success:
            print(f"   - Validation data preprocessing failed")

if __name__ == "__main__":
    main()
