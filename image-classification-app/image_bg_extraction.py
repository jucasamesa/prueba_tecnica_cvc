#!/usr/bin/env python3
"""
Background Extraction Processing Script
This script processes all images from data/images, removes backgrounds, and saves results.
"""

import os
import sys
import csv
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import rembg
try:
    from rembg import remove, new_session
    print("✅ rembg imported successfully!")
except ImportError as e:
    print(f"❌ Error importing rembg: {e}")
    sys.exit(1)

def create_output_directories():
    """Create the necessary output directories if they don't exist."""
    processed_images_dir = Path("data/processed_images")
    processed_data_dir = Path("data/processed")
    
    processed_images_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Processed images directory: {processed_images_dir.absolute()}")
    print(f"📁 Processed data directory: {processed_data_dir.absolute()}")
    
    return processed_images_dir, processed_data_dir

def process_single_image(image_path, processed_images_dir):
    """Process a single image to remove background and return mask data."""
    try:
        # Load original image
        original_image = Image.open(image_path)
        
        # Remove background to get image without background
        no_background_image = remove(original_image)
        
        # Get the mask using rembg (this gives us the foreground mask)
        mask = remove(original_image, only_mask=True)
        
        # Convert mask to numpy array
        mask_array = np.array(mask)
        
        # Invert the mask to get background mask
        background_mask = 255 - mask_array
        
        # Save the image without background
        output_path = processed_images_dir / f"no_background_{image_path.stem}.png"
        no_background_image.save(output_path)
        
        # Flatten the background mask array for CSV storage
        if background_mask.ndim == 3:
            # If mask has multiple channels, take the first channel
            background_mask_flat = background_mask[:, :, 0].flatten()
        else:
            background_mask_flat = background_mask.flatten()
        
        return {
            'image_name': image_path.name,
            'processed_image_name': f"no_background_{image_path.stem}.png",
            'background_mask': background_mask_flat.tolist(),
            'mask_shape': background_mask.shape if background_mask.ndim == 2 else background_mask.shape[:2]
        }
        
    except Exception as e:
        # Error will be tracked in the progress bar
        return None

def process_all_images():
    """Process all images from data/images directory."""
    print("\n🚀 Starting batch image processing...")
    
    # Create output directories
    processed_images_dir, processed_data_dir = create_output_directories()
    
    # Check if input directory exists
    input_dir = Path("data/images")
    if not input_dir.exists():
        print("❌ data/images directory not found.")
        return False
    
    # Get all image files
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(ext)))
        image_files.extend(list(input_dir.glob(ext.upper())))
    
    if not image_files:
        print("❌ No image files found in data/images.")
        return False
    
    print(f"📊 Found {len(image_files)} images to process")
    
    # Store results for CSV
    mask_data = []
    successful_processed = 0
    
    # Process each image with progress bar
    with tqdm(image_files, desc="Processing images", unit="image") as pbar:
        for image_path in pbar:
            pbar.set_description(f"Processing {image_path.name}")
            
            result = process_single_image(image_path, processed_images_dir)
            if result:
                mask_data.append(result)
                successful_processed += 1
            
            # Update progress bar with current status
            pbar.set_postfix({
                'Processed': successful_processed,
                'Failed': len(image_files) - successful_processed - (len(image_files) - (pbar.n + 1))
            })
    
    # Save mask data to CSV
    if mask_data:
        csv_path = processed_data_dir / "background_masks_data.csv"
        print(f"\n💾 Saving background mask data to {csv_path}")
        
        try:
            # Prepare data for CSV - we'll save mask statistics instead of full arrays due to size
            csv_data = []
            for item in mask_data:
                csv_row = {
                    'original_image_name': item['image_name'],
                    'processed_image_name': item['processed_image_name'],
                    'mask_height': item['mask_shape'][0],
                    'mask_width': item['mask_shape'][1],
                    'mask_mean': np.mean(item['background_mask']),
                    'mask_std': np.std(item['background_mask']),
                    'mask_min': np.min(item['background_mask']),
                    'mask_max': np.max(item['background_mask']),
                    'background_pixels': np.sum(np.array(item['background_mask']) > 127),
                    'total_pixels': len(item['background_mask'])
                }
                csv_data.append(csv_row)
            
            # Write to CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['original_image_name', 'processed_image_name', 'mask_height', 'mask_width', 
                             'mask_mean', 'mask_std', 'mask_min', 'mask_max', 'background_pixels', 'total_pixels']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"   ✅ Background mask data saved successfully")
            
            # Also save detailed mask arrays to a separate file for reference
            detailed_csv_path = processed_data_dir / "detailed_background_masks.csv"
            print(f"💾 Saving detailed background mask arrays to {detailed_csv_path}")
            
            with open(detailed_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['image_name', 'processed_image_name'] + [f'pixel_{i}' for i in range(len(mask_data[0]['background_mask']))]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for item in mask_data[:5]:  # Save only first 5 for demo due to size constraints
                    row = {
                        'image_name': item['image_name'],
                        'processed_image_name': item['processed_image_name']
                    }
                    for i, pixel_val in enumerate(item['background_mask']):
                        row[f'pixel_{i}'] = pixel_val
                    writer.writerow(row)
            
            print(f"   ✅ Detailed mask arrays saved (first 5 images as example)")
            
        except Exception as e:
            print(f"   ❌ Error saving CSV data: {e}")
    
    print(f"\n🎉 Processing completed!")
    print(f"   Successfully processed: {successful_processed}/{len(image_files)} images")
    print(f"   Processed images saved to: {processed_images_dir}")
    print(f"   Background mask data saved to: {processed_data_dir}")
    
    return successful_processed > 0

def main():
    """Main function to process all images and remove backgrounds."""
    print("🚀 Starting Image Background Removal Processing")
    print("=" * 60)
    
    # Process all images from data/images
    if process_all_images():
        print("✅ Image processing completed successfully!")
        print("\n📁 Results:")
        print("   - Processed images (without background) saved to: data/processed_images/")
        print("   - Background mask statistics saved to: data/processed/background_masks_data.csv")
        print("   - Detailed mask arrays (sample) saved to: data/processed/detailed_background_masks.csv")
    else:
        print("❌ Image processing failed!")
    
    print("\n" + "=" * 60)
    print("🎉 Processing completed!")

if __name__ == "__main__":
    main() 