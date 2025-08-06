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
    """Process a single image to extract only the background and return mask data."""
    try:
        # Load original image
        original_image = Image.open(image_path)
        
        # Get the mask using rembg (this gives us the foreground mask)
        mask = remove(original_image, only_mask=True)
        
        # Convert mask to numpy array
        mask_array = np.array(mask)
        
        # Invert the mask to get background mask
        background_mask = 255 - mask_array
        
        # Convert back to PIL Image
        background_mask_pil = Image.fromarray(background_mask.astype(np.uint8))
        
        # Apply the background mask to the original image
        original_array = np.array(original_image)
        
        # Create a 4-channel image (RGBA) if it's not already
        if original_array.shape[2] == 3:
            # Add alpha channel
            rgba_image = np.zeros((original_array.shape[0], original_array.shape[1], 4), dtype=np.uint8)
            rgba_image[:, :, :3] = original_array
            rgba_image[:, :, 3] = 255  # Full opacity
        else:
            rgba_image = original_array
        
        # Apply the background mask to the alpha channel
        rgba_image[:, :, 3] = background_mask[:, :, 0] if background_mask.ndim == 3 else background_mask
        
        # Convert back to PIL Image (ensure uint8 dtype for proper RGBA interpretation)
        background_image = Image.fromarray(rgba_image.astype(np.uint8))
        # Ensure it's interpreted as RGBA
        if background_image.mode != 'RGBA':
            background_image = background_image.convert('RGBA')
        
        # Save the background extracted image
        output_path = processed_images_dir / f"background_extracted_{image_path.stem}.png"
        background_image.save(output_path)
        
        # Flatten the background mask array for CSV storage
        if background_mask.ndim == 3:
            # If mask has multiple channels, take the first channel
            background_mask_flat = background_mask[:, :, 0].flatten()
        else:
            background_mask_flat = background_mask.flatten()
        
        # Extract background color information for color analysis
        original_array = np.array(original_image)
        if original_array.shape[2] == 3:  # RGB
            # Apply background mask to get background pixels only
            background_pixels_mask = background_mask > 127
            if background_mask.ndim == 3:
                background_pixels_mask = background_pixels_mask[:, :, 0]
            
            # Extract RGB values of background pixels
            background_r = original_array[:, :, 0][background_pixels_mask]
            background_g = original_array[:, :, 1][background_pixels_mask]
            background_b = original_array[:, :, 2][background_pixels_mask]
            
            # Calculate average background color
            avg_background_color = {
                'avg_r': np.mean(background_r) if len(background_r) > 0 else 0,
                'avg_g': np.mean(background_g) if len(background_g) > 0 else 0,
                'avg_b': np.mean(background_b) if len(background_b) > 0 else 0
            }
        else:
            avg_background_color = {'avg_r': 0, 'avg_g': 0, 'avg_b': 0}
        
        return {
            'image_name': image_path.name,
            'processed_image_name': f"background_extracted_{image_path.stem}.png",
            'background_mask': background_mask_flat.tolist(),
            'mask_shape': background_mask.shape if background_mask.ndim == 2 else background_mask.shape[:2],
            'background_color': avg_background_color
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
    
    # Get all image files (using case-insensitive matching to avoid duplicates)
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = set()  # Use set to automatically avoid duplicates
    for ext in image_extensions:
        # Add both lowercase and uppercase patterns, set will handle duplicates
        image_files.update(input_dir.glob(ext))
        image_files.update(input_dir.glob(ext.upper()))
    
    # Convert back to list for processing
    image_files = list(image_files)
    
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
            # Prepare data for CSV - include color analysis for white/cream detection
            csv_data = []
            for item in mask_data:
                # Classify background color
                bg_color = item['background_color']
                avg_r, avg_g, avg_b = bg_color['avg_r'], bg_color['avg_g'], bg_color['avg_b']
                
                # Determine if background is white/cream colored
                is_white = (avg_r > 240 and avg_g > 240 and avg_b > 240)
                is_cream = (avg_r > 240 and avg_g > 235 and avg_b > 220 and 
                           avg_r >= avg_g >= avg_b and (avg_r - avg_b) < 30)
                
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
                    'total_pixels': len(item['background_mask']),
                    'avg_background_r': avg_r,
                    'avg_background_g': avg_g,
                    'avg_background_b': avg_b,
                    'is_white_background': is_white,
                    'is_cream_background': is_cream
                }
                csv_data.append(csv_row)
            
            # Write to CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['original_image_name', 'processed_image_name', 'mask_height', 'mask_width', 
                             'mask_mean', 'mask_std', 'mask_min', 'mask_max', 'background_pixels', 'total_pixels',
                             'avg_background_r', 'avg_background_g', 'avg_background_b', 'is_white_background', 'is_cream_background']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"   ✅ Background mask data saved successfully")
            
            # Save complete mask arrays in numpy format for efficient storage and color analysis
            masks_numpy_path = processed_data_dir / "background_masks_arrays.npz"
            print(f"💾 Saving all background mask arrays to {masks_numpy_path}")
            
            # Prepare arrays for numpy storage
            image_names = [item['image_name'] for item in mask_data]
            mask_arrays = {}
            
            for i, item in enumerate(mask_data):
                # Create a safe filename for numpy array key
                safe_name = item['image_name'].replace('.', '_').replace('-', '_')
                mask_arrays[safe_name] = np.array(item['background_mask']).reshape(item['mask_shape'])
            
            # Save all mask arrays in compressed numpy format
            np.savez_compressed(masks_numpy_path, **mask_arrays)
            
            # Also save a mapping file for easy reference
            mapping_path = processed_data_dir / "mask_arrays_mapping.csv"
            with open(mapping_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['original_image_name', 'numpy_array_key', 'mask_shape'])
                for item in mask_data:
                    safe_name = item['image_name'].replace('.', '_').replace('-', '_')
                    writer.writerow([item['image_name'], safe_name, f"{item['mask_shape'][0]}x{item['mask_shape'][1]}"])
            
            print(f"   ✅ All {len(mask_data)} background mask arrays saved in compressed format")
            print(f"   ✅ Array mapping saved to {mapping_path}")
            
        except Exception as e:
            print(f"   ❌ Error saving CSV data: {e}")
    
    print(f"\n🎉 Processing completed!")
    print(f"   Successfully processed: {successful_processed}/{len(image_files)} images")
    print(f"   Processed images saved to: {processed_images_dir}")
    print(f"   Background mask data saved to: {processed_data_dir}")
    
    return successful_processed > 0

def main():
    """Main function to process all images and extract backgrounds."""
    print("🚀 Starting Image Background Extraction Processing")
    print("=" * 60)
    
    # Process all images from data/images
    if process_all_images():
        print("✅ Image processing completed successfully!")
        print("\n📁 Results:")
        print("   - Background extracted images saved to: data/processed_images/")
        print("   - Background color analysis saved to: data/processed/background_masks_data.csv")
        print("   - Complete mask arrays saved to: data/processed/background_masks_arrays.npz")
        print("   - Array mapping reference saved to: data/processed/mask_arrays_mapping.csv")
    else:
        print("❌ Image processing failed!")
    
    print("\n" + "=" * 60)
    print("🎉 Processing completed!")

if __name__ == "__main__":
    main() 