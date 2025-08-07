"""
Script de procesamiento de extracción de fondo
Este script procesa todas las imágenes de datos/imágenes, elimina los fondos y guarda los resultados.

Ejemplos de como usar:

python image_bg_extraction.py --test --input-dir "data/images"\
    --output-images-dir "data/train_processed_images"\
    --output-data-dir "data/train_processed"

python image_bg_extraction.py --test --input-dir "data/validation_images"\
    --output-images-dir "data/val_processed_images"\
    --output-data-dir "data/val_processed"
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

def create_output_directories(processed_images_dir: str = None, processed_data_dir: str = None):
    """
    Create the necessary output directories if they don't exist.
    
    Args:
        processed_images_dir (str, optional): Directory for processed images
        processed_data_dir (str, optional): Directory for processed data
        
    Returns:
        tuple: (processed_images_dir, processed_data_dir) as Path objects
    """
    if processed_images_dir:
        processed_images_dir = Path(processed_images_dir)
    else:
        processed_images_dir = Path("data/processed_images")
        
    if processed_data_dir:
        processed_data_dir = Path(processed_data_dir)
    else:
        processed_data_dir = Path("data/processed")
    
    processed_images_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Processed images directory: {processed_images_dir.absolute()}")
    print(f"📁 Processed data directory: {processed_data_dir.absolute()}")
    
    return processed_images_dir, processed_data_dir

def process_single_image(image_path, processed_images_dir, target_size=(512, 512)):
    """
    Process a single image to extract only the background and return mask data.
    
    Args:
        image_path: Path to the input image
        processed_images_dir: Directory to save processed images
        target_size: Tuple of (width, height) for resizing images (default: 512x512)
    """
    try:
        # Load original image
        original_image = Image.open(image_path)
        
        # Validate image
        if original_image.size[0] == 0 or original_image.size[1] == 0:
            print(f"⚠️  Warning: Invalid image size for {image_path.name}")
            return None
        
        # Convert to RGB mode if it's not already (handles grayscale, RGBA, etc.)
        if original_image.mode not in ['RGB', 'RGBA']:
            try:
                original_image = original_image.convert('RGB')
            except Exception as e:
                print(f"⚠️  Warning: Could not convert {image_path.name} to RGB: {e}")
                return None
        
        # Resize image to target size while maintaining aspect ratio
        # Calculate the scaling factor to fit the image within the target size
        original_width, original_height = original_image.size
        target_width, target_height = target_size
        
        # Calculate scaling factor to fit the image within target size
        scale_factor = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Ensure minimum size
        if new_width < 1 or new_height < 1:
            print(f"⚠️  Warning: Image {image_path.name} too small to resize")
            return None
        
        # Resize the image
        resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a new image with the target size and paste the resized image in the center
        # Use RGB mode and white background (255, 255, 255)
        final_image = Image.new('RGB', target_size, (255, 255, 255))
        
        # Calculate position to center the resized image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        
        # Paste the resized image onto the final image
        final_image.paste(resized_image, (x_offset, y_offset))
        
        # Get the mask using rembg (this gives us the foreground mask)
        mask = remove(final_image, only_mask=True)
        
        # Convert mask to numpy array
        mask_array = np.array(mask)
        
        # Invert the mask to get background mask
        background_mask = 255 - mask_array
        
        # Convert back to PIL Image
        background_mask_pil = Image.fromarray(background_mask.astype(np.uint8))
        
        # Apply the background mask to the final image
        final_array = np.array(final_image)
        
        # Create a 4-channel image (RGBA) if it's not already
        if final_array.shape[2] == 3:
            # Add alpha channel
            rgba_image = np.zeros((final_array.shape[0], final_array.shape[1], 4), dtype=np.uint8)
            rgba_image[:, :, :3] = final_array
            rgba_image[:, :, 3] = 255  # Full opacity
        else:
            rgba_image = final_array
        
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
        if final_array.shape[2] == 3:  # RGB
            # Apply background mask to get background pixels only
            background_pixels_mask = background_mask > 127
            if background_mask.ndim == 3:
                background_pixels_mask = background_pixels_mask[:, :, 0]
            
            # Extract RGB values of background pixels
            background_r = final_array[:, :, 0][background_pixels_mask]
            background_g = final_array[:, :, 1][background_pixels_mask]
            background_b = final_array[:, :, 2][background_pixels_mask]
            
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
            'background_color': avg_background_color,
            'original_size': (original_width, original_height),
            'resized_size': (new_width, new_height),
            'final_size': target_size
        }
        
    except Exception as e:
        # Error will be tracked in the progress bar
        print(f"Error processing {image_path.name}: {e}")
        return None

def process_all_images(input_dir: str = None, processed_images_dir: str = None, processed_data_dir: str = None, limit: int = None, target_size: tuple = (512, 512)):
    """
    Process all images from the specified input directory.
    
    Args:
        input_dir (str, optional): Directory containing images to process (default: data/images)
        processed_images_dir (str, optional): Directory to save processed images (default: data/processed_images)
        processed_data_dir (str, optional): Directory to save processed data (default: data/processed)
        limit (int, optional): Limit number of images to process (for testing)
        target_size (tuple, optional): Target size for resizing images (default: (512, 512))
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    print("\n🚀 Starting batch image processing...")
    print(f"📏 Target image size: {target_size[0]}x{target_size[1]} pixels")
    
    # Create output directories
    processed_images_dir, processed_data_dir = create_output_directories(processed_images_dir, processed_data_dir)
    
    # Check if input directory exists
    if input_dir:
        input_dir = Path(input_dir)
    else:
        input_dir = Path("data/images")
        
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
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
    
    # Apply limit if specified
    if limit:
        image_files = image_files[:limit]
        print(f"🔧 Limited to {limit} images for testing")
    
    # Store results for CSV
    mask_data = []
    successful_processed = 0
    
    # Process each image with progress bar
    with tqdm(image_files, desc="Processing images", unit="image") as pbar:
        for image_path in pbar:
            pbar.set_description(f"Processing {image_path.name}")
            
            result = process_single_image(image_path, processed_images_dir, target_size)
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
                    'is_cream_background': is_cream,
                    'original_width': item['original_size'][0],
                    'original_height': item['original_size'][1],
                    'resized_width': item['resized_size'][0],
                    'resized_height': item['resized_size'][1],
                    'final_width': item['final_size'][0],
                    'final_height': item['final_size'][1]
                }
                csv_data.append(csv_row)
            
            # Write to CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['original_image_name', 'processed_image_name', 'mask_height', 'mask_width', 
                             'mask_mean', 'mask_std', 'mask_min', 'mask_max', 'background_pixels', 'total_pixels',
                             'avg_background_r', 'avg_background_g', 'avg_background_b', 'is_white_background', 'is_cream_background',
                             'original_width', 'original_height', 'resized_width', 'resized_height', 'final_width', 'final_height']
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Background extraction from images")
    parser.add_argument("--input-dir", type=str, help="Directory containing images to process (default: data/images)")
    parser.add_argument("--output-images-dir", type=str, help="Directory to save processed images (default: data/processed_images)")
    parser.add_argument("--output-data-dir", type=str, help="Directory to save processed data (default: data/processed)")
    parser.add_argument("--limit", type=int, help="Limit number of images to process (for testing)")
    parser.add_argument("--test", action="store_true", help="Test mode - process only 10 images")
    parser.add_argument("--target-size", type=int, nargs=2, default=[512, 512], 
                       help="Target size for resizing images (width height) (default: 512 512)")
    
    args = parser.parse_args()
    
    print("🚀 Starting Image Background Extraction Processing")
    print("=" * 60)
    
    # Configure limit
    limit = None
    if args.test:
        limit = 10
    elif args.limit:
        limit = args.limit
    
    # Configure target size
    target_size = tuple(args.target_size)
    print(f"📏 Target image size: {target_size[0]}x{target_size[1]} pixels")
    
    # Process all images with custom paths
    success = process_all_images(
        input_dir=args.input_dir,
        processed_images_dir=args.output_images_dir,
        processed_data_dir=args.output_data_dir,
        limit=limit,
        target_size=target_size
    )
    
    if success:
        print("✅ Image processing completed successfully!")
        print("\n📁 Results:")
        
        # Determine actual paths used
        actual_input_dir = args.input_dir if args.input_dir else "data/images"
        actual_output_images_dir = args.output_images_dir if args.output_images_dir else "data/processed_images"
        actual_output_data_dir = args.output_data_dir if args.output_data_dir else "data/processed"
        
        print(f"   - Background extracted images saved to: {actual_output_images_dir}/")
        print(f"   - Background color analysis saved to: {actual_output_data_dir}/background_masks_data.csv")
        print(f"   - Complete mask arrays saved to: {actual_output_data_dir}/background_masks_arrays.npz")
        print(f"   - Array mapping reference saved to: {actual_output_data_dir}/mask_arrays_mapping.csv")
        print(f"   - All images resized to: {target_size[0]}x{target_size[1]} pixels")
    else:
        print("❌ Image processing failed!")
    
    print("\n" + "=" * 60)
    print("🎉 Processing completed!")

if __name__ == "__main__":
    main() 