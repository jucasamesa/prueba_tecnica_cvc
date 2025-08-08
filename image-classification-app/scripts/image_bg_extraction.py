"""
Script de procesamiento de extracción de fondo
Este script procesa todas las imágenes de datos/imágenes, elimina los fondos y guarda los resultados.

Ejemplos de como usar:

python scripts/image_bg_extraction.py --test --input-dir "data/images"\
    --output-images-dir "data/train_processed_images"\
    --output-data-dir "data/train_processed"

python scripts/image_bg_extraction.py --test --input-dir "data/validation_images"\
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
import colour

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import rembg
try:
    from rembg import remove, new_session
    print("✅ rembg imported successfully!")
except ImportError as e:
    print(f"❌ Error importing rembg: {e}")
    sys.exit(1)

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

def create_output_directories(processed_images_dir: str = None, processed_data_dir: str = None):
    """
    Create the necessary output directories if they don't exist.
    
    Args:
        processed_images_dir (str, optional): Directory for processed images
        processed_data_dir (str, optional): Directory for processed data
        
    Returns:
        tuple: (processed_images_dir, processed_data_dir) as Path objects
    """
    project_root = Path(__file__).parent.parent
    
    if processed_images_dir:
        processed_images_dir = Path(processed_images_dir)
    else:
        processed_images_dir = project_root / "data/processed_images"
        
    if processed_data_dir:
        processed_data_dir = Path(processed_data_dir)
    else:
        processed_data_dir = project_root / "data/processed"
    
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
        
        # Create background-only image with transparent foreground
        final_array = np.array(final_image)
        
        # Create RGBA image (4 channels) for transparency support
        background_only_rgba = np.zeros((final_array.shape[0], final_array.shape[1], 4), dtype=np.uint8)
        
        # Copy RGB channels
        background_only_rgba[:, :, :3] = final_array
        
        # Set alpha channel based on background mask
        # Where background_mask is 255 (background), set alpha to 255 (opaque)
        # Where background_mask is 0 (foreground), set alpha to 0 (transparent)
        if background_mask.ndim == 3:
            background_only_rgba[:, :, 3] = background_mask[:, :, 0]
        else:
            background_only_rgba[:, :, 3] = background_mask
        
        # Convert to PIL Image for saving
        background_only_image = Image.fromarray(background_only_rgba, 'RGBA')
        
        # Save the background-only image with transparency
        output_path = processed_images_dir / f"background_extracted_{image_path.stem}.png"
        background_only_image.save(output_path)
        
        # For compatibility with existing code, also create RGB version with transparent areas as white
        # This is for the numpy array storage and analysis
        background_only_rgb = background_only_image.convert('RGB')
        background_only_array = np.array(background_only_rgb)
        
        # Extract background color information for color analysis
        # Only analyze pixels that are actually background (not transparent/white foreground)
        if background_mask.ndim == 3:
            background_pixels_mask = background_mask[:, :, 0] > 127
        else:
            background_pixels_mask = background_mask > 127
        
        # Get background pixels from the original image (only actual background pixels)
        background_r = final_array[:, :, 0][background_pixels_mask]
        background_g = final_array[:, :, 1][background_pixels_mask]
        background_b = final_array[:, :, 2][background_pixels_mask]
        
        # Calculate average background color from actual background pixels only
        if len(background_r) > 0:
            avg_background_color = {
                'avg_r': np.mean(background_r),
                'avg_g': np.mean(background_g),
                'avg_b': np.mean(background_b)
            }
        else:
            # If no background pixels found, use the entire image
            avg_background_color = {
                'avg_r': np.mean(final_array[:, :, 0]),
                'avg_g': np.mean(final_array[:, :, 1]),
                'avg_b': np.mean(final_array[:, :, 2])
            }
        
        # Create background-only mask (all pixels are background now)
        background_only_mask = np.ones(background_mask.shape[:2], dtype=np.uint8) * 255
        
        return {
            'image_name': image_path.name,
            'processed_image_name': f"background_extracted_{image_path.stem}.png",
            'background_mask': background_only_mask.flatten().tolist(),  # All pixels are background
            'mask_shape': background_only_mask.shape,
            'background_color': avg_background_color,
            'original_size': (original_width, original_height),
            'resized_size': (new_width, new_height),
            'final_size': target_size,
            'background_only_array': background_only_array  # Store the background-only image array
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
    project_root = Path(__file__).parent.parent
    if input_dir:
        input_dir = Path(input_dir)
    else:
        input_dir = project_root / "data/images"
        
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
                
                # Determine if background is white/cream colored using CIELAB color space
                is_white = (avg_r > 240 and avg_g > 240 and avg_b > 240)
                is_cream_background = is_cream((avg_r, avg_g, avg_b))
                
                # Get background-only array shape
                array_shape = item['background_only_array'].shape
                
                csv_row = {
                    'original_image_name': item['image_name'],
                    'processed_image_name': item['processed_image_name'],
                    'array_height': array_shape[0],
                    'array_width': array_shape[1],
                    'array_channels': array_shape[2],
                    'avg_background_r': avg_r,
                    'avg_background_g': avg_g,
                    'avg_background_b': avg_b,
                    'is_white_background': is_white,
                    'is_cream_background': is_cream_background,
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
                fieldnames = ['original_image_name', 'processed_image_name', 'array_height', 'array_width', 'array_channels',
                             'avg_background_r', 'avg_background_g', 'avg_background_b', 'is_white_background', 'is_cream_background',
                             'original_width', 'original_height', 'resized_width', 'resized_height', 'final_width', 'final_height']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"   ✅ Background mask data saved successfully")
            
            # Save complete background-only image arrays in numpy format for efficient storage
            background_arrays_numpy_path = processed_data_dir / "background_masks_arrays.npz"
            print(f"💾 Saving all background-only image arrays to {background_arrays_numpy_path}")
            
            # Prepare arrays for numpy storage
            image_names = [item['image_name'] for item in mask_data]
            background_arrays = {}
            
            for i, item in enumerate(mask_data):
                # Create a safe filename for numpy array key
                safe_name = item['image_name'].replace('.', '_').replace('-', '_')
                # Save the background-only image array instead of mask
                background_arrays[safe_name] = item['background_only_array']
            
            # Save all background-only arrays in compressed numpy format
            np.savez_compressed(background_arrays_numpy_path, **background_arrays)
            
            # Also save a mapping file for easy reference
            mapping_path = processed_data_dir / "background_arrays_mapping.csv"
            with open(mapping_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['original_image_name', 'numpy_array_key', 'array_shape'])
                for item in mask_data:
                    safe_name = item['image_name'].replace('.', '_').replace('-', '_')
                    array_shape = item['background_only_array'].shape
                    writer.writerow([item['image_name'], safe_name, f"{array_shape[0]}x{array_shape[1]}x{array_shape[2]}"])
            
            print(f"   ✅ All {len(mask_data)} background-only image arrays saved in compressed format")
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
        project_root = Path(__file__).parent.parent
        actual_input_dir = args.input_dir if args.input_dir else str(project_root / "data/images")
        actual_output_images_dir = args.output_images_dir if args.output_images_dir else str(project_root / "data/processed_images")
        actual_output_data_dir = args.output_data_dir if args.output_data_dir else str(project_root / "data/processed")
        
        print(f"   - Background-only images saved to: {actual_output_images_dir}/")
        print(f"   - Background color analysis saved to: {actual_output_data_dir}/background_masks_data.csv")
        print(f"   - Background-only image arrays saved to: {actual_output_data_dir}/background_masks_arrays.npz")
        print(f"   - Array mapping reference saved to: {actual_output_data_dir}/background_arrays_mapping.csv")
        print(f"   - All images resized to: {target_size[0]}x{target_size[1]} pixels")
    else:
        print("❌ Image processing failed!")
    
    print("\n" + "=" * 60)
    print("🎉 Processing completed!")

if __name__ == "__main__":
    main() 