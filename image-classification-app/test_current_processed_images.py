# Test script to verify current processed images
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def test_current_processed_images():
    """
    Test what the current processed images actually contain.
    """
    print("🔍 Testing current processed images...")
    
    # Get project root
    project_root = Path(".")
    
    # Load the CSV to get image paths
    train_processed = pd.read_csv(project_root / "data/train_processed/background_masks_data_with_labels.csv")
    
    # Get sample images
    sample_row = train_processed.sample(1, random_state=50)
    
    # Image paths
    original_img_path = project_root / "data/images" / sample_row['original_image_name'].iloc[0]
    processed_img_path = project_root / "data/train_processed_images" / sample_row['processed_image_name'].iloc[0]
    
    print(f"Original image: {original_img_path}")
    print(f"Processed image: {processed_img_path}")
    
    # Check if files exist
    if not original_img_path.exists():
        print(f"❌ Original image not found: {original_img_path}")
        return
    
    if not processed_img_path.exists():
        print(f"❌ Processed image not found: {processed_img_path}")
        return
    
    # Read images
    original_img = cv2.imread(str(original_img_path))
    processed_img = cv2.imread(str(processed_img_path), cv2.IMREAD_UNCHANGED)
    
    if original_img is None:
        print(f"❌ Could not read original image: {original_img_path}")
        return
    
    if processed_img is None:
        print(f"❌ Could not read processed image: {processed_img_path}")
        return
    
    # Convert BGR to RGB for original
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Handle processed image
    if processed_img.shape[2] == 4:  # RGBA
        print("📊 Processed image has alpha channel (RGBA)")
        
        # Get alpha channel
        alpha = processed_img[:, :, 3]
        
        # Count transparent pixels
        transparent_pixels = np.sum(alpha < 255)
        total_pixels = alpha.shape[0] * alpha.shape[1]
        transparency_ratio = transparent_pixels / total_pixels
        
        print(f"Transparent pixels: {transparent_pixels:,} / {total_pixels:,} ({transparency_ratio:.2%})")
        
        # Convert RGBA to RGB by compositing with white background
        alpha_normalized = alpha / 255.0
        processed_rgb = processed_img[:, :, :3] * alpha_normalized[:, :, np.newaxis] + (1 - alpha_normalized[:, :, np.newaxis]) * 255
        processed_rgb = processed_rgb.astype(np.uint8)
        
        # Also show the alpha channel
        alpha_display = alpha.astype(np.uint8)
        
    else:
        print("📊 Processed image is RGB (no alpha channel)")
        processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        alpha_display = None
    
    # Create subplot
    if alpha_display is not None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        ax1.imshow(original_rgb)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Processed image (composited with white background)
        ax2.imshow(processed_rgb)
        ax2.set_title('Processed Image (White Background)')
        ax2.axis('off')
        
        # Alpha channel
        ax3.imshow(alpha_display, cmap='gray')
        ax3.set_title('Alpha Channel (White = Opaque, Black = Transparent)')
        ax3.axis('off')
        
        # Processed image with transparency
        ax4.imshow(processed_img)
        ax4.set_title('Processed Image (With Transparency)')
        ax4.axis('off')
        
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        ax1.imshow(original_rgb)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Processed image
        ax2.imshow(processed_rgb)
        ax2.set_title('Processed Image')
        ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed information
    print(f"\n📊 Detailed Information:")
    print(f"Original image shape: {original_img.shape}")
    print(f"Processed image shape: {processed_img.shape}")
    print(f"Original image dtype: {original_img.dtype}")
    print(f"Processed image dtype: {processed_img.dtype}")
    
    if processed_img.shape[2] == 4:
        print(f"\n🎯 Analysis:")
        print(f"- The processed image has an alpha channel")
        print(f"- {transparency_ratio:.2%} of pixels are transparent")
        print(f"- This means the foreground object was made transparent")
        print(f"- When displayed, transparent areas may appear as:")
        print(f"  * Black (if alpha channel is ignored)")
        print(f"  * Original object (if transparency isn't handled)")
        print(f"  * White (if composited with white background)")
        
        if transparency_ratio > 0.1:  # More than 10% transparent
            print(f"\n⚠️  Issue detected: High transparency ratio suggests foreground object is still visible")
            print(f"   This explains why you see the original object when displaying the image")
        else:
            print(f"\n✅ Low transparency ratio - background extraction may be working correctly")
    
    else:
        print(f"\n⚠️  Issue: Processed image doesn't have alpha channel")
        print(f"   This suggests the background removal didn't work as expected")

if __name__ == "__main__":
    test_current_processed_images()
