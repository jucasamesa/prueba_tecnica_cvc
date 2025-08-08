# Analyze background-only images vs images with objects
'''
Este script permitio analizar la ventaja de utilizar una imagen con el objeto principal removido sobre una imagen original
'''
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from rembg import remove
import seaborn as sns

def analyze_image_differences():
    """
    Analyze the differences between images with and without foreground objects.
    """
    print("🔍 Analyzing image differences with and without foreground objects...")
    
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
    
    if not original_img_path.exists():
        print(f"❌ Original image not found: {original_img_path}")
        return
    
    # Read original image
    original_img = cv2.imread(str(original_img_path))
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Create background-only image (remove foreground object)
    original_pil = Image.open(original_img_path)
    foreground = remove(original_pil)
    
    # Create white background
    white_background = Image.new('RGBA', foreground.size, (255, 255, 255, 255))
    background_only = Image.alpha_composite(white_background, foreground)
    background_only_rgb = np.array(background_only.convert('RGB'))
    
    # Read current processed image (with transparency)
    if processed_img_path.exists():
        processed_img = cv2.imread(str(processed_img_path), cv2.IMREAD_UNCHANGED)
        if processed_img is not None and processed_img.shape[2] == 4:
            # Convert RGBA to RGB by compositing with white background
            alpha = processed_img[:, :, 3] / 255.0
            processed_rgb = processed_img[:, :, :3] * alpha[:, :, np.newaxis] + (1 - alpha[:, :, np.newaxis]) * 255
            processed_rgb = processed_rgb.astype(np.uint8)
        else:
            processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB) if processed_img is not None else None
    else:
        processed_rgb = None
    
    # Analyze differences
    print(f"\n📊 Image Analysis:")
    print(f"Original image shape: {original_rgb.shape}")
    print(f"Background-only image shape: {background_only_rgb.shape}")
    
    # Calculate statistics
    original_mean = np.mean(original_rgb, axis=(0, 1))
    background_only_mean = np.mean(background_only_rgb, axis=(0, 1))
    
    original_std = np.std(original_rgb, axis=(0, 1))
    background_only_std = np.std(background_only_rgb, axis=(0, 1))
    
    print(f"\n🎨 Color Statistics:")
    print(f"Original - Mean RGB: {original_mean}, Std RGB: {original_std}")
    print(f"Background-only - Mean RGB: {background_only_mean}, Std RGB: {background_only_std}")
    
    # Calculate differences
    mean_diff = background_only_mean - original_mean
    std_diff = background_only_std - original_std
    
    print(f"Mean difference: {mean_diff}")
    print(f"Std difference: {std_diff}")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    ax1.imshow(original_rgb)
    ax1.set_title('Original Image (with object)')
    ax1.axis('off')
    
    # Background-only image
    ax2.imshow(background_only_rgb)
    ax2.set_title('Background-Only Image (object removed)')
    ax2.axis('off')
    
    # Color distribution comparison
    colors = ['Red', 'Green', 'Blue']
    x = np.arange(len(colors))
    width = 0.35
    
    ax3.bar(x - width/2, original_mean, width, label='Original', alpha=0.7)
    ax3.bar(x + width/2, background_only_mean, width, label='Background-Only', alpha=0.7)
    ax3.set_xlabel('Color Channel')
    ax3.set_ylabel('Mean Value')
    ax3.set_title('Mean Color Values Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(colors)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Histogram comparison
    ax4.hist(original_rgb.ravel(), bins=50, alpha=0.7, label='Original', density=True)
    ax4.hist(background_only_rgb.ravel(), bins=50, alpha=0.7, label='Background-Only', density=True)
    ax4.set_xlabel('Pixel Value')
    ax4.set_ylabel('Density')
    ax4.set_title('Pixel Value Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze for model training
    print(f"\n🤖 Model Training Analysis:")
    
    # Calculate feature differences
    original_features = {
        'mean_r': original_mean[0],
        'mean_g': original_mean[1], 
        'mean_b': original_mean[2],
        'std_r': original_std[0],
        'std_g': original_std[1],
        'std_b': original_std[2],
        'brightness': np.mean(original_rgb),
        'contrast': np.std(original_rgb)
    }
    
    background_features = {
        'mean_r': background_only_mean[0],
        'mean_g': background_only_mean[1],
        'mean_b': background_only_mean[2], 
        'std_r': background_only_std[0],
        'std_g': background_only_std[1],
        'std_b': background_only_std[2],
        'brightness': np.mean(background_only_rgb),
        'contrast': np.std(background_only_rgb)
    }
    
    print(f"Original features: {original_features}")
    print(f"Background-only features: {background_features}")
    
    # Calculate feature differences
    feature_diffs = {}
    for key in original_features:
        feature_diffs[key] = background_features[key] - original_features[key]
    
    print(f"\nFeature differences (Background-Only - Original):")
    for key, diff in feature_diffs.items():
        print(f"  {key}: {diff:+.2f}")
    
    return original_features, background_features, feature_diffs

def analyze_model_impact():
    """
    Analyze the impact of background-only images on model training.
    """
    print("\n🤖 Analyzing impact on model training...")
    
    # Get project root
    project_root = Path(".")
    
    # Load the CSV to get image paths
    train_processed = pd.read_csv(project_root / "data/train_processed/background_masks_data_with_labels.csv")
    
    # Sample multiple images for analysis
    sample_rows = train_processed.sample(min(10, len(train_processed)), random_state=42)
    
    original_features_list = []
    background_features_list = []
    
    for idx, row in sample_rows.iterrows():
        original_img_path = project_root / "data/images" / row['original_image_name']
        
        if not original_img_path.exists():
            continue
        
        try:
            # Read original image
            original_img = cv2.imread(str(original_img_path))
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            # Create background-only image
            original_pil = Image.open(original_img_path)
            foreground = remove(original_pil)
            white_background = Image.new('RGBA', foreground.size, (255, 255, 255, 255))
            background_only = Image.alpha_composite(white_background, foreground)
            background_only_rgb = np.array(background_only.convert('RGB'))
            
            # Calculate features
            original_mean = np.mean(original_rgb, axis=(0, 1))
            original_std = np.std(original_rgb, axis=(0, 1))
            
            background_mean = np.mean(background_only_rgb, axis=(0, 1))
            background_std = np.std(background_only_rgb, axis=(0, 1))
            
            original_features = {
                'mean_r': original_mean[0],
                'mean_g': original_mean[1],
                'mean_b': original_mean[2],
                'std_r': original_std[0],
                'std_g': original_std[1],
                'std_b': original_std[2],
                'brightness': np.mean(original_rgb),
                'contrast': np.std(original_rgb)
            }
            
            background_features = {
                'mean_r': background_mean[0],
                'mean_g': background_mean[1],
                'mean_b': background_mean[2],
                'std_r': background_std[0],
                'std_g': background_std[1],
                'std_b': background_std[2],
                'brightness': np.mean(background_only_rgb),
                'contrast': np.std(background_only_rgb)
            }
            
            original_features_list.append(original_features)
            background_features_list.append(background_features)
            
        except Exception as e:
            print(f"⚠️  Error processing {row['original_image_name']}: {e}")
    
    if not original_features_list:
        print("❌ No images could be processed")
        return
    
    # Convert to DataFrames
    original_df = pd.DataFrame(original_features_list)
    background_df = pd.DataFrame(background_features_list)
    
    # Calculate average differences
    avg_diffs = background_df.mean() - original_df.mean()
    
    print(f"\n📊 Average Feature Differences (Background-Only - Original):")
    for feature, diff in avg_diffs.items():
        print(f"  {feature}: {diff:+.2f}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    features = ['mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b', 'brightness', 'contrast']
    
    for i, feature in enumerate(features):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        ax.hist(original_df[feature], alpha=0.7, label='Original', bins=10)
        ax.hist(background_df[feature], alpha=0.7, label='Background-Only', bins=10)
        ax.set_title(f'{feature.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Model training recommendations
    print(f"\n🎯 Model Training Recommendations:")
    
    # Analyze feature stability
    feature_stability = {}
    for feature in features:
        original_cv = original_df[feature].std() / original_df[feature].mean()
        background_cv = background_df[feature].std() / background_df[feature].mean()
        feature_stability[feature] = {
            'original_cv': original_cv,
            'background_cv': background_cv,
            'improvement': original_cv - background_cv
        }
    
    print(f"\nFeature Stability Analysis (Coefficient of Variation):")
    for feature, stats in feature_stability.items():
        print(f"  {feature}:")
        print(f"    Original CV: {stats['original_cv']:.3f}")
        print(f"    Background-Only CV: {stats['background_cv']:.3f}")
        print(f"    Improvement: {stats['improvement']:+.3f}")
    
    # Overall recommendation
    print(f"\n💡 Overall Recommendation:")
    
    # Count improvements
    improvements = sum(1 for stats in feature_stability.values() if stats['improvement'] > 0)
    total_features = len(feature_stability)
    
    if improvements > total_features / 2:
        print(f"✅ RECOMMENDED: Use background-only images for model training")
        print(f"   - {improvements}/{total_features} features show improved stability")
        print(f"   - Background-only images provide more consistent feature patterns")
        print(f"   - Reduces noise from foreground objects")
    else:
        print(f"⚠️  CONSIDER: Original images might be better for model training")
        print(f"   - Only {improvements}/{total_features} features show improvement")
        print(f"   - Foreground objects might provide useful information")
    
    print(f"\n🔍 Key Benefits of Background-Only Images:")
    print(f"  1. More consistent background patterns")
    print(f"  2. Reduced noise from foreground objects")
    print(f"  3. Better focus on background quality features")
    print(f"  4. Cleaner feature distributions")
    
    print(f"\n⚠️  Potential Drawbacks:")
    print(f"  1. Loss of object-background interaction information")
    print(f"  2. May miss contextual background quality cues")
    print(f"  3. Artificial white backgrounds may not represent real scenarios")

if __name__ == "__main__":
    print("🔍 Background-Only Image Analysis")
    print("1. Analyze single image differences")
    print("2. Analyze model training impact")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        analyze_image_differences()
    elif choice == "2":
        analyze_model_impact()
    else:
        print("Running both analyses...")
        analyze_image_differences()
        analyze_model_impact()
