import pandas as pd
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def load_image(image_path)->np.ndarray:
    """
    Load an image and convert to RGB format.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Image in RGB format, or None if loading fails
    """
    try:
        # Load image in BGR format (OpenCV default)
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def analyze_background_color(image, threshold_percentage=0.3)->dict:
    """
    Analyze if the background is white or pastel colored.
    
    Args:
        image (numpy.ndarray): RGB image
        threshold_percentage (float): Percentage of pixels that need to be light to consider it light background
        
    Returns:
        dict: Analysis results including background type and confidence
    """
    if image is None:
        return {'background_type': 'unknown', 'confidence': 0.0, 'reason': 'Failed to load image'}
    
    # Convert to different color spaces for better analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Get image dimensions
    height, width = image.shape[:2]
    total_pixels = height * width
    
    # Analyze different regions of the image (center, corners, edges)
    regions = {
        'center': image[height//4:3*height//4, width//4:3*width//4],
        'top_left': image[:height//3, :width//3],
        'top_right': image[:height//3, 2*width//3:],
        'bottom_left': image[2*height//3:, :width//3],
        'bottom_right': image[2*height//3:, 2*width//3:]
    }
    
    light_pixel_counts = {}
    white_pixel_counts = {}
    pastel_pixel_counts = {}
    
    for region_name, region in regions.items():
        if region.size == 0:
            continue
            
        # Convert region to different color spaces
        region_hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        region_lab = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)
        
        # Define light color thresholds
        # Light colors: high value (V) in HSV, high lightness (L) in LAB
        light_mask = (region_hsv[:, :, 2] > 200) & (region_lab[:, :, 0] > 180)
        
        # White colors: very high value and low saturation
        white_mask = (region_hsv[:, :, 2] > 220) & (region_hsv[:, :, 1] < 30)
        
        # Pastel colors: medium-high value, low-medium saturation
        pastel_mask = (region_hsv[:, :, 2] > 180) & (region_hsv[:, :, 1] < 80) & (region_hsv[:, :, 2] <= 220)
        
        light_pixel_counts[region_name] = np.sum(light_mask)
        white_pixel_counts[region_name] = np.sum(white_mask)
        pastel_pixel_counts[region_name] = np.sum(pastel_mask)
    
    # Calculate overall percentages
    total_light_pixels = sum(light_pixel_counts.values())
    total_white_pixels = sum(white_pixel_counts.values())
    total_pastel_pixels = sum(pastel_pixel_counts.values())
    
    light_percentage = total_light_pixels / total_pixels if total_pixels > 0 else 0
    white_percentage = total_white_pixels / total_pixels if total_pixels > 0 else 0
    pastel_percentage = total_pastel_pixels / total_pixels if total_pixels > 0 else 0
    
    # Determine background type
    if white_percentage > threshold_percentage:
        background_type = 'white'
        confidence = min(white_percentage * 2, 1.0)  # Scale confidence
        reason = f"White pixels: {white_percentage:.2%}"
    elif light_percentage > threshold_percentage:
        background_type = 'pastel'
        confidence = min(light_percentage * 1.5, 1.0)  # Scale confidence
        reason = f"Light/pastel pixels: {light_percentage:.2%}"
    else:
        background_type = 'dark'
        confidence = 1.0 - light_percentage
        reason = f"Dark pixels: {1-light_percentage:.2%}"
    
    return {
        'background_type': background_type,
        'confidence': confidence,
        'reason': reason,
        'white_percentage': white_percentage,
        'pastel_percentage': pastel_percentage,
        'light_percentage': light_percentage
    }

def create_visualizations(results_df):
    """
    Create visualizations of the analysis results.
    
    Args:
        results_df (pandas.DataFrame): Results DataFrame
    """
    try:
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Background Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(results_df['human_annotation'], results_df['predicted_background'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Dark', 'Light'], yticklabels=['Dark', 'Light'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Background Type Distribution
        type_counts = results_df['background_type'].value_counts()
        axes[0, 1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Background Type Distribution')
        
        # 3. Confidence Distribution
        axes[1, 0].hist(results_df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Confidence Distribution')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Accuracy by Background Type
        accuracy_by_type = results_df.groupby('background_type').apply(
            lambda x: accuracy_score(x['human_annotation'], x['predicted_background'])
        )
        axes[1, 1].bar(accuracy_by_type.index, accuracy_by_type.values, color='lightgreen')
        axes[1, 1].set_title('Accuracy by Background Type')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(accuracy_by_type.values):
            axes[1, 1].text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('background_analysis_plots.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved to: background_analysis_plots.png")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def analyze_all_images(csv_filename="data.csv", images_folder="images")->pd.DataFrame:
    """
    Analyze all images and compare with human annotations.
    
    Args:
        csv_filename (str): Path to the CSV file
        images_folder (str): Path to the images folder
    """
    # Check if files exist
    if not os.path.exists(csv_filename):
        print(f"Error: CSV file '{csv_filename}' not found!")
        return
    
    if not os.path.exists(images_folder):
        print(f"Error: Images folder '{images_folder}' not found!")
        return
    
    try:
        # Read CSV file
        print(f"Reading CSV file: {csv_filename}")
        df = pd.read_csv(csv_filename)
        print(df.columns)
        
        # Check if required columns exist
        required_columns = ['item_id', 'site_id', 'domain_id', 'picture_id', 'correct_background?']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return
        
        # Initialize results
        results = []
        analyzed_count = 0
        missing_images = 0
        
        print("Analyzing images...")
        print("-" * 80)
        
        for index, row in df.iterrows():
            picture_id = row['picture_id']
            correct_background = row['correct_background?']
            
            # Skip if picture_id is NaN or empty
            if pd.isna(picture_id) or str(picture_id).strip() == '':
                continue
            
            # Construct image path
            image_filename = f"D_{picture_id}-F.jpg"
            image_path = os.path.join(images_folder, image_filename)
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"⚠  Missing image: {image_filename}")
                missing_images += 1
                continue
            
            # Load and analyze image
            image = load_image(image_path)
            analysis = analyze_background_color(image)
            
            # Determine predicted background (True for white/pastel, False for dark)
            predicted_background = analysis['background_type'] in ['white', 'pastel']
            
            # Store results
            result = {
                'item_id': row['item_id'],
                'picture_id': picture_id,
                'image_filename': image_filename,
                'human_annotation': correct_background,
                'predicted_background': predicted_background,
                'background_type': analysis['background_type'],
                'confidence': analysis['confidence'],
                'reason': analysis['reason'],
                'white_percentage': analysis['white_percentage'],
                'pastel_percentage': analysis['pastel_percentage'],
                'light_percentage': analysis['light_percentage']
            }
            results.append(result)
            
            # Print progress
            status = "✓" if predicted_background == correct_background else "✗"
            print(f"{status} {image_filename}: {analysis['background_type']} ({analysis['confidence']:.2f}) - Human: {correct_background}")
            
            analyzed_count += 1
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)

        results_df2 = results_df.copy()
        
        # Calculate metrics
        if len(results_df) > 0:
            accuracy = accuracy_score(results_df['human_annotation'], results_df['predicted_background'])
            
            print("\n" + "=" * 80)
            print("ANALYSIS RESULTS")
            print("=" * 80)
            print(f"Total images analyzed: {analyzed_count}")
            print(f"Missing images: {missing_images}")
            print(f"Overall accuracy: {accuracy:.2%}")
            
            # Detailed classification report
            print("\nClassification Report:")
            print(classification_report(results_df['human_annotation'], results_df['predicted_background'], 
                                     target_names=['Dark Background', 'Light Background']))
            
            # Background type distribution
            print("\nBackground Type Distribution:")
            type_counts = results_df['background_type'].value_counts()
            for bg_type, count in type_counts.items():
                print(f"  {bg_type}: {count} ({count/len(results_df):.1%})")
            
            # Confidence analysis
            print(f"\nAverage confidence: {results_df['confidence'].mean():.2f}")
            print(f"Confidence range: {results_df['confidence'].min():.2f} - {results_df['confidence'].max():.2f}")
            
            # Save detailed results
            output_filename = "../../data/processed/background_analysis_results.csv"
            results_df.to_csv(output_filename, index=False)
            print(f"\nDetailed results saved to: {output_filename}")
            
            # Create visualizations
            create_visualizations(results_df)
        
            return results_df2, results_df2

        else:
            print("No images were successfully analyzed!")
            
    except Exception as e:
        print(f"Error during analysis: {e}")