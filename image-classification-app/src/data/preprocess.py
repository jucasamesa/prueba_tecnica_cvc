import pandas as pd
import requests
import os
from urllib.parse import urljoin
import time
from pathlib import Path

def download_image(url, filepath, max_retries=3):
    """
    Download an image from URL and save it to the specified filepath.
    
    Args:
        url (str): The URL to download the image from
        filepath (str): The local filepath to save the image
        max_retries (int): Maximum number of retry attempts
    
    Returns:
        bool: True if download successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            #print(f"✓ Downloaded: {os.path.basename(filepath)}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retrying
            else:
                print(f"✗ Failed to download after {max_retries} attempts: {url}")
                return False

def saving_images_in_local(path_csv, image_folder):
    """
    Main function to read CSV and download images.
    """
    # Configuration
    base_url = "https://http2.mlstatic.com/D_{picture_id}-F.jpg"
    
    # Check if CSV file exists
    if not os.path.exists(path_csv):
        print(f"Error: CSV file '{path_csv}' not found!")
        return
    
    try:
        # Read CSV file
        print(f"Reading CSV file: {path_csv}")
        df = pd.read_csv(path_csv, dtype=str)
        
        # Check if required columns exist
        required_columns = ['item_id', 'site_id', 'domain_id', 'picture_id', 'correct_background?']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Create images folder
        os.makedirs(image_folder, exist_ok=True)
        
        # Get unique picture_ids to avoid duplicates
        unique_picture_ids = df['picture_id'].dropna().unique()
        total_images = len(unique_picture_ids)
        
        print(f"Found {total_images} unique images to download")
        print(f"Images will be saved to: {image_folder}/")
        print("-" * 50)
        
        # Download images
        successful_downloads = 0
        failed_downloads = 0
        
        for i, picture_id in enumerate(unique_picture_ids, 1):
            # Skip if picture_id is NaN or empty
            if pd.isna(picture_id) or str(picture_id).strip() == '':
                continue
            
            # Create URL
            url = base_url.format(picture_id=picture_id)
            
            # Create filename
            filename = f"D_{picture_id}-F.jpg"
            filepath = os.path.join(image_folder, filename)
            
            # Skip if file already exists
            if os.path.exists(filepath):
                print(f"⏭  Skipped (already exists): {filename}")
                successful_downloads += 1
                continue
            
            #print(f"[{i}/{total_images}] Downloading: {filename}")
            
            # Download the image
            if download_image(url, filepath):
                successful_downloads += 1
            else:
                failed_downloads += 1
            
            # Small delay to be respectful to the server
            time.sleep(0.1)
        
        # Summary
        print("-" * 50)
        print("Download Summary:")
        print(f"Total images processed: {total_images}")
        print(f"Successfully downloaded: {successful_downloads}")
        print(f"Failed downloads: {failed_downloads}")
        print(f"Images saved in: {os.path.abspath(image_folder)}")
        
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty!")
    except pd.errors.ParserError as e:
        print(f"Error: Could not parse CSV file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# from PIL import Image
# import requests
# from io import BytesIO
# import os

# def download_image_with_pillow(url: str, save_path: str) -> bool:
#     """
#     Downloads an image from a URL and saves it using Pillow.
    
#     Args:
#         url (str): Static URL of the image.
#         save_path (str): Local path to save the image (e.g., "/folder/image.jpg").
    
#     Returns:
#         bool: True if successful, False otherwise.
#     """
#     try:
#         # Fetch the image
#         response = requests.get(url, stream=True, timeout=10)
#         print(response)
#         response.raise_for_status()  # Raise HTTP errors
        
#         # Open with Pillow to validate it's an image
#         img = Image.open(BytesIO(response.content))
        
#         # Create directory if it doesn't exist
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
#         # Save the image (format inferred from extension)
#         img.save(save_path)
#         print(f"Image saved to {save_path}")
#         return True
        
#     except requests.exceptions.RequestException as e:
#         print(f"Download failed (HTTP/network error): {e}")
#     except IOError as e:
#         print(f"Failed to process/save image: {e}")
#     except Exception as e:
#         print(f"Unexpected error: {e}")
    
#     return False

# def load_data(file_path):
#     """Load the raw training data from a CSV file."""
#     data = pd.read_csv(file_path)
#     return data

# def clean_data(data):
#     """Clean the data by handling missing values and duplicates."""
#     data = data.dropna()  # Remove rows with missing values
#     data = data.drop_duplicates()  # Remove duplicate rows
#     return data

# def transform_data(data):
#     """Transform the data into a suitable format for training."""
#     # Example transformation: Convert categorical variables to numerical
#     data['site_id'] = data['site_id'].astype('category').cat.codes
#     data['domain_id'] = data['domain_id'].astype('category').cat.codes
#     return data

# def preprocess_data(raw_file_path, processed_file_path):
#     """Main function to preprocess the data."""
#     raw_data = load_data(raw_file_path)
#     cleaned_data = clean_data(raw_data)
#     processed_data = transform_data(cleaned_data)
#     processed_data.to_csv(processed_file_path, index=False)  # Save the processed data

# if __name__ == "__main__":
#     preprocess_data('data/raw/training_data.csv', 'data/processed/preprocessed_data.csv')