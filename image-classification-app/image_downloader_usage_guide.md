# Image Downloader Usage Guide

The `image_downloader.py` script has been updated to accept custom paths and datasets. Here's how to use it:

## 🎯 Your Specific Use Case

To download images from `data/productive_data.csv` to `data/validation/images`:

```bash
python image_downloader.py \
  --training-data "data/productive_data.csv" \
  --images-dir "data/validation_images" \
  --output-dir "data" \
  --output-name "productive_images_dataset.csv"
```

## 📋 Command Line Options

### Required Arguments (Optional - uses defaults if not provided)
- `--training-data`: Path to the CSV file with image data (default: `data/training_data.csv`)
- `--images-dir`: Directory where to save downloaded images (default: `data/images/`)
- `--output-dir`: Directory where to save the output CSV (default: `data/`)

### Optional Arguments
- `--output-name`: Name of the output CSV file (default: `downloaded_{input_name}_dataset.csv`)
- `--limit`: Limit number of images to process (useful for testing)
- `--test`: Test mode - downloads only 10 images

## 🚀 Usage Examples

### 1. Default Usage (Original Behavior)
```bash
python image_downloader.py
```
Downloads from `data/training_data.csv` to `data/images/` and saves results to `data/downloaded_training_data_dataset.csv`

### 2. Custom Dataset and Folders
```bash
python image_downloader.py \
  --training-data "data/productive_data.csv" \
  --images-dir "data/validation/images" \
  --output-dir "data/validation" \
  --output-name "productive_images_dataset.csv"
```

### 3. Test Mode with Custom Paths
```bash
python image_downloader.py \
  --training-data "data/productive_data.csv" \
  --images-dir "data/validation/images" \
  --test
```

### 4. Limited Download
```bash
python image_downloader.py \
  --training-data "data/productive_data.csv" \
  --images-dir "data/validation/images" \
  --limit 100
```

### 5. Different Project Structure
```bash
python image_downloader.py \
  --training-data "experiments/exp1/dataset.csv" \
  --images-dir "experiments/exp1/images/" \
  --output-dir "experiments/exp1/results/" \
  --output-name "experiment1_results.csv"
```

## 🔧 Programmatic Usage

You can also use the `ImageDownloader` class directly in your Python code:

```python
from image_downloader import ImageDownloader

# Custom paths
downloader = ImageDownloader(
    training_data_path="data/productive_data.csv",
    images_dir="data/validation/images",
    output_dir="data/validation"
)

# Run with custom output filename
output_file = downloader.run(
    limit=50,  # Optional limit
    output_filename="productive_images_dataset.csv"
)

print(f"Images downloaded to: {downloader.images_dir}")
print(f"Results saved to: {output_file}")
```

## 📊 Output Structure

The script creates:
1. **Images**: Downloaded to the specified `--images-dir`
2. **CSV File**: Contains all the original data plus download information:
   - `download_url`: The URL that was downloaded
   - `local_path`: Relative path to the downloaded image
   - `filename`: Name of the downloaded file
   - `download_success`: Whether the download was successful
   - `error_message`: Error message if download failed
   - `file_exists`: Whether the file already existed

## ✅ Success Indicators

- **Images downloaded**: Check the specified `--images-dir` folder
- **CSV created**: Check the specified `--output-dir` for the results CSV
- **Console output**: Shows progress and summary statistics

## 🎉 Ready for Background Extraction

After downloading, you can run background extraction on the new images:

```bash
# Modify image_bg_extraction.py to point to your new images folder
# Or create a new script that accepts custom paths
python image_bg_extraction.py  # (after modifying the paths)
``` 