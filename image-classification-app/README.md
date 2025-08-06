# Image Classification App

This project is an image processing and classification application focused on background removal and image analysis. The current implementation includes automated background extraction from product images using AI-powered tools, with plans for future machine learning model development.

## Project Structure

```
image-classification-app/
├── data/
│   ├── images/                           # Source images for processing
│   ├── processed/                        # Processed data and CSV files
│   │   ├── background_masks_data.csv    # Background mask statistics
│   │   └── detailed_background_masks.csv # Detailed mask arrays (sample)
│   ├── processed_images/                 # Images with backgrounds removed
│   ├── eda_results/                      # Exploratory data analysis results
│   └── training_data.csv                 # Original training data
├── eda_modules/                          # Exploratory data analysis modules
│   ├── eda_atomic.py
│   ├── eda_full.py
│   ├── image_analyzer.py
│   ├── segmentation.py
│   └── example.ipynb
├── notebooks/
│   ├── exploratory.ipynb                 # Data exploration notebook
│   └── prueba_tecnica_cvc.ipynb         # Technical test notebook
├── image_bg_extraction.py                # Main background removal script
├── image_downloader.py                   # Image downloading utilities
├── utils.py                              # General utility functions
├── config.py                             # Configuration settings
├── requirements.txt                      # Project dependencies
├── setup.py                              # Package setup configuration
└── README.md
```

## Data

- **Source Images**: Product images for processing are located in `data/images/`. These include various product photos from the MercadoLibre dataset.
- **Processed Images**: Images with backgrounds removed are saved in `data/processed_images/` with the prefix `no_background_`.
- **Background Mask Data**: Statistical information about background masks is stored in `data/processed/background_masks_data.csv`.
- **Training Data**: The original dataset is available in `data/training_data.csv` for future model development.

## Notebooks

- **Exploratory Data Analysis**: The Jupyter notebook `notebooks/exploratory.ipynb` contains data exploration and visualization.
- **Technical Test**: The notebook `notebooks/prueba_tecnica_cvc.ipynb` includes specific technical analysis and testing.

## Main Components

- **Background Extraction**: The `image_bg_extraction.py` script processes all images in `data/images/` to remove backgrounds using the rembg library. Features include:
  - Batch processing with progress tracking (tqdm)
  - Automatic directory creation for outputs
  - CSV export of background mask statistics
  - Error handling and processing summaries

- **Image Downloader**: The `image_downloader.py` script handles downloading and organizing product images from external sources.

- **EDA Modules**: Located in `eda_modules/`, these provide specialized tools for:
  - Atomic-level image analysis (`eda_atomic.py`)
  - Full image analysis workflows (`eda_full.py`)
  - Image segmentation (`segmentation.py`)
  - Interactive image analysis (`image_analyzer.py`)

- **Utilities**: The `utils.py` and `config.py` files contain helper functions and configuration settings for the project.

## Requirements

The project dependencies are listed in `requirements.txt`. Key libraries include:
- `rembg[cpu,cli]` - AI-powered background removal
- `opencv-python` - Computer vision operations
- `pillow` - Image processing
- `tqdm` - Progress tracking
- `pandas` - Data manipulation
- `scikit-image` - Advanced image processing
- `seaborn` - Data visualization

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up the project:
```bash
python setup.py install
```

## Usage

### Background Removal Processing

To process all images and remove backgrounds:

```bash
python image_bg_extraction.py
```

This will:
- Process all images in `data/images/`
- Save processed images (without backgrounds) to `data/processed_images/`
- Generate background mask statistics in `data/processed/background_masks_data.csv`
- Show real-time progress with tqdm progress bars

### Exploratory Data Analysis

Use the Jupyter notebooks in the `notebooks/` directory for data exploration and analysis.

## Future Development

This project is designed to support future machine learning model development for image classification. The current preprocessing and background removal pipeline provides a solid foundation for training classification models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.