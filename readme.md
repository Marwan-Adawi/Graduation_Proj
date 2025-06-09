# Arabic OCR Correction System - Installation Guide

This guide will walk you through the process of setting up the Arabic OCR Correction System on your machine.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)
- Poppler (required for PDF processing)

## Installation Steps

### 1. Clone or Download the Repository (Optional)

If you haven't already downloaded the project:

```bash
git clone https://github.com/Marwan-Adawi/Graduation_Proj.git
```

### 2. Create a Virtual Environment (Recommended)

Creating a virtual environment helps isolate the project dependencies:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

The project dependencies are listed in the `requirements.txt` file. Install them using pip:

```bash
pip install -r requirements.txt
```

This will install all the required libraries, including:
- torch and transformers (for deep learning models)
- opencv-python and scikit-image (for image processing)
- pdf2image (for PDF processing)
- numpy (for numerical operations)
- ultralytics (for object detection)
- and other dependencies

### 4. Install Poppler

Poppler is required for PDF processing:

**MacOS**
```bash
brew install poppler
```

**Linux**
```bash
sudo apt-get install poppler-utils
```

**Windows**
Download and install Poppler from the official website: https://poppler.freedesktop.org/

### 5. Fix Common Installation Issues

If you encounter warnings about urllib3 or ipywidgets:

```bash
# Fix ipywidgets warning
pip install ipywidgets --upgrade

# Fix LibreSSL warning (if on macOS)
pip install urllib3==1.26.15
```

## Using the OCR Script

The system includes a command-line script for processing PDF files with Arabic OCR:

```bash
python ocr_script.py --input your_file.pdf --output results.txt --transform --verbose
```

### Command Line Arguments

- `--input`, `-i`: Input PDF file path (default: test_OCR_2025.pdf)
- `--output`, `-o`: Output file path (auto-generated if not specified)
- `--transform`, `-t`: Apply scan transformation to images before OCR
- `--format`: Output format (txt or json, default: txt)
- `--verbose`, `-v`: Enable detailed progress output

## System Components

- **ocr_script.py**: Main script for processing PDFs with Arabic OCR
- **ocr_functions.py**: Utility functions for PDF processing and image transformation
- **Qariv03.py**: Arabic OCR implementation using deep learning models

## Troubleshooting

- **Memory Issues**: If you encounter memory errors when loading models, try using a smaller model or enabling model offloading.
- **CUDA Errors**: If using NVIDIA GPUs and encountering CUDA errors, ensure you have compatible CUDA drivers installed.
- **MPS Issues**: On Apple Silicon Macs, ensure you have the latest version of PyTorch that supports MPS acceleration.

For more detailed information, refer to the documentation of each library or open an issue in the project repository.
