# Arabic OCR Correction System - Installation Guide

This guide will walk you through the process of setting up the Arabic OCR Correction System on your machine.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Installation Steps

### 1. Clone or Download the Repository (Optional)

If you haven't already downloaded the project:

```bash
git https://github.com/Marwan-Adawi/Graduation_Proj.git

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
- opencv-python (for image processing)
- numpy (for numerical operations)
- scikit-image (for image processing)
- tensorflow (for machine learning)
- torch (for deep learning)
- transformers (for NLP models)
- ultralytics (for YOLO object detection)
- and other dependencies

### 4. Fix Common Installation Issues

If you encounter warnings about urllib3 or ipywidgets:

```bash
# Fix ipywidgets warning
pip install ipywidgets --upgrade

# Fix LibreSSL warning (if on macOS)
pip install urllib3==1.26.15
```

### 5. Verify Installation

To verify that all dependencies are correctly installed:

```bash
python -c "import torch; import transformers; import tensorflow as tf; print('Installation successful!')"
```

## Running the Notebooks

After installation, you can run the Jupyter notebooks:

```bash
jupyter notebook
```

Navigate to `ArabicOCRCorrection.ipynb` or `OCR.ipynb` to start working with the system.

## Troubleshooting

- **Memory Issues**: If you encounter memory errors when loading models, try using a smaller model or enabling mo
del offloading.
- **CUDA Errors**: If using NVIDIA GPUs and encountering CUDA errors, ensure you have compatible CUDA drivers installed.
- **MPS Issues**: On Apple Silicon Macs, ensure you have the latest version of PyTorch that supports MPS acceleration.

For more detailed information, refer to the documentation of each library or open an issue in the project repository.