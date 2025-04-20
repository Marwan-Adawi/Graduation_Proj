import numpy as np
import cv2
import matplotlib.pyplot as plt


def scan_transform_image(image):
    # Convert PIL image to NumPy array
    image = np.array(image)

    # Convert to RGB if the image is in BGR format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

    # Get image dimensions
    height, width = image.shape[:2]
    
    # Define source points (corners of the original image)
    pts = np.array([
        [0, 0],             # top-left
        [width-1, 0],       # top-right
        [width-1, height-1],# bottom-right
        [0, height-1]       # bottom-left
    ], dtype="float32")

    # Define target size with minimum width of 2000px while preserving aspect ratio
    output_width = max(width, 2000)  
    output_height = int(output_width * (height/width))  

    # Define destination points (corners of the output image)
    dst = np.array([
        [0, 0],
        [output_width-1, 0],
        [output_width-1, output_height-1],
        [0, output_height-1]
    ], dtype="float32")

    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(pts, dst)
    
    # Apply perspective transformation with cubic interpolation
    color_scanned = cv2.warpPerspective(image, M, (output_width, output_height), flags=cv2.INTER_CUBIC)
    return color_scanned