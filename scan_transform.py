import numpy as np
import cv2


def scan_transform_image(image):
    """
    Transform image to A4 scanned document format with robust dimension handling
    """
    # Convert PIL image to NumPy array
    image = np.array(image)

    # Convert to RGB if the image is in BGR format
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

    # Get image dimensions
    height, width = image.shape[:2]
    
    print(f"Input image dimensions: {width}x{height}")
    
    # Check for extremely narrow or invalid images
    if width < 10 or height < 10:
        raise ValueError(f"Input image too small: {width}x{height}. Minimum size is 10x10.")
    
    # Define minimum dimensions that work with the model (multiples of 28)
    min_width = 28 * 30   # 840
    min_height = 28 * 42  # 1176
    
    # A4 aspect ratio: 210:297 ≈ 0.707 (width:height)
    a4_ratio = 210 / 297
    
    # Start with original dimensions and scale up to meet minimums
    scale_factor = max(min_width / width, min_height / height, 1.0)
    
    # Apply scale factor
    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)
    
    # Adjust to A4 ratio while maintaining minimums
    if scaled_width / scaled_height > a4_ratio:
        # Too wide for A4 - increase height
        output_width = scaled_width
        output_height = max(int(scaled_width / a4_ratio), min_height)
    else:
        # Too tall for A4 - increase width
        output_height = scaled_height
        output_width = max(int(scaled_height * a4_ratio), min_width)
    
    # Ensure dimensions are multiples of 28
    factor = 28
    output_width = ((output_width + factor - 1) // factor) * factor
    output_height = ((output_height + factor - 1) // factor) * factor
    
    # Final safety check
    output_width = max(output_width, min_width)
    output_height = max(output_height, min_height)
    
    print(f"Output dimensions: {output_width}x{output_height}")
    
    # Verify the output dimensions meet requirements
    if output_width < 28 or output_height < 28:
        raise ValueError(f"Failed to create valid dimensions: {output_width}x{output_height}")
    
    # Define source points (corners of the original image)
    pts = np.array([
        [0, 0],             # top-left
        [width-1, 0],       # top-right
        [width-1, height-1],# bottom-right
        [0, height-1]       # bottom-left
    ], dtype="float32")

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
    
    # Final verification of output
    final_height, final_width = color_scanned.shape[:2]
    print(f"Final output verified: {final_width}x{final_height}")
    
    if final_width < 28 or final_height < 28:
        raise ValueError(f"Transform failed - output too small: {final_width}x{final_height}")
    
    return color_scanned