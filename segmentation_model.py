import tensorflow as tf
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt


tf_savedmodel_model = YOLO("best.pt",task='detect')

def segment_page(image, image_id=None):
    print("Starting image processing...")

    # Create a copy of the original image to apply the mask
    processed_image = image.copy()

    # Create a blank mask with same dimensions as image
    mask = np.zeros(image.shape, dtype=np.uint8)
    # Fill mask with white
    mask[:] = (255, 255, 255)  # White color for all channels

    # Dictionary to store all image data
    images_dict = {}

    # Predict with lower confidence threshold
    results = tf_savedmodel_model.predict(image, verbose=True)
    result = results[0]

    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create figure with larger size and clear any existing plots
    plt.clf()
    plt.figure(figsize=(15, 10))

    # Display the image
    plt.imshow(image_rgb)

    # Create a binary mask for detected objects
    binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Store current image data
    current_image_data = {
        "boxes": [],
        "binary_mask": None,
        "masked_image": None
    }

    if len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = box.astype(int)

            # Add to binary mask - fill with white (255)
            cv2.rectangle(binary_mask, (x1, y1), (x2, y2), 255, -1)  # -1 means filled

            # Store box info
            box_info = {
                "coordinates": [x1, y1, x2, y2],
                "class": int(cls),
                "confidence": float(conf)
            }
            current_image_data["boxes"].append(box_info)

            # Draw rectangle with thicker lines
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)

            # Add label with larger font
            label = f"Class: {int(cls)}, {conf:.2f}"
            plt.text(x1, y1-10, label,
                    color='red',
                    fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.8))

    # Create the final masked image:
    # 1. Where binary_mask is 255 (objects), use the white mask
    # 2. Where binary_mask is 0 (background), use the original image
    for c in range(0, 3):  # For each color channel
        processed_image[:, :, c] = np.where(binary_mask == 255,
                                         mask[:, :, c],
                                         processed_image[:, :, c])

    # Store masks and processed image in the data
    current_image_data["binary_mask"] = binary_mask
    current_image_data["masked_image"] = processed_image

    # Store image data in dictionary if image_id is provided
    if image_id is not None:
        images_dict[image_id] = current_image_data

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.show()

    # Show the masked image
    plt.figure(figsize=(15, 10))
    masked_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    plt.imshow(masked_rgb)
    plt.title(f"White Masked Objects - {image_id}")
    plt.axis('off')
    plt.show()

    print("Processing complete.")

    # Return the images dictionary
    return processed_image,images_dict
