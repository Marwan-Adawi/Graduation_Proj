
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

def preprocess_image(path):

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  


    height, width = image.shape[:2]
    pts = np.array([
        [0, 0],             # top-left
        [width-1, 0],       # top-right
        [width-1, height-1],# bottom-right
        [0, height-1]       # bottom-left
    ], dtype="float32")


    output_width = max(width, 2000)  
    output_height = int(output_width * (height/width))  

    dst = np.array([
        [0, 0],
        [output_width-1, 0],
        [output_width-1, output_height-1],
        [0, output_height-1]
    ], dtype="float32")


    M = cv2.getPerspectiveTransform(pts, dst)
    color_scanned = cv2.warpPerspective(image, M, (output_width, output_height), flags=cv2.INTER_CUBIC)
    return color_scanned

