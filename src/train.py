"""Model training for GraphDiffLayout experiments."""

import numpy as np
import cv2
from src.utils.graph_utils import create_layout_graph

def dummy_generate_image_noise_collage(layout, image_size=(256, 256)):
    """
    Simulate the NoiseCollage pipeline by generating a blank white image and drawing
    each bounding box with a blue rectangle.
    
    Args:
        layout: list of dictionaries with 'bbox' and 'label' keys
        image_size: size of the image as (width, height)
        
    Returns:
        image: numpy array image with drawn boxes
    """
    image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
    for obj in layout:
        x, y, w, h = obj['bbox']
        top_left = (int(x), int(y))
        bottom_right = (int(x+w), int(y+h))
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
        cv2.putText(image, obj['label'], (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 0, 0), 1)
    
    return image

def dummy_generate_image_graph_diff_layout(layout, image_size=(256, 256), threshold=50):
    """
    Simulate the GraphDiffLayout pipeline by generating a blank white image and drawing
    each bounding box with a green rectangle. Additionally, overlay the computed graph connections.
    
    Args:
        layout: list of dictionaries with 'bbox' and 'label' keys
        image_size: size of the image as (width, height)
        threshold: maximum distance for edge creation
        
    Returns:
        image: numpy array image with drawn boxes and graph connections
    """
    image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
    
    centers = []
    for obj in layout:
        x, y, w, h = obj['bbox']
        top_left = (int(x), int(y))
        bottom_right = (int(x+w), int(y+h))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, obj['label'], (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 1)
        centers.append((int(x + w/2), int(y + h/2)))
    
    G = create_layout_graph(layout, threshold=threshold)
    for edge in G.edges():
        pt1 = centers[edge[0]]
        pt2 = centers[edge[1]]
        cv2.line(image, pt1, pt2, (0, 0, 255), 1)
    
    return image
