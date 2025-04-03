"""Data preprocessing for GraphDiffLayout experiments."""

import numpy as np
import random
import cv2
from src.utils.graph_utils import create_layout_graph

def generate_random_layout(n_objects, image_size=(256, 256), seed=None):
    """
    Generate a random layout with n_objects.
    
    Args:
        n_objects: number of objects to generate
        image_size: size of the image as (width, height)
        seed: random seed for reproducibility
        
    Returns:
        layout: list of dictionaries with 'bbox' and 'label' keys
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    layout = []
    for i in range(n_objects):
        w, h = random.randint(20, 50), random.randint(20, 50)
        x = random.randint(0, image_size[0]-w)
        y = random.randint(0, image_size[1]-h)
        layout.append({'bbox': [x, y, w, h], 'label': f'object_{i}'})
    
    return layout

def create_test_layouts():
    """
    Create test layouts for experiments.
    
    Returns:
        dict: Dictionary containing different layouts for experiments
    """
    layout_crowded = [
        {'bbox': [30, 30, 60, 60], 'label': 'table'},
        {'bbox': [50, 50, 60, 60], 'label': 'box'},
        {'bbox': [70, 70, 60, 60], 'label': 'bottle'}
    ]
    
    layout_small = [
        {'bbox': [20, 20, 20, 20], 'label': 'icon1'},
        {'bbox': [60, 30, 15, 15], 'label': 'icon2'},
        {'bbox': [100, 50, 18, 18], 'label': 'icon3'}
    ]
    
    layout_overlap = [
        {'bbox': [40, 40, 50, 50], 'label': 'cup'},
        {'bbox': [70, 60, 50, 50], 'label': 'saucer'},
        {'bbox': [90, 80, 50, 50], 'label': 'spoon'}
    ]
    
    return {
        'crowded': layout_crowded,
        'small': layout_small,
        'overlap': layout_overlap
    }

def generate_ground_truth(layout, image_size=(256, 256)):
    """
    Generate a ground truth image by drawing boxes.
    
    Args:
        layout: list of dictionaries with 'bbox' and 'label' keys
        image_size: size of the image as (width, height)
        
    Returns:
        image: numpy array image with drawn boxes
    """
    image = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255
    for obj in layout:
        x, y, w, h = obj['bbox']
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 0), 2)
    
    return image
