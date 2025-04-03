"""Utility functions for graph operations in GraphDiffLayout."""

import networkx as nx
import numpy as np

def create_layout_graph(layout, threshold=50):
    """
    Create a graph from a given layout.
    
    Args:
        layout: list of dicts, each dict contains {'bbox': [x, y, w, h], 'label': str}
        threshold: maximum Euclidean distance between centers for an edge to be added.
        
    Returns:
        G: networkx Graph with nodes and edges representing the layout
    """
    G = nx.Graph()
    
    for idx, obj in enumerate(layout):
        G.add_node(idx, label=obj['label'], bbox=obj['bbox'])
    
    for i in range(len(layout)):
        for j in range(i+1, len(layout)):
            bbox_i = np.array(layout[i]['bbox'])
            bbox_j = np.array(layout[j]['bbox'])
            center_i = np.array([bbox_i[0]+bbox_i[2]/2, bbox_i[1]+bbox_i[3]/2])
            center_j = np.array([bbox_j[0]+bbox_j[2]/2, bbox_j[1]+bbox_j[3]/2])
            dist = np.linalg.norm(center_i - center_j)
            if dist < threshold:
                G.add_edge(i, j, weight=1/dist)
    
    return G

def compute_box_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) for two boxes.
    Boxes are in [x, y, w, h] format.
    
    Args:
        boxA: First bounding box [x, y, w, h]
        boxB: Second bounding box [x, y, w, h]
        
    Returns:
        iou: Intersection over Union score
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    
    return iou
