"""Metrics for evaluating segmentation results."""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

def compute_mIoU(pred_mask, gt_mask):
    """
    Compute the mean Intersection-over-Union (mIoU) between the predicted and ground truth masks.
    
    Args:
        pred_mask (numpy.ndarray): Predicted binary mask
        gt_mask (numpy.ndarray): Ground truth binary mask
        
    Returns:
        float: mIoU score
    """
    pred = pred_mask.astype(bool)
    gt = np.array(gt_mask, dtype=bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return intersection / union

def compute_boundary_f1(pred_mask, gt_mask):
    """
    Compute a boundary F1 score based on morphological edge detection.
    
    Args:
        pred_mask (numpy.ndarray): Predicted binary mask
        gt_mask (numpy.ndarray): Ground truth binary mask
        
    Returns:
        float: Boundary F1 score
    """
    pred = pred_mask.astype(bool)
    gt = np.array(gt_mask, dtype=bool)
    pred_edge = np.logical_xor(binary_dilation(pred), binary_erosion(pred))
    gt_edge = np.logical_xor(binary_dilation(gt), binary_erosion(gt))
    intersection = np.logical_and(pred_edge, gt_edge).sum()
    union = np.logical_or(pred_edge, gt_edge).sum()
    if union == 0:
        return 1.0
    return intersection / union
