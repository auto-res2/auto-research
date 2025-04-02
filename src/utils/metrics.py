"""
Metrics for evaluating the PTDA model.
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def compute_ssim(reference_frames, generated_frames):
    """
    Compute Structural Similarity Index (SSIM) between corresponding frames.
    
    Args:
        reference_frames: List of reference frames (numpy arrays)
        generated_frames: List of generated frames (numpy arrays)
        
    Returns:
        float: Mean SSIM value across all frame pairs
    """
    ssim_values = []
    for ref, gen in zip(reference_frames, generated_frames):
        min_dim = min(ref.shape[0], ref.shape[1])
        win_size = min(7, min_dim - (min_dim % 2) + 1) if min_dim < 7 else 7
        
        try:
            ssim, _ = compare_ssim(ref, gen, full=True, win_size=win_size, channel_axis=2)
        except TypeError:
            ssim, _ = compare_ssim(ref, gen, full=True, win_size=win_size, multichannel=True)
        
        ssim_values.append(ssim)
    return np.mean(ssim_values) if ssim_values else 0.0


def compute_psnr(reference_frames, generated_frames):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between corresponding frames.
    
    Args:
        reference_frames: List of reference frames (numpy arrays)
        generated_frames: List of generated frames (numpy arrays)
        
    Returns:
        float: Mean PSNR value across all frame pairs
    """
    psnr_values = []
    for ref, gen in zip(reference_frames, generated_frames):
        psnr = compare_psnr(ref, gen)
        psnr_values.append(psnr)
    return np.mean(psnr_values)


def compute_temporal_consistency(frames):
    """
    Compute an optical flow-based temporal consistency metric.
    
    Args:
        frames: List of frames (numpy arrays)
        
    Returns:
        float: Temporal consistency score
    """
    flow_diffs = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_magnitude = np.linalg.norm(flow, axis=2)
        flow_diffs.append(np.mean(flow_magnitude))
        prev_gray = gray
    return np.mean(flow_diffs)


def compute_temporal_consistency_long(frames, window_size=3):
    """
    Compute average optical flow variance over a sliding window across the sequence.
    
    Args:
        frames: List of frames (numpy arrays)
        window_size: Size of the sliding window
        
    Returns:
        float: Long-range temporal consistency score
    """
    consistency_scores = []
    for i in range(len(frames) - window_size):
        flows = []
        for j in range(i, i+window_size-1):
            prev_gray = cv2.cvtColor(frames[j], cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(frames[j+1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray,
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(flow)
        flow_magnitudes = [np.linalg.norm(f, axis=2) for f in flows]
        flow_variance = np.var(np.stack(flow_magnitudes, axis=0))
        consistency_scores.append(flow_variance)
    return np.mean(consistency_scores) if consistency_scores else 0.0
