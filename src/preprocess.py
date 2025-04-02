"""
Data preprocessing for the PTDA experiments.
"""

import os
import cv2
import numpy as np
import torch
from src.utils.data import load_video_frames, create_dummy_video, create_dummy_frame


def preprocess_video(video_path, output_dir, num_frames=None, resize=None):
    """
    Preprocess a video by extracting frames and optionally resizing them.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the preprocessed frames
        num_frames: Maximum number of frames to extract (None for all frames)
        resize: Optional tuple (width, height) to resize frames
        
    Returns:
        List of paths to the saved frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    frames = load_video_frames(video_path, num_frames)
    
    frame_paths = []
    for i, frame in enumerate(frames):
        if resize is not None:
            frame = cv2.resize(frame, resize)
        
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)
    
    return frame_paths


def prepare_dummy_data(data_dir, num_videos=2, num_frames=10, height=256, width=256):
    """
    Prepare dummy data for the experiments.
    
    Args:
        data_dir: Directory to save the dummy data
        num_videos: Number of dummy videos to create
        num_frames: Number of frames per video
        height: Frame height
        width: Frame width
        
    Returns:
        List of paths to the dummy videos
    """
    os.makedirs(data_dir, exist_ok=True)
    
    video_paths = []
    for i in range(num_videos):
        video_path = os.path.join(data_dir, f"dummy_video_{i}.mp4")
        create_dummy_video(video_path, num_frames, height, width)
        video_paths.append(video_path)
    
    initial_frame_path = os.path.join(data_dir, "initial_frame.jpg")
    initial_frame = create_dummy_frame(height, width)
    cv2.imwrite(initial_frame_path, cv2.cvtColor(initial_frame, cv2.COLOR_RGB2BGR))
    
    return video_paths


def normalize_frame(frame):
    """
    Normalize a frame to [0, 1] range.
    
    Args:
        frame: Frame as numpy array
        
    Returns:
        Normalized frame as torch tensor
    """
    frame = frame.astype(np.float32) / 255.0
    
    tensor = torch.from_numpy(frame).permute(2, 0, 1)
    
    return tensor


def preprocess_frame_for_model(frame, device='cuda'):
    """
    Preprocess a frame for model input.
    
    Args:
        frame: Frame as numpy array
        device: Device to move the tensor to
        
    Returns:
        Preprocessed frame as torch tensor
    """
    tensor = normalize_frame(frame)
    
    tensor = tensor.unsqueeze(0).to(device)
    
    return tensor


def preprocess_batch(frames, device='cuda'):
    """
    Preprocess a batch of frames for model input.
    
    Args:
        frames: List of frames as numpy arrays
        device: Device to move the tensor to
        
    Returns:
        Preprocessed batch as torch tensor
    """
    tensors = [normalize_frame(frame) for frame in frames]
    
    batch = torch.stack(tensors).to(device)
    
    return batch


def main():
    """
    Main function for data preprocessing.
    """
    data_dir = "data"
    video_paths = prepare_dummy_data(data_dir)
    
    for video_path in video_paths:
        output_dir = os.path.join(data_dir, os.path.splitext(os.path.basename(video_path))[0])
        frame_paths = preprocess_video(video_path, output_dir)
        print(f"Preprocessed {video_path} -> {len(frame_paths)} frames")


if __name__ == "__main__":
    main()
