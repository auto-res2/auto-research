"""
Data utilities for the PTDA model.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_video_frames(video_path, num_frames=None):
    """
    Load frames from a video file.
    
    Args:
        video_path: Path to the video file
        num_frames: Maximum number of frames to load (None for all frames)
        
    Returns:
        List of frames (numpy arrays)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
        count += 1
        if num_frames is not None and count >= num_frames:
            break
    
    cap.release()
    return frames


def create_dummy_video(output_path, num_frames=10, height=240, width=320, fps=5):
    """
    Create a dummy video file with random frames.
    
    Args:
        output_path: Path to save the video
        num_frames: Number of frames to generate
        height: Frame height
        width: Frame width
        fps: Frames per second
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for _ in range(num_frames):
        frame = np.uint8(np.random.rand(height, width, 3) * 255)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()


def create_dummy_frame(height=240, width=320):
    """
    Create a dummy frame with random pixel values.
    
    Args:
        height: Frame height
        width: Frame width
        
    Returns:
        Numpy array of shape (height, width, 3)
    """
    return np.uint8(np.random.rand(height, width, 3) * 255)

def preprocess_frame_for_model(frame, device='cuda'):
    """
    Preprocess a frame for model input.
    
    Args:
        frame: Frame as numpy array
        device: Device to move the tensor to
        
    Returns:
        Preprocessed frame as torch tensor
    """
    frame = frame.astype(np.float32) / 255.0
    
    tensor = torch.from_numpy(frame).permute(2, 0, 1)
    
    tensor = tensor.unsqueeze(0).to(device)
    
    return tensor



class VideoFrameDataset(Dataset):
    """
    Dataset for video frames.
    
    For testing purposes, this can generate synthetic data if no video path is provided.
    """
    def __init__(self, video_path=None, num_samples=4, num_frames=5, height=64, width=64, transform=None):
        """
        Initialize the dataset.
        
        Args:
            video_path: Path to the video file (None for synthetic data)
            num_samples: Number of samples to generate (for synthetic data)
            num_frames: Number of frames per sequence
            height: Frame height (for synthetic data)
            width: Frame width (for synthetic data)
            transform: Optional transform to apply to the frames
        """
        self.video_path = video_path
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.transform = transform
        
        if video_path is not None and os.path.exists(video_path):
            self.frames = load_video_frames(video_path)
            self.num_samples = max(1, len(self.frames) - num_frames + 1)
        else:
            self.frames = None
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.frames is not None:
            sequence = self.frames[idx:idx+self.num_frames]
            
            tensor_frames = []
            for frame in sequence:
                if frame.shape[0] != self.height or frame.shape[1] != self.width:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                tensor_frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
                tensor_frames.append(tensor_frame)
            
            tensor_sequence = torch.stack(tensor_frames)  # Shape: [num_frames, 3, H, W]
        else:
            tensor_sequence = torch.rand(self.num_frames, 3, self.height, self.width)
        
        if self.transform:
            tensor_sequence = self.transform(tensor_sequence)
        
        return tensor_sequence


def get_dataloader(dataset, batch_size=4, shuffle=True, num_workers=2):
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset object
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker threads
        
    Returns:
        DataLoader object
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
