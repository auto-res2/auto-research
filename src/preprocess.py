#!/usr/bin/env python
"""
Preprocessing module for Tweedie-Guided Global Consistent Video Editing (TG-GCVE).

This module handles:
1. Video loading and frame extraction
2. Spatiotemporal slice extraction
3. Global context aggregation
4. Data preparation for the TG-GCVE model
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    print(f"Setting random seed to {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class VideoProcessor:
    """Class for video processing and frame extraction."""
    
    def __init__(self, video_path=None, frame_size=(256, 256)):
        """
        Initialize the video processor.
        
        Args:
            video_path (str): Path to the video file
            frame_size (tuple): Target size for frames (height, width)
        """
        self.video_path = video_path
        self.frame_size = frame_size
        self.frames = []
        self.transform = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def load_video(self, video_path=None):
        """
        Load video from path and extract frames.
        
        Args:
            video_path (str, optional): Path to video file. If None, uses self.video_path.
            
        Returns:
            list: List of extracted frames as tensors
        """
        if video_path:
            self.video_path = video_path
            
        if not self.video_path:
            raise ValueError("No video path provided")
            
        print(f"Loading video from {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        self.frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Apply transformations
            tensor_frame = self.transform(pil_img)
            self.frames.append(tensor_frame)
            frame_count += 1
            
        cap.release()
        print(f"Extracted {frame_count} frames from video")
        return self.frames
    
    def extract_spatiotemporal_slices(self, slice_type='horizontal', num_slices=8):
        """
        Extract spatiotemporal slices from video frames.
        
        Args:
            slice_type (str): Type of slice ('horizontal', 'vertical', or 'both')
            num_slices (int): Number of slices to extract
            
        Returns:
            dict: Dictionary containing spatiotemporal slices
        """
        if not self.frames:
            raise ValueError("No frames loaded. Call load_video first.")
            
        frame_tensors = torch.stack(self.frames)  # [T, C, H, W]
        T, C, H, W = frame_tensors.shape
        
        slices = {}
        
        if slice_type in ['horizontal', 'both']:
            # Extract horizontal slices (time-width slices at different heights)
            h_indices = np.linspace(0, H-1, num_slices, dtype=int)
            h_slices = []
            
            for h_idx in h_indices:
                # Extract slice at height h_idx across all frames
                h_slice = frame_tensors[:, :, h_idx:h_idx+1, :]  # [T, C, 1, W]
                h_slice = h_slice.squeeze(2)  # [T, C, W]
                h_slices.append(h_slice)
                
            slices['horizontal'] = h_slices
            
        if slice_type in ['vertical', 'both']:
            # Extract vertical slices (time-height slices at different widths)
            w_indices = np.linspace(0, W-1, num_slices, dtype=int)
            v_slices = []
            
            for w_idx in w_indices:
                # Extract slice at width w_idx across all frames
                v_slice = frame_tensors[:, :, :, w_idx:w_idx+1]  # [T, C, H, 1]
                v_slice = v_slice.squeeze(3)  # [T, C, H]
                v_slices.append(v_slice)
                
            slices['vertical'] = v_slices
            
        print(f"Extracted {len(slices.get('horizontal', [])) + len(slices.get('vertical', []))} spatiotemporal slices")
        return slices
    
    def compute_global_attention_maps(self, frames=None):
        """
        Compute global attention maps for frames.
        
        Args:
            frames (torch.Tensor, optional): Video frames. If None, uses self.frames.
            
        Returns:
            torch.Tensor: Global attention maps for each frame
        """
        if frames is None:
            if not self.frames:
                raise ValueError("No frames loaded. Call load_video first.")
            frames = torch.stack(self.frames)  # [T, C, H, W]
            
        T, C, H, W = frames.shape
        
        # Simple implementation of global attention maps
        # In a real implementation, this would use a more sophisticated attention mechanism
        
        # Convert to feature space (dummy implementation)
        # In practice, this would use a CNN backbone
        frame_features = frames.view(T, C, -1)  # [T, C, H*W]
        frame_features = frame_features.permute(0, 2, 1)  # [T, H*W, C]
        
        # Compute attention weights (dummy implementation)
        # In practice, this would use self-attention or similar
        attention_weights = torch.softmax(torch.randn(T, H*W, H*W, device=frames.device), dim=-1)
        
        # Reshape attention maps to spatial dimensions
        attention_maps = attention_weights.mean(dim=-1).view(T, H, W)
        
        print(f"Computed global attention maps with shape {attention_maps.shape}")
        return attention_maps

class TGGCVEDataset(Dataset):
    """Dataset class for TG-GCVE model training."""
    
    def __init__(self, data_dir, frame_size=(256, 256), sequence_length=16, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing video files
            frame_size (tuple): Target size for frames (height, width)
            sequence_length (int): Number of frames to use in each sequence
            transform (callable, optional): Optional transform to apply to frames
        """
        self.data_dir = data_dir
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(frame_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        # Find all video files
        self.video_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    self.video_files.append(os.path.join(root, file))
                    
        print(f"Found {len(self.video_files)} video files in {data_dir}")
        
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        """
        Get a video sequence and its processed data.
        
        Args:
            idx (int): Index of the video file
            
        Returns:
            dict: Dictionary containing frames, slices, and attention maps
        """
        video_path = self.video_files[idx]
        processor = VideoProcessor(video_path, self.frame_size)
        
        # Load video and extract frames
        frames = processor.load_video()
        
        # If video has more frames than sequence_length, randomly select a subsequence
        if len(frames) > self.sequence_length:
            start_idx = random.randint(0, len(frames) - self.sequence_length)
            frames = frames[start_idx:start_idx + self.sequence_length]
        
        # If video has fewer frames than sequence_length, pad with zeros
        elif len(frames) < self.sequence_length:
            padding = [torch.zeros_like(frames[0]) for _ in range(self.sequence_length - len(frames))]
            frames.extend(padding)
            
        # Stack frames into a tensor
        frames_tensor = torch.stack(frames)  # [T, C, H, W]
        
        # Extract spatiotemporal slices
        slices = processor.extract_spatiotemporal_slices(slice_type='both')
        
        # Compute global attention maps
        attention_maps = processor.compute_global_attention_maps(frames_tensor)
        
        return {
            'frames': frames_tensor,
            'slices': slices,
            'attention_maps': attention_maps,
            'video_path': video_path
        }

def create_dataloader(data_dir, batch_size=4, frame_size=(256, 256), sequence_length=16, num_workers=4):
    """
    Create a DataLoader for the TG-GCVE dataset.
    
    Args:
        data_dir (str): Directory containing video files
        batch_size (int): Batch size
        frame_size (tuple): Target size for frames (height, width)
        sequence_length (int): Number of frames to use in each sequence
        num_workers (int): Number of worker threads for data loading
        
    Returns:
        DataLoader: DataLoader for the TG-GCVE dataset
    """
    dataset = TGGCVEDataset(data_dir, frame_size, sequence_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created DataLoader with {len(dataset)} videos, batch size {batch_size}")
    return dataloader

def add_noise(tensor, noise_level=0.1):
    """
    Add Gaussian noise to a tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor
        noise_level (float): Standard deviation of the noise
        
    Returns:
        torch.Tensor: Noisy tensor
    """
    noise = torch.randn_like(tensor) * noise_level
    return tensor + noise

def preprocess_for_training(data_dir, batch_size=4, frame_size=(256, 256), sequence_length=16):
    """
    Preprocess data for TG-GCVE model training.
    
    Args:
        data_dir (str): Directory containing video files
        batch_size (int): Batch size
        frame_size (tuple): Target size for frames (height, width)
        sequence_length (int): Number of frames to use in each sequence
        
    Returns:
        DataLoader: DataLoader for the TG-GCVE dataset
    """
    print(f"Preprocessing data for TG-GCVE model training from {data_dir}")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create dataloader
    dataloader = create_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        frame_size=frame_size,
        sequence_length=sequence_length
    )
    
    return dataloader

if __name__ == "__main__":
    # Example usage
    data_dir = "data/sample_videos"
    
    # Check if data directory exists, if not create it
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created directory: {data_dir}")
        print("Please add video files to this directory for preprocessing")
    
    # If data directory exists and contains video files, preprocess them
    elif any(f.endswith(('.mp4', '.avi', '.mov')) for f in os.listdir(data_dir)):
        dataloader = preprocess_for_training(
            data_dir=data_dir,
            batch_size=2,
            frame_size=(256, 256),
            sequence_length=16
        )
        
        # Print sample batch
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(f"  Frames shape: {batch['frames'].shape}")
            print(f"  Horizontal slices: {len(batch['slices']['horizontal'])}")
            print(f"  Vertical slices: {len(batch['slices']['vertical'])}")
            print(f"  Attention maps shape: {batch['attention_maps'].shape}")
            
            # Only process one batch for demonstration
            break
    else:
        print(f"No video files found in {data_dir}")
        print("Please add video files (mp4, avi, mov) to this directory for preprocessing")
