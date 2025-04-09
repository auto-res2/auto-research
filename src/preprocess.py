import numpy as np
import torch

def load_dataset(name="VisualGenome", num_images=10):
    """
    Creates a synthetic dataset for scene graph generation experiments.
    
    Args:
        name (str): Name of the dataset (for future real implementation)
        num_images (int): Number of synthetic images to generate
        
    Returns:
        list: List of (image, ground_truth) tuples
    """
    print(f"Loading synthetic dataset '{name}' with {num_images} images...")
    dataset = []
    
    for i in range(num_images):
        image_tensor = torch.randn(3, 224, 224)
        
        num_objs = np.random.randint(5, 15)
        detections = {
            "bboxes": np.random.rand(num_objs, 4) * 224,  # x1, y1, x2, y2
            "confidences": np.random.rand(num_objs),
            "labels": np.random.randint(0, 20, size=(num_objs,))
        }
        
        image = {"tensor": image_tensor, "detections": detections, "id": i}
        
        gt = {"id": i, "graph": f"gt_scene_graph_{i}"}
        
        dataset.append((image, gt))
    
    return dataset

def add_noise_to_bboxes(bboxes, noise_std):
    """
    Add Gaussian noise to bounding boxes for robustness testing.
    
    Args:
        bboxes (numpy.ndarray): Array of bounding boxes, shape (N, 4)
        noise_std (float): Standard deviation of the noise
        
    Returns:
        numpy.ndarray: Noisy bounding boxes
    """
    noisy_bboxes = bboxes.copy()
    noise = np.random.normal(0, noise_std, bboxes.shape)
    noisy_bboxes += noise
    return noisy_bboxes

def add_noise_to_confidences(confidences, flip_prob):
    """
    Flip confidence scores with a certain probability for robustness testing.
    
    Args:
        confidences (numpy.ndarray): Array of confidence scores, shape (N,)
        flip_prob (float): Probability of flipping each confidence
        
    Returns:
        numpy.ndarray: Noisy confidences
    """
    noisy_confidences = confidences.copy()
    flip_mask = np.random.rand(len(confidences)) < flip_prob
    noisy_confidences[flip_mask] = 1.0 - noisy_confidences[flip_mask]
    return noisy_confidences
