import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return True

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # list of PIL Images
        self.labels = labels  # list of integer labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class TextToImageModel:
    """Simulate a text-to-image generator by returning a random RGB image."""
    def generate(self, prompt, size=(64, 64)):
        array = np.uint8(255 * np.random.rand(size[0], size[1], 3))
        return Image.fromarray(array)

class DiffusionModel:
    """Simulate a diffusion model that enhances a generated image."""
    def generate(self, prompt, conditioning_images=None, strength=0.5, size=(64, 64)):
        base = np.uint8(255 * np.random.rand(size[0], size[1], 3))
        img = Image.fromarray(base)
        if conditioning_images is not None and len(conditioning_images) > 0:
            proto = conditioning_images[random.randint(0, len(conditioning_images)-1)]
            proto = proto.resize(size)
            base_proto = np.array(proto, dtype=np.float32)
            base_img = np.array(img, dtype=np.float32)
            combined = (1-strength)*base_img + strength*base_proto
            combined = np.clip(combined, 0, 255).astype(np.uint8)
            img = Image.fromarray(combined)
        blur_radius = strength * 2.0  # adjust scale as needed
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return img

def load_real_prototype_images(num_classes=4, per_class=5, size=(64,64)):
    """Generate simulated prototype images for each class."""
    real_prototypes = {}
    for cls in range(num_classes):
        prototypes = []
        for _ in range(per_class):
            arr = np.uint8(255 * np.random.rand(size[0], size[1], 3))
            prototypes.append(Image.fromarray(arr))
        real_prototypes[cls] = prototypes
    return real_prototypes

def identify_weak_classes(synthetic_dataset, real_prototypes):
    """Identify classes that need enhancement."""
    weak_classes = [1, 3]
    print("Identified weak classes:", weak_classes)
    return weak_classes

def get_transform():
    """Return basic transformation for images."""
    return transforms.Compose([
        transforms.ToTensor(),
    ])
