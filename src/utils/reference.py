"""Reference retrieval utilities for RG-MDS."""

import torch
from torchvision import transforms
from PIL import Image

def retrieve_reference(img, reference_gallery, feature_extractor, transform):
    """
    Given an input image, extracts its feature representation and picks the closest one from reference_gallery.
    
    Args:
        img (PIL.Image): Input image
        reference_gallery (list): List of reference images
        feature_extractor (torch.nn.Module): Model for feature extraction
        transform (torchvision.transforms): Transforms to apply to images
        
    Returns:
        PIL.Image: Best matching reference image
    """
    img_tensor = transform(img).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        feature = feature_extractor(img_tensor).squeeze()  # feature shape depends on network
    min_dist = float('inf')
    best_ref = None
    for ref_img in reference_gallery:
        ref_tensor = transform(ref_img).unsqueeze(0)
        with torch.no_grad():
            ref_feature = feature_extractor(ref_tensor).squeeze()
        dist = torch.norm(feature - ref_feature, p=2).item()
        if dist < min_dist:
            min_dist = dist
            best_ref = ref_img
    return best_ref
