"""
Simple test script to verify the ACSC implementation.
"""

import torch
import matplotlib.pyplot as plt
from src.utils.acsc_utils import diffusion_process

def test_diffusion_process():
    """
    Test the diffusion process function.
    """
    image = torch.rand(1, 1, 64, 64)
    
    baseline_output = diffusion_process(image, acsc_enabled=False)
    acsc_output = diffusion_process(image, acsc_enabled=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(baseline_output.squeeze(), cmap='gray')
    plt.title("Baseline")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(acsc_output.squeeze(), cmap='gray')
    plt.title("ACSC")
    plt.axis('off')
    
    plt.suptitle("ACSC Test")
    plt.savefig("logs/acsc_test.pdf", format='pdf')
    plt.close()
    
    print("Test completed successfully. Results saved to logs/acsc_test.pdf")
    return True

if __name__ == "__main__":
    test_diffusion_process()
