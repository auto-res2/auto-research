"""
Main script for running SAC-Seg experiments.

This script implements three experiments:
1. Computational Efficiency and Memory Footprint Measurement
2. Segmentation Accuracy and Fine-Tuning Effectiveness
3. Domain Adaptation and Robustness Testing

All plots are saved as PDF files in the logs directory.
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

print("\n" + "="*50)
print("SAC-Seg Experiment Runner")
print("="*50)
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print("="*50 + "\n")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models import BaseModel, SACSeg
from preprocess import RandomSegmentationDataset, get_data_loaders
from train import train_one_epoch, evaluate, train_model
from evaluate import segmentation_metrics, compute_dice_coefficient
from utils.visualization import save_plot, plot_comparison

print(f"Successfully imported all required modules")
print(f"sys.path: {sys.path}")
print("="*50 + "\n")

os.makedirs('logs', exist_ok=True)


def check_gpu_compatibility():
    """
    Check if CUDA is available and print GPU information.
    """
    print("\n=== GPU Compatibility Check ===")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA is available with {device_count} device(s)")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # GB
            print(f"Device {i}: {device_name}")
            print(f"  Compute Capability: {device_capability[0]}.{device_capability[1]}")
            print(f"  Total Memory: {total_memory:.2f} GB")
            
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        if any('T4' in name for name in device_names):
            print("NVIDIA Tesla T4 detected - compatible with SAC-Seg implementation")
        else:
            print("NVIDIA Tesla T4 not detected, but code should run on available GPU")
    else:
        print("CUDA is not available. Running on CPU.")
    print("=" * 30 + "\n")


def experiment1_efficiency(
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 4,
    num_classes: int = 16,
    num_samples: int = 20,
    seed_config: Optional[Dict] = None,
    embedding_dim: int = 64
):
    """
    Experiment 1: Computational Efficiency and Memory Footprint Measurement
    
    Compares the computational efficiency and memory usage of the BaseModel and SAC-Seg.
    
    Args:
        image_size: Input image size
        batch_size: Batch size for training
        num_classes: Number of segmentation classes
        num_samples: Number of samples in the dataset
        seed_config: Configuration for seed-based methods
    """
    print("\n[Experiment 1] Computational Efficiency and Memory Footprint Measurement")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_data_loaders(
        image_size=image_size,
        num_samples=num_samples,
        num_classes=num_classes,
        batch_size=batch_size
    )
    
    models = {
        "BaseModel": BaseModel(
            input_channels=3,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            image_size=image_size
        ).to(device),
        
        "SACSeg": SACSeg(
            input_channels=3,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            image_size=image_size,
            seed_config=seed_config
        ).to(device)
    }
    
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    
    for name, model in models.items():
        print(f"\nMeasuring {name} efficiency...")
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        epoch_loss, epoch_time, peak_memory = train_one_epoch(
            model=model, 
            dataloader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device,
            seed_config=seed_config if name == "SACSeg" else None
        )
        
        results[name] = {
            "epoch_time_sec": epoch_time,
            "peak_memory_MB": peak_memory,
            "epoch_loss": epoch_loss
        }
        
        print(f"{name}: {epoch_time:.2f} sec/epoch, {peak_memory:.2f} MB peak memory, Loss: {epoch_loss:.4f}")
    
    times = {name: results[name]["epoch_time_sec"] for name in results}
    time_fig = plot_comparison(
        times,
        title="Training Time Comparison",
        xlabel="Model",
        ylabel="Epoch Training Time (sec)",
        filename="logs/comp_efficiency_epoch_time.pdf"
    )
    
    memories = {name: results[name]["peak_memory_MB"] for name in results}
    memory_fig = plot_comparison(
        memories,
        title="Memory Usage Comparison",
        xlabel="Model",
        ylabel="Peak Memory (MB)",
        filename="logs/comp_efficiency_peak_memory.pdf"
    )
    
    print("\nExperiment 1 completed. Results saved to logs directory.")


def experiment2_segmentation(
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 4,
    num_classes: int = 16,
    num_samples: int = 50,
    num_epochs: int = 5,
    seed_config: Optional[Dict] = None,
    embedding_dim: int = 64
):
    """
    Experiment 2: Segmentation Accuracy and Fine-Tuning Effectiveness
    
    Compares the segmentation accuracy and fine-tuning effectiveness of the BaseModel and SAC-Seg.
    
    Args:
        image_size: Input image size
        batch_size: Batch size for training
        num_classes: Number of segmentation classes
        num_samples: Number of samples in the dataset
        num_epochs: Number of training epochs
        seed_config: Configuration for seed-based methods
    """
    print("\n[Experiment 2] Segmentation Accuracy and Fine-Tuning Effectiveness")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_data_loaders(
        image_size=image_size,
        num_samples=num_samples,
        num_classes=num_classes,
        batch_size=batch_size
    )
    
    models = {
        "BaseModel": BaseModel(
            input_channels=3,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            image_size=image_size
        ).to(device),
        
        "SACSeg": SACSeg(
            input_channels=3,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            image_size=image_size,
            seed_config=seed_config
        ).to(device)
    }
    
    criterion = nn.CrossEntropyLoss()
    metric_fn = lambda pred, target: segmentation_metrics(pred, target, num_classes)
    
    histories = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        history = train_model(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            metric_fn=metric_fn,
            device=device, 
            num_epochs=num_epochs, 
            seed_config=seed_config if name == "SACSeg" else None
        )
        
        histories[name] = history
    
    train_losses = {name: histories[name]["train_loss"] for name in histories}
    epochs = [float(i) for i in range(1, num_epochs + 1)]
    
    loss_fig = plot_comparison(
        train_losses,
        x_values=epochs,
        title="Training Loss Comparison",
        xlabel="Epoch",
        ylabel="Training Loss",
        filename="logs/seg_accuracy_train_loss.pdf"
    )
    
    val_ious = {
        name: [metrics["mean_iou"] for metrics in histories[name]["val_metrics"]]
        for name in histories
    }
    
    iou_fig = plot_comparison(
        val_ious,
        x_values=epochs,
        title="Validation IoU Comparison",
        xlabel="Epoch",
        ylabel="Mean IoU",
        filename="logs/seg_accuracy_val_iou.pdf"
    )
    
    val_accs = {
        name: [metrics["pixel_acc"] for metrics in histories[name]["val_metrics"]]
        for name in histories
    }
    
    acc_fig = plot_comparison(
        val_accs,
        x_values=epochs,
        title="Validation Pixel Accuracy Comparison",
        xlabel="Epoch",
        ylabel="Pixel Accuracy",
        filename="logs/seg_accuracy_val_pixelacc.pdf"
    )
    
    print("\nExperiment 2 completed. Results saved to logs directory.")


def experiment3_domain_adaptation(
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 4,
    num_classes: int = 2,  # Binary segmentation for domain adaptation
    num_samples: int = 50,
    num_epochs: int = 5,
    seed_configs: Optional[List[Dict]] = None,
    embedding_dim: int = 64
):
    """
    Experiment 3: Domain Adaptation and Robustness Testing
    
    Tests the domain adaptation capabilities and robustness of the SAC-Seg model
    with different seed configurations.
    
    Args:
        image_size: Input image size
        batch_size: Batch size for training
        num_classes: Number of segmentation classes (2 for binary segmentation)
        num_samples: Number of samples in the dataset
        num_epochs: Number of training epochs
        seed_configs: List of seed configurations to test
    """
    print("\n[Experiment 3] Domain Adaptation and Robustness Testing")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if seed_configs is None:
        seed_configs = [
            {"num_seeds": 5, "seed_prob_threshold": 0.3, "perturbation_scale": 0.01},
            {"num_seeds": 10, "seed_prob_threshold": 0.5, "perturbation_scale": 0.01},
            {"num_seeds": 20, "seed_prob_threshold": 0.7, "perturbation_scale": 0.01}
        ]
    
    train_loader, val_loader = get_data_loaders(
        image_size=image_size,
        num_samples=num_samples,
        num_classes=num_classes,
        batch_size=batch_size
    )
    
    base_model = BaseModel(
        input_channels=3,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        image_size=image_size
    ).to(device)
    
    sac_models = {
        f"SACSeg-{config['num_seeds']}seeds": SACSeg(
            input_channels=3,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            image_size=image_size,
            seed_config=config
        ).to(device)
        for config in seed_configs
    }
    
    models = {"BaseModel": base_model, **sac_models}
    
    criterion = nn.CrossEntropyLoss()
    
    def binary_metric_fn(pred, target):
        if pred.dim() > target.dim():
            pred_binary = pred[:, 1]  # Take the second channel for the foreground
        else:
            pred_binary = pred > 0
            
        target_binary = target > 0
        dice = compute_dice_coefficient(pred_binary, target_binary)
        
        metrics = segmentation_metrics(pred, target, num_classes)
        metrics['dice'] = dice
        
        return metrics
    
    histories = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        if name.startswith("SACSeg"):
            seed_config = seed_configs[int(name.split("-")[1].split("seeds")[0]) // 5 - 1]
        else:
            seed_config = None
        
        history = train_model(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            metric_fn=binary_metric_fn,
            device=device, 
            num_epochs=num_epochs, 
            seed_config=seed_config
        )
        
        histories[name] = history
    
    train_losses = {name: histories[name]["train_loss"] for name in histories}
    epochs = [float(i) for i in range(1, num_epochs + 1)]
    
    loss_fig = plot_comparison(
        train_losses,
        x_values=epochs,
        title="Domain Adaptation Training Loss",
        xlabel="Epoch",
        ylabel="Training Loss",
        filename="logs/domain_adaptation_train_loss.pdf"
    )
    
    val_dices = {
        name: [metrics["dice"] for metrics in histories[name]["val_metrics"]]
        for name in histories
    }
    
    dice_fig = plot_comparison(
        val_dices,
        x_values=epochs,
        title="Domain Adaptation Dice Coefficient",
        xlabel="Epoch",
        ylabel="Dice Coefficient",
        filename="logs/domain_adaptation_dice.pdf"
    )
    
    print("\nExperiment 3 completed. Results saved to logs directory.")


def run_test():
    """
    Run a quick test to verify that the code executes correctly.
    """
    print("\n[TEST] Running quick tests for each experiment...")
    
    test_params = {
        "image_size": (64, 64),
        "batch_size": 2,
        "num_classes": 4,
        "num_samples": 8,
        "num_epochs": 1,
        "embedding_dim": 64
    }
    
    test_seed_config = {
        "num_seeds": 5,
        "seed_prob_threshold": 0.5,
        "perturbation_scale": 0.01
    }
    
    try:
        print("\n" + "="*50)
        print("DETAILED TEST EXECUTION LOG")
        print("="*50)
        print("\nCreating test dataset...")
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = RandomSegmentationDataset(
            image_size=test_params["image_size"],
            num_samples=4,
            num_classes=test_params["num_classes"],
            transform=transform
        )
        
        test_params["embedding_dim"] = test_params["num_classes"]
        
        sample_img, sample_mask = test_dataset[0]
        print(f"Sample image shape: {sample_img.shape}")
        print(f"Sample mask shape: {sample_mask.shape}")
        
        print("\nTesting Experiment 1...")
        experiment1_efficiency(
            image_size=test_params["image_size"],
            batch_size=test_params["batch_size"],
            num_classes=test_params["num_classes"],
            num_samples=test_params["num_samples"],
            seed_config=test_seed_config,
            embedding_dim=test_params["embedding_dim"]
        )
        
        print("\nTesting Experiment 2...")
        experiment2_segmentation(
            image_size=test_params["image_size"],
            batch_size=test_params["batch_size"],
            num_classes=test_params["num_classes"],
            num_samples=test_params["num_samples"],
            num_epochs=test_params["num_epochs"],
            seed_config=test_seed_config,
            embedding_dim=test_params["embedding_dim"]
        )
        
        print("\nTesting Experiment 3...")
        experiment3_domain_adaptation(
            image_size=test_params["image_size"],
            batch_size=test_params["batch_size"],
            num_classes=2,  # Binary segmentation for domain adaptation
            num_samples=test_params["num_samples"],
            num_epochs=test_params["num_epochs"],
            seed_configs=[test_seed_config],
            embedding_dim=2  # Match embedding_dim with num_classes for binary segmentation
        )
        
        print("\n[TEST] All tests completed successfully!")
        return True
    except Exception as e:
        print(f"\n[TEST] Error during testing: {str(e)}")
        return False


if __name__ == "__main__":
    check_gpu_compatibility()
    
    test_success = run_test()
    
    if test_success:
        print("\nRunning full experiments...")
        
        default_seed_config = {
            "num_seeds": 10,
            "seed_prob_threshold": 0.5,
            "perturbation_scale": 0.01
        }
        
        print("\nRunning Experiment 1 with adjusted parameters...")
        experiment1_efficiency(
            seed_config=default_seed_config, 
            embedding_dim=16,  # Match with default num_classes=16
            num_classes=16,
            image_size=(128, 128),  # Smaller image size for faster execution
            batch_size=2
        )
        
        print("\nRunning Experiment 2 with adjusted parameters...")
        experiment2_segmentation(
            seed_config=default_seed_config, 
            embedding_dim=16,  # Match with default num_classes=16
            num_classes=16,
            image_size=(128, 128),  # Smaller image size for faster execution
            batch_size=2,
            num_epochs=2  # Reduce epochs for faster execution
        )
        
        seed_configs = [
            {"num_seeds": 5, "seed_prob_threshold": 0.3, "perturbation_scale": 0.01},
            {"num_seeds": 10, "seed_prob_threshold": 0.5, "perturbation_scale": 0.01},
            {"num_seeds": 20, "seed_prob_threshold": 0.7, "perturbation_scale": 0.01}
        ]
        
        print("\nRunning Experiment 3 with adjusted parameters...")
        experiment3_domain_adaptation(
            seed_configs=seed_configs, 
            embedding_dim=2,  # Match with num_classes=2 for binary segmentation
            num_classes=2,
            image_size=(128, 128),  # Smaller image size for faster execution
            batch_size=2,
            num_epochs=2  # Reduce epochs for faster execution
        )
        
        print("\nAll experiments completed successfully!")
    else:
        print("\nSkipping full experiments due to test failure.")
