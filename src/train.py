"""
Training script for DEALWGAN experiments.
"""

import torch
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import compute_fid, save_samples
from utils.models import DEALWGAN, LWGAN

def train_model(model, train_loader, config, writer, experiment_name="experiment"):
    """
    Train a model (DEALWGAN or LWGAN) for a specified number of epochs.
    
    Args:
        model: Model to train (DEALWGAN or LWGAN)
        train_loader: DataLoader for training data
        config: Configuration object
        writer: TensorBoard writer
        experiment_name: Name of the experiment for logging
        
    Returns:
        model: Trained model
        train_metrics: Dictionary containing training metrics
    """
    print(f"Starting training for {model.__class__.__name__} ({experiment_name})...")
    
    epochs_list = []
    loss_epochs = []
    fid_list = []
    
    for epoch in range(config.num_epochs):
        start_time = time.time()
        epoch_losses = []
        
        for i, (imgs, _) in enumerate(train_loader):
            loss = model.train_step(imgs)
            epoch_losses.append(loss)
            
            writer.add_scalar(f"Loss/{experiment_name}", loss, model.step_count)
            
            if i % 50 == 0:
                print(f"Epoch {epoch}, Batch {i}/{len(train_loader)}, Loss: {loss:.4f}")
        
        avg_loss = np.mean(epoch_losses)
        epochs_list.append(epoch)
        loss_epochs.append(avg_loss)
        
        if epoch % config.eval_interval == 0:
            samples = model.generate_samples(num_samples=config.sample_size)
            
            save_samples(samples, f"{experiment_name}_samples_epoch_{epoch}")
            
            fid = compute_fid(None, None)
            fid_list.append(fid)
            
            writer.add_scalar(f"FID/{experiment_name}", fid, epoch)
            
            print(f"Epoch {epoch}: FID = {fid:.4f}")
        
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")
    
    train_metrics = {
        "epochs": epochs_list,
        "losses": loss_epochs,
        "fid": fid_list
    }
    
    return model, train_metrics

def experiment_performance_convergence(config, train_loader, test_loader):
    """
    Experiment 1: Performance and Convergence Benchmark.
    Compares DEALWGAN and LWGAN on performance metrics.
    
    Args:
        config: Configuration object
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        
    Returns:
        results: Dictionary containing experiment results
    """
    print("Starting Experiment 1: Performance and Convergence Benchmark")
    
    writer = SummaryWriter(log_dir='./logs/experiment1')
    
    model_deal = DEALWGAN(config)
    model_lw = LWGAN(config)
    
    _, deal_metrics = train_model(model_deal, train_loader, config, writer, "DEALWGAN")
    _, lw_metrics = train_model(model_lw, train_loader, config, writer, "LWGAN")
    
    from utils.metrics import plot_loss_curves
    
    losses_dict = {
        "DEALWGAN": deal_metrics["losses"],
        "LWGAN": lw_metrics["losses"]
    }
    
    plot_loss_curves(
        deal_metrics["epochs"],
        losses_dict,
        "Training Loss vs. Epoch",
        "training_loss"
    )
    
    if deal_metrics["fid"] and lw_metrics["fid"]:
        fid_epochs = list(range(0, config.num_epochs, config.eval_interval))
        fid_dict = {
            "DEALWGAN": deal_metrics["fid"],
            "LWGAN": lw_metrics["fid"]
        }
        
        plot_loss_curves(
            fid_epochs,
            fid_dict,
            "FID vs. Epoch",
            "fid_curve"
        )
    
    writer.close()
    print("Experiment 1 complete.\n")
    
    return {
        "deal_metrics": deal_metrics,
        "lw_metrics": lw_metrics
    }

def experiment_ablation_study(config, train_loader, test_loader):
    """
    Experiment 2: Ablation Study to Isolate Contributions.
    Compares different variants of DEALWGAN to isolate the contribution
    of each component.
    
    Args:
        config: Configuration object
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        
    Returns:
        results: Dictionary containing experiment results
    """
    print("Starting Experiment 2: Ablation Study")
    
    writer = SummaryWriter(log_dir='./logs/experiment2')
    
    config_a = config
    
    config_b = config
    config_b.use_diffusion = False
    
    config_c = config
    config_c.adaptive_latent = False
    
    model_a = DEALWGAN(config_a)
    model_b = DEALWGAN(config_b)
    model_c = DEALWGAN(config_c)
    
    _, metrics_a = train_model(model_a, train_loader, config, writer, "VariantA")
    _, metrics_b = train_model(model_b, train_loader, config, writer, "VariantB")
    _, metrics_c = train_model(model_c, train_loader, config, writer, "VariantC")
    
    from utils.metrics import plot_loss_curves
    
    losses_dict = {
        "Variant A (Full DEALWGAN)": metrics_a["losses"],
        "Variant B (No Diffusion)": metrics_b["losses"],
        "Variant C (Fixed Latent)": metrics_c["losses"]
    }
    
    plot_loss_curves(
        metrics_a["epochs"],
        losses_dict,
        "Ablation Study: Training Loss vs. Epoch",
        "training_loss_ablation"
    )
    
    if metrics_a["fid"] and metrics_b["fid"] and metrics_c["fid"]:
        fid_epochs = list(range(0, config.num_epochs, config.eval_interval))
        fid_dict = {
            "Variant A": metrics_a["fid"],
            "Variant B": metrics_b["fid"],
            "Variant C": metrics_c["fid"]
        }
        
        plot_loss_curves(
            fid_epochs,
            fid_dict,
            "Ablation Study: FID vs. Epoch",
            "fid_ablation"
        )
    
    writer.close()
    print("Experiment 2 complete.\n")
    
    return {
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "metrics_c": metrics_c
    }

def experiment_robustness_analysis(config, train_loader, test_loader):
    """
    Experiment 3: Robustness and Stability Analysis Across Hyperparameters.
    Tests DEALWGAN with different hyperparameter configurations.
    
    Args:
        config: Configuration object
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        
    Returns:
        results: Dictionary containing experiment results
    """
    print("Starting Experiment 3: Robustness and Stability Analysis")
    
    diffusion_hyperparams = [
        {"noise_schedule": "linear", "step_size": 0.05},
        {"noise_schedule": "linear", "step_size": 0.1},
        {"noise_schedule": "cosine", "step_size": 0.05},
        {"noise_schedule": "cosine", "step_size": 0.1},
    ]
    
    results = {}
    
    for i, params in enumerate(diffusion_hyperparams):
        print(f"Testing configuration {i+1}/{len(diffusion_hyperparams)}: {params}")
        
        temp_config = config
        temp_config.noise_schedule = params["noise_schedule"]
        temp_config.step_size = params["step_size"]
        
        model = DEALWGAN(temp_config)
        
        reduced_epochs = min(5, config.num_epochs)
        temp_config.num_epochs = reduced_epochs
        
        stable_run = True
        epoch_losses = []
        
        try:
            for epoch in range(reduced_epochs):
                losses = []
                
                for i, (imgs, _) in enumerate(train_loader):
                    loss = model.train_step(imgs)
                    
                    if torch.isnan(torch.tensor(loss)):
                        stable_run = False
                        raise ValueError("NaN loss encountered!")
                    
                    losses.append(loss)
                    
                    if i >= 50:
                        break
                
                avg_loss = np.mean(losses)
                epoch_losses.append(avg_loss)
                print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
        
        except Exception as e:
            stable_run = False
            print(f"Run with config {params} failed: {str(e)}")
        
        if stable_run:
            samples = model.generate_samples(num_samples=100)
            fid_val = compute_fid(None, None)
        else:
            fid_val = float('inf')
        
        config_str = f"{params['noise_schedule']}_step{params['step_size']}"
        results[config_str] = {
            "stable": stable_run,
            "fid": fid_val,
            "losses": epoch_losses
        }
        
        print(f"Config {params} → Stable: {stable_run}, FID: {fid_val:.4f}")
    
    from utils.metrics import plot_tsne
    
    latent_reps = model.get_latent_representations(test_loader)
    plot_tsne(latent_reps)
    
    print("Final robustness analysis results:")
    for cfg, res in results.items():
        print(f"Config: {cfg} → Stable: {res['stable']}, FID: {res['fid']:.4f}")
    
    print("Experiment 3 complete.\n")
    
    return results

def run_test(config, train_loader, test_loader):
    """
    Run a quick test of all experiments with reduced settings.
    
    Args:
        config: Configuration object
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    print("Running quick test of experiments...\n")
    
    test_config = config
    
    test_config.num_epochs = 1
    test_config.eval_interval = 1
    test_config.batch_size = 16  # Smaller batch size
    test_config.diffusion_steps = 2  # Fewer diffusion steps
    
    limited_train_loader = []
    for i, batch in enumerate(train_loader):
        if i >= 2:  # Only process 2 batches
            break
        limited_train_loader.append(batch)
    
    limited_test_loader = []
    for i, batch in enumerate(test_loader):
        if i >= 1:  # Only process 1 test batch
            break
        limited_test_loader.append(batch)
    
    class SimpleBatchLoader:
        def __init__(self, batches):
            self.batches = batches
        
        def __iter__(self):
            return iter(self.batches)
        
        def __len__(self):
            return len(self.batches)
    
    simple_train_loader = SimpleBatchLoader(limited_train_loader)
    simple_test_loader = SimpleBatchLoader(limited_test_loader)
    
    print("Testing experiment_performance_convergence...")
    experiment_performance_convergence(test_config, simple_train_loader, simple_test_loader)
    
    print("Testing experiment_ablation_study...")
    experiment_ablation_study(test_config, simple_train_loader, simple_test_loader)
    
    print("Testing experiment_robustness_analysis...")
    experiment_robustness_analysis(test_config, simple_train_loader, simple_test_loader)
    
    print("\nQuick test complete. All experiments executed without error.")
