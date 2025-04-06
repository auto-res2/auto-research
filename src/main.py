"""
Main execution script for ProtoSurvPath experiments.
"""

import os
import sys
import numpy as np
import torch
from scipy.stats import ttest_rel
from captum.attr import IntegratedGradients

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.protosurvpath_config import *
from src.preprocess import load_or_generate_data, perform_gmm_clustering
from src.train import (
    ProtoSurvPath, BaseMethod, PANTHER, ProtoSurvPathSimpleFusion,
    cross_validation_experiment
)
from src.utils.visualization import (
    plot_integrated_gradients, plot_prototype_visualization,
    save_directory_check
)

def compute_integrated_gradients(model, sample_gene):
    """
    Compute integrated gradients for model interpretability.
    
    Args:
        model: Trained model
        sample_gene: Sample gene expression data
        
    Returns:
        attributions: Attribution values for each gene feature
    """
    model.eval()
    
    if len(sample_gene.shape) == 1:
        sample_gene = sample_gene.unsqueeze(0)
    
    batch_size = sample_gene.shape[0]
    dummy_image = torch.zeros((batch_size, IMAGE_CHANNELS, *IMAGE_SIZE), dtype=torch.float32)
    
    def gene_encoder_wrapper(gene_input):
        batch_size = gene_input.shape[0]
        dummy_img = torch.zeros((batch_size, IMAGE_CHANNELS, *IMAGE_SIZE), dtype=torch.float32)
        return model(gene_input, dummy_img)
    
    ig = IntegratedGradients(gene_encoder_wrapper)
    
    try:
        attributions, delta = ig.attribute(
            sample_gene, target=0, return_convergence_delta=True
        )
        return attributions.squeeze().detach().numpy()
    except Exception as e:
        print(f"Error computing integrated gradients: {e}")
        return np.zeros(sample_gene.shape[1])

def run_all_experiments(quick_test=QUICK_TEST, num_epochs=NUM_EPOCHS):
    """
    Run all experiments for the ProtoSurvPath paper.
    
    Args:
        quick_test: Flag for quick test with reduced dataset
        num_epochs: Number of training epochs
    """
    save_directory_check('logs')
    
    if quick_test:
        print("Running in QUICK TEST mode with reduced dataset and epochs")
        num_epochs = min(3, num_epochs)
    
    print("\n" + "="*80)
    print("Running ProtoSurvPath Experiments")
    print("="*80)
    
    print("\nExperiment 1: Performance Comparison with Baselines")
    print("-"*50)
    
    print("\nExperiment 1a: ProtoSurvPath (Full Model)")
    proto_loss, proto_metric, proto_model, test_loader = cross_validation_experiment(
        ProtoSurvPath, model_name="ProtoSurvPath", epochs=num_epochs, quick_test=quick_test
    )
    
    print("\nExperiment 1b: Base Method")
    base_loss, base_metric, base_model, _ = cross_validation_experiment(
        BaseMethod, model_name="BaseMethod", epochs=num_epochs, quick_test=quick_test
    )
    
    print("\nExperiment 1c: PANTHER")
    panther_loss, panther_metric, panther_model, _ = cross_validation_experiment(
        PANTHER, model_name="PANTHER", epochs=num_epochs, quick_test=quick_test
    )
    
    print("\nStatistical Comparison:")
    t_stat, p_val = ttest_rel(proto_metric, base_metric)
    print(f"Paired t-test between ProtoSurvPath and BaseMethod: t-stat = {t_stat:.4f}, p-value = {p_val:.4f}")
    
    t_stat, p_val = ttest_rel(proto_metric, panther_metric)
    print(f"Paired t-test between ProtoSurvPath and PANTHER: t-stat = {t_stat:.4f}, p-value = {p_val:.4f}")
    
    print("\nExperiment 2: Ablation Study (Variant C: Simple Fusion)")
    print("-"*50)
    variant_loss, variant_metric, variant_model, _ = cross_validation_experiment(
        ProtoSurvPathSimpleFusion, model_name="ProtoSurvPathSimpleFusion", 
        epochs=num_epochs, quick_test=quick_test
    )
    
    diff = np.array(proto_metric) - np.array(variant_metric)
    print("\nDifference in c-index between full ProtoSurvPath and variant (SimpleFusion):")
    for i, d in enumerate(diff):
        print(f"  Epoch {i+1}: {d:.4f}")
    print(f"  Mean difference: {np.mean(diff):.4f}")
    
    print("\nExperiment 3: Interpretability and Clinical Insight Evaluation")
    print("-"*50)
    
    sample_batch = next(iter(test_loader))
    sample_gene = sample_batch['gene'][0]
    
    print("\nComputing integrated gradients for ProtoSurvPath...")
    attributions = compute_integrated_gradients(proto_model, sample_gene)
    plot_integrated_gradients(
        attributions, 
        title="Integrated Gradients Attribution for ProtoSurvPath", 
        condition="protosurvpath"
    )
    print("Integrated gradients computed and plot saved (logs/interpretability_protosurvpath.pdf).")
    
    print("\nComputing integrated gradients for BaseMethod...")
    base_attributions = compute_integrated_gradients(base_model, sample_gene)
    plot_integrated_gradients(
        base_attributions, 
        title="Integrated Gradients Attribution for BaseMethod", 
        condition="basemethod"
    )
    print("Integrated gradients computed and plot saved (logs/interpretability_basemethod.pdf).")
    
    print("\nExtracting and visualizing prototypes...")
    gene_data = sample_batch['gene'].numpy()
    prototypes, cluster_assignments = perform_gmm_clustering(gene_data, n_components=NUM_PROTOTYPES)
    plot_prototype_visualization(
        prototypes, 
        title="Gene Expression Prototypes", 
        condition="gene"
    )
    if hasattr(prototypes, 'shape'):
        shape_info = str(prototypes.shape)
    elif isinstance(prototypes, list):
        shape_info = f"list with {len(prototypes)} elements"
    else:
        shape_info = "unknown"
    print(f"GaussianMixture clustering completed. Prototypes shape: {shape_info}")
    print("Prototype visualization saved (logs/prototype_gene.pdf).")
    
    print("\n" + "="*80)
    print("All experiments completed successfully!")
    print("="*80)
    
    print("\nSummary of Results:")
    print("-"*50)
    print(f"ProtoSurvPath final c-index: {proto_metric[-1]:.4f}")
    print(f"BaseMethod final c-index: {base_metric[-1]:.4f}")
    print(f"PANTHER final c-index: {panther_metric[-1]:.4f}")
    print(f"ProtoSurvPathSimpleFusion final c-index: {variant_metric[-1]:.4f}")
    
    print("\nSaved Files:")
    print("-"*50)
    print("Training loss plots:")
    print("  - logs/training_loss_protosurvpath.pdf")
    print("  - logs/training_loss_basemethod.pdf")
    print("  - logs/training_loss_panther.pdf")
    print("  - logs/training_loss_protosurvpathsimplefusion.pdf")
    
    print("\nC-index plots:")
    print("  - logs/cindex_protosurvpath.pdf")
    print("  - logs/cindex_basemethod.pdf")
    print("  - logs/cindex_panther.pdf")
    print("  - logs/cindex_protosurvpathsimplefusion.pdf")
    
    print("\nInterpretability plots:")
    print("  - logs/interpretability_protosurvpath.pdf")
    print("  - logs/interpretability_basemethod.pdf")
    print("  - logs/prototype_gene.pdf")
    
    print("\nModel checkpoints:")
    print("  - models/ProtoSurvPath_epoch_*.pt")
    print("  - models/BaseMethod_epoch_*.pt")
    print("  - models/PANTHER_epoch_*.pt")
    print("  - models/ProtoSurvPathSimpleFusion_epoch_*.pt")

def test_code_execution():
    """
    Run a quick test to verify code execution.
    """
    print("Running quick test to verify code execution...")
    
    global QUICK_TEST
    QUICK_TEST = True
    
    run_all_experiments(quick_test=True, num_epochs=1)
    
    print("Quick test completed successfully!")

if __name__ == "__main__":
    for directory in ['logs', 'models', 'data']:
        save_directory_check(directory)
    
    print("\nExperiment Configuration:")
    print("-"*50)
    print(f"Gene Input Dimension: {GENE_INPUT_DIM}")
    print(f"Image Channels: {IMAGE_CHANNELS}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Hidden Dimension: {HIDDEN_DIM}")
    print(f"Number of Prototypes: {NUM_PROTOTYPES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Test Size: {TEST_SIZE}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Quick Test Mode: {QUICK_TEST}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    if QUICK_TEST or "--test" in sys.argv:
        test_code_execution()
    else:
        run_all_experiments()
