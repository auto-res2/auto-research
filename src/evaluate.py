"""
Evaluation script for VG-DD experiments.
Implements evaluation metrics and visualization for all three experiments.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.vgdd.config import EXPERIMENT1_CONFIG, EXPERIMENT2_CONFIG, EXPERIMENT3_CONFIG

def evaluate_adaptive_weighting(decoder, visual_features, tokens, save_dir="logs"):
    """
    Evaluate the adaptive visual prompt weighting mechanism.
    
    Args:
        decoder: The AdaptiveDecoder model
        visual_features: Visual features from the VisualModule
        tokens: Input token sequence
        save_dir: Directory to save evaluation results
    
    Returns:
        dict: Evaluation metrics
    """
    device = tokens.device
    
    memory = tokens.clone()  # Dummy memory for demonstration
    adaptive_logits, cosine_sim = decoder(tokens, memory, visual_features)
    
    cosine_vals = cosine_sim.squeeze(-1).detach().cpu().numpy()
    
    avg_cosine = np.mean(cosine_vals)
    min_cosine = np.min(cosine_vals)
    max_cosine = np.max(cosine_vals)
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(cosine_vals.flatten(), kde=True)
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Token-Visual Cosine Similarities", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cosine_similarity_distribution.pdf"), format='pdf')
    plt.close()
    
    metrics = {
        "avg_cosine_similarity": avg_cosine,
        "min_cosine_similarity": min_cosine,
        "max_cosine_similarity": max_cosine
    }
    
    print("Adaptive Weighting Evaluation:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def evaluate_iterative_decoding(grounding_scores, threshold=0.5, save_dir="logs"):
    """
    Evaluate the iterative decoding process.
    
    Args:
        grounding_scores: Token grounding scores
        threshold: Grounding threshold
        save_dir: Directory to save evaluation results
    
    Returns:
        dict: Evaluation metrics
    """
    if isinstance(grounding_scores, torch.Tensor):
        grounding_scores = grounding_scores.detach().cpu().numpy()
    
    avg_grounding = np.mean(grounding_scores)
    min_grounding = np.min(grounding_scores)
    max_grounding = np.max(grounding_scores)
    
    tokens_above_threshold = (grounding_scores >= threshold).sum() / grounding_scores.size
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(grounding_scores.flatten(), kde=True)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel("Grounding Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Token Grounding Scores", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "grounding_score_distribution.pdf"), format='pdf')
    plt.close()
    
    if len(grounding_scores.shape) > 1:
        plt.figure(figsize=(12, 6), dpi=300)
        sns.heatmap(grounding_scores, cmap="viridis", annot=False)
        plt.xlabel("Token Position", fontsize=12)
        plt.ylabel("Batch", fontsize=12)
        plt.title("Grounding Scores by Token Position", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "grounding_score_heatmap.pdf"), format='pdf')
        plt.close()
    
    metrics = {
        "avg_grounding_score": avg_grounding,
        "min_grounding_score": min_grounding,
        "max_grounding_score": max_grounding,
        "tokens_above_threshold": tokens_above_threshold
    }
    
    print("Iterative Decoding Evaluation:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def evaluate_contrastive_learning(model, intact_img, perturbed_img, text_input, save_dir="logs"):
    """
    Evaluate the contrastive learning model.
    
    Args:
        model: The ContrastiveLVLM model
        intact_img: Original image tensor
        perturbed_img: Perturbed image tensor
        text_input: Text input tensor
        save_dir: Directory to save evaluation results
    
    Returns:
        dict: Evaluation metrics
    """
    device = intact_img.device
    model.eval()
    
    with torch.no_grad():
        intact_emb, text_emb = model(intact_img, text_input)
        perturbed_emb, _ = model(perturbed_img, text_input)
        
        intact_text_sim = F.cosine_similarity(intact_emb, text_emb).item()
        perturbed_text_sim = F.cosine_similarity(perturbed_emb, text_emb).item()
        intact_perturbed_sim = F.cosine_similarity(intact_emb, perturbed_emb).item()
        
        intact_norm = torch.norm(intact_emb, dim=1).mean().item()
        perturbed_norm = torch.norm(perturbed_emb, dim=1).mean().item()
        text_norm = torch.norm(text_emb, dim=1).mean().item()
    
    plt.figure(figsize=(10, 6), dpi=300)
    similarities = [intact_text_sim, perturbed_text_sim, intact_perturbed_sim]
    labels = ['Intact-Text', 'Perturbed-Text', 'Intact-Perturbed']
    sns.barplot(x=labels, y=similarities)
    plt.xlabel("Pair", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.title("Embedding Similarities in Contrastive Learning", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "contrastive_similarities.pdf"), format='pdf')
    plt.close()
    
    plt.figure(figsize=(10, 6), dpi=300)
    norms = [intact_norm, perturbed_norm, text_norm]
    labels = ['Intact Image', 'Perturbed Image', 'Text']
    sns.barplot(x=labels, y=norms)
    plt.xlabel("Embedding Type", fontsize=12)
    plt.ylabel("L2 Norm", fontsize=12)
    plt.title("Embedding Norms in Contrastive Learning", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "contrastive_norms.pdf"), format='pdf')
    plt.close()
    
    metrics = {
        "intact_text_similarity": intact_text_sim,
        "perturbed_text_similarity": perturbed_text_sim,
        "intact_perturbed_similarity": intact_perturbed_sim,
        "intact_embedding_norm": intact_norm,
        "perturbed_embedding_norm": perturbed_norm,
        "text_embedding_norm": text_norm,
        "contrast_ratio": intact_text_sim / (perturbed_text_sim + 1e-8)
    }
    
    print("Contrastive Learning Evaluation:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def evaluate_all_experiments(exp1_results, exp2_results, exp3_results, save_dir="logs"):
    """
    Evaluate and compare results from all three experiments.
    
    Args:
        exp1_results: Results from Experiment 1
        exp2_results: Results from Experiment 2
        exp3_results: Results from Experiment 3
        save_dir: Directory to save evaluation results
    
    Returns:
        dict: Combined evaluation metrics
    """
    summary = {
        "experiment1": {
            "avg_cosine_similarity": exp1_results["decoder_metrics"]["avg_cosine_similarity"] 
                if "decoder_metrics" in exp1_results else None
        },
        "experiment2": {
            "tokens_above_threshold": exp2_results["grounding_metrics"]["tokens_above_threshold"] 
                if "grounding_metrics" in exp2_results else None
        },
        "experiment3": {
            "contrast_ratio": exp3_results["contrastive_metrics"]["contrast_ratio"] 
                if "contrastive_metrics" in exp3_results else None
        }
    }
    
    print("\n=== VG-DD Experiments Summary ===")
    for exp_name, metrics in summary.items():
        print(f"\n{exp_name}:")
        for metric_name, value in metrics.items():
            if value is not None:
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: N/A")
    
    plt.figure(figsize=(15, 10), dpi=300)
    
    plt.subplot(2, 2, 1)
    plt.title("Experiment 1: Adaptive Visual Prompt Weighting", fontsize=12)
    if "cosine_vals" in exp1_results:
        sns.lineplot(x=range(1, len(exp1_results["cosine_vals"])+1), 
                    y=exp1_results["cosine_vals"], marker='o')
        plt.xlabel("Decoding Step", fontsize=10)
        plt.ylabel("Cosine Similarity", fontsize=10)
    else:
        plt.text(0.5, 0.5, "Data not available", ha='center', va='center')
    
    plt.subplot(2, 2, 2)
    plt.title("Experiment 2: Joint Decoding with Feedback Loop", fontsize=12)
    if "final_groundings" in exp2_results:
        sns.barplot(x=list(range(1, len(exp2_results["final_groundings"])+1)), 
                   y=exp2_results["final_groundings"])
        plt.xlabel("Token Position", fontsize=10)
        plt.ylabel("Grounding Score", fontsize=10)
    else:
        plt.text(0.5, 0.5, "Data not available", ha='center', va='center')
    
    plt.subplot(2, 2, 3)
    plt.title("Experiment 3: Contrastive Learning", fontsize=12)
    if "loss_list" in exp3_results:
        sns.lineplot(x=range(1, len(exp3_results["loss_list"])+1), 
                    y=exp3_results["loss_list"], marker='o')
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Contrastive Loss", fontsize=10)
    else:
        plt.text(0.5, 0.5, "Data not available", ha='center', va='center')
    
    plt.subplot(2, 2, 4)
    plt.title("VG-DD: Combined Performance", fontsize=12)
    metrics_names = ["Exp1: Cosine Sim", "Exp2: Grounding", "Exp3: Contrast"]
    metrics_values = [
        summary["experiment1"]["avg_cosine_similarity"] if summary["experiment1"]["avg_cosine_similarity"] else 0,
        summary["experiment2"]["tokens_above_threshold"] if summary["experiment2"]["tokens_above_threshold"] else 0,
        summary["experiment3"]["contrast_ratio"] if summary["experiment3"]["contrast_ratio"] else 0
    ]
    sns.barplot(x=metrics_names, y=metrics_values)
    plt.xlabel("Metric", fontsize=10)
    plt.ylabel("Value", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vgdd_combined_results.pdf"), format='pdf')
    plt.close()
    
    return summary
