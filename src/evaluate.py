"""
Evaluation module for the STEM method.
Provides functions for model evaluation and result visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
from src.preprocess import extract_embeddings

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

def evaluate_model(model, data):
    """
    Evaluate model over provided data.
    Assumes data is of the form (text, true_alpha, ...)
    Returns MSE between model output and zero (dummy target) as demonstration.
    """
    pred_list = []
    true_list = []
    with torch.no_grad():
        for sample in data:
            text = sample[0]
            true_alpha = sample[1]
            embeddings = extract_embeddings([text])
            alpha_tensor = true_alpha.view(-1, 1)
            output = model(embeddings, alpha_tensor)
            if isinstance(output, tuple):  # For models with metadata
                output = output[0]
            pred_list.append(output.item())
            true_list.append(0.0)  # dummy true target for demonstration
    mse_val = mean_squared_error(true_list, pred_list)
    print("Evaluation MSE: {:.4f}".format(mse_val))
    return mse_val

def plot_training_curve(loss_history, figure_topic="training_loss", condition=None, pair=None):
    """
    Plot the training loss curve using matplotlib and save as .pdf
    """
    plt.figure()
    epochs = list(range(1, len(loss_history)+1))
    sns.lineplot(x=epochs, y=loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    title = "Training Loss"
    if condition is not None:
        title += f" ({condition})"
    plt.title(title)
    filename = f"logs/{figure_topic}"
    if condition is not None:
        filename += f"_{condition}"
    if pair is not None:
        filename += f"_pair{pair}"
    filename += ".pdf"
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved training curve as {filename}")
    plt.close()  # close figure
    
    return filename

def plot_domain_shift_results(targets, predictions, filename="inference_latency_domain_pair1.pdf"):
    """
    Plot domain evaluation: scatter plot of predictions against targets
    """
    plt.figure()
    sns.scatterplot(x=targets, y=predictions)
    plt.xlabel("True Value (dummy zero)")
    plt.ylabel("Predicted Value")
    plt.title("Domain Shift Predictions (Academic)")
    filepath = f"logs/{filename}"
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Saved domain shift scatter plot as {filepath}")
    plt.close()
    
    return filepath
