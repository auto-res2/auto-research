# ClusterCloak: A Novel Poisoning Defense Method

ClusterCloak is a novel poisoning defense method that builds on the MetaCloak meta-learning framework while integrating ideas from meta-learning-inspired regularization, such as consensus weight and feature clustering. The key innovation is to guide the bi-level poisoning optimization not only to induce robust semantic degradation under common transformations (as in MetaCloak) but also to force the internal representations (feature and weight spaces) of surrogate diffusion models into adversarially "misaligned clusters."

## Methodology

ClusterCloak works through:

1. **Dual-Loss Optimization Framework**:
   - Minimizes a denoising-error maximization loss computed over surrogate diffusion models with an embedded transformation sampling process
   - Introduces an auxiliary clustering regularizer that encourages feature representations to group into tight, misleading clusters

2. **Consensus Clustering Across Models**:
   - Aggregates clustering gradients computed on multiple surrogate diffusion models
   - Creates a "consensus clustering" to yield common clustered feature distortion

3. **Training and Robustness**:
   - Uses an unrolling strategy with PGD-style updates
   - Targets both the output quality and the structure of internal representations

## Experiments

The implementation includes these experiments:

1. Feature Misalignment and Clustering Analysis
2. Robust Poisoning Against Fine-Tuning Recovery
3. Robustness Under Common Transformations and Purification Attacks

## Usage

Run the experiment using:

```bash
python src/main.py
```

The results and plots will be saved in the `logs` directory in PDF format suitable for academic papers.
