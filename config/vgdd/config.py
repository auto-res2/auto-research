"""
Configuration file for Visual Grounding and Dynamic Decoding (VG-DD) experiments.
"""

RANDOM_SEED = 42
DEVICE = "cuda"  # Use GPU for experiments

EXPERIMENT1_CONFIG = {
    "vocab_size": 1000,
    "embed_dim": 512,
    "visual_dim": 2048,  # ResNet50 feature dimension
    "batch_size": 4,
    "test_batch_size": 1,  # For quick test runs
    "image_size": 224,
    "save_path": "logs/cosine_similarity_adaptive_pair1.pdf"
}

EXPERIMENT2_CONFIG = {
    "num_iterations": 3,
    "threshold": 0.5,
    "embed_dim": 768,  # GPT-2 embedding dimension
    "test_prompt": "A man is riding a horse in a field.",
    "save_path": "logs/grounding_scores_iterative_pair1.pdf"
}

EXPERIMENT3_CONFIG = {
    "visual_module_out_dim": 2048,
    "language_model_embed_dim": 768,
    "num_epochs": 5,
    "learning_rate": 1e-4,
    "save_path": "logs/contrastive_loss_training.pdf"
}

T4_OPTIMIZATION = {
    "batch_size": 16,
    "mixed_precision": True,
    "memory_efficient_attention": True,
    "gradient_checkpointing": True,
    "max_tokens": 512
}
