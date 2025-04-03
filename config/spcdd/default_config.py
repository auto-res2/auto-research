"""Configuration for SPCDD experiment."""

experiment_name = "spcdd_mri_superres"
random_seed = 42
device = "cuda"  # Use GPU for training

image_size = 64
batch_size = 8
num_workers = 4
synthetic_dataset_size = 20  # Small size for quick testing

anatomy_extractor_channels = [1, 16, 32]
anatomy_lr = 1e-4

diffusion_channels = [2, 64, 64, 1]  # [input, hidden1, hidden2, output]
diffusion_lr = 1e-4
gradient_loss_weight = 0.1

teacher_channels = [2, 64, 64, 1]
student_channels = [1, 32, 32, 1]
distillation_alpha = 0.5  # Weight for output-level loss
distillation_beta = 0.5   # Weight for feature-level loss
student_lr = 1e-4

num_epochs = {
    "ablation": 1,       # For quick testing
    "intensity": 1,      # For quick testing
    "distillation": 1    # For quick testing
}
save_interval = 1

metrics = ["psnr", "ssim"]

use_anatomy_prior = True
use_intensity_modulation = True
