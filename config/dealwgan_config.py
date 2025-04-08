"""
Configuration file for DEALWGAN experiments.
"""

class DEALWGANConfig:
    """Configuration for DEALWGAN experiments"""
    
    seed = 42
    dataset = "cifar10"
    batch_size = 64
    num_workers = 4
    
    latent_dim = 128
    adaptive_latent = True
    use_diffusion = True
    
    num_epochs = 30
    lr_gen = 0.0002
    lr_disc = 0.0002
    lr_enc = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    
    diffusion_steps = 10
    noise_schedule = "linear"
    step_size = 0.05
    
    variant_a = {"adaptive_latent": True, "use_diffusion": True}
    variant_b = {"adaptive_latent": True, "use_diffusion": False}
    variant_c = {"adaptive_latent": False, "use_diffusion": True}
    
    eval_interval = 5
    sample_size = 500
    
    device = "cuda"
