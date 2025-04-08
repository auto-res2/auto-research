"""
Implementation of DEALWGAN and LWGAN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    """Encoder network for mapping inputs to latent space."""
    
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False)  # 64 x 16 x 16
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)  # 128 x 8 x 8
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)  # 256 x 4 x 4
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False)  # 512 x 2 x 2
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.fc = nn.Linear(512 * 2 * 2, latent_dim)
        
        self.selection_matrix = nn.Parameter(torch.ones(latent_dim))
        
    def forward(self, x, adaptive=True):
        print(f"Encoder input shape: {x.shape}")
        
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension [batch_size, 1, height, width]
            print(f"Encoder after unsqueeze: {x.shape}")
        
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            print(f"Encoder after repeat: {x.shape}")
            
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.lrelu4(self.bn4(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        latent = self.fc(x)
        
        if adaptive:
            latent = latent * self.selection_matrix
            
        return latent

class Generator(nn.Module):
    """Generator network for mapping latent vectors to images."""
    
    def __init__(self, latent_dim=128):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)
        self.relu = nn.ReLU(inplace=True)
        
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)  # 256 x 4 x 4
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)  # 128 x 8 x 8
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)  # 64 x 16 x 16
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.deconv4 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False)  # 3 x 32 x 32
        self.tanh = nn.Tanh()
        
    def forward(self, z):
        print(f"Generator input shape: {z.shape}")
        
        x = self.relu(self.fc(z))
        x = x.view(x.size(0), 512, 2, 2)
        
        x = self.relu1(self.bn1(self.deconv1(x)))
        x = self.relu2(self.bn2(self.deconv2(x)))
        x = self.relu3(self.bn3(self.deconv3(x)))
        x = self.tanh(self.deconv4(x))
        
        print(f"Generator output shape: {x.shape}")
        
        return x

class Critic(nn.Module):
    """Critic network for WGAN."""
    
    def __init__(self):
        super(Critic, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False)  # 64 x 16 x 16
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)  # 128 x 8 x 8
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)  # 256 x 4 x 4
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False)  # 512 x 2 x 2
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.fc = nn.Linear(512 * 2 * 2, 1)
        
    def forward(self, x):
        print(f"Critic input shape: {x.shape}")
        
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension [batch_size, 1, height, width]
            print(f"After unsqueeze: {x.shape}")
        
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            print(f"After repeat: {x.shape}")
            
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.lrelu4(self.bn4(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LatentDiffusion:
    """
    Latent space diffusion module for refining the latent representation.
    """
    
    def __init__(self, config):
        self.config = config
        self.steps = config.diffusion_steps
        self.noise_schedule = config.noise_schedule
        self.step_size = config.step_size
        
    def _get_noise_schedule(self, t):
        """Get the noise level at time t."""
        if self.noise_schedule == "linear":
            return t
        elif self.noise_schedule == "cosine":
            return 0.5 * (1 + torch.cos(t * torch.pi))
        else:
            return t
    
    def add_noise(self, x, t):
        """Add noise to the latent representation according to the schedule."""
        noise_level = self._get_noise_schedule(t)
        noise = torch.randn_like(x)
        return x * (1 - noise_level) + noise * noise_level, noise
    
    def denoise_step(self, x_noisy, t, score_fn):
        """Perform a single denoising step."""
        score = score_fn(x_noisy, t)
        
        noise_level = self._get_noise_schedule(t)
        next_t = torch.clamp(t - self.step_size, min=0.0)
        next_noise_level = self._get_noise_schedule(next_t)
        
        alpha = (1 - next_noise_level) / (1 - noise_level)
        sigma = self.step_size * torch.sqrt(next_noise_level)
        
        x_new = alpha * (x_noisy + self.step_size * score) + sigma * torch.randn_like(x_noisy)
        
        return x_new, next_t
    
    def refine_latent(self, z, score_fn):
        """Refine the latent representation using the diffusion process."""
        t = torch.ones(1, device=z.device)
        x_t = z.clone()
        
        for _ in range(self.steps):
            x_t, t = self.denoise_step(x_t, t, score_fn)
            
        return x_t

class LWGAN:
    """
    Latent Wasserstein GAN baseline model.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.latent_dim = config.latent_dim
        self.adaptive_latent = config.adaptive_latent
        
        self.encoder = Encoder(self.latent_dim).to(self.device)
        self.generator = Generator(self.latent_dim).to(self.device)
        self.critic = Critic().to(self.device)
        
        self.opt_enc = torch.optim.Adam(
            self.encoder.parameters(), 
            lr=config.lr_enc,
            betas=(config.beta1, config.beta2)
        )
        self.opt_gen = torch.optim.Adam(
            self.generator.parameters(), 
            lr=config.lr_gen,
            betas=(config.beta1, config.beta2)
        )
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(), 
            lr=config.lr_disc,
            betas=(config.beta1, config.beta2)
        )
        
        self.step_count = 0
        
    def train_critic(self, real_imgs):
        """Train the critic for one step."""
        batch_size = real_imgs.size(0)
        
        print(f"train_critic input shape: {real_imgs.shape}")
        
        if real_imgs.dim() == 3:
            real_imgs = real_imgs.unsqueeze(1)  # Add channel dimension
            print(f"train_critic after unsqueeze: {real_imgs.shape}")
        
        if real_imgs.size(1) == 1:
            real_imgs = real_imgs.repeat(1, 3, 1, 1)
            print(f"train_critic after repeat: {real_imgs.shape}")
        
        with torch.no_grad():
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_imgs = self.generator(z)
        
        real_validity = self.critic(real_imgs)
        fake_validity = self.critic(fake_imgs)
        
        critic_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
        interp_validity = self.critic(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=interp_validity,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interp_validity),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        penalty_weight = 10.0
        
        critic_loss += penalty_weight * gradient_penalty
        
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()
        
        return critic_loss.item()
    
    def train_generator(self, real_imgs):
        """Train the generator and encoder for one step."""
        batch_size = real_imgs.size(0)
        
        print(f"train_generator input shape: {real_imgs.shape}")
        
        if real_imgs.dim() == 3:
            real_imgs = real_imgs.unsqueeze(1)  # Add channel dimension
            print(f"train_generator after unsqueeze: {real_imgs.shape}")
        
        if real_imgs.size(1) == 1:
            real_imgs = real_imgs.repeat(1, 3, 1, 1)
            print(f"train_generator after repeat: {real_imgs.shape}")
            
        z_real = self.encoder(real_imgs, self.adaptive_latent)
        
        z_random = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_imgs = self.generator(z_random)
            
        reconstructed_imgs = self.generator(z_real)
            
        fake_validity = self.critic(fake_imgs)
        g_loss = -torch.mean(fake_validity)
        
        rec_loss = F.mse_loss(reconstructed_imgs, real_imgs)
        
        generator_loss = g_loss + 10.0 * rec_loss
        
        self.opt_gen.zero_grad()
        self.opt_enc.zero_grad()
        generator_loss.backward()
        self.opt_gen.step()
        self.opt_enc.step()
        
        self.step_count += 1
        
        return generator_loss.item()
    
    def train_step(self, batch):
        """Perform a full training step."""
        if isinstance(batch, (list, tuple)):
            real_imgs = batch[0].to(self.device)
        else:
            real_imgs = batch.to(self.device)
        
        print(f"train_step input shape: {real_imgs.shape}")
        
        if real_imgs.dim() == 3:
            real_imgs = real_imgs.unsqueeze(1)  # Add channel dimension
            print(f"train_step after unsqueeze: {real_imgs.shape}")
        
        if real_imgs.size(1) != 3:
            print(f"Warning: Expected 3 channels, got {real_imgs.size(1)}")
            
            if real_imgs.size(1) == 1:
                real_imgs = real_imgs.repeat(1, 3, 1, 1)
                print(f"After repeating: {real_imgs.shape}")
        
        for _ in range(1):  # Reduced from 5 to 1 for testing
            critic_loss = self.train_critic(real_imgs)
        
        generator_loss = self.train_generator(real_imgs)
        
        return generator_loss + critic_loss
    
    def generate_samples(self, num_samples=64):
        """Generate samples from random noise."""
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        with torch.no_grad():
            samples = self.generator(z)
        return samples
    
    def get_latent_representations(self, dataloader, max_batches=3):
        """Get latent representations for data samples."""
        latent_list = []
        with torch.no_grad():
            for i, (imgs, _) in enumerate(dataloader):
                if i >= max_batches:
                    break
                imgs = imgs.to(self.device)
                
                print(f"get_latent_representations input shape: {imgs.shape}")
                
                if imgs.dim() == 3:
                    imgs = imgs.unsqueeze(1)  # Add channel dimension
                    print(f"get_latent_representations after unsqueeze: {imgs.shape}")
                
                if imgs.size(1) == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)
                    print(f"get_latent_representations after repeat: {imgs.shape}")
                    
                latents = self.encoder(imgs, self.adaptive_latent)
                latent_list.append(latents.cpu())
                
                break
                
        if latent_list:
            return torch.cat(latent_list, dim=0).numpy()
        else:
            return np.zeros((0, self.latent_dim))

class DEALWGAN(LWGAN):
    """
    Diffusion-Enhanced Adaptive Latent Wasserstein GAN.
    Extends LWGAN with a latent space diffusion refinement process.
    """
    
    def __init__(self, config):
        super(DEALWGAN, self).__init__(config)
        self.use_diffusion = config.use_diffusion
        
        if self.use_diffusion:
            self.diffusion = LatentDiffusion(config)
    
    def score_function(self, z_noisy, t):
        """
        Score function for the diffusion process.
        Approximates the gradient of the log probability density.
        """
        z_denoised = self.generator(z_noisy)
        z_reconstructed = self.encoder(z_denoised, self.adaptive_latent)
        
        t_expanded = t.expand(z_noisy.size(0), 1)
        score = (z_reconstructed - z_noisy) / (t_expanded + 1e-5)
        return score
    
    def train_generator(self, real_imgs):
        """Train the generator and encoder with diffusion refinement."""
        batch_size = real_imgs.size(0)
        
        print(f"DEALWGAN train_generator input shape: {real_imgs.shape}")
        
        if real_imgs.dim() == 3:
            real_imgs = real_imgs.unsqueeze(1)  # Add channel dimension
            print(f"DEALWGAN train_generator after unsqueeze: {real_imgs.shape}")
        
        if real_imgs.size(1) == 1:
            real_imgs = real_imgs.repeat(1, 3, 1, 1)
            print(f"DEALWGAN train_generator after repeat: {real_imgs.shape}")
            
        z_real = self.encoder(real_imgs, self.adaptive_latent)
        
        if self.use_diffusion:
            z_refined = self.diffusion.refine_latent(z_real, self.score_function)
        else:
            z_refined = z_real
        
        z_random = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_imgs = self.generator(z_random)
            
        reconstructed_imgs = self.generator(z_refined)
            
        fake_validity = self.critic(fake_imgs)
        g_loss = -torch.mean(fake_validity)
        
        rec_loss = F.mse_loss(reconstructed_imgs, real_imgs)
        
        diffusion_loss = 0.0
        if self.use_diffusion:
            diffusion_loss = F.mse_loss(z_refined, z_real) * 0.1
        
        generator_loss = g_loss + 10.0 * rec_loss + diffusion_loss
        
        self.opt_gen.zero_grad()
        self.opt_enc.zero_grad()
        generator_loss.backward()
        self.opt_gen.step()
        self.opt_enc.step()
        
        self.step_count += 1
        
        return generator_loss.item()
    
    def generate_samples(self, num_samples=64):
        """Generate samples with diffusion refinement."""
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        
        if self.use_diffusion:
            z = self.diffusion.refine_latent(z, self.score_function)
        
        with torch.no_grad():
            samples = self.generator(z)
        return samples
        
    def get_latent_representations(self, dataloader, max_batches=3):
        """Get latent representations for data samples."""
        latent_list = []
        with torch.no_grad():
            for i, (imgs, _) in enumerate(dataloader):
                if i >= max_batches:
                    break
                imgs = imgs.to(self.device)
                
                print(f"DEALWGAN get_latent_representations input shape: {imgs.shape}")
                
                if imgs.dim() == 3:
                    imgs = imgs.unsqueeze(1)  # Add channel dimension
                    print(f"DEALWGAN get_latent_representations after unsqueeze: {imgs.shape}")
                
                if imgs.size(1) == 1:
                    imgs = imgs.repeat(1, 3, 1, 1)
                    print(f"DEALWGAN get_latent_representations after repeat: {imgs.shape}")
                    
                latents = self.encoder(imgs, self.adaptive_latent)
                latent_list.append(latents.cpu())
                
                break
                
        if latent_list:
            return torch.cat(latent_list, dim=0).numpy()
        else:
            return np.zeros((0, self.latent_dim))
