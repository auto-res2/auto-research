import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random

class Generator(nn.Module):
    def __init__(self, img_size=128, channels=3):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Discriminator(nn.Module):
    def __init__(self, img_size=128, channels=3):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        
        self.main = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

class ContextConditionalGAN:
    def __init__(self, img_size=128, channels=3, lr=0.0001, device='cuda'):
        self.img_size = img_size
        self.channels = channels
        self.lr = lr
        self.device = device
        
        self.generator = Generator(img_size, channels).to(device)
        self.discriminator = Discriminator(img_size, channels).to(device)
        
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        self.criterion = nn.BCELoss()
        self.l1_criterion = nn.L1Loss()
        
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)
        
    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def create_mask(self, batch_size, mask_size_range=(16, 64)):
        """Create random square masks for in-painting"""
        masks = torch.ones(batch_size, 1, self.img_size, self.img_size)
        
        for i in range(batch_size):
            mask_size = random.randint(mask_size_range[0], mask_size_range[1])
            
            x = random.randint(0, self.img_size - mask_size)
            y = random.randint(0, self.img_size - mask_size)
            
            masks[i, :, y:y+mask_size, x:x+mask_size] = 0
            
        return masks.to(self.device)
    
    def apply_mask(self, images, masks):
        """Apply masks to images to create holes"""
        return images * masks
    
    def train_step(self, real_images):
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        
        masks = self.create_mask(batch_size)
        masked_images = self.apply_mask(real_images, masks)
        
        real_labels = torch.ones(batch_size).to(self.device)
        fake_labels = torch.zeros(batch_size).to(self.device)
        
        self.optimizer_D.zero_grad()
        
        real_output = self.discriminator(real_images)
        d_loss_real = self.criterion(real_output, real_labels)
        
        fake_images = self.generator(masked_images)
        fake_output = self.discriminator(fake_images.detach())
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.optimizer_D.step()
        
        self.optimizer_G.zero_grad()
        
        fake_images = self.generator(masked_images)
        
        fake_output = self.discriminator(fake_images)
        g_loss_adv = self.criterion(fake_output, real_labels)
        
        g_loss_l1 = self.l1_criterion(fake_images * (1 - masks), real_images * (1 - masks))
        
        g_loss = g_loss_adv + 100 * g_loss_l1  # L1 loss weight
        g_loss.backward()
        self.optimizer_G.step()
        
        return {
            'g_loss': g_loss.item(),
            'g_loss_adv': g_loss_adv.item(),
            'g_loss_l1': g_loss_l1.item(),
            'd_loss': d_loss.item(),
            'd_loss_real': d_loss_real.item(),
            'd_loss_fake': d_loss_fake.item()
        }
    
    def save_sample_images(self, real_images, epoch, save_dir='samples'):
        """Save sample in-painted images"""
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            batch_size = min(8, real_images.size(0))
            real_images = real_images[:batch_size].to(self.device)
            
            masks = self.create_mask(batch_size)
            masked_images = self.apply_mask(real_images, masks)
            
            fake_images = self.generator(masked_images)
            
            completed_images = real_images * masks + fake_images * (1 - masks)
            
            real_np = real_images.cpu().numpy()
            masked_np = masked_images.cpu().numpy()
            fake_np = fake_images.cpu().numpy()
            completed_np = completed_images.cpu().numpy()
            
            fig, axes = plt.subplots(4, batch_size, figsize=(batch_size * 2, 8))
            
            for i in range(batch_size):
                axes[0, i].imshow(np.transpose(real_np[i], (1, 2, 0)) * 0.5 + 0.5)
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                axes[1, i].imshow(np.transpose(masked_np[i], (1, 2, 0)) * 0.5 + 0.5)
                axes[1, i].set_title('Masked')
                axes[1, i].axis('off')
                
                axes[2, i].imshow(np.transpose(fake_np[i], (1, 2, 0)) * 0.5 + 0.5)
                axes[2, i].set_title('Generated')
                axes[2, i].axis('off')
                
                axes[3, i].imshow(np.transpose(completed_np[i], (1, 2, 0)) * 0.5 + 0.5)
                axes[3, i].set_title('Completed')
                axes[3, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
            plt.close()

def get_dataloader(data_path='./data', batch_size=16, img_size=128):
    """Create dataloader for training"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    except:
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader

def train_ccgan(lr=0.0001, n_epochs=100, batch_size=16, img_size=128, data_path='./data'):
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ccgan = ContextConditionalGAN(img_size=img_size, lr=lr, device=device)
    
    dataloader = get_dataloader(data_path, batch_size, img_size)
    
    print("Starting training...")
    for epoch in range(n_epochs):
        epoch_losses = {'g_loss': 0, 'g_loss_adv': 0, 'g_loss_l1': 0, 
                       'd_loss': 0, 'd_loss_real': 0, 'd_loss_fake': 0}
        
        for i, (real_images, _) in enumerate(dataloader):
            losses = ccgan.train_step(real_images)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}] Batch [{i}/{len(dataloader)}] "
                      f"G_loss: {losses['g_loss']:.4f} D_loss: {losses['d_loss']:.4f}")
        
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
        
        print(f"Epoch [{epoch+1}/{n_epochs}] Average Losses:")
        print(f"  Generator: {epoch_losses['g_loss']:.4f} (Adv: {epoch_losses['g_loss_adv']:.4f}, L1: {epoch_losses['g_loss_l1']:.4f})")
        print(f"  Discriminator: {epoch_losses['d_loss']:.4f} (Real: {epoch_losses['d_loss_real']:.4f}, Fake: {epoch_losses['d_loss_fake']:.4f})")
        
        if (epoch + 1) % 10 == 0:
            sample_batch = next(iter(dataloader))[0]
            ccgan.save_sample_images(sample_batch, epoch + 1)
        
        if (epoch + 1) % 25 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': ccgan.generator.state_dict(),
                'discriminator_state_dict': ccgan.discriminator.state_dict(),
                'optimizer_G_state_dict': ccgan.optimizer_G.state_dict(),
                'optimizer_D_state_dict': ccgan.optimizer_D.state_dict(),
            }, f'checkpoints/ccgan_epoch_{epoch+1:03d}.pth')
    
    print("Training completed!")
    
    torch.save({
        'generator_state_dict': ccgan.generator.state_dict(),
        'discriminator_state_dict': ccgan.discriminator.state_dict(),
    }, 'ccgan_final.pth')

if __name__ == "__main__":
    lr = 0.0001
    n_epochs = 100
    batch_size = 16
    img_size = 128
    
    train_ccgan(lr=lr, n_epochs=n_epochs, batch_size=batch_size, img_size=img_size)
