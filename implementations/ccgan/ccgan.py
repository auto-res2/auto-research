import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 128
CHANNELS = 3
Z_DIM = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, channels_img=3, features_g=64):
        super(Generator, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(channels_img, features_g, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_g, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_g * 2, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_g * 4, features_g * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_g * 8, features_g * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features_g * 8, features_g * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(features_g, channels_img, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Discriminator(nn.Module):
    def __init__(self, channels_img=3, features_d=64):
        super(Discriminator, self).__init__()
        
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(features_d * 8, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.disc(x).view(-1, 1).squeeze(1)

def create_mask(batch_size, image_size, mask_size_range=(32, 64)):
    """Create random square masks for context-conditional training"""
    masks = torch.ones(batch_size, 1, image_size, image_size)
    
    for i in range(batch_size):
        mask_size = random.randint(mask_size_range[0], mask_size_range[1])
        
        max_pos = image_size - mask_size
        start_x = random.randint(0, max_pos)
        start_y = random.randint(0, max_pos)
        
        masks[i, 0, start_y:start_y+mask_size, start_x:start_x+mask_size] = 0
    
    return masks

def apply_mask(images, masks):
    """Apply masks to images (set masked regions to 0)"""
    return images * masks

def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_sample_images(generator, test_loader, epoch, device, save_dir="samples"):
    """Save sample inpainting results"""
    os.makedirs(save_dir, exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        test_images, _ = next(iter(test_loader))
        test_images = test_images[:4].to(device)  # Take first 4 images
        
        masks = create_mask(4, IMAGE_SIZE).to(device)
        masked_images = apply_mask(test_images, masks)
        
        generated = generator(masked_images)
        
        inpainted = masked_images + generated * (1 - masks)
        
        comparison = torch.cat([
            test_images,      # Original
            masked_images,    # Masked
            inpainted        # Inpainted
        ], dim=0)
        
        torchvision.utils.save_image(
            comparison, 
            f"{save_dir}/epoch_{epoch:03d}.png",
            nrow=4, 
            normalize=True, 
            value_range=(-1, 1)
        )
    
    generator.train()

def main():
    print(f"Using device: {DEVICE}")
    print(f"Training Context-Conditional GAN with:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Image Size: {IMAGE_SIZE}")
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    try:
        dataset = torchvision.datasets.CIFAR10(
            root="./data", 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", 
            train=False, 
            download=True, 
            transform=transform
        )
    except:
        print("Error loading CIFAR-10. You may need to install torchvision or use a different dataset.")
        return
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    generator = Generator(channels_img=CHANNELS).to(DEVICE)
    discriminator = Discriminator(channels_img=CHANNELS).to(DEVICE)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    criterion = nn.BCELoss()
    opt_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    print("Starting training...")
    
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)
            
            masks = create_mask(batch_size, IMAGE_SIZE).to(DEVICE)
            masked_images = apply_mask(real_images, masks)
            
            real_label = torch.ones(batch_size).to(DEVICE)
            fake_label = torch.zeros(batch_size).to(DEVICE)
            
            opt_disc.zero_grad()
            
            output_real = discriminator(real_images)
            loss_disc_real = criterion(output_real, real_label)
            
            fake_images = generator(masked_images)
            inpainted_images = masked_images + fake_images * (1 - masks)
            output_fake = discriminator(inpainted_images.detach())
            loss_disc_fake = criterion(output_fake, fake_label)
            
            loss_disc = loss_disc_real + loss_disc_fake
            loss_disc.backward()
            opt_disc.step()
            
            opt_gen.zero_grad()
            
            output_fake = discriminator(inpainted_images)
            loss_gen_adv = criterion(output_fake, real_label)
            
            loss_gen_recon = nn.L1Loss()(fake_images * (1 - masks), real_images * (1 - masks))
            
            loss_gen = loss_gen_adv + 100 * loss_gen_recon  # Weight reconstruction loss heavily
            loss_gen.backward()
            opt_gen.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")
        
        if (epoch + 1) % 10 == 0:
            save_sample_images(generator, test_loader, epoch + 1, DEVICE)
            print(f"Saved sample images for epoch {epoch + 1}")
    
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(generator.state_dict(), "checkpoints/generator_final.pth")
    torch.save(discriminator.state_dict(), "checkpoints/discriminator_final.pth")
    
    print("Training completed!")
    print("Models saved to checkpoints/")
    print("Sample images saved to samples/")

if __name__ == "__main__":
    main()
