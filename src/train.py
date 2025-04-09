import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import torch.nn.functional as F

class FPDMTeacher(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=3):
        super(FPDMTeacher, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(inplace=True),
        )
        
        self.fixed_point_layer = nn.Sequential(
            nn.Conv2d(hidden_channels*2, hidden_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels*2, hidden_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels*2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )
        
    def forward(self, x, steps=10):
        """Forward pass with fixed point iterations.
        
        Args:
            x: Input tensor
            steps: Number of fixed point iterations
            
        Returns:
            Tensor: Output after fixed point iterations
        """
        features = self.encoder(x)
        
        for _ in range(steps):
            features = self.fixed_point_layer(features)
            
        output = self.decoder(features)
        return output

class OneStepStudent(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=3):
        super(OneStepStudent, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels*2, hidden_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels*2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        """One-step forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor: Output tensor
        """
        return self.model(x)

class OneStepStudentAux(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=3):
        super(OneStepStudentAux, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(inplace=True),
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(hidden_channels*2, hidden_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(inplace=True),
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_channels*2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )
        
    def forward(self, x, return_intermediates=False):
        """Forward pass with optional intermediate outputs.
        
        Args:
            x: Input tensor
            return_intermediates: Whether to return intermediate features
            
        Returns:
            Tensor or Tuple of tensors: Output tensor and optionally intermediate features
        """
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        middle_out = self.middle(enc2_out)
        dec1_out = self.dec1(middle_out)
        output = self.dec2(dec1_out)
        
        if return_intermediates:
            return output, (enc1_out, enc2_out, middle_out, dec1_out)
        return output

def train_teacher(teacher, dataloader, optimizer, device, epochs=1):
    """Train the FPDM Teacher model.
    
    Args:
        teacher: Teacher model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of epochs
        
    Returns:
        List: Training losses
    """
    teacher.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} (Teacher)")
        
        for images, _ in loop:
            images = images.to(device)
            optimizer.zero_grad()
            
            output = teacher(images, steps=10)
            
            loss = F.mse_loss(output, images)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            loop.set_postfix(loss=loss.item())
            
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Avg loss: {avg_loss:.4f}")
        
    return losses

def train_student(student, teacher, dataloader, optimizer, device, epochs=1):
    """Train the One-Step Student model.
    
    Args:
        student: Student model
        teacher: Teacher model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of epochs
        
    Returns:
        List: Training losses
    """
    student.train()
    teacher.eval()
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} (Student)")
        
        for images, _ in loop:
            images = images.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_output = teacher(images, steps=10)
                
            student_output = student(images)
            
            loss = F.mse_loss(student_output, teacher_output)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            loop.set_postfix(loss=loss.item())
            
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Avg loss: {avg_loss:.4f}")
        
    return losses

def train_student_with_aux(student_aux, teacher, dataloader, optimizer, device, epochs=1, aux_weight=0.1):
    """Train the One-Step Student model with auxiliary supervision.
    
    Args:
        student_aux: Student model with auxiliary outputs
        teacher: Teacher model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of epochs
        aux_weight: Weight for auxiliary loss
        
    Returns:
        Tuple: Training losses (total, main, aux)
    """
    student_aux.train()
    teacher.eval()
    total_losses = []
    main_losses = []
    aux_losses = []
    
    for epoch in range(epochs):
        epoch_total_losses = []
        epoch_main_losses = []
        epoch_aux_losses = []
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} (Student+Aux)")
        
        for images, _ in loop:
            images = images.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_output = teacher(images, steps=10)
                teacher_features = teacher.encoder(images)
                
            student_output, student_interm = student_aux(images, return_intermediates=True)
            
            main_loss = F.mse_loss(student_output, teacher_output)
            
            aux_loss = F.mse_loss(student_interm[0], teacher_features)
                
            total_loss = main_loss + aux_weight * aux_loss
            total_loss.backward()
            optimizer.step()
            
            total_loss_val = float(total_loss.item() if hasattr(total_loss, 'item') else total_loss)
            main_loss_val = float(main_loss.item() if hasattr(main_loss, 'item') else main_loss)
            aux_loss_val = float(aux_loss)
            
            epoch_total_losses.append(total_loss_val)
            epoch_main_losses.append(main_loss_val)
            epoch_aux_losses.append(aux_loss_val)
            
            loop.set_postfix(
                total_loss=total_loss_val,
                main_loss=main_loss_val,
                aux_loss=aux_loss_val
            )
            
        total_losses.append(sum(epoch_total_losses) / len(epoch_total_losses))
        main_losses.append(sum(epoch_main_losses) / len(epoch_main_losses))
        aux_losses.append(sum(epoch_aux_losses) / len(epoch_aux_losses))
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Total loss: {total_losses[-1]:.4f}, "
              f"Main loss: {main_losses[-1]:.4f}, "
              f"Aux loss: {aux_losses[-1]:.4f}")
        
    return total_losses, main_losses, aux_losses

def save_model(model, path):
    """Save model to disk.
    
    Args:
        model: PyTorch model
        path: Path to save to
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
