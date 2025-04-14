"""
Training script for the Progressive Brightness Distillation Diffusion experiment.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.pbd_diffusion_config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    OUTPUT_DIR,
    LOGS_DIR,
    MODELS_DIR
)
from src.utils.models import TeacherModel, StudentModel
from src.utils.visualization import plot_loss

def train_teacher_student_model(train_loader, test_loader=None, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """
    Train a teacher-student model for progressive distillation.
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data (optional)
        num_epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        tuple: (teacher_network, student_network, loss_history)
    """
    print(f"\nStarting Teacher-Student Progressive Distillation Training (epochs: {num_epochs})")
    
    teacher_network = TeacherModel()
    student_network = StudentModel()
    
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    teacher_network.to(device)
    student_network.to(device)
    
    optimizer = torch.optim.Adam(student_network.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    loss_history = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            batch_count += 1
            imgs, _ = batch
            imgs = imgs.to(device)
            
            teacher_correction = teacher_network(imgs)  # teacher output
            student_output = student_network(imgs)      # student prediction
            
            loss = criterion(student_output, teacher_correction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / batch_count
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(teacher_network.state_dict(), os.path.join(MODELS_DIR, 'teacher_model.pth'))
    torch.save(student_network.state_dict(), os.path.join(MODELS_DIR, 'student_model.pth'))
    
    os.makedirs(LOGS_DIR, exist_ok=True)
    plot_loss(loss_history, "Teacher-Student Distillation Loss", f"{LOGS_DIR}/training_loss_teacher_student.pdf")
    
    return teacher_network, student_network, loss_history
