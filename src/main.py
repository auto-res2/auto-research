import os
import torch
import numpy as np
import random
import argparse
import time
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import get_dataloaders
from src.train import (
    FPDMTeacher, OneStepStudent, OneStepStudentAux,
    train_teacher, train_student, train_student_with_aux, save_model
)
from src.evaluate import (
    evaluate_model, profile_inference, plot_training_loss,
    plot_comparison_barplot, visualize_outputs
)
from config.adfp_diff_config import MODEL_CONFIG, TRAIN_CONFIG, TEST_CONFIG, DATA_CONFIG, PATHS

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'config', 'data', 'models', 'paper', 'logs'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
def get_device():
    """Get the device to use."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def experiment_1_sampling_quality_efficiency(device):
    """Experiment 1: Comparison of Sampling Quality & Efficiency.
    
    This experiment compares the teacher and student models in terms of
    training loss, inference time, and memory usage.
    
    Args:
        device: Device to run the experiment on
    """
    print("\n" + "="*80)
    print("Experiment 1: Comparison of Sampling Quality & Efficiency")
    print("="*80)
    
    train_loader, val_loader = get_dataloaders(
        batch_size=TRAIN_CONFIG['batch_size'],
        image_size=TRAIN_CONFIG['image_size'],
        num_workers=TRAIN_CONFIG['num_workers']
    )
    
    teacher = FPDMTeacher(
        hidden_channels=MODEL_CONFIG['hidden_channels']
    ).to(device)
    
    student = OneStepStudent(
        hidden_channels=MODEL_CONFIG['hidden_channels']
    ).to(device)
    
    teacher_optimizer = torch.optim.Adam(
        teacher.parameters(),
        lr=TRAIN_CONFIG['lr'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    student_optimizer = torch.optim.Adam(
        student.parameters(),
        lr=TRAIN_CONFIG['lr'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    print("\nTraining Teacher Model...")
    teacher_losses = train_teacher(
        teacher=teacher,
        dataloader=train_loader,
        optimizer=teacher_optimizer,
        device=device,
        epochs=TRAIN_CONFIG['teacher_epochs']
    )
    
    save_model(teacher, PATHS['teacher_model'])
    
    plot_training_loss(
        losses=teacher_losses,
        title="Teacher Training Loss",
        filename=PATHS['plots']['teacher_loss']
    )
    
    print("\nTraining Student Model...")
    student_losses = train_student(
        student=student,
        teacher=teacher,
        dataloader=train_loader,
        optimizer=student_optimizer,
        device=device,
        epochs=TRAIN_CONFIG['student_epochs']
    )
    
    save_model(student, PATHS['student_model'])
    
    plot_training_loss(
        losses=student_losses,
        title="Student Distillation Loss",
        filename=PATHS['plots']['student_loss']
    )
    
    print("\nEvaluating Teacher Model...")
    teacher_loss = evaluate_model(teacher, val_loader, device)
    
    print("\nEvaluating Student Model...")
    student_loss = evaluate_model(student, val_loader, device)
    
    print("\nProfiling Inference...")
    images, _ = next(iter(val_loader))
    images = images.to(device)
    
    _, teacher_time, teacher_memory = profile_inference(
        model=teacher,
        images=images,
        device=device,
        name="Teacher",
        steps=MODEL_CONFIG['teacher_steps']
    )
    
    _, student_time, student_memory = profile_inference(
        model=student,
        images=images,
        device=device,
        name="Student"
    )
    
    plot_comparison_barplot(
        data=[teacher_time, student_time],
        labels=["Teacher", "Student"],
        title="Inference Time Comparison",
        filename=PATHS['plots']['inference_time'],
        ylabel="Time (seconds)"
    )
    
    plot_comparison_barplot(
        data=[teacher_memory / 1024 / 1024, student_memory / 1024 / 1024],
        labels=["Teacher", "Student"],
        title="GPU Memory Usage Comparison",
        filename=PATHS['plots']['memory_usage'],
        ylabel="Memory (MB)"
    )
    
    print("\nVisualizing Outputs...")
    with torch.no_grad():
        teacher_output = teacher(images)
        student_output = student(images)
    
    visualize_outputs(
        original=images,
        teacher_output=teacher_output,
        student_output=student_output,
        filename=PATHS['plots']['output_comparison']
    )
    
    print("\nExperiment 1 completed.")
    return teacher, student

def experiment_2_ablation_study(teacher, device):
    """Experiment 2: Ablation Study on Adaptive Auxiliary Supervision.
    
    This experiment compares the standard student model with a student model
    that uses auxiliary supervision.
    
    Args:
        teacher: Trained teacher model
        device: Device to run the experiment on
    """
    print("\n" + "="*80)
    print("Experiment 2: Ablation Study on Adaptive Auxiliary Supervision")
    print("="*80)
    
    train_loader, val_loader = get_dataloaders(
        batch_size=TRAIN_CONFIG['batch_size'],
        image_size=TRAIN_CONFIG['image_size'],
        num_workers=TRAIN_CONFIG['num_workers']
    )
    
    student_aux = OneStepStudentAux(
        hidden_channels=MODEL_CONFIG['hidden_channels']
    ).to(device)
    
    student_aux_optimizer = torch.optim.Adam(
        student_aux.parameters(),
        lr=TRAIN_CONFIG['lr'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    print("\nTraining Student Model with Auxiliary Supervision...")
    total_losses, main_losses, aux_losses = train_student_with_aux(
        student_aux=student_aux,
        teacher=teacher,
        dataloader=train_loader,
        optimizer=student_aux_optimizer,
        device=device,
        epochs=TRAIN_CONFIG['student_epochs'],
        aux_weight=TRAIN_CONFIG['aux_weight']
    )
    
    save_model(student_aux, PATHS['student_aux_model'])
    
    plot_training_loss(
        losses=total_losses,
        title="Student with Auxiliary Supervision - Total Loss",
        filename=PATHS['plots']['student_aux_loss']
    )
    
    print("\nEvaluating Student Model with Auxiliary Supervision...")
    student_aux_loss = evaluate_model(student_aux, val_loader, device)
    
    print("\nProfiling Inference...")
    images, _ = next(iter(val_loader))
    images = images.to(device)
    
    _, student_aux_time, student_aux_memory = profile_inference(
        model=student_aux,
        images=images,
        device=device,
        name="Student+Aux"
    )
    
    print("\nExperiment 2 completed.")
    return student_aux

def experiment_3_resource_constraints(teacher, student, student_aux, device):
    """Experiment 3: Resource-Constrained vs. Unconstrained Inference Analysis.
    
    This experiment compares the models under resource-constrained and
    unconstrained inference settings.
    
    Args:
        teacher: Trained teacher model
        student: Trained student model
        student_aux: Trained student model with auxiliary supervision
        device: Device to run the experiment on
    """
    print("\n" + "="*80)
    print("Experiment 3: Resource-Constrained vs. Unconstrained Inference Analysis")
    print("="*80)
    
    _, val_loader = get_dataloaders(
        batch_size=TEST_CONFIG['batch_size'],
        image_size=TRAIN_CONFIG['image_size'],
        num_workers=TRAIN_CONFIG['num_workers']
    )
    
    images, _ = next(iter(val_loader))
    images = images.to(device)
    
    print("\nResource-Constrained Inference (Limited Steps)...")
    
    with torch.no_grad():
        start_time = time.time()
        teacher_constrained = teacher(images, steps=TEST_CONFIG['constrained_steps'])
        teacher_constrained_time = time.time() - start_time
    
    print(f"Teacher (Constrained): {teacher_constrained_time:.4f}s")
    
    with torch.no_grad():
        start_time = time.time()
        student_output = student(images)
        student_time = time.time() - start_time
    
    print(f"Student: {student_time:.4f}s")
    
    with torch.no_grad():
        start_time = time.time()
        student_aux_output = student_aux(images)
        student_aux_time = time.time() - start_time
    
    print(f"Student+Aux: {student_aux_time:.4f}s")
    
    print("\nUnconstrained Inference (Full Steps)...")
    
    with torch.no_grad():
        start_time = time.time()
        teacher_unconstrained = teacher(images, steps=TEST_CONFIG['unconstrained_steps'])
        teacher_unconstrained_time = time.time() - start_time
    
    print(f"Teacher (Unconstrained): {teacher_unconstrained_time:.4f}s")
    
    plot_comparison_barplot(
        data=[teacher_constrained_time, teacher_unconstrained_time, student_time, student_aux_time],
        labels=["Teacher (Constrained)", "Teacher (Unconstrained)", "Student", "Student+Aux"],
        title="Inference Time Comparison under Resource Constraints",
        filename="logs/resource_constraints_time.pdf",
        ylabel="Time (seconds)"
    )
    
    print("\nExperiment 3 completed.")

def main():
    """Main function to run all experiments."""
    print("="*80)
    print("Adaptive Distilled Fixed-Point Diffusion (ADFP-Diff) Experiment")
    print("="*80)
    
    set_seed()
    
    create_directories()
    
    device = get_device()
    
    teacher, student = experiment_1_sampling_quality_efficiency(device)
    student_aux = experiment_2_ablation_study(teacher, device)
    experiment_3_resource_constraints(teacher, student, student_aux, device)
    
    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    main()
