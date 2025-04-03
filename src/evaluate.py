"""
Evaluation module for ASMO-Solver experiments.
"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

def ensure_directory(directory):
    """
    Ensure that the directory exists.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_high_quality_pdf(filename, directory="logs"):
    """
    Save the current figure as a high-quality PDF.
    
    Args:
        filename (str): Filename (without extension)
        directory (str): Directory to save the file
        
    Returns:
        str: Full path to the saved file
    """
    ensure_directory(directory)
    full_path = os.path.join(directory, f"{filename}.pdf")
    plt.savefig(full_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Figure saved as {full_path}")
    return full_path

def experiment1(step_counts, t_span, dynamics, x0):
    """
    Experiment 1: Comparing Adaptive Time-Step Adjustments and Trajectory Accuracy.
    
    Args:
        step_counts (list): List of step counts to evaluate
        t_span (tuple): Time span (t_start, t_end)
        dynamics (nn.Module): Dynamics model
        x0 (torch.Tensor): Initial state
        
    Returns:
        tuple: (errors_base, errors_asmo)
    """
    print("======== Experiment 1: Adaptive Time-Step Adjustment ========")
    
    t_teacher, traj_teacher = generate_teacher_trajectory(x0, t_span, dynamics)
    print("Teacher trajectory generated with 1000 steps.")
    
    errors_base = []
    errors_asmo = []
    
    for n in step_counts:
        from src.train import base_method_trajectory
        t_fixed, traj_base = base_method_trajectory(x0, t_span, dynamics, n)
        
        traj_base_reshaped = traj_base.transpose(0, 1).unsqueeze(0)  # [1, dims, steps]
        traj_base_interp = torch.nn.functional.interpolate(
            traj_base_reshaped, 
            size=traj_teacher.shape[0], 
            mode='linear', 
            align_corners=False
        ).squeeze(0).transpose(0, 1)  # [steps, dims]
        
        error_base = torch.mean((traj_base_interp - traj_teacher)**2)
        errors_base.append(error_base.item())
        
        from src.train import asmo_solver_trajectory
        t_adaptive, traj_asmo = asmo_solver_trajectory(x0, t_span, dynamics, n)
        
        traj_asmo_reshaped = traj_asmo.transpose(0, 1).unsqueeze(0)  # [1, dims, steps]
        traj_asmo_interp = torch.nn.functional.interpolate(
            traj_asmo_reshaped, 
            size=traj_teacher.shape[0], 
            mode='linear', 
            align_corners=False
        ).squeeze(0).transpose(0, 1)  # [steps, dims]
        
        error_asmo = torch.mean((traj_asmo_interp - traj_teacher)**2)
        errors_asmo.append(error_asmo.item())
        
        print(f"Step Count {n}: Base Method MSE Error = {error_base.item():.5f}, "
              f"ASMO-Solver MSE Error = {error_asmo.item():.5f}")
    
    plt.figure(figsize=(6, 4))
    plt.plot(step_counts, errors_base, marker='o', label="Base Method")
    plt.plot(step_counts, errors_asmo, marker='s', label="ASMO-Solver")
    plt.xlabel("Step Count")
    plt.ylabel("Integrated MSE Error")
    plt.title("Trajectory Error vs. Step Count")
    plt.legend()
    plt.tight_layout()
    
    save_high_quality_pdf("trajectory_error_small")
    
    print("Experiment 1 completed.\n")
    return errors_base, errors_asmo

def generate_teacher_trajectory(x0, t_span, dynamics, n_steps=1000):
    """
    Generate teacher trajectory on a very fine grid.
    
    Args:
        x0 (torch.Tensor): Initial state
        t_span (tuple): Time span (t_start, t_end)
        dynamics (nn.Module): Dynamics model
        n_steps (int): Number of steps
        
    Returns:
        tuple: (t_teacher, traj_teacher)
    """
    t_teacher = torch.linspace(t_span[0], t_span[1], n_steps)
    traj_teacher = odeint(dynamics, x0, t_teacher, method='dopri5')
    return t_teacher, traj_teacher

def experiment2(latent_dim, timesteps, window_size, min_dim, max_dim, threshold, seed=0):
    """
    Experiment 2: Evaluating the Benefit of Dynamic Manifold Construction.
    
    Args:
        latent_dim (int): Dimensionality of the latent space
        timesteps (int): Number of time steps
        window_size (int): Size of the sliding window
        min_dim (int): Minimum manifold dimension
        max_dim (int): Maximum manifold dimension
        threshold (float): Threshold for determining effective dimensions
        seed (int): Random seed
        
    Returns:
        tuple: (dims_used, reconstruction_errors)
    """
    print("======== Experiment 2: Dynamic Manifold Construction ========")
    
    np.random.seed(seed)
    
    from src.preprocess import generate_synthetic_trajectory, static_projection, dynamic_projection
    X = generate_synthetic_trajectory(timesteps, latent_dim, seed)
    
    proj_static = static_projection(X, target_dim=2)
    
    proj_dynamic_all = []
    dims_used = []
    reconstruction_errors = []
    
    for i in range(0, timesteps - window_size + 1, window_size):
        window = X[i:i+window_size]
        proj_window, dim_used = dynamic_projection(window, min_dim, max_dim, threshold)
        
        proj_dynamic_all.append(proj_window)
        dims_used.append(dim_used)
        
        X_centered = window - window.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        S_reduced = np.diag(S[:dim_used])
        U_reduced = U[:, :dim_used]
        V_reduced = Vt[:dim_used, :]
        window_reconstructed = np.dot(U_reduced, np.dot(S_reduced, V_reduced)) + window.mean(axis=0, keepdims=True)
        error = np.mean((window - window_reconstructed)**2)
        reconstruction_errors.append(error)
    
    print("Dynamic projection effective dimensions over windows:", dims_used)
    print("Mean reconstruction error (dynamic projection):", np.mean(reconstruction_errors))
    
    plt.figure(figsize=(6, 4))
    x_bar = np.arange(len(reconstruction_errors))
    plt.bar(x_bar, reconstruction_errors, tick_label=[f"win{i+1}" for i in x_bar])
    plt.xlabel("Window")
    plt.ylabel("Reconstruction MSE")
    plt.title("Dynamic Projection Reconstruction Error per Window")
    plt.tight_layout()
    
    save_high_quality_pdf("reconstruction_error_small")
    
    print("Experiment 2 completed.\n")
    return dims_used, reconstruction_errors

def experiment3(latent_dim, t_span, teacher_steps, student_steps, train_epochs, learning_rate):
    """
    Experiment 3: Teacher–Student Distillation Performance in Low NFE Settings.
    
    Args:
        latent_dim (int): Dimensionality of the latent space
        t_span (tuple): Time span (t_start, t_end)
        teacher_steps (int): Number of steps for teacher trajectory
        student_steps (int): Number of steps for student trajectory
        train_epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        
    Returns:
        float: Final teacher-student MSE error
    """
    print("======== Experiment 3: Teacher–Student Distillation Performance ========")
    
    x0 = torch.tensor([1.0, 0.0])
    
    from src.train import TeacherDynamics, StudentDynamics
    teacher_model = TeacherDynamics()
    student_model = StudentDynamics(latent_dim)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    
    from src.train import generate_teacher_trajectory
    teacher_t, teacher_traj = generate_teacher_trajectory(x0, t_span, teacher_model, n_steps=teacher_steps)
    print("Teacher trajectory generated for distillation.")
    
    from src.train import train_student
    student_model, manifold_proj = train_student(
        x0, t_span, teacher_traj, teacher_t, student_model, optimizer, num_epochs=train_epochs
    )
    
    t_student_eval = torch.linspace(t_span[0], t_span[1], student_steps)
    student_traj_final = odeint(student_model, x0, t_student_eval, method='dopri5')
    
    teacher_traj_eval = odeint(teacher_model, x0, t_student_eval, method='dopri5')
    final_error = nn.MSELoss()(student_traj_final, teacher_traj_eval)
    print("Final teacher-student MSE error:", final_error.item())
    
    student_traj_np = student_traj_final.detach().numpy()
    teacher_traj_np = teacher_traj_eval.detach().numpy()
    
    plt.figure(figsize=(6, 4))
    plt.plot(teacher_traj_np[:,0], teacher_traj_np[:,1], 'bo-', label="Teacher")
    plt.plot(student_traj_np[:,0], student_traj_np[:,1], 'rs--', label="Student")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Teacher vs. Student Trajectories")
    plt.legend()
    plt.tight_layout()
    
    save_high_quality_pdf("teacher_student_trajectory_small")
    
    print("Experiment 3 completed.\n")
    return final_error.item()
