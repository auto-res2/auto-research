"""
Training module for ASMO-Solver experiments.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

class DiffusionDynamics(nn.Module):
    """Simple diffusion dynamics for experiments."""
    def forward(self, t, x):
        return -0.5 * x

class TeacherDynamics(nn.Module):
    """Teacher model for distillation experiment."""
    def forward(self, t, x):
        return -0.5 * x

class StudentDynamics(nn.Module):
    """Student model to be trained via distillation."""
    def __init__(self, latent_dim):
        super(StudentDynamics, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, t, x):
        return self.net(x)

def distillation_loss(student_traj, teacher_traj, manifold_student, manifold_teacher, lambda_geo=0.1):
    """
    Compute distillation loss combining trajectory reconstruction and geometric regularization.
    
    Args:
        student_traj (torch.Tensor): Student trajectory
        teacher_traj (torch.Tensor): Teacher trajectory
        manifold_student (torch.Tensor): Student manifold representation
        manifold_teacher (torch.Tensor): Teacher manifold representation
        lambda_geo (float): Weight for geometric regularization
        
    Returns:
        torch.Tensor: Combined loss
    """
    mse_loss = nn.MSELoss()(student_traj, teacher_traj)
    
    cos = nn.CosineSimilarity(dim=1)
    geo_loss = 1 - cos(manifold_student, manifold_teacher).mean()
    
    return mse_loss + lambda_geo * geo_loss

def generate_teacher_trajectory(x0, t_span, dynamics, n_steps=100):
    """
    Generate a teacher trajectory with high resolution.
    
    Args:
        x0 (torch.Tensor): Initial state
        t_span (tuple): Time span (t_start, t_end)
        dynamics (nn.Module): Dynamics model
        n_steps (int): Number of steps
        
    Returns:
        tuple: (t_teacher, teacher_traj)
    """
    t_teacher = torch.linspace(t_span[0], t_span[1], n_steps)
    teacher_traj = odeint(dynamics, x0, t_teacher, method='dopri5')
    return t_teacher, teacher_traj

def base_method_trajectory(x0, t_span, dynamics, n_steps):
    """
    Generate a trajectory using the Base Method with fixed uniform integration steps.
    
    Args:
        x0 (torch.Tensor): Initial state
        t_span (tuple): Time span (t_start, t_end)
        dynamics (nn.Module): Dynamics model
        n_steps (int): Number of steps
        
    Returns:
        tuple: (t_fixed, traj_fixed)
    """
    t_fixed = torch.linspace(t_span[0], t_span[1], n_steps)
    traj_fixed = odeint(dynamics, x0, t_fixed, method='dopri5')
    return t_fixed, traj_fixed

def asmo_solver_trajectory(x0, t_span, dynamics, n_steps):
    """
    Generate a trajectory using the ASMO-Solver with adaptive time steps.
    
    Args:
        x0 (torch.Tensor): Initial state
        t_span (tuple): Time span (t_start, t_end)
        dynamics (nn.Module): Dynamics model
        n_steps (int): Number of steps
        
    Returns:
        tuple: (t_adaptive, traj)
    """
    t_initial = torch.linspace(t_span[0], t_span[1], n_steps)
    traj = []
    xt = x0
    t_prev = t_span[0]
    traj.append(xt)
    
    for t_next in t_initial[1:]:
        dt = t_next - t_prev
        derivative = dynamics(t_prev, xt)
        error_estimate = torch.norm(derivative) * dt
        
        if error_estimate > 0.1:
            dt = dt * 0.5
            
        t_temp = torch.linspace(t_prev, t_prev + dt, 2)
        xt_new = odeint(dynamics, xt, t_temp, method='dopri5')[-1]
        traj.append(xt_new)
        xt = xt_new
        t_prev += dt
        
    t_adaptive = torch.linspace(t_span[0], t_prev, len(traj))
    traj = torch.stack(traj)
    
    return t_adaptive, traj

def train_student(x0, t_span, teacher_traj, teacher_t, student_model, optimizer, num_epochs=50):
    """
    Train a student model via distillation from a teacher model.
    
    Args:
        x0 (torch.Tensor): Initial state
        t_span (tuple): Time span (t_start, t_end)
        teacher_traj (torch.Tensor): Teacher trajectory
        teacher_t (torch.Tensor): Teacher time steps
        student_model (nn.Module): Student model
        optimizer (torch.optim.Optimizer): Optimizer
        num_epochs (int): Number of training epochs
        
    Returns:
        tuple: (trained_student_model, manifold_projection)
    """
    print("Starting teacher-student distillation training:")
    
    manifold_proj = nn.Linear(x0.shape[0], 3)  # projecting latent dimension to 3 dims
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        t_student = torch.linspace(t_span[0], t_span[1], 10)
        student_traj = odeint(student_model, x0, t_student, method='dopri5')
        
        sampling_ratio = teacher_traj.shape[0] // student_traj.shape[0]
        teacher_sampled = teacher_traj[::sampling_ratio]
        
        proj_student = manifold_proj(student_traj)
        proj_teacher = manifold_proj(teacher_sampled)
        
        loss = distillation_loss(student_traj, teacher_sampled, proj_student, proj_teacher)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.item():.5f}")
    
    return student_model, manifold_proj
