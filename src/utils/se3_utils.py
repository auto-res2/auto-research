import torch
import numpy as np
from scipy.spatial.distance import cdist

def compute_energy(conformation):
    """
    Compute a surrogate energy for a conformation.
    Here we use a simple function: sum of pairwise inverse distances (penalizing clashes).
    """
    energy = 0.0
    num_atoms = conformation.shape[0]
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            d = np.linalg.norm(conformation[i] - conformation[j])
            energy += 1.0 / (d + 1e-6)
    return energy

def compute_rmsd(conf1, conf2):
    """Compute RMSD between two conformations (each a numpy array with shape (num_atoms, 3))."""
    diff = conf1 - conf2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def tensor_rmsd(a, b):
    """Compute RMSD between two tensor conformations."""
    return torch.sqrt(torch.mean((a - b)**2)).item()

def generate_conformations(method='HBFG-SE3', num_samples=50, num_atoms=10, seed=42):
    """
    Generate dummy protein conformations.
    The random seed is set differently for each method to simulate different behaviors.
    """
    conformations = []
    for i in range(num_samples):
        np.random.seed(i + (0 if method == 'HBFG-SE3' else 100))
        # For HBFG-SE3 we simulate a lower variance (and lower energy bias)
        scale = 0.8 if method == 'HBFG-SE3' else 1.0
        conformation = np.random.randn(num_atoms, 3) * scale
        conformations.append(conformation)
    return conformations

def create_protein_batch(batch_size, num_atoms, device):
    """Create a batch of random protein-like structures."""
    return torch.randn(batch_size, num_atoms, 3, device=device)

def rotate_points(points, rotation_matrix):
    """Apply rotation matrix to points."""
    return torch.matmul(points, rotation_matrix.transpose(-1, -2))

def random_rotation_matrix(batch_size, device):
    """Generate random rotation matrices for SE(3) equivariance."""
    # Generate random rotation matrices using QR decomposition
    q, r = torch.linalg.qr(torch.randn(batch_size, 3, 3, device=device))
    # Ensure proper rotation (determinant 1)
    det = torch.linalg.det(q).unsqueeze(-1).unsqueeze(-1)
    q = q * torch.sign(det)
    return q
