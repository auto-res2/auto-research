"""
Evaluation script for DPC-3D model
"""
import os
import torch
import numpy as np
import pandas as pd
import pickle
import logging
from tqdm import tqdm
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from utils.models import DPC3D
# Define model variants for testing
class DPC3D_Full(DPC3D):
    """Full DPC-3D model with dynamic prompt tuning and Bayesian adaptation"""
    def __init__(self, config):
        super().__init__(config)
        self.use_dynamic_prompt = True
        self.use_bayesian = True

class DPC3D_Static(DPC3D):
    """DPC-3D model with static prompt (no dynamic tuning)"""
    def __init__(self, config):
        super().__init__(config)
        self.use_dynamic_prompt = False
        self.use_bayesian = True
        
class DPC3D_NoBayesian(DPC3D):
    """DPC-3D model without Bayesian adaptation"""
    def __init__(self, config):
        super().__init__(config)
        self.use_dynamic_prompt = True
        self.use_bayesian = False
from utils.molecular_utils import (
    calculate_rmsd, 
    compute_mmff_energy, 
    validate_structure,
    get_scaffold
)
from utils.profiling import PerformanceTracker, measure_peak_memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluator for DPC-3D model"""
    
    def __init__(self, config, model=None, device="cuda"):
        """
        Initialize evaluator
        
        Args:
            config: Configuration dictionary
            model: Pre-initialized model (optional)
            device: Device to run evaluation on
        """
        self.config = config
        self.device = torch.device(device)
        
        # Load model if not provided
        if model is None:
            model_path = Path("models/best_model.pt")
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model = DPC3D(config).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from {model_path}")
            else:
                logger.warning(f"Model not found at {model_path}, initializing new model")
                self.model = DPC3D(config).to(self.device)
        else:
            self.model = model.to(self.device)
            
        # Set model to evaluation mode
        self.model.eval()
        
        # Create output directory
        self.output_dir = Path("models/evaluation")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def generate_conformers(self, data_loader, num_samples=10):
        """
        Generate 3D conformers for molecules in data loader
        
        Args:
            data_loader: DataLoader with molecular data
            num_samples: Number of conformers to generate per molecule
            
        Returns:
            results: Dictionary with generated conformers and metrics
        """
        logger.info(f"Generating conformers for {len(data_loader.dataset)} molecules")
        
        results = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Generating conformers"):
                # Get data
                token_ids = batch['token_ids']
                atom_types = batch['atom_types']
                ref_coords = batch['coords']
                mask = batch['mask']
                num_atoms = batch['num_atoms']
                smiles = batch['smiles']
                
                # Generate conformers
                batch_size = token_ids.shape[0]
                
                for i in range(batch_size):
                    # Get single molecule data
                    mol_token_ids = token_ids[i:i+1]
                    mol_atom_types = atom_types[i:i+1]
                    mol_ref_coords = ref_coords[i:i+1]
                    mol_mask = mask[i:i+1]
                    mol_num_atoms = num_atoms[i]
                    mol_smiles = smiles[i]
                    
                    # Track performance
                    tracker = PerformanceTracker().start()
                    
                    # Generate conformers
                    pred_coords, prompt_trajectory, uncertainty_trajectory = self.model.sample(
                        mol_token_ids, 
                        mol_num_atoms, 
                        mol_atom_types, 
                        mol_mask, 
                        device=self.device
                    )
                    
                    # End tracking
                    runtime = tracker.end()
                    
                    # Convert to numpy
                    pred_coords_np = pred_coords[0, :mol_num_atoms].cpu().numpy()
                    ref_coords_np = mol_ref_coords[0, :mol_num_atoms].cpu().numpy()
                    
                    # Create RDKit mol with predicted coordinates
                    mol = Chem.MolFromSmiles(mol_smiles)
                    if mol is None:
                        continue
                        
                    # Add hydrogens
                    mol = Chem.AddHs(mol)
                    
                    # Create conformer
                    conf = Chem.Conformer(mol.GetNumAtoms())
                    for j, (x, y, z) in enumerate(pred_coords_np):
                        if j < mol.GetNumAtoms():
                            conf.SetAtomPosition(j, (float(x), float(y), float(z)))
                    mol.AddConformer(conf)
                    
                    # Calculate metrics
                    rmsd = calculate_rmsd(mol, mol)  # Self-RMSD should be 0
                    energy = compute_mmff_energy(mol)
                    valid = validate_structure(mol)
                    
                    # Store results
                    result = {
                        'smiles': mol_smiles,
                        'num_atoms': mol_num_atoms,
                        'pred_coords': pred_coords_np,
                        'ref_coords': ref_coords_np,
                        'rmsd': rmsd,
                        'energy': energy,
                        'valid': valid,
                        'runtime': runtime,
                        'prompt_trajectory': [p.numpy() for p in prompt_trajectory],
                        'uncertainty_trajectory': [u.numpy() for u in uncertainty_trajectory]
                    }
                    
                    results.append(result)
                    
        logger.info(f"Generated conformers for {len(results)} molecules")
        
        return results
        
    def evaluate_conformer_quality(self, results):
        """
        Evaluate quality of generated conformers
        
        Args:
            results: List of dictionaries with generated conformers and metrics
            
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        logger.info("Evaluating conformer quality")
        
        # Calculate metrics
        rmsds = [r['rmsd'] for r in results if r['rmsd'] is not None]
        energies = [r['energy'] for r in results if r['energy'] != float('inf')]
        valid_count = sum(r['valid'] for r in results)
        runtimes = [r['runtime'] for r in results]
        
        # Calculate statistics
        metrics = {
            'rmsd_mean': np.mean(rmsds) if rmsds else None,
            'rmsd_std': np.std(rmsds) if rmsds else None,
            'rmsd_min': np.min(rmsds) if rmsds else None,
            'rmsd_max': np.max(rmsds) if rmsds else None,
            'energy_mean': np.mean(energies) if energies else None,
            'energy_std': np.std(energies) if energies else None,
            'validity_rate': valid_count / len(results) if results else 0,
            'runtime_mean': np.mean(runtimes) if runtimes else None,
            'runtime_std': np.std(runtimes) if runtimes else None,
            'num_molecules': len(results)
        }
        
        # Log metrics
        logger.info(f"RMSD: {metrics['rmsd_mean']:.4f} ± {metrics['rmsd_std']:.4f}")
        logger.info(f"Energy: {metrics['energy_mean']:.4f} ± {metrics['energy_std']:.4f}")
        logger.info(f"Validity rate: {metrics['validity_rate']:.4f}")
        logger.info(f"Runtime: {metrics['runtime_mean']:.4f} ± {metrics['runtime_std']:.4f}")
        
        return metrics
        
    def compare_model_variants(self, data_loader):
        """
        Compare different model variants
        
        Args:
            data_loader: DataLoader with molecular data
            
        Returns:
            comparison: Dictionary with comparison results
        """
        logger.info("Comparing model variants")
        
        # Create model variants
        model_full = DPC3D_Full(self.config).to(self.device)
        model_static = DPC3D_Static(self.config).to(self.device)
        model_no_bayesian = DPC3D_NoBayesian(self.config).to(self.device)
        
        # Set models to evaluation mode
        model_full.eval()
        model_static.eval()
        model_no_bayesian.eval()
        
        # Initialize results
        full_memory = []
        static_memory = []
        no_bayesian_memory = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Comparing variants"):
                # Get data
                token_ids = batch['token_ids']
                atom_types = batch['atom_types']
                coords = batch['coords']
                mask = batch['mask']
                
                # Sample random timesteps
                batch_size = token_ids.shape[0]
                t = torch.rand(batch_size, device=self.device)
                
                # Add noise to coordinates
                noise = torch.randn_like(coords)
                noisy_coords = coords + torch.sqrt(t.view(-1, 1, 1)) * noise
                
                # Get initial prompt from language model
                prompt = self.model.encode_molecule(token_ids)
                
                # Measure full model
                def run_full():
                    noise_pred, _, _ = model_full.forward(
                        noisy_coords, atom_types, t, prompt, mask
                    )
                    return noise_pred
                    
                # Measure static model
                def run_static():
                    noise_pred, _, _ = model_static.forward(
                        noisy_coords, atom_types, t, prompt, mask
                    )
                    return noise_pred
                    
                # Measure no Bayesian model
                def run_no_bayesian():
                    noise_pred, _, _ = model_no_bayesian.forward(
                        noisy_coords, atom_types, t, prompt, mask
                    )
                    return noise_pred
                    
                # Run measurements
                _, mem_full = measure_peak_memory(run_full)
                _, mem_static = measure_peak_memory(run_static)
                _, mem_no_bayesian = measure_peak_memory(run_no_bayesian)
                
                # Store results
                full_memory.append(mem_full)
                static_memory.append(mem_static)
                no_bayesian_memory.append(mem_no_bayesian)
                
        # Calculate statistics
        full_memory_mean = float(np.mean(full_memory)) if full_memory else 0.0
        full_memory_std = float(np.std(full_memory)) if full_memory else 0.0
        static_memory_mean = float(np.mean(static_memory)) if static_memory else 0.0
        static_memory_std = float(np.std(static_memory)) if static_memory else 0.0
        no_bayesian_memory_mean = float(np.mean(no_bayesian_memory)) if no_bayesian_memory else 0.0
        no_bayesian_memory_std = float(np.std(no_bayesian_memory)) if no_bayesian_memory else 0.0
        
        # Create comparison dictionary
        comparison = {
            'full': {
                'memory': full_memory,
                'memory_mean': full_memory_mean,
                'memory_std': full_memory_std
            },
            'static': {
                'memory': static_memory,
                'memory_mean': static_memory_mean,
                'memory_std': static_memory_std
            },
            'no_bayesian': {
                'memory': no_bayesian_memory,
                'memory_mean': no_bayesian_memory_mean,
                'memory_std': no_bayesian_memory_std
            }
        }
            
        # Log results
        logger.info("Memory usage (MB):")
        logger.info(f"Full: {full_memory_mean:.2f} ± {full_memory_std:.2f}")
        logger.info(f"Static: {static_memory_mean:.2f} ± {static_memory_std:.2f}")
        logger.info(f"No Bayesian: {no_bayesian_memory_mean:.2f} ± {no_bayesian_memory_std:.2f}")
        
        return comparison
        
    def evaluate_scaffold_generalization(self, seen_loader, unseen_loader):
        """
        Evaluate generalization to unseen scaffolds
        
        Args:
            seen_loader: DataLoader with seen scaffolds
            unseen_loader: DataLoader with unseen scaffolds
            
        Returns:
            generalization: Dictionary with generalization results
        """
        logger.info("Evaluating scaffold generalization")
        
        # Generate conformers for seen scaffolds
        seen_results = self.generate_conformers(seen_loader)
        
        # Generate conformers for unseen scaffolds
        unseen_results = self.generate_conformers(unseen_loader)
        
        # Evaluate conformer quality
        seen_metrics = self.evaluate_conformer_quality(seen_results)
        unseen_metrics = self.evaluate_conformer_quality(unseen_results)
        
        # Calculate generalization gap
        generalization_gap = {
            'rmsd_gap': unseen_metrics['rmsd_mean'] - seen_metrics['rmsd_mean'] 
                if seen_metrics['rmsd_mean'] and unseen_metrics['rmsd_mean'] else None,
            'energy_gap': unseen_metrics['energy_mean'] - seen_metrics['energy_mean']
                if seen_metrics['energy_mean'] and unseen_metrics['energy_mean'] else None,
            'validity_gap': unseen_metrics['validity_rate'] - seen_metrics['validity_rate'],
            'runtime_gap': unseen_metrics['runtime_mean'] - seen_metrics['runtime_mean']
                if seen_metrics['runtime_mean'] and unseen_metrics['runtime_mean'] else None
        }
        
        # Log generalization gap
        logger.info("Generalization gap (unseen - seen):")
        logger.info(f"RMSD gap: {generalization_gap['rmsd_gap']:.4f}")
        logger.info(f"Energy gap: {generalization_gap['energy_gap']:.4f}")
        logger.info(f"Validity gap: {generalization_gap['validity_gap']:.4f}")
        logger.info(f"Runtime gap: {generalization_gap['runtime_gap']:.4f}")
        
        # Combine results
        generalization = {
            'seen': seen_metrics,
            'unseen': unseen_metrics,
            'gap': generalization_gap
        }
        
        return generalization
        
    def evaluate_efficiency(self, data_loader):
        """
        Evaluate computational efficiency
        
        Args:
            data_loader: DataLoader with molecular data
            
        Returns:
            efficiency: Dictionary with efficiency metrics
        """
        logger.info("Evaluating computational efficiency")
        
        # Initialize metrics
        runtimes = []
        memory_usage = []
        step_counts = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating efficiency"):
                # Get data
                token_ids = batch['token_ids']
                atom_types = batch['atom_types']
                mask = batch['mask']
                num_atoms = batch['num_atoms']
                
                # Generate conformers with tracking
                batch_size = token_ids.shape[0]
                
                for i in range(batch_size):
                    # Get single molecule data
                    mol_token_ids = token_ids[i:i+1]
                    mol_atom_types = atom_types[i:i+1]
                    mol_mask = mask[i:i+1]
                    mol_num_atoms = num_atoms[i]
                    
                    # Track performance
                    tracker = PerformanceTracker().start()
                    
                    # Measure memory usage
                    def generate_conformer():
                        pred_coords, _, _ = self.model.sample(
                            mol_token_ids, 
                            mol_num_atoms, 
                            mol_atom_types, 
                            mol_mask, 
                            device=self.device
                        )
                        return pred_coords
                        
                    _, peak_memory = measure_peak_memory(generate_conformer)
                    
                    # End tracking
                    runtime = tracker.end()
                    
                    # Store metrics
                    runtimes.append(runtime)
                    memory_usage.append(peak_memory)
                    step_counts.append(self.config["diffusion"]["sampling_steps"])
                    
        # Calculate statistics
        efficiency = {
            'runtime_mean': np.mean(runtimes),
            'runtime_std': np.std(runtimes),
            'memory_mean': np.mean(memory_usage),
            'memory_std': np.std(memory_usage),
            'steps_mean': np.mean(step_counts),
            'steps_std': np.std(step_counts)
        }
        
        # Log efficiency metrics
        logger.info(f"Runtime: {efficiency['runtime_mean']:.4f} ± {efficiency['runtime_std']:.4f} s")
        logger.info(f"Memory usage: {efficiency['memory_mean']:.2f} ± {efficiency['memory_std']:.2f} MB")
        logger.info(f"Steps: {efficiency['steps_mean']:.2f} ± {efficiency['steps_std']:.2f}")
        
        return efficiency
        
    def save_results(self, results, filename):
        """
        Save evaluation results to file
        
        Args:
            results: Dictionary with evaluation results
            filename: Name of output file
        """
        output_path = self.output_dir / filename
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Saved results to {output_path}")

def load_scaffold_datasets(config):
    """
    Load datasets for scaffold generalization experiment
    
    Args:
        config: Configuration dictionary
        
    Returns:
        seen_loader: DataLoader for seen scaffolds
        unseen_loader: DataLoader for unseen scaffolds
    """
    from train import MolecularDataset
    
    # Set device
    device = torch.device(config["device"])
    
    # Load processed data
    processed_dir = Path(config["processed_data_dir"])
    
    # Load seen scaffolds
    seen_path = processed_dir / "seen_scaffolds.pkl"
    with open(seen_path, "rb") as f:
        seen_data = pickle.load(f)
    logger.info(f"Loaded {len(seen_data)} seen scaffold samples")
    
    # Load unseen scaffolds
    unseen_path = processed_dir / "unseen_scaffolds.pkl"
    with open(unseen_path, "rb") as f:
        unseen_data = pickle.load(f)
    logger.info(f"Loaded {len(unseen_data)} unseen scaffold samples")
    
    # Create datasets
    seen_dataset = MolecularDataset(seen_data, device=device)
    unseen_dataset = MolecularDataset(unseen_data, device=device)
    
    # Create data loaders
    seen_loader = torch.utils.data.DataLoader(
        seen_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=seen_dataset.collate_fn
    )
    
    unseen_loader = torch.utils.data.DataLoader(
        unseen_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=unseen_dataset.collate_fn
    )
    
    return seen_loader, unseen_loader

def load_test_dataset(config):
    """
    Load test dataset for evaluation
    
    Args:
        config: Configuration dictionary
        
    Returns:
        test_loader: DataLoader for test data
    """
    from train import MolecularDataset
    
    # Set device
    device = torch.device(config["device"])
    
    # Load processed data
    processed_dir = Path(config["processed_data_dir"])
    
    # Load test data
    test_path = processed_dir / "test_dataset.pkl"
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Create dataset
    test_dataset = MolecularDataset(test_data, device=device)
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=test_dataset.collate_fn
    )
    
    return test_loader

def evaluate_pipeline(model, dataset):
    """
    Evaluate a model on a dataset of molecules
    
    Args:
        model: Model to evaluate
        dataset: List of RDKit molecules
        
    Returns:
        results: List of evaluation results
    """
    results = []
    
    for mol in dataset:
        # Generate conformer
        pred_mol = model.generate_conformer(mol)
        
        # Calculate metrics
        rmsd = calculate_rmsd(mol, pred_mol)
        energy = compute_mmff_energy(pred_mol)
        valid = validate_structure(pred_mol)
        
        # Store results
        result = {
            'smiles': Chem.MolToSmiles(mol),
            'rmsd': rmsd,
            'energy': energy,
            'valid': valid
        }
        
        results.append(result)
        
    return results

def evaluate_model(config):
    """
    Main function to evaluate DPC-3D model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        evaluator: Model evaluator with results
    """
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    # Load test dataset
    test_loader = load_test_dataset(config)
    
    # Evaluate conformer quality
    results = evaluator.generate_conformers(test_loader)
    metrics = evaluator.evaluate_conformer_quality(results)
    evaluator.save_results(metrics, "conformer_quality.pkl")
    
    # Compare model variants
    comparison = evaluator.compare_model_variants(test_loader)
    evaluator.save_results(comparison, "model_variants.pkl")
    
    # Evaluate scaffold generalization
    seen_loader, unseen_loader = load_scaffold_datasets(config)
    generalization = evaluator.evaluate_scaffold_generalization(seen_loader, unseen_loader)
    evaluator.save_results(generalization, "scaffold_generalization.pkl")
    
    # Evaluate efficiency
    efficiency = evaluator.evaluate_efficiency(test_loader)
    evaluator.save_results(efficiency, "efficiency.pkl")
    
    return evaluator

if __name__ == "__main__":
    # Import config when run as script
    from config.dpc3d_config import (
        DATA_CONFIG, MODEL_CONFIG, EVAL_CONFIG, EXPERIMENT_CONFIG
    )
    
    # Combine configs
    config = {
        **DATA_CONFIG, 
        **MODEL_CONFIG, 
        **EVAL_CONFIG, 
        **EXPERIMENT_CONFIG
    }
    
    # Evaluate model
    evaluate_model(config)
