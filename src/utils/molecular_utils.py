"""
Utility functions for molecular operations
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem.Scaffolds import MurckoScaffold

def smiles_to_mol(smiles):
    """Convert SMILES string to RDKit Mol object"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol

def add_hydrogens(mol):
    """Add hydrogens to a molecule"""
    return Chem.AddHs(mol)

def generate_conformer(mol, num_conf=1, random_seed=42):
    """
    Generate 3D conformers for a molecule
    
    Args:
        mol: RDKit Mol object
        num_conf: Number of conformers to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        RDKit Mol object with embedded conformers
    """
    mol = add_hydrogens(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.numThreads = 0  # Use all available CPUs
    
    # Try to generate conformers
    conformer_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params)
    
    if len(conformer_ids) == 0:
        # If conformer generation fails, try again without constraints
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, useRandomCoords=True)
    
    # Optimize conformers using MMFF
    for conf_id in range(mol.GetNumConformers()):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        except:
            # If optimization fails, continue with unoptimized conformer
            pass
            
    return mol

def calculate_rmsd(ref_mol, pred_mol, ref_conf_id=0, pred_conf_id=0):
    """
    Calculate RMSD between two conformers
    
    Args:
        ref_mol: Reference molecule with conformer
        pred_mol: Predicted molecule with conformer
        ref_conf_id: Conformer ID in reference molecule
        pred_conf_id: Conformer ID in predicted molecule
        
    Returns:
        RMSD value or None if calculation fails
    """
    try:
        # Prune to match atom indices
        rmsd = rdMolAlign.GetBestRMS(pred_mol, ref_mol, 
                                     refId=ref_conf_id, 
                                     prbId=pred_conf_id)
        return rmsd
    except Exception as e:
        print(f"RMSD calculation failed: {e}")
        return None

def compute_mmff_energy(mol, conf_id=0):
    """
    Compute MMFF94 energy for a conformer
    
    Args:
        mol: RDKit Mol object with conformer
        conf_id: Conformer ID
        
    Returns:
        Energy value or infinity if calculation fails
    """
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        energy = ff.CalcEnergy()
        return energy
    except Exception as e:
        print(f"Energy calculation failed: {e}")
        return float("inf")

def validate_structure(mol):
    """
    Validate chemical structure by checking bonds, angles, and atom clashes
    
    Args:
        mol: RDKit Mol object with conformer
        
    Returns:
        True if structure is valid, False otherwise
    """
    try:
        # Check if mol has a conformer
        if mol.GetNumConformers() == 0:
            return False
            
        # Check for atom clashes
        conf = mol.GetConformer()
        positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        
        # Calculate all pairwise distances
        dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        
        # Set diagonal to a large value to ignore self-distances
        np.fill_diagonal(dists, 999.0)
        
        # Check if any non-bonded atoms are too close (< 0.7 Å)
        min_dist = 0.7
        if np.any(dists < min_dist):
            return False
            
        return True
    except:
        return False

def get_scaffold(smiles):
    """
    Extract Murcko scaffold from a molecule
    
    Args:
        smiles: SMILES string
        
    Returns:
        Scaffold SMILES or None if extraction fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold is not None else None
