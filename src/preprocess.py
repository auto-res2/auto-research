"""
Data preprocessing for DPC-3D experiment
"""
import os
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
import selfies as sf
from tqdm import tqdm
import pickle
import logging
from pathlib import Path

from src.utils.molecular_utils import (
    smiles_to_mol, 
    generate_conformer, 
    get_scaffold
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MolecularDataPreprocessor:
    """
    Preprocessor for molecular data to be used in DPC-3D experiments
    """
    
    def __init__(self, config):
        """
        Initialize the preprocessor with configuration
        
        Args:
            config: Configuration dictionary with data parameters
        """
        self.config = config
        self.max_atoms = config["max_atoms"]
        self.max_conformers = config["max_conformers"]
        self.processed_dir = Path(config["processed_data_dir"])
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        
        # Create vocabulary for SELFIES encoding
        self.vocab = None
        self.token_to_idx = None
        self.idx_to_token = None
        
    def create_vocabulary(self, smiles_list):
        """
        Create vocabulary from SELFIES representations of molecules
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            vocab: Set of unique tokens
            token_to_idx: Mapping from token to index
            idx_to_token: Mapping from index to token
        """
        logger.info("Creating vocabulary from SELFIES representations")
        
        # Convert SMILES to SELFIES
        selfies_list = [sf.encoder(s) for s in tqdm(smiles_list) if self._is_valid_smiles(s)]
        
        # Extract unique tokens
        all_tokens = set()
        for s in selfies_list:
            tokens = list(sf.split_selfies(s))
            all_tokens.update(tokens)
            
        # Add special tokens
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        all_tokens.update(special_tokens)
        
        # Create mappings
        vocab = sorted(list(all_tokens))
        token_to_idx = {token: idx for idx, token in enumerate(vocab)}
        idx_to_token = {idx: token for token, idx in token_to_idx.items()}
        
        # Save vocabulary
        self.vocab = vocab
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token
        
        # Save to disk
        vocab_path = self.processed_dir / "vocabulary.pkl"
        with open(vocab_path, "wb") as f:
            pickle.dump({
                "vocab": vocab,
                "token_to_idx": token_to_idx,
                "idx_to_token": idx_to_token
            }, f)
            
        logger.info(f"Vocabulary created with {len(vocab)} tokens and saved to {vocab_path}")
        
        return vocab, token_to_idx, idx_to_token
        
    def _is_valid_smiles(self, smiles):
        """Check if SMILES string is valid"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None and mol.GetNumAtoms() <= self.max_atoms
        except:
            return False
            
    def _smiles_to_selfies_tokens(self, smiles):
        """Convert SMILES to SELFIES tokens"""
        try:
            selfies = sf.encoder(smiles)
            tokens = list(sf.split_selfies(selfies))
            return tokens
        except:
            return None
            
    def _tokenize_selfies(self, tokens):
        """Convert SELFIES tokens to indices"""
        if tokens is None:
            return None
            
        # Add BOS and EOS tokens
        tokens = ["<BOS>"] + tokens + ["<EOS>"]
        
        # Convert to indices
        if self.token_to_idx is None:
            return None
        indices = [self.token_to_idx.get(token, self.token_to_idx["<UNK>"]) 
                  for token in tokens]
        
        return indices
        
    def _pad_sequence(self, seq, max_len):
        """Pad sequence to max_len"""
        if self.token_to_idx is None:
            return None
            
        if len(seq) >= max_len:
            return seq[:max_len]
        else:
            return seq + [self.token_to_idx["<PAD>"]] * (max_len - len(seq))
            
    def _get_atom_types(self, mol):
        """Get atom types as indices"""
        atom_types = []
        for atom in mol.GetAtoms():
            # Map common atom types to indices
            # 0: Unknown, 1: C, 2: N, 3: O, 4: F, 5: P, 6: S, 7: Cl, 8: Br, 9: I
            symbol = atom.GetSymbol()
            if symbol == 'C':
                atom_types.append(1)
            elif symbol == 'N':
                atom_types.append(2)
            elif symbol == 'O':
                atom_types.append(3)
            elif symbol == 'F':
                atom_types.append(4)
            elif symbol == 'P':
                atom_types.append(5)
            elif symbol == 'S':
                atom_types.append(6)
            elif symbol == 'Cl':
                atom_types.append(7)
            elif symbol == 'Br':
                atom_types.append(8)
            elif symbol == 'I':
                atom_types.append(9)
            else:
                atom_types.append(0)
                
        # Pad to max_atoms
        if len(atom_types) < self.max_atoms:
            atom_types = atom_types + [0] * (self.max_atoms - len(atom_types))
        else:
            atom_types = atom_types[:self.max_atoms]
            
        return atom_types
        
    def _get_coordinates(self, mol, conf_id=0):
        """Get 3D coordinates from conformer"""
        if mol.GetNumConformers() == 0:
            return None
            
        conf = mol.GetConformer(conf_id)
        coords = []
        
        for i in range(min(mol.GetNumAtoms(), self.max_atoms)):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
            
        # Pad to max_atoms
        if len(coords) < self.max_atoms:
            coords = coords + [[0, 0, 0]] * (self.max_atoms - len(coords))
            
        return coords
        
    def _create_attention_mask(self, num_atoms):
        """Create attention mask for transformer"""
        mask = torch.zeros(self.max_atoms, self.max_atoms)
        mask[:num_atoms, :num_atoms] = 1
        return mask
        
    def process_dataset(self, smiles_path, split=True):
        """
        Process dataset from SMILES file
        
        Args:
            smiles_path: Path to CSV file with SMILES strings
            split: Whether to split into train/val/test
            
        Returns:
            processed_data: Dictionary with processed data
        """
        logger.info(f"Processing dataset from {smiles_path}")
        
        # Load SMILES data
        df = pd.read_csv(smiles_path)
        
        # Ensure 'smiles' column exists
        if 'smiles' not in df.columns:
            smiles_col = df.columns[0]  # Assume first column is SMILES
            df = df.rename(columns={smiles_col: 'smiles'})
            
        # Filter valid molecules
        valid_smiles = [s for s in df['smiles'] if self._is_valid_smiles(s)]
        logger.info(f"Found {len(valid_smiles)} valid molecules out of {len(df)}")
        
        # Create vocabulary if not already created
        if self.vocab is None:
            self.create_vocabulary(valid_smiles)
            
        # Process each molecule
        processed_data = []
        
        for smiles in tqdm(valid_smiles, desc="Processing molecules"):
            # Convert to RDKit mol
            mol = smiles_to_mol(smiles)
            if mol is None:
                continue
                
            # Generate 3D conformers
            mol_with_conf = generate_conformer(mol, num_conf=self.max_conformers)
            if mol_with_conf.GetNumConformers() == 0:
                continue
                
            # Get SELFIES tokens
            selfies_tokens = self._smiles_to_selfies_tokens(smiles)
            if selfies_tokens is None:
                continue
                
            # Tokenize and pad
            token_ids = self._tokenize_selfies(selfies_tokens)
            if token_ids is None:
                continue
                
            padded_ids = self._pad_sequence(token_ids, self.config["lm"]["max_seq_len"])
            
            # Get atom types
            atom_types = self._get_atom_types(mol_with_conf)
            
            # Get coordinates for each conformer
            conformers = []
            for conf_id in range(min(mol_with_conf.GetNumConformers(), self.max_conformers)):
                coords = self._get_coordinates(mol_with_conf, conf_id)
                if coords is not None:
                    conformers.append(coords)
                    
            if not conformers:
                continue
                
            # Get scaffold
            scaffold = get_scaffold(smiles)
            
            # Create attention mask
            num_atoms = min(mol_with_conf.GetNumAtoms(), self.max_atoms)
            mask = self._create_attention_mask(num_atoms)
            
            # Store processed data
            processed_data.append({
                'smiles': smiles,
                'selfies_tokens': selfies_tokens,
                'token_ids': padded_ids,
                'mol': mol_with_conf,
                'atom_types': atom_types,
                'conformers': conformers,
                'num_atoms': num_atoms,
                'mask': mask,
                'scaffold': scaffold
            })
            
        logger.info(f"Successfully processed {len(processed_data)} molecules")
        
        # Split dataset if requested
        if split:
            return self._split_dataset(processed_data)
        else:
            # Save full dataset
            output_path = self.processed_dir / "full_dataset.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(processed_data, f)
            logger.info(f"Full dataset saved to {output_path}")
            return processed_data
            
    def _split_dataset(self, data):
        """Split dataset into train/val/test"""
        # Shuffle data
        np.random.seed(self.config["seed"])
        np.random.shuffle(data)
        
        # Calculate split indices
        n = len(data)
        train_idx = int(n * self.config["train_split"])
        val_idx = train_idx + int(n * self.config["val_split"])
        
        # Split data
        train_data = data[:train_idx]
        val_data = data[train_idx:val_idx]
        test_data = data[val_idx:]
        
        logger.info(f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        # Save splits
        splits = {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
        
        for split_name, split_data in splits.items():
            output_path = self.processed_dir / f"{split_name}_dataset.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(split_data, f)
            logger.info(f"{split_name.capitalize()} dataset saved to {output_path}")
            
        return splits
        
    def create_scaffold_splits(self, data):
        """
        Create scaffold-based splits for testing generalization
        
        Args:
            data: List of processed molecule data
            
        Returns:
            scaffold_splits: Dictionary with scaffold-based splits
        """
        # Group by scaffold
        scaffold_to_mols = {}
        for mol_data in data:
            scaffold = mol_data['scaffold']
            if scaffold not in scaffold_to_mols:
                scaffold_to_mols[scaffold] = []
            scaffold_to_mols[scaffold].append(mol_data)
            
        # Sort scaffolds by frequency
        scaffolds = sorted(scaffold_to_mols.keys(), 
                          key=lambda s: len(scaffold_to_mols[s]), 
                          reverse=True)
        
        # Split scaffolds into seen and unseen
        num_scaffolds = len(scaffolds)
        seen_scaffolds = scaffolds[:int(num_scaffolds * 0.8)]
        unseen_scaffolds = scaffolds[int(num_scaffolds * 0.8):]
        
        # Create splits
        seen_data = []
        for scaffold in seen_scaffolds:
            seen_data.extend(scaffold_to_mols[scaffold])
            
        unseen_data = []
        for scaffold in unseen_scaffolds:
            unseen_data.extend(scaffold_to_mols[scaffold])
            
        logger.info(f"Scaffold split: {len(seen_data)} seen molecules, {len(unseen_data)} unseen molecules")
        
        # Save splits
        scaffold_splits = {
            "seen": seen_data,
            "unseen": unseen_data
        }
        
        for split_name, split_data in scaffold_splits.items():
            output_path = self.processed_dir / f"{split_name}_scaffolds.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(split_data, f)
            logger.info(f"{split_name.capitalize()} scaffold dataset saved to {output_path}")
            
        return scaffold_splits

def preprocess_data(config):
    """
    Main function to preprocess data for DPC-3D experiment
    
    Args:
        config: Configuration dictionary
        
    Returns:
        preprocessor: Initialized preprocessor with data
    """
    # Create preprocessor
    preprocessor = MolecularDataPreprocessor(config)
    
    # Process dataset
    data_splits = preprocessor.process_dataset(config["smiles_path"])
    
    # Create scaffold splits for generalization experiment
    if isinstance(data_splits, dict) and "test" in data_splits:
        scaffold_splits = preprocessor.create_scaffold_splits(data_splits["test"])
    
    return preprocessor

if __name__ == "__main__":
    # Import config when run as script
    from config.dpc3d_config import DATA_CONFIG, MODEL_CONFIG, EXPERIMENT_CONFIG
    
    # Combine configs
    config = {**DATA_CONFIG, **MODEL_CONFIG, **EXPERIMENT_CONFIG}
    
    # Preprocess data
    preprocess_data(config)
