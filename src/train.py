"""
Training script for Adaptive Hierarchical Contextualization (AHC) experiment.
This module handles the dual training schedule and model training.
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import logging
from preprocess import load_config, prepare_data

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
log_dir = os.path.join(repo_root, 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'train.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CrossSegmentFusion(nn.Module):
    """
    Lightweight fusion module based on multi-head attention.
    It takes a list of segment representations (tensors) and fuses them.
    """
    def __init__(self, hidden_size, num_heads):
        super(CrossSegmentFusion, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            batch_first=True
        )
    
    def forward(self, segment_outputs):
        segment_embeds = torch.stack(
            [torch.mean(seg, dim=1) for seg in segment_outputs], 
            dim=1
        )
        query = torch.mean(segment_embeds, dim=1, keepdim=True)
        fused_output, _ = self.attention(query, segment_embeds, segment_embeds)
        return fused_output

def dual_phase_training(data_package, config):
    """
    Implement the dual training schedule with adaptive context windows.
    Phase 1: Train on standard context lengths
    Phase 2: Fine-tune on ultra-long sequences
    """
    logger.info("Starting dual phase training...")
    
    tokenizer = data_package['tokenizer']
    tokenized_ds_phase1 = data_package['tokenized_ds_phase1']
    tokenized_ds_phase2 = data_package['tokenized_ds_phase2']
    
    model = AutoModelForCausalLM.from_pretrained(config['model']['base_model'])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    os.makedirs('models/phase1', exist_ok=True)
    os.makedirs('models/phase2', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Starting Phase 1 training (standard context lengths)...")
    training_args_phase1 = TrainingArguments(
        output_dir="models/phase1",
        overwrite_output_dir=True,
        num_train_epochs=config['training']['phase1']['num_epochs'],
        per_device_train_batch_size=config['training']['phase1']['batch_size'],
        learning_rate=config['training']['phase1']['learning_rate'],
        logging_dir='logs/phase1',
        logging_steps=10,
        save_steps=50,
        report_to="none"
    )
    
    trainer_phase1 = Trainer(
        model=model,
        args=training_args_phase1,
        data_collator=data_collator,
        train_dataset=tokenized_ds_phase1,
    )
    
    if config['experiment']['test_mode']:
        logger.info("Test mode enabled. Running minimal training for Phase 1.")
        train_result_phase1 = trainer_phase1.train(num_steps=5)
    else:
        train_result_phase1 = trainer_phase1.train()
    
    model.save_pretrained("models/phase1/final")
    
    logger.info("Starting Phase 2 fine-tuning (ultra-long sequences)...")
    training_args_phase2 = TrainingArguments(
        output_dir="models/phase2",
        overwrite_output_dir=True,
        num_train_epochs=config['training']['phase2']['num_epochs'],
        per_device_train_batch_size=config['training']['phase2']['batch_size'],
        learning_rate=config['training']['phase2']['learning_rate'],
        logging_dir='logs/phase2',
        logging_steps=10,
        save_steps=30,
        report_to="none"
    )
    
    trainer_phase2 = Trainer(
        model=model,
        args=training_args_phase2,
        data_collator=data_collator,
        train_dataset=tokenized_ds_phase2,
    )
    
    if config['experiment']['test_mode']:
        logger.info("Test mode enabled. Running minimal training for Phase 2.")
        train_result_phase2 = trainer_phase2.train(num_steps=5)
    else:
        train_result_phase2 = trainer_phase2.train()
    
    model.save_pretrained("models/phase2/final")
    
    if config['output']['save_plots']:
        plot_training_loss(
            [train_result_phase1.training_loss, train_result_phase2.training_loss],
            ['Phase 1', 'Phase 2'],
            'training_loss_baseline'
        )
    
    logger.info("Dual phase training completed successfully.")
    return model

def plot_training_loss(losses, labels, filename_prefix):
    """Plot and save training loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(losses[0]) + 1)
    
    for loss, label in zip(losses, labels):
        plt.plot(epochs, loss, label=label, marker='o')
    
    plt.title("Dual Training Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if not os.path.exists('paper'):
        os.makedirs('paper')
    
    pdf_filename = f"paper/{filename_prefix}_full.pdf"
    plt.savefig(pdf_filename)
    
    pdf_filename_small = f"paper/{filename_prefix}_small.pdf"
    plt.savefig(pdf_filename_small, dpi=100)
    
    plt.close()
    logger.info(f"Saved training loss plot as {pdf_filename}")

def train_model(config_path='config/ahc_config.yaml'):
    """Main function to train the AHC model."""
    config = load_config(config_path)
    
    data_package = prepare_data(config_path)
    
    model = dual_phase_training(data_package, config)
    
    logger.info("Model training completed successfully.")
    return model, data_package

if __name__ == "__main__":
    train_model()
