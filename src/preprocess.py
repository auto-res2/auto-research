"""
Preprocessing script for Adaptive Hierarchical Contextualization (AHC) experiment.
This module handles text segmentation and embedding for the AHC methodology.
"""

import os
import yaml
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer
from datasets import load_dataset
import logging

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
log_dir = os.path.join(repo_root, 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'preprocess.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path='config/ahc_config.yaml'):
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_tokenizer(model_name):
    """Get tokenizer for the specified model."""
    logger.info(f"Loading tokenizer for model: {model_name}")
    return AutoTokenizer.from_pretrained(model_name)

def get_sentence_embedder(model_name="all-MiniLM-L6-v2"):
    """Get sentence embedder model."""
    logger.info(f"Loading sentence embedder: {model_name}")
    return SentenceTransformer(model_name)

def load_dataset_for_experiment(config):
    """Load dataset based on configuration."""
    dataset_name = config['data']['dataset']
    subset = config['data'].get('subset', None)
    split = config['data'].get('split', 'train')
    
    logger.info(f"Loading dataset: {dataset_name}, subset: {subset}, split: {split}")
    
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    return dataset

def segment_text(text, tokenizer, embedder, threshold=8000, n_clusters=5):
    """
    Segment a long text into semantically coherent chunks.
    For texts shorter than threshold (token count) returns the full text.
    Otherwise uses SentenceTransformer to embed sentences and clusters them.
    """
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= threshold:
        logger.info("Text is short enough; no segmentation performed.")
        return [text]
    
    sentences = text.split('. ')
    logger.info(f"Segmenting text into {len(sentences)} sentences...")
    
    embeddings = embedder.encode(sentences)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    segments = {}
    for label, sentence in zip(labels, sentences):
        segments.setdefault(label, []).append(sentence)
    
    segment_texts = [". ".join(seg) for seg in segments.values()]
    logger.info(f"Text segmented into {len(segment_texts)} segments.")
    return segment_texts

def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenize a dataset with the given tokenizer and max length."""
    logger.info(f"Tokenizing dataset with max length: {max_length}")
    return dataset.map(
        lambda examples: tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=max_length
        ),
        batched=True
    )

def prepare_data(config_path='config/ahc_config.yaml'):
    """Main function to prepare data for the AHC experiment."""
    config = load_config(config_path)
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    tokenizer = get_tokenizer(config['model']['base_model'])
    embedder = get_sentence_embedder()
    
    dataset = load_dataset_for_experiment(config)
    
    tokenized_ds_phase1 = tokenize_dataset(
        dataset, 
        tokenizer, 
        config['training']['phase1']['context_length']
    )
    
    tokenized_ds_phase2 = tokenize_dataset(
        dataset, 
        tokenizer, 
        config['training']['phase2']['context_length']
    )
    
    logger.info("Data preparation completed successfully.")
    return {
        'tokenizer': tokenizer,
        'embedder': embedder,
        'dataset': dataset,
        'tokenized_ds_phase1': tokenized_ds_phase1,
        'tokenized_ds_phase2': tokenized_ds_phase2,
        'config': config
    }

if __name__ == "__main__":
    prepare_data()
