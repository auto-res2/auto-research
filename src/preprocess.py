"""
Preprocessing module for the STEM method.
Handles data loading, preprocessing, and feature extraction.
"""

import torch
from transformers import AutoTokenizer, AutoModel

torch.manual_seed(42)

tokenizer = None
model_transformer = None

def initialize_models():
    """Initialize the tokenizer and transformer models."""
    global tokenizer, model_transformer
    if tokenizer is None or model_transformer is None:
        print("Initializing tokenizer and transformer models...")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model_transformer = AutoModel.from_pretrained("roberta-base")
        model_transformer.eval()  # set to eval mode as dropout is not needed

def extract_embeddings(text_list):
    """
    Extract embeddings for a list of texts using the pretrained transformer.
    Uses mean pooling over the last hidden state.
    Returns a tensor of shape (batch_size, embedding_dim)
    """
    initialize_models()
    
    global tokenizer, model_transformer
    if tokenizer is None or model_transformer is None:
        raise ValueError("Tokenizer or model not initialized properly")
        
    inputs = tokenizer(text_list, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model_transformer(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def preprocess_data(data_path=None):
    """
    Preprocess data for training and evaluation.
    For demonstration, this function generates synthetic data.
    In a real scenario, this would load and preprocess real data.
    """
    print("Preprocessing data...")
    return None
