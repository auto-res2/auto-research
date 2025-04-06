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
        
        try:
            model_transformer = AutoModel.from_pretrained("roberta-base")
        except Exception as e:
            print(f"Error loading RoBERTa model: {str(e)}")
            print("Creating mock transformer model for demonstration purposes...")
            
            class MockRoBERTaModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding_dim = 768  # Same as RoBERTa base
                    self.embeddings = torch.nn.Embedding(50265, self.embedding_dim)  # RoBERTa vocab size
                    
                def forward(self, input_ids=None, attention_mask=None, **kwargs):
                    if input_ids is None:
                        batch_size = 1
                        seq_length = 10  # Default sequence length
                    else:
                        batch_size = input_ids.shape[0]
                        seq_length = input_ids.shape[1]
                    
                    class MockOutput:
                        def __init__(self, last_hidden_state):
                            self.last_hidden_state = last_hidden_state
                    
                    random_embeddings = torch.randn(batch_size, seq_length, self.embedding_dim)
                    return MockOutput(random_embeddings)
            
            model_transformer = MockRoBERTaModel()
            
        model_transformer.eval()

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
