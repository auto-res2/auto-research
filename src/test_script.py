"""
Simple test script to verify the Adaptive Hierarchical Contextualization (AHC) implementation.
This script runs a minimal test of the main components without full training.
"""

import os
import sys
import logging
from preprocess import load_config, get_tokenizer, get_sentence_embedder, segment_text
from transformers import AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_segmentation():
    """Test the text segmentation functionality."""
    logger.info("Testing text segmentation...")
    
    config = load_config('../config/ahc_config.yaml')
    
    tokenizer = get_tokenizer(config['model']['base_model'])
    embedder = get_sentence_embedder()
    
    test_text = "This is a test sentence. This is another test sentence. " * 10
    
    segments = segment_text(
        test_text, 
        tokenizer, 
        embedder, 
        threshold=50,  # Small threshold to force segmentation
        n_clusters=2   # Small number of clusters for quick testing
    )
    
    logger.info(f"Segmentation test completed. Number of segments: {len(segments)}")
    return segments

def test_model_loading():
    """Test loading the model."""
    logger.info("Testing model loading...")
    
    config = load_config('../config/ahc_config.yaml')
    
    model = AutoModelForCausalLM.from_pretrained(config['model']['base_model'])
    
    logger.info(f"Model loading test completed. Model type: {type(model).__name__}")
    return model

def main():
    """Run all tests."""
    logger.info("Starting simple test run for AHC implementation...")
    
    segments = test_segmentation()
    
    model = test_model_loading()
    
    logger.info("All tests completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
