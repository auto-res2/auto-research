"""
Main script for Adaptive Hierarchical Contextualization (AHC) experiment.
This script integrates preprocessing, training, and evaluation stages.
"""

import os
import sys
import argparse
import datetime
import logging
from preprocess import prepare_data, load_config
from train import train_model
from evaluate import evaluate_model

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
log_dir = os.path.join(repo_root, 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'main.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Adaptive Hierarchical Contextualization (AHC) experiment')
    parser.add_argument('--config', type=str, default='config/ahc_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only run evaluation')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip evaluation and only run training')
    parser.add_argument('--model_path', type=str, default='models/phase2/final',
                        help='Path to trained model for evaluation')
    return parser.parse_args()

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['config', 'data', 'logs', 'models', 'paper']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("Created necessary directories.")

def run_experiment(args):
    """Run the complete AHC experiment."""
    start_time = datetime.datetime.now()
    logger.info(f"Starting Adaptive Hierarchical Contextualization experiment at {start_time}")
    
    create_directories()
    
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    logger.info("Experiment Configuration:")
    logger.info(f"- Base Model: {config['model']['base_model']}")
    logger.info(f"- Segment Threshold: {config['model']['segment_threshold']}")
    logger.info(f"- Number of Clusters: {config['model']['n_clusters']}")
    logger.info(f"- Dropout Samples: {config['experiment']['dropout_samples']}")
    logger.info(f"- Test Mode: {config['experiment']['test_mode']}")
    
    if not args.skip_training:
        logger.info("Starting model training...")
        model, data_package = train_model(args.config)
        logger.info("Model training completed.")
    else:
        logger.info("Skipping training as requested.")
        model = None
    
    if not args.skip_evaluation:
        logger.info("Starting model evaluation...")
        if args.skip_training:
            evaluation_results = evaluate_model(args.model_path, args.config)
        else:
            evaluation_results = evaluate_model("models/phase2/final", args.config)
        
        logger.info("Evaluation Results:")
        logger.info(f"- Number of segments: {len(evaluation_results['segments'])}")
        logger.info(f"- Perplexity (Base): {evaluation_results['perplexity']['base']:.2f}")
        logger.info(f"- Perplexity (AHC): {evaluation_results['perplexity']['ahc']:.2f}")
        logger.info(f"- Example response: {evaluation_results['responses'][0][:100]}...")
    else:
        logger.info("Skipping evaluation as requested.")
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logger.info(f"Experiment completed in {duration}")
    
    return {
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration,
        'config': config
    }

def main():
    """Main entry point for the experiment."""
    args = parse_args()
    
    experiment_results = run_experiment(args)
    
    print("\n" + "="*80)
    print("ADAPTIVE HIERARCHICAL CONTEXTUALIZATION (AHC) EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Start time: {experiment_results['start_time']}")
    print(f"End time: {experiment_results['end_time']}")
    print(f"Duration: {experiment_results['duration']}")
    print(f"Configuration: {args.config}")
    print("="*80 + "\n")
    
    logger.info("Experiment completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
