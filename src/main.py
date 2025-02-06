import yaml
from pathlib import Path
from train import train_model
from evaluate import evaluate_model
from preprocess import load_and_preprocess_data

def main():
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = train_model(config)
    
    test_data = load_and_preprocess_data(Path('data/test.txt'))
    metrics = evaluate_model(model, test_data)
    
    print("Evaluation metrics:", metrics)

if __name__ == "__main__":
    main()
