import os
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import sys

from preprocess import (
    set_random_seed, 
    CustomImageDataset, 
    TextToImageModel, 
    DiffusionModel,
    load_real_prototype_images, 
    identify_weak_classes,
    get_transform
)
from train import (
    DEVICE, 
    check_gpu_memory, 
    get_classifier_model, 
    train_model, 
    plot_training_curves
)
from evaluate import (
    evaluate_model, 
    plot_confusion_matrix, 
    plot_feature_discrepancy,
    get_feature_extractor, 
    extract_features,
    print_classification_report
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config.deca_config import *
except ImportError:
    print("Warning: Could not import config, using default values")
    RANDOM_SEED = 42
    IMAGE_SIZE = (64, 64)
    NUM_CLASSES = 4
    IMAGES_PER_CLASS = 100
    BATCH_SIZE = 32
    DIFFUSION_STRENGTH_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
    WEAK_CLASSES = [1, 3]
    DIFFUSION_SAMPLES_PER_CLASS = 50
    NUM_EPOCHS = 3
    LEARNING_RATE = 1e-3
    RUN_EXPERIMENT_1 = True
    RUN_EXPERIMENT_2 = True
    RUN_EXPERIMENT_3 = True
    RUN_TEST_ONLY = False

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

set_random_seed(RANDOM_SEED)

text_to_image_model = TextToImageModel()
diffusion_model = DiffusionModel()

def generate_synthetic_dataset(num_images_per_class, class_prompts, transform=None):
    """Generate synthetic images for each class using the text-to-image model."""
    synthetic_images, labels = [], []
    for label, prompt in enumerate(class_prompts):
        for _ in range(num_images_per_class):
            img = text_to_image_model.generate(prompt, size=IMAGE_SIZE)
            synthetic_images.append(img)
            labels.append(label)
    
    return CustomImageDataset(synthetic_images, labels, transform=transform)

def experiment1():
    """Experiment 1: Classification Performance with Diffusion-Augmented Data"""
    print("\n=== Experiment 1: Classification Performance with Diffusion-Augmented Data ===\n")
    
    class_prompts = [f"prompt_class_{i}" for i in range(NUM_CLASSES)]
    transform = get_transform()
    
    baseline_dataset = generate_synthetic_dataset(IMAGES_PER_CLASS, class_prompts, transform)
    baseline_loader = DataLoader(baseline_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    real_prototypes = load_real_prototype_images(num_classes=NUM_CLASSES, per_class=5, size=IMAGE_SIZE)
    weak_classes = identify_weak_classes(baseline_dataset, real_prototypes)
    
    diffusion_images, diffusion_labels = [], []
    for idx, prompt in enumerate(class_prompts):
        if idx in weak_classes:
            prototypes = real_prototypes[idx]
            for _ in range(DIFFUSION_SAMPLES_PER_CLASS):
                img = diffusion_model.generate(
                    prompt, conditioning_images=prototypes, 
                    strength=0.7, size=IMAGE_SIZE
                )
                diffusion_images.append(img)
                diffusion_labels.append(idx)
    
    diffusion_dataset = CustomImageDataset(diffusion_images, diffusion_labels, transform=transform)
    
    composite_dataset = ConcatDataset([baseline_dataset, diffusion_dataset])
    composite_loader = DataLoader(composite_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("\nTraining classifier on baseline synthetic dataset...")
    model_baseline = get_classifier_model(num_classes=NUM_CLASSES)
    model_baseline, loss_history_baseline, acc_history_baseline = train_model(
        model_baseline, baseline_loader, num_epochs=NUM_EPOCHS
    )
    
    print("\nTraining classifier on composite dataset...")
    model_composite = get_classifier_model(num_classes=NUM_CLASSES)
    model_composite, loss_history_composite, acc_history_composite = train_model(
        model_composite, composite_loader, num_epochs=NUM_EPOCHS
    )
    
    print("\nEvaluating baseline model...")
    baseline_results = evaluate_model(model_baseline, baseline_loader)
    print(f"Baseline Model - Accuracy: {baseline_results['accuracy']:.2f}%")
    
    print("\nEvaluating composite model...")
    composite_results = evaluate_model(model_composite, composite_loader)
    print(f"Composite Model - Accuracy: {composite_results['accuracy']:.2f}%")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS+1), loss_history_baseline, 'b-', marker='o', label='Baseline')
    plt.plot(range(1, NUM_EPOCHS+1), loss_history_composite, 'r-', marker='s', label='Composite')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS+1), acc_history_baseline, 'b-', marker='o', label='Baseline')
    plt.plot(range(1, NUM_EPOCHS+1), acc_history_composite, 'r-', marker='s', label='Composite')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("logs/experiment1_comparison.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    plot_confusion_matrix(
        baseline_results['labels'], 
        baseline_results['predictions'], 
        NUM_CLASSES, 
        filename="baseline_confusion_matrix.pdf"
    )
    
    plot_confusion_matrix(
        composite_results['labels'], 
        composite_results['predictions'], 
        NUM_CLASSES, 
        filename="composite_confusion_matrix.pdf"
    )
    
    print("\nExperiment 1 completed.\n")

def experiment2():
    """Experiment 2: Ablation Study on Diffusion Strength Parameter"""
    print("\n=== Experiment 2: Ablation Study on Diffusion Strength Parameter ===\n")
    
    feature_extractor = get_feature_extractor()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    selected_class = 1
    prompt = f"prompt_class_{selected_class}"
    
    real_prototypes = load_real_prototype_images(num_classes=NUM_CLASSES, per_class=5, size=IMAGE_SIZE)
    prototypes = real_prototypes[selected_class][:5]
    
    prototype_features = [extract_features(proto, feature_extractor, preprocess) for proto in prototypes]
    mean_proto = np.mean(prototype_features, axis=0)
    
    strength_values = DIFFUSION_STRENGTH_VALUES
    discrepancies = []
    num_samples = 10
    
    for strength in strength_values:
        generated_features = []
        for _ in range(num_samples):
            gen_img = diffusion_model.generate(
                prompt, conditioning_images=prototypes, 
                strength=strength, size=IMAGE_SIZE
            )
            generated_features.append(extract_features(gen_img, feature_extractor, preprocess))
        
        discrepancy = np.mean([np.linalg.norm(g_feat - mean_proto) for g_feat in generated_features])
        discrepancies.append(discrepancy)
        print(f"Diffusion strength {strength}: Avg. feature discrepancy = {discrepancy:.4f}")
    
    plot_feature_discrepancy(strength_values, discrepancies, filename="feature_discrepancy_ablation.pdf")
    print("\nExperiment 2 completed.\n")

def experiment3():
    """Experiment 3: Impact of Domain Mixing Strategies via Curriculum Learning"""
    print("\n=== Experiment 3: Adaptive Domain Mixing via Curriculum Learning ===\n")
    
    class_prompts = [f"prompt_class_{i}" for i in range(NUM_CLASSES)]
    transform = get_transform()
    
    baseline_images, baseline_labels = [], []
    for label, prompt in enumerate(class_prompts):
        for _ in range(IMAGES_PER_CLASS):
            img = text_to_image_model.generate(prompt, size=IMAGE_SIZE)
            baseline_images.append(img)
            baseline_labels.append(label)
    
    baseline_dataset = CustomImageDataset(baseline_images, baseline_labels, transform=transform)
    baseline_loader = DataLoader(baseline_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    diffusion_images, diffusion_labels = [], []
    for idx, prompt in enumerate(class_prompts):
        if idx in WEAK_CLASSES:
            real_protos = load_real_prototype_images(num_classes=NUM_CLASSES, per_class=5, size=IMAGE_SIZE)[idx]
            for _ in range(DIFFUSION_SAMPLES_PER_CLASS):
                img = diffusion_model.generate(
                    prompt, conditioning_images=real_protos, 
                    strength=0.7, size=IMAGE_SIZE
                )
                diffusion_images.append(img)
                diffusion_labels.append(idx)
    
    diffusion_dataset = CustomImageDataset(diffusion_images, diffusion_labels, transform=transform)
    diffusion_loader = DataLoader(diffusion_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = get_classifier_model(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    def get_mixed_batch(epoch, total_epochs, batch_baseline, batch_diffused):
        ratio = epoch / total_epochs  # Increases over epochs
        mixed_images = []
        mixed_labels = []
        
        min_len = min(len(batch_baseline[0]), len(batch_diffused[0]))
        for i in range(min_len):
            if random.random() < ratio:
                mixed_images.append(batch_diffused[0][i])
                mixed_labels.append(batch_diffused[1][i])
            else:
                mixed_images.append(batch_baseline[0][i])
                mixed_labels.append(batch_baseline[1][i])
        
        mixed_images = torch.stack(mixed_images)
        mixed_labels = torch.tensor(mixed_labels)
        
        try:
            return mixed_images.to(DEVICE), mixed_labels.to(DEVICE)
        except RuntimeError as e:
            print(f"Warning: Could not move tensors to device {DEVICE}. Using CPU instead. Error: {e}")
            return mixed_images, mixed_labels
    
    total_epochs = 5  # Using fewer epochs for demonstration
    loss_history = []
    accuracy_history = []
    
    print("Starting training with adaptive (curriculum) mixing strategy...")
    for epoch in range(1, total_epochs+1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for (images_b, labels_b), (images_d, labels_d) in zip(baseline_loader, diffusion_loader):
            mixed_images, mixed_labels = get_mixed_batch(
                epoch, total_epochs, (images_b, labels_b), (images_d, labels_d)
            )
            
            optimizer.zero_grad()
            outputs = model(mixed_images)
            loss = criterion(outputs, mixed_labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += mixed_labels.size(0)
            correct += (predicted == mixed_labels).sum().item()
        
        avg_epoch_loss = epoch_loss / len(baseline_loader)
        accuracy = 100 * correct / total
        loss_history.append(avg_epoch_loss)
        accuracy_history.append(accuracy)
        
        print(f"Epoch {epoch}/{total_epochs} - Loss: {avg_epoch_loss:.4f} - Accuracy: {accuracy:.2f}%")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, total_epochs+1), loss_history, 'g-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Curriculum Training Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, total_epochs+1), accuracy_history, 'g-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Curriculum Training Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("logs/curriculum_training_curves.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    torch.save(model.state_dict(), os.path.join("models", "curriculum_model.pth"))
    
    print("\nExperiment 3 completed.\n")

def test_code():
    """Test function to quickly verify the code execution."""
    print("\n=== Running Quick Test ===\n")
    
    print(f"Using device: {DEVICE}")
    check_gpu_memory()
    
    try:
        global NUM_CLASSES, IMAGES_PER_CLASS, BATCH_SIZE, NUM_EPOCHS
        NUM_CLASSES = 2  # Reduce number of classes
        IMAGES_PER_CLASS = 10  # Reduce images per class
        BATCH_SIZE = 4  # Reduce batch size
        NUM_EPOCHS = 1  # Run only one epoch
        
        experiment1()
        print("\nQuick test completed successfully.\n")
    except Exception as e:
        print(f"Error during test execution: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run all experiments."""
    start_time = time.time()
    
    print("=" * 80)
    print("DECA: Diffusion-Enhanced Concept Augmentation")
    print("=" * 80)
    print(f"Running on device: {DEVICE}")
    print(f"Configuration: {NUM_CLASSES} classes, {IMAGES_PER_CLASS} images per class")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
    print("=" * 80)
    
    check_gpu_memory()
    
    if RUN_TEST_ONLY:
        test_code()
    else:
        if RUN_EXPERIMENT_1:
            experiment1()
        
        if RUN_EXPERIMENT_2:
            experiment2()
        
        if RUN_EXPERIMENT_3:
            experiment3()
    
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print("\nAll experiments completed.")

if __name__ == '__main__':
    main()
