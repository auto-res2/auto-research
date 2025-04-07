
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr_ssim(original, processed):
    """
    original, processed: tensors of shape (B, C, H, W), assumed to be in [0,1]
    Returns lists of PSNR and SSIM per image.
    """
    original_np = original.detach().cpu().numpy().transpose(0,2,3,1)  # (B, H, W, C)
    processed_np = processed.detach().cpu().numpy().transpose(0,2,3,1)
    psnr_list = []
    ssim_list = []
    for orig, proc in zip(original_np, processed_np):
        psnr_val = peak_signal_noise_ratio(orig, proc, data_range=1)
        ssim_val = structural_similarity(orig, proc, channel_axis=2, data_range=1)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
    return psnr_list, ssim_list

def experiment_ablation(device, diffusion_model, test_loader):
    print("\n=== Experiment 1: Ablation Study on Dual-Stage Denoising and Consistency Loss ===")
    batch = next(iter(test_loader))
    x, y = batch
    x = x.to(device)
    try:
        import torchattacks
        HAS_TA = True
        from train import DummyClassifier
        dummy_classifier = DummyClassifier().to(device)
        attack = torchattacks.PGD(dummy_classifier, eps=8/255, alpha=2/255, steps=10)
        x_adv = attack(x, y.to(device))
    except ImportError:
        print("torchattacks not found; will simulate adversarial perturbations.")
        HAS_TA = False
        x_adv = x + torch.randn_like(x) * 0.05
        
    noise_level = 0.3

    with torch.no_grad():
        from train import run_purification
        purified_base = run_purification(x_adv, noise_level, diffusion_model, variant='base')
        purified_dual = run_purification(x_adv, noise_level, diffusion_model, variant='dual')
        purified_cedp = run_purification(x_adv, noise_level, diffusion_model, variant='cedp')

    psnr_base, ssim_base = compute_psnr_ssim(x, purified_base)
    psnr_dual, ssim_dual = compute_psnr_ssim(x, purified_dual)
    psnr_cedp, ssim_cedp = compute_psnr_ssim(x, purified_cedp)
    print("Average PSNR (Base): {:.2f} dB".format(np.mean(psnr_base)))
    print("Average PSNR (Dual): {:.2f} dB".format(np.mean(psnr_dual)))
    print("Average PSNR (CEDP): {:.2f} dB".format(np.mean(psnr_cedp)))
    print("Average SSIM (Base): {:.4f}".format(np.mean(ssim_base)))
    print("Average SSIM (Dual): {:.4f}".format(np.mean(ssim_dual)))
    print("Average SSIM (CEDP): {:.4f}".format(np.mean(ssim_cedp)))

    variants = ["Base", "Dual", "CEDP"]
    avg_psnr = [np.mean(psnr_base), np.mean(psnr_dual), np.mean(psnr_cedp)]
    plt.figure()
    plt.bar(variants, avg_psnr, color=['blue', 'orange', 'green'])
    plt.xlabel("Purification Variant")
    plt.ylabel("Average PSNR (dB)")
    plt.title("Ablation Study: PSNR Comparison")
    plt.tight_layout()
    plt.savefig("logs/psnr_ablation_pair1.pdf")
    plt.close()

    avg_ssim = [np.mean(ssim_base), np.mean(ssim_dual), np.mean(ssim_cedp)]
    plt.figure()
    plt.bar(variants, avg_ssim, color=['blue', 'orange', 'green'])
    plt.xlabel("Purification Variant")
    plt.ylabel("Average SSIM")
    plt.title("Ablation Study: SSIM Comparison")
    plt.tight_layout()
    plt.savefig("logs/ssim_ablation_pair1.pdf")
    plt.close()
    print("Experiment 1 plots saved as 'logs/psnr_ablation_pair1.pdf' and 'logs/ssim_ablation_pair1.pdf'.")

    return {
        "psnr": {
            "base": np.mean(psnr_base),
            "dual": np.mean(psnr_dual),
            "cedp": np.mean(psnr_cedp)
        },
        "ssim": {
            "base": np.mean(ssim_base),
            "dual": np.mean(ssim_dual),
            "cedp": np.mean(ssim_cedp)
        }
    }

def experiment_robustness(device, diffusion_model, classifier, test_loader):
    print("\n=== Experiment 2: Adversarial Robustness Benchmarking ===")
    classifier.eval().to(device)
    accuracies = {'base': [], 'cedp': []}
    total_samples = 0
    correct_base = 0
    correct_cedp = 0
    noise_level = 0.3

    try:
        import torchattacks
        HAS_TA = True
        attack = torchattacks.PGD(classifier, eps=8/255, alpha=2/255, steps=10)
    except ImportError:
        print("torchattacks not found; will simulate adversarial perturbations.")
        HAS_TA = False
        attack = None

    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        if attack is not None:
            x_adv = attack(x, y)
        else:
            x_adv = x + torch.randn_like(x)*0.05

        with torch.no_grad():
            from train import run_purification
            purified_base = run_purification(x_adv, noise_level, diffusion_model, variant='base')
            purified_cedp = run_purification(x_adv, noise_level, diffusion_model, variant='cedp')
            out_base = classifier(purified_base)
            out_cedp = classifier(purified_cedp)
            pred_base = out_base.argmax(dim=1)
            pred_cedp = out_cedp.argmax(dim=1)
            correct_base += (pred_base == y).sum().item()
            correct_cedp += (pred_cedp == y).sum().item()
            total_samples += y.size(0)
        if total_samples > 128:
            break

    acc_base = 100.0 * correct_base/total_samples
    acc_cedp = 100.0 * correct_cedp/total_samples
    print("Total samples evaluated: {}".format(total_samples))
    print("Classification accuracy on purified adversarial images:")
    print("  Base Method: {:.2f}%".format(acc_base))
    print("  CEDP Method: {:.2f}%".format(acc_cedp))
    
    variants = ["Base", "CEDP"]
    acc_values = [acc_base, acc_cedp]
    plt.figure()
    plt.bar(variants, acc_values, color=['red', 'green'])
    plt.xlabel("Purification Variant")
    plt.ylabel("Accuracy (%)")
    plt.title("Adversarial Robustness Comparison")
    plt.tight_layout()
    plt.savefig("logs/accuracy_robustness_pair1.pdf")
    plt.close()
    print("Experiment 2 plot saved as 'logs/accuracy_robustness_pair1.pdf'.")
    
    return {
        "accuracy": {
            "base": acc_base,
            "cedp": acc_cedp
        }
    }

def experiment_adaptive_control(device, diffusion_model, test_loader):
    print("\n=== Experiment 3: Effectiveness of Adaptive Randomness Control ===")
    batch = next(iter(test_loader))
    x, _ = batch
    x = x.to(device)
    x_adv = x + torch.randn_like(x)*0.05

    import time
    from train import run_adaptive_purification, run_fixed_purification
    
    start_time = time.time()
    purified_adaptive, noise_records = run_adaptive_purification(
        x_adv, diffusion_model, initial_noise_level=0.3, iterations=5, threshold=0.01)
    adaptive_time = time.time() - start_time

    start_time = time.time()
    purified_fixed = run_fixed_purification(x_adv, diffusion_model, noise_level=0.3, iterations=5)
    fixed_time = time.time() - start_time

    print("Adaptive purification time: {:.4f} seconds".format(adaptive_time))
    print("Fixed purification time: {:.4f} seconds".format(fixed_time))
    
    psnr_adaptive, _ = compute_psnr_ssim(x, purified_adaptive)
    psnr_fixed, _ = compute_psnr_ssim(x, purified_fixed)
    print("Average PSNR (Adaptive): {:.2f} dB".format(np.mean(psnr_adaptive)))
    print("Average PSNR (Fixed): {:.2f} dB".format(np.mean(psnr_fixed)))

    iterations = list(range(len(noise_records)))
    plt.figure()
    plt.plot(iterations, noise_records, marker='o', linestyle='-', color='purple')
    plt.xlabel("Iteration")
    plt.ylabel("Noise Level")
    plt.title("Adaptive Randomness Control")
    plt.tight_layout()
    plt.savefig("logs/noise_adaptive_pair1.pdf")
    plt.close()
    print("Experiment 3 plot saved as 'logs/noise_adaptive_pair1.pdf'.")
    
    return {
        "time": {
            "adaptive": adaptive_time,
            "fixed": fixed_time
        },
        "psnr": {
            "adaptive": np.mean(psnr_adaptive),
            "fixed": np.mean(psnr_fixed)
        },
        "noise_records": noise_records
    }
