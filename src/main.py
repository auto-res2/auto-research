"""
Main script for running the LRE-CDT experiment.
This file orchestrates three experiments:
1. Quantitative and Qualitative Comparison with Baselines
2. Ablation Study on the Localized Residual Module
3. Efficiency Evaluation via Diffusion Step Reduction
"""

import torch
import time
import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from preprocess import get_dataloader
from train import (
    LRE_CDT, CAT_DM, LRE_CDT_Full, LRE_CDT_NoResidue, 
    LRE_CDT_GlobalResidue, train_model, set_seed
)
from evaluate import (
    compute_lpips, compute_ssim, compute_dummy_fid, compute_region_metric,
    save_comparison_figure, save_ablation_figure, save_efficiency_plots
)

set_seed(42)

def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def experiment1(device, config):
    """
    Quantitative and Qualitative Comparison with Baselines.
    
    Args:
        device: Device to run experiment on
        config: Configuration dictionary
    """
    print("\n" + "="*80)
    print("Running Experiment 1: Quantitative and Qualitative Comparison with Baselines")
    print("="*80)
    
    dataloader = get_dataloader(config)
    
    lre_cdt_model = LRE_CDT().to(device)
    baseline_model = CAT_DM().to(device)
    
    lre_cdt_model = train_model(lre_cdt_model, dataloader, device, config)
    baseline_model = train_model(baseline_model, dataloader, device, config)
    
    lre_results = []
    baseline_results = []
    
    lre_all_outputs = []
    baseline_all_outputs = []
    ref_all_images = []
    
    lre_cdt_model.eval()
    baseline_model.eval()
    
    with torch.no_grad():
        for i, (images, garment_masks) in enumerate(dataloader):
            if i > 1:
                break
            images = images.to(device)
            garment_masks = garment_masks.to(device)
            
            out_lre = lre_cdt_model.generate(images, control_signal=garment_masks)
            out_base = baseline_model.generate(images, control_signal=garment_masks)
            
            for j in range(images.size(0)):
                lpips_lre = compute_lpips(out_lre[j], images[j])
                ssim_lre = compute_ssim(out_lre[j], images[j])
                lre_results.append({'lpips': lpips_lre, 'ssim': ssim_lre})
                
                lpips_base = compute_lpips(out_base[j], images[j])
                ssim_base = compute_ssim(out_base[j], images[j])
                baseline_results.append({'lpips': lpips_base, 'ssim': ssim_base})
            
            lre_all_outputs.append(out_lre)
            baseline_all_outputs.append(out_base)
            ref_all_images.append(images)
            
            if i == 0:
                save_comparison_figure(
                    images, out_lre, garment_masks,
                    "LRE-CDT vs Input Comparison",
                    "experiment1_lre_cdt_output.pdf",
                    save_dir=config["evaluation"]["save_dir"]
                )
                save_comparison_figure(
                    images, out_base, garment_masks,
                    "Baseline vs Input Comparison",
                    "experiment1_baseline_output.pdf",
                    save_dir=config["evaluation"]["save_dir"]
                )
    
    try:
        lre_all = torch.cat(lre_all_outputs, dim=0)
        base_all = torch.cat(baseline_all_outputs, dim=0)
        ref_all = torch.cat(ref_all_images, dim=0)
        
        fid_lre = compute_dummy_fid(lre_all, ref_all)
        fid_base = compute_dummy_fid(base_all, ref_all)
        
        avg_lpips_lre = np.mean([r['lpips'] for r in lre_results])
        avg_ssim_lre = np.mean([r['ssim'] for r in lre_results])
        avg_lpips_base = np.mean([r['lpips'] for r in baseline_results])
        avg_ssim_base = np.mean([r['ssim'] for r in baseline_results])
        
        print("LRE-CDT Metrics -> FID: {:.4f}, LPIPS: {:.4f}, SSIM: {:.4f}".format(
            fid_lre, avg_lpips_lre, avg_ssim_lre)
        )
        print("Baseline Metrics -> FID: {:.4f}, LPIPS: {:.4f}, SSIM: {:.4f}".format(
            fid_base, avg_lpips_base, avg_ssim_base)
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")


def experiment2(device, config):
    """
    Ablation Study on the Localized Residual Module.
    
    Args:
        device: Device to run experiment on
        config: Configuration dictionary
    """
    print("\n" + "="*80)
    print("Running Experiment 2: Ablation Study on the Localized Residual Module")
    print("="*80)
    
    abl_dataloader = get_dataloader(config, subset='challenging')
    
    model_full = LRE_CDT_Full().to(device)
    model_no_res = LRE_CDT_NoResidue().to(device)
    model_global_res = LRE_CDT_GlobalResidue().to(device)
    
    model_full = train_model(model_full, abl_dataloader, device, config)
    model_no_res = train_model(model_no_res, abl_dataloader, device, config)
    model_global_res = train_model(model_global_res, abl_dataloader, device, config)
    
    results = {"full": [], "no_resid": [], "global_resid": []}
    
    model_full.eval()
    model_no_res.eval()
    model_global_res.eval()
    
    with torch.no_grad():
        for i, (images, garment_masks) in enumerate(abl_dataloader):
            if i > 0:
                break
                
            images = images.to(device)
            garment_masks = garment_masks.to(device)
            
            out_full = model_full.generate(images, control_signal=garment_masks)
            out_no = model_no_res.generate(images, control_signal=garment_masks)
            out_global = model_global_res.generate(images, control_signal=garment_masks)
            
            outputs_dict = {
                "Full LRE-CDT": out_full,
                "No Residual": out_no,
                "Global Residual": out_global
            }
            
            for variant, output in zip(["full", "no_resid", "global_resid"],
                                      [out_full, out_no, out_global]):
                batch_ssim, batch_lpips = [], []
                for j in range(images.size(0)):
                    ssim_val, lpips_val = compute_region_metric(output[j], images[j], garment_masks[j])
                    batch_ssim.append(ssim_val)
                    batch_lpips.append(lpips_val)
                results[variant].append({
                    "ssim": np.mean(batch_ssim),
                    "lpips": np.mean(batch_lpips)
                })
            
            save_ablation_figure(
                images, outputs_dict, garment_masks,
                "Ablation Study Comparison",
                "experiment2_ablation_comparison.pdf",
                save_dir=config["evaluation"]["save_dir"]
            )
    
    print("Ablation Study Region-specific Metrics:")
    for variant in results:
        variant_ssim = np.mean([r["ssim"] for r in results[variant]])
        variant_lpips = np.mean([r["lpips"] for r in results[variant]])
        print("Variant {}: SSIM = {:.4f}, LPIPS = {:.4f}".format(variant, variant_ssim, variant_lpips))


def experiment3(device, config):
    """
    Efficiency Evaluation via Diffusion Step Reduction.
    
    Args:
        device: Device to run experiment on
        config: Configuration dictionary
    """
    print("\n" + "="*80)
    print("Running Experiment 3: Efficiency Evaluation via Diffusion Step Reduction")
    print("="*80)
    
    step_counts = config["training"]["num_inference_steps"]
    efficiency_results = {"lre_cdt": {}, "baseline": {}}
    
    lre_cdt_model = LRE_CDT().to(device)
    baseline_model = CAT_DM().to(device)
    
    eff_dataloader = get_dataloader(config, subset='efficiency')
    lre_cdt_model = train_model(lre_cdt_model, eff_dataloader, device, config)
    baseline_model = train_model(baseline_model, eff_dataloader, device, config)
    
    lre_cdt_model.eval()
    baseline_model.eval()
    
    def generate_with_steps(model, images, garment_masks, num_steps):
        return model.generate(images, control_signal=garment_masks, num_inference_steps=num_steps)
    
    time_results = {"LRE-CDT": [], "Baseline": []}
    quality_results = {"LRE-CDT": [], "Baseline": []}
    
    with torch.no_grad():
        for step_count in step_counts:
            lre_time_total = 0.0
            baseline_time_total = 0.0
            lre_lpips_list = []
            lre_ssim_list = []
            base_lpips_list = []
            base_ssim_list = []
            
            for i, (images, garment_masks) in enumerate(eff_dataloader):
                if i > 0:
                    break
                    
                images = images.to(device)
                garment_masks = garment_masks.to(device)
                
                start_time = time.time()
                out_lre = generate_with_steps(lre_cdt_model, images, garment_masks, num_steps=step_count)
                lre_time_total += time.time() - start_time
                
                start_time = time.time()
                out_base = generate_with_steps(baseline_model, images, garment_masks, num_steps=step_count)
                baseline_time_total += time.time() - start_time
                
                for j in range(images.size(0)):
                    lpips_lre = compute_lpips(out_lre[j], images[j])
                    ssim_lre = compute_ssim(out_lre[j], images[j])
                    lre_lpips_list.append(lpips_lre)
                    lre_ssim_list.append(ssim_lre)
                    
                    lpips_base = compute_lpips(out_base[j], images[j])
                    ssim_base = compute_ssim(out_base[j], images[j])
                    base_lpips_list.append(lpips_base)
                    base_ssim_list.append(ssim_base)
            
            efficiency_results["lre_cdt"][step_count] = {
                "avg_time": lre_time_total / (i + 1),
                "lpips": np.mean(lre_lpips_list),
                "ssim": np.mean(lre_ssim_list)
            }
            efficiency_results["baseline"][step_count] = {
                "avg_time": baseline_time_total / (i + 1),
                "lpips": np.mean(base_lpips_list),
                "ssim": np.mean(base_ssim_list)
            }
            
            time_results["LRE-CDT"].append(efficiency_results["lre_cdt"][step_count]["avg_time"])
            time_results["Baseline"].append(efficiency_results["baseline"][step_count]["avg_time"])
            quality_results["LRE-CDT"].append(efficiency_results["lre_cdt"][step_count]["lpips"])
            quality_results["Baseline"].append(efficiency_results["baseline"][step_count]["lpips"])
    
    print("Efficiency Evaluation Results:")
    for method in efficiency_results:
        print(f"Method: {method}")
        for steps in efficiency_results[method]:
            res = efficiency_results[method][steps]
            print(f"  Steps: {steps} - Avg Time: {res['avg_time']:.4f}s, LPIPS: {res['lpips']:.4f}, SSIM: {res['ssim']:.4f}")
    
    save_efficiency_plots(
        step_counts, time_results, quality_results,
        "experiment3_efficiency_tradeoff.pdf",
        save_dir=config["evaluation"]["save_dir"]
    )


def main():
    """
    Main function to run all experiments.
    """
    print("="*80)
    print("LRE-CDT: Localized Residual Enhanced Controllable Diffusion Try-on")
    print("="*80)
    
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./models/", exist_ok=True)
    
    config_path = "./config/lre_cdt_config.json"
    config = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    experiment1(device, config)
    experiment2(device, config)
    experiment3(device, config)
    
    print("\n" + "="*80)
    print("All experiments completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
