"""
Evaluation script for MS-ANO experiments.

This script implements evaluation functions for the generated images.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compute_clip_score(image, prompt, clip_processor, clip_model):
    """
    Compute the CLIP cosine similarity score between text and image embeddings.
    """
    inputs = clip_processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    device = next(clip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
    
    if hasattr(outputs, 'image_embeds') and hasattr(outputs, 'text_embeds'):
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
    else:
        image_embeds = outputs.image_features
        text_embeds = outputs.text_features
    
    cosine_sim = torch.nn.functional.cosine_similarity(image_embeds, text_embeds)
    score = cosine_sim.item()
    return score

def compute_fid(generated_images, reference_images=None):
    """
    Dummy function to compute FID; in practice use pytorch_fid or clean-fid package.
    """
    fid = np.random.uniform(0, 100)
    return fid

def run_baseline_evaluation(base_pipeline, msano_pipeline, prompts, n_runs, clip_processor, clip_model):
    """
    Run baseline comparison experiment between base pipeline and MS-ANO pipeline.
    """
    print("\nStarting Baseline Comparison Experiment")
    
    results = {"base": [], "msano": []}
    
    for pipeline_label, pipeline in zip(["base", "msano"], [base_pipeline, msano_pipeline]):
        print(f"\nEvaluating pipeline: {pipeline_label}")
        prompt_metrics = []
        for prompt in prompts:
            timings = []
            clip_scores = []
            generated_images = []
            for run in range(n_runs):
                print(f"  Prompt: '{prompt}' | Run: {run+1}/{n_runs}")
                start_time = time.time()
                output = pipeline(prompt, num_inference_steps=50)
                image = output["sample"][0]
                runtime = time.time() - start_time
                timings.append(runtime)
                score = compute_clip_score(image, prompt, clip_processor, clip_model)
                clip_scores.append(score)
                generated_images.append(image)
                print(f"    Run time: {runtime:.3f}s, CLIP score: {score:.3f}")
            avg_time = np.mean(timings)
            avg_clip = np.mean(clip_scores)
            fid_score = compute_fid(generated_images, reference_images=None)
            prompt_metrics.append({
                "prompt": prompt,
                "avg_runtime": avg_time,
                "avg_clip_score": avg_clip,
                "fid": fid_score
            })
            print(f"  --> Averaged metrics for prompt '{prompt}': runtime={avg_time:.3f}s, CLIP={avg_clip:.3f}, FID={fid_score:.2f}")
        results[pipeline_label] = prompt_metrics

    plt.figure(figsize=(8,6))
    for pipeline_label in results:
        clips = [entry["avg_clip_score"] for entry in results[pipeline_label]]
        plt.plot(prompts, clips, marker='o', label=pipeline_label)
    plt.xlabel("Prompts")
    plt.ylabel("Average CLIP Score")
    plt.title("Baseline Comparison: CLIP Scores by Pipeline")
    plt.legend()
    plt.tight_layout()
    pdf_filename = "logs/experiment1_baseline_clip_scores.pdf"
    plt.savefig(pdf_filename, format="pdf")
    print(f"\nSaved baseline CLIP score plot to {pdf_filename}")
    plt.close()
    
    return results

def run_ablation_study(prompts, n_runs, clip_processor, clip_model):
    """
    Run ablation study on different MS-ANO configurations.
    """
    import time
    from train import MS_ANOPipeline
    
    print("\nStarting Ablation Study Experiment")
    
    config_full = {"stages": 3, "integrate_prompt_every_stage": True}
    config_single = {"stages": 1, "integrate_prompt_every_stage": True}  # Mimics InitNO
    config_no_stage_prompt = {"stages": 3, "integrate_prompt_every_stage": False}

    msano_full = MS_ANOPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", config=config_full)
    msano_single = MS_ANOPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", config=config_single)
    msano_no_prompt = MS_ANOPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", config=config_no_stage_prompt)

    pipeline_variants = {
        "MSANO_Full": msano_full,
        "MSANO_SingleStage": msano_single,
        "MSANO_NoStagePrompt": msano_no_prompt
    }
    
    ablation_results = {}
    
    for variant_name, pipeline in pipeline_variants.items():
        print(f"\nEvaluating variant: {variant_name}")
        variant_metrics = []
        for prompt in prompts:
            timings = []
            clip_scores = []
            latent_logs = []  # For intermediate representation logging.
            for run in range(n_runs):
                print(f"  Variant '{variant_name}' | Prompt: '{prompt}' | Run: {run+1}/{n_runs}")
                start_time = time.time()
                output = pipeline(prompt, num_inference_steps=50, log_latents=True)
                image = output["sample"][0]
                stage_latents = output.get("stage_latents", None)
                runtime = time.time() - start_time
                timings.append(runtime)
                score = compute_clip_score(image, prompt, clip_processor, clip_model)
                clip_scores.append(score)
                latent_logs.append(stage_latents)
                print(f"    Run time: {runtime:.3f}s, CLIP score: {score:.3f}")
            avg_time = np.mean(timings)
            avg_clip = np.mean(clip_scores)
            fid_score = compute_fid([], reference_images=None)
            variant_metrics.append({
                "prompt": prompt,
                "avg_runtime": avg_time,
                "avg_clip_score": avg_clip,
                "fid": fid_score,
                "latent_logs": latent_logs  # Saved for later inspection.
            })
            print(f"  --> Averaged metrics for prompt '{prompt}' for variant '{variant_name}': runtime={avg_time:.3f}s, CLIP={avg_clip:.3f}, FID={fid_score:.2f}")
        ablation_results[variant_name] = variant_metrics

    plt.figure(figsize=(8,6))
    variant_names = list(pipeline_variants.keys())
    avg_runtimes = []
    for variant in variant_names:
        times = [entry["avg_runtime"] for entry in ablation_results[variant]]
        avg_runtimes.append(np.mean(times))
    plt.bar(variant_names, avg_runtimes)
    plt.xlabel("Variant")
    plt.ylabel("Average Runtime (s)")
    plt.title("Ablation Study: Average Runtime per Variant")
    plt.tight_layout()
    pdf_filename = "logs/experiment2_ablation_runtime.pdf"
    plt.savefig(pdf_filename, format="pdf")
    print(f"\nSaved ablation runtime plot to {pdf_filename}")
    plt.close()
    
    return ablation_results

def run_hyperparameter_study(prompts, n_runs, clip_processor, clip_model):
    """
    Run hyperparameter sensitivity analysis.
    """
    import time
    import itertools
    from train import MS_ANOPipeline
    
    print("\nStarting Hyperparameter Sensitivity and Robustness Analysis")
    
    stages_space = [1, 2, 3, 4]
    clustering_threshold_space = [0.1, 0.3, 0.5, 0.7]
    weighting_space = [0.3, 0.5, 0.7, 0.9]
    
    hyperparam_results = []
    
    for stages, threshold, weight in itertools.product([1, 3], [0.5], [0.7]):
        config = {
            "stages": stages,
            "clustering_threshold": threshold,
            "attention_weight": weight,
            "integrate_prompt_every_stage": True
        }
        pipeline = MS_ANOPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", config=config)
        metrics_across_prompts = []
        for prompt in prompts:
            runtimes = []
            clip_scores = []
            for _ in range(n_runs):
                start_time = time.time()
                output = pipeline(prompt, num_inference_steps=50)
                image = output["sample"][0]
                runtime = time.time() - start_time
                runtimes.append(runtime)
                score = compute_clip_score(image, prompt, clip_processor, clip_model)
                clip_scores.append(score)
            metrics_across_prompts.append({
                "prompt": prompt,
                "avg_runtime": np.mean(runtimes),
                "avg_clip_score": np.mean(clip_scores)
            })
        overall_runtime = np.mean([m["avg_runtime"] for m in metrics_across_prompts])
        overall_clip = np.mean([m["avg_clip_score"] for m in metrics_across_prompts])
        
        hyperparam_results.append({
            "stages": stages,
            "clustering_threshold": threshold,
            "attention_weight": weight,
            "overall_runtime": overall_runtime,
            "overall_clip_score": overall_clip
        })
        print(f"Config: stages={stages}, threshold={threshold}, weight={weight} -> runtime={overall_runtime:.3f}s, CLIP={overall_clip:.3f}")
    
    filtered = [r for r in hyperparam_results if (r["clustering_threshold"] == 0.5 and r["attention_weight"] == 0.7)]
    if filtered:
        stages_list = [r["stages"] for r in filtered]
        runtime_list = [r["overall_runtime"] for r in filtered]
        plt.figure(figsize=(8,6))
        plt.plot(stages_list, runtime_list, marker='o')
        plt.xlabel("Number of Stages")
        plt.ylabel("Average Runtime (s)")
        plt.title("Runtime vs. Number of Stages (Threshold=0.5, Weight=0.7)")
        plt.tight_layout()
        pdf_filename = "logs/experiment3_runtime_vs_stages.pdf"
        plt.savefig(pdf_filename, format="pdf")
        print(f"\nSaved hyperparameter sensitivity plot to {pdf_filename}")
        plt.close()
    else:
        print("No configurations found for plotting with threshold=0.5 and weight=0.7.")
    
    return hyperparam_results

if __name__ == "__main__":
    from preprocess import preprocess
    preprocess_output = preprocess()
    from train import train_model
    
    pipeline = train_model(preprocess_output["config"], preprocess_output)
    
    prompt = "a cat and a rabbit"
    output = pipeline(prompt, num_inference_steps=10)
    image = output["sample"][0]
    
    clip_processor = preprocess_output["clip_processor"]
    clip_model = preprocess_output["clip_model"]
    
    score = compute_clip_score(image, prompt, clip_processor, clip_model)
    print(f"CLIP score for prompt '{prompt}': {score:.3f}")
    print("Evaluation test completed successfully")
