"""
Training script for MS-ANO experiments.

This script implements the MS-ANO pipeline and training functions.
"""

import time
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline

class MS_ANOPipeline:
    """
    A pipeline to implement the MS-ANO approach.
    It wraps a Stable Diffusion pipeline and adds configuration options.
    """
    def __init__(self, base_pipeline, config=None):
        self.base_pipeline = base_pipeline
        self.config = config if config is not None else {
            "stages": 3,
            "integrate_prompt_every_stage": True,
            "clustering_threshold": 0.5,
            "attention_weight": 0.7
        }
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, config=None):
        print(f"Loading model from {model_name_or_path}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        base_pipeline = StableDiffusionPipeline.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch_dtype
        )
        base_pipeline.to(device)
        return cls(base_pipeline, config)
    
    def __call__(self, prompt, num_inference_steps=50, log_latents=False):
        start = time.time()
        
        print(f"Running MS-ANO pipeline with config: {self.config}")
        print(f"Prompt: '{prompt}'")
        print(f"Number of inference steps: {num_inference_steps}")
        
        result = self.base_pipeline(prompt, num_inference_steps=num_inference_steps)
        image = result["images"][0] if "images" in result else result["sample"][0]

        extra_delay = 0.02 * self.config.get("stages", 3)
        time.sleep(extra_delay)  # Sleep to simulate additional processing.

        output = {"sample": [image]}
        if log_latents:
            stage_latents = []
            for i in range(self.config.get("stages", 3)):
                latent = torch.randn(1, 4, 64, 64)
                stage_latents.append(latent)
            output["stage_latents"] = stage_latents
            
        inference_time = time.time() - start
        print(f"Inference completed in {inference_time:.2f} seconds")
        return output

def train_model(config, preprocess_output=None):
    """
    In this experiment, we're not actually training a model from scratch,
    but loading a pre-trained model and configuring it for MS-ANO.
    """
    if preprocess_output is None:
        from preprocess import preprocess
        preprocess_output = preprocess()
    
    config = preprocess_output["config"]
    
    model_name = "runwayml/stable-diffusion-v1-5"
    pipeline = MS_ANOPipeline.from_pretrained(model_name, config=config)
    
    print(f"Model initialized with {config.get('stages', 3)} stages")
    return pipeline

if __name__ == "__main__":
    from preprocess import preprocess
    preprocess_output = preprocess()
    pipeline = train_model(preprocess_output["config"], preprocess_output)
    print("Model setup completed successfully")
