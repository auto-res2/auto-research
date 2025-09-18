import os
import requests
from PIL import Image
import io
import torch
from transformers import AutoProcessor, BlipForQuestionAnswering


def download_image_if_needed(image_path="./data/image.jpg"):
    """Download a sample image with a dog if it doesn't exist."""
    if os.path.exists(image_path):
        print(f"Image already exists at {image_path}")
        return True
    
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    
    url = 'https://images.unsplash.com/photo-1552053831-71594a27632d?w=500'
    
    try:
        print(f"Downloading image from {url}...")
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            img.save(image_path)
            print(f'Image downloaded successfully to {image_path}')
            print(f'Image size: {img.size}')
            return True
        else:
            print(f'Failed to download image. Status code: {response.status_code}')
            return False
    except Exception as e:
        print(f'Error downloading image: {e}')
        return False


def load_vqa_model():
    """Load the BLIP VQA model and processor."""
    try:
        print("Loading BLIP VQA model...")
        model_name = "Salesforce/blip-vqa-base"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model = BlipForQuestionAnswering.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        
        model.to(device)
        model.eval()
        
        print("Model loaded successfully!")
        return model, processor, device
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


def analyze_image_for_cat_behind_dog(image_path="./data/image.jpg", question="Is there a cat behind the dog?"):
    """Analyze the image to determine if there's a cat behind the dog."""
    
    if not download_image_if_needed(image_path):
        return None
    
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Loaded image: {image_path}")
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    model, processor, device = load_vqa_model()
    if model is None:
        return None
    
    try:
        print(f"\nQuestion: {question}")
        print("Processing image and question...")
        
        inputs = processor(images=image, text=question, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=50, num_beams=5)
        
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Model's answer: {answer}")
        return answer
        
    except Exception as e:
        print(f"Error during VQA inference: {e}")
        return None


def main():
    """Main function to run the VQA analysis."""
    print("=== Visual Question Answering Analysis ===")
    print("Task: Determine if there's a cat behind the dog in the image")
    print()
    
    result = analyze_image_for_cat_behind_dog()
    
    if result is not None:
        print(f"\n=== RESULT ===")
        print(f"Question: Is there a cat behind the dog?")
        print(f"Answer: {result}")
        print("\nAnalysis completed successfully!")
    else:
        print("\nAnalysis failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
