"""
Training script for VG-DD experiments.
Implements models and training loops for all three experiments.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.vgdd.config import EXPERIMENT1_CONFIG, EXPERIMENT2_CONFIG, EXPERIMENT3_CONFIG, T4_OPTIMIZATION


class VisualModule(nn.Module):
    """Visual feature extraction module based on ResNet50."""
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
    
    def forward(self, x):
        feat = self.features(x)  # [batch, C, H', W']
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)  # [batch, C]
        return feat

class AdaptiveDecoder(nn.Module):
    """Decoder with adaptive visual prompt weighting."""
    def __init__(self, vocab_size, embed_dim, visual_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8)
        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, tgt, memory, visual_features):
        visual_prompt = self.visual_proj(visual_features).unsqueeze(0)  # [1, batch, embed_dim]
        output = self.decoder_layer(tgt, memory)
        repeated_visual = visual_prompt.expand(output.size(0), -1, -1)
        cosine_sim = F.cosine_similarity(output, repeated_visual, dim=-1).unsqueeze(-1)
        logits = self.out_proj(output)
        adaptive_logits = logits * (0.5 + 0.5 * cosine_sim)
        return adaptive_logits, cosine_sim

def train_experiment1(save_path, test_mode=True):
    """Run Experiment 1: Adaptive Visual Prompt Weighting."""
    print("Running Experiment 1: Adaptive Visual Prompt Weighting")
    
    config = EXPERIMENT1_CONFIG
    if test_mode:
        config["batch_size"] = config["test_batch_size"]
    elif torch.cuda.is_available():
        config["batch_size"] = min(config["batch_size"], T4_OPTIMIZATION["batch_size"])
        print(f"Using T4 optimized batch size: {config['batch_size']}")
    
    dummy_image = torch.rand((config["batch_size"], 3, config["image_size"], config["image_size"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_image = dummy_image.to(device)
    
    visual_module = VisualModule().to(device)
    visual_features = visual_module(dummy_image)  # [batch, C]
    print("Extracted visual features shape:", visual_features.shape)
    
    vocab_size = config["vocab_size"]
    embed_dim = config["embed_dim"]
    visual_dim = visual_features.shape[1]
    decoder = AdaptiveDecoder(vocab_size, embed_dim, visual_dim).to(device)
    
    T = 10
    batch_size = config["batch_size"]
    token_indices = torch.randint(0, vocab_size, (T, batch_size)).to(device)
    embedded_tokens = decoder.embed(token_indices)  # [T, batch, embed_dim]
    memory = embedded_tokens.clone()
    
    adaptive_logits, cosine_sim = decoder(embedded_tokens, memory, visual_features)
    print("Adaptive logits shape:", adaptive_logits.shape)
    print("Cosine similarity per token (adaptive weighting):", 
          cosine_sim.squeeze(-1).detach().cpu().numpy())
    
    cosine_vals = cosine_sim.squeeze(-1).detach().cpu().numpy().mean(axis=1)  # Average over batch
    plt.figure(figsize=(10, 6), dpi=300)
    sns.lineplot(x=range(1, T+1), y=cosine_vals, marker='o', linewidth=2)
    plt.xlabel("Decoding Step", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    plt.title("Token-Visual Cosine Similarity over Decoding Steps", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"Experiment 1 plot saved: {save_path}\n")
    
    return {
        "visual_module": visual_module,
        "decoder": decoder,
        "visual_dim": visual_dim,
        "embed_dim": embed_dim
    }


def compute_grounding_score(token_embedding, visual_feature):
    """Compute grounding score between token embedding and visual feature."""
    return F.cosine_similarity(token_embedding, visual_feature, dim=-1)

class SimpleDecoder(nn.Module):
    """Simple decoder for joint decoding experiment."""
    def __init__(self, vocab_size=10000, embed_dim=768):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.transformer_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8)
        self.output = nn.Linear(embed_dim, vocab_size)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
    
    def forward(self, input_ids, visual_feature=None):
        batch_size, seq_len = input_ids.shape
        
        token_embeds = self.embed(input_ids)  # [batch, seq_len, embed_dim]
        
        positions = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1).to(input_ids.device)
        position_embeds = self.embed(positions)  # [batch, seq_len, embed_dim]
        
        x = token_embeds + position_embeds
        x = x.transpose(0, 1)  # [seq_len, batch, embed_dim]
        
        memory = torch.zeros(seq_len, batch_size, self.embed_dim).to(input_ids.device)
        
        out = self.transformer_layer(x, memory)
        out = out.transpose(0, 1)  # [batch, seq_len, embed_dim]
        
        logits = self.output(out)  # [batch, seq_len, vocab_size]
        
        grounding_scores = None
        if visual_feature is not None:
            visual_expanded = visual_feature.unsqueeze(1).expand(-1, seq_len, -1)
            grounding_scores = F.cosine_similarity(out, visual_expanded, dim=-1)
        
        return logits, out, grounding_scores

def generate_text(model, input_ids, max_length=30, visual_feature=None):
    """Generate text using the simple decoder model."""
    device = input_ids.device
    batch_size = input_ids.size(0)
    current_ids = input_ids
    
    for _ in range(max_length - input_ids.size(1)):
        logits, _, _ = model(current_ids, visual_feature)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        current_ids = torch.cat([current_ids, next_token], dim=1)
    
    return current_ids

def iterative_decoding(model, tokenizer, input_prompt, visual_feature, 
                       num_iterations=3, threshold=0.5):
    """Run iterative decoding with feedback loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_ids = tokenizer.encode(input_prompt, return_tensor=True).to(device)
    
    print("Initial prompt:", input_prompt)
    all_grounding_scores = []
    
    generated_ids = generate_text(model, input_ids, visual_feature=visual_feature)
    
    for iteration in range(num_iterations):
        decoded_text = tokenizer.decode(generated_ids[0])
        tokens = decoded_text.split()
        print(f"Iteration {iteration+1}: Generated text: {decoded_text}")
        
        token_embeddings = model.embed(generated_ids)  # [batch, T, embed_dim]
        
        _, _, grounding_scores = model(generated_ids, visual_feature)
        all_grounding_scores.append(grounding_scores[0].detach().cpu().numpy())
        
        avg_grounding = grounding_scores.mean().item()
        print(f"Iteration {iteration+1}: Average grounding score: {avg_grounding:.4f}")
        
        if (grounding_scores < threshold).sum().item() == 0:
            print("All tokens meet the grounding threshold. Stopping iterations.")
            break
        
        reexamine_phrase = " Please reexamine the previous statement."
        revised_prompt = decoded_text + reexamine_phrase
        input_ids = tokenizer.encode(revised_prompt, return_tensor=True).to(device)
        generated_ids = generate_text(model, input_ids, visual_feature=visual_feature)
    
    final_text = tokenizer.decode(generated_ids[0])
    return final_text, all_grounding_scores[-1]

class DummyTokenizer:
    """Dummy tokenizer for experiment 2."""
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
    
    def encode(self, text, return_tensor=False):
        """Convert text to token IDs (dummy implementation)."""
        tokens = torch.randint(0, self.vocab_size, (1, min(len(text.split()), 20)))
        return tokens
    
    def decode(self, token_ids):
        """Convert token IDs back to text (dummy implementation)."""
        if token_ids.dim() > 1:
            return f"Generated text with {token_ids.size(1)} tokens"
        else:
            return f"Generated text with {token_ids.size(0)} tokens"

def train_experiment2(save_path, test_mode=True):
    """Run Experiment 2: Joint Decoding with Iterative Feedback Loop."""
    print("Running Experiment 2: Joint Decoding with Iterative Feedback Loop")
    
    config = EXPERIMENT2_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available() and not test_mode:
        if T4_OPTIMIZATION["memory_efficient_attention"]:
            print("Using memory-efficient attention for transformer layers")
    
    model = SimpleDecoder(embed_dim=config["embed_dim"]).to(device)
    tokenizer = DummyTokenizer()
    
    max_tokens = T4_OPTIMIZATION["max_tokens"] if torch.cuda.is_available() and not test_mode else 30
    
    dummy_visual_feature = torch.rand((1, config["embed_dim"])).to(device)
    
    final_output, final_groundings = iterative_decoding(
        model, tokenizer, config["test_prompt"], dummy_visual_feature,
        num_iterations=config["num_iterations"], threshold=config["threshold"]
    )
    print("Final output after iterative decoding:", final_output)
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x=list(range(1, len(final_groundings)+1)), y=final_groundings)
    plt.xlabel("Token Position", fontsize=12)
    plt.ylabel("Grounding Score (Cosine Similarity)", fontsize=12)
    plt.title("Final Grounding Scores per Token", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"Experiment 2 plot saved: {save_path}\n")
    
    return model


class ContrastiveLVLM(nn.Module):
    """Contrastive Learning for Language-Vision Models."""
    def __init__(self, language_model, visual_module, visual_module_out_dim, language_model_embed_dim):
        super().__init__()
        self.language_model = language_model
        self.visual_module = visual_module
        self.proj = nn.Linear(visual_module_out_dim, language_model_embed_dim)
    
    def forward(self, image, text_input):
        visual_feat = self.visual_module(image)          # shape: [batch, visual_module_out_dim]
        visual_emb = self.proj(visual_feat)              # shape: [batch, language_model_embed_dim]
        language_out = self.language_model(text_input)   # shape: [batch, language_model_embed_dim]
        return visual_emb, language_out

class DummyLanguageModel(nn.Module):
    """Dummy language model for experiment 3."""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
    
    def forward(self, text_input):
        batch_size = text_input.size(0)
        return torch.rand(batch_size, self.embed_dim, device=text_input.device)

def train_experiment3(visual_module, save_path, test_mode=True):
    """Run Experiment 3: Data Augmentation for Contrastive Learning."""
    print("Running Experiment 3: Data Augmentation for Contrastive Learning")
    
    config = EXPERIMENT3_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available() and not test_mode:
        if T4_OPTIMIZATION["gradient_checkpointing"]:
            print("Enabling gradient checkpointing for memory efficiency")
    
    from preprocess import DummyImageDataset
    
    batch_size = 1 if test_mode else T4_OPTIMIZATION["batch_size"] if torch.cuda.is_available() else 4
    
    dataset = DummyImageDataset(size=max(2, batch_size))
    sample = dataset[0]
    intact_img = sample["original"].unsqueeze(0).to(device)
    perturbed_img = sample["perturbed"].unsqueeze(0).to(device)
    print("Obtained intact and perturbed images.")
    
    visual_module_out_dim = config["visual_module_out_dim"]
    language_model_embed_dim = config["language_model_embed_dim"]
    
    language_model = DummyLanguageModel(language_model_embed_dim).to(device)
    
    model = ContrastiveLVLM(
        language_model, visual_module, visual_module_out_dim, language_model_embed_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CosineEmbeddingLoss()  # Encourages alignment based on cosine similarity
    
    text_input = torch.randint(0, 1000, (1, 10), device=device)  # [batch, seq_length]
    num_epochs = config["num_epochs"]
    loss_list = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        intact_feat, language_out = model(intact_img, text_input)
        perturbed_feat, _ = model(perturbed_img, text_input)
        
        target_sim = torch.tensor([1.0], device=device)
        target_diff = torch.tensor([-1.0], device=device)
        
        loss_intact = criterion(intact_feat, language_out, target_sim)
        loss_perturbed = criterion(perturbed_feat, language_out, target_diff)
        loss = loss_intact + loss_perturbed
        
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_list.append(loss_val)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss_val:.4f}")
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.lineplot(x=range(1, num_epochs+1), y=loss_list, marker='o', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Contrastive Loss", fontsize=12)
    plt.title("Training Loss for Contrastive Fine-Tuning", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()
    print(f"Experiment 3 plot saved: {save_path}\n")
    
    return model
