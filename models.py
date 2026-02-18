"""Model wrappers and Procrustes alignment."""

import os
import torch
import torch.nn as nn
import open_clip


# ======================================================
# Model Wrappers
# ======================================================

class ClipWrapper:
    """OpenCLIP model wrapper."""
    
    def __init__(self, model_name, pretrained, device):
        cache = os.path.join(os.path.expanduser("~"), ".open_clip")
        os.makedirs(cache, exist_ok=True)
        os.environ["TORCH_HOME"] = cache
        os.environ["OPENCLIP_CACHE_DIR"] = cache

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cache
        )
        self.model = self.model.to(device).eval()
        self.device = device
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode_image(self, images):
        return self.model.encode_image(images)

    @torch.no_grad()
    def encode_text(self, texts):
        return self.model.encode_text(self.tokenizer(texts).to(self.device))


class FlavaWrapper:
    """FLAVA model wrapper."""
    
    def __init__(self, device):
        from transformers import FlavaModel, AutoProcessor
        self.model = FlavaModel.from_pretrained("facebook/flava-full").to(device).eval()
        self.proc = AutoProcessor.from_pretrained("facebook/flava-full", use_fast=True)
        self.device = device

    def preprocess(self, img):
        return self.proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

    @torch.no_grad()
    def encode_image(self, images):
        z = self.model.get_image_features(pixel_values=images.to(self.device))[:, 0, :]
        return z / (z.norm(dim=-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def encode_text(self, texts):
        tx = self.proc(text=texts, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
        tx = {k: v.to(self.device) for k, v in tx.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
        z = self.model.get_text_features(**tx)
        if z.dim() == 3:
            z = z[:, 0, :]
        return z / (z.norm(dim=-1, keepdim=True) + 1e-8)


class SigLIPWrapper:
    """SigLIP model wrapper."""
    
    def __init__(self, model_name, device):
        from transformers import AutoProcessor, SiglipModel
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def preprocess(self, img):
        return self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)

    @torch.no_grad()
    def encode_image(self, images):
        z = self.model.get_image_features(pixel_values=images.to(self.device))
        return z / (z.norm(dim=-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding="max_length", truncation=True).to(self.device)
        z = self.model.get_text_features(**inputs)
        return z / (z.norm(dim=-1, keepdim=True) + 1e-8)


def get_model(model_name, pretrained, device):
    """Factory function to create model wrapper."""
    if "flava" in model_name.lower():
        return FlavaWrapper(device)
    elif "siglip" in model_name.lower():
        return SigLIPWrapper(pretrained, device)
    return ClipWrapper(model_name, pretrained, device)


# ======================================================
# Procrustes Alignment
# ======================================================

def compute_standard_procrustes(x_src, x_tgt, device):
    """Compute Procrustes rotation matrix R and means."""
    X, Y = x_src.to(device), x_tgt.to(device)
    mu_x, mu_y = X.mean(0, keepdim=True), Y.mean(0, keepdim=True)
    M = (X - mu_x).T @ (Y - mu_y)
    U, _, Vh = torch.linalg.svd(M)
    R = U @ Vh
    return R, mu_x, mu_y


class ProcrustesAligner(nn.Module):
    """
    Procrustes aligner: y = (x - mu_x) @ R + mu_y
    
    Args:
        R: Rotation matrix [d, d]
        mu_x: Source mean [1, d] (use zeros for no centering)
        mu_y: Target mean [1, d] (use zeros for no centering)
    """
    
    def __init__(self, R):
        super().__init__()
        self.register_buffer("R", R)

    def forward(self, x, mu_x, mu_y):
        return (x - mu_x) @ self.R + mu_y
