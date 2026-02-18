"""
Utility functions, metrics, and text prompt templates.
"""

import argparse
import random
import numpy as np
import torch
from tqdm import tqdm


# ======================================================
# General Utilities
# ======================================================

def set_seed(s=42):
    """Set random seed for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def normalize_model_name(name):
    """Normalize model names for consistent cache file naming."""
    n = name.lower()
    if "flava" in n:
        return "flava"
    if "siglip" in n:
        return "siglip"
    return name


def str2bool(v):
    """Parse boolean from string for argparse."""
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# ======================================================
# Tensor Operations
# ======================================================

def l2_normalize(x, eps=1e-8):
    """L2-normalize a torch tensor along the last dim."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


# ======================================================
# Metrics
# ======================================================

def mean_pairwise_cosine(a, b):
    """Mean cosine similarity over corresponding pairs."""
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return (a_n * b_n).sum(dim=-1).mean().item()


@torch.no_grad()
def class_top1_retrieval(query, candidates, labels_query, labels_ref):
    """
    Compute top-1 retrieval accuracy.
    
    Args:
        query: [Nq, d] query embeddings
        candidates: [Nr, d] candidate embeddings
        labels_query: [Nq] query labels
        labels_ref: [Nr] candidate labels
    
    Returns:
        Top-1 accuracy (float)
    """
    query = l2_normalize(query)
    candidates = l2_normalize(candidates)

    sim = query @ candidates.T
    ranks = sim.argsort(dim=-1, descending=True)

    labels_query = labels_query.to(ranks.device)
    labels_ref = labels_ref.to(ranks.device)

    top1_preds = labels_ref[ranks[:, 0]]
    return (top1_preds == labels_query).float().mean().item()


@torch.no_grad()
def image_to_class_top1(imgs, class_embs, true_labels):
    """
    Zero-shot image classification accuracy.
    
    Args:
        imgs: [N, d] image embeddings
        class_embs: [C, d] class text embeddings
        true_labels: [N] ground truth labels
    
    Returns:
        Top-1 accuracy (float)
    """
    imgs = l2_normalize(imgs)
    class_embs = l2_normalize(class_embs)
    sim = imgs @ class_embs.T
    preds = sim.argmax(dim=1)
    true = true_labels.to(sim.device)
    return (preds == true).float().mean().item()


# ======================================================
# Text Prompt Templates
# ======================================================

PROMPTS = {
    "oxford": ["a photo of a {}, a type of pet."],
    
    "cifar100": [
        'a photo of a {}.', 'a blurry photo of a {}.', 'a black and white photo of a {}.',
        'a low contrast photo of a {}.', 'a high contrast photo of a {}.', 'a bad photo of a {}.',
        'a good photo of a {}.', 'a photo of a small {}.', 'a photo of a big {}.',
        'a photo of the {}.', 'a blurry photo of the {}.', 'a black and white photo of the {}.',
        'a low contrast photo of the {}.', 'a high contrast photo of the {}.', 'a bad photo of the {}.',
        'a good photo of the {}.', 'a photo of the small {}.', 'a photo of the big {}.',
    ],
    
    "caltech101": [
        'a photo of a {}.', 'a painting of a {}.', 'a plastic {}.', 'a sculpture of a {}.',
        'a sketch of a {}.', 'a tattoo of a {}.', 'a toy {}.', 'a rendition of a {}.',
        'a embroidered {}.', 'a cartoon {}.', 'a {} in a video game.', 'a plushie {}.',
        'a origami {}.', 'art of a {}.', 'graffiti of a {}.', 'a drawing of a {}.',
        'a doodle of a {}.', 'a photo of the {}.', 'a painting of the {}.', 'the plastic {}.',
        'a sculpture of the {}.', 'a sketch of the {}.', 'a tattoo of the {}.', 'the toy {}.',
        'a rendition of the {}.', 'the embroidered {}.', 'the cartoon {}.', 'the {} in a video game.',
        'the plushie {}.', 'the origami {}.', 'art of the {}.', 'graffiti of the {}.',
        'a drawing of the {}.', 'a doodle of the {}.',
    ],
    
    "dtd": [
        'a photo of a {} texture.', 'a photo of a {} pattern.', 'a photo of a {} thing.',
        'a photo of a {} object.', 'a photo of the {} texture.', 'a photo of the {} pattern.',
        'a photo of the {} thing.', 'a photo of the {} object.',
    ],
    
    "stl10": ['a photo of a {}.', 'a photo of the {}.'],
}


@torch.no_grad()
def get_class_text_embeddings(class_names, clip_model, device, dataset_name, single_prompt=True):
    """
    Compute class text embeddings.
    
    Args:
        class_names: List of class names
        clip_model: Model wrapper with encode_text method
        device: Torch device
        dataset_name: Name of dataset (for prompt selection)
        single_prompt: If True, use first prompt only; else average over all prompts
    
    Returns:
        [C, d] normalized class embeddings
    """
    prompts = PROMPTS.get(dataset_name, ['a photo of a {}.'])
    
    if single_prompt:
        texts = [prompts[0].format(name.replace("_", " ")) for name in class_names]
        embs = clip_model.encode_text(texts)
        return l2_normalize(embs).to(device)
    
    # Average over all prompts
    all_embs = []
    for prompt in prompts:
        texts = [prompt.format(name.replace("_", " ")) for name in class_names]
        e = l2_normalize(clip_model.encode_text(texts))
        all_embs.append(e)
    
    stacked = torch.stack(all_embs, dim=0)
    avg = l2_normalize(stacked.mean(dim=0))
    return avg.to(device)


# ======================================================
# Embedding Computation
# ======================================================

def compute_image_embeddings(loader, clip1, clip2, device):
    """
    Compute image embeddings for two models.
    
    Returns:
        img1, img2: [N, d] embeddings from each model
        labels: [N] class labels
    """
    all_img1, all_img2 = [], []
    all_labels = []

    for batch in tqdm(loader, desc="Computing image embeddings"):
        imgs = [item["image"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        imgs1 = torch.stack([clip1.preprocess(img).to(device) for img in imgs])
        imgs2 = torch.stack([clip2.preprocess(img).to(device) for img in imgs])

        with torch.no_grad():
            z1 = l2_normalize(clip1.encode_image(imgs1)).cpu()
            z2 = l2_normalize(clip2.encode_image(imgs2)).cpu()

        all_img1.append(z1)
        all_img2.append(z2)
        all_labels.append(labels)

    return torch.cat(all_img1), torch.cat(all_img2), torch.cat(all_labels)


def compute_text_embeddings(loader, clip1, clip2, device):
    """
    Compute per-instance text embeddings for two models.
    
    Returns:
        txt1, txt2: [N, d] embeddings from each model
    """
    all_txt1, all_txt2 = [], []

    for batch in tqdm(loader, desc="Computing text embeddings"):
        texts = [item["text"] for item in batch]
        with torch.no_grad():
            z1 = l2_normalize(clip1.encode_text(texts)).cpu()
            z2 = l2_normalize(clip2.encode_text(texts)).cpu()
        all_txt1.append(z1)
        all_txt2.append(z2)

    return torch.cat(all_txt1), torch.cat(all_txt2)

# ======================================================
# Result Aggregation
# ======================================================

def aggregate_results(dicts_list):
    """Recursively compute mean and std across result dictionaries."""
    if not dicts_list:
        return {}, {}
    
    mean_dict, std_dict = {}, {}
    for key in dicts_list[0].keys():
        if isinstance(dicts_list[0][key], dict):
            mean_dict[key], std_dict[key] = aggregate_results([d[key] for d in dicts_list])
        else:
            values = np.array([d[key] for d in dicts_list])
            mean_dict[key] = float(np.mean(values))
            std_dict[key] = float(np.std(values))
    return mean_dict, std_dict
