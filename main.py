"""Alignment Across Contrastive Models Via A Modality Invariant Orthogonal Map."""
import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from models import get_model, compute_standard_procrustes, ProcrustesAligner
from datasets import get_dataset
from utils import (
    set_seed, normalize_model_name, str2bool,
    mean_pairwise_cosine, class_top1_retrieval, image_to_class_top1,
    get_class_text_embeddings, compute_image_embeddings, compute_text_embeddings,
    aggregate_results
)


def load_or_compute(path, fn, device='cpu'):
    """Load from cache or compute and save."""
    if os.path.exists(path):
        return torch.load(path, map_location=device)
    result = fn()
    torch.save(result, path)
    return result


def run_single_seed(args, seed):
    """Run experiment for a single seed."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*50}\nSeed: {seed}\n{'='*50}")

    # Setup paths
    exp_id = f"A={args.clip_model_1}-{args.pretrained_1}__B={args.clip_model_2}-{args.pretrained_2}".replace("/", "_")
    cache_dir = f"../embeddings/{args.dataset}/{exp_id}"
    os.makedirs(cache_dir, exist_ok=True)
    m1 = f"{normalize_model_name(args.clip_model_1)}_{args.pretrained_1}".replace("/", "_")
    m2 = f"{normalize_model_name(args.clip_model_2)}_{args.pretrained_2}".replace("/", "_")
    cache_prefix = os.path.join(cache_dir, f"{m1}_{m2}")

    # Data loaders
    train_ds = get_dataset(args.dataset, args.data_root, "train")
    test_ds = get_dataset(args.dataset, args.data_root, "test")
    train_loader = DataLoader(train_ds, args.batch_size, num_workers=args.num_workers, collate_fn=list)
    test_loader = DataLoader(test_ds, args.batch_size, num_workers=args.num_workers, collate_fn=list)

    # Models
    clip1 = get_model(args.clip_model_1, args.pretrained_1, device)
    clip2 = get_model(args.clip_model_2, args.pretrained_2, device)

    # Load/compute embeddings
    to_dict = lambda keys, tensors: dict(zip(keys, [t.cpu() for t in tensors]))
    
    tr_img = load_or_compute(f"{cache_prefix}_tr_img.pt", 
        lambda: to_dict(['i1', 'i2', 'l'], compute_image_embeddings(train_loader, clip1, clip2, device)), device)
    te_img = load_or_compute(f"{cache_prefix}_te_img.pt",
        lambda: to_dict(['i1', 'i2', 'l'], compute_image_embeddings(test_loader, clip1, clip2, device)), device)
    tr_txt = load_or_compute(f"{cache_prefix}_tr_txt.pt",
        lambda: to_dict(['t1', 't2'], compute_text_embeddings(train_loader, clip1, clip2, device)), device)

    # Test embeddings to device
    i1_te = te_img['i1'].to(device)
    i2_te = te_img['i2'].to(device)
    labels = te_img['l'].to(device)

    # Means for alignment
    i_mu1 = tr_img['i1'].mean(0, keepdim=True).to(device)
    i_mu2 = tr_img['i2'].mean(0, keepdim=True).to(device)
    t_mu1 = tr_txt['t1'].mean(0, keepdim=True).to(device)
    t_mu2 = tr_txt['t2'].mean(0, keepdim=True).to(device)
    zero = torch.zeros_like(i_mu1)

    # Procrustes aligner
    R, _, _ = compute_standard_procrustes(tr_img['i1'], tr_img['i2'], device)
    aligner = ProcrustesAligner(R).to(device)

    # Align test images
    i1_aff = aligner(i1_te, i_mu1, i_mu2) # with mean adjustment for images
    i1_std = aligner(i1_te, zero, zero) # without mean adjustment for images

    # Class prototypes
    classes = train_ds.class_names
    proto_lbl = torch.arange(len(classes), device=device)
    p1 = get_class_text_embeddings(classes, clip1, device, args.dataset, args.single_prompt)
    p2 = get_class_text_embeddings(classes, clip2, device, args.dataset, args.single_prompt)
    p1_aff = aligner(p1, t_mu1, t_mu2) # with mean adjustment for text
    p1_std = aligner(p1, zero, zero) # without mean adjustment for text

    # Compute metrics
    def metrics(i1_aligned, p1_aligned):
        return {
            "ImageImage": {
                "Baseline": class_top1_retrieval(i1_te, i2_te, labels, labels),
                "Procrustes": class_top1_retrieval(i1_aligned, i2_te, labels, labels)
            },
            "TextText": {
                "Baseline": class_top1_retrieval(p1, p2, proto_lbl, proto_lbl),
                "Procrustes": class_top1_retrieval(p1_aligned, p2, proto_lbl, proto_lbl)
            },
            "ImageText": {
                "A_to_A": image_to_class_top1(i1_te, p1, labels),
                "B_to_B": image_to_class_top1(i2_te, p2, labels),
                "Aligned_imgA_to_textB": image_to_class_top1(i1_aligned, p2, labels),
                "Aligned_imgA_to_aligned_textA": image_to_class_top1(i1_aligned, p1_aligned, labels),
                "ImgB_to_aligned_textA": image_to_class_top1(i2_te, p1_aligned, labels)
            },
            "Cosine": {
                "Image_before": mean_pairwise_cosine(i1_te, i2_te),
                "Image_after": mean_pairwise_cosine(i1_aligned, i2_te),
                "Text_before": mean_pairwise_cosine(p1, p2),
                "Text_after": mean_pairwise_cosine(p1_aligned, p2)
            },
        }

    return metrics(i1_aff, p1_aff), metrics(i1_std, p1_std)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="oxford", choices=["oxford", "cifar100", "caltech101", "dtd", "stl10"])
    p.add_argument("--data_root", default="../data")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--single_prompt", type=str2bool, default=False)
    p.add_argument("--clip_model_1", default="ViT-B-32")
    p.add_argument("--pretrained_1", default="openai")
    p.add_argument("--clip_model_2", default="ViT-B-32")
    p.add_argument("--pretrained_2", default="laion400m_e31")
    args = p.parse_args()

    # Output paths
    exp_id = f"A={args.clip_model_1}-{args.pretrained_1}__B={args.clip_model_2}-{args.pretrained_2}".replace("/", "_")
    base = f"./results/{args.dataset}/{exp_id}"
    paths = {"rotation": os.path.join(base, "rotation", f"{args.dataset}_results.json"),
             "rotation_without_centering": os.path.join(base, "rotation_without_centering", f"{args.dataset}_results.json")}

    if all(os.path.exists(path) for path in paths.values()):
        print(f"Results exist: {paths['rotation']}")
        return

    # Run experiments
    seeds = [int(s) for s in args.seeds.split(",")]
    results = {"rotation": {}, "rotation_without_centering": {}}
    
    for seed in seeds:
        aff, std = run_single_seed(args, seed)
        results["rotation"][f"seed_{seed}"] = aff
        results["rotation_without_centering"][f"seed_{seed}"] = std
        print(f"Seed {seed}: {json.dumps(aff, indent=2)}")

    # Aggregate and save
    for key in ["rotation", "rotation_without_centering"]:
        data = results[key]
        mean, std = aggregate_results([data[f"seed_{s}"] for s in seeds])
        data["mean"], data["std"] = mean, std
        os.makedirs(os.path.dirname(paths[key]), exist_ok=True)
        with open(paths[key], "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {paths[key]}")

    print(f"\n=== MEAN (rotation) ===\n{json.dumps(results['rotation']['mean'], indent=2)}")


if __name__ == "__main__":
    main()
