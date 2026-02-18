"""Few-shot alignment: learn orthogonal map from a subset of classes, evaluate on all."""
import os
import json
import argparse
import random
import torch
from torch.utils.data import DataLoader

from models import get_model, compute_standard_procrustes, ProcrustesAligner
from datasets import get_dataset
from utils import (
    set_seed, normalize_model_name, str2bool,
    mean_pairwise_cosine, class_top1_retrieval, image_to_class_top1,
    get_class_text_embeddings, compute_image_embeddings, compute_text_embeddings,
    aggregate_results,
)


def load_or_compute(path, fn, device="cpu"):
    """Load from cache or compute and save."""
    if os.path.exists(path):
        return torch.load(path, map_location=device)
    result = fn()
    torch.save(result, path)
    return result


def run_single_seed(args, seed, class_split_seed):
    """Run few-anchors experiment for a single seed."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*50}\nSeed: {seed}, class_split_seed: {class_split_seed}\n{'='*50}")

    exp_id = f"A={args.clip_model_1}-{args.pretrained_1}__B={args.clip_model_2}-{args.pretrained_2}".replace("/", "_")
    cache_dir = f"../embeddings/{args.dataset}/{exp_id}"
    os.makedirs(cache_dir, exist_ok=True)
    m1 = f"{normalize_model_name(args.clip_model_1)}_{args.pretrained_1}".replace("/", "_")
    m2 = f"{normalize_model_name(args.clip_model_2)}_{args.pretrained_2}".replace("/", "_")
    cache_prefix = os.path.join(cache_dir, f"{m1}_{m2}")

    train_ds = get_dataset(args.dataset, args.data_root, "train")
    test_ds = get_dataset(args.dataset, args.data_root, "test")
    train_loader = DataLoader(train_ds, args.batch_size, num_workers=args.num_workers, collate_fn=list)
    test_loader = DataLoader(test_ds, args.batch_size, num_workers=args.num_workers, collate_fn=list)

    num_classes = train_ds.num_classes
    classes = train_ds.class_names

    clip1 = get_model(args.clip_model_1, args.pretrained_1, device)
    clip2 = get_model(args.clip_model_2, args.pretrained_2, device)

    to_dict = lambda keys, tensors: dict(zip(keys, [t.cpu() for t in tensors]))

    tr_img = load_or_compute(
        f"{cache_prefix}_tr_img.pt",
        lambda: to_dict(["i1", "i2", "l"], compute_image_embeddings(train_loader, clip1, clip2, device)),
        device,
    )
    te_img = load_or_compute(
        f"{cache_prefix}_te_img.pt",
        lambda: to_dict(["i1", "i2", "l"], compute_image_embeddings(test_loader, clip1, clip2, device)),
        device,
    )
    tr_txt = load_or_compute(
        f"{cache_prefix}_tr_txt.pt",
        lambda: to_dict(["t1", "t2"], compute_text_embeddings(train_loader, clip1, clip2, device)),
        device,
    )

    i1_te = te_img["i1"].to(device)
    i2_te = te_img["i2"].to(device)
    labels_te = te_img["l"].to(device)

    n_seen = min(args.num_train_classes, num_classes) if args.num_train_classes is not None else num_classes
    n_unseen = num_classes - n_seen

    random.seed(class_split_seed)
    torch.manual_seed(class_split_seed)
    all_class_ids = list(range(num_classes))
    random.shuffle(all_class_ids)
    seen_class_list = sorted(all_class_ids[:n_seen])
    unseen_class_list = sorted(all_class_ids[n_seen:])
    seen_class_ids = torch.tensor(seen_class_list, device=device)
    unseen_class_ids = torch.tensor(unseen_class_list, device=device)

    mask_tr_seen = torch.isin(tr_img["l"].to(device), seen_class_ids)
    i1_tr_seen = tr_img["i1"][mask_tr_seen.cpu()].to(device)
    i2_tr_seen = tr_img["i2"][mask_tr_seen.cpu()].to(device)

    mask_te_seen = torch.isin(labels_te, seen_class_ids)
    mask_te_unseen = torch.isin(labels_te, unseen_class_ids)

    i_mu1 = i1_tr_seen.mean(0, keepdim=True)
    i_mu2 = i2_tr_seen.mean(0, keepdim=True)
    t_mu1 = tr_txt["t1"].to(device).mean(0, keepdim=True)
    t_mu2 = tr_txt["t2"].to(device).mean(0, keepdim=True)
    zero = torch.zeros_like(i_mu1)

    R, _, _ = compute_standard_procrustes(i1_tr_seen, i2_tr_seen, device)
    aligner = ProcrustesAligner(R).to(device)

    i1_aff = aligner(i1_te, i_mu1, i_mu2)
    i1_std = aligner(i1_te, zero, zero)

    proto_lbl = torch.arange(num_classes, device=device)
    p1 = get_class_text_embeddings(classes, clip1, device, args.dataset, args.single_prompt)
    p2 = get_class_text_embeddings(classes, clip2, device, args.dataset, args.single_prompt)
    p1_aff = aligner(p1, t_mu1, t_mu2)
    p1_std = aligner(p1, zero, zero)

    # Text subsets for seen/unseen
    p1_seen = p1[seen_class_ids]
    p2_seen = p2[seen_class_ids]
    if n_unseen > 0:
        p1_unseen = p1[unseen_class_ids]
        p2_unseen = p2[unseen_class_ids]

    def metrics(i1_aligned, p1_aligned):
        to_acc = lambda v: v * 1.0
        out = {
            "ImageImage": {
                "Baseline": to_acc(class_top1_retrieval(i1_te, i2_te, labels_te, labels_te)),
                "Procrustes": to_acc(class_top1_retrieval(i1_aligned, i2_te, labels_te, labels_te)),
            },
            "TextText": {
                "Baseline": to_acc(class_top1_retrieval(p1, p2, proto_lbl, proto_lbl)),
                "Procrustes": to_acc(class_top1_retrieval(p1_aligned, p2, proto_lbl, proto_lbl)),
            },
            "ImageText": {
                "A_to_A": to_acc(image_to_class_top1(i1_te, p1, labels_te)),
                "B_to_B": to_acc(image_to_class_top1(i2_te, p2, labels_te)),
                "Aligned_imgA_to_textB": to_acc(image_to_class_top1(i1_aligned, p2, labels_te)),
                "Aligned_imgA_to_aligned_textA": to_acc(image_to_class_top1(i1_aligned, p1_aligned, labels_te)),
                "ImgB_to_aligned_textA": to_acc(image_to_class_top1(i2_te, p1_aligned, labels_te)),
            },
            "Cosine": {
                "Image_before": mean_pairwise_cosine(i1_te, i2_te),
                "Image_after": mean_pairwise_cosine(i1_aligned, i2_te),
                "Text_before": mean_pairwise_cosine(p1, p2),
                "Text_after": mean_pairwise_cosine(p1_aligned, p2)
            },
        }
        if n_unseen > 0 and mask_te_seen.sum() > 0 and mask_te_unseen.sum() > 0:
            p1_seen_aligned = p1_aligned[seen_class_ids]
            p1_unseen_aligned = p1_aligned[unseen_class_ids]
            out["ImageImage_Seen"] = {
                "Baseline": to_acc(class_top1_retrieval(i1_te[mask_te_seen], i2_te, labels_te[mask_te_seen], labels_te)),
                "Procrustes": to_acc(class_top1_retrieval(i1_aligned[mask_te_seen], i2_te, labels_te[mask_te_seen], labels_te)),
            }
            out["ImageImage_Unseen"] = {
                "Baseline": to_acc(class_top1_retrieval(i1_te[mask_te_unseen], i2_te, labels_te[mask_te_unseen], labels_te)),
                "Procrustes": to_acc(class_top1_retrieval(i1_aligned[mask_te_unseen], i2_te, labels_te[mask_te_unseen], labels_te)),
            }
            out["TextText_Seen"] = {
                "Baseline": to_acc(class_top1_retrieval(p1_seen, p2, seen_class_ids, proto_lbl)),
                "Procrustes": to_acc(class_top1_retrieval(p1_seen_aligned, p2, seen_class_ids, proto_lbl)),
            }
            out["TextText_Unseen"] = {
                "Baseline": to_acc(class_top1_retrieval(p1_unseen, p2, unseen_class_ids, proto_lbl)),
                "Procrustes": to_acc(class_top1_retrieval(p1_unseen_aligned, p2, unseen_class_ids, proto_lbl)),
            }
            out["ImageText_Seen"] = {
                "A_to_A": to_acc(image_to_class_top1(i1_te[mask_te_seen], p1, labels_te[mask_te_seen])),
                "B_to_B": to_acc(image_to_class_top1(i2_te[mask_te_seen], p2, labels_te[mask_te_seen])),
                "Aligned_imgA_to_textB": to_acc(image_to_class_top1(i1_aligned[mask_te_seen], p2, labels_te[mask_te_seen])),
                "Aligned_imgA_to_aligned_textA": to_acc(image_to_class_top1(i1_aligned[mask_te_seen], p1_aligned, labels_te[mask_te_seen])),
                "ImgB_to_aligned_textA": to_acc(image_to_class_top1(i2_te[mask_te_seen], p1_aligned, labels_te[mask_te_seen])),
            }

            out["ImageText_Unseen"] = {
                "A_to_A": to_acc(image_to_class_top1(i1_te[mask_te_unseen], p1, labels_te[mask_te_unseen])),
                "B_to_B": to_acc(image_to_class_top1(i2_te[mask_te_unseen], p2, labels_te[mask_te_unseen])),
                "Aligned_imgA_to_textB": to_acc(image_to_class_top1(i1_aligned[mask_te_unseen], p2, labels_te[mask_te_unseen])),
                "Aligned_imgA_to_aligned_textA": to_acc(image_to_class_top1(i1_aligned[mask_te_unseen], p1_aligned, labels_te[mask_te_unseen])),
                "ImgB_to_aligned_textA": to_acc(image_to_class_top1(i2_te[mask_te_unseen], p1_aligned, labels_te[mask_te_unseen])),
            }

            out["Cosine_Seen"] = {
                "Image_before": mean_pairwise_cosine(i1_te[mask_te_seen], i2_te[mask_te_seen]),
                "Image_after": mean_pairwise_cosine(i1_aligned[mask_te_seen], i2_te[mask_te_seen]),
                "Text_before": mean_pairwise_cosine(p1_seen, p2_seen),
                "Text_after": mean_pairwise_cosine(p1_seen_aligned, p2_seen),
            }
            out["Cosine_Unseen"] = {
                "Image_before": mean_pairwise_cosine(i1_te[mask_te_unseen], i2_te[mask_te_unseen]),
                "Image_after": mean_pairwise_cosine(i1_aligned[mask_te_unseen], i2_te[mask_te_unseen]),
                "Text_before": mean_pairwise_cosine(p1_unseen, p2_unseen),
                "Text_after": mean_pairwise_cosine(p1_unseen_aligned, p2_unseen),
            }
        return out

    return metrics(i1_aff, p1_aff), metrics(i1_std, p1_std)


def main():
    p = argparse.ArgumentParser(description="Few-shot alignment: train on N classes, evaluate on all.")
    p.add_argument("--dataset", default="oxford", choices=["oxford", "cifar100", "caltech101", "dtd", "stl10"])
    p.add_argument("--data_root", default="../data")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--class_split_seeds", default="42,43,44", help="Seeds for seen/unseen class split")
    p.add_argument("--num_train_classes", type=int, default=15, help="If set, use only this many classes for training alignment; eval on all.")
    p.add_argument("--single_prompt", type=str2bool, default=False)
    p.add_argument("--clip_model_1", default="ViT-B-32")
    p.add_argument("--pretrained_1", default="openai")
    p.add_argument("--clip_model_2", default="ViT-B-32")
    p.add_argument("--pretrained_2", default="laion400m_e31")
    args = p.parse_args()

    if args.num_train_classes is None:
        args.num_train_classes = 9999

    exp_id = f"A={args.clip_model_1}-{args.pretrained_1}__B={args.clip_model_2}-{args.pretrained_2}".replace("/", "_")
    n_str = f"N{args.num_train_classes}" if args.num_train_classes < 9999 else "all"
    base = f"./results/{args.dataset}/{exp_id}/few_anchors_{n_str}"
    paths = {
        "rotation": os.path.join(base, "rotation", f"{args.dataset}_results.json"),
        "rotation_without_centering": os.path.join(base, "rotation_without_centering", f"{args.dataset}_results.json"),
    }

    if all(os.path.exists(path) for path in paths.values()):
        print(f"Results exist: {paths['rotation']}")
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    class_split_seeds = [int(s) for s in args.class_split_seeds.split(",")]
    if len(class_split_seeds) < len(seeds):
        class_split_seeds = class_split_seeds + [class_split_seeds[-1]] * (len(seeds) - len(class_split_seeds))

    results_aff = {}
    results_std = {}
    for seed, cseed in zip(seeds, class_split_seeds):
        aff, std = run_single_seed(args, seed, cseed)
        results_aff[f"seed_{seed}"] = aff
        results_std[f"seed_{seed}"] = std

    for key, path in [("rotation", paths["rotation"]), ("rotation_without_centering", paths["rotation_without_centering"])]:
        data = results_aff if key == "rotation" else results_std
        seed_dicts = [data[f"seed_{s}"] for s in seeds]
        mean_d, std_d = aggregate_results(seed_dicts)
        data["mean"], data["std"] = mean_d, std_d
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {path}")

    print(f"\n=== MEAN (few_anchors {n_str}, rotation) ===\n{json.dumps(results_aff['mean'], indent=2)}")


if __name__ == "__main__":
    main()
