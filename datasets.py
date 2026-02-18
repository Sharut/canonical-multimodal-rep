"""
Dataset classes for various vision datasets.
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class OxfordPetsDataset(Dataset):
    """Oxford-IIIT Pet Dataset."""
    
    def __init__(self, root, split="trainval"):
        from torchvision.datasets import OxfordIIITPet

        self.root = root
        self.split = split

        _ = OxfordIIITPet(root=root, split=split, target_types="category", download=True)

        dataset_dir = os.path.join(root, "oxford-iiit-pet")
        self.image_dir = os.path.join(dataset_dir, "images")
        ann_path = os.path.join(dataset_dir, "annotations", f"{split}.txt")

        with open(ann_path, "r") as f:
            self.filenames = [line.strip().split(" ")[0] for line in f.readlines()]

        self.class_names = sorted(set(
            name.rsplit("_", 1)[0].lower().replace("_", " ")
            for name in self.filenames
        ))
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        base_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, base_name + ".jpg")
        image = Image.open(img_path).convert("RGB")
        class_name = base_name.rsplit("_", 1)[0].lower().replace("_", " ")
        label = self.class_to_idx[class_name]

        return {
            "image": image,
            "label": label,
            "text": class_name,
            "class_name": class_name,
            "image_path": base_name + ".jpg"
        }


class CIFAR100Dataset(Dataset):
    """CIFAR-100 Dataset."""
    
    def __init__(self, root, split="test"):
        import pickle
        
        assert split in ["train", "test"], "split must be 'train' or 'test'"
        self.root = root
        self.split = split

        meta_path = os.path.join(root, "meta")
        with open(meta_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        self.class_names = data["fine_label_names"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        split_path = os.path.join(root, split)
        with open(split_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        self.images = data["data"].reshape(-1, 3, 32, 32)
        self.labels = data["fine_labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].transpose(1, 2, 0)
        image = Image.fromarray(img)
        label = self.labels[idx]
        class_name = self.class_names[label]

        return {
            "image": image,
            "text": class_name,
            "class_name": class_name,
            "label": label,
            "image_path": None
        }


class Caltech101Dataset(Dataset):
    """Caltech-101 Dataset."""
    
    def __init__(self, root, split="train"):
        import random
        
        self.root = root
        self.image_dir = os.path.join(root, "101_ObjectCategories")
        
        self.class_names = sorted([
            d for d in os.listdir(self.image_dir) 
            if os.path.isdir(os.path.join(self.image_dir, d)) 
            and d != "BACKGROUND_Google"
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        self.samples = []
        for cls in self.class_names:
            cls_dir = os.path.join(self.image_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append({
                        "path": os.path.join(cls_dir, img_name),
                        "label": self.class_to_idx[cls],
                        "class_name": cls.replace("_", " "),
                    })
        
        random.seed(42)
        random.shuffle(self.samples)
        split_idx = int(0.8 * len(self.samples))
        self.samples = self.samples[:split_idx] if split == "train" else self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        
        return {
            "image": image,
            "text": sample["class_name"],
            "class_name": sample["class_name"],
            "label": sample["label"],
            "image_path": sample["path"]
        }


class DTDDataset(Dataset):
    """Describable Textures Dataset."""
    
    def __init__(self, root, split="train"):
        from torchvision.datasets import DTD
        
        self.root = root
        self.split = split
        assert split in ["train", "val", "test"], "split must be 'train', 'val', or 'test'"
        
        _ = DTD(root=root, split=split, download=True)
        dtd_dataset = DTD(root=root, split=split, download=False)
        
        self.class_names = dtd_dataset.classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.dtd_dataset = dtd_dataset
    
    def __len__(self):
        return len(self.dtd_dataset)
    
    def __getitem__(self, idx):
        image, label = self.dtd_dataset[idx]
        image = image.convert("RGB") if hasattr(image, "convert") else image
        
        return {
            "image": image,
            "label": label,
            "text": self.class_names[label],
            "class_name": self.class_names[label],
            "image_path": f"dtd_{self.split}_{idx}.png"
        }


class STL10Dataset(Dataset):
    """STL-10 Dataset."""
    
    def __init__(self, root, split="train"):
        from torchvision.datasets import STL10
        
        self.root = root
        self.split = split
        assert split in ["train", "test"], "split must be 'train' or 'test'"
        
        self.stl10_dataset = STL10(root=root, split=split, download=True)
        self.class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
    
    def __len__(self):
        return len(self.stl10_dataset)
    
    def __getitem__(self, idx):
        image, label = self.stl10_dataset[idx]
        image = image.convert("RGB") if hasattr(image, "convert") else image
        
        return {
            "image": image,
            "label": label,
            "text": self.class_names[label],
            "class_name": self.class_names[label],
            "image_path": f"stl10_{self.split}_{idx}.png"
        }


# ======================================================
# Dataset Factory
# ======================================================

DATASET_REGISTRY = {
    "oxford": {
        "class": OxfordPetsDataset,
        "train_split": "trainval",
        "test_split": "test",
        "root_suffix": "",
    },
    "cifar100": {
        "class": CIFAR100Dataset,
        "train_split": "train",
        "test_split": "test",
        "root_suffix": "cifar100/cifar-100-python",
    },
    "caltech101": {
        "class": Caltech101Dataset,
        "train_split": "train",
        "test_split": "test",
        "root_suffix": "caltech101/caltech-101",
    },
    "dtd": {
        "class": DTDDataset,
        "train_split": "train",
        "test_split": "test",
        "root_suffix": "",
    },
    "stl10": {
        "class": STL10Dataset,
        "train_split": "train",
        "test_split": "test",
        "root_suffix": "",
    },
}


def get_dataset(name, data_root, split):
    """Factory function to create dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    
    config = DATASET_REGISTRY[name]
    root = os.path.join(data_root, config["root_suffix"]) if config["root_suffix"] else data_root
    split_name = config["train_split"] if split == "train" else config["test_split"]
    
    return config["class"](root=root, split=split_name)
