import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from config import DATASET_ROOT, BATCH_SIZE, NUM_WORKERS
from data.transforms import get_transform, get_mask_transform


class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", transform=None, mask_transform=None):
        self.root = Path(root) / category
        self.category = category
        self.split = split
        self.transform = transform or get_transform()
        self.mask_transform = mask_transform or get_mask_transform()
        self.samples = []  # list of (img_path, mask_path_or_None, label, defect_type)
        self._load_samples()

    def _load_samples(self):
        split_dir = self.root / self.split
        if self.split == "train":
            good_dir = split_dir / "good"
            for img_path in sorted(good_dir.glob("*.png")):
                self.samples.append((img_path, None, 0, "good"))
        else:
            for defect_dir in sorted(split_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect_type = defect_dir.name
                label = 0 if defect_type == "good" else 1
                for img_path in sorted(defect_dir.glob("*.png")):
                    if label == 1:
                        # ground_truth mask path mirrors test structure
                        mask_path = (
                            self.root / "ground_truth" / defect_type /
                            (img_path.stem + "_mask.png")
                        )
                        mask_path = mask_path if mask_path.exists() else None
                    else:
                        mask_path = None
                    self.samples.append((img_path, mask_path, label, defect_type))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label, defect_type = self.samples[idx]
        
        # Correction chirurgicale pour les catégories Grayscale (Screw, Grid, Zipper)
        if any(cat in str(img_path).lower() for cat in ["screw", "grid", "zipper"]):
            # On charge en mode L (8-bit pixels, black and white)
            raw_img = Image.open(img_path).convert("L")
            # On duplique le canal pour le WideResNet50 (R=G=B)
            image = Image.merge("RGB", (raw_img, raw_img, raw_img))
        else:
            image = Image.open(img_path).convert("RGB")
            
        image = self.transform(image)

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, image.shape[1], image.shape[2])

        return image, label, mask, defect_type


def get_dataloader(category, split="train", shuffle=None):
    if shuffle is None:
        shuffle = (split == "train")
    dataset = MVTecDataset(DATASET_ROOT, category, split)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
