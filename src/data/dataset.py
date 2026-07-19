"""
Data pipeline: directory scanning, filename-based weak labelling,
the AnomalyDataset, image transforms, and DataLoader factory.
"""
import logging
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

logger = logging.getLogger("ConvNeXtAutoencoder")


def to_grayscale(img: Image.Image) -> Image.Image:
    """Convert an RGB PIL image to grayscale (no equalization), replicated
    back into 3 channels (RGB) so it stays compatible with a pretrained
    3-channel backbone."""
    gray = img.convert("L")             # [H, W] single channel
    return gray.convert("RGB")          # replicate L -> R=G=B


def grayscale_equalize(img: Image.Image) -> Image.Image:
    """Convert an RGB PIL image to grayscale then apply histogram equalization.

    The single equalized channel is replicated back into 3 channels (RGB) so
    the output stays compatible with a pretrained 3-channel backbone
    (e.g. ConvNeXt expects 3-channel input + ImageNet normalization).
    """
    gray = np.array(img.convert("L"))          # [H, W] uint8
    equalized = cv2.equalizeHist(gray)          # [H, W] uint8, contrast-stretched
    equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(equalized_rgb)


class Grayscale:
    """torchvision-style transform wrapper around `to_grayscale` (no equalization).

    Insert into a `v2.Compose([...])` pipeline (operates on PIL images,
    so place it BEFORE `v2.ToImage()` / `v2.ToDtype()` / `v2.Normalize()`).
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        return to_grayscale(img)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class GrayscaleEqualize:
    """torchvision-style transform wrapper around `grayscale_equalize`.

    Insert into a `v2.Compose([...])` pipeline (operates on PIL images,
    so place it BEFORE `v2.ToImage()` / `v2.ToDtype()` / `v2.Normalize()`).
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        return grayscale_equalize(img)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def infer_label_from_filename(filename: str, cfg) -> Optional[str]:
    """Infer 'normal' / 'anomaly' label from filename keywords in cfg."""
    name = filename.lower()
    has_normal = any(k in name for k in cfg.NORMAL_KEYWORDS)
    has_anomaly = any(k in name for k in cfg.ANOMALY_KEYWORDS)

    if has_normal and has_anomaly:
        logger.warning(f"Ambiguous keywords in filename: {filename} → Skipped.")
        return None

    if has_anomaly:
        return "anomaly"
    if has_normal:
        return "normal"

    return None


def scan_directory(directory: str, cfg) -> pd.DataFrame:
    """Recursively scan `directory` for valid images and label them."""
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"Not found: {directory}")

    rows = []
    valid_files = [
        f for f in d.rglob("*")
        if f.is_file() and f.suffix.lower() in cfg.VALID_EXT
    ]

    for f in sorted(valid_files):
        label = infer_label_from_filename(f.name, cfg)
        rows.append({
            "path": str(f),
            "filename": f.name,
            "label": label
        })

    if not rows:
        logger.warning(f"No valid files found in {directory}")
        return pd.DataFrame(columns=["path", "filename", "label"])

    df = pd.DataFrame(rows)
    n_unk = df["label"].isna().sum()
    if n_unk:
        logger.warning(f"{directory}: {n_unk} unlabelled/ambiguous files dropped.")

    return df.dropna(subset=["label"]).reset_index(drop=True)


class AnomalyDataset(Dataset):
    """Returns (normalized_tensor, display_tensor, path, label, (orig_w, orig_h))."""

    def __init__(self, df: pd.DataFrame, norm_tf, orig_tf, image_size=(224, 224)):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()

        self.norm_tf = norm_tf
        self.orig_tf = orig_tf
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                ow, oh = img.size
                norm_t = self.norm_tf(img)
                orig_t = self.orig_tf(img)

            return norm_t, orig_t, path, label, (ow, oh)

        except Exception as e:
            logger.error(f"Load failed {path}: {e}")
            random_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(random_idx)


def build_transforms(cfg):
    """Build the ImageNet-normalized transform (with/without augmentation),
    and the plain 'display' transform used for visualisation.

    Color mode is controlled by cfg.USE_GRAYSCALE / cfg.USE_GRAYSCALE_EQUALIZATION
    (see config.py for the 3 supported combinations: RGB / Grayscale /
    Grayscale+Equalization). The color-mode step is applied only to the
    model-input pipelines (imagenet_tf / train_aug_tf); `display_tf` used
    for the "original image" gallery is left untouched so users can still
    see the true original photo regardless of the selected mode.
    """
    color_mode = getattr(cfg, 'COLOR_MODE', 'rgb')
    if color_mode == 'grayscale_equalized':
        color_step = [GrayscaleEqualize()]
    elif color_mode == 'grayscale':
        color_step = [Grayscale()]
    elif color_mode == 'rgb':
        color_step = []
    else:
        raise ValueError(f"Unknown cfg.COLOR_MODE: {color_mode!r}")

    imagenet_tf = v2.Compose([
        v2.Resize(cfg.IMAGE_SIZE),
        *color_step,
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    train_aug_tf = v2.Compose([
        v2.Resize(cfg.IMAGE_SIZE),
        v2.ColorJitter(brightness=cfg.AUG_COLOR_JITTER, contrast=cfg.AUG_COLOR_JITTER),
        *color_step,
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    display_tf = v2.Compose([
        v2.Resize(cfg.IMAGE_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    return imagenet_tf, train_aug_tf, display_tf


def make_loader(ds, cfg, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=False,
    )


def build_datasets_and_loaders(cfg):
    """Scan train/val/test dirs, build datasets and dataloaders (including the
    normal-only loader used to train the autoencoder)."""
    df_train = scan_directory(cfg.TRAIN_DIR, cfg)
    df_val = scan_directory(cfg.VAL_DIR, cfg)
    df_test = scan_directory(cfg.TEST_DIR, cfg)

    imagenet_tf, train_aug_tf, display_tf = build_transforms(cfg)

    train_ds = AnomalyDataset(df_train, imagenet_tf, display_tf, cfg.IMAGE_SIZE)
    val_ds = AnomalyDataset(df_val, imagenet_tf, display_tf, cfg.IMAGE_SIZE)
    test_ds = AnomalyDataset(df_test, imagenet_tf, display_tf, cfg.IMAGE_SIZE)

    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg)
    test_loader = make_loader(test_ds, cfg)

    df_train_normal = df_train[df_train["label"] == "normal"].reset_index(drop=True)
    normal_norm_tf = train_aug_tf if cfg.USE_AUGMENTATION else imagenet_tf
    normal_ds = AnomalyDataset(df_train_normal, normal_norm_tf, display_tf, cfg.IMAGE_SIZE)
    normal_loader = make_loader(normal_ds, cfg, shuffle=True)

    return {
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "normal_ds": normal_ds,
        "train_loader": train_loader, "val_loader": val_loader,
        "test_loader": test_loader, "normal_loader": normal_loader,
    }