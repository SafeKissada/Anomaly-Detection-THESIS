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
    n_unk = int(df["label"].isna().sum())
    if n_unk:
        logger.warning(f"{directory}: {n_unk} unlabelled/ambiguous files dropped.")

    clean_df = df.dropna(subset=["label"]).reset_index(drop=True)
    # Attach drop-count as DataFrame metadata (pandas .attrs) rather than just
    # logging it, so callers (e.g. scripts/train.py) can surface this number
    # in final_results.json for a transparent, auditable report of how many
    # files were excluded from each split and why. .attrs does not affect
    # indexing/values, so this is fully backward-compatible with existing
    # callers that only use the returned DataFrame as before.
    clean_df.attrs["n_scanned"] = len(df)
    clean_df.attrs["n_dropped_ambiguous_or_unlabelled"] = n_unk
    clean_df.attrs["source_dir"] = str(directory)
    return clean_df


class AnomalyDataset(Dataset):
    """Returns (normalized_tensor, display_tensor, preproc_display_tensor,
    path, label, (orig_w, orig_h)).

    - normalized_tensor      : model input (ImageNet-normalized, color-mode applied)
    - display_tensor         : always the plain RGB image (0..1, no normalize) —
                               used to show the "real photo" regardless of color mode
    - preproc_display_tensor : the actual preprocessed image fed to the model
                               (grayscale / grayscale+equalized), 0..1, no normalize,
                               used for visual comparison. If no color-mode transform
                               is configured (RGB mode), this equals display_tensor.
    """

    # Hard cap on how many times __getitem__ may fall back to a random
    # replacement sample before giving up. Without this cap, a batch of
    # corrupt/unreachable files (e.g. a transient network-drive hiccup on
    # Google Drive) could recurse arbitrarily deep or silently substitute an
    # unbounded number of samples with no record of it happening.
    MAX_LOAD_RETRIES = 5

    def __init__(self, df: pd.DataFrame, norm_tf, orig_tf, image_size=(224, 224),
                 preproc_tf=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()

        self.norm_tf = norm_tf
        self.orig_tf = orig_tf
        self.preproc_tf = preproc_tf
        self.image_size = image_size
        self.n_fallbacks = 0  # running count of successful fallback substitutions

    def __len__(self):
        return len(self.paths)

    def _load_one(self, idx):
        """Load+transform a single sample. Raises on failure (no fallback here)."""
        path = self.paths[idx]
        label = self.labels[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            ow, oh = img.size
            norm_t = self.norm_tf(img)
            orig_t = self.orig_tf(img)
            preproc_t = self.preproc_tf(img) if self.preproc_tf is not None else orig_t
        return norm_t, orig_t, preproc_t, path, label, (ow, oh)

    def __getitem__(self, idx):
        try:
            return self._load_one(idx)
        except Exception as e:
            logger.error(f"Load failed {self.paths[idx]}: {e}")

        # Bounded retry loop (replaces the old unbounded recursive fallback).
        for attempt in range(self.MAX_LOAD_RETRIES):
            random_idx = random.randint(0, len(self) - 1)
            try:
                sample = self._load_one(random_idx)
                self.n_fallbacks += 1
                logger.warning(
                    f"Substituted index {idx} -> {random_idx} "
                    f"(fallback attempt {attempt + 1}/{self.MAX_LOAD_RETRIES}, "
                    f"total fallbacks so far this run: {self.n_fallbacks})")
                return sample
            except Exception as e:
                logger.error(f"Fallback load also failed {self.paths[random_idx]}: {e}")

        raise RuntimeError(
            f"AnomalyDataset: failed to load a usable sample for index {idx} "
            f"after {self.MAX_LOAD_RETRIES} random fallback attempts. "
            f"Too many corrupt/unreachable files in this split — check the "
            f"data source (e.g. a dropped network drive) rather than retrying "
            f"indefinitely.")


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
    color_mode = getattr(cfg, 'COLOR_MODE', 'RGB')
    if color_mode != 'RGB':
        # AWARENESS NOTE (this is a conceptual/theoretical
        # caveat, not something a code patch can fully "fix"): the backbone
        # (ConvNeXtExtractor) is pretrained on natural RGB ImageNet images.
        # Converting inputs to grayscale — and especially histogram-equalized
        # grayscale, which applies an aggressive non-linear contrast remap —
        # shifts the input texture/intensity statistics away from what the
        # backbone was pretrained on, even though the tensor is replicated
        # back to 3 channels so shapes still match. extractor.fit_normalization()
        # re-fits the *feature* mean/std for this color mode, which corrects
        # first-order scale/shift but does NOT fine-tune the frozen backbone
        # weights themselves, so some domain-shift risk remains. This should
        # be discussed explicitly in the thesis limitations section when
        # comparing RGB vs grayscale/equalized experiment results.
        logger.warning(
            f"COLOR_MODE={color_mode!r}: input images are converted before "
            f"the ImageNet-pretrained backbone sees them. This is a known "
            f"domain-shift risk (see Findings 2.10) — interpret color-mode "
            f"ablation results (e.g. E1/E2/E4/E5/E7/E8) with this in mind.")
    if color_mode == 'GRAYSCALE_EQUAILAZATION':
        color_step = [GrayscaleEqualize()]
    elif color_mode == 'GRAYSCALE':
        color_step = [Grayscale()]
    elif color_mode == 'RGB':
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

    if color_step:
        # Same color-mode transform as the model input, but WITHOUT ImageNet
        # normalization — kept in plain 0..1 range for visual display.
        preproc_display_tf = v2.Compose([
            v2.Resize(cfg.IMAGE_SIZE),
            *color_step,
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    else:
        preproc_display_tf = None  # RGB mode: nothing extra to preview

    return imagenet_tf, train_aug_tf, display_tf, preproc_display_tf


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

    imagenet_tf, train_aug_tf, display_tf, preproc_display_tf = build_transforms(cfg)

    train_ds = AnomalyDataset(df_train, imagenet_tf, display_tf, cfg.IMAGE_SIZE, preproc_display_tf)
    val_ds = AnomalyDataset(df_val, imagenet_tf, display_tf, cfg.IMAGE_SIZE, preproc_display_tf)
    test_ds = AnomalyDataset(df_test, imagenet_tf, display_tf, cfg.IMAGE_SIZE, preproc_display_tf)

    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg)
    test_loader = make_loader(test_ds, cfg)

    df_train_normal = df_train[df_train["label"] == "normal"].reset_index(drop=True)
    normal_norm_tf = train_aug_tf if cfg.USE_AUGMENTATION else imagenet_tf
    normal_ds = AnomalyDataset(df_train_normal, normal_norm_tf, display_tf, cfg.IMAGE_SIZE, preproc_display_tf)
    normal_loader = make_loader(normal_ds, cfg, shuffle=True)

    return {
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "normal_ds": normal_ds,
        "train_loader": train_loader, "val_loader": val_loader,
        "test_loader": test_loader, "normal_loader": normal_loader,
    }