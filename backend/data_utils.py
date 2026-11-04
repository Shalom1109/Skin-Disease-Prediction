"""Dataset helpers shared by train.py, train_kfold.py and evaluate.py.

Why this exists
---------------
HAM10000 contains several photographs of the same lesion (`lesion_id`). Splitting
by *image* puts near-duplicates in both train and validation and inflates the
score. Here splits are made by *lesion*, stratified by class, with fixed
train / val / test sets (~70 / 15 / 15) saved to splits.json so every script
evaluates on exactly the same held-out data.
"""
import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset

logger = logging.getLogger("skinsense.data")

IMG_SIZE = 384
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

HAM_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DISPLAY_NAMES = ["Actinic Keratoses", "Basal Cell Carcinoma", "Benign Keratosis",
                 "Dermatofibroma", "Melanoma", "Nevi", "Vascular"]


def train_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform(img_size=IMG_SIZE):
    # Same preprocessing as inference.py (resize + center crop).
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def find_metadata_csv(data_dir, explicit=None):
    candidates = [explicit, os.environ.get("SKINSENSE_METADATA"),
                  Path(data_dir) / "HAM10000_metadata.csv",
                  Path(data_dir).parent / "HAM10000_metadata.csv"]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


def load_groups(samples, metadata_csv):
    """Group id per sample (lesion_id when metadata is available, else unique)."""
    if metadata_csv is None:
        return list(range(len(samples))), False
    lesion_of = {}
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lesion_of[row["image_id"]] = row["lesion_id"]
    groups, missing = [], 0
    for path, _ in samples:
        lesion = lesion_of.get(Path(path).stem)
        if lesion is None:
            missing += 1
            lesion = f"__solo__{path}"
        groups.append(lesion)
    if missing:
        logger.warning("%d images were not found in the metadata CSV and are treated as separate lesions.", missing)
    return groups, True


def make_splits(labels, groups, seed=42):
    """Lesion-grouped, class-stratified 70/15/15 split -> dict of index lists."""
    from sklearn.model_selection import StratifiedGroupKFold

    idx = np.arange(len(labels))
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    outer = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    rest, test = next(outer.split(idx, labels, groups))
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, va = next(inner.split(rest, labels[rest], groups[rest]))
    return {"train": rest[tr].tolist(), "val": rest[va].tolist(), "test": test.tolist()}


def check_no_overlap(splits, groups):
    sets = {k: {groups[i] for i in v} for k, v in splits.items()}
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b and sets[a] & sets[b]:
                raise AssertionError(f"Lesion leakage between {a} and {b}")


def load_or_make_splits(data_dir, metadata=None, seed=42, splits_file="splits.json", rebuild=False):
    """Returns (splits, class_names, samples, targets, info)."""
    base = datasets.ImageFolder(data_dir)
    samples, targets = base.samples, [t for _, t in base.samples]
    folder_names = base.classes
    if len(folder_names) != len(DISPLAY_NAMES):
        raise ValueError(f"Expected {len(DISPLAY_NAMES)} class folders, found {folder_names}")
    known = set(HAM_CODES) | {n.lower() for n in DISPLAY_NAMES}
    if not {n.lower() for n in folder_names} <= known | {"vascular lesions"}:
        logger.warning("Class folders %s are not the usual HAM10000 codes; assuming alphabetical order "
                       "matches %s.", folder_names, HAM_CODES)

    csv_path = find_metadata_csv(data_dir, metadata)
    groups, grouped = load_groups(samples, csv_path)

    splits_path = Path(splits_file)
    root = Path(data_dir).resolve()
    rel = lambda p: str(Path(p).resolve().relative_to(root)).replace("\\", "/")
    names = [rel(p) for p, _ in samples]

    if splits_path.exists() and not rebuild:
        saved = json.loads(splits_path.read_text())
        pos = {n: i for i, n in enumerate(names)}
        splits = {k: [pos[n] for n in saved[k] if n in pos] for k in ("train", "val", "test")}
        info = {"grouped_by_lesion": saved.get("grouped_by_lesion", False), "seed": saved.get("seed"),
                "loaded_from": str(splits_path)}
    else:
        splits = make_splits(targets, groups, seed)
        if grouped:
            check_no_overlap(splits, groups)
        else:
            logger.warning("HAM10000_metadata.csv NOT found: splitting by image. The same lesion can appear "
                           "in train and test, so scores will be optimistic. Pass --metadata to fix this.")
        splits_path.write_text(json.dumps({
            "seed": seed, "grouped_by_lesion": grouped,
            **{k: [names[i] for i in v] for k, v in splits.items()},
        }))
        info = {"grouped_by_lesion": grouped, "seed": seed, "loaded_from": None}

    return splits, DISPLAY_NAMES, samples, targets, info


def make_subsets(data_dir, splits, img_size=IMG_SIZE):
    """Separate dataset objects so train uses augmentation and val/test never do."""
    train_ds = datasets.ImageFolder(data_dir, transform=train_transform(img_size))
    eval_ds = datasets.ImageFolder(data_dir, transform=eval_transform(img_size))
    return (Subset(train_ds, splits["train"]),
            Subset(eval_ds, splits["val"]),
            Subset(eval_ds, splits["test"]))


def class_counts(targets, indices, num_classes=7):
    counts = np.zeros(num_classes, dtype=np.int64)
    for i in indices:
        counts[targets[i]] += 1
    return counts
