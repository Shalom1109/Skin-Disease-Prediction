"""Lesion-grouped, class-stratified k-fold cross-validation.

    python train_kfold.py --data-dir dataset --metadata HAM10000_metadata.csv

Gives mean +/- std of balanced accuracy and melanoma recall, which is a much
more honest estimate than a single split. The held-out *test* split from
splits.json is never touched here - folds are drawn from train+val only.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.data_utils import (eval_transform, find_metadata_csv, load_groups,  # noqa: E402
                                load_or_make_splits, train_transform, DISPLAY_NAMES)
from backend.metrics import classification_report, collect_logits, softmax  # noqa: E402
from backend.model import SkinSenseEnsemble  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.environ.get("SKINSENSE_DATA", "dataset"))
    p.add_argument("--metadata", default=None)
    p.add_argument("--splits-file", default="splits.json")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--finetune", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    splits, _names, samples, targets, _ = load_or_make_splits(
        args.data_dir, args.metadata, args.seed, args.splits_file)
    groups, grouped = load_groups(samples, find_metadata_csv(args.data_dir, args.metadata))
    if not grouped:
        print("WARNING: no metadata CSV - folds are split by image and will leak lesions.")

    pool = np.array(splits["train"] + splits["val"])
    y = np.array(targets)
    g = np.array(groups, dtype=object)
    train_ds = datasets.ImageFolder(args.data_dir, transform=train_transform())
    eval_ds = datasets.ImageFolder(args.data_dir, transform=eval_transform())

    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    bal, mel = [], []
    for fold, (tr, va) in enumerate(skf.split(pool, y[pool], g[pool]), 1):
        tr_idx, va_idx = pool[tr], pool[va]
        print(f"--- FOLD {fold}/{args.folds}: train={len(tr_idx)} val={len(va_idx)} ---")
        counts = np.bincount(y[tr_idx], minlength=7)
        w = 1.0 / np.maximum(counts, 1)
        sampler = WeightedRandomSampler([w[y[i]] for i in tr_idx], len(tr_idx), replacement=True)
        tl = DataLoader(Subset(train_ds, tr_idx.tolist()), batch_size=args.batch_size, sampler=sampler)
        vl = DataLoader(Subset(eval_ds, va_idx.tolist()), batch_size=args.batch_size)

        model = SkinSenseEnsemble(num_classes=7, pretrained=True, finetune_last_stage=args.finetune).to(device)
        opt = optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best = None
        for epoch in range(args.epochs):
            model.train()
            for x, t in tl:
                x, t = x.to(device), t.to(device)
                opt.zero_grad()
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = crit(model(x), t)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            logits, labels = collect_logits(model, vl, device)
            rep = classification_report(labels, softmax(logits), DISPLAY_NAMES)
            print(f"  epoch {epoch + 1}: balanced acc {rep['balanced_accuracy']:.4f} | "
                  f"melanoma recall {rep['per_class']['Melanoma']['recall']:.4f}")
            if best is None or rep["balanced_accuracy"] > best["balanced_accuracy"]:
                best = rep
        bal.append(best["balanced_accuracy"])
        mel.append(best["per_class"]["Melanoma"]["recall"])

    print(f"\nBalanced accuracy: {np.mean(bal):.4f} +/- {np.std(bal):.4f}   folds={np.round(bal, 4).tolist()}")
    print(f"Melanoma recall  : {np.mean(mel):.4f} +/- {np.std(mel):.4f}")


if __name__ == "__main__":
    main()
