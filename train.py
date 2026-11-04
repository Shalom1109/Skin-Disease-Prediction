"""Train the SkinSense ensemble.

    python train.py --data-dir path/to/dataset --metadata path/to/HAM10000_metadata.csv

Differences from the original script (all of them fix evaluation problems):
  * lesion-grouped, class-stratified train/val/test split (no leakage)  -> backend/data_utils.py
  * validation/test use the deterministic eval transform (no augmentation)
  * class weights are computed from the training split, not hard-coded
  * the best checkpoint is chosen by balanced accuracy (not raw accuracy,
    which is dominated by the 67 % Nevi class) and melanoma recall is logged
  * works on CPU too (AMP only when CUDA is available); no hard-coded paths
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.data_utils import class_counts, load_or_make_splits, make_subsets  # noqa: E402
from backend.metrics import classification_report, collect_logits, softmax  # noqa: E402
from backend.model import SkinSenseEnsemble, load_state  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("train_log.txt", mode="a")])
log = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.environ.get("SKINSENSE_DATA", "dataset"),
                   help="ImageFolder root with 7 class sub-folders (default: ./dataset)")
    p.add_argument("--metadata", default=None, help="HAM10000_metadata.csv (needed for lesion-grouped splits)")
    p.add_argument("--splits-file", default="splits.json")
    p.add_argument("--rebuild-splits", action="store_true")
    p.add_argument("--out-dir", default="backend")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum-steps", type=int, default=8, help="effective batch = batch-size * accum-steps")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1,
                   help="lr multiplier for unfrozen backbone layers when --finetune is set")
    p.add_argument("--finetune", action="store_true",
                   help="also train the last stage of each backbone (not just the heads)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    best_path = os.path.join(args.out_dir, "skin_sense_ensemble_best.pth")
    ckpt_path = os.path.join(args.out_dir, "checkpoint_latest.pth")
    final_path = os.path.join(args.out_dir, "skin_sense_ensemble_final.pth")

    splits, class_names, _samples, targets, info = load_or_make_splits(
        args.data_dir, args.metadata, args.seed, args.splits_file, args.rebuild_splits)
    log.info("Split sizes: train=%d val=%d test=%d | grouped by lesion: %s",
             len(splits["train"]), len(splits["val"]), len(splits["test"]), info["grouped_by_lesion"])

    train_ds, val_ds, _test_ds = make_subsets(args.data_dir, splits)
    counts = class_counts(targets, splits["train"], len(class_names))
    log.info("Train class counts: %s", dict(zip(class_names, counts.tolist())))
    class_w = 1.0 / np.maximum(counts, 1)
    sampler = WeightedRandomSampler([class_w[targets[i]] for i in splits["train"]],
                                    num_samples=len(splits["train"]), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=use_amp)
    val_loader = DataLoader(val_ds, batch_size=min(args.batch_size, 8), shuffle=False,
                            num_workers=args.workers, pin_memory=use_amp)

    model = SkinSenseEnsemble(num_classes=len(class_names), pretrained=True,
                              finetune_last_stage=args.finetune).to(device)
    groups = [{"params": list(model.head_parameters()), "lr": args.lr}]
    backbone = model.backbone_trainable_parameters()
    if backbone:
        groups.append({"params": backbone, "lr": args.lr * args.backbone_lr_mult})
    optimizer = optim.AdamW(groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_score = 0, -1.0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_balanced_accuracy", -1.0)
        log.info("Resumed from checkpoint at epoch %d", start_epoch + 1)
    elif os.path.exists(best_path):
        try:
            model.load_state_dict(load_state(best_path, device))
            log.info("Initialised from %s", best_path)
        except Exception as exc:
            log.warning("Could not load %s: %s", best_path, exc)

    mel = class_names.index("Melanoma")
    log.info("Training on %s for epochs %d..%d", device, start_epoch + 1, args.epochs)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, n_batches = 0.0, len(train_loader)
        optimizer.zero_grad()
        for b, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(images), labels) / args.accum_steps
            scaler.scale(loss).backward()
            if (b + 1) % args.accum_steps == 0 or (b + 1) == n_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running += loss.item() * args.accum_steps
            if (b + 1) % 50 == 0:
                log.info("Epoch [%d/%d] batch [%d/%d] loss %.4f", epoch + 1, args.epochs, b + 1, n_batches,
                         loss.item() * args.accum_steps)
        scheduler.step()

        if use_amp:
            torch.cuda.empty_cache()     # free training-time cache before validation
        log.info("Epoch [%d/%d] training done, validating %d images...", epoch + 1, args.epochs, len(val_ds))
        logits, labels = collect_logits(model, val_loader, device, log_every=20)
        rep = classification_report(labels, softmax(logits), class_names)
        score = rep["balanced_accuracy"]
        log.info("Epoch [%d/%d] loss %.4f | val acc %.2f%% | balanced acc %.2f%% | macro-F1 %.3f | melanoma recall %.2f%%",
                 epoch + 1, args.epochs, running / max(n_batches, 1), 100 * rep["accuracy"], 100 * score,
                 rep["macro_f1"], 100 * rep["per_class"]["Melanoma"]["recall"])

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
            log.info("New best model saved (balanced accuracy %.2f%%)", 100 * score)

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_balanced_accuracy": best_score}, ckpt_path)

    torch.save(model.state_dict(), final_path)
    log.info("Done. Now run:  python evaluate.py --data-dir %s", args.data_dir)


if __name__ == "__main__":
    main()
