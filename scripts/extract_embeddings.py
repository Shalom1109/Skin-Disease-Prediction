"""Extract one feature vector per image from a (foundation) model, for fast model comparison.

    python scripts/extract_embeddings.py --model timm:convnext_base --out runs/emb/convnext_in1k.npz
    python scripts/train_embedding_head.py --emb runs/emb/convnext_in1k.npz

Supported models
  timm:<name>        any timm model with pretrained weights (works offline once cached)
  derm-foundation    Google's Derm Foundation (6144-d). GATED: you must accept the terms on Hugging Face
                     (huggingface.co/google/derm-foundation), log in with `huggingface-cli login`, and
                     `pip install tensorflow`. The calling convention below follows the public model card and has
                     NOT been run here - check it against the card if it errors.
For any other model (e.g. PanDerm), add a small class with an `.embed(list_of_PIL_images) -> ndarray` method.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from backend.data_utils import DISPLAY_NAMES  # noqa: E402


class TimmEmbedder:
    def __init__(self, name, device, img_size=384):
        import timm
        from torchvision import transforms
        self.model = timm.create_model(name, pretrained=True, num_classes=0).to(device).eval()
        cfg = self.model.pretrained_cfg
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(),
            transforms.Normalize(cfg.get("mean", (0.485, 0.456, 0.406)), cfg.get("std", (0.229, 0.224, 0.225)))])

    def embed(self, images):
        x = torch.stack([self.transform(i) for i in images]).to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            return self.model(x).float().cpu().numpy()


class DermFoundationEmbedder:
    """Unverified adapter - see the module docstring."""

    def __init__(self):
        import tensorflow as tf
        from huggingface_hub import snapshot_download
        self.tf = tf
        self.infer = tf.saved_model.load(snapshot_download(repo_id="google/derm-foundation")).signatures["serving_default"]

    def embed(self, images):
        import io
        tf, out = self.tf, []
        for img in images:
            buf = io.BytesIO()
            img.convert("RGB").resize((448, 448)).save(buf, "PNG")
            example = tf.train.Example(features=tf.train.Features(feature={
                "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[buf.getvalue()]))}))
            out.append(self.infer(inputs=tf.constant([example.SerializeToString()]))["embedding"].numpy().flatten())
        return np.stack(out)


class _Images(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return Image.open(self.files[i]).convert("RGB"), i


def extract(embedder, data_dir, batch_size=32, workers=4, limit=None):
    """-> (features [N,D], labels [N], relative names [N]) in sorted ImageFolder order."""
    root = Path(data_dir)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    files, labels = [], []
    for ci, c in enumerate(classes):
        for f in sorted((root / c).glob("*.jpg")):
            files.append(f)
            labels.append(ci)
    if limit:
        files, labels = files[:limit], labels[:limit]
    loader = DataLoader(_Images(files), batch_size=batch_size, num_workers=workers,
                        collate_fn=lambda b: ([x[0] for x in b], [x[1] for x in b]))
    feats = []
    for n, (imgs, _idx) in enumerate(loader, 1):
        feats.append(embedder.embed(imgs))
        if n % 20 == 0:
            print(f"  {min(n * batch_size, len(files))}/{len(files)}", flush=True)
    names = [str(f.relative_to(root)).replace("\\", "/") for f in files]
    return np.concatenate(feats), np.array(labels), np.array(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="timm:<name> or derm-foundation")
    ap.add_argument("--data-dir", default=os.environ.get("SKINSENSE_DATA", "dataset"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model.startswith("timm:"):
        embedder = TimmEmbedder(args.model.split(":", 1)[1], device, args.img_size)
    elif args.model == "derm-foundation":
        embedder = DermFoundationEmbedder()
    else:
        sys.exit("Unknown --model. Use timm:<name> or derm-foundation.")
    feats, labels, names = extract(embedder, args.data_dir, args.batch_size, args.workers)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, features=feats, labels=labels, names=names, model=args.model,
                        class_names=np.array(DISPLAY_NAMES))
    print(f"Saved {feats.shape} to {args.out}")


if __name__ == "__main__":
    main()
