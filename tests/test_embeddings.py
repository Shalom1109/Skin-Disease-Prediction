"""End-to-end test of the embedding-comparison tooling on a tiny synthetic dataset (no GPU, no downloads)."""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import extract_embeddings as xe  # noqa: E402
import train_embedding_head as th  # noqa: E402
from backend import data_utils  # noqa: E402

CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


class MeanColour:
    """Stand-in embedder: a class-correlated image statistic (mean colour + coarse 4x4 grey grid)."""

    def embed(self, images):
        out = []
        for im in images:
            a = np.asarray(im.resize((8, 8)), dtype=float) / 255.0
            out.append(np.concatenate([a.mean(axis=(0, 1)), a.mean(axis=2)[::2, ::2].ravel()]))
        return np.array(out)


def make_dataset(root, per_class=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    n = 0
    for ci, code in enumerate(CODES):
        (root / code).mkdir(parents=True)
        base = np.array([30 + 30 * ci, 200 - 25 * ci, 90 + 10 * ci], dtype=float)
        for k in range(per_class):
            n += 1
            iid = f"ISIC_{n:07d}"
            arr = np.clip(rng.normal(base, 18, (48, 48, 3)), 0, 255).astype("uint8")
            Image.fromarray(arr).save(root / code / f"{iid}.jpg")
            rows.append((f"HAM_{code}_{k // 2}", iid, code))      # two photos per lesion
    with open(root / "HAM10000_metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lesion_id", "image_id", "dx"])
        w.writerows(rows)


def test_extract_then_train_head_end_to_end(tmp_path):
    data = tmp_path / "dataset"
    make_dataset(data)
    splits_file = tmp_path / "splits.json"
    splits, *_ = data_utils.load_or_make_splits(str(data), str(data / "HAM10000_metadata.csv"), 1, str(splits_file))
    assert all(len(splits[k]) > 0 for k in ("train", "val", "test"))

    feats, labels, names = xe.extract(MeanColour(), data, batch_size=16, workers=0)
    assert feats.shape[0] == len(names) == len(CODES) * 24 and (labels == np.repeat(range(7), 24)).all()
    emb = tmp_path / "emb.npz"
    np.savez_compressed(emb, features=feats, labels=labels, names=names, model="dummy",
                        class_names=np.array(data_utils.DISPLAY_NAMES))

    res = th.run(str(emb), str(splits_file), metadata=str(data / "HAM10000_metadata.csv"), data_dir=str(data), n_boot=50)
    assert res["n_train"] + res["n_val"] + res["n_test"] == len(names)
    assert res["test"]["balanced_accuracy"] > 0.8, "classes are colour-separable, the head must learn them"
    assert res["confidence_intervals_95"]["accuracy"]["low"] <= res["test"]["accuracy"] <= res["confidence_intervals_95"]["accuracy"]["high"]
    assert 0 < res["melanoma_threshold"] <= 1 and res["temperature"] > 0


def test_rows_follow_the_saved_split_names():
    names = np.array(["a/1.jpg", "a/2.jpg", "b/3.jpg"])
    rows = th.rows_for(names, {"train": ["b/3.jpg"], "val": ["a/1.jpg"], "test": ["a/2.jpg", "missing.jpg"]})
    assert rows["train"].tolist() == [2] and rows["val"].tolist() == [0] and rows["test"].tolist() == [1]
