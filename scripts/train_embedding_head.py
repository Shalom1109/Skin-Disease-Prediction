"""Train and evaluate a classifier on pre-extracted embeddings, with exactly the same split, calibration and
metrics as evaluate.py - so any backbone can be compared fairly in minutes.

    python scripts/extract_embeddings.py --model timm:convnext_base --out runs/emb/convnext_in1k.npz
    python scripts/train_embedding_head.py --emb runs/emb/convnext_in1k.npz --metadata dataset/HAM10000_metadata.csv

Pipeline: standardise -> multinomial logistic regression (class-balanced) -> choose the regularisation on the
validation split -> temperature scaling + melanoma threshold on validation -> report on the held-out test split.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from backend import metrics as M  # noqa: E402
from backend.data_utils import find_metadata_csv, load_groups  # noqa: E402


def rows_for(names, splits):
    pos = {n: i for i, n in enumerate(names.tolist())}
    return {k: np.array([pos[n] for n in splits[k] if n in pos]) for k in ("train", "val", "test")}


def run(emb_path, splits_file, metadata=None, data_dir="dataset", target_sensitivity=0.90, n_boot=500,
        c_grid=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    d = np.load(emb_path, allow_pickle=False)
    X, y, names = d["features"].astype(np.float64), d["labels"], d["names"]
    class_names = [str(c) for c in d["class_names"]]
    splits = json.loads(Path(splits_file).read_text())
    rows = rows_for(names, splits)
    mel = class_names.index("Melanoma")

    scaler = StandardScaler().fit(X[rows["train"]])
    Xs = scaler.transform(X)

    best = None
    for c in c_grid:
        clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced")
        clf.fit(Xs[rows["train"]], y[rows["train"]])
        score = balanced_accuracy_score(y[rows["val"]], clf.predict(Xs[rows["val"]]))
        if best is None or score > best[0]:
            best = (score, c, clf)
    val_bal, c, clf = best

    val_logits, test_logits = clf.decision_function(Xs[rows["val"]]), clf.decision_function(Xs[rows["test"]])
    val_y, test_y = y[rows["val"]], y[rows["test"]]
    T = M.fit_temperature(val_logits, val_y)
    thr, val_sens, val_spec = M.pick_sensitivity_threshold(M.softmax(val_logits, T)[:, mel], val_y == mel,
                                                           target_sensitivity)
    probs = M.softmax(test_logits, T)
    rep = M.classification_report(test_y, probs, class_names)
    rep["melanoma_rule"] = M.sensitivity_specificity(probs[:, mel], test_y == mel, thr)

    ci = None
    csv_path = find_metadata_csv(data_dir, metadata)
    if n_boot and csv_path:
        groups, _ = load_groups([(n, 0) for n in names.tolist()], csv_path)
        ci = M.bootstrap_ci(test_y, probs, [groups[i] for i in rows["test"]], mel, thr, n_boot=n_boot)
    return {"model": str(d["model"]), "embedding_dim": int(X.shape[1]), "C": c, "val_balanced_accuracy": val_bal,
            "temperature": T, "melanoma_threshold": thr, "test": rep, "confidence_intervals_95": ci,
            "n_train": int(len(rows["train"])), "n_val": int(len(rows["val"])), "n_test": int(len(rows["test"]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--splits-file", default="splits.json")
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--data-dir", default=os.environ.get("SKINSENSE_DATA", "dataset"))
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--out", default=None, help="write the JSON result here (default: next to the embeddings)")
    args = ap.parse_args()

    res = run(args.emb, args.splits_file, args.metadata, args.data_dir, n_boot=args.bootstrap)
    out = Path(args.out or Path(args.emb).with_suffix(".result.json"))
    out.write_text(json.dumps(res, indent=2))
    t, r = res["test"], res["test"]["melanoma_rule"]
    print(f"{res['model']} ({res['embedding_dim']}-d, C={res['C']})")
    print(f"TEST accuracy {t['accuracy']:.4f} | balanced acc {t['balanced_accuracy']:.4f} | macro-F1 {t['macro_f1']:.4f} | "
          f"macro-AUC {t['macro_auc_ovr']:.4f} | melanoma AUC {t['per_class']['Melanoma']['auc']:.4f}")
    print(f"melanoma rule @ {res['melanoma_threshold']:.4f}: sensitivity {r['sensitivity']:.3f}, specificity {r['specificity']:.3f}")
    if res["confidence_intervals_95"]:
        print("95% CI: " + " | ".join(f"{k} {v['low']:.3f}-{v['high']:.3f}" for k, v in res["confidence_intervals_95"].items()))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
