"""Honest evaluation + calibration.

    python evaluate.py --data-dir dataset --metadata HAM10000_metadata.csv

1. Runs the trained model on the VALIDATION and held-out TEST splits (splits.json).
2. Fits the temperature on the validation split only.
3. Picks the melanoma-escalation threshold on the validation split for the
   requested sensitivity (default 90 %).
4. Reports everything on the TEST split: accuracy, balanced accuracy, per-class
   precision/recall/F1/AUC, confusion matrix, calibration error, and the
   melanoma rule's sensitivity/specificity.
5. Writes reports/evaluation.json, reports/evaluation.md, reports/confusion_matrix.png
   and backend/calibration.json (read by the API at start-up).
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend import metrics as M  # noqa: E402
from backend.data_utils import find_metadata_csv, load_groups, load_or_make_splits, make_subsets  # noqa: E402
from backend.model import get_model  # noqa: E402


def find_weights(explicit):
    if explicit:
        return explicit
    for name in ("skin_sense_ensemble_best.pth", "skin_sense_ensemble_final.pth"):
        p = Path("backend") / name
        if p.exists():
            return str(p)
    raise SystemExit("No weights found. Pass --weights path/to/model.pth")


def save_confusion_png(cm, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Row-normalised")
    ax.set_title("Confusion matrix (held-out test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.environ.get("SKINSENSE_DATA", "dataset"))
    p.add_argument("--metadata", default=None)
    p.add_argument("--splits-file", default="splits.json")
    p.add_argument("--weights", default=None)
    p.add_argument("--target-sensitivity", type=float, default=0.90,
                   help="melanoma sensitivity the escalation threshold must reach on the validation set")
    p.add_argument("--min-confidence", type=float, default=0.35)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--calibration-out", default="backend/calibration.json")
    p.add_argument("--no-write-calibration", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tta", action="store_true", help="test-time augmentation: average over 4 flip views (4x slower)")
    p.add_argument("--bootstrap", type=int, default=1000, help="lesion-level bootstrap resamples for 95%% CIs (0 = off)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, names, _s, _t, info = load_or_make_splits(args.data_dir, args.metadata, args.seed, args.splits_file)
    _train, val_ds, test_ds = make_subsets(args.data_dir, splits)
    weights = find_weights(args.weights)
    print(f"Weights: {weights} | device: {device} | val={len(val_ds)} test={len(test_ds)} | "
          f"lesion-grouped: {info['grouped_by_lesion']}")

    model = get_model(num_classes=len(names), checkpoint_path=weights).to(device)
    mk = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    val_logits, val_parts, val_y = M.collect_outputs(model, mk(val_ds), device, tta=args.tta)
    test_logits, test_parts, test_y = M.collect_outputs(model, mk(test_ds), device, tta=args.tta)

    mel = names.index("Melanoma")
    T = M.fit_temperature(val_logits, val_y)
    val_probs = M.softmax(val_logits, T)
    if not (val_y == mel).any() or not (test_y == mel).any():
        raise SystemExit("The validation or test split contains no melanoma images, so the melanoma "
                         "threshold cannot be chosen or checked. Use the full dataset (or rebuild "
                         "splits with --seed / delete splits.json).")
    thr, val_sens, val_spec = M.pick_sensitivity_threshold(
        val_probs[:, mel], val_y == mel, args.target_sensitivity)

    report = {}
    for label, temp in (("uncalibrated", 1.0), ("temperature_scaled", T)):
        probs = M.softmax(test_logits, temp)
        rep = M.classification_report(test_y, probs, names)
        rep["melanoma_rule"] = M.sensitivity_specificity(probs[:, mel], test_y == mel, thr) if label == "temperature_scaled" else None
        report[label] = rep

    # ---- uncertainty / abstention, confidence intervals -------------------------------------------
    test_probs = M.softmax(test_logits, T)
    u_conf = 1.0 - test_probs.max(axis=1)
    u_dis = M.disagreement(test_parts)
    unc = {"confidence (1 - max prob)": u_conf, "network disagreement": u_dis,
           "combined": u_conf + u_dis}
    uncertainty = {
        name: {"error_detection_auroc": M.error_detection_auroc(test_y, test_probs, u),
               "risk_coverage": M.risk_coverage(test_y, test_probs, u, mel)}
        for name, u in unc.items()
    }
    ci = None
    if args.bootstrap > 0:
        csv_path = find_metadata_csv(args.data_dir, args.metadata)
        groups_all, _ = load_groups(_s, csv_path)
        test_groups = [groups_all[i] for i in splits["test"]]
        ci = M.bootstrap_ci(test_y, test_probs, test_groups, mel, thr, n_boot=args.bootstrap)

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    result = {
        "tta": bool(args.tta),
        "uncertainty": uncertainty,
        "confidence_intervals_95": ci,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "weights": weights,
        "lesion_grouped_split": info["grouped_by_lesion"],
        "temperature": T,
        "melanoma_threshold": thr,
        "validation_melanoma_rule": {"sensitivity": val_sens, "specificity": val_spec,
                                     "target_sensitivity": args.target_sensitivity},
        "test": report,
    }
    (out / "evaluation.json").write_text(json.dumps(result, indent=2))
    # raw outputs, so later analyses (metadata fusion, calibration studies) need no second GPU pass
    np.savez_compressed(out / "outputs.npz",
                        val_logits=val_logits, val_parts=val_parts, val_y=val_y, val_idx=np.array(splits["val"]),
                        test_logits=test_logits, test_parts=test_parts, test_y=test_y, test_idx=np.array(splits["test"]),
                        temperature=T, melanoma_threshold=thr, tta=bool(args.tta))
    save_confusion_png(report["temperature_scaled"]["confusion_matrix"], names, out / "confusion_matrix.png")

    ts = report["temperature_scaled"]
    lines = [
        "# SkinSense evaluation (held-out test split)", "",
        f"- Date: {result['date']}", f"- Weights: `{weights}`",
        f"- Lesion-grouped split: **{info['grouped_by_lesion']}**"
        + ("" if info["grouped_by_lesion"] else "  (**no metadata CSV - results are optimistic**)"),
        f"- Test images: {ts['n_samples']}", "",
        "| Metric | Uncalibrated | Temperature-scaled |", "|---|---|---|",
    ]
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc_ovr", "ece", "nll"):
        lines.append(f"| {key} | {report['uncalibrated'][key]:.4f} | {ts[key]:.4f} |")
    lines += ["", f"Temperature T = {T:.3f}", "", "## Per class (temperature-scaled)", "",
              "| Class | Precision | Recall | F1 | AUC | Support |", "|---|---|---|---|---|---|"]
    for n, m in ts["per_class"].items():
        auc = "-" if m["auc"] is None else f"{m['auc']:.3f}"
        lines.append(f"| {n} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {auc} | {m['support']} |")
    rule = ts["melanoma_rule"]
    lines += ["", "## Melanoma escalation rule", "",
              f"Flag when P(melanoma) >= **{thr:.4f}** (chosen on validation for {args.target_sensitivity:.0%} sensitivity).", "",
              f"- Validation: sensitivity {val_sens:.3f}, specificity {val_spec:.3f}",
              f"- **Test**: sensitivity {rule['sensitivity']:.3f}, specificity {rule['specificity']:.3f} "
              f"({rule['n_positive']} melanomas, {rule['n_negative']} others)", "",
              ""]
    lines.append(f"Test-time augmentation: **{'on' if args.tta else 'off'}**")
    if ci:
        lines += ["", f"## 95% confidence intervals (lesion-level bootstrap, {args.bootstrap} resamples)", "",
                  "| Metric | Low | High |", "|---|---|---|"]
        lines += [f"| {k} | {v['low']:.3f} | {v['high']:.3f} |" for k, v in ci.items()]
    lines += ["", "## Uncertainty and abstention", "",
              "Keeping only the most certain cases (abstaining on the rest). Higher accuracy at lower coverage means the "
              "uncertainty score is useful.", ""]
    for name, u in uncertainty.items():
        lines += [f"**{name}** - AUROC for spotting wrong predictions: {u['error_detection_auroc']:.3f}", "",
                  "| Coverage | Accuracy | Balanced acc. | Melanomas still covered |", "|---|---|---|---|"]
        lines += [f"| {r['coverage']:.0%} | {r['accuracy']:.1%} | {r['balanced_accuracy']:.1%} | {r['melanomas_kept']:.0%} |"
                  for r in u["risk_coverage"]]
        lines.append("")
    lines += ["![confusion matrix](confusion_matrix.png)", ""]
    (out / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")

    if not args.no_write_calibration:
        Path(args.calibration_out).write_text(json.dumps({
            "temperature": round(T, 4),
            "melanoma_threshold": round(thr, 4),
            "min_confidence": args.min_confidence,
            "source": f"fitted on validation split by evaluate.py on {result['date']}",
            "metrics": {"test_balanced_accuracy": round(ts["balanced_accuracy"], 4),
                        "test_melanoma_sensitivity": round(rule["sensitivity"], 4),
                        "test_melanoma_specificity": round(rule["specificity"], 4)},
        }, indent=2))
        print(f"Wrote {args.calibration_out}")

    print(f"\nTEST  accuracy {ts['accuracy']:.4f} | balanced acc {ts['balanced_accuracy']:.4f} | "
          f"macro-F1 {ts['macro_f1']:.4f} | AUC {ts['macro_auc_ovr']:.4f} | ECE {ts['ece']:.4f}")
    print(f"      melanoma rule @ {thr:.4f}: sensitivity {rule['sensitivity']:.3f}, specificity {rule['specificity']:.3f}")
    if ci:
        print("95% CI: " + " | ".join(f"{k} {v['low']:.3f}-{v['high']:.3f}" for k, v in ci.items()))
    for name, u in uncertainty.items():
        r = {x['coverage']: x for x in u['risk_coverage']}
        print(f"uncertainty [{name}]: error-detection AUROC {u['error_detection_auroc']:.3f} | "
              f"accuracy at 100/80/50% coverage: {r[1.0]['accuracy']:.1%} / {r[0.8]['accuracy']:.1%} / {r[0.5]['accuracy']:.1%}")
    print(f"Report written to {out}/")


if __name__ == "__main__":
    main()
