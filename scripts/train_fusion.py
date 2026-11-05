"""Train and honestly evaluate the age/sex/body-site fusion model.

    python scripts/train_fusion.py --outputs runs/enh/notta2/outputs.npz --metadata dataset/HAM10000_metadata.csv

Three variants are scored on the held-out TEST split:
  base       the calibrated image model alone (what ships today)
  recal      logistic regression on the 7 log-probabilities only  (isolates the effect of re-calibration)
  fusion     recal + age, sex, body site                           (the candidate)
Training uses only the VALIDATION split (grouped 5-fold CV by lesion to pick the regularisation and to get
out-of-fold probabilities for the melanoma threshold). Differences are tested with a paired lesion-level bootstrap.
Pass --write backend/fusion.json to export the fusion model; do that ONLY if the report says it helped.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
from backend import metrics as M  # noqa: E402
from backend.data_utils import find_metadata_csv, load_groups, load_or_make_splits  # noqa: E402
from backend.fusion import build_features  # noqa: E402

EXPORT_VARIANT = "offset_ctx_only"      # the honest one: context effect only, no class-frequency shift
SITES = ["abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", "hand", "lower extremity",
         "neck", "scalp", "trunk", "unknown", "upper extremity"]


def load_context(csv_path, stems):
    rows = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                age = float(r["age"])
            except (ValueError, KeyError):
                age = np.nan
            rows[r["image_id"]] = (age, r.get("sex", ""), r.get("localization", ""))
    age = np.array([rows[s][0] for s in stems])
    return age, [rows[s][1] for s in stems], [rows[s][2] for s in stems]


def fit_logreg(X, y, C, seed=0):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=seed).fit(X, y)


def full_proba(clf, X, k):
    p = np.zeros((len(X), k))
    p[:, clf.classes_] = clf.predict_proba(X)
    return p


def fit_offset(X, base_lp, y, lam, bias=True):
    """softmax(base_lp + X W^T + b), L2 penalty on W only. The image model is a fixed input (coefficient 1)."""
    from scipy.optimize import minimize
    n, f = X.shape
    k = base_lp.shape[1]

    def fun(w):
        W, b = w[:k * f].reshape(k, f), w[k * f:]
        z = base_lp + X @ W.T + b
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        g = np.exp(logp)
        g[np.arange(n), y] -= 1.0
        loss = -logp[np.arange(n), y].sum() + 0.5 * lam * (W ** 2).sum()
        gb = g.sum(0) if bias else np.zeros(k)          # bias=False: no class-frequency (prior) adjustment
        return loss, np.concatenate([(g.T @ X + lam * W).ravel(), gb])

    res = minimize(fun, np.zeros(k * f + k), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x[:k * f].reshape(k, f), res.x[k * f:]


def offset_proba(W, b, X, base_lp):
    z = base_lp + X @ W.T + b
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def cv_offset(X, base_lp, y, groups, lam_grid, bias=True):
    from sklearn.model_selection import GroupKFold
    best = None
    for lam in lam_grid:
        oof = np.zeros((len(X), base_lp.shape[1]))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            W, b = fit_offset(X[tr], base_lp[tr], y[tr], lam, bias)
            oof[te] = offset_proba(W, b, X[te], base_lp[te])
        nll = -np.log(np.clip(oof[np.arange(len(y)), y], 1e-12, 1)).mean()
        if best is None or nll < best[0]:
            best = (nll, lam, oof)
    return best


def cv_select(X, y, groups, k, c_grid):
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    best = None
    for C in c_grid:
        oof = np.zeros((len(X), k))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            oof[te] = full_proba(fit_logreg(X[tr], y[tr], C), X[te], k)
        score = balanced_accuracy_score(y, oof.argmax(1))
        if best is None or score > best[0]:
            best = (score, C, oof)
    return best


def paired_bootstrap(y, p_a, p_b, groups, n_boot=1000, seed=0):
    """95% CI of metric(b) - metric(a) over lesion resamples (b = candidate, a = reference)."""
    preds_a, preds_b = p_a.argmax(1), p_b.argmax(1)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(inv.max() + 1)]
    rng = np.random.default_rng(seed)
    d_bal, d_f1 = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        _, ba, fa = M._cm_stats(y[idx], preds_a[idx], p_a.shape[1])
        _, bb, fb = M._cm_stats(y[idx], preds_b[idx], p_b.shape[1])
        d_bal.append(bb - ba)
        d_f1.append(fb - fa)
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"balanced_accuracy_diff": ci(d_bal), "macro_f1_diff": ci(d_f1)}


def run(outputs_path, data_dir, metadata, splits_file, seed=42, target_sensitivity=0.90,
        c_grid=(0.01, 0.03, 0.1, 0.3, 1.0), lam_grid=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0), n_boot=1000):
    d = np.load(outputs_path)
    T = float(d["temperature"])
    splits, class_names, samples, _t, _i = load_or_make_splits(data_dir, metadata, seed, splits_file)
    mel = class_names.index("Melanoma")
    k = len(class_names)
    csv_path = find_metadata_csv(data_dir, metadata)
    groups_all, grouped = load_groups(samples, csv_path)
    stems = lambda idx: [Path(samples[i][0]).stem for i in idx]

    val_idx, test_idx = d["val_idx"], d["test_idx"]
    val_y, test_y = d["val_y"], d["test_y"]
    val_lp = np.log(np.clip(M.softmax(d["val_logits"], T), 1e-9, 1))
    test_lp = np.log(np.clip(M.softmax(d["test_logits"], T), 1e-9, 1))
    v_age, v_sex, v_site = load_context(csv_path, stems(val_idx))
    t_age, t_sex, t_site = load_context(csv_path, stems(test_idx))
    val_groups = [groups_all[i] for i in val_idx]
    test_groups = [groups_all[i] for i in test_idx]

    age_mean = float(np.nanmean(v_age))
    age_std = float(np.nanstd(v_age)) or 1.0
    spec = {"age_mean": age_mean, "age_std": age_std, "sites": SITES}
    nan = lambda n: np.full(n, np.nan)

    variants = {
        "recal": (build_features(val_lp, nan(len(val_y)), [None] * len(val_y), [None] * len(val_y), spec),
                  build_features(test_lp, nan(len(test_y)), [None] * len(test_y), [None] * len(test_y), spec)),
        "fusion": (build_features(val_lp, v_age, v_sex, v_site, spec),
                   build_features(test_lp, t_age, t_sex, t_site, spec)),
    }
    base_probs = M.softmax(d["test_logits"], T)
    thr_base, _, _ = M.pick_sensitivity_threshold(M.softmax(d["val_logits"], T)[:, mel], val_y == mel, target_sensitivity)
    results = {"base": {"probs": base_probs, "thr": thr_base, "C": None}}

    models = {}
    for name, (Xv, Xt) in variants.items():
        score, C, oof = cv_select(Xv, val_y, val_groups, k, c_grid)
        thr, _, _ = M.pick_sensitivity_threshold(oof[:, mel], val_y == mel, target_sensitivity)
        clf = fit_logreg(Xv, val_y, C)
        results[name] = {"probs": full_proba(clf, Xt, k), "thr": thr, "C": C, "cv_balanced_accuracy": score}
        models[name] = clf

    meta_v, meta_t = variants["fusion"][0][:, 7:], variants["fusion"][1][:, 7:]
    zero_v, zero_t = np.zeros((len(val_y), 1)), np.zeros((len(test_y), 1))
    offset_models = {}
    center = meta_v.mean(axis=0)                     # centred so "an average patient" gets no correction
    variants_off = {"offset_bias": (zero_v, zero_t, True, None),
                    "offset_meta": (meta_v, meta_t, True, None),
                    "offset_ctx_only": (meta_v - center, meta_t - center, False, center)}
    for name, (Xv, Xt, bias, ctr) in variants_off.items():
        nll, lam, oof = cv_offset(Xv, val_lp, val_y, val_groups, lam_grid, bias)
        thr, _, _ = M.pick_sensitivity_threshold(oof[:, mel], val_y == mel, target_sensitivity)
        W, b = fit_offset(Xv, val_lp, val_y, lam, bias)
        results[name] = {"probs": offset_proba(W, b, Xt, test_lp), "thr": thr, "C": lam, "cv_nll": nll}
        if ctr is not None:                          # absorb the centring into the intercept so the runtime needs no change
            b = b - ctr @ W.T
        offset_models[name] = (W, b)

    report = {}
    for name, r in results.items():
        rep = M.classification_report(test_y, r["probs"], class_names)
        rep["melanoma_rule"] = M.sensitivity_specificity(r["probs"][:, mel], test_y == mel, r["thr"])
        ci = M.bootstrap_ci(test_y, r["probs"], test_groups, mel, r["thr"], n_boot=min(n_boot, 500))
        report[name] = {"C": r["C"], "melanoma_threshold": r["thr"], "metrics": rep, "ci": ci}
    deltas = {
        "ctx_only_vs_base": paired_bootstrap(test_y, results["base"]["probs"], results["offset_ctx_only"]["probs"], test_groups, n_boot),
        "offset_meta_vs_base": paired_bootstrap(test_y, results["base"]["probs"], results["offset_meta"]["probs"], test_groups, n_boot),
        "offset_meta_vs_offset_bias": paired_bootstrap(test_y, results["offset_bias"]["probs"], results["offset_meta"]["probs"], test_groups, n_boot),
        "offset_bias_vs_base": paired_bootstrap(test_y, results["base"]["probs"], results["offset_bias"]["probs"], test_groups, n_boot),
    }
    # export the offset model in the runtime's generic form: identity on the 7 log-probabilities + learned correction
    export_name = EXPORT_VARIANT
    W, b = offset_models[export_name]
    coef = np.hstack([np.eye(k), W])
    export = {"version": 1, "classes": class_names, **spec, "lambda": results[export_name]["C"], "variant": export_name,
              "melanoma_threshold": float(results[export_name]["thr"]), "coef": coef.tolist(), "intercept": b.tolist(),
              "source": "offset model fitted on the validation split by scripts/train_fusion.py"}
    return {"grouped_by_lesion": grouped, "variants": report, "deltas": deltas, "export": export,
            "n_val": int(len(val_y)), "n_test": int(len(test_y))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", required=True, help="outputs.npz written by evaluate.py")
    ap.add_argument("--data-dir", default=os.environ.get("SKINSENSE_DATA", "dataset"))
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--splits-file", default="splits.json")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default="runs/enh/fusion_report.json")
    ap.add_argument("--write", default=None, help="export the fusion model (e.g. backend/fusion.json)")
    args = ap.parse_args()

    res = run(args.outputs, args.data_dir, args.metadata, args.splits_file, n_boot=args.bootstrap)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({k: v for k, v in res.items() if k != "export"}, indent=2))
    print(f"{'variant':8} {'acc':>7} {'bal.acc':>8} {'macroF1':>8} {'melAUC':>7} {'mel sens':>9} {'mel spec':>9}")
    for name, v in res["variants"].items():
        m, r = v["metrics"], v["metrics"]["melanoma_rule"]
        print(f"{name:8} {m['accuracy']:7.3f} {m['balanced_accuracy']:8.3f} {m['macro_f1']:8.3f} "
              f"{m['per_class']['Melanoma']['auc']:7.3f} {r['sensitivity']:9.3f} {r['specificity']:9.3f}")
    for k, v in res["deltas"].items():
        print(f"{k}: balanced-acc difference 95% CI {v['balanced_accuracy_diff'][0]:+.3f}..{v['balanced_accuracy_diff'][1]:+.3f}"
              f" | macro-F1 {v['macro_f1_diff'][0]:+.3f}..{v['macro_f1_diff'][1]:+.3f}")
    if args.write:
        Path(args.write).write_text(json.dumps(res["export"], indent=2))
        print(f"Exported {args.write}")


if __name__ == "__main__":
    main()
