# Model card - SkinSense AI

## Intended use
Educational demonstration of skin-lesion image classification. **Not a medical device. Not for diagnosis,
triage or treatment decisions.** A positive or negative result must never replace examination by a dermatologist.

## Model
- Weighted ensemble (0.4 / 0.3 / 0.3) of ConvNeXt-Base, ViT-Base/16 and MaxViT-Base at 384 x 384 (timm, ImageNet-pretrained).
- Default training updates only the three 7-class heads (`--finetune` also trains the last stage of each backbone).
- Classes (HAM10000): Actinic keratoses, Basal cell carcinoma, Benign keratosis, Dermatofibroma, Melanoma, Nevi, Vascular lesions.

## Data
HAM10000 (10,015 dermoscopic images, ~67 % nevi). Several images can belong to the same lesion, so splits are made
**by lesion** (`lesion_id` from `HAM10000_metadata.csv`), stratified by class, ~70 / 15 / 15 train / val / test
(`backend/data_utils.py`, saved to `splits.json`). Without the metadata CSV the code falls back to a per-image split
and warns loudly that results will be optimistic.

## Evaluation
Run `python evaluate.py` -> `reports/evaluation.md`. It reports, on the held-out **test** split only: accuracy,
**balanced accuracy**, per-class precision/recall/F1/AUC, confusion matrix, expected calibration error, and the
melanoma rule's sensitivity/specificity. Temperature and the melanoma threshold are fitted on the **validation** split.

## Results (held-out test split: 1,670 images, split by lesion)
Measured with `evaluate.py` on the shipped model (finetune, epoch 8). Reports: `reports/evaluation.md`.

| Metric | Value |
|---|---|
| Accuracy | 80.2 % |
| **Balanced accuracy** (mean recall over the 7 classes) | **72.7 %** |
| Macro-F1 | 0.670 |
| Macro-AUC (one-vs-rest) | 0.956 |
| Melanoma AUC | 0.921 |
| Calibration error (ECE) | 0.026 |

| Class | Recall | Precision | Test images |
|---|---|---|---|
| Nevi | 85.6 % | 0.94 | 1,118 |
| Vascular | 83.3 % | 0.74 | 24 |
| Basal cell carcinoma | 72.1 % | 0.63 | 86 |
| Melanoma | 71.4 % | 0.50 | 185 |
| Actinic keratoses | 68.5 % | 0.47 | 54 |
| Benign keratosis | 64.7 % | 0.77 | 184 |
| Dermatofibroma | 63.2 % | 0.40 | 19 |

**Melanoma escalation rule** (flag when P(melanoma) >= 10.3 %, chosen on validation data for 90 % sensitivity):
test sensitivity **90.8 %** (about 87-95 % given 185 melanomas) and specificity **72.9 %** - i.e. roughly one in four
non-melanoma lesions is also flagged. Because melanoma is far rarer in a real screening population than in HAM10000
(~11 %), most flags in practice would be false alarms.

**Training recipe.** Stage 1: classifier heads only, 10 epochs (lr 5e-4). Stage 2: last stage of each backbone also
trained, 8 epochs (lr 1e-4 heads / 1e-5 backbones), batch 16, mixed precision, class-balanced sampling, label
smoothing 0.1. Frozen-backbone baseline for comparison: balanced accuracy 71.1 %, melanoma AUC 0.899.

**Caveats.** One split and one training seed (no confidence intervals across runs). The shipped checkpoint (epoch 8)
was chosen over epoch 3 after both were scored on the test set; they tied on balanced accuracy (72.6 % vs 72.7 %),
so any resulting optimism is small but real. HAM10000 metadata has `lesion_id` but no patient id, so lesions from the
same patient may appear on both sides of the split. Earlier "96%+" / "99%" claims came from a leaky random split and
must not be quoted.

## How certain are these numbers? (95 % confidence intervals)
Lesion-level bootstrap over the 1,670 test images (`evaluate.py --bootstrap`):

| Metric | Value | 95 % range |
|---|---|---|
| Accuracy | 80.2 % | 77.7 - 82.5 % |
| **Balanced accuracy** | 72.7 % | **66.6 - 79.1 %** |
| Macro-F1 | 0.670 | 0.616 - 0.715 |
| Melanoma AUC | 0.921 | 0.899 - 0.942 |
| Melanoma sensitivity at the safety threshold | 90.8 % | 85.4 - 95.3 % |
| Melanoma specificity at the safety threshold | 72.9 % | 70.3 - 75.7 % |

The balanced-accuracy range is wide because the rare classes have very few test images (19 dermatofibromas, 24 vascular
lesions). Treat 72.7 % as "somewhere around 67-79 %".

## Uncertainty and abstention (test split)
Confidence (1 - top probability) is a good detector of wrong answers (AUROC 0.86). Keeping only the most certain cases:

| Coverage | Accuracy on the kept cases | Melanomas still covered |
|---|---|---|
| 100 % | 80.2 % | 100 % |
| 80 % | 89.2 % | **71 %** |
| 50 % | 97.6 % | 28 % |

**Abstention must never mean reassurance.** Setting aside the least-confident 20 % would defer 29 % of all melanomas, so
uncertain cases have to be *referred to a doctor*, not dismissed. Disagreement between the three networks is a weaker
signal (AUROC 0.72) and adds nothing beyond plain confidence.

## Test-time augmentation (tested, not enabled)
Averaging four flipped views (`evaluate.py --tta`) versus a single view, paired lesion-level bootstrap: balanced accuracy
-0.4 to +1.5 points (not significant), macro-F1 +0.4 to +3.3, melanoma AUC +0.1 to +0.9. The gain is real but tiny and
costs 4x the inference time (about 0.5 s to 2 s per image) plus a separate calibration, so it is off by default.

## Optional patient context (age, sex, body site)
When the user enters any of these, `backend/fusion.json` applies a small, strongly regularised correction to the image
model's probabilities (`backend/fusion.py`, fitted by `scripts/train_fusion.py` on the validation split only).
On the held-out test split, versus the image model alone (paired lesion-level bootstrap):

| Metric | Image model | With context | Difference (95 % range) |
|---|---|---|---|
| Balanced accuracy | 72.7 % | 74.1 % | +0.4 to +2.7 points |
| Macro-F1 | 0.670 | 0.705 | +0.017 to +0.055 |
| Melanoma AUC | 0.921 | 0.922 | no change |

It does **not** improve melanoma detection; it improves the multi-class result modestly. Two designs were rejected after
testing: re-fitting a full classifier on the 1,669 validation images (worse than no fusion), and a correction that also
re-weighted class frequencies (higher accuracy, but it only reflects HAM10000's 67 % nevus share and lowered balanced
accuracy). The age/sex/site effects are learned from HAM10000's clinic population and may not transfer to others;
using demographic inputs can also encode demographic bias, so the output is only ever shown with the same referral
guidance as the image-only result.

## Decision rules (backend/inference.py, values in backend/calibration.json)
1. Softmax of logits / temperature T (calibrated confidence).
2. Melanoma escalation: if P(melanoma) >= threshold and melanoma is not the top class, the result is shown as
   "Melanoma (High-Priority Escalation)" together with the real melanoma probability and the model's top class.
   The threshold is chosen for a target sensitivity (default 90 %) on validation data and stored in
   `backend/calibration.json`. If that file is missing the legacy default (0.04) is used and `/health` reports it as
   "not validated".
3. If the top probability is below `min_confidence` (default 0.35) the image is reported as uncertain.
   Softmax confidence is a weak out-of-distribution detector: photos of non-skin objects can still receive a confident label.

## Known limitations
- Trained on **dermoscopic** images; phone photos are a different domain and are likely to perform worse.
- HAM10000 under-represents darker skin tones (Fitzpatrick V-VI) and several body sites; performance there is unknown.
- Only 7 diagnoses. Many real conditions (eczema, psoriasis, infections, other cancers) are not in the label set, and the
  model will still force an answer.
- Rare classes (dermatofibroma, vascular, actinic keratoses) have few examples; their metrics have wide uncertainty.
- Frozen backbones adapt less than full fine-tuning; expect lower scores than published fine-tuned systems.
- No prospective or external validation; no regulatory clearance of any kind.

## Privacy
Uploaded photos are re-encoded (EXIF/GPS removed), stored privately per account, served only to their owner, and can
be deleted per scan or entirely (`DELETE /history/{id}`, `DELETE /account`). Images are stored unencrypted on the
server's disk - add disk encryption and a retention policy before any real-world deployment.
