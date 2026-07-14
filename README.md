# Diabetic Retinopathy Screening (Intermediate Project — Task 5)

A CNN-based classifier that grades diabetic retinopathy (DR) severity
from retinal fundus images, using transfer learning from an
ImageNet-pretrained EfficientNetB0 backbone, with Grad-CAM
interpretability.

## ⚠️ Important: no dataset is included

Real fundus-image datasets (APTOS 2019, IDRiD, Messidor) are hosted on
Kaggle/research sites and are **not bundled with this repo** — you need
to download one yourself (free, ~1-9GB depending on dataset). See
"Getting real data" below.

To prove the pipeline itself is correct and bug-free without needing a
real dataset first, `generate_smoke_test_data.py` builds a tiny
synthetic "fundus-like" image set, and `smoke_test_proof/` contains the
actual output of running the full pipeline on it (training curves,
confusion matrix, ROC curve, Grad-CAM overlays). **These smoke-test
results are meaningless medically** (synthetic images, tiny sample,
trained from random weights, near chance-level accuracy by design) —
they only confirm every stage of the code runs without errors:
preprocessing → model build → training → evaluation → Grad-CAM.

## Pipeline

1. **Fundus-specific preprocessing** — circle-crop (removes black
   borders around the circular fundus image) + CLAHE contrast
   enhancement (boosts visibility of lesions/hemorrhages) + resize
2. **Data augmentation** — flips, rotation, zoom, contrast jitter
   (fundus images are rotation-invariant, unlike natural photos)
3. **Transfer learning** — EfficientNetB0 pretrained on ImageNet;
   phase 1 trains a new classification head with the backbone frozen,
   phase 2 fine-tunes the top 30 backbone layers at a low learning rate
4. **Class weighting** — DR datasets are heavily imbalanced toward
   "No DR"; class weights correct for this during training
5. **Evaluation** — classification report (precision/recall/F1 per
   severity grade), confusion matrix, quadratic weighted kappa (the
   standard DR-grading metric used in the APTOS Kaggle competition),
   and ROC/AUC for the clinically meaningful binary split ("referable"
   DR — Moderate/Severe/Proliferate — vs "non-referable" — No/Mild)
6. **Grad-CAM** — visualizes which retinal regions drove each
   prediction, for model interpretability/trust

## Getting real data

Pick one:
- **Kaggle: "APTOS 2019 Blindness Detection"** — the standard benchmark
- **Kaggle: "Diabetic Retinopathy 224x224 Gaussian Filtered"** — already
  preprocessed and folder-per-class, easiest to plug in directly

Then arrange it as:

```
data/
├── train/
│   ├── No_DR/            *.jpg or *.png
│   ├── Mild/
│   ├── Moderate/
│   ├── Severe/
│   └── Proliferate_DR/
└── val/                  (same 5 subfolders)
```

(If your chosen dataset ships as a single CSV + image folder instead of
folder-per-class, sort images into the structure above first — a
one-off script using `pandas` + `shutil.copy` based on the label column
will do it.)

## How to run

```bash
pip install -r requirements.txt

# OPTIONAL - smoke-test the code path with synthetic data first:
python generate_smoke_test_data.py --out_dir data --n_per_class 12
python diabetic_retinopathy_detection.py --data_dir data --img_size 96 \
    --batch_size 4 --epochs 2 --fine_tune_epochs 1 --weights none

# REAL run, once you've downloaded a real dataset into data/:
python diabetic_retinopathy_detection.py --data_dir data --epochs 15
```

`--weights imagenet` (the default) requires internet access to download
pretrained weights the first time; use `--weights none` only to sanity
check the code without training a useful model.

## Project structure

```
.
├── diabetic_retinopathy_detection.py   # full pipeline
├── generate_smoke_test_data.py         # synthetic data for code testing
├── smoke_test_proof/                   # proof the pipeline runs (NOT real results)
├── requirements.txt
└── README.md
```

`data/`, `figures/`, and `*.keras` are gitignored — they're generated
locally when you run the scripts (not meaningful to commit, and can get
large with a real dataset).

## Honest limitations of this smoke test

- Trained from random weights (`--weights none`), not real ImageNet
  transfer learning — pretrained weight downloads weren't reachable
  from the sandbox this was built in, but `--weights imagenet` (the
  script default) will work fine on a normal machine with internet
- 12 images per class, 96×96 resolution, 3 total epochs — enough to
  prove the code runs, nowhere near enough to learn anything real
- Accuracy near chance level is *expected and correct* for this setup

## Tech used

Python, TensorFlow/Keras (EfficientNetB0 transfer learning, Grad-CAM),
OpenCV, numpy, pandas, matplotlib, seaborn, scikit-learn
