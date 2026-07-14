"""
Diabetic Retinopathy Screening - CNN Classifier
=================================================
Task 5 (Intermediate Project): Build a CNN-based classifier to detect
diabetic retinopathy (DR) from retinal fundus images, using transfer
learning from an ImageNet-pretrained backbone.

Expected data layout (5-class DR severity grading - the standard
formulation used by the APTOS 2019 / "Diabetic Retinopathy 224x224
Gaussian Filtered" datasets):

    data_dir/
        train/
            No_DR/          *.png or *.jpg
            Mild/
            Moderate/
            Severe/
            Proliferate_DR/
        val/                (same 5 subfolders)
        test/               (same 5 subfolders, optional)

Where to get real data (not included in this repo - see README):
    - Kaggle: "APTOS 2019 Blindness Detection"
    - Kaggle: "Diabetic Retinopathy 224x224 Gaussian Filtered"
    - IDRiD / Messidor-2 (research use)

Run:
    python diabetic_retinopathy_detection.py --data_dir data --epochs 15

Outputs:
    - figures/ - training curves, confusion matrix, ROC curves, Grad-CAM
    - dr_model.keras - trained model
    - analysis_summary.md
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    cohen_kappa_score
)
from sklearn.utils.class_weight import compute_class_weight

sns.set_theme(style="whitegrid")
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
REFERABLE_CLASSES = {"Moderate", "Severe", "Proliferate_DR"}  # clinically "refer to specialist"


def savefig(fig_dir, name):
    path = os.path.join(fig_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved -> {path}")


# ---------------------------------------------------------------------
# Fundus-specific image preprocessing
# ---------------------------------------------------------------------
def circle_crop(img):
    """Crop to the circular fundus region and remove black borders."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > 10
    if mask.sum() == 0:
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def enhance_contrast_clahe(img):
    """CLAHE on the L channel (LAB space) - standard fundus-image trick
    to boost contrast of lesions/hemorrhages without blowing out color."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_fundus_image(path, img_size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = circle_crop(img)
    img = enhance_contrast_clahe(img)
    img = cv2.resize(img, (img_size, img_size))
    return img


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def build_dataframe(split_dir):
    rows = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append({"path": os.path.join(cls_dir, fname), "label": cls})
    return pd.DataFrame(rows)


def make_dataset(df, img_size, batch_size, augment=False, shuffle=False):
    label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    paths = df["path"].values
    labels = df["label"].map(label_to_idx).values

    def _load(path, label):
        def _cv_load(p):
            p = p.numpy().decode("utf-8")
            img = preprocess_fundus_image(p, img_size)
            return img.astype(np.float32)
        img = tf.py_function(_cv_load, [path], tf.float32)
        img.set_shape([img_size, img_size, 3])
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda x, y: (keras.applications.efficientnet.preprocess_input(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------
# Model - transfer learning with EfficientNetB0
# ---------------------------------------------------------------------
def build_model(img_size, num_classes, weights="imagenet"):
    try:
        base = keras.applications.EfficientNetB0(
            include_top=False, weights=weights,
            input_shape=(img_size, img_size, 3)
        )
    except Exception as e:
        print(f"  [!] Could not load '{weights}' weights ({e}). "
              f"Falling back to random initialization (weights=None). "
              f"This will train from scratch, not via transfer learning - "
              f"only expected in network-restricted environments.")
        base = keras.applications.EfficientNetB0(
            include_top=False, weights=None,
            input_shape=(img_size, img_size, 3)
        )

    base.trainable = False  # phase 1: frozen backbone

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base


# ---------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------
def build_gradcam_model(model, base_model, img_size):
    """Rebuilds a fresh (input -> [base_output, final_output]) graph that
    reuses the same layer objects (and therefore the same trained
    weights) as `model`. Directly slicing tensors out of an existing
    nested Functional model (e.g. `keras.Model(model.inputs,
    [model.get_layer(base.name).output, model.output])`) raises a
    'not connected to inputs' error in Keras 3 when a submodel was
    called as a layer - rebuilding the graph from scratch avoids it."""
    grad_input = keras.Input(shape=(img_size, img_size, 3))
    conv_output = base_model(grad_input)
    h = conv_output
    started = False
    for layer in model.layers:
        if layer is base_model:
            started = True
            continue
        if not started:
            continue
        h = layer(h)
    return keras.Model(grad_input, [conv_output, h])


def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_gradcam_overlay(orig_img, heatmap, out_path):
    heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * orig_img + 0.4 * heatmap_color).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Diabetic Retinopathy CNN classifier")
    parser.add_argument("--data_dir", default="data",
                         help="Folder containing train/ (and val/, test/) subfolders")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine_tune_epochs", type=int, default=5)
    parser.add_argument("--weights", default="imagenet",
                         help="'imagenet' (default) or 'none' to train from scratch")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    weights = None if args.weights.lower() == "none" else args.weights

    # -------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")

    train_df = build_dataframe(train_dir)
    val_df = build_dataframe(val_dir) if os.path.isdir(val_dir) else pd.DataFrame()
    test_df = build_dataframe(test_dir) if os.path.isdir(test_dir) else pd.DataFrame()

    if len(train_df) == 0:
        raise SystemExit(
            f"No training images found under {train_dir}/<class_name>/*.jpg\n"
            f"Expected class folders: {CLASS_NAMES}\n"
            f"See README.md for how to download a real dataset."
        )

    print(f"Train images: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("\nClass distribution (train):")
    print(train_df["label"].value_counts())

    eval_df = val_df if len(val_df) else test_df

    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: BUILD tf.data PIPELINES")
    print("=" * 70)
    train_ds = make_dataset(train_df, args.img_size, args.batch_size,
                             augment=True, shuffle=True)
    eval_ds = None
    if len(eval_df):
        eval_ds = make_dataset(eval_df, args.img_size, args.batch_size)
    print("Datasets ready (fundus preprocessing: circle-crop + CLAHE + resize).")

    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: CLASS WEIGHTS (handle DR class imbalance)")
    print("=" * 70)
    label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    y_train_idx = train_df["label"].map(label_to_idx).values
    present_classes = np.unique(y_train_idx)
    weights_arr = compute_class_weight("balanced", classes=present_classes, y=y_train_idx)
    class_weight = {int(c): float(w) for c, w in zip(present_classes, weights_arr)}
    print(f"Class weights: {class_weight}")

    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: BUILD MODEL (EfficientNetB0 transfer learning)")
    print("=" * 70)
    model, base = build_model(args.img_size, len(CLASS_NAMES), weights=weights)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True,
                                       monitor="val_loss" if eval_ds else "loss"),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5,
                                           monitor="val_loss" if eval_ds else "loss"),
    ]

    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: TRAIN - PHASE 1 (frozen backbone, head only)")
    print("=" * 70)
    history1 = model.fit(
        train_ds, validation_data=eval_ds, epochs=args.epochs,
        class_weight=class_weight, callbacks=callbacks, verbose=2
    )

    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: TRAIN - PHASE 2 (fine-tune top backbone layers)")
    print("=" * 70)
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    history2 = model.fit(
        train_ds, validation_data=eval_ds, epochs=args.fine_tune_epochs,
        class_weight=class_weight, callbacks=callbacks, verbose=2
    )

    # Combined training curves
    acc = history1.history.get("accuracy", []) + history2.history.get("accuracy", [])
    loss = history1.history.get("loss", []) + history2.history.get("loss", [])
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="train acc")
    if eval_ds:
        val_acc = history1.history.get("val_accuracy", []) + history2.history.get("val_accuracy", [])
        plt.plot(val_acc, label="val acc")
    plt.axvline(len(history1.history.get("accuracy", [])), color="gray", linestyle="--",
                label="fine-tune starts")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="train loss")
    if eval_ds:
        val_loss = history1.history.get("val_loss", []) + history2.history.get("val_loss", [])
        plt.plot(val_loss, label="val loss")
    plt.axvline(len(history1.history.get("loss", [])), color="gray", linestyle="--",
                label="fine-tune starts")
    plt.title("Loss")
    plt.legend()
    savefig(args.output_dir, "01_training_curves.png")

    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7: EVALUATION")
    print("=" * 70)
    if eval_ds is None:
        print("No val/ or test/ folder found - skipping quantitative evaluation.")
        model.save("dr_model.keras")
        return

    y_true, y_pred_proba = [], []
    for x_batch, y_batch in eval_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred_proba.append(preds)
        y_true.append(y_batch.numpy())
    y_true = np.concatenate(y_true)
    y_pred_proba = np.concatenate(y_pred_proba)
    y_pred = y_pred_proba.argmax(axis=1)

    present = sorted(set(y_true) | set(y_pred))
    target_names_present = [CLASS_NAMES[i] for i in present]
    print(classification_report(y_true, y_pred, labels=present,
                                 target_names=target_names_present, zero_division=0))

    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    print(f"Quadratic weighted kappa (standard DR-grading metric): {kappa:.3f}")

    cm = confusion_matrix(y_true, y_pred, labels=present)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names_present, yticklabels=target_names_present)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - DR Severity Grade")
    savefig(args.output_dir, "02_confusion_matrix.png")

    # Binary "referable DR" ROC (clinically: refer to specialist or not)
    y_true_ref = np.array([CLASS_NAMES[i] in REFERABLE_CLASSES for i in y_true]).astype(int)
    y_score_ref = y_pred_proba[:, [CLASS_NAMES.index(c) for c in REFERABLE_CLASSES]].sum(axis=1)
    if len(set(y_true_ref)) > 1:
        auc = roc_auc_score(y_true_ref, y_score_ref)
        fpr, tpr, _ = roc_curve(y_true_ref, y_score_ref)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"Referable DR (AUC={auc:.3f})", color="darkorange")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC - Referable DR (Moderate/Severe/Proliferate vs No/Mild)")
        plt.legend()
        savefig(args.output_dir, "03_roc_referable_dr.png")
        print(f"Referable-DR ROC AUC: {auc:.3f}")

    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 8: GRAD-CAM VISUALIZATION (model interpretability)")
    print("=" * 70)
    gradcam_model = build_gradcam_model(model, base, args.img_size)
    sample_rows = eval_df.sample(min(4, len(eval_df)), random_state=42)
    for i, row in enumerate(sample_rows.itertuples()):
        img = preprocess_fundus_image(row.path, args.img_size)
        arr = keras.applications.efficientnet.preprocess_input(
            img.astype(np.float32))[np.newaxis, ...]
        try:
            heatmap = make_gradcam_heatmap(arr, gradcam_model)
            out_path = os.path.join(args.output_dir, f"04_gradcam_{i}_{row.label}.png")
            save_gradcam_overlay(img, heatmap, out_path)
            print(f"  saved -> {out_path}")
        except Exception as e:
            print(f"  [!] Grad-CAM failed for {row.path}: {e}")

    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 9: SAVE MODEL & SUMMARY")
    print("=" * 70)
    model.save("dr_model.keras")
    print("Saved -> dr_model.keras")

    summary = f"""# Diabetic Retinopathy Screening - Summary

## Data
Train: {len(train_df)} images | Eval: {len(eval_df)} images
Classes: {CLASS_NAMES}

## Model
EfficientNetB0 (weights='{args.weights}'), transfer learning:
phase 1 frozen backbone ({len(history1.history.get('accuracy', []))} epochs),
phase 2 fine-tuned top 30 layers ({len(history2.history.get('accuracy', []))} epochs).

## Evaluation (severity grading, {len(present)} classes present in eval set)
See figures/02_confusion_matrix.png for the full breakdown.
Quadratic weighted kappa: {kappa:.3f}
{"Referable-DR ROC AUC: %.3f" % auc if len(set(y_true_ref)) > 1 else ""}

## Interpretability
Grad-CAM overlays in figures/04_gradcam_*.png show which retinal
regions most influenced each prediction (should highlight lesions /
hemorrhages / microaneurysms for a well-trained model on real data).

## Files
- `diabetic_retinopathy_detection.py` - full pipeline
- `dr_model.keras` - trained model
- `figures/` - training curves, confusion matrix, ROC, Grad-CAM
"""
    with open("analysis_summary.md", "w") as f:
        f.write(summary)
    print("Summary written to analysis_summary.md")
    print("DONE.")


if __name__ == "__main__":
    main()
