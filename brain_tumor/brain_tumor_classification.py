# =============================================================================
#  Brain Tumor Classification — CPU-Optimised Pipeline
#  Environment : VS Code | Local CPU (or GPU if available)
# =============================================================================
#
#  Speed vs Quality tradeoff switches are all in the Config class.
#  Default settings are tuned for fast CPU training (~3–5 min per model).
#  If you have a GPU, set USE_GPU_SETTINGS = True for best accuracy.
#
#  Folder layout (relative to this file):
#
#   project_root/
#   ├── brain_tumor_classification.py
#   ├── brain_tumor_notebook.ipynb
#   ├── app.py                          ← Streamlit (auto-generated)
#   ├── data/
#   │   ├── Training/  glioma/ meningioma/ notumor/ pituitary/
#   │   └── Testing/   glioma/ meningioma/ notumor/ pituitary/
#   └── outputs/                        ← created automatically
# =============================================================================

import os, sys, time, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
matplotlib.use("Agg")
matplotlib.rcParams["figure.dpi"] = 100

# ── Resolve project root ──────────────────────────────────────────────────────
try:
    ROOT_DIR = Path(__file__).resolve().parent
except NameError:
    ROOT_DIR = Path.cwd()


# =============================================================================
#  CONFIG
#  ──────
#  Toggle USE_GPU_SETTINGS = True if you have a dedicated GPU.
#  All speed-critical parameters live here — never buried in functions.
# =============================================================================

USE_GPU_SETTINGS = False   # ← set True if you have a GPU

class Config:
    # ── Paths ─────────────────────────────────────────────────────────────────
    TRAIN_DIR  : Path = ROOT_DIR / "data" / "Training"
    TEST_DIR   : Path = ROOT_DIR / "data" / "Testing"
    OUTPUT_DIR : Path = ROOT_DIR / "outputs"

    # ── Image size ────────────────────────────────────────────────────────────
    #   CPU  → 96×96   (~6× faster than 224×224, accuracy ~2–3% lower)
    #   GPU  → 224×224 (full resolution, best accuracy)
    IMG_SIZE  : tuple = (224, 224) if USE_GPU_SETTINGS else (96, 96)
    IMG_SHAPE : tuple = (224, 224, 3) if USE_GPU_SETTINGS else (96, 96, 3)

    # ── Training ──────────────────────────────────────────────────────────────
    BATCH_SIZE    : int   = 64  if USE_GPU_SETTINGS else 32
    EPOCHS        : int   = 30  if USE_GPU_SETTINGS else 20
    LEARNING_RATE : float = 1e-3
    VAL_SPLIT     : float = 0.20
    SEED          : int   = 42

    # ── Classes ───────────────────────────────────────────────────────────────
    CLASS_NAMES : list = ["glioma", "meningioma", "notumor", "pituitary"]
    NUM_CLASSES : int  = 4

    # ── Augmentation — kept lighter on CPU to save preprocessing time ─────────
    ROTATION_RANGE   : int   = 10  if not USE_GPU_SETTINGS else 15
    ZOOM_RANGE       : float = 0.10 if not USE_GPU_SETTINGS else 0.15
    HORIZONTAL_FLIP  : bool  = True
    WIDTH_SHIFT      : float = 0.08 if not USE_GPU_SETTINGS else 0.10
    HEIGHT_SHIFT     : float = 0.08 if not USE_GPU_SETTINGS else 0.10

    # ── MobileNetV2 ───────────────────────────────────────────────────────────
    #   alpha=0.35 → lightweight (~600 K params) — fast on CPU
    #   alpha=1.00 → full model  (~2.3 M params) — use on GPU
    MN_ALPHA         : float = 1.0  if USE_GPU_SETTINGS else 0.35
    MN_PHASE1_EPOCHS : int   = 10   if not USE_GPU_SETTINGS else 15
    MN_FINETUNE_FROM : int   = 60   if not USE_GPU_SETTINGS else 100

    # ── Grad-CAM samples ──────────────────────────────────────────────────────
    GRADCAM_SAMPLES  : int   = 4

    # ── tf.data pipeline ──────────────────────────────────────────────────────
    #   CACHE_DATA=True  → cache decoded images in RAM after epoch 1 (fast)
    #   Set False if RAM < 4 GB
    CACHE_DATA       : bool  = True
    PREFETCH         : int   = tf.data.AUTOTUNE
    NUM_PARALLEL_CALLS : int = tf.data.AUTOTUNE


cfg = Config()


# =============================================================================
#  SETUP
# =============================================================================

def setup() -> bool:
    """Seed RNGs, create output dir, print environment info."""
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    tf.random.set_seed(cfg.SEED)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Allow GPU memory growth — avoids OOM on shared machines
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    print("=" * 56)
    print("  Brain Tumor Classification — Environment")
    print("=" * 56)
    print(f"  Root       : {ROOT_DIR}")
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  GPU        : {[g.name for g in gpus] if gpus else 'None — CPU only'}")
    print(f"  Image size : {cfg.IMG_SIZE}  {'(CPU-optimised)' if not USE_GPU_SETTINGS else '(full)'}")
    print(f"  Batch size : {cfg.BATCH_SIZE}")
    print(f"  Epochs     : {cfg.EPOCHS}")
    print(f"  MN alpha   : {cfg.MN_ALPHA}")
    print(f"  Output dir : {cfg.OUTPUT_DIR}")
    print("=" * 56)

    ok = True
    for split, path in [("Training", cfg.TRAIN_DIR), ("Testing", cfg.TEST_DIR)]:
        if not path.exists():
            print(f"[!] Missing: {path}")
            ok = False
        else:
            counts = {
                c: len(list((path / c).glob("*.*")))
                for c in cfg.CLASS_NAMES
                if (path / c).exists()
            }
            print(f"  {split:8s}: {counts}")
    return ok


# =============================================================================
#  SECTION 1 — DATA PIPELINE  (tf.data — much faster than ImageDataGenerator)
# =============================================================================

# ── Augmentation layer (runs on the same device as the model) ─────────────────
def _augmentation_layer():
    return keras.Sequential([
        layers.RandomRotation(cfg.ROTATION_RANGE / 360),
        layers.RandomZoom(cfg.ZOOM_RANGE),
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(cfg.HEIGHT_SHIFT, cfg.WIDTH_SHIFT),
    ], name="augmentation")


def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor):
    """Decode a JPEG/PNG file → float32 tensor in [0, 1]."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)          # handles PNG too
    img = tf.image.resize(img, cfg.IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def _make_dataset(file_paths, labels, training: bool) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from file paths and one-hot labels.
    Uses caching + prefetching for fast iteration on CPU.
    """
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=cfg.SEED)

    ds = ds.map(_load_and_preprocess,
                num_parallel_calls=cfg.NUM_PARALLEL_CALLS)

    if cfg.CACHE_DATA:
        ds = ds.cache()          # cache decoded images — epoch 2+ is instant

    if training:
        # Apply augmentation AFTER caching so each epoch sees different augmentations
        aug = _augmentation_layer()
        ds  = ds.map(lambda x, y: (aug(x, training=True), y),
                     num_parallel_calls=cfg.NUM_PARALLEL_CALLS)

    ds = ds.batch(cfg.BATCH_SIZE).prefetch(cfg.PREFETCH)
    return ds


def build_datasets():
    """
    Scan data/Training and data/Testing, build tf.data pipelines.

    Returns
    -------
    train_ds, val_ds, test_ds : tf.data.Datasets
    class_weights             : dict {class_index: weight}
    n_train, n_val, n_test    : sample counts (ints)
    """
    def _scan(root: Path):
        paths, labels = [], []
        for idx, cls in enumerate(cfg.CLASS_NAMES):
            cls_dir = root / cls
            if not cls_dir.exists():
                print(f"[!] Missing class folder: {cls_dir}")
                continue
            files = sorted(
                [p for p in cls_dir.iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            )
            paths.extend([str(f) for f in files])
            labels.extend([idx] * len(files))
        return np.array(paths), np.array(labels)

    # ── Scan directories ──────────────────────────────────────────────────────
    train_paths, train_labels = _scan(cfg.TRAIN_DIR)
    test_paths,  test_labels  = _scan(cfg.TEST_DIR)

    # ── Train / val split ─────────────────────────────────────────────────────
    rng     = np.random.RandomState(cfg.SEED)
    indices = rng.permutation(len(train_paths))
    split   = int(len(indices) * (1 - cfg.VAL_SPLIT))
    tr_idx, va_idx = indices[:split], indices[split:]

    tr_paths, tr_labels = train_paths[tr_idx], train_labels[tr_idx]
    va_paths, va_labels = train_paths[va_idx], train_labels[va_idx]

    # ── One-hot encode ────────────────────────────────────────────────────────
    tr_oh = tf.one_hot(tr_labels, cfg.NUM_CLASSES).numpy()
    va_oh = tf.one_hot(va_labels, cfg.NUM_CLASSES).numpy()
    te_oh = tf.one_hot(test_labels, cfg.NUM_CLASSES).numpy()

    # ── Build datasets ────────────────────────────────────────────────────────
    train_ds = _make_dataset(tr_paths, tr_oh, training=True)
    val_ds   = _make_dataset(va_paths, va_oh, training=False)
    test_ds  = _make_dataset(test_paths, te_oh, training=False)

    # ── Class weights for imbalance ───────────────────────────────────────────
    raw_w         = compute_class_weight("balanced",
                                          classes=np.unique(tr_labels),
                                          y=tr_labels)
    class_weights = dict(zip(np.unique(tr_labels), raw_w))

    n_train, n_val, n_test = len(tr_paths), len(va_paths), len(test_paths)

    print(f"\n[Data]  Train : {n_train}  |  Val : {n_val}  |  Test : {n_test}")
    print(f"        Batches/epoch (train): {int(np.ceil(n_train/cfg.BATCH_SIZE))}")
    print(f"        Class weights : "
          f"{ {cfg.CLASS_NAMES[k]: round(v,3) for k,v in class_weights.items()} }")

    return (train_ds, val_ds, test_ds,
            class_weights, tr_labels,
            n_train, n_val, n_test,
            test_paths, test_labels)


def visualise_samples(train_ds, n: int = 12, save: bool = True):
    """Show a grid of augmented training images."""
    images, labels = next(iter(train_ds))
    images = images.numpy()
    labels = labels.numpy()
    n   = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    fig.suptitle("Sample Training Images  (after augmentation)",
                 fontsize=13, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(np.clip(images[i], 0, 1))
            ax.set_title(cfg.CLASS_NAMES[np.argmax(labels[i])], fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    if save:
        _save_fig("sample_images.png")
    plt.show()


# =============================================================================
#  SECTION 2 — MODEL DEFINITIONS
# =============================================================================

def _head(x, name: str):
    """
    Shared classification head.
    GAP → BN → Dense(128) → Dropout(0.4) → Dense(64) → Dropout(0.3) → Softmax(4)

    Note: head is smaller than the original 256→128 to keep CPU training fast.
    """
    x   = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    x   = layers.BatchNormalization(name=f"{name}_bn")(x)
    x   = layers.Dense(128, activation="relu",
                        kernel_regularizer=keras.regularizers.l2(1e-4),
                        name=f"{name}_fc1")(x)
    x   = layers.Dropout(0.40, name=f"{name}_drop1")(x)
    x   = layers.Dense(64, activation="relu",
                        kernel_regularizer=keras.regularizers.l2(1e-4),
                        name=f"{name}_fc2")(x)
    x   = layers.Dropout(0.30, name=f"{name}_drop2")(x)
    out = layers.Dense(cfg.NUM_CLASSES, activation="softmax",
                       name=f"{name}_out")(x)
    return out


# ── Model 1 : Custom CNN ──────────────────────────────────────────────────────
def build_custom_cnn(input_shape: tuple = None) -> Model:
    """
    Lightweight 3-block CNN — fast on CPU, solid accuracy.

    Filters : 32 → 64 → 128
    Per block: Conv → Conv → BN → MaxPool → Dropout
    Head    : GAP → BN → Dense(128) → Dense(64) → Softmax(4)

    CPU training time at 96×96 : ~3–5 min  (vs ~45 min at 224×224)
    """
    input_shape = input_shape or cfg.IMG_SHAPE
    inp = Input(shape=input_shape, name="cnn_input")

    specs = [(32, 0.20), (64, 0.25), (128, 0.30)]
    x = inp
    for i, (f, dr) in enumerate(specs, 1):
        x = layers.Conv2D(f, 3, padding="same", activation="relu",
                           name=f"b{i}_c1")(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu",
                           name=f"b{i}_c2")(x)
        x = layers.BatchNormalization(name=f"b{i}_bn")(x)
        x = layers.MaxPooling2D(2, name=f"b{i}_pool")(x)
        x = layers.Dropout(dr, name=f"b{i}_drop")(x)

    return Model(inp, _head(x, "cnn"), name="CustomCNN")


# ── Model 2 : LeNet-5 ─────────────────────────────────────────────────────────
def build_lenet(input_shape: tuple = None) -> Model:
    """
    Classic LeNet-5 adapted for modern colour MRI images.
    Faithful architecture: tanh, average pooling, BN for stability.
    Very fast on CPU due to its small parameter count.
    """
    input_shape = input_shape or cfg.IMG_SHAPE
    inp = Input(shape=input_shape, name="lenet_input")

    x = layers.Conv2D(6,  5, activation="tanh", padding="same",  name="c1")(inp)
    x = layers.AveragePooling2D(2, name="s2")(x)
    x = layers.BatchNormalization(name="bn1")(x)

    x = layers.Conv2D(16, 5, activation="tanh", padding="valid", name="c3")(x)
    x = layers.AveragePooling2D(2, name="s4")(x)
    x = layers.BatchNormalization(name="bn2")(x)

    x = layers.Conv2D(120, 5, activation="tanh", padding="valid", name="c5")(x)
    x = layers.BatchNormalization(name="bn3")(x)

    x   = layers.Flatten(name="flatten")(x)
    x   = layers.Dense(84, activation="tanh", name="f6")(x)
    x   = layers.Dropout(0.40, name="drop")(x)
    out = layers.Dense(cfg.NUM_CLASSES, activation="softmax", name="lenet_out")(x)

    return Model(inp, out, name="LeNet5")


# ── Model 3 : MobileNetV2  (Transfer Learning) ────────────────────────────────
def build_mobilenet(input_shape: tuple = None) -> tuple:
    """
    MobileNetV2 (ImageNet) with a custom head.

    alpha=0.35  → ~600 K params,  fast on CPU  (default)
    alpha=1.00  → ~2.3 M params,  best accuracy (GPU)

    Returns (model, base_model)
    """
    input_shape = input_shape or cfg.IMG_SHAPE

    base = MobileNetV2(
        include_top  = False,
        weights      = "imagenet",
        input_shape  = input_shape,
        alpha        = cfg.MN_ALPHA,
        name         = "mobilenet_base",
    )
    base.trainable = False

    inp   = base.input
    x     = base.output
    model = Model(inp, _head(x, "mn"), name="MobileNetV2_TL")
    return model, base


def unfreeze_for_finetuning(model: Model,
                             base_name: str = "mobilenet_base") -> Model:
    """
    Unfreeze top layers of the base model (from MN_FINETUNE_FROM onwards)
    and recompile at LR / 10.
    """
    model.trainable = True
    for layer in model.layers[:cfg.MN_FINETUNE_FROM]:
        layer.trainable = False

    n = sum(1 for l in model.layers if l.trainable)
    print(f"[Fine-tune] {n}/{len(model.layers)} model layers trainable "
          f"(unfrozen from index {cfg.MN_FINETUNE_FROM})")

    model.compile(
        optimizer = keras.optimizers.Adam(cfg.LEARNING_RATE / 10),
        loss      = "categorical_crossentropy",
        metrics   = _metrics(),
    )
    return model


# =============================================================================
#  SECTION 3 — TRAINING UTILITIES
# =============================================================================

def _metrics():
    return [
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ]


def compile_model(model: Model, lr: float = None) -> Model:
    model.compile(
        optimizer = keras.optimizers.Adam(lr or cfg.LEARNING_RATE),
        loss      = "categorical_crossentropy",
        metrics   = _metrics(),
    )
    return model


def _callbacks(model_name: str):
    ckpt_path = str(cfg.OUTPUT_DIR / f"best_{model_name}.h5")
    cbs = [
        EarlyStopping(
            monitor              = "val_loss",
            patience             = 5,           # less patient → stops sooner on CPU
            restore_best_weights = True,
            verbose              = 1,
        ),
        ModelCheckpoint(
            filepath       = ckpt_path,
            monitor        = "val_accuracy",
            save_best_only = True,
            verbose        = 0,                 # quiet — less noise per epoch
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 2,
            min_lr   = 1e-7,
            verbose  = 1,
        ),
    ]
    return cbs, ckpt_path


def _estimate_time(model: Model, train_ds, sample_batches: int = 3) -> float:
    """
    Warm up the model on a few batches and estimate seconds per epoch.
    Returns estimated minutes per epoch.
    """
    print("  [⏱]  Estimating training speed …", end=" ", flush=True)
    batch_times = []
    for i, (x, y) in enumerate(train_ds):
        if i >= sample_batches:
            break
        t0 = time.time()
        model.train_on_batch(x, y)
        batch_times.append(time.time() - t0)

    avg_sec   = np.mean(batch_times)
    n_batches = sum(1 for _ in train_ds)
    est_min   = avg_sec * n_batches / 60
    print(f"{avg_sec:.2f}s/batch → ~{est_min:.1f} min/epoch")
    return est_min


def train_model(model: Model, model_name: str,
                train_ds, val_ds, class_weights: dict,
                epochs: int = None,
                estimate_speed: bool = True):
    """
    Training loop with optional speed estimate before the first epoch.

    Returns (history, checkpoint_path)
    """
    cbs, ckpt_path = _callbacks(model_name)
    epochs         = epochs or cfg.EPOCHS

    print(f"\n{'═'*56}")
    print(f"  Training  →  {model_name}  ({model.count_params():,} params)")
    print(f"  Image size: {cfg.IMG_SIZE}  |  Epochs: {epochs}  |  Batch: {cfg.BATCH_SIZE}")
    print(f"{'═'*56}")

    if estimate_speed:
        est = _estimate_time(model, train_ds)
        print(f"  → Estimated total training time: ~{est * epochs:.0f} min "
              f"(early stopping may cut this short)")

    t0      = time.time()
    history = model.fit(
        train_ds,
        epochs          = epochs,
        validation_data = val_ds,
        class_weight    = class_weights,
        callbacks       = cbs,
        verbose         = 1,
    )
    elapsed      = (time.time() - t0) / 60
    best_val_acc = max(history.history["val_accuracy"]) * 100
    actual_ep    = len(history.history["loss"])
    print(f"\n  ✓ Completed {actual_ep} epochs in {elapsed:.1f} min")
    print(f"  ✓ Best val accuracy : {best_val_acc:.2f}%")
    print(f"  ✓ Checkpoint        : {ckpt_path}")
    return history, ckpt_path


# =============================================================================
#  SECTION 4 — EVALUATION
# =============================================================================

def _get_labels_and_preds(model: Model, test_ds):
    """Run inference on the full test dataset, return (y_true, y_pred, y_prob)."""
    y_prob_list, y_true_list = [], []
    for x, y in test_ds:
        y_prob_list.append(model.predict_on_batch(x))
        y_true_list.append(y.numpy())
    y_prob = np.concatenate(y_prob_list, axis=0)
    y_true = np.argmax(np.concatenate(y_true_list, axis=0), axis=1)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


def evaluate_model(model: Model, test_ds, model_name: str) -> dict:
    """
    Evaluate on the test split.

    Returns dict: loss, accuracy, precision, recall, auc, f1, y_true, y_pred, y_prob

    Note: uses model.metrics_names (the correct post-training attribute) instead
    of model.metrics, which returns layer objects and may omit 'accuracy' after
    a training run.
    """
    print(f"\n[Evaluate]  {model_name} …")
    results = model.evaluate(test_ds, verbose=0)

    # model.metrics_names is always a plain list of strings e.g.
    # ['loss', 'accuracy', 'precision', 'recall', 'auc']
    # model.metrics (layer objects) can be stale / differently named after .fit()
    metric_names = model.metrics_names
    metrics      = dict(zip(metric_names, results))

    # Normalise common name variants so downstream code always finds 'accuracy'
    # TF sometimes returns 'accuracy', 'acc', or 'compile_metrics/accuracy' etc.
    for raw_key in list(metrics.keys()):
        clean = raw_key.split("/")[-1]           # strip any layer-path prefix
        if clean != raw_key:
            metrics[clean] = metrics.pop(raw_key)

    # Debug print so you can see exactly what keys were returned
    print(f"  Metric keys : {list(metrics.keys())}")

    y_true, y_pred, y_prob = _get_labels_and_preds(model, test_ds)
    report = classification_report(y_true, y_pred,
                                    target_names=cfg.CLASS_NAMES,
                                    labels=list(range(len(cfg.CLASS_NAMES))),
                                    zero_division=0,
                                    output_dict=True)

    metrics.update({
        "f1_score" : report["weighted avg"]["f1-score"],
        "y_true"   : y_true,
        "y_pred"   : y_pred,
        "y_prob"   : y_prob,
    })

    # Use .get() with a fallback of 0 so a missing key never crashes here
    acc  = metrics.get("accuracy",  metrics.get("acc",  0))
    prec = metrics.get("precision", 0)
    rec  = metrics.get("recall",    0)

    # Also store under canonical names for compare_models()
    metrics["accuracy"]  = acc
    metrics["precision"] = prec
    metrics["recall"]    = rec

    print(f"  Accuracy   : {acc  * 100:.2f}%")
    print(f"  Precision  : {prec * 100:.2f}%")
    print(f"  Recall     : {rec  * 100:.2f}%")
    print(f"  F1 (wgt)   : {metrics['f1_score'] * 100:.2f}%")
    print()
    print(classification_report(y_true, y_pred, target_names=cfg.CLASS_NAMES, 
                                labels=list(range(len(cfg.CLASS_NAMES))), zero_division=0))
    return metrics


def plot_confusion_matrix(metrics: dict, model_name: str, save: bool = True):
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES,
                linewidths=0.5)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    if save: _save_fig(f"cm_{model_name}.png")
    plt.show()


def plot_training_curves(history, model_name: str, save: bool = True):
    h           = history.history
    epoch_range = range(1, len(h["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=13, fontweight="bold")
    for ax, tk, vk, label in [
        (ax1, "accuracy", "val_accuracy", "Accuracy"),
        (ax2, "loss",     "val_loss",     "Loss"),
    ]:
        ax.plot(epoch_range, h[tk], label="Train", linewidth=2)
        ax.plot(epoch_range, h[vk], label="Val",   linewidth=2, linestyle="--")
        ax.set_title(label); ax.set_xlabel("Epoch"); ax.set_ylabel(label)
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save: _save_fig(f"curves_{model_name}.png")
    plt.show()


def compare_models(all_metrics: dict, save: bool = True) -> pd.DataFrame:
    """Grouped bar chart + summary table across all models."""
    model_names     = list(all_metrics.keys())
    metrics_to_show = ["accuracy", "f1_score", "precision", "recall"]
    colors          = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    vals  = {m: [all_metrics[n].get(m, 0)*100 for n in model_names]
              for m in metrics_to_show}
    x, w  = np.arange(len(model_names)), 0.20

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, color) in enumerate(zip(metrics_to_show, colors)):
        bars = ax.bar(x + i * w, vals[metric], w,
                      label=metric.replace("_", " ").title(),
                      color=color, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.4,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x + w*1.5); ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 112); ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save: _save_fig("model_comparison.png")
    plt.show()

    df = pd.DataFrame({
        "Model": model_names,
        **{m.replace("_"," ").title():
           [f"{all_metrics[n].get(m,0)*100:.2f}%" for n in model_names]
           for m in metrics_to_show}
    })
    print("\n── Model Summary ──────────────────────────────────────")
    print(df.to_string(index=False))
    print("───────────────────────────────────────────────────────")
    return df


def show_sample_predictions(model: Model, test_ds, model_name: str,
                              n: int = 12, save: bool = True):
    imgs, labels = next(iter(test_ds))
    preds        = model.predict_on_batch(imgs)
    pred_idxs    = np.argmax(np.array(preds),     axis=1)
    true_idxs    = np.argmax(np.array(labels),    axis=1)
    imgs_np      = np.array(imgs)
    n    = min(n, len(imgs_np))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.8))
    fig.suptitle(f"Sample Predictions — {model_name}",
                 fontsize=14, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(np.clip(imgs_np[i], 0, 1))
            tl    = cfg.CLASS_NAMES[true_idxs[i]]
            pl    = cfg.CLASS_NAMES[pred_idxs[i]]
            conf  = np.array(preds)[i][pred_idxs[i]] * 100
            color = "green" if pl == tl else "red"
            ax.set_title(f"T: {tl}\nP: {pl} ({conf:.0f}%)",
                         fontsize=8.5, color=color)
        ax.axis("off")
    plt.tight_layout()
    if save: _save_fig(f"predictions_{model_name}.png")
    plt.show()


# =============================================================================
#  SECTION 5 — GRAD-CAM
# =============================================================================

def _find_last_conv(model: Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, layers.Conv2D):
                    return sub.name
    raise ValueError(f"No Conv2D found in '{model.name}'")


def _build_grad_model(model: Model, conv_name: str):
    try:
        conv_out = model.get_layer(conv_name).output
    except ValueError:
        conv_out = None
        for layer in model.layers:
            if hasattr(layer, "layers"):
                for sub in layer.layers:
                    if sub.name == conv_name:
                        conv_out = sub.output; break
            if conv_out is not None: break
    if conv_out is None:
        raise ValueError(f"Layer '{conv_name}' not found")
    return tf.keras.Model(inputs=model.inputs,
                           outputs=[conv_out, model.output])


def visualise_gradcam(model: Model, test_ds, model_name: str,
                       n_samples: int = 4, save: bool = True):
    """
    3-column Grad-CAM visualisation:
        Original MRI  |  Heatmap  |  Overlay
    Green = correct prediction, Red = incorrect.
    """
    print(f"\n[Grad-CAM]  {model_name}  ({n_samples} samples) …")
    try:
        last_conv  = _find_last_conv(model)
        gmodel     = _build_grad_model(model, last_conv)
        print(f"  Layer: '{last_conv}'")
    except Exception as e:
        print(f"  [!] Skipped: {e}"); return

    imgs, labels = next(iter(test_ds))
    imgs    = imgs.numpy()
    labels  = labels.numpy()
    indices = np.random.choice(len(imgs), min(n_samples, len(imgs)), replace=False)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 3.8))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Grad-CAM — {model_name}",
                 fontsize=13, fontweight="bold", y=1.01)
    for ax, t in zip(axes[0], ["Original MRI", "Grad-CAM", "Overlay"]):
        ax.set_title(t, fontsize=11, fontweight="bold")

    for row, idx in enumerate(indices):
        img       = imgs[idx]
        true_lbl  = cfg.CLASS_NAMES[np.argmax(labels[idx])]
        img_batch = np.expand_dims(img, 0)

        with tf.GradientTape() as tape:
            inp_t           = tf.cast(img_batch, tf.float32)
            conv_out, preds = gmodel(inp_t)
            pred_idx        = int(tf.argmax(preds[0]))
            cc              = preds[:, pred_idx]

        grads   = tape.gradient(cc, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        hm      = (conv_out[0] @ pooled[..., tf.newaxis]).numpy().squeeze()
        hm      = np.maximum(hm, 0)
        hm     /= (hm.max() + 1e-8)

        img_u8  = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        hm_r    = cv2.resize(hm, (cfg.IMG_SIZE[1], cfg.IMG_SIZE[0]))
        hm_col  = cv2.applyColorMap(np.uint8(255 * hm_r), cv2.COLORMAP_JET)
        hm_rgb  = cv2.cvtColor(hm_col, cv2.COLOR_BGR2RGB)
        overlay = (0.45 * hm_rgb + 0.55 * img_u8).astype(np.uint8)

        pred_lbl = cfg.CLASS_NAMES[pred_idx]
        conf     = float(preds[0][pred_idx]) * 100
        color    = "green" if pred_lbl == true_lbl else "red"

        axes[row, 0].imshow(img_u8);  axes[row, 0].set_ylabel(f"True: {true_lbl}", fontsize=9)
        axes[row, 1].imshow(hm, cmap="jet")
        axes[row, 2].imshow(overlay); axes[row, 2].set_xlabel(
            f"Pred: {pred_lbl} ({conf:.1f}%)", fontsize=9, color=color)
        for ax in axes[row]: ax.axis("off")

    plt.tight_layout()
    if save: _save_fig(f"gradcam_{model_name}.png")
    plt.show()


# =============================================================================
#  SECTION 6 — SINGLE-IMAGE INFERENCE
# =============================================================================

def predict_single_image(model: Model, image_path: str) -> dict:
    """Predict tumour class for one image file and display a bar chart."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)

    img       = tf.image.decode_jpeg(tf.io.read_file(str(path)), channels=3)
    img       = tf.image.resize(img, cfg.IMG_SIZE)
    img       = tf.cast(img, tf.float32) / 255.0
    img_batch = tf.expand_dims(img, 0)

    preds    = model.predict(img_batch, verbose=0)[0]
    top_idx  = int(np.argmax(preds))
    result   = {
        "predicted_class"  : cfg.CLASS_NAMES[top_idx],
        "confidence"       : float(preds[top_idx]),
        "all_probabilities": dict(zip(cfg.CLASS_NAMES, preds.tolist())),
    }

    print(f"\n  Image     : {path.name}")
    print(f"  Predicted : {result['predicted_class']}  "
          f"({result['confidence']*100:.1f}%)")
    for cls, prob in result["all_probabilities"].items():
        print(f"    {cls:12s}  {'█'*int(prob*35):<35s}  {prob*100:.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.imshow(img.numpy())
    ax1.set_title(f"{result['predicted_class']}  "
                  f"({result['confidence']*100:.1f}%)", fontsize=11)
    ax1.axis("off")
    colors = ["#C44E52" if i != top_idx else "#55A868"
               for i in range(cfg.NUM_CLASSES)]
    ax2.barh(cfg.CLASS_NAMES, preds * 100, color=colors)
    ax2.set_xlabel("Probability (%)")
    ax2.set_title("Class Probabilities")
    ax2.set_xlim(0, 105)
    for i, v in enumerate(preds * 100):
        ax2.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.show()
    return result


# =============================================================================
#  SECTION 7 — STREAMLIT APP
# =============================================================================

_APP_CODE = '''\
# Run from project root:  streamlit run app.py
import streamlit as st, numpy as np, cv2, tensorflow as tf
from pathlib import Path
from PIL import Image

ROOT        = Path(__file__).resolve().parent
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH  = ROOT / "outputs" / "best_final_model.h5"

# Read IMG_SIZE from saved model input shape
@st.cache_resource
def load_model():
    m = tf.keras.models.load_model(str(MODEL_PATH))
    return m

CLASS_INFO = {
    "glioma":     ("🔴", "Glioma",     "Arises from glial cells. Most common primary brain tumour."),
    "meningioma": ("🟠", "Meningioma", "Usually benign; grows on brain membranes."),
    "notumor":    ("🟢", "No Tumour",  "No tumour detected in the MRI scan."),
    "pituitary":  ("🔵", "Pituitary",  "Located at the base of the brain; often benign adenoma."),
}

st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor MRI Classifier")
st.caption("Upload an MRI image — glioma · meningioma · pituitary · no tumour")

try:
    model = load_model()
    IMG_SIZE = tuple(model.input_shape[1:3])
    st.success(f"Model loaded  |  input: {IMG_SIZE}", icon="✅")
except Exception as e:
    st.error(f"Cannot load model: {e}"); st.stop()

uploaded = st.file_uploader("Upload MRI (JPG / PNG)", type=["jpg","jpeg","png"])
if uploaded:
    pil   = Image.open(uploaded).convert("RGB")
    arr   = np.array(pil.resize(IMG_SIZE)) / 255.0
    batch = np.expand_dims(arr, 0)
    with st.spinner("Analysing …"):
        preds   = model.predict(batch, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        top_cls = CLASS_NAMES[top_idx]
        conf    = float(preds[top_idx]) * 100
    emoji, label, desc = CLASS_INFO[top_cls]
    c1, c2 = st.columns(2)
    with c1:
        st.image(pil, caption="Uploaded MRI", use_column_width=True)
    with c2:
        st.subheader(f"{emoji}  {label}")
        st.metric("Confidence", f"{conf:.1f}%")
        st.info(desc)
        st.divider()
        for cls, p in zip(CLASS_NAMES, preds):
            st.progress(float(p), text=f"{CLASS_INFO[cls][0]} {cls}: {p*100:.1f}%")
    st.subheader("🔥 Grad-CAM")
    try:
        last = None
        for l in reversed(model.layers):
            if isinstance(l, tf.keras.layers.Conv2D): last = l.name; break
        if last is None:
            for l in reversed(model.layers):
                if hasattr(l,"layers"):
                    for s in reversed(l.layers):
                        if isinstance(s, tf.keras.layers.Conv2D): last=s.name; break
                if last: break
        gm = tf.keras.Model(inputs=model.inputs,
                            outputs=[model.get_layer(last).output, model.output])
        with tf.GradientTape() as tape:
            inp_t = tf.cast(batch, tf.float32)
            co, p2 = gm(inp_t); cc = p2[:, top_idx]
        g = tape.gradient(cc, co)
        pg = tf.reduce_mean(g, axis=(0,1,2))
        hm = (co[0] @ pg[..., tf.newaxis]).numpy().squeeze()
        hm = np.maximum(hm, 0); hm /= (hm.max()+1e-8)
        u8 = (arr*255).astype(np.uint8)
        hr = cv2.resize(hm, (IMG_SIZE[1], IMG_SIZE[0]))
        hc = cv2.applyColorMap(np.uint8(255*hr), cv2.COLORMAP_JET)
        hrgb = cv2.cvtColor(hc, cv2.COLOR_BGR2RGB)
        ov = (0.45*hrgb + 0.55*u8).astype(np.uint8)
        st.image(ov, caption="Grad-CAM overlay", use_column_width=True)
    except Exception as e:
        st.warning(f"Grad-CAM unavailable: {e}")
st.divider(); st.caption("TensorFlow · Keras · Streamlit")
'''

def save_streamlit_app():
    path = ROOT_DIR / "app.py"
    path.write_text(_APP_CODE, encoding="utf-8")
    print(f"[✓] app.py saved → {path}")
    print("    Launch: streamlit run app.py")


# =============================================================================
#  HELPERS
# =============================================================================

def _save_fig(filename: str):
    path = cfg.OUTPUT_DIR / filename
    plt.savefig(str(path), dpi=110, bbox_inches="tight")
    print(f"    [✓] outputs/{filename}")


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def run_full_pipeline():
    """Run the complete pipeline end to end. Returns (all_metrics, all_histories, trained_models)."""
    print("\n" + "█"*60)
    print("  BRAIN TUMOR CLASSIFICATION — CPU-OPTIMISED PIPELINE")
    print("█"*60)

    if not setup():
        print("[!] Fix paths and re-run.")
        return None, None, None

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/7]  Building datasets …")
    (train_ds, val_ds, test_ds,
     class_weights, tr_labels,
     n_train, n_val, n_test,
     test_paths, test_labels) = build_datasets()

    visualise_samples(train_ds)

    all_metrics, all_histories, trained_models = {}, {}, {}

    # ── Custom CNN ────────────────────────────────────────────────────────────
    print("\n[2/7]  Custom CNN …")
    cnn        = compile_model(build_custom_cnn())
    h_cnn, ck  = train_model(cnn, "CustomCNN", train_ds, val_ds, class_weights)
    all_metrics["CustomCNN"]   = evaluate_model(cnn, test_ds, "CustomCNN")
    all_histories["CustomCNN"] = h_cnn
    plot_confusion_matrix(all_metrics["CustomCNN"],  "CustomCNN")
    plot_training_curves(h_cnn,                       "CustomCNN")
    show_sample_predictions(cnn, test_ds,             "CustomCNN")
    visualise_gradcam(cnn, test_ds,                   "CustomCNN", cfg.GRADCAM_SAMPLES)
    trained_models["CustomCNN"] = cnn

    # ── LeNet-5 ───────────────────────────────────────────────────────────────
    print("\n[3/7]  LeNet-5 …")
    lenet      = compile_model(build_lenet())
    h_ln,  ck  = train_model(lenet, "LeNet5", train_ds, val_ds, class_weights)
    all_metrics["LeNet5"]   = evaluate_model(lenet, test_ds, "LeNet5")
    all_histories["LeNet5"] = h_ln
    plot_confusion_matrix(all_metrics["LeNet5"],  "LeNet5")
    plot_training_curves(h_ln,                     "LeNet5")
    show_sample_predictions(lenet, test_ds,        "LeNet5")
    visualise_gradcam(lenet, test_ds,              "LeNet5", cfg.GRADCAM_SAMPLES)
    trained_models["LeNet5"] = lenet

    # ── MobileNetV2 Phase 1 ───────────────────────────────────────────────────
    print("\n[4/7]  MobileNetV2 Phase 1 (frozen base) …")
    mn, _     = build_mobilenet()
    compile_model(mn)
    h_p1, ck  = train_model(mn, "MobileNetV2_TL", train_ds, val_ds,
                              class_weights, epochs=cfg.MN_PHASE1_EPOCHS)

    # ── MobileNetV2 Phase 2 ───────────────────────────────────────────────────
    print("\n[5/7]  MobileNetV2 Phase 2 (fine-tuning) …")
    mn        = unfreeze_for_finetuning(mn)
    h_p2, ck  = train_model(mn, "MobileNetV2_TL", train_ds, val_ds,
                              class_weights, epochs=cfg.EPOCHS)

    merged = {k: h_p1.history[k] + h_p2.history[k] for k in h_p1.history}
    class _H:
        history = merged

    all_metrics["MobileNetV2_TL"]   = evaluate_model(mn, test_ds, "MobileNetV2_TL")
    all_histories["MobileNetV2_TL"] = _H()
    plot_confusion_matrix(all_metrics["MobileNetV2_TL"], "MobileNetV2_TL")
    plot_training_curves(_H(),                            "MobileNetV2_TL")
    show_sample_predictions(mn, test_ds,                  "MobileNetV2_TL")
    visualise_gradcam(mn, test_ds, "MobileNetV2_TL", cfg.GRADCAM_SAMPLES)
    trained_models["MobileNetV2_TL"] = mn

    # ── Compare + save ────────────────────────────────────────────────────────
    print("\n[6/7]  Comparing models …")
    compare_models(all_metrics)

    print("\n[7/7]  Saving …")
    best_name = max(all_metrics, key=lambda n: all_metrics[n].get("accuracy", 0))
    best_path = cfg.OUTPUT_DIR / "best_final_model.h5"
    trained_models[best_name].save(str(best_path))
    print(f"[✓] Best model: {best_name}  →  {best_path}")
    save_streamlit_app()

    print("\n" + "█"*60)
    print("  PIPELINE COMPLETE 🎉  |  outputs/ has all plots + models")
    print("█"*60)
    return all_metrics, all_histories, trained_models


if __name__ == "__main__":
    run_full_pipeline()