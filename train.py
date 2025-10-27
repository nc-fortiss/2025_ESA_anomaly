#!/usr/bin/env python3
# Training a simple ConvNet for ESA anomalies, Akida 1.0–friendly
# Requires: TensorFlow 2.15, your dataloader_tf.py from earlier

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter

from Dataloader import make_tf_datasets

# ---------------- GPU setup ----------------
def setup_gpu():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"[GPU] Using {len(gpus)} GPU(s).")
        else:
            print("[GPU] No GPU found. Running on CPU.")
    except Exception as e:
        print("[GPU] Could not set memory growth:", e)

# ---------------- utils ----------------
def estimate_norm_stats(ds, take_batches=50):
    """
    Estimate per-channel mean/std from streamed windows.
    x comes as [B, window, C]; we flatten batch+time and compute per-channel stats.
    """
    cnt = 0
    sum_ = None
    sumsq_ = None
    for i, (x, y) in enumerate(ds.take(take_batches)):
        x = tf.cast(x, tf.float32)  # [B, W, C]
        B, W, C = x.shape
        if sum_ is None:
            sum_ = tf.zeros([C], dtype=tf.float32)
            sumsq_ = tf.zeros([C], dtype=tf.float32)
        x2d = tf.reshape(x, [-1, C])  # [(B*W), C]
        sum_ += tf.reduce_sum(x2d, axis=0)
        sumsq_ += tf.reduce_sum(tf.square(x2d), axis=0)
        cnt += x2d.shape[0]
    if cnt == 0:
        raise RuntimeError("Empty dataset while estimating normalization stats.")
    mean = sum_ / cnt
    var = sumsq_ / cnt - tf.square(mean)
    std = tf.sqrt(tf.maximum(var, 1e-8))
    return mean.numpy(), std.numpy()

def make_normalized_image_ds(ds, mean, std, time_first=True):
    """
    ds yields (x, y) where x: [B, W, C]. Normalize and reshape to [B, H, W, 1].
    - If time_first=True: H=W(window), W=n_channels
      Otherwise swap to H=n_channels, W=window.
    """
    mean = tf.constant(mean, dtype=tf.float32)
    std  = tf.constant(std, dtype=tf.float32)

    def _map(x, y):
        x = tf.cast(x, tf.float32)              # [B, W, C]
        x = (x - mean) / tf.maximum(std, 1e-6)  # per-channel z-score
        if not time_first:
            x = tf.transpose(x, [0, 2, 1])      # [B, C, W]
        # add channel dim for Conv2D
        x = tf.expand_dims(x, axis=-1)          # [B, H, W, 1]
        return x, y

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)

def compute_class_weights(ds, max_batches=200):
    """
    For binary labels: returns {0: w0, 1: w1}. Uses counts from a sample of dataset.
    """
    c = Counter()
    seen = 0
    for xb, yb in ds.take(max_batches):
        y_np = yb.numpy().reshape(-1)
        c.update(y_np.tolist())
        seen += len(y_np)
    if seen == 0:
        return {0: 1.0, 1: 1.0}
    n0 = float(c.get(0, 1))
    n1 = float(c.get(1, 1))
    total = n0 + n1
    # inverse frequency
    w0 = total / (2.0 * n0) if n0 > 0 else 1.0
    w1 = total / (2.0 * n1) if n1 > 0 else 1.0
    print(f"[class-weights] counts: 0={int(n0)} 1={int(n1)} → weights: 0={w0:.3f} 1={w1:.3f}")
    return {0: w0, 1: w1}

# ---------------- Akida-friendly model ----------------
def build_akida_friendly_model(window, n_channels):
    """
    Build a small ConvNet using only Akida 1.0–compatible ops:
    Conv2D + ReLU + MaxPool2D + Flatten + Dense(+ReLU) + Dense(sigmoid).
    Input format: [B, H=window, W=n_channels, C=1]
    """
    inputs = keras.Input(shape=(window, n_channels, 1), dtype="float32", name="input")

    # Conv stack (keep it simple for Akida conversion)
    x = layers.Conv2D(16, (5, 5), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name="akida_friendly_cnn")
    return model

# ---------------- main ----------------
def main():
    setup_gpu()

    DATA_ROOT = "/home/kannan/Kannan_Workspace/AI4FDIR/ESA-Mission1"

    # Build datasets (ALL channels), cache to disk to avoid re-loading 76 zips each run
    train_ds, val_ds, test_ds, meta = make_tf_datasets(
        data_root=DATA_ROOT,
        channel_ids=None,           # ALL channels
        resample="1min",
        start="2004-12-01T00:00:00Z",
        end="2004-12-10T00:00:00Z",
        window=60,
        stride=30,
        batch_size=128,
        label_mode="binary",
        cache="/tmp/esa_cache",
        threaded=True,
        max_workers=8,
        verbose=True
    )
    window = meta["window"]
    n_channels = meta["n_channels"]
    print(f"[meta] window={window} n_channels={n_channels}")

    # Estimate normalization stats on the *already windowed* train dataset
    mean, std = estimate_norm_stats(train_ds, take_batches=80)
    print("[norm] mean[:5]=", np.round(mean[:5], 4), "std[:5]=", np.round(std[:5], 4))

    # Normalize in the pipeline and reshape to [B, H, W, 1] for Conv2D
    # time_first=True → H=window, W=n_channels
    train_ds = make_normalized_image_ds(train_ds, mean, std, time_first=True)
    val_ds   = make_normalized_image_ds(val_ds,   mean, std, time_first=True)
    test_ds  = make_normalized_image_ds(test_ds,  mean, std, time_first=True)

    # Inspect one batch
    xb, yb = next(iter(train_ds))
    print("[batch] X:", xb.shape, "y:", yb.shape)  # (batch, window, n_channels, 1)

    # Build Akida-friendly model
    model = build_akida_friendly_model(window=window, n_channels=n_channels)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auroc"), "accuracy"]
    )

    # Handle imbalance
    class_weights = compute_class_weights(train_ds, max_batches=200)

    # Callbacks (checkpoint H5; early stop)
    ckpt_path = "akida_friendly_cnn.h5"  # classic H5 for later Akida conversion
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_auroc", mode="max",
            save_best_only=True, save_weights_only=False
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auroc", mode="max", patience=5, restore_best_weights=True
        )
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Evaluate
    print("\n[eval] Test set:")
    model.evaluate(test_ds, verbose=2)

    # Save normalization stats to reuse at inference/export
    np.savez("norm_stats.npz", mean=mean, std=std)
    print("[save] Model →", ckpt_path, "| Norm stats → norm_stats.npz")

if __name__ == "__main__":
    main()
