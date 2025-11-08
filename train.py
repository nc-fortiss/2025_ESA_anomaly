#!/usr/bin/env python3
# Training a simple ConvNet for ESA anomalies, Akida 1.0
# Requires: TensorFlow 2.15

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import Counter
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)) 

import dataloader_tf
from dataloader_tf import make_tf_datasets
print("[debug] using loader:", dataloader_tf.__file__)

DATA_ROOT = (HERE / "ESA-Mission1").resolve()
print("[debug] DATA_ROOT =", DATA_ROOT)
assert (DATA_ROOT / "labels.csv").exists(), f"labels.csv not found at {DATA_ROOT}"
assert (DATA_ROOT / "channels").exists(), f"channels/ not found at {DATA_ROOT}"

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

# ---------------- model ----------------
def build_model(window, n_channels):

    inputs = keras.Input(shape=(window, n_channels, 1), dtype="float32", name="input")

    # Block 1
    x = layers.Conv2D(16, (5, 5), padding="same", activation="linear", name="conv1")(inputs)
    x = layers.Activation("relu", name="conv1_relu")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="pool1")(x)

    # Block 2
    x = layers.Conv2D(32, (3, 3), padding="same", activation="linear", name="conv2")(x)
    x = layers.Activation("relu", name="conv2_relu")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="pool2")(x)

    # Block 3
    x = layers.Conv2D(64, (3, 3), padding="same", activation="linear", name="conv3")(x)
    x = layers.Activation("relu", name="conv3_relu")(x)
    x = layers.MaxPooling2D((2, 2), padding="same", name="pool3")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(64, activation="linear", name="fc1")(x)
    x = layers.Activation("relu", name="fc1_relu")(x)
    x = layers.Dense(1, activation="linear", name="logit")(x)
    outputs = layers.Activation("sigmoid", name="output")(x)

    return keras.Model(inputs, outputs, name="akida_friendly_cnn_v2")


# ---------------- main ----------------
def main():
    setup_gpu()

    # Build datasets (ALL channels), cache to disk to avoid re-loading 76 zips each run
    train_ds, val_ds, test_ds, meta = make_tf_datasets(
        data_root="/home/kannan/AI4FDIR/ESA-Mission1",
        channel_ids=None,         # ALL channels
        resample="1min",          
        window=60,
        stride=30,
        batch_size=128,
        label_mode="binary",
        cache="/tmp/esa_cache_all",   # disk cache (big)
        threaded=True,
        max_workers=8,
        verbose=True,
        use_full_timespan=True,      
    )
    window = meta["window"]
    n_channels = meta["n_channels"]
    print(f"[meta] window={window} n_channels={n_channels}")

    def count_labels(ds, name):
        import numpy as np
        c0=c1=0
        for _, y in ds:
            y = y.numpy().reshape(-1)
            c0 += np.sum(y==0)
            c1 += np.sum(y==1)
        print(f"[{name}] y==0: {c0}, y==1: {c1}")
    count_labels(train_ds, "train")
    count_labels(val_ds, "val")
    count_labels(test_ds, "test")


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

    # Build model
    model = build_model(window=window, n_channels=n_channels)
    model.summary()

    # model.compile(
    #     optimizer=keras.optimizers.Adam(1e-3),
    #     loss="binary_crossentropy",
    #     metrics=[keras.metrics.AUC(name="auroc"), "accuracy"]
    # )
    def focal_loss(gamma=2., alpha=0.25):
        import tensorflow as tf
        def _focal_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            return alpha * tf.pow(1 - p_t, gamma) * bce
        return _focal_loss

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=focal_loss(gamma=2, alpha=0.25),
        metrics=[tf.keras.metrics.AUC(name="auroc"), "accuracy"]
    )

    # Handle imbalance
    class_weights = compute_class_weights(train_ds, max_batches=200)

    # Callbacks (checkpoint H5; early stop)
    ckpt_path = "ESA_cnn.h5" 
    import datetime

    log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq="epoch"
    )

    callbacks = [
        tensorboard_cb,
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_auroc", mode="max",
            save_best_only=True, save_weights_only=False
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auroc", mode="max", patience=5, restore_best_weights=True
        ),
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
    print("[save] Model", ckpt_path, "| Norm stats norm_stats.npz")

if __name__ == "__main__":
    main()

