#!/usr/bin/env python3

import io
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

# -------------- Core I/O helpers --------------

def _load_channel_zip(zip_path: Path) -> pd.DataFrame:
    """Read a single channel_*.zip containing a pickled pandas object with time and value."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        inner = zf.namelist()[0]
        with zf.open(inner) as f:
            df = pd.read_pickle(io.BytesIO(f.read()))
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    value_cols = [c for c in df.columns if c != "time"]
    if not value_cols:
        raise ValueError(f"No value column found in {zip_path}")
    v = value_cols[0]
    return df[["time", v]]

def _merge_channels(
    root: Path,
    channel_ids: Optional[List[int]],
    resample: Optional[str],
    time_min: Optional[pd.Timestamp],
    time_max: Optional[pd.Timestamp],
    threaded: bool = True,
    max_workers: int = 8,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load requested channels, resample, outer-join on time, fill gaps.
    Threaded for faster I/O (useful with many zip files).
    """
    ch_dir = root / "channels"
    if channel_ids is None:
        channel_ids = sorted(int(p.stem.split("_")[1]) for p in ch_dir.glob("channel_*.zip"))

    def load_one(cid: int) -> Optional[pd.DataFrame]:
        zp = ch_dir / f"channel_{cid}.zip"
        if not zp.exists():
            return None
        df = _load_channel_zip(zp)
        if time_min is not None:
            df = df[df["time"] >= time_min]
        if time_max is not None:
            df = df[df["time"] <= time_max]
        if resample:
            df = (
                df.set_index("time")
                  .resample(resample)
                  .mean(numeric_only=True)
                  .dropna()
                  .reset_index()
            )
        val_col = [c for c in df.columns if c != "time"][0]
        df = df.rename(columns={val_col: f"channel_{cid}"})[["time", f"channel_{cid}"]]
        return df

    frames: List[pd.DataFrame] = []

    if threaded:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(load_one, cid): cid for cid in channel_ids}
            for i, fut in enumerate(as_completed(futs), 1):
                df = fut.result()
                if verbose and i % 5 == 0:
                    print(f"[load] {i}/{len(channel_ids)} channels loaded…")
                if df is not None:
                    frames.append(df)
    else:
        for i, cid in enumerate(channel_ids, 1):
            df = load_one(cid)
            if verbose and (i % 5 == 0 or i == 1 or i == len(channel_ids)):
                print(f"[load] {i}/{len(channel_ids)} -> channel_{cid}")
            if df is not None:
                frames.append(df)

    if not frames:
        raise RuntimeError("No channel data loaded. Check paths or time window.")

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="time", how="outer")

    merged = merged.sort_values("time")
    merged = merged.interpolate(limit_direction="both").ffill().bfill()
    return merged

def _build_labels(
    root: Path,
    timeline: pd.Series,
    label_mode: str = "binary"
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, int]]:
    """
    Create labels aligned to `timeline`.
    - binary: 0/1 anomaly
    - class/subclass: integer class id for anomaly, -1 for normal
    """
    labels = pd.read_csv(root / "labels.csv")
    labels["StartTime"] = pd.to_datetime(labels["StartTime"], utc=True)
    labels["EndTime"]   = pd.to_datetime(labels["EndTime"],   utc=True)

    y_bin = np.zeros(len(timeline), dtype=np.int64)

    y_cls = None
    class_to_idx: Dict[str, int] = {}
    if label_mode in ("class", "subclass"):
        at = pd.read_csv(root / "anomaly_types.csv")
        labels = labels.merge(at[["ID", "Class", "Subclass"]], on="ID", how="left")
        key = "Class" if label_mode == "class" else "Subclass"
        labels[key] = labels[key].fillna("Unknown").astype(str)
        uniq = sorted(labels[key].unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(uniq)}
        y_cls = np.full(len(timeline), fill_value=-1, dtype=np.int64)

    t = timeline.values.astype("datetime64[ns]")
    for _, r in labels.iterrows():
        s = r["StartTime"].to_datetime64()
        e = r["EndTime"].to_datetime64()
        mask = (t >= s) & (t <= e)
        if mask.any():
            y_bin[mask] = 1
            if y_cls is not None:
                key = "Class" if label_mode == "class" else "Subclass"
                y_cls[mask & (y_cls == -1)] = class_to_idx[str(r[key])]

    return y_bin, y_cls, class_to_idx

def _time_splits(times: pd.Series, frac_train: float = 0.7, frac_val: float = 0.15):
    n = len(times)
    i_train = int(n * frac_train)
    i_val   = int(n * (frac_train + frac_val))
    return i_train, i_val

# -------------- Windowing + tf.data --------------

def _make_windows(
    X: np.ndarray,
    y_bin: np.ndarray,
    y_cls: Optional[np.ndarray],
    window: int,
    stride: int,
    label_mode: str
):
    """
    Build sliding windows over the sample dimension.
    Label: majority (binary) or most frequent class within window (else -1).
    Returns (windows, labels).
    """
    n = len(X)
    if n < window:
        return np.empty((0, window, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)

    starts = np.arange(0, n - window + 1, stride, dtype=np.int64)

    win_list = []
    lab_list = []
    for s in starts:
        e = s + window
        win_list.append(X[s:e])
        if y_cls is None or label_mode == "binary":
            lab_list.append(int(y_bin[s:e].mean() >= 0.5))
        else:
            seg = y_cls[s:e]
            vals, counts = np.unique(seg[seg >= 0], return_counts=True)
            lab_list.append(int(vals[np.argmax(counts)]) if len(vals) else -1)

    W = np.stack(win_list).astype(np.float32)  # [N, window, C]
    Y = np.asarray(lab_list, dtype=np.int64)   # [N]
    return W, Y

def _to_tf_dataset(Xw: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((Xw, Y))
    if shuffle and len(Xw) > 0:
        ds = ds.shuffle(min(8192, len(Xw)), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -------------- Public API --------------

def make_tf_datasets(
    data_root: str,
    channel_ids: Optional[List[int]] = None,
    resample: str = "1min",
    start: Optional[str] = None,
    end:   Optional[str] = None,
    window: int = 60,
    stride: int = 30,
    batch_size: int = 64,
    label_mode: str = "binary",
    splits=(0.7, 0.15, 0.15),
    cache: Optional[str] = None,
    threaded: bool = True,
    max_workers: int = 8,
    verbose: bool = True,
    use_full_timespan: bool = False, 
):
    import time
    t0 = time.time()
    root = Path(data_root).expanduser().resolve()

    # Choose time range
    if start is None and end is None and use_full_timespan:
        # Load ALL available data from channels (no time windowing)
        t0_user = None
        t1_user = None
        if verbose:
            print("[time] using FULL timespan from channels (no start/end limits)")
    else:
        labs = pd.read_csv(root / "labels.csv")
        lab_min = pd.to_datetime(labs["StartTime"], utc=True).min()
        lab_max = pd.to_datetime(labs["EndTime"],   utc=True).max()
        t0_user = pd.to_datetime(start, utc=True) if start else lab_min
        t1_user = pd.to_datetime(end,   utc=True) if end   else lab_max
        if verbose:
            print(f"[time] {t0_user} → {t1_user}")


    # Merge all channels
    df = _merge_channels(
        root,
        channel_ids,
        resample,
        t0_user,
        t1_user,
        threaded=threaded,
        max_workers=max_workers,
        verbose=verbose
    )
    if verbose:
        print(f"[merge] frame shape={df.shape}  elapsed={time.time() - t0:.1f}s")

    # Labels aligned to df["time"]
    y_bin, y_cls, class_to_idx = _build_labels(root, df["time"], label_mode)

    # Chronological split
    i_train, i_val = _time_splits(df["time"], frac_train=splits[0], frac_val=splits[1])

    feats = df.drop(columns=["time"]).astype(np.float32).to_numpy()
    feature_names = [c for c in df.columns if c != "time"]

    # Windowing by split
    X_tr, y_tr = _make_windows(feats[:i_train], y_bin[:i_train],
                               None if y_cls is None else y_cls[:i_train],
                               window, stride, label_mode)
    X_va, y_va = _make_windows(feats[i_train:i_val], y_bin[i_train:i_val],
                               None if y_cls is None else y_cls[i_train:i_val],
                               window, stride, label_mode)
    X_te, y_te = _make_windows(feats[i_val:], y_bin[i_val:],
                               None if y_cls is None else y_cls[i_val:],
                               window, stride, label_mode)

    if verbose:
        print(f"[windows] train={X_tr.shape} val={X_va.shape} test={X_te.shape}")

    # tf.data datasets
    train_ds = _to_tf_dataset(X_tr, y_tr, batch_size, shuffle=True)
    val_ds   = _to_tf_dataset(X_va, y_va, batch_size, shuffle=False)
    test_ds  = _to_tf_dataset(X_te, y_te, batch_size, shuffle=False)

    # Optional cache
    if cache:
        if cache == "memory":
            if verbose: print("[cache] caching in RAM…")
            train_ds = train_ds.cache()
            val_ds   = val_ds.cache()
            test_ds  = test_ds.cache()
        else:
            cache_dir = Path(cache).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            if verbose: print(f"[cache] caching on disk → {cache_dir}")
            train_ds = train_ds.cache(str(cache_dir / "train.cache"))
            val_ds   = val_ds.cache(str(cache_dir / "val.cache"))
            test_ds  = test_ds.cache(str(cache_dir / "test.cache"))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.prefetch(tf.data.AUTOTUNE)

    meta = dict(
        feature_names=feature_names,
        class_to_idx=class_to_idx,
        n_channels=len(feature_names),
        window=window,
        stride=stride,
        resample=resample,
        shapes=dict(train=X_tr.shape, val=X_va.shape, test=X_te.shape)
    )

    if verbose:
        print(f"[done] total build time: {time.time() - t0:.1f}s")

    return train_ds, val_ds, test_ds, meta

#transpose for (batch, channels, window) 
def map_time_last_to_channel_first(ds: tf.data.Dataset) -> tf.data.Dataset:
    """For models that expect (batch, channels, window) instead of (batch, window, channels)."""
    def _swap(x, y):
        x = tf.transpose(x, [0, 2, 1])
        return x, y
    return ds.map(_swap, num_parallel_calls=tf.data.AUTOTUNE)



