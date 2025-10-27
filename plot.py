#!/usr/bin/env python3
import argparse, io, zipfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
def load_df_from_channel_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        inner = zf.namelist()[0]
        with zf.open(inner) as f:
            df = pd.read_pickle(io.BytesIO(f.read()))
    # normalize time
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": "time"})
    for cand in ["time", "timestamp", "date", "datetime"]:
        if cand in df.columns:
            df["time"] = pd.to_datetime(df[cand], utc=True, errors="coerce")
            if cand != "time":
                df = df.drop(columns=[cand])
            break
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

def parse_channels(s: str):
    # e.g., "12,13,15"
    return [int(x.strip()) for x in s.split(",") if x.strip()]

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Path to ESA-Mission1")
    ap.add_argument("--channels", required=True, help="Comma-separated channel IDs, e.g. 12,13,16")
    ap.add_argument("--start", required=True, help="ISO start, e.g. 2004-12-01T00:00:00Z")
    ap.add_argument("--end",   required=True, help="ISO end,   e.g. 2004-12-10T00:00:00Z")
    ap.add_argument("--resample", default=None, help="Optional resample rule, e.g. 1min, 10s, 5min (lowercase!)")
    ap.add_argument("--every", type=int, default=1, help="Take every Nth sample after resampling/filtering")
    ap.add_argument("--save", default=None, help="Optional path to save PNG")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    ch_dir = root / "channels"
    lab_csv = root / "labels.csv"

    t0 = pd.to_datetime(args.start, utc=True)
    t1 = pd.to_datetime(args.end, utc=True)

    sel = parse_channels(args.channels)
    print(f"[i] Plotting channels: {sel}  window: {t0} → {t1}")

    frames = []
    for cid in sel:
        zip_path = ch_dir / f"channel_{cid}.zip"
        if not zip_path.exists():
            print(f"Missing {zip_path.name}, skipping.")
            continue
        df = load_df_from_channel_zip(zip_path)
        # keep only the column that matches this channel
        value_col = f"channel_{cid}"
        if value_col not in df.columns:
            value_candidates = [c for c in df.columns if c != "time"]
            value_col = value_candidates[0]
            df = df.rename(columns={value_col: f"channel_{cid}"})
            value_col = f"channel_{cid}"
        # window
        df = df[(df["time"] >= t0) & (df["time"] <= t1)]
        if args.resample:
            df = df.set_index("time").resample(args.resample).mean(numeric_only=True).dropna().reset_index()
        if args.every > 1:
            df = df.iloc[::args.every, :]
        print(f"    • channel_{cid}: {df.shape[0]} samples in window")
        frames.append(df[["time", value_col]])

    if not frames:
        print("[!] No data to plot in the requested window.")
        return

    # merge on time
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="time", how="outer")
    out = out.sort_values("time")

    # read labels and filter for selected channels + time overlap
    labels = None
    if lab_csv.exists():
        labels = pd.read_csv(lab_csv)
        for col in ["StartTime", "EndTime"]:
            labels[col] = pd.to_datetime(labels[col], utc=True, errors="coerce")
        labels = labels[
            (labels["Channel"].str.replace("channel_", "").astype(int).isin(sel)) &
            (labels["EndTime"] >= t0) & (labels["StartTime"] <= t1)
        ].copy()
        print(f"[i] Label intervals overlapping window: {len(labels)}")

    # ---- plot ----
    n = len(sel)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8*n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, cid in zip(axes, sel):
        col = f"channel_{cid}"
        if col in out.columns:
            ax.plot(out["time"], out[col], linewidth=1)
            ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        # shade anomaly intervals for this channel
        if labels is not None and not labels.empty:
            sub = labels[labels["Channel"] == f"channel_{cid}"]
            for _, row in sub.iterrows():
                s, e = row["StartTime"], row["EndTime"]
                ax.axvspan(s, e, alpha=0.15)
    axes[-1].set_xlabel("time")
    fig.suptitle(f"ESA raw channels {sel}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.save:
        out_path = Path(args.save).with_suffix(".png")
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"Saved figure → {out_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
