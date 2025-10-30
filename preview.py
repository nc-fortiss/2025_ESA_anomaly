#!/usr/bin/env python3
import argparse, io, zipfile, math
from pathlib import Path
import pandas as pd
from typing import Optional

def load_df_from_channel_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if not names:
            raise ValueError(f"{zip_path} is empty")
        inner = names[0]
        with zf.open(inner) as f:
            df = pd.read_pickle(io.BytesIO(f.read()))
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": "time"})
    for cand in ["time","timestamp","date","datetime"]:
        if cand in df.columns:
            try:
                df["time"] = pd.to_datetime(df[cand], utc=True, errors="coerce")
                df = df.drop(columns=[c for c in [cand] if c!="time"])
            except Exception:
                pass
            break
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    return df



def rough_sampling_hz(times: pd.Series) -> Optional[float]:
    if times.isna().any() or len(times) < 3:
        return None
    dt = times.diff().dropna().dt.total_seconds()
    if dt.empty:
        return None
    q = dt[(dt > 0) & (dt < dt.quantile(0.99))]
    if q.empty: 
        return None
    mean_dt = q.mean()
    return 1.0/mean_dt if mean_dt > 0 else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Path to ESA-Mission1 folder")
    ap.add_argument("--sample_channel", type=int, default=1, help="Which channel_* to open")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    channels_dir = root / "channels"
    print(f"[i] Data root: {root}")
    print(f"[i] Channels dir: {channels_dir}")

    zips = sorted(channels_dir.glob("channel_*.zip"))
    print(f"[✓] Found {len(zips)} channel zip(s). Example names:")
    for p in zips[:5]:
        print("   •", p.name)
    if not zips:
        raise SystemExit("No channel_*.zip found")

    # open sample zip
    sample_name = f"channel_{args.sample_channel}.zip"
    sample = channels_dir / sample_name
    if not sample.exists():
        sample = zips[0]
        print(f"[!] Requested {sample_name} not found, using {sample.name}")

    print(f"\n[i] Inspecting: {sample.name}")
    with zipfile.ZipFile(sample, "r") as zf:
        print("    inner files:", zf.namelist())

    df = load_df_from_channel_zip(sample)
    print("\n[✓] Loaded DataFrame from zip")
    print("    shape:", df.shape)
    print("    columns:", list(df.columns))
    print("    dtypes:\n", df.dtypes)

    if "time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["time"]):
        tmin, tmax = df["time"].min(), df["time"].max()
        print(f"    time range: {tmin}  →  {tmax}")
        hz = rough_sampling_hz(df["time"])
        if hz:
            print(f"    ~sampling frequency: {hz:.2f} Hz")
    else:
        print("    (no recognizable time column)")

    print("\n[head]")
    print(df.head(5).to_string(index=False))

    for csv_name in ["channels.csv", "labels.csv", "telecommands.csv", "anomaly_types.csv", "telecommands.csv"]:
        csv_path = root / csv_name
        if csv_path.exists():
            try:
                side = pd.read_csv(csv_path)
                print(f"\n[✓] {csv_name}: shape={side.shape}")
                print(side.head(5).to_string(index=False))
            except Exception as e:
                print(f"[!] Could not read {csv_name}: {e}")

if __name__ == "__main__":
    main()

