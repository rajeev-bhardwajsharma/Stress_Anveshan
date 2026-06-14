"""
================================================================================
WESAD Feature Extraction Pipeline  —  Main Loop  (Chest / Wrist Separated)
================================================================================
Reads per-subject window pickles from:
    WDM_dataset/Interim/<subject>/<subject>_windows.pkl

Saves SEPARATE parquet files per device AND per feature type:

    WDM_dataset/Features/<subject>/
        chest/
            <subject>_chest_statistical.parquet
            <subject>_chest_heart.parquet
            <subject>_chest_eda.parquet
            <subject>_chest_acc.parquet
        wrist/
            <subject>_wrist_statistical.parquet
            <subject>_wrist_heart.parquet
            <subject>_wrist_eda.parquet
            <subject>_wrist_acc.parquet

Every parquet shares metadata columns: subject, window_idx, start_time,
end_time, label — so any combination can be joined on window_idx.

Ablation study usage
--------------------
Full fusion   →  load chest/ + wrist/ everything and merge on window_idx
Chest only    →  load chest/ folder only
Wrist only    →  load wrist/ folder only
ECG only      →  load chest_heart + chest_statistical (ECG columns)
EDA only      →  load chest_eda + wrist_eda

Phase C note
------------
Features are stored RAW (unnormalized).
Normalize INSIDE each LOSO fold: fit on training subjects, transform test only.
================================================================================
"""

from itertools import accumulate
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from statistical_features import extract_statistical_features_all_signals
from heart_features        import extract_heart_features
from eda_features          import extract_eda_features
from acc_features          import extract_acc_features
from emg_features import extract_emg_features  # point to be noted added later->2.0

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
INTERIM_ROOT = Path("/home/rs/ml-projects/WDM_dataset/Interim")
FEATURE_ROOT = Path("/home/rs/ml-projects/WDM_dataset/Features")

ALL_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]

# ── Sampling rates ────────────────────────────────────────────────────────────
SENSOR_FS: dict[str, int] = {
    "chest_ECG":  700,
    "chest_EDA":  700,
    "chest_EMG":  700,
    "chest_RESP": 700,
    "chest_TEMP": 700,
    "chest_ACC":  700,
    "wrist_BVP":   64,
    "wrist_ACC":   32,
    "wrist_EDA":    4,
    "wrist_TEMP":   4,
}

# ── Signal groupings by device ────────────────────────────────────────────────

CHEST_STAT_SIGNALS = ["chest_ECG", "chest_EDA", "chest_EMG",
                      "chest_RESP", "chest_TEMP", "chest_ACC"]

WRIST_STAT_SIGNALS = ["wrist_BVP", "wrist_EDA", "wrist_TEMP", "wrist_ACC"]

CHEST_HEART_SIGNALS = {"chest_ECG": 700}
WRIST_HEART_SIGNALS = {"wrist_BVP":  64}

CHEST_EDA_SIGNALS   = {"chest_EDA": 700}
WRIST_EDA_SIGNALS   = {"wrist_EDA":   4}

CHEST_EMG_SIGNALS = {"chest_EMG": 700}

CHEST_ACC_SIGNALS   = {"chest_ACC": 700}
WRIST_ACC_SIGNALS   = {"wrist_ACC":  32}


# ─────────────────────────────────────────────────────────────────────────────
# Per-window extraction  →  returns 8 row dicts (4 chest + 4 wrist)
# ─────────────────────────────────────────────────────────────────────────────

def extract_window_features(
    window:     dict[str, Any],
    window_idx: int,
) -> dict[str, dict]:
    """
    Run all feature modules on one window, keeping chest and wrist separate.

    Returns
    -------
    dict with keys:
        "chest_statistical", "chest_heart", "chest_eda", "chest_acc"
        "wrist_statistical", "wrist_heart", "wrist_eda", "wrist_acc"
    Each value is a flat feature dict (metadata + features).
    """
    meta = {
        "subject":    window["subject"],
        "window_idx": window_idx,
        "start_time": window["start_time"],
        "end_time":   window["end_time"],
        "label":      window["final_label"],
    }

    rows: dict[str, dict] = {}

    # ── CHEST ─────────────────────────────────────────────────────────────

    # Statistical — all chest signals
    chest_stat_signals = {k: window[k] for k in CHEST_STAT_SIGNALS if k in window}
    rows["chest_statistical"] = {
        **meta,
        **extract_statistical_features_all_signals(chest_stat_signals, SENSOR_FS),
    }

    # Heart — ECG only
    chest_heart_feats: dict[str, float] = {}
    for sensor, fs in CHEST_HEART_SIGNALS.items():
        if sensor in window:
            chest_heart_feats.update(
                extract_heart_features(window[sensor], sensor.lower(), fs)
            )
    rows["chest_heart"] = {**meta, **chest_heart_feats}

    # EDA — chest EDA only
    chest_eda_feats: dict[str, float] = {}
    for sensor, fs in CHEST_EDA_SIGNALS.items():
        if sensor in window:
            chest_eda_feats.update(
                extract_eda_features(window[sensor], sensor.lower(), fs)
            )
    rows["chest_eda"] = {**meta, **chest_eda_feats}

    # ACC — chest ACC only
    chest_acc_feats: dict[str, float] = {}
    for sensor, fs in CHEST_ACC_SIGNALS.items():
        if sensor in window:
            chest_acc_feats.update(
                extract_acc_features(window[sensor], sensor.lower(), fs)
            )
    rows["chest_acc"] = {**meta, **chest_acc_feats}

   

    # EMG — chest EMG only
    chest_emg_feats: dict[str, float] = {}
    for sensor, fs in CHEST_EMG_SIGNALS.items():
        if sensor in window:
            chest_emg_feats.update(
                extract_emg_features(window[sensor], sensor.lower(), fs)
            )
    rows["chest_emg"] = {**meta, **chest_emg_feats} # This key matches BUNDLE_DEVICE

    # ── WRIST ─────────────────────────────────────────────────────────────

    # Statistical — all wrist signals
    wrist_stat_signals = {k: window[k] for k in WRIST_STAT_SIGNALS if k in window}
    rows["wrist_statistical"] = {
        **meta,
        **extract_statistical_features_all_signals(wrist_stat_signals, SENSOR_FS),
    }

    # Heart — BVP only
    wrist_heart_feats: dict[str, float] = {}
    for sensor, fs in WRIST_HEART_SIGNALS.items():
        if sensor in window:
            wrist_heart_feats.update(
                extract_heart_features(window[sensor], sensor.lower(), fs)
            )
    rows["wrist_heart"] = {**meta, **wrist_heart_feats}

    # EDA — wrist EDA only
    wrist_eda_feats: dict[str, float] = {}
    for sensor, fs in WRIST_EDA_SIGNALS.items():
        if sensor in window:
            wrist_eda_feats.update(
                extract_eda_features(window[sensor], sensor.lower(), fs)
            )
    rows["wrist_eda"] = {**meta, **wrist_eda_feats}

    # ACC — wrist ACC only
    wrist_acc_feats: dict[str, float] = {}
    for sensor, fs in WRIST_ACC_SIGNALS.items():
        if sensor in window:
            wrist_acc_feats.update(
                extract_acc_features(window[sensor], sensor.lower(), fs)
            )
    rows["wrist_acc"] = {**meta, **wrist_acc_feats}

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject loop
# ─────────────────────────────────────────────────────────────────────────────

# Maps bundle key → device subfolder
BUNDLE_DEVICE: dict[str, str] = {
    "chest_statistical": "chest",
    "chest_heart":       "chest",
    "chest_eda":         "chest",
    "chest_acc":         "chest",
    "chest_emg":         "chest",#added later 2.0
    "wrist_statistical": "wrist",
    "wrist_heart":       "wrist",
    "wrist_eda":         "wrist",
    "wrist_acc":         "wrist",
}

def process_subject(subject_id: str) -> None:
    """
    Surgically extracts only EMG features for a subject if they don't exist.
    Uses absolute paths to prevent duplicate WDM_dataset folders.
    """
    # 1. Define specific subject directory
    # Path: /home/rs/ml-projects/WDM_dataset/Features/S2/chest/
    subject_chest_dir = FEATURE_ROOT / subject_id / "chest"
    emg_out_path = subject_chest_dir / f"{subject_id}_chest_emg.parquet"
    
    # Path to source window data
    pkl_path = INTERIM_ROOT / subject_id / f"{subject_id}_windows.pkl"

    # 2. Check if source exists or work is already done
    if not pkl_path.exists():
        print(f"  [{subject_id}]  SKIPPED — pkl not found at {pkl_path}")
        return

    if emg_out_path.exists():
        print(f"  [{subject_id}]  SKIPPING — EMG features already exist.")
        return

    # 3. Load the windowed data
    with open(pkl_path, "rb") as fh:
        windows: list[dict] = pickle.load(fh)

    print(f"  [{subject_id}]  {len(windows)} windows — extracting ONLY EMG features …")

    # 4. Target only EMG extraction
    emg_rows = []
    for idx, window in enumerate(windows):
        try:
            # Metadata block matching your existing parquet structure
            row = {
                "subject":    window["subject"],
                "window_idx": idx,
                "start_time": window["start_time"],
                "end_time":   window["end_time"],
                "label":      window["final_label"],
            }

            # Extract EMG features only using emg_features.py logic
            if "chest_EMG" in window:
                emg_feats = extract_emg_features(
                    window["chest_EMG"], 
                    "chest_emg", 
                    SENSOR_FS["chest_EMG"]
                )
                row.update(emg_feats)
            
            emg_rows.append(row)

        except Exception as ex:
            print(f"    window {idx} failed: {ex}")
            continue

    # 5. Save the specific EMG bundle
    if emg_rows:
        # Create /home/rs/ml-projects/WDM_dataset/Features/S2/chest/ if it doesn't exist
        subject_chest_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(emg_rows)
        df.to_parquet(emg_out_path, index=False, engine="pyarrow", compression="snappy")
        
        print(f"    [{subject_id}]  Saved to: {emg_out_path}")
"""
    # ── Save each bundle to its device subfolder ───────────────────────────
    for bundle_key, device in BUNDLE_DEVICE.items():
        rows = accumulate[bundle_key]
        if not rows:
            print(f"    [{subject_id}]  no rows for {bundle_key} — skipping")
            continue

        out_dir  = FEATURE_ROOT / subject_id / device
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{subject_id}_{bundle_key}.parquet"
        out_path = out_dir / filename

        df = pd.DataFrame(rows)
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

        print(f"    [{subject_id}]  {bundle_key:<22} "
              f"→ {df.shape[0]} rows × {df.shape[1]} cols  →  "
              f"{device}/{filename}")
        """


# ─────────────────────────────────────────────────────────────────────────────
# All-subjects runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all(subjects: list[str] = ALL_SUBJECTS) -> None:
    print("=" * 68)
    print("  WESAD Feature Extraction Pipeline  (chest / wrist separated)")
    print("=" * 68)

    failed = []
    for subject_id in subjects:
        try:
            process_subject(subject_id)
        except Exception as ex:
            print(f"  [{subject_id}]  FAILED — {ex}")
            failed.append(subject_id)

    # ── Final output tree ──────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("Output structure:")
    for subject_id in subjects:
        if subject_id in failed:
            continue
        for device in ["chest", "wrist"]:
            device_dir = FEATURE_ROOT / subject_id / device
            if device_dir.exists():
                for f in sorted(device_dir.glob("*.parquet")):
                    df = pd.read_parquet(f)
                    print(f"  {f.relative_to(FEATURE_ROOT)}"
                          f"  ({df.shape[0]} rows × {df.shape[1]} cols)")

    if failed:
        print(f"\nFailed: {failed}")
    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
# Merge all subjects into master parquets
# ─────────────────────────────────────────────────────────────────────────────

def merge_all_subjects(subjects: list[str] = ALL_SUBJECTS) -> None:
    """
    Concatenate per-subject parquets into master files.

    Output
    ------
    WDM_dataset/Features/
        chest/
            all_chest_statistical.parquet
            all_chest_heart.parquet
            all_chest_eda.parquet
            all_chest_acc.parquet
        wrist/
            all_wrist_statistical.parquet
            all_wrist_heart.parquet
            all_wrist_eda.parquet
            all_wrist_acc.parquet

    Filter by subject column inside each LOSO fold.
    """
    print("\nMerging all subjects …")

    for bundle_key, device in BUNDLE_DEVICE.items():
        dfs = []
        for subject_id in subjects:
            path = (FEATURE_ROOT / subject_id / device
                    / f"{subject_id}_{bundle_key}.parquet")
            if path.exists():
                dfs.append(pd.read_parquet(path))

        if not dfs:
            print(f"  [merge]  no files found for {bundle_key}")
            continue

        out_dir  = FEATURE_ROOT / device
        out_dir.mkdir(parents=True, exist_ok=True)

        merged   = pd.concat(dfs, ignore_index=True)
        out_path = out_dir / f"all_{bundle_key}.parquet"
        merged.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

        print(f"  [merge]  {bundle_key:<22} "
              f"→ {merged.shape[0]} rows × {merged.shape[1]} cols  "
              f"→  {device}/all_{bundle_key}.parquet")



# Entry point


if __name__ == "__main__":
    run_all()
    merge_all_subjects()