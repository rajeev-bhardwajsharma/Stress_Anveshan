"""
================================================================================
WESAD Multi-Subject Windowing Pipeline
================================================================================
Processes subjects S2 to S17, runs the multirate sliding window function
on each, and saves the resulting list of window dicts as a pickle file to:

    WDM_dataset/Interim/<subject_id>/<subject_id>_windows.pkl

Skips S12 automatically (missing in WESAD — only 15 subjects, S12 excluded).
================================================================================
"""

import pickle
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ROOT = Path("WDM_dataset/WESAD")
OUTPUT_ROOT  = Path("WDM_dataset/Interim")

# WESAD subjects — S12 does not exist in the dataset
ALL_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]

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

LABEL_FS:   int   = 700
WINDOW_SEC: float = 60.0
STEP_SEC:   float = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load one subject's pkl
# ─────────────────────────────────────────────────────────────────────────────

def load_subject(subject_id: str) -> dict:
    pkl_path = DATASET_ROOT / subject_id / f"{subject_id}.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(f"pkl not found: {pkl_path}")

    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh, encoding="latin1")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Extract signals + labels from raw dict
# ─────────────────────────────────────────────────────────────────────────────

def extract_signals(data: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    def _squeeze(arr):
        arr = np.array(arr)
        return arr.squeeze(axis=1) if (arr.ndim == 2 and arr.shape[1] == 1) else arr

    chest = data["signal"]["chest"]
    wrist = data["signal"]["wrist"]

    signals = {
        "chest_ECG":  _squeeze(chest["ECG"]),
        "chest_EDA":  _squeeze(chest["EDA"]),
        "chest_EMG":  _squeeze(chest["EMG"]),
        "chest_RESP": _squeeze(chest["Resp"]),
        "chest_TEMP": _squeeze(chest["Temp"]),
        "chest_ACC":  np.array(chest["ACC"]),   # shape (N, 3)
        "wrist_BVP":  _squeeze(wrist["BVP"]),
        "wrist_ACC":  np.array(wrist["ACC"]),   # shape (N, 3)
        "wrist_EDA":  _squeeze(wrist["EDA"]),
        "wrist_TEMP": _squeeze(wrist["TEMP"]),
    }

    labels = np.array(data["label"]).squeeze().astype(np.int32)

    return signals, labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Multirate sliding window (no resampling)
# ─────────────────────────────────────────────────────────────────────────────

def multirate_sliding_windows(
    signals:    dict[str, np.ndarray],
    labels:     np.ndarray,
    subject_id: str,
    window_sec: float = WINDOW_SEC,
    step_sec:   float = STEP_SEC,
) -> list[dict[str, Any]]:

    total_duration_sec = len(labels) / LABEL_FS

    windows: list[dict[str, Any]] = []
    start_time = 0.0

    while True:
        end_time = start_time + window_sec

        # Stop before a partial window
        if end_time > total_duration_sec:
            break

        window: dict[str, Any] = {
            "subject":    subject_id,
            "start_time": start_time,
            "end_time":   end_time,
        }

        # Slice every sensor using its own index space
        for sensor, arr in signals.items():
            fs        = SENSOR_FS[sensor]
            start_idx = int(start_time * fs)
            end_idx   = min(int(end_time * fs), arr.shape[0])
            window[sensor] = arr[start_idx:end_idx]

        # Majority label over 700 Hz label stream (exclude label 0)
        lbl_start     = int(start_time * LABEL_FS)
        lbl_end       = min(int(end_time * LABEL_FS), len(labels))
        label_segment = labels[lbl_start:lbl_end]
        valid_labels  = label_segment[label_segment > 0]

        if len(valid_labels) == 0:
            start_time += step_sec
            continue

        majority_label     = int(stats.mode(valid_labels, keepdims=True).mode[0])
        window["final_label"] = majority_label

        windows.append(window)
        start_time += step_sec

    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Save one subject's windows to pickle
# ─────────────────────────────────────────────────────────────────────────────

def save_windows(windows: list[dict], subject_id: str) -> Path:
    out_dir  = OUTPUT_ROOT / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{subject_id}_windows.pkl"

    with open(out_path, "wb") as fh:
        pickle.dump(windows, fh)

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Process all subjects
# ─────────────────────────────────────────────────────────────────────────────

def process_all_subjects(subjects: list[str] = ALL_SUBJECTS) -> dict[str, list]:
    """
    Iterate over all subjects, run the windowing pipeline, save each to disk.

    Returns
    -------
    all_windows : dict mapping subject_id → list of window dicts
                  (everything in RAM if you need cross-subject access)
    """
    all_windows: dict[str, list] = {}
    failed:      list[str]       = []

    print(f"Processing {len(subjects)} subjects: {subjects}\n")
    print("=" * 64)

    for subject_id in subjects:
        print(f"\n[{subject_id}]  Loading …")

        try:
            # Load
            data = load_subject(subject_id)

            # Extract
            signals, labels = extract_signals(data)

            # Window
            windows = multirate_sliding_windows(
                signals    = signals,
                labels     = labels,
                subject_id = subject_id,
            )

            # Save
            out_path = save_windows(windows, subject_id)

            all_windows[subject_id] = windows

            # Per-subject summary
            label_names = {1: "baseline", 2: "stress", 3: "amusement"}
            unique, counts = np.unique(
                [w["final_label"] for w in windows], return_counts=True
            )
            label_summary = "  |  ".join(
                f"{label_names.get(int(l), l)}: {c}"
                for l, c in zip(unique, counts)
            )
            print(f"[{subject_id}]  {len(windows)} windows  →  {label_summary}")
            print(f"[{subject_id}]  Saved → {out_path}")

        except FileNotFoundError as e:
            print(f"[{subject_id}]  SKIPPED — {e}")
            failed.append(subject_id)

        except Exception as e:
            print(f"[{subject_id}]  FAILED  — {e}")
            failed.append(subject_id)

    # Final summary
    print("\n" + "=" * 64)
    print(f"Done.  Processed: {len(all_windows)}  |  "
          f"Skipped/Failed: {len(failed)}")
    if failed:
        print(f"Failed subjects: {failed}")

    print("\nOutput structure:")
    for subject_id in all_windows:
        path = OUTPUT_ROOT / subject_id / f"{subject_id}_windows.pkl"
        print(f"  {path}")

    return all_windows



# How to reload a saved subject later


def load_subject_windows(subject_id: str) -> list[dict]:
    """
    Load a previously saved subject's window list from disk.

    Usage
    -----
    windows = load_subject_windows("S2")
    print(windows[0]["final_label"])
    print(windows[0]["chest_ECG"].shape)
    """
    pkl_path = OUTPUT_ROOT / subject_id / f"{subject_id}_windows.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"No saved windows found for {subject_id} at {pkl_path}\n"
            f"Run process_all_subjects() first."
        )

    with open(pkl_path, "rb") as fh:
        windows = pickle.load(fh)

    print(f"[load]  {subject_id}  →  {len(windows)} windows loaded from {pkl_path}")
    return windows



# Entry point


if __name__ == "__main__":
    all_windows = process_all_subjects()
    