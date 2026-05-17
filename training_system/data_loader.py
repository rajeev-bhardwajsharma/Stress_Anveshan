"""
data_loader.py
--------------
Loads and merges feature parquet files from the WDM dataset.

Directory layout expected:
  <BASE_PATH>/
    chest/
      all_chest_acc.parquet
      all_chest_eda.parquet
      ...
    wrist/
      all_wrist_acc.parquet
      ...

Labels in the raw files:
  1 = baseline
  2 = stress
  3 = amusement
  4 = meditation
  0 = transition

After binarize_labels() only baseline(→0) and stress(→1) remain.
binarize_labels() is called ONCE inside load_dataset().
Do NOT call it again in train.py or benchmark.py.
"""

import os
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
# Change this one line if your dataset lives elsewhere.
# You can also override it with an environment variable:
#   export WDM_BASE_PATH=/data/my_dataset
BASE_PATH = os.environ.get("WDM_BASE_PATH", "../WDM_dataset/Features")

# Columns that identify a window — never used as ML features
MERGE_KEYS = ["subject", "window_idx", "start_time", "end_time", "label"]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_data(modality: str, feature_type: str) -> pd.DataFrame:
    """Load a single parquet file for one (modality, feature_type) pair."""
    path = os.path.join(BASE_PATH, modality, f"all_{modality}_{feature_type}.parquet")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_parquet(path)

    for key in MERGE_KEYS:
        if key not in df.columns:
            raise ValueError(f"Expected column '{key}' not found in {path}")

    return df


def load_multiple_features(modality: str, feature_types: list) -> pd.DataFrame:
    """Load and inner-join multiple feature files for one modality."""
    if not feature_types:
        raise ValueError("feature_types list is empty")

    dfs = [load_data(modality, ft) for ft in feature_types]

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=MERGE_KEYS)

    return merged


def load_dataset(config: dict) -> pd.DataFrame:
    """
    Load, merge, and binarize the dataset.

    Parameters
    ----------
    config : dict
        Maps modality → list of feature types.
        Example: {"chest": ["acc", "eda"], "wrist": ["acc"]}

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with binary labels (0=baseline, 1=stress).
        Transition, amusement, and meditation rows are dropped.
    """
    if not config:
        raise ValueError("Config dict is empty")

    modality_dfs = [
        load_multiple_features(modality, feature_types)
        for modality, feature_types in config.items()
    ]

    final_df = modality_dfs[0]
    for df in modality_dfs[1:]:
        final_df = final_df.merge(df, on=MERGE_KEYS)

    # Binarize exactly once here — do not repeat in benchmark.py / train.py
    final_df = binarize_labels(final_df)
    return final_df


# ── Label helpers ─────────────────────────────────────────────────────────────

def binarize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only baseline (label=1) and stress (label=2) rows.
    Remap:  baseline → 0,  stress → 1.

    All other labels (amusement=3, meditation=4, transition=0) are dropped.
    """
    df = df[df["label"].isin([1, 2])].copy()
    df["label"] = df["label"].map({1: 0, 2: 1})
    return df


# ── Debug helper ──────────────────────────────────────────────────────────────

def inspect_data(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("Unique subjects:", df["subject"].nunique())
    print("\nLabel distribution:\n", df["label"].value_counts())
    print("Total NaNs:", df.isna().sum().sum())
    nan_cols = df.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if not nan_cols.empty:
        print("\nColumns with NaNs:\n", nan_cols)


# ── Quick smoke-test ──────────────────────────────────────────────────────────
"""
if __name__ == "__main__":
    config = {"wrist": ["statistical"]}
    df = load_dataset(config)
    inspect_data(df)
    """