import pandas as pd
import os
BASE_PATH = "../WDM_dataset/Features"
MERGE_KEYS = ["subject", "window_idx", "start_time", "end_time", "label"]

def load_data(modality, feature_type):
    """
    Load a single parquet file based on modality and feature type.
    """

    path = os.path.join(BASE_PATH, modality, f"all_{modality}_{feature_type}.parquet")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_parquet(path)

    # basic validation
    for key in MERGE_KEYS:
        if key not in df.columns:
            raise ValueError(f"{key} missing in {path}")

    return df


def load_multiple_features(modality, feature_types):
    """
    Load and merge multiple feature files for a single modality.
    """

    if not feature_types:
        raise ValueError("feature_types list is empty")

    dfs = []

    for ft in feature_types:
        df = load_data(modality, ft)
        dfs.append(df)

    merged_df = dfs[0]

    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on=MERGE_KEYS)

    return merged_df


def load_dataset(config):

    if not config:
        raise ValueError("Config is empty")

    modality_dfs = []

    for modality, feature_types in config.items():
        df_mod = load_multiple_features(modality, feature_types)
        modality_dfs.append(df_mod)

    # merge across modalities
    final_df = modality_dfs[0]

    for df in modality_dfs[1:]:
        final_df = final_df.merge(df, on=MERGE_KEYS)
    
    final_df = binarize_labels(final_df)
    return final_df

def inspect_data(df):
    print("Shape:", df.shape)
    print("Unique subjects:", df["subject"].nunique())
    print("\nLabel distribution:\n", df["label"].value_counts())
    print("Total NaNs:", df.isna().sum().sum())

    print("\nColumns with NaNs:")
    print(df.isna().sum()[df.isna().sum() > 0])

# the function binaires the dataset into useful construct so we can use it for example collapsing everything except stress and baseline
# Convert dataset into binary stress classification:
# keep only baseline (1) and stress (2),
# remove amusement, meditation, and transition states.
# Final mapping:
# baseline -> 0
# stress   -> 1
def binarize_labels(df):
    # drop transient and meditation
    df = df[df["label"].isin([1, 2])].copy()
    
    # stress=1, non-stress=0
    df["label"] = df["label"].map({1: 0, 2: 1})
    
    return df

if __name__ == "__main__":


    config = {"wrist": ["statistical"]}

    df = load_dataset(config)
    inspect_data(df)