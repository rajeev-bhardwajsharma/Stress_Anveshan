"""
================================================================================
WESAD — Complete Visualization Suite
================================================================================
Run this ONCE from the feature_extraction/ folder after feature extraction:

    python3 visualize_all.py

Produces every meaningful graph organized into sections:
    Section 1 — Dataset Overview       (class distribution, subject counts)
    Section 2 — Signal Quality         (raw signal samples per condition)
    Section 3 — Feature Distributions  (how features differ stress vs non-stress)
    Section 4 — Correlation Analysis   (feature redundancy, modality overlap)
    Section 5 — Subject Variability    (how much subjects differ from each other)
    Section 6 — Feature Importance     (variance, mutual information ranking)
    Section 7 — Modality Comparison    (chest vs wrist, sensor contributions)

All plots saved to:
    WDM_dataset/Visualizations/
        01_dataset_overview/
        02_signal_quality/
        03_feature_distributions/
        04_correlation_analysis/
        05_subject_variability/
        06_feature_importance/
        07_modality_comparison/
================================================================================
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.stats import f_oneway
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import gc

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths and global settings
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path.cwd() / "WDM_dataset"

INTERIM_ROOT = BASE_DIR / "Interim"
FEATURE_ROOT = BASE_DIR / "Features"
VIZ_ROOT     = BASE_DIR / "Visualizations"

ALL_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]

# Human-readable label names
LABEL_NAMES  = {1: "Baseline", 2: "Stress", 3: "Amusement"}
LABEL_COLORS = {1: "#2196F3", 2: "#F44336", 3: "#4CAF50"}  # blue, red, green

# Consistent style across all plots
plt.rcParams.update({
    "figure.dpi":       100,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
# ─────────────────────────────────────────────────────────────────────────────
# Global Settings & Debugging
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "DEBUG": False,        # If True, only uses a small sample of data
    "VERBOSE": True,      # Print detailed progress
    "SAMPLE_SIZE": 100,   # Number of rows to use in DEBUG mode
    "SINGLE_SUBJECT": "S2" # Only process this subject for signal plots
}
PALETTE = sns.color_palette("Set2", 3)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def save(fig: plt.Figure, folder: str, filename: str) -> None:
    """Save figure to the correct subfolder and close it to free memory."""
    out_dir = VIZ_ROOT / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    plt.close('all')
    plt.clf()         # Clear the current figure
    gc.collect()      # Force Python to give RAM back to the computer
    print(f"  saved → {path.relative_to(VIZ_ROOT.parent)}")


def load_all_features(bundle: str) -> pd.DataFrame | None:
    """
    Load the merged master parquet for a given bundle key.
    e.g. bundle = "chest_heart"  →  Features/chest/all_chest_heart.parquet
    """
    device = bundle.split("_")[0]   # "chest" or "wrist"
    path   = FEATURE_ROOT / device / f"all_{bundle}.parquet"
    if not path.exists():
        if CONFIG["VERBOSE"]: print(f"  [Error] Path missing: {path}")
        return None
    

    df= pd.read_parquet(path)
    if len(df) > 5000:
        df = df.sample(frac=0.1, random_state=42)
    #for debugging purpose purposes
    if CONFIG["DEBUG"]:
        df=df.sample(n=min(CONFIG["SAMPLE_SIZE"],len(df)))

# --- THE CRITICAL FIX --- for error  unsupported operand type(s) for -: 'str' and 'str'
    # 1. Identify which columns are features (the ones that should be numbers)
    fcols = feature_cols(df)
    
    # 2. Force these columns to be numeric. 
    # If there is a "string" hidden in there, it becomes NaN (Not a Number)
    for col in fcols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. Fill any resulting NaNs with 0 or the column mean so the math doesn't break
    df[fcols] = df[fcols].fillna(0)
    # ------------------------

    return df

    


def load_windows_for_subject(subject_id: str) -> list[dict] | None:
    """Load raw window pkl for signal quality plots."""
    path = INTERIM_ROOT / subject_id / f"{subject_id}_windows.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Return only feature columns — everything except metadata."""
    meta = {"subject", "window_idx", "start_time", "end_time", "label"}
    return [c for c in df.columns if c not in meta]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Dataset Overview
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution() -> None:
    """
    Bar chart showing how many windows belong to each class (Baseline /
    Stress / Amusement) across the entire dataset.
    Reveals class imbalance — important context for why SMOTE is needed.
    """
    df = load_all_features("chest_heart")
    if df is None:
        return

    # Count windows per label
    counts = df["label"].value_counts().sort_index()
    labels = [LABEL_NAMES.get(i, str(i)) for i in counts.index]
    colors = [LABEL_COLORS.get(i, "#999") for i in counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.5)

    # Annotate each bar with its count and percentage
    total = counts.sum()
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{count}\n({count/total:.1%})",
                ha="center", va="bottom", fontsize=10)

    ax.set_title("Class Distribution Across All Subjects\n"
                 "(shows imbalance — justifies SMOTE in training folds)")
    ax.set_ylabel("Number of 60-second Windows")
    ax.set_xlabel("Condition")
    ax.set_ylim(0, counts.max() * 1.2)

    save(fig, "01_dataset_overview", "class_distribution.png")


def plot_windows_per_subject() -> None:
    """
    Horizontal bar chart: how many windows each subject contributed.
    Shows that some subjects recorded longer sessions — subject variability
    in data quantity.
    """
    df = load_all_features("chest_heart")
    if df is None:
        return

    per_subject = df.groupby("subject")["window_idx"].count().sort_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(per_subject.index, per_subject.values,
                   color="#5C6BC0", edgecolor="white")

    # Label each bar
    for bar, val in zip(bars, per_subject.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)

    ax.set_title("Windows per Subject\n"
                 "(each window = 60s with 50% overlap)")
    ax.set_xlabel("Number of Windows")
    ax.set_ylabel("Subject ID")
    ax.axvline(per_subject.mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"Mean = {per_subject.mean():.0f}")
    ax.legend()

    save(fig, "01_dataset_overview", "windows_per_subject.png")


def plot_label_distribution_per_subject() -> None:
    """
    Stacked bar chart: label breakdown per subject.
    Shows that the stress/baseline/amusement ratio varies across subjects.
    This is why LOSO is necessary — you cannot assume uniform label
    distributions across subjects.
    """
    df = load_all_features("chest_heart")
    if df is None:
        return

    # Pivot: subjects as rows, labels as columns
    pivot = df.groupby(["subject", "label"]).size().unstack(fill_value=0)
    pivot.columns = [LABEL_NAMES.get(c, str(c)) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(
        kind="bar", stacked=True, ax=ax,
        color=[LABEL_COLORS[1], LABEL_COLORS[2], LABEL_COLORS[3]],
        edgecolor="white", linewidth=0.5,
    )

    ax.set_title("Label Distribution per Subject\n"
                 "(motivates LOSO — each subject has a unique distribution)")
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Number of Windows")
    ax.legend(title="Condition", bbox_to_anchor=(1.01, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    save(fig, "01_dataset_overview", "label_per_subject.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Raw Signal Quality
# ─────────────────────────────────────────────────────────────────────────────

def plot_raw_signals_one_subject(subject_id: str = "S2") -> None:
    """
    Multi-panel plot showing 10 seconds of each raw signal from one subject
    during Baseline vs Stress side by side.
    This is the most intuitive plot for a professor — shows WHY physiological
    signals look different under stress.
    """
    windows = load_windows_for_subject(subject_id)
    if windows is None:
        return

    # Find one baseline window and one stress window
    baseline_win = next((w for w in windows if w["final_label"] == 1), None)
    stress_win   = next((w for w in windows if w["final_label"] == 2), None)

    if baseline_win is None or stress_win is None:
        print(f"  [skip] could not find both label types for {subject_id}")
        return

    # Signals to display and their sampling rates and units
    signals_to_plot = [
        ("chest_ECG",  700, "ECG (mV)",         "Heart Electrical Activity"),
        ("chest_EDA",  700, "EDA (µS)",          "Skin Conductance"),
        ("chest_EMG",  700, "EMG (mV)",          "Muscle Activity"),
        ("chest_RESP", 700, "RESP (a.u.)",       "Respiration"),
        ("chest_TEMP", 700, "TEMP (°C)",         "Skin Temperature"),
        ("wrist_BVP",   64, "BVP (a.u.)",        "Blood Volume Pulse"),
        ("wrist_EDA",    4, "w-EDA (µS)",        "Wrist Skin Conductance"),
    ]

    n_signals = len(signals_to_plot)
    fig, axes = plt.subplots(n_signals, 2,
                             figsize=(16, n_signals * 2.2),
                             sharex=False)

    display_sec = 10   # show first 10 seconds of each window

    for row, (sig_key, fs, ylabel, title) in enumerate(signals_to_plot):
        n_samples = int(display_sec * fs)

        for col, (win, condition, color) in enumerate([
            (baseline_win, "Baseline", LABEL_COLORS[1]),
            (stress_win,   "Stress",   LABEL_COLORS[2]),
        ]):
            ax  = axes[row, col]
            arr = win.get(sig_key)
            if arr is None:
                ax.text(0.5, 0.5, "Not available",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            arr = np.asarray(arr).flatten()[:n_samples]
            t   = np.arange(len(arr)) / fs

            ax.plot(t, arr, color=color, linewidth=0.8, alpha=0.9)
            ax.set_ylabel(ylabel, fontsize=9)

            if row == 0:
                ax.set_title(f"{condition}\n({subject_id})",
                             fontsize=12, color=color, fontweight="bold")
            if row == n_signals - 1:
                ax.set_xlabel("Time (seconds)")

            # Add signal name on leftmost column
            if col == 0:
                ax.text(-0.18, 0.5, title, transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=9, color="#555")

    fig.suptitle(f"Raw Physiological Signals — {subject_id}\n"
                 f"Baseline vs Stress (first {display_sec}s of window)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    save(fig, "02_signal_quality", f"raw_signals_{subject_id}.png")


def plot_acc_3axis(subject_id: str = "S2") -> None:
    """
    Three-panel plot showing x, y, z axes of chest ACC for baseline vs stress.
    Illustrates how body movement patterns differ between conditions.
    """
    windows = load_windows_for_subject(subject_id)
    if windows is None:
        return

    baseline_win = next((w for w in windows if w["final_label"] == 1), None)
    stress_win   = next((w for w in windows if w["final_label"] == 2), None)
    if baseline_win is None or stress_win is None:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharey="row")
    axes_labels = ["X-axis", "Y-axis", "Z-axis"]
    display_samples = 700 * 10   # 10 seconds at 700 Hz

    for col, (win, condition, color) in enumerate([
        (baseline_win, "Baseline", LABEL_COLORS[1]),
        (stress_win,   "Stress",   LABEL_COLORS[2]),
    ]):
        acc = np.asarray(win.get("chest_ACC", np.zeros((1, 3))))
        acc = acc[:display_samples]
        t   = np.arange(len(acc)) / 700

        for row in range(3):
            ax = axes[row, col]
            ax.plot(t, acc[:, row], color=color, linewidth=0.7, alpha=0.85)
            ax.set_ylabel(f"{axes_labels[row]} (g)")
            if row == 0:
                ax.set_title(condition, color=color, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Time (seconds)")

    fig.suptitle(f"Chest Accelerometer (3-Axis) — {subject_id}\n"
                 "Baseline vs Stress — captures body movement differences",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "02_signal_quality", f"acc_3axis_{subject_id}.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Feature Distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_boxplots(bundle: str, top_n: int = 8) -> None:
    """
    Box plots of the top N most discriminative features in a bundle,
    grouped by class label.
    Shows how stress shifts the distribution of each feature.
    Only plots binary: Baseline vs Stress (drops Amusement for clarity).
    """
    df = load_all_features(bundle)
    if df is None:
        return

    # Keep only Baseline and Stress for clean binary comparison
    df = df[df["label"].isin([1, 2])].copy()
    df["Condition"] = df["label"].map(LABEL_NAMES)

    fcols = feature_cols(df)
    if len(fcols) == 0:
        return

    # Select top N features by ANOVA F-statistic between Baseline and Stress
    f_scores = {}
    for col in fcols:
        grp_base   = df.loc[df["label"] == 1, col].dropna()
        grp_stress = df.loc[df["label"] == 2, col].dropna()
        if len(grp_base) > 1 and len(grp_stress) > 1:
            f_val, _ = f_oneway(grp_base, grp_stress)
            f_scores[col] = f_val if not np.isnan(f_val) else 0

    if not f_scores:
        return

    top_features = sorted(f_scores, key=f_scores.get, reverse=True)[:top_n]

    n_cols = 4
    n_rows = int(np.ceil(top_n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        # Box plot with individual data points overlaid
        sns.boxplot(
            data=df, x="Condition", y=feat, ax=ax,
            palette={LABEL_NAMES[1]: LABEL_COLORS[1],
                     LABEL_NAMES[2]: LABEL_COLORS[2]},
            width=0.5, linewidth=1.2,
            flierprops=dict(marker="o", markersize=2, alpha=0.3),
        )
        # Clean up feature name for display
        display_name = feat.replace("_", " ").replace("chest ", "").replace("wrist ", "w-")
        ax.set_title(f"{display_name}\n(F={f_scores[feat]:.1f})", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Top {top_n} Most Discriminative Features — {bundle}\n"
                 "Ranked by ANOVA F-score (Baseline vs Stress)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "03_feature_distributions", f"boxplots_{bundle}.png")


def plot_violin_key_features() -> None:
    """
    Violin plots of the most physiologically important stress features:
    RMSSD, LF/HF ratio, EDA n_scr, EMG RMS.
    Violins show the full distribution shape — more informative than box plots
    for showing bimodal distributions or skewed data.
    """
    # Gather key features from different bundles
    key_features_map = {
        "chest_heart":  ["chest_ecg_rmssd", "chest_ecg_lf_hf_ratio"],
        "chest_eda":    ["chest_eda_n_scr",  "chest_eda_scl_mean"],
        "chest_emg":    ["chest_emg_rms",    "chest_emg_n_bursts"],
        "wrist_heart":  ["wrist_bvp_rmssd"],
    }

    collected = []
    for bundle, features in key_features_map.items():
        df = load_all_features(bundle)
        if df is None:
            continue
        df_sub = df[["label"] + [f for f in features if f in df.columns]].copy()
        collected.append(df_sub)

    if not collected:
        return

    # Merge all on index (they should align since same windows)
    merged = collected[0]
    for other in collected[1:]:
        merged = merged.merge(other, on="label", how="outer")

    merged = merged[merged["label"].isin([1, 2])].copy()
    merged["Condition"] = merged["label"].map(LABEL_NAMES)

    feat_cols_available = [c for c in merged.columns
                           if c not in ("label", "Condition")][:8]

    if not feat_cols_available:
        return

    n_cols = 4
    n_rows = int(np.ceil(len(feat_cols_available) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(feat_cols_available):
        ax = axes[i]
        subset = merged[["Condition", feat]].dropna()
        if len(subset) == 0:
            continue
        sns.violinplot(
            data=subset, x="Condition", y=feat, ax=ax,
            palette={LABEL_NAMES[1]: LABEL_COLORS[1],
                     LABEL_NAMES[2]: LABEL_COLORS[2]},
            inner="box", linewidth=1.0,
        )
        display = feat.replace("_", " ").replace("chest ", "").replace("wrist ", "w-")
        ax.set_title(display, fontsize=9)
        ax.set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution of Key Stress Features (Baseline vs Stress)\n"
                 "Violin = full distribution, box inside = IQR + median",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "03_feature_distributions", "violin_key_features.png")


def plot_feature_means_radar() -> None:
    """
    Radar / spider chart comparing normalised feature means across
    Baseline vs Stress for a selected set of interpretable features.
    Gives a quick visual of which dimensions separate the two conditions.
    """
    bundles_and_features = [
        ("chest_heart", "chest_ecg_hr_mean"),
        ("chest_heart", "chest_ecg_rmssd"),
        ("chest_heart", "chest_ecg_lf_hf_ratio"),
        ("chest_eda",   "chest_eda_n_scr"),
        ("chest_eda",   "chest_eda_scl_mean"),
        ("chest_emg",   "chest_emg_rms"),
        ("wrist_heart", "wrist_bvp_hr_mean"),
        ("wrist_eda",   "wrist_eda_n_scr"),
    ]

    # Load each feature
    values = {}
    for bundle, feat in bundles_and_features:
        df = load_all_features(bundle)
        if df is None or feat not in df.columns:
            continue
        for label in [1, 2]:
            sub = df.loc[df["label"] == label, feat].dropna()
            if len(sub) > 0:
                values.setdefault(feat, {})[label] = sub.mean()

    available = [(b, f) for b, f in bundles_and_features
                 if f in values and 1 in values[f] and 2 in values[f]]
    if len(available) < 3:
        return

    feats  = [f for _, f in available]
    labels = [f.replace("chest_ecg_", "ECG ").replace("chest_eda_", "EDA ")
               .replace("chest_emg_", "EMG ").replace("wrist_bvp_", "wBVP ")
               .replace("wrist_eda_", "wEDA ") for f in feats]

    # Normalise to [0, 1] across both conditions for fair radar display
    vals_base   = np.array([values[f][1] for f in feats])
    vals_stress = np.array([values[f][2] for f in feats])

    combined_min = np.minimum(vals_base, vals_stress)
    combined_max = np.maximum(vals_base, vals_stress)
    rng          = combined_max - combined_min
    rng[rng == 0] = 1   # avoid division by zero

    norm_base   = (vals_base   - combined_min) / rng
    norm_stress = (vals_stress - combined_min) / rng

    N      = len(feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    norm_base_c   = norm_base.tolist()   + norm_base[:1].tolist()
    norm_stress_c = norm_stress.tolist() + norm_stress[:1].tolist()

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True))

    ax.plot(angles, norm_base_c,   "o-", linewidth=2,
            color=LABEL_COLORS[1], label="Baseline")
    ax.fill(angles, norm_base_c,   alpha=0.15, color=LABEL_COLORS[1])

    ax.plot(angles, norm_stress_c, "o-", linewidth=2,
            color=LABEL_COLORS[2], label="Stress")
    ax.fill(angles, norm_stress_c, alpha=0.15, color=LABEL_COLORS[2])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_yticklabels([])
    ax.set_title("Normalised Feature Means — Baseline vs Stress\n"
                 "(each axis normalised to [0,1] independently)",
                 pad=20, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    save(fig, "03_feature_distributions", "radar_feature_means.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Correlation Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_intra_bundle_correlation(bundle: str, max_features: int = 25) -> None:
    """
    Heatmap of Pearson correlation between features within one bundle.
    High correlation (>0.9) means redundant features that can be removed.
    This is your visual argument for needing feature selection.
    """
    df = load_all_features(bundle)
    if df is None:
        return

    fcols = feature_cols(df)[:max_features]
    if len(fcols) < 2:
        return

    corr = df[fcols].corr()

    # Shorten column names for display
    short_names = [c.replace("chest_", "").replace("wrist_", "w_")
                     .replace("_mean", "_μ").replace("_std", "_σ")
                   for c in fcols]

    fig, ax = plt.subplots(figsize=(max(10, len(fcols) * 0.5),
                                    max(8,  len(fcols) * 0.45)))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle

    sns.heatmap(
        corr, mask=mask, annot=False, fmt=".1f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        xticklabels=short_names, yticklabels=short_names,
        linewidths=0.3, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title(f"Feature Correlation Matrix — {bundle}\n"
                 "Dark red = high positive correlation (redundant features)\n"
                 "Dark blue = high negative correlation",
                 fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    save(fig, "04_correlation_analysis", f"corr_heatmap_{bundle}.png")


def plot_cross_modality_correlation() -> None:
    """
    Heatmap showing correlation between KEY features from DIFFERENT modalities.
    Answers: do ECG and BVP measure the same thing?
    Do chest EDA and wrist EDA agree?
    This motivates the modality ablation study.
    """
    # Select representative features from each modality
    feature_bundles = {
        "ECG-HR":    ("chest_heart", "chest_ecg_hr_mean"),
        "ECG-RMSSD": ("chest_heart", "chest_ecg_rmssd"),
        "ECG-LF/HF": ("chest_heart", "chest_ecg_lf_hf_ratio"),
        "BVP-HR":    ("wrist_heart", "wrist_bvp_hr_mean"),
        "BVP-RMSSD": ("wrist_heart", "wrist_bvp_rmssd"),
        "cEDA-SCL":  ("chest_eda",   "chest_eda_scl_mean"),
        "wEDA-SCL":  ("wrist_eda",   "wrist_eda_scl_mean"),
        "cEDA-nSCR": ("chest_eda",   "chest_eda_n_scr"),
        "wEDA-nSCR": ("wrist_eda",   "wrist_eda_n_scr"),
        "EMG-RMS":   ("chest_emg",   "chest_emg_rms"),
    }

    # Build a DataFrame with one column per cross-modality feature
    frames = {}
    ref_idx = None
    for label, (bundle, feat) in feature_bundles.items():
        df = load_all_features(bundle)
        if df is None or feat not in df.columns:
            continue
        # Use window_idx + subject as join key
        df_sub = df[["subject", "window_idx", feat]].copy()
        df_sub = df_sub.set_index(["subject", "window_idx"])
        frames[label] = df_sub[feat]

    if len(frames) < 3:
        return

    combined = pd.DataFrame(frames).dropna()
    corr     = combined.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        annot_kws={"size": 9},
    )
    ax.set_title("Cross-Modality Feature Correlation\n"
                 "ECG vs BVP: do chest and wrist capture the same heart signal?\n"
                 "Chest EDA vs Wrist EDA: do they agree?",
                 fontsize=11)

    save(fig, "04_correlation_analysis", "cross_modality_correlation.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Subject Variability
# ─────────────────────────────────────────────────────────────────────────────

def plot_subject_feature_variability() -> None:
    """
    Line plot: mean value of key features per subject, coloured by condition.
    Shows inter-subject variability — some subjects have very different
    baseline HR, EDA etc. This is the core motivation for LOSO:
    a model trained on S2-S16 must generalise to S17 despite these differences.
    """
    key_features = [
        ("chest_heart", "chest_ecg_hr_mean",    "ECG Mean HR (bpm)"),
        ("chest_heart", "chest_ecg_rmssd",      "ECG RMSSD (ms)"),
        ("chest_eda",   "chest_eda_scl_mean",   "Chest EDA SCL Mean (µS)"),
        ("chest_emg",   "chest_emg_rms",        "EMG RMS"),
    ]

    n = len(key_features)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))

    for ax, (bundle, feat, ylabel) in zip(axes, key_features):
        df = load_all_features(bundle)
        if df is None or feat not in df.columns:
            continue

        df_bin = df[df["label"].isin([1, 2])].copy()
        df_bin["Condition"] = df_bin["label"].map(LABEL_NAMES)

        # Compute per-subject per-condition mean
        means = df_bin.groupby(["subject", "Condition"])[feat].mean().reset_index()

        # Plot each condition as a separate line across subjects
        for condition, color in [(LABEL_NAMES[1], LABEL_COLORS[1]),
                                  (LABEL_NAMES[2], LABEL_COLORS[2])]:
            sub = means[means["Condition"] == condition].sort_values("subject")
            ax.plot(sub["subject"], sub[feat], "o-", color=color,
                    label=condition, linewidth=1.5, markersize=6)

        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("Subject")
        ax.set_ylabel(ylabel.split("(")[0].strip())
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)

    fig.suptitle("Inter-Subject Variability in Key Features\n"
                 "Each point = one subject's mean. Lines connect subjects.\n"
                 "Large spread = high variability — motivates LOSO evaluation",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "05_subject_variability", "subject_feature_variability.png")


def plot_subject_stress_delta() -> None:
    """
    Bar chart: stress - baseline delta for each subject for key features.
    A large positive delta means a subject shows a strong stress response.
    A delta near zero means the feature does not change much for that person.
    This shows WHY generalisation is hard — stress responses differ by person.
    """
    key_features = [
        ("chest_heart", "chest_ecg_hr_mean",  "ΔHR (bpm)"),
        ("chest_heart", "chest_ecg_rmssd",    "ΔRMSSD (ms)"),
        ("chest_eda",   "chest_eda_n_scr",    "Δ#SCR events"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (bundle, feat, ylabel) in zip(axes, key_features):
        df = load_all_features(bundle)
        if df is None or feat not in df.columns:
            continue

        # Compute mean per subject per condition, then delta
        means = df[df["label"].isin([1, 2])].groupby(
            ["subject", "label"])[feat].mean().unstack()

        if 1 not in means.columns or 2 not in means.columns:
            continue
        # Compute mean per subject per condition
        group_means = df[df["label"].isin([1, 2])].groupby(["subject", "label"])[feat].mean()
        
        # Use unstack but immediately force the result to be floating-point numbers
        means = group_means.unstack().astype(float)

        if 1 not in means.columns or 2 not in means.columns:
            continue

        # Now this math is guaranteed to be number minus number
        delta = (means[2] - means[1]).dropna().sort_values()

        colors = [LABEL_COLORS[2] if v > 0 else LABEL_COLORS[1]
                  for v in delta.values]

        ax.bar(delta.index, delta.values, color=colors,
               edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"{ylabel}\n(Stress − Baseline per Subject)")
        ax.set_xlabel("Subject")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.text(0.02, 0.97, "Red = stress higher\nBlue = stress lower",
                transform=ax.transAxes, fontsize=8,
                va="top", color="#555")

    fig.suptitle("Per-Subject Stress Response Delta\n"
                 "Positive = feature increases under stress | "
                 "Negative = decreases under stress",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "05_subject_variability", "subject_stress_delta.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_mutual_information_ranking(bundle: str, top_n: int = 20) -> None:
    """
    Horizontal bar chart ranking features by mutual information with the label.
    Mutual information captures non-linear relationships — better than
    correlation for physiological signals which are rarely purely linear.
    This is your pre-model feature importance evidence.
    """
    df = load_all_features(bundle)
    if df is None:
        return

    df_bin = df[df["label"].isin([1, 2])].copy()
    fcols  = feature_cols(df_bin)

    # Drop columns with too many NaNs
    df_clean = df_bin[fcols + ["label"]].dropna(thresh=len(df_bin) * 0.7)
    fcols    = [c for c in fcols if c in df_clean.columns]

    # Impute remaining NaNs with column median for MI computation
    X = df_clean[fcols].fillna(df_clean[fcols].median())
    y = df_clean["label"]

    if X.shape[1] == 0 or len(y) == 0:
        return

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=fcols).sort_values(ascending=True)
    top       = mi_series.tail(top_n)

    # Clean names for display
    short = [n.replace("chest_", "").replace("wrist_", "w_")
              .replace("_mean", "_μ").replace("_std", "_σ")
             for n in top.index]

    fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.4)))
    bars = ax.barh(range(len(top)), top.values,
                   color=sns.color_palette("viridis", len(top)),
                   edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Mutual Information Score")
    ax.set_title(f"Feature Importance by Mutual Information — {bundle}\n"
                 f"Top {top_n} features (higher = more predictive of stress label)\n"
                 "Captures both linear and non-linear relationships",
                 fontsize=11)

    # Annotate scores
    for i, (bar, val) in enumerate(zip(bars, top.values)):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    save(fig, "06_feature_importance",
         f"mutual_information_{bundle}.png")


def plot_variance_across_features(bundle: str) -> None:
    """
    Bar chart of feature variances across all windows.
    Features with near-zero variance are constant and useless — they should
    be removed before training. This plot identifies them visually.
    """
    df = load_all_features(bundle)
    if df is None:
        return

    fcols = feature_cols(df)
    var   = df[fcols].var().sort_values(ascending=False)

    # Normalise for display
    var_norm = var / (var.max() + 1e-10)

    # Colour by threshold: red if near zero (useless), green otherwise
    colors = ["#EF5350" if v < 0.01 else "#66BB6A" for v in var_norm]
    short  = [n.replace("chest_", "").replace("wrist_", "w_")
               for n in var.index]

    fig, ax = plt.subplots(figsize=(max(12, len(fcols) * 0.35), 5))
    ax.bar(range(len(var)), var_norm.values, color=colors, edgecolor="none")
    ax.axhline(0.01, color="red", linestyle="--", linewidth=1,
               label="Low-variance threshold (0.01)")
    ax.set_xticks(range(len(var)))
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_ylabel("Normalised Variance")
    ax.set_title(f"Feature Variance — {bundle}\n"
                 "Red bars = near-zero variance → useless features to drop\n"
                 "Green bars = sufficient variance → potentially useful",
                 fontsize=11)
    ax.legend()

    save(fig, "06_feature_importance", f"variance_{bundle}.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Modality Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_chest_vs_wrist_heart() -> None:
    """
    Scatter plot: ECG-derived HR (chest) vs BVP-derived HR (wrist)
    coloured by condition.
    If both sensors agree perfectly, points lie on the y=x diagonal.
    Deviation tells you how reliable wrist measurement is.
    """
    ecg_df = load_all_features("chest_heart")
    bvp_df = load_all_features("wrist_heart")
    if ecg_df is None or bvp_df is None:
        return

    # Join on subject + window_idx
    merged = ecg_df[["subject", "window_idx", "label", "chest_ecg_hr_mean"]].merge(
        bvp_df[["subject", "window_idx", "wrist_bvp_hr_mean"]],
        on=["subject", "window_idx"],
    ).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, label_id in zip(axes, [1, 2]):
        sub    = merged[merged["label"] == label_id]
        color  = LABEL_COLORS[label_id]
        cname  = LABEL_NAMES[label_id]

        ax.scatter(sub["chest_ecg_hr_mean"], sub["wrist_bvp_hr_mean"],
                   alpha=0.5, s=20, color=color, label=cname)

        # Identity line (y = x) — perfect agreement
        lim = [merged[["chest_ecg_hr_mean", "wrist_bvp_hr_mean"]].min().min() - 2,
               merged[["chest_ecg_hr_mean", "wrist_bvp_hr_mean"]].max().max() + 2]
        ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5, label="Perfect agreement")

        # Correlation
        r = sub["chest_ecg_hr_mean"].corr(sub["wrist_bvp_hr_mean"])
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                va="top", fontsize=11, color=color)

        ax.set_title(f"{cname} Condition")
        ax.set_xlabel("Chest ECG — Mean HR (bpm)")
        ax.set_ylabel("Wrist BVP — Mean HR (bpm)")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.legend(fontsize=8)

    fig.suptitle("Chest ECG vs Wrist BVP — Heart Rate Agreement\n"
                 "Points on the diagonal = perfect agreement between devices\n"
                 "Deviation = wrist sensor measurement error",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "07_modality_comparison", "chest_vs_wrist_hr.png")


def plot_eda_chest_vs_wrist() -> None:
    """
    Paired box plot comparing chest EDA vs wrist EDA for SCL mean and n_scr.
    Shows whether the 4 Hz wrist sensor captures the same information
    as the 700 Hz chest sensor.
    """
    c_df = load_all_features("chest_eda")
    w_df = load_all_features("wrist_eda")
    if c_df is None or w_df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_pairs = [
        ("chest_eda_scl_mean", "wrist_eda_scl_mean",
         "EDA SCL Mean (µS)", "Tonic Component"),
        ("chest_eda_n_scr",    "wrist_eda_n_scr",
         "EDA n_scr (count)", "Phasic Peaks"),
    ]

    for ax, (c_feat, w_feat, ylabel, title) in zip(axes, plot_pairs):
        merged = c_df[["label", c_feat]].merge(
            w_df[["label", w_feat]],
            left_index=True, right_index=True,
            suffixes=("_l", "_r"),
        ).dropna()

        # Reshape to long format for grouped box plot
        label_col = "label_l" if "label_l" in merged.columns else "label"
        long = pd.melt(
            merged.rename(columns={c_feat: "Chest", w_feat: "Wrist",
                                   label_col: "label"}),
            id_vars="label",
            value_vars=["Chest", "Wrist"],
            var_name="Device", value_name="Value",
        )
        long = long[long["label"].isin([1, 2])].copy()
        long["Condition"] = long["label"].map(LABEL_NAMES)

        sns.boxplot(
            data=long, x="Condition", y="Value", hue="Device",
            ax=ax, palette={"Chest": "#FF7043", "Wrist": "#42A5F5"},
            width=0.5, linewidth=1.0,
        )
        ax.set_title(f"{title}\nChest (700 Hz) vs Wrist (4 Hz)")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")

    fig.suptitle("Chest EDA vs Wrist EDA — Do Both Sensors Agree?\n"
                 "Closer boxes = sensors capture similar information",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save(fig, "07_modality_comparison", "eda_chest_vs_wrist.png")


def plot_modality_f_score_summary() -> None:
    """
    Summary bar chart: mean ANOVA F-score per modality.
    Aggregates discriminative power of all features in each modality.
    This is your single-panel answer to "which sensor matters most?"
    before running the full ablation study with a trained model.
    """
    modality_bundles = {
        "ECG (Heart)":    "chest_heart",
        "Chest EDA":      "chest_eda",
        "EMG (Muscle)":   "chest_emg",
        "Chest ACC":      "chest_acc",
        "Chest Stat":     "chest_statistical",
        "BVP (Heart)":    "wrist_heart",
        "Wrist EDA":      "wrist_eda",
        "Wrist ACC":      "wrist_acc",
        "Wrist Stat":     "wrist_statistical",
    }

    modality_scores = {}

    for name, bundle in modality_bundles.items():
        df = load_all_features(bundle)
        if df is None:
            continue

        df_bin = df[df["label"].isin([1, 2])].copy()
        fcols  = feature_cols(df_bin)
        scores = []

        for col in fcols:
            grp1 = df_bin.loc[df_bin["label"] == 1, col].dropna()
            grp2 = df_bin.loc[df_bin["label"] == 2, col].dropna()
            if len(grp1) > 1 and len(grp2) > 1:
                f_val, _ = f_oneway(grp1, grp2)
                if not np.isnan(f_val):
                    scores.append(f_val)

        if scores:
            modality_scores[name] = np.mean(scores)

    if not modality_scores:
        return

    # Sort descending
    sorted_scores = dict(sorted(modality_scores.items(),
                                key=lambda x: x[1], reverse=True))

    # Colour chest vs wrist differently
    colors = ["#FF7043" if "Wrist" not in k and "BVP" not in k
              else "#42A5F5"
              for k in sorted_scores]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sorted_scores.keys(), sorted_scores.values(),
                  color=colors, edgecolor="white", linewidth=1)

    for bar, val in zip(bars, sorted_scores.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Mean ANOVA F-Score per Modality\n"
                 "Higher = modality features better separate Stress vs Baseline\n"
                 "Orange = Chest | Blue = Wrist",
                 fontsize=12)
    ax.set_ylabel("Mean F-Score across all features in modality")
    ax.set_xlabel("Modality")
    plt.xticks(rotation=25, ha="right")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#FF7043", label="Chest (RespiBAN)"),
                       Patch(facecolor="#42A5F5", label="Wrist (Empatica E4)")]
    ax.legend(handles=legend_elements, fontsize=9)

    save(fig, "07_modality_comparison", "modality_f_score_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Master runner — calls every plot in order
# ─────────────────────────────────────────────────────────────────────────────

import gc

def run_all_visualizations() -> None:
    """
    Orchestrates the entire visualization suite with memory management
    to prevent computer freezing.
    """
    print("="*80)
    print("      WESAD VISUALIZATION SUITE — AUTOMATED RUNNER")
    print("="*80)

    # --- SECTION 1: DATASET OVERVIEW ---
    try:
        print("\n── Section 1: Dataset Overview ────────────────────────")
        plot_class_distribution()
        plot_windows_per_subject()
        plot_label_distribution_per_subject()
    except Exception as e:
        print(f"  [ERROR] Section 1 failed: {e}")
    finally:
        gc.collect() # Free RAM

    # --- SECTION 2: SIGNAL QUALITY ---
    try:
        print("\n── Section 2: Raw Signal Quality ────────────────────────────")
        plot_raw_signals_one_subject("S2")
        plot_acc_3axis("S2")
    except Exception as e:
        print(f"  [ERROR] Section 2 failed: {e}")
    finally:
        gc.collect()

    # --- SECTION 3: DISTRIBUTIONS ---
    try:
        print("\n── Section 3: Feature Distributions ────────────────────────")
        # List of bundles to process
        dist_bundles = ["chest_heart", "chest_eda", "chest_emg", 
                        "chest_acc", "wrist_heart", "wrist_eda"]
        for bundle in dist_bundles:
            plot_feature_boxplots(bundle, top_n=8)
            gc.collect() # Clean RAM after every single plot in the loop
        
        plot_violin_key_features()
        plot_feature_means_radar()
    except Exception as e:
        print(f"  [ERROR] Section 3 failed: {e}")
    finally:
        gc.collect()

    # --- SECTION 4: CORRELATIONS ---
    try:
        print("\n── Section 4: Correlation Analysis ─────────────────────────")
        corr_bundles = ["chest_heart", "chest_eda", "chest_emg", 
                        "chest_statistical", "wrist_statistical"]
        for bundle in corr_bundles:
            plot_intra_bundle_correlation(bundle)
            gc.collect()
        plot_cross_modality_correlation()
    except Exception as e:
        print(f"  [ERROR] Section 4 failed: {e}")
    finally:
        gc.collect()

    # --- SECTION 5: VARIABILITY ---
    try:
        print("\n── Section 5: Subject Variability ───────────────────────────")
        plot_subject_feature_variability()
        plot_subject_stress_delta()
    except Exception as e:
        print(f"  [ERROR] Section 5 failed: {e}")
    finally:
        gc.collect()

    # --- SECTION 6: IMPORTANCE ---
    try:
        print("\n── Section 6: Feature Importance ────────────────────────────")
        imp_bundles = ["chest_heart", "chest_eda", "chest_emg", "wrist_heart"]
        for bundle in imp_bundles:
            plot_mutual_information_ranking(bundle, top_n=15)
            plot_variance_across_features(bundle)
            gc.collect()
    except Exception as e:
        print(f"  [ERROR] Section 6 failed: {e}")
    finally:
        gc.collect()

    # --- SECTION 7: COMPARISON ---
    try:
        print("\n── Section 7: Modality Comparison ───────────────────────────")
        plot_chest_vs_wrist_heart()
        plot_eda_chest_vs_wrist()
        plot_modality_f_score_summary()
    except Exception as e:
        print(f"  [ERROR] Section 7 failed: {e}")
    finally:
        gc.collect()

    print("\n" + "="*80)
    print(f"DONE! All plots saved in: {VIZ_ROOT}")
    print("="*80)
if __name__ == "__main__":
    print(FEATURE_ROOT.resolve())
    run_all_visualizations()
