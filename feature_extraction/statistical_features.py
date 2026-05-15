"""
================================================================================
Statistical Features Module
================================================================================
Computes basic statistical descriptors for ANY 1-D signal window.
Run this on every signal: ECG, EDA, EMG, RESP, TEMP, BVP, and each ACC axis.

Returns a flat dict with keys prefixed by the caller-supplied signal_name.
================================================================================
"""

import numpy as np
from scipy.stats import skew, kurtosis


def extract_statistical_features(
    signal: np.ndarray,
    signal_name: str,
    fs: float,
) -> dict[str, float]:
    """
    Extract basic statistical features from a 1-D signal window.

    Parameters
    ----------
    signal      : 1-D numpy array of raw samples
    signal_name : prefix for all output keys  e.g. "chest_ecg", "wrist_eda"
    fs          : sampling frequency of the signal (Hz)

    Returns
    -------
    flat dict  e.g. {"chest_ecg_mean": 0.02, "chest_ecg_std": 0.18, ...}
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()

    # Guard: if window is empty or all-NaN return NaNs
    if len(signal) == 0 or np.all(np.isnan(signal)):
        return _nan_dict(signal_name)

    # Remove NaNs for computation
    x = signal[~np.isnan(signal)]
    n = len(x)

    # ── Time axis for slope calculation ───────────────────────────────────
    t = np.arange(n) / fs   # seconds

    # Linear slope via least-squares fit
    if n > 1:
        slope = float(np.polyfit(t, x, 1)[0])
    else:
        slope = float("nan")

    features = {
        f"{signal_name}_mean":         float(np.mean(x)),
        f"{signal_name}_std":          float(np.std(x, ddof=1) if n > 1 else 0.0),
        f"{signal_name}_min":          float(np.min(x)),
        f"{signal_name}_max":          float(np.max(x)),
        f"{signal_name}_range":        float(np.max(x) - np.min(x)),
        f"{signal_name}_slope":        slope,
        f"{signal_name}_abs_integral": float(np.trapezoid(np.abs(x), t)),
        f"{signal_name}_skewness":     float(skew(x)        if n > 2 else float("nan")),
        f"{signal_name}_kurtosis":     float(kurtosis(x)    if n > 2 else float("nan")),
        f"{signal_name}_p25":          float(np.percentile(x, 25)),
        f"{signal_name}_p75":          float(np.percentile(x, 75)),
        f"{signal_name}_iqr":          float(np.percentile(x, 75) - np.percentile(x, 25)),
        f"{signal_name}_rms":          float(np.sqrt(np.mean(x ** 2))),
    }

    return features


def extract_statistical_features_all_signals(
    signals: dict[str, np.ndarray],
    sensor_fs: dict[str, int],
) -> dict[str, float]:
    """
    Convenience wrapper: run extract_statistical_features on every signal
    in the window dict and merge all results into one flat dict.

    Parameters
    ----------
    signals   : { "chest_ECG": array, "wrist_EDA": array, ... }
                ACC arrays shape (N, 3) are split into _x, _y, _z axes.
    sensor_fs : sampling rates per sensor key

    Returns
    -------
    one merged flat dict of all statistical features
    """
    all_features: dict[str, float] = {}

    for sensor_key, arr in signals.items():
        fs         = sensor_fs.get(sensor_key, 1)
        name       = sensor_key.lower()   # e.g. "chest_ecg"
        arr        = np.asarray(arr)

        if arr.ndim == 2 and arr.shape[1] == 3:
            # ACC — process each axis independently
            for axis_idx, axis_label in enumerate(["x", "y", "z"]):
                axis_name = f"{name}_{axis_label}"
                feats = extract_statistical_features(
                    arr[:, axis_idx], axis_name, fs
                )
                all_features.update(feats)

            # Also compute magnitude
            mag       = np.sqrt(np.sum(arr ** 2, axis=1))
            mag_feats = extract_statistical_features(mag, f"{name}_mag", fs)
            all_features.update(mag_feats)

        else:
            feats = extract_statistical_features(arr.flatten(), name, fs)
            all_features.update(feats)

    return all_features


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nan_dict(signal_name: str) -> dict[str, float]:
    keys = [
        "mean", "std", "min", "max", "range", "slope",
        "abs_integral", "skewness", "kurtosis",
        "p25", "p75", "iqr", "rms",
    ]
    return {f"{signal_name}_{k}": float("nan") for k in keys}
