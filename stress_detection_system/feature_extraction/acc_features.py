"""
================================================================================
Accelerometer (ACC) Features Module
================================================================================
Extracts movement features from 3-axis ACC signals (chest 700 Hz, wrist 32 Hz).

Features extracted per axis (x, y, z)
    mean, std, min, max, range, abs_integral

Features from magnitude  Mag = sqrt(x² + y² + z²)
    mag_mean          average movement intensity
    mag_std           variability of movement
    mag_energy        sum of squared magnitude (total kinetic proxy)
    mag_peak_freq     dominant frequency of motion (FFT)
    mag_spectral_entropy  disorder in frequency spectrum
================================================================================
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_acc_features(
    signal:      np.ndarray,
    signal_name: str,
    fs:          float,
) -> dict[str, float]:
    """
    Extract features from a 3-axis ACC window.

    Parameters
    ----------
    signal      : array of shape (N, 3)  — columns = [x, y, z]
                  Also accepts (N,) for single-axis fallback.
    signal_name : prefix  e.g. "chest_acc", "wrist_acc"
    fs          : sampling frequency (700 for chest, 32 for wrist)

    Returns
    -------
    flat dict  e.g. {"chest_acc_x_mean": 0.01, "chest_acc_mag_energy": 312.4, ...}
    """
    signal = np.asarray(signal, dtype=np.float64)

    # Handle both (N,3) and (N,) inputs
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
        axes   = ["x"]
    elif signal.ndim == 2 and signal.shape[1] == 3:
        axes   = ["x", "y", "z"]
    else:
        return _nan_dict(signal_name)

    if signal.shape[0] == 0:
        return _nan_dict(signal_name)

    features: dict[str, float] = {}

    # ── Per-axis statistical features ─────────────────────────────────────
    for i, axis in enumerate(axes):
        col  = signal[:, i]
        t    = np.arange(len(col)) / fs
        name = f"{signal_name}_{axis}"

        features[f"{name}_mean"]         = float(np.mean(col))
        features[f"{name}_std"]          = float(np.std(col, ddof=1) if len(col) > 1 else 0.0)
        features[f"{name}_min"]          = float(np.min(col))
        features[f"{name}_max"]          = float(np.max(col))
        features[f"{name}_range"]        = float(np.max(col) - np.min(col))
        features[f"{name}_abs_integral"] = float(np.trapezoid(np.abs(col), t))

    # ── Magnitude features ─────────────────────────────────────────────────
    if len(axes) == 3:
        mag = np.sqrt(np.sum(signal ** 2, axis=1))
    else:
        mag = np.abs(signal[:, 0])

    t   = np.arange(len(mag)) / fs

    features[f"{signal_name}_mag_mean"]   = float(np.mean(mag))
    features[f"{signal_name}_mag_std"]    = float(np.std(mag, ddof=1) if len(mag) > 1 else 0.0)
    features[f"{signal_name}_mag_energy"] = float(np.sum(mag ** 2))

    # Dominant frequency and spectral entropy from magnitude
    freq_feats = _spectral_features(mag, fs, signal_name)
    features.update(freq_feats)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Spectral features on magnitude signal
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_features(
    mag:         np.ndarray,
    fs:          float,
    signal_name: str,
) -> dict[str, float]:

    if len(mag) < 16:
        return {
            f"{signal_name}_mag_peak_freq":        float("nan"),
            f"{signal_name}_mag_spectral_entropy": float("nan"),
        }

    nperseg = min(len(mag), 256)
    freqs, psd = welch(mag, fs=fs, nperseg=nperseg)

    # Dominant (peak) frequency
    peak_freq = float(freqs[np.argmax(psd)])

    # Spectral entropy — measures how spread the energy is across frequencies
    psd_norm = psd / (np.sum(psd) + 1e-12)
    spec_ent = float(scipy_entropy(psd_norm + 1e-12))

    return {
        f"{signal_name}_mag_peak_freq":        peak_freq,
        f"{signal_name}_mag_spectral_entropy": spec_ent,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NaN fallback
# ─────────────────────────────────────────────────────────────────────────────

def _nan_dict(signal_name: str) -> dict[str, float]:
    axis_keys  = [f"{signal_name}_{ax}_{s}"
                  for ax in ["x", "y", "z"]
                  for s  in ["mean", "std", "min", "max", "range", "abs_integral"]]
    mag_keys   = [f"{signal_name}_mag_{s}"
                  for s in ["mean", "std", "energy", "peak_freq", "spectral_entropy"]]
    return {k: float("nan") for k in axis_keys + mag_keys}
