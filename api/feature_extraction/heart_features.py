"""
================================================================================
Heart Features Module  (ECG + BVP)
================================================================================
Extracts heart rate and HRV features from ECG (700 Hz) and BVP (64 Hz).

Uses neurokit2 for robust R-peak / peak detection.

Features extracted
------------------
Time-domain
    hr_mean       mean heart rate (bpm)
    hr_std        std of heart rate
    rr_mean       mean RR interval (ms)
    rr_std        std of RR intervals
    rmssd         root mean square of successive RR differences
    pnn50         % of successive RR differences > 50 ms

Frequency-domain (Welch PSD on RR tachogram)
    lf_power      power in low-frequency band  0.04–0.15 Hz
    hf_power      power in high-frequency band 0.15–0.40 Hz
    lf_hf_ratio   LF / HF  →  key stress marker
================================================================================
"""

import numpy as np
import neurokit2 as nk
from scipy.signal import welch
from scipy.interpolate import interp1d


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_heart_features(
    signal:      np.ndarray,
    signal_name: str,
    fs:          float,
) -> dict[str, float]:
    """
    Extract heart rate and HRV features from a single ECG or BVP window.

    Parameters
    ----------
    signal      : 1-D raw ECG or BVP array for one 60s window
    signal_name : prefix for output keys  e.g. "chest_ecg", "wrist_bvp"
    fs          : sampling frequency (700 for ECG, 64 for BVP)

    Returns
    -------
    flat dict of heart features  e.g. {"chest_ecg_hr_mean": 72.3, ...}
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()

    if len(signal) == 0 or np.all(np.isnan(signal)):
        return _nan_dict(signal_name)

    try:
        rr_ms = _detect_rr_intervals(signal, fs, signal_name)
    except Exception:
        return _nan_dict(signal_name)

    if rr_ms is None or len(rr_ms) < 4:
        # Too few beats to compute reliable HRV
        return _nan_dict(signal_name)

    return {
        **_time_domain_features(rr_ms, signal_name),
        **_frequency_domain_features(rr_ms, signal_name),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Peak detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_rr_intervals(
    signal:      np.ndarray,
    fs:          float,
    signal_name: str,
) -> np.ndarray | None:
    """
    Detect R-peaks (ECG) or systolic peaks (BVP) using neurokit2.
    Returns RR intervals in milliseconds.
    """
    if "bvp" in signal_name.lower():
        # BVP — use PPG peak detection
        _, info = nk.ppg_peaks(signal, sampling_rate=int(fs))
        peaks   = info["PPG_Peaks"]
    else:
        # ECG — use Pan-Tompkins R-peak detection
        _, info = nk.ecg_peaks(signal, sampling_rate=int(fs))
        peaks   = info["ECG_R_Peaks"]

    if len(peaks) < 2:
        return None

    # Convert peak sample indices → RR intervals in milliseconds
    rr_ms = np.diff(peaks) / fs * 1000.0

    # Physiological plausibility filter (RR between 300 ms and 2000 ms)
    rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]

    return rr_ms if len(rr_ms) >= 3 else None


# ─────────────────────────────────────────────────────────────────────────────
# Time-domain HRV
# ─────────────────────────────────────────────────────────────────────────────

def _time_domain_features(
    rr_ms:       np.ndarray,
    signal_name: str,
) -> dict[str, float]:

    hr       = 60_000.0 / rr_ms          # bpm per beat
    rr_diff  = np.diff(rr_ms)

    rmssd    = float(np.sqrt(np.mean(rr_diff ** 2)))
    nn50     = int(np.sum(np.abs(rr_diff) > 50))
    pnn50    = float(nn50 / len(rr_diff) * 100) if len(rr_diff) > 0 else float("nan")

    return {
        f"{signal_name}_hr_mean":  float(np.mean(hr)),
        f"{signal_name}_hr_std":   float(np.std(hr, ddof=1) if len(hr) > 1 else 0.0),
        f"{signal_name}_rr_mean":  float(np.mean(rr_ms)),
        f"{signal_name}_rr_std":   float(np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else 0.0),
        f"{signal_name}_rmssd":    rmssd,
        f"{signal_name}_nn50":     float(nn50),
        f"{signal_name}_pnn50":    pnn50,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-domain HRV  (Welch PSD on interpolated RR tachogram)
# ─────────────────────────────────────────────────────────────────────────────

def _frequency_domain_features(
    rr_ms:       np.ndarray,
    signal_name: str,
    interp_fs:   float = 4.0,     # standard HRV resampling rate
) -> dict[str, float]:

    # Build cumulative time axis for RR intervals (in seconds)
    rr_sec    = rr_ms / 1000.0
    t_rr      = np.cumsum(rr_sec)
    t_rr      = np.insert(t_rr, 0, 0)[:-1]   # align to beat onset

    total_dur = t_rr[-1]
    if total_dur < 10.0:
        # Not enough duration for reliable spectral estimate
        return _nan_freq_dict(signal_name)

    # Interpolate RR tachogram to uniform grid (4 Hz standard)
    t_uniform = np.arange(0, total_dur, 1.0 / interp_fs)
    interpolator = interp1d(t_rr, rr_sec, kind="cubic",
                            bounds_error=False, fill_value="extrapolate")
    rr_uniform = interpolator(t_uniform)

    # Welch PSD
    nperseg  = min(len(rr_uniform), int(interp_fs * 60))   # up to 60s segment
    freqs, psd = welch(rr_uniform, fs=interp_fs, nperseg=nperseg)

    lf_mask  = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask  = (freqs >= 0.15) & (freqs < 0.40)

    lf_power = float(np.trapezoid(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else float("nan")
    hf_power = float(np.trapezoid(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else float("nan")

    if hf_power and hf_power > 0:
        lf_hf = lf_power / hf_power
    else:
        lf_hf = float("nan")

    return {
        f"{signal_name}_lf_power":   lf_power,
        f"{signal_name}_hf_power":   hf_power,
        f"{signal_name}_lf_hf_ratio": lf_hf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NaN fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def _nan_dict(signal_name: str) -> dict[str, float]:
    return {
        **{k: float("nan") for k in [
            f"{signal_name}_hr_mean",
            f"{signal_name}_hr_std",
            f"{signal_name}_rr_mean",
            f"{signal_name}_rr_std",
            f"{signal_name}_rmssd",
            f"{signal_name}_nn50",
            f"{signal_name}_pnn50",
        ]},
        **_nan_freq_dict(signal_name),
    }


def _nan_freq_dict(signal_name: str) -> dict[str, float]:
    return {
        f"{signal_name}_lf_power":    float("nan"),
        f"{signal_name}_hf_power":    float("nan"),
        f"{signal_name}_lf_hf_ratio": float("nan"),
    }
