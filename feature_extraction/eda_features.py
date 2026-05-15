"""
================================================================================
Electrodermal Activity (EDA) Features Module
================================================================================
Decomposes EDA into:
    SCL  — Tonic component   (slow baseline, long-term arousal)
    SCR  — Phasic component  (fast spikes, acute stress responses)

Uses neurokit2's eda_process() for decomposition and peak detection.

Features extracted
------------------
Tonic (SCL)
    scl_mean       mean baseline conductance
    scl_std        std of baseline
    scl_slope      trend direction (rising = increasing arousal)

Phasic (SCR)
    n_scr          number of SCR peaks (stress events)
    scr_amp_mean   mean peak amplitude
    scr_amp_std    std of peak amplitudes
    scr_auc        area under the phasic curve (total arousal energy)
    scr_rise_mean  mean rise time of peaks
================================================================================
"""

import numpy as np
import neurokit2 as nk
from scipy.stats import linregress


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_eda_features(
    signal:      np.ndarray,
    signal_name: str,
    fs:          float,
) -> dict[str, float]:
    """
    Extract tonic (SCL) and phasic (SCR) features from one EDA window.

    Parameters
    ----------
    signal      : 1-D raw EDA array (µS) for one 60s window
    signal_name : prefix for output keys  e.g. "chest_eda", "wrist_eda"
    fs          : sampling frequency (700 for chest EDA, 4 for wrist EDA)

    Returns
    -------
    flat dict  e.g. {"chest_eda_scl_mean": 0.83, "chest_eda_n_scr": 3, ...}
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()

    if len(signal) == 0 or np.all(np.isnan(signal)):
        return _nan_dict(signal_name)

    # Minimum length check — neurokit2 needs enough samples to decompose
    min_samples = int(fs * 10)   # at least 10 seconds
    if len(signal) < min_samples:
        return _nan_dict(signal_name)

    try:
        eda_df, info = nk.eda_process(signal, sampling_rate=int(fs))
    except Exception:
        return _nan_dict(signal_name)

    tonic_feats  = _extract_scl(eda_df, signal_name, fs)
    phasic_feats = _extract_scr(eda_df, info, signal_name, fs)

    return {**tonic_feats, **phasic_feats}


# ─────────────────────────────────────────────────────────────────────────────
# Tonic component (SCL)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_scl(
    eda_df:      object,       # pandas DataFrame from nk.eda_process
    signal_name: str,
    fs:          float,
) -> dict[str, float]:

    try:
        scl = eda_df["EDA_Tonic"].values.astype(np.float64)
    except KeyError:
        return {
            f"{signal_name}_scl_mean":  float("nan"),
            f"{signal_name}_scl_std":   float("nan"),
            f"{signal_name}_scl_slope": float("nan"),
        }

    t = np.arange(len(scl)) / fs

    if len(scl) > 1:
        slope, _, _, _, _ = linregress(t, scl)
    else:
        slope = float("nan")

    return {
        f"{signal_name}_scl_mean":  float(np.mean(scl)),
        f"{signal_name}_scl_std":   float(np.std(scl, ddof=1) if len(scl) > 1 else 0.0),
        f"{signal_name}_scl_slope": float(slope),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phasic component (SCR)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_scr(
    eda_df:      object,
    info:        dict,
    signal_name: str,
    fs:          float,
) -> dict[str, float]:

    try:
        phasic = eda_df["EDA_Phasic"].values.astype(np.float64)
    except KeyError:
        return _nan_scr_dict(signal_name)

    # Area under the phasic curve = total phasic energy in window
    t   = np.arange(len(phasic)) / fs
    auc = float(np.trapezoid(np.abs(phasic), t))

    # SCR peaks from neurokit2
    peak_indices = info.get("SCR_Peaks", np.array([]))
    n_scr        = len(peak_indices)

    if n_scr == 0:
        return {
            f"{signal_name}_n_scr":         0.0,
            f"{signal_name}_scr_amp_mean":  0.0,
            f"{signal_name}_scr_amp_std":   0.0,
            f"{signal_name}_scr_auc":       auc,
            f"{signal_name}_scr_rise_mean": float("nan"),
        }

    # Amplitudes at peak indices
    peak_indices  = np.array(peak_indices, dtype=int)
    peak_indices  = peak_indices[peak_indices < len(phasic)]
    amplitudes    = phasic[peak_indices]

    # Rise times — neurokit2 stores onset indices in info["SCR_Onsets"]
    onset_indices = info.get("SCR_Onsets", np.array([]))
    if len(onset_indices) == len(peak_indices) and len(onset_indices) > 0:
        rise_times = (peak_indices - np.array(onset_indices, dtype=int)) / fs
        rise_mean  = float(np.mean(rise_times[rise_times >= 0]))
    else:
        rise_mean  = float("nan")

    return {
        f"{signal_name}_n_scr":         float(n_scr),
        f"{signal_name}_scr_amp_mean":  float(np.mean(amplitudes)),
        f"{signal_name}_scr_amp_std":   float(np.std(amplitudes, ddof=1) if n_scr > 1 else 0.0),
        f"{signal_name}_scr_auc":       auc,
        f"{signal_name}_scr_rise_mean": rise_mean,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NaN fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def _nan_scr_dict(signal_name: str) -> dict[str, float]:
    return {
        f"{signal_name}_n_scr":         float("nan"),
        f"{signal_name}_scr_amp_mean":  float("nan"),
        f"{signal_name}_scr_amp_std":   float("nan"),
        f"{signal_name}_scr_auc":       float("nan"),
        f"{signal_name}_scr_rise_mean": float("nan"),
    }


def _nan_dict(signal_name: str) -> dict[str, float]:
    return {
        f"{signal_name}_scl_mean":      float("nan"),
        f"{signal_name}_scl_std":       float("nan"),
        f"{signal_name}_scl_slope":     float("nan"),
        **_nan_scr_dict(signal_name),
    }
