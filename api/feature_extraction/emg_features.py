"""
================================================================================
EMG (Electromyography) Features Module
================================================================================
EMG measures electrical activity produced by skeletal muscles.
Under stress, muscle tension increases — especially in the trapezius (shoulder)
and frontalis (forehead) muscles where the WESAD chest device sits.

Signal characteristics
----------------------
- Raw EMG is a zero-mean, high-frequency, high-amplitude burst signal
- Resting muscle: low amplitude, low activity
- Tense / stressed muscle: high amplitude, frequent bursts, higher frequency content

Features extracted
------------------
Time-domain (amplitude & energy)
    mean_abs          mean absolute value — average muscle activation level
    rms               root mean square — signal power, standard EMG metric
    iemg              integrated EMG (sum of absolute values × time step)
                      — total muscle work in the window
    var               variance of the signal — spread of activation
    waveform_length   sum of absolute differences between consecutive samples
                      — captures signal complexity and speed of change

Frequency-domain (muscle fatigue & activation pattern)
    mean_freq         spectral mean frequency — shifts down with muscle fatigue
    median_freq       frequency that divides spectrum into two equal halves
                      — most robust fatigue indicator in literature
    peak_freq         dominant frequency component
    total_power       total power across all frequencies

Peak activity features
    n_bursts          number of muscle activation bursts above threshold
    burst_amp_mean    mean amplitude of detected bursts
    burst_duration    mean duration of activation bursts (seconds)

Reference
---------
Phinyomark A, et al. "Feature reduction and selection for EMG signal
classification." Expert Systems with Applications, 2012.
================================================================================
"""

import numpy as np
from scipy.signal import welch, find_peaks


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_emg_features(
    signal:      np.ndarray,
    signal_name: str,
    fs:          float,
) -> dict[str, float]:
    """
    Extract time-domain, frequency-domain, and burst features from EMG.

    Parameters
    ----------
    signal      : 1-D raw EMG array for one 60s window (chest_EMG at 700 Hz)
    signal_name : prefix for output keys  e.g. "chest_emg"
    fs          : sampling frequency (700 Hz for WESAD chest EMG)

    Returns
    -------
    flat dict  e.g. {"chest_emg_rms": 0.023, "chest_emg_median_freq": 112.4, ...}
    """
    signal = np.asarray(signal, dtype=np.float64).flatten()

    if len(signal) == 0 or np.all(np.isnan(signal)):
        return _nan_dict(signal_name)

    # Remove NaN samples before processing
    signal = signal[~np.isnan(signal)]

    if len(signal) < int(fs * 2):
        # Need at least 2 seconds of data for meaningful features
        return _nan_dict(signal_name)

    return {
        **_time_domain_features(signal, signal_name, fs),
        **_frequency_domain_features(signal, signal_name, fs),
        **_burst_features(signal, signal_name, fs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Time-domain features
# ─────────────────────────────────────────────────────────────────────────────

def _time_domain_features(
    signal:      np.ndarray,
    signal_name: str,
    fs:          float,
) -> dict[str, float]:
    """
    Amplitude and energy features computed directly on the raw signal.
    These are the standard features from Phinyomark et al. 2012.
    """
    n  = len(signal)
    dt = 1.0 / fs   # time step between samples

    # Mean Absolute Value — average rectified amplitude
    # Captures average muscle activation level
    mean_abs = float(np.mean(np.abs(signal)))

    # RMS — root mean square, the standard power metric for EMG
    # Higher RMS = more forceful muscle contraction
    rms = float(np.sqrt(np.mean(signal ** 2)))

    # IEMG — Integrated EMG, total muscle work in the window
    # Sum of absolute values multiplied by time step
    iemg = float(np.sum(np.abs(signal)) * dt)

    # Variance — spread of activation values
    var = float(np.var(signal, ddof=1))

    # Waveform Length — cumulative length of the signal waveform
    # Captures complexity, speed of change, and frequency content jointly
    # Computed as sum of absolute first differences
    waveform_length = float(np.sum(np.abs(np.diff(signal))))

    return {
        f"{signal_name}_mean_abs":        mean_abs,
        f"{signal_name}_rms":             rms,
        f"{signal_name}_iemg":            iemg,
        f"{signal_name}_var":             var,
        f"{signal_name}_waveform_length": waveform_length,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-domain features
# ─────────────────────────────────────────────────────────────────────────────

def _frequency_domain_features(
    signal:      np.ndarray,
    signal_name: str,
    fs:          float,
) -> dict[str, float]:
    """
    Spectral features that capture muscle fatigue and activation patterns.

    Mean and median frequency are the two primary fatigue indicators:
    - During sustained muscle contraction, fatigue shifts spectrum to lower freqs
    - Stressed subjects show higher frequency content (more rapid, tense activation)
    """
    # Use Welch method for robust PSD estimate
    # nperseg controls frequency resolution — longer = more resolution
    nperseg    = min(len(signal), int(fs * 2))  # 2-second segments
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    # Restrict to physiologically meaningful EMG band: 10–500 Hz
    # Below 10 Hz is motion artifact, above 500 Hz is noise for surface EMG
    band_mask = (freqs >= 10) & (freqs <= 500)
    freqs_b   = freqs[band_mask]
    psd_b     = psd[band_mask]

    if len(psd_b) == 0 or np.sum(psd_b) == 0:
        return _nan_freq_dict(signal_name)

    total_power = float(np.trapezoid(psd_b, freqs_b))

    # Mean frequency — power-weighted average frequency
    # Σ(f × PSD) / Σ(PSD)
    mean_freq = float(np.sum(freqs_b * psd_b) / np.sum(psd_b))

    # Median frequency — frequency that splits spectrum into two equal halves
    # Find where cumulative power reaches 50% of total
    cumulative_power = np.cumsum(psd_b)
    half_power       = cumulative_power[-1] / 2.0
    median_idx       = np.searchsorted(cumulative_power, half_power)
    median_freq      = float(freqs_b[min(median_idx, len(freqs_b) - 1)])

    # Peak frequency — frequency with maximum power
    peak_freq = float(freqs_b[np.argmax(psd_b)])

    return {
        f"{signal_name}_mean_freq":   mean_freq,
        f"{signal_name}_median_freq": median_freq,
        f"{signal_name}_peak_freq":   peak_freq,
        f"{signal_name}_total_power": total_power,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Burst detection features
# ─────────────────────────────────────────────────────────────────────────────

def _burst_features(
    signal:      np.ndarray,
    signal_name: str,
    fs:          float,
) -> dict[str, float]:
    """
    Detect and characterise muscle activation bursts.

    A burst is a sustained period where the rectified EMG exceeds a
    threshold (mean + 1 std). More bursts = more muscle tension events.
    Higher burst amplitude = more forceful contractions.

    Under stress, subjects show more frequent and higher-amplitude bursts
    especially in the trapezius / shoulder region.
    """
    # Rectify signal (take absolute value) — standard EMG processing step
    rectified = np.abs(signal)

    # Threshold: mean + 1 standard deviation
    # Bursts are periods of above-average activation
    threshold = float(np.mean(rectified) + np.std(rectified))

    # Find peaks in rectified signal that exceed threshold
    # min_distance = 0.1s × fs to avoid counting same burst twice
    min_distance = max(1, int(0.1 * fs))
    peak_indices, properties = find_peaks(
        rectified,
        height=threshold,
        distance=min_distance,
    )

    n_bursts = len(peak_indices)

    if n_bursts == 0:
        return {
            f"{signal_name}_n_bursts":       0.0,
            f"{signal_name}_burst_amp_mean": 0.0,
            f"{signal_name}_burst_duration": float("nan"),
        }

    # Mean amplitude of burst peaks
    burst_amplitudes = rectified[peak_indices]
    burst_amp_mean   = float(np.mean(burst_amplitudes))

    # Estimate burst duration using half-maximum width
    # Width in samples where signal > threshold around each peak
    total_above = np.sum(rectified > threshold)
    if n_bursts > 0:
        avg_burst_duration = float((total_above / n_bursts) / fs)  # seconds
    else:
        avg_burst_duration = float("nan")

    return {
        f"{signal_name}_n_bursts":       float(n_bursts),
        f"{signal_name}_burst_amp_mean": burst_amp_mean,
        f"{signal_name}_burst_duration": avg_burst_duration,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NaN fallbacks
# ─────────────────────────────────────────────────────────────────────────────

def _nan_freq_dict(signal_name: str) -> dict[str, float]:
    return {
        f"{signal_name}_mean_freq":   float("nan"),
        f"{signal_name}_median_freq": float("nan"),
        f"{signal_name}_peak_freq":   float("nan"),
        f"{signal_name}_total_power": float("nan"),
    }


def _nan_dict(signal_name: str) -> dict[str, float]:
    keys = [
        "mean_abs", "rms", "iemg", "var", "waveform_length",
        "mean_freq", "median_freq", "peak_freq", "total_power",
        "n_bursts", "burst_amp_mean", "burst_duration",
    ]
    return {f"{signal_name}_{k}": float("nan") for k in keys}
