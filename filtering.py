from scipy.ndimage import uniform_filter1d
from scipy.signal import convolve
from scipy.signal import butter, filtfilt, find_peaks,hilbert
from scipy.signal import detrend

import numpy as np

def compute_smoothed_hilbert(signal, fs, cutoff=0.5):
    """
    Computes the smoothed Hilbert envelope of a signal.

    Parameters:
        signal (array): Input time series
        fs (float): Sampling frequency in Hz
        cutoff (float): Cutoff frequency for smoothing (Hz)

    Returns:
        envelope_smoothed (array): Smoothed Hilbert envelope
    """
    # Hilbert transform to get the analytic signal
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)

    # Low-pass filter to smooth the envelope
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(2, normal_cutoff, btype='low')
    envelope_smoothed = filtfilt(b, a, envelope)

    return envelope_smoothed

def adaptive_bandpass_filter(signal, true_rr_bpm, config, bandwidth=0.1, order=4):
    """
    Apply an adaptive band-pass filter centered on the true respiratory frequency.

    Parameters:
    - signal: 1D array-like, IMU signal
    - fs: Sampling rate in Hz
    - true_rr_bpm: Ground truth respiratory rate in breaths per minute
    - bandwidth: Half-width of the filter band in Hz (default ±0.1 Hz)
    - order: Filter order (default 4)
    Returns:
    - filtered_signal: Signal after adaptive band-pass filtering
    """
    fs=config.fs

    true_freq = true_rr_bpm / 60.0  # Convert BPM to Hz
    
    out=true_freq-bandwidth

    lowcut = max(true_freq - bandwidth, 0.05)  # Avoid going below 0 Hz
    highcut = min(true_freq + bandwidth, fs / 2 - 0.1)  # Avoid Nyquist
    

    
    # Band-pass Butterworth filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    # Optional: remove trend first
    # detrended = detrend(signal)
    

    # Apply filter
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def bandpass_filter(signal, config, order=4):
    nyq = 0.5 * config.fs
    low = config.lowcut / nyq
    high = config.highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def smoothing(signal, config):
    ma_window=config.ma_window
    signal= signal - uniform_filter1d(signal, size=ma_window, axis=0, mode='nearest')
    long_window = int(config.long_window_size * config.fs)  # e.g., 2s * 50Hz = 100 samples
    signal = uniform_filter1d(signal, size=long_window, axis=0, mode='nearest')

    return signal
    
def triangle_filter(signal, config):
    """
    Apply a triangular filter with a width of 2 seconds to the input signal.

    Parameters:
    - signal: 1D numpy array, the input signal.
    - fs: int, sampling frequency in Hz.

    Returns:
    - filtered_signal: 1D numpy array, filtered output.
    """
    window_size = int(config.triangle_w_size * config.fs)  # 2 seconds
    if window_size < 1:
        raise ValueError("Sampling frequency too low or invalid")

    # Create a symmetric triangle window
    triangle = np.bartlett(window_size)

    # Normalize to preserve overall amplitude
    triangle /= triangle.sum()

    # Convolve signal with the triangle filter
    filtered_signal = convolve(signal, triangle, mode='same')

    return filtered_signal