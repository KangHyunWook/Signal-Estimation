from scipy.ndimage import uniform_filter1d
from scipy.signal import convolve

import numpy as np


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