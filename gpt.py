import numpy as np
from scipy.signal import convolve

def triangle_filter(signal, fs):
    """
    Apply a triangular filter with a width of 2 seconds to the input signal.

    Parameters:
    - signal: 1D numpy array, the input signal.
    - fs: int, sampling frequency in Hz.

    Returns:
    - filtered_signal: 1D numpy array, filtered output.
    """
    window_size = int(2 * fs)  # 2 seconds
    if window_size < 1:
        raise ValueError("Sampling frequency too low or invalid")

    # Create a symmetric triangle window
    triangle = np.bartlett(window_size)

    # Normalize to preserve overall amplitude
    triangle /= triangle.sum()

    # Convolve signal with the triangle filter
    filtered_signal = convolve(signal, triangle, mode='same')

    return filtered_signal


fs = 50  # 50 Hz sampling rate
t = np.linspace(0, 10, fs*10)  # 10-second signal
resp_signal = np.sin(2 * np.pi * 0.25 * t)  # Simulated 0.25 Hz respiratory signal
print('resp_s:', resp_signal.shape)


filtered = triangle_filter(resp_signal, fs)

print('f:', filtered.shape)

