import numpy as np

def moving_average(signal, window_size=21):
    """
    Compute centered moving average of a 1D signal.
    
    Parameters:
        signal (np.array): Input signal
        window_size (int): Must be odd for centered averaging
    
    Returns:
        smoothed (np.array): Smoothed signal
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd for centered averaging.")
    
    # Pad the signal at the edges
    pad = window_size // 2
    padded = np.pad(signal, (pad, pad), mode='edge')
    
    # Create moving average
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded, kernel, mode='valid')
    
    return smoothed


from scipy.signal import hilbert
import matplotlib.pyplot as plt

# Simulate signal
fs = 50
t = np.linspace(0, 30, fs * 30)
signal = np.sin(2 * np.pi * 0.3 * t) + 0.2 * np.random.randn(len(t))

# Hilbert envelope
envelope = np.abs(hilbert(signal))

# Smoothed envelope
smoothed = moving_average(envelope, window_size=51)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t, envelope, label='Envelope', alpha=0.5)
plt.plot(t, smoothed, label='Smoothed Envelope', linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Smoothed Hilbert Envelope with Moving Average")
plt.grid(True)
plt.tight_layout()
plt.show()
