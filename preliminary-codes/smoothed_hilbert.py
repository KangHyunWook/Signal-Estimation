import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt

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

path=r"C:\workspace\NYUAD\raw_IMU_DATA\GT_IMU_pattern1.csv"

chest_signals=[]
t=[]
for line in open(path):
    line=line.strip()
    splits = line.split(',')

    if splits[0]=='Time':
        print('splits[1]:',splits[1])
        continue
    if splits[0]=='300':
        break
    t.append(float(splits[0]))
    chest_signals.append(float(splits[1]))

    # print(splits[0], splits[1])

print('che:', len(chest_signals))

# Simulated respiratory-like signal
fs = 50  # Hz

# Compute smoothed Hilbert envelope
envelope_smoothed = compute_smoothed_hilbert(chest_signals, fs)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(t, chest_signals, label='Original Signal')
plt.plot(t, envelope_smoothed, label='Smoothed Hilbert Envelope', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Smoothed Hilbert Envelope of Respiratory Signal')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('smoothed_hilbert.png')
plt.show()