'''
Compute ground-truths heart rates 
'''

from scipy.signal import hilbert

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

path=r"C:\workspace\NYUAD\MIC_1000gain_facingEarCanal.csv"

chest_signals=[]
t=[]
for line in open(path):
    line=line.strip()
    splits = line.split(',')
    
    if splits[0]=='Time (s)':
        print('splits[1]:',splits[1])
        continue
    if splits[0]=='32':
        break
    t.append(float(splits[0]))
    chest_signals.append(float(splits[2])) #GT
    
print('c:', len(chest_signals))

def bandpass_filter(signal, fs, lowcut=0.1, highcut=0.5, order=4):
    """
    Bandpass filter to isolate heart rate components (e.g., 30–180 BPM).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)



fs=500
window_size = 2*fs
peaks_list=[]





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



def detect_heart_rate(signal, fs,i):
    """
    Detect heart rate from filtered signal.
    
    Parameters:
        signal (1D array): Input ECG or PPG signal
        fs (int): Sampling frequency in Hz
        plot (bool): If True, plot signal and detected peaks
    
    Returns:
        hr_bpm (float): Estimated heart rate in BPM
        peaks (array): Indices of detected peaks
    """
    # signal = compute_smoothed_hilbert(signal, fs)
    # signal = bandpass_filter(signal, fs)
    # Step 2: Peak detection
    min_distance = int(fs * 0.5)  # Min 0.4s between beats (~150 BPM)
    peaks, _ = find_peaks(signal, distance=min_distance, prominence=0.8)
    peaks_list.extend(peaks+i*window_size)
    print('peaks:', peaks)
    # Step 3: Compute heart rate
    duration_sec = window_size #len(signal) / fs
    hr_bpm = (len(peaks) / duration_sec) * 60

    return hr_bpm, peaks


#compute heart rate (BPM) for a window size of 3 seconds


print('w:', window_size)

hr_bpm_list=[]
#fetch signal for a window size to compute heart rate
for i in range(len(chest_signals)//window_size):
    segment = chest_signals[i*window_size:(i+1)*window_size]
    hr_bpm, peaks = detect_heart_rate(segment, fs, i)
    print('hr_bpm:',hr_bpm)
    hr_bpm_list.append(hr_bpm)


# print((len(peaks_list)-6)/len(peaks_list)*100)
# exit()

t=np.arange(len(chest_signals))

plt.figure(figsize=(10, 4))
plt.xlabel('Indices for each window')
plt.ylabel('Heart rates (BPM)')
plt.title('Heart rates for a window size of 2 seconds from the chest belt (GT)')
plt.plot(t, chest_signals, label='Heart rates', linewidth=2)
plt.plot([t[p] for p in peaks_list], [chest_signals[p] for p in peaks_list], 'ro', label='Detected Breaths')
plt.grid(True)
plt.tight_layout()
plt.savefig('heart_rates.png')
plt.show()
