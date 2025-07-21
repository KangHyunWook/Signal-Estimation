import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

file_path=r"C:\workspace\NYUAD\MIC_1000gain_facingEarCanal.csv"

# Simulate a noisy respiratory signal
fs = 500  # sampling frequency in Hz
t = np.linspace(0, 30, 30 * fs)  # 30 seconds
rr_true = 18  # breaths per minute
breaths_per_second = rr_true / 60
signal = np.sin(2 * np.pi * breaths_per_second * t) + 0.1 * np.random.randn(len(t))

# Filter the signal (optional but recommended)
def bandpass_filter(sig, fs, low=0.1, high=0.5):  # breathing is typically 0.1–0.5 Hz
    b, a = butter(2, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, sig)

signals =[]

for line in open(file_path):
    line = line.strip()
    splits = line.split(',')
    if splits[2]=='GT':
        continue
    print(splits[0], splits[2])
    if splits[0]=='30':
        break
    signals.append(float(splits[2]))

filtered_signal = bandpass_filter(signals, fs)

# Find peaks (e.g., inhalations)
print('f:', len(filtered_signal))

window_size=fs*3
rr_bpm_list=[]
for i in range(len(filtered_signal)//window_size):
    segment=filtered_signal[i*window_size:(i+1)*window_size]
    print('s:', segment.shape)
 
    peaks, _ = find_peaks(segment,distance=fs/2)  # at least 0.5 sec between peaks
    print('peak:',peaks)

    duration_sec=3
    print('d:', duration_sec)

    rr_bpm = (len(peaks) / duration_sec) * 60
    rr_bpm_list.append(rr_bpm)


print(len(rr_bpm_list))
# Plot histogram of respiratory rate estimates
plt.figure(figsize=(8, 5))
plt.hist(rr_bpm_list, bins=np.arange(5, 45, 2), color='skyblue', edgecolor='black')
plt.xlabel('Respiratory Rate (BPM)')
plt.ylabel('Frequency')
plt.title('Distribution of Estimated Respiratory Rates')
plt.grid(True)
plt.tight_layout()
plt.show()
