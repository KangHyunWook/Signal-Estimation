from filtering import smoothing, triangle_filter
from scipy.signal import butter, filtfilt, find_peaks
from scipy.signal import detrend

import numpy as np


from scipy.signal import butter, filtfilt, detrend

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

    lowcut = max(true_freq - bandwidth, 0.05)  # Avoid going below 0 Hz
    highcut = min(true_freq + bandwidth, fs / 2 - 0.1)  # Avoid Nyquist
    
    # Band-pass Butterworth filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    # Optional: remove trend first
    detrended = detrend(signal)
    

    # Apply filter
    filtered_signal = filtfilt(b, a, detrended)
    return filtered_signal


def bandpass_filter(signal, config, order=4):
    nyq = 0.5 * config.fs
    low = config.lowcut / nyq
    high = config.highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def estimate_rr_peak_detection(signal_data, RR_GT_list, config):
    """
    Estimate respiratory rate using peak detection.

    Parameters:
    - signal_data: 1D array-like, IMU signal (e.g., one accelerometer axis)
    - fs: int, sampling frequency in Hz (default 50 Hz)

    Returns:
    - rr_bpm: float, estimated respiratory rate in breaths per minute
    """
    # Detrend and bandpass filter to isolate breathing frequency range
    fs = config.fs
    signal_data = detrend(signal_data)
    # filtered = bandpass_filter(signal_data, fs)
    
    total_len = len(signal_data)
    rrs=[]
    gt_idx=0
    window_size=config.fs*config.window_sec
    for start in range(0, total_len, window_size):
        end = start + window_size
        print("start:", start)
        print('end:', end)
        
        if end > total_len:
            break

        segment = signal_data[start:end]
        
        if config.bp:
            if config.adaptive_bp:
                segment = adaptive_bandpass_filter(segment, RR_GT_list)
            else: 
                segment = bandpass_filter(segment, config)

        #todo:
        # Detect peaks (assumed to correspond to inhalation peaks)
        min_distance = int(fs * 2)  # minimum 2 seconds between breaths (i.e., max 30 BPM)
        peaks, _ = find_peaks(segment, distance=min_distance)

        # If not enough peaks are found, return 0
        if len(peaks) < 2:
            return 0.0

        # Calculate intervals between peaks
        peak_intervals = np.diff(peaks) / fs  # in seconds
        avg_interval = np.mean(peak_intervals)

        # RR = 60 / average breathing period
        rr_bpm = 60.0 / avg_interval
        rrs.append(rr_bpm)
    
    return rrs

def estimate_rr_fft(signal_data, RR_GT_list, config):
    """
    Estimate respiratory rate using FFT.

    Parameters:
    - signal_data: 1D array-like, the IMU signal (e.g., from one axis of accelerometer or gyroscope)
    - fs: int, sampling frequency in Hz (default is 50 Hz)

    Returns:
    - rr_bpm: float, estimated respiratory rate in breaths per minute
    """
    # Detrend signal to remove linear drift

    signal_data = detrend(signal_data)

    total_len=len(signal_data)
    window_size=config.window_sec*config.fs
    
    rrs=[]
    gt_idx=0
    for start in range(0, total_len, window_size):
        end = start + window_size
        print("start:", start)
        print('end:', end)
        
        if end > total_len:
            break

        segment = signal_data[start:end]
        
        if config.bp:
            if config.adaptive_bp:
                config.lowcut=0.15
                config.highcut=0.5
                segment = adaptive_bandpass_filter(segment, RR_GT_list[gt_idx], config)
            else: 
                segment = bandpass_filter(segment, config)

        if config.smoothing:
            segment=smoothing(segment, config)
        
        gt_idx+=1
        # print('s:', segment)
        n = len(segment)

        freqs = np.fft.rfftfreq(n, d=1/config.fs)
        fft_magnitude = np.abs(np.fft.rfft(segment))
        
        # Restrict frequency range to typical human breathing (0.1 Hz to 0.5 Hz, i.e., 6-30 breaths per minute)
        min_hz = config.lowcut
        max_hz = config.highcut
        valid_idx = (freqs >= min_hz) & (freqs <= max_hz)

        if not np.any(valid_idx):
            return 0.0  # fallback if no valid frequencies

        dominant_freq = freqs[valid_idx][np.argmax(fft_magnitude[valid_idx])]

        # Convert Hz to breaths per minute (BPM)
        rr_bpm = dominant_freq * 60
        rrs.append(rr_bpm)
        


    return rrs