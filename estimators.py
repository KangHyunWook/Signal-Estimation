from scipy.signal import find_peaks
from filtering import smoothing, triangle_filter, bandpass_filter, adaptive_bandpass_filter

import numpy as np


from scipy.signal import butter, filtfilt, detrend


#todo: create RR_Estimator class

class RR_Estimator:
    def __init__(self, config, signal, GT_RRs):
        self.config=config
        self.signal = signal
        self.GT_RRs = GT_RRs
        
    def estimate_rr_fft(self):
        """
        Estimate respiratory rate using FFT.

        Parameters:
        - signal_data: 1D array-like, the IMU signal (e.g., from one axis of accelerometer or gyroscope)
        - fs: int, sampling frequency in Hz (default is 50 Hz)

        Returns:
        - rr_bpm: float, estimated respiratory rate in breaths per minute
        """
        # Detrend signal to remove linear drift

        signal_data = detrend(self.signal)

        total_len=len(signal_data)
        window_size=self.config.window_sec*self.config.fs
        
        rrs=[]
        gt_idx=0
        for start in range(0, total_len, window_size):
            end = start + window_size

            if end > total_len:
                break

            segment = signal_data[start:end]
            
            if self.config.bp:
                if self.config.adaptive_bp:
                    self.config.lowcut=0.15
                    self.config.highcut=0.5
                    segment = adaptive_bandpass_filter(segment, self.GT_RRs[gt_idx], self.config)
                else: 
                    segment = bandpass_filter(segment, self.config)

            if self.config.smoothing:
                segment=smoothing(segment, self.config)
            
            gt_idx+=1
            # print('s:', segment)
            n = len(segment)

            freqs = np.fft.rfftfreq(n, d=1/self.config.fs)
            fft_magnitude = np.abs(np.fft.rfft(segment))
            
            # Restrict frequency range to typical human breathing (0.1 Hz to 0.5 Hz, i.e., 6-30 breaths per minute)
            min_hz = self.config.lowcut
            max_hz = self.config.highcut
            valid_idx = (freqs >= min_hz) & (freqs <= max_hz)

            if not np.any(valid_idx):
                return 0.0  # fallback if no valid frequencies

            dominant_freq = freqs[valid_idx][np.argmax(fft_magnitude[valid_idx])]

            # Convert Hz to breaths per minute (BPM)
            rr_bpm = dominant_freq * 60
            rrs.append(rr_bpm)
            


        return rrs

    def estimate_rr_peak_detection(self):
        """
        Estimate respiratory rate using peak detection.

        Parameters:
        - signal_data: 1D array-like, IMU signal (e.g., one accelerometer axis)
        - fs: int, sampling frequency in Hz (default 50 Hz)

        Returns:
        - rr_bpm: float, estimated respiratory rate in breaths per minute
        """
        # Detrend and bandpass filter to isolate breathing frequency range
        fs = self.config.fs
        
        # signal_data = detrend(self.signal)
        signal_data = self.signal
        # filtered = bandpass_filter(signal_data, fs)
        
        total_len = len(signal_data)
        rrs=[]
        gt_idx=0
        window_size=self.config.fs*self.config.window_sec
        for start in range(0, total_len, window_size):
            end = start + window_size
            
            if end > total_len:
                break

            segment = signal_data[start:end]

            if self.config.bp:
                if self.config.adaptive_bp:
                    segment = adaptive_bandpass_filter(segment, self.GT_RRs[gt_idx], self.config)
                else: 
                    segment = bandpass_filter(segment, self.config)

            gt_idx+=1
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

    def estimate(self, estimator):
        if estimator=='FFT':
            rrs=self.estimate_rr_fft()
        elif estimator=='Peak':
            rrs = self.estimate_rr_peak_detection()
        return rrs



