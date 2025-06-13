from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import get_config

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq



def smoothing(signal):
    ma_window=3
    signal= signal - uniform_filter1d(signal, size=ma_window, axis=0, mode='nearest')
    long_window = int(10 * fs)  # e.g., 2s * 50Hz = 100 samples
    smoothed= uniform_filter1d(signal, size=long_window, axis=0, mode='nearest')

    return smoothed

def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def estimate_rr_from_imu(config, imu_signal, fs, window_sec=10):

    window_size = window_sec * fs    
    
    total_len = len(imu_signal)
    rr_estimates = []
    
    for start in range(0, total_len, window_size):
        end = start + window_size
        if end > total_len:
            break

        segment = imu_signal[start:end]
        
        if config.bp:
            segment = bandpass_filter(segment, fs, config.lowcut, config.highcut)
            
        # Step 2: FFT
        N = len(segment)
        freqs = fftfreq(N, 1/fs)
        fft_vals = np.abs(fft(segment))
        
        # Step 3: Focus on positive frequencies in respiration band (0.1–0.7 Hz)
        
        mask = (freqs >= 0.1) & (freqs <= 0.7)
        
        freqs_band = freqs[mask]
        
        fft_band = fft_vals[mask]

        # Find index of peak in FFT band
        peak_idx = np.argmax(fft_band)
        
        # Use neighboring bins for parabolic interpolation
        # rr_bpm = get_rr_bpm_by_parabolic_interpolation(peak_idx, freqs_band, fft_band)
        rr_bpm = get_rr_bpm_by_max(freqs_band, fft_band)
        
        rr_estimates.append(rr_bpm)

    return rr_estimates

def r_squared(y_true, y_pred):
    """
    Computes the R-squared (coefficient of determination).

    Parameters:
        y_true (array-like): Ground-truth values
        y_pred (array-like): Predicted values

    Returns:
        float: R-squared value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)


def get_rr_bpm_by_max(freqs_band, fft_band):
    # Step 4: Find peak frequency
    if len(freqs_band) == 0:
        rr_bpm = 0
    else:
        peak_freq = freqs_band[np.argmax(fft_band)]
        rr_bpm = peak_freq * 60  # Convert Hz to breaths per minute (BPM)

    return rr_bpm
    
def get_rr_bpm_by_parabolic_interpolation(peak_idx, freqs_band, fft_band):

    if 0 < peak_idx < len(fft_band) - 1:
        y0, y1, y2 = fft_band[peak_idx - 1:peak_idx + 2]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom
            peak_freq = freqs_band[peak_idx] + delta * (freqs_band[1] - freqs_band[0])
        else:
            peak_freq = freqs_band[peak_idx]
    else:
        peak_freq = freqs_band[peak_idx]

    print('peak freq:',  peak_freq)

    rr_bpm = peak_freq * 60
    
    return rr_bpm

# Simulated IMU data (replace with your real signal)
np.random.seed(42)
fs = 50  # 50 Hz sampling rate
duration = 300  # 5 minutes

#todo: read raw IMU signal

raw_IMU_path=r"C:\workspace\NYUAD\raw_imu.csv"

imu_signal=[]
accelX_list=[]
accelY_list=[]
accelZ_list=[]
gyroX_list=[]
gyroY_list=[]
gyroZ_list=[]

for line in open(raw_IMU_path):
    line=line.strip()
    splits = line.split(',')    
    
    if splits[0]=='accelx':
        continue
    
    accelx = float(splits[0])
    

    accely = float(splits[1])
    accelz = float(splits[2])
    gyrox = float(splits[3])
    gyroy = float(splits[4])
    gyroz = float(splits[5])
    
    # imu_signal.append(np.array([splits[0],splits[1],splits[2], gyrox, gyroy, gyroz]))
    # imu_signal.append(accely)
    
    accelX_list.append(accelx)
    accelY_list.append(accely)
    accelZ_list.append(accelz)
    gyroX_list.append(gyrox)
    gyroY_list.append(gyroy)
    gyroZ_list.append(gyroz)


thres=15000
# print(len(accelX_list))
# exit()
accelX_list = accelX_list[:thres]
accelY_list = accelY_list[:thres]
accelZ_list = accelZ_list[:thres]
gyroX_list = gyroX_list[:thres]
gyroY_list = gyroY_list[:thres]
gyroZ_list = gyroZ_list[:thres]



config=get_config()
    
#apply moving average to accelX
if config.smoothing:
    accelX_list=smoothing(accelX_list)



# imu_signal=np.asarray(imu_signal)

# imu_signal=np.linalg.norm(imu_signal,axis=1)

# imu_signal=list(imu_signal)


# Step 1: Bandpass filter


accelx_rr_estimates = estimate_rr_from_imu(config ,accelX_list, fs)
accely_rr_estimates = estimate_rr_from_imu(config, accelY_list, fs)
accelz_rr_estimates = estimate_rr_from_imu(config, accelZ_list, fs)
gyrox_rr_estimates = estimate_rr_from_imu(config, gyroX_list, fs)
gyroy_rr_estimates = estimate_rr_from_imu(config, gyroY_list, fs)
gyroz_rr_estimates = estimate_rr_from_imu(config, gyroZ_list, fs)


accelx_rr_estimates = estimate_rr_from_imu(config, accelX_list, fs)
accely_rr_estimates = estimate_rr_from_imu(config, accelY_list, fs)
accelz_rr_estimates = estimate_rr_from_imu(config, accelZ_list, fs)
gyrox_rr_estimates = estimate_rr_from_imu(config, gyroX_list, fs)
gyroy_rr_estimates = estimate_rr_from_imu(config, gyroY_list, fs)
gyroz_rr_estimates = estimate_rr_from_imu(config, gyroZ_list, fs)


time_points = np.arange(0, 50, 10)


RR_GT_path=r"C:\workspace\NYUAD\RR_GT.csv"

RR_GT_list=[]

cnt=0
for line in open(RR_GT_path):
    line = line.strip()
    splits = line.split(',')
    if splits[0]=='time':
        continue
    if cnt<30:
        RR_GT_list.append(float(splits[1]))
    cnt+=1


accelx_mse = mean_squared_error(RR_GT_list, accelx_rr_estimates)
accely_mse = mean_squared_error(RR_GT_list, accely_rr_estimates)
accelz_mse = mean_squared_error(RR_GT_list, accelz_rr_estimates)
gyrox_mse = mean_squared_error(RR_GT_list, gyrox_rr_estimates)
gyroy_mse = mean_squared_error(RR_GT_list, gyroy_rr_estimates)
gyroz_mse = mean_squared_error(RR_GT_list, gyroz_rr_estimates)

accelx_r_squared = r_squared(RR_GT_list, accelx_rr_estimates)
accely_r_squared = r_squared(RR_GT_list, accely_rr_estimates)
accelz_r_squared = r_squared(RR_GT_list, accelz_rr_estimates)
gyrox_r_squared = r_squared(RR_GT_list, gyrox_rr_estimates)
gyroy_r_squared = r_squared(RR_GT_list, gyroy_rr_estimates)
gyroz_r_squared = r_squared(RR_GT_list, gyroz_rr_estimates)


print('accelx mse:', accelx_mse)
print('accely mse:', accely_mse)
print('accelz mse:', accelz_mse)
print('gyrox mse:', gyrox_mse)
print('gyroy mse:', gyroy_mse)
print('gyroz mse:', gyroz_mse)

print('accelx r_squared:', accelx_r_squared)
print('accely r_squared:', accely_r_squared)
print('accelz r_squared:', accelz_r_squared)
print('gyrox r_squared:', gyrox_r_squared)
print('gyroy r_squared:', gyroy_r_squared)
print('gyroz r_squared:', gyroz_r_squared)

print(len(accelx_rr_estimates))

train_len=int(len(accelx_rr_estimates)*0.7)
print('l:', train_len)

from sklearn.linear_model import Ridge

start=0


input_len=len(accelx_rr_estimates)//3

target=0
fold=0
for i in range(0,30,10):
    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    clf = Ridge(alpha=0.1)
    fold+=1    
    for j in range(0,len(accelx_rr_estimates),input_len):
        
        data=np.asarray(accelx_rr_estimates[j:j+input_len])
        label=np.asarray(RR_GT_list[j:j+input_len])

        if i==j:
            test_data.append(data)
            test_label.append(label)
        else:
            train_data.append(data)
            train_label.append(label)

    test_data=np.asarray(test_data)
    test_label =np.asarray(test_label)

    train_data = np.asarray(train_data)    
    train_label = np.asarray(train_label)
    
    clf.fit(train_data, train_label)
    pred = clf.predict(test_data)
    raw_mse = mean_squared_error(test_label, test_data)
    raw_mae = mean_absolute_error(test_label, test_data)
    mse = mean_squared_error(test_label, pred)
    mae = mean_absolute_error(test_label, pred)
    print('=====Fold {0}====='.format(fold))
    print('raw mse:',raw_mse)
    print('mse:',mse)
    print('raw mae:', raw_mae)
    print('mae:', mae)





print(train_data.shape, test_data.shape)
print(train_label.shape, test_label.shape)
exit()


import matplotlib.pyplot as plt

fig, axis = plt.subplots(2,3, figsize=(16,5))

axis[0,0].plot(time_points, RR_GT_list, label='RR GT',color='#1f77b4')
axis[0,0].plot(time_points, accelx_rr_estimates, label='RR esti.', color='#d62728')
axis[0,0].set_title('Estimation is from accelX')

axis[0,1].plot(time_points, RR_GT_list, label='RR GT', color='#1f77b4')
axis[0,1].plot(time_points, accely_rr_estimates, color='#d62728')
axis[0,1].set_title('Estimation is from accelY')

axis[0,2].plot(time_points, RR_GT_list, label='RR GT', color='#1f77b4')
axis[0,2].plot(time_points, accelz_rr_estimates, label='RR esti.', color='#d62728')
axis[0,2].set_title('Estimation is from accelZ')


axis[1,0].plot(time_points, RR_GT_list, label='RR GT', color='#1f77b4')
axis[1,0].plot(time_points, gyrox_rr_estimates, label='RR esti.', color='#d62728')
axis[1,0].set_title('Estimation is from gyroX')

axis[1,1].plot(time_points, RR_GT_list, label='RR GT', color='#1f77b4')
axis[1,1].plot(time_points, gyroy_rr_estimates, label='RR esti.', color='#d62728')
axis[1,1].set_title('Estimation is from gyroY')

axis[1,2].plot(time_points, RR_GT_list, label='RR GT', color='#1f77b4')
axis[1,2].plot(time_points, gyroz_rr_estimates, label='RR esti.', color='#d62728')
axis[1,2].set_title('Estimation is from gyroZ')

from matplotlib.lines import Line2D

# Create custom legend lines with thickened lines
legend_lines = [Line2D([0], [0], color='#1f77b4', lw=5),   # Blue line in legend, thickened
                Line2D([0], [0], color='#d62728', lw=5)]  # Orange line in legend, thickened

plt.legend(handles=legend_lines, labels=['RR GT', 'RR esti.'], loc='upper right', handlelength=3, fontsize=10, bbox_to_anchor=(1.36,1))

# axis[0,0].set_ylabel('Respiratory rate (RR) (beats per minute)')

fig.text(0.5, 0.5, 'Time (s)', ha='center', fontsize=12)  # For top row (0,x)
fig.text(0.5, 0.02, 'Time (s)', ha='center', fontsize=12)  # For bottom row (1,x)

# fig.text(0.1, 0.5, 'Respiratory rate (BPM)', va='center', rotation='vertical', fontsize=12)  # Shared y-label

fig.supylabel('Respiratory rate (BPM)')

# plt.tight_layout(rect=[0.01, 0.01, 0.01, 0.95])  # Leave space on the left for y-label

plt.tight_layout(pad=2.0)
plt.savefig('rr.png')
plt.show()
print(len(time_points))
print(len(RR_GT_list))

# print(mse)