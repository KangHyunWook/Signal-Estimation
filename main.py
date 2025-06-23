from filtering import smoothing, triangle_filter

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from biosppy.signals import ecg
from create_dataset import IMU, MIC
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import get_config

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

#functions
def get_heart_rate(ecg_signal, config):
    # Example: Simulated ECG signal (you can replace this with your real ECG data)
    # For real use, load ECG from a file (e.g., with numpy, pandas, or scipy)
    sampling_rate = config.fs  # in Hz

    # Simulated ECG-like signal (replace with your own signal)

    # Step 1: Detect R-peaks
    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=False)
    r_peaks = out['rpeaks']  # indices of R-peaks


    # Step 2: Compute R-R intervals (in seconds)
    r_peak_times = r_peaks / sampling_rate
    rr_intervals = np.diff(r_peak_times)

    # Step 3: Compute instantaneous heart rate (in bpm)
    heart_rates = 60 / rr_intervals
    
    return r_peaks, rr_intervals, heart_rates


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


train_config = get_config()
if 'imu' in train_config.dataset_dir.lower():
    dataset=IMU(train_config)
    train_config.fs = 50  # 50 Hz sampling rate
    print('IMU')
elif 'mic' in train_config.dataset_dir.lower():
    dataset=MIC(train_config)
    train_config.fs = 500
    print('MIC')

data=dataset.get_data()


# Simulated IMU data (replace with your real signal)
np.random.seed(42)

duration = 300  # 5 minutes

#todo: read raw IMU signal
#evaluate

if 'imu' in train_config.dataset_dir.lower():
    if train_config.axis==0:
        signal = data[0]
    elif train_config.axis==1:
        signal = data[1]
    elif train_config.axis==2:
        signal = data[2]
    elif train_config.axis==3:
        signal = data[3]
    elif train_config.axis==4:
        signal = data[4]
    elif train_config.axis==5:
        signal = data[5]
    elif train_config.axis==6:
        pass #all

        
    #apply moving average to accelX
    if train_config.smoothing:
        signal=smoothing(signal,train_config)

    if train_config.triangle_filter:
        signal = triangle_filter(signal, train_config)
    
    # if train_config.triangle_filter:
        # accelX_l

    # imu_signal=np.asarray(imu_signal)



    # Step 1: Bandpass filter

    fs = train_config.fs
    rr_estimates = estimate_rr_from_imu(train_config ,signal, fs)
    # accely_rr_estimates = estimate_rr_from_imu(train_config, accelY_list, fs)
    # accelz_rr_estimates = estimate_rr_from_imu(train_config, accelZ_list, fs)
    # gyrox_rr_estimates = estimate_rr_from_imu(train_config, gyroX_list, fs)
    # gyroy_rr_estimates = estimate_rr_from_imu(train_config, gyroY_list, fs)
    # gyroz_rr_estimates = estimate_rr_from_imu(train_config, gyroZ_list, fs)




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


    mse = mean_squared_error(RR_GT_list, rr_estimates)
    r_squared = r_squared(RR_GT_list, rr_estimates)


    print('axis {0} | MSE: {1}'.format(train_config.axis, mse))
    print('r_squared:', r_squared)

    train_len=int(len(rr_estimates)*0.7)
    print('l:', train_len)


    start=0


    input_len=len(rr_estimates)//3

    target=0
    fold=0
    maes=[]
    for i in range(0,30,10):
        train_data=[]
        train_label=[]
        test_data=[]
        test_label=[]
        clf = Ridge(alpha=0.1)
        fold+=1    
        for j in range(0,len(rr_estimates),input_len):
            
            data=np.asarray(rr_estimates[j:j+input_len])
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
        maes.append(mae)
        print('=====Fold {0}====='.format(fold))
        print('raw mse:',raw_mse)
        print('mse:',mse)
        print('raw mae:', raw_mae)
        print('mae:', mae)

    print('mean mae /std')
    mean_mae= round(np.mean(maes),2)
    mean_std =round(np.std(maes),2)
    print(round(np.mean(maes),2), mean_std)


    with open('results.csv', train_config.w_mode) as f:
        if train_config.w_mode=='w':
            f.write('ma window, long window size, triangle size,  mean mae, mean std\n')
        f.write(str(train_config.ma_window)+','+str(train_config.long_window_size)+','+str(train_config.triangle_w_size)+','+str(mean_mae)+','+str(mean_std)+'\n')

    f.close()
    
    print('File written')
    print(train_data.shape, test_data.shape)
    print(train_label.shape, test_label.shape)
    exit()


    

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

elif 'mic' in train_config.dataset_dir.lower():
    
    times=data[0]
    sensors = data[1]
    GTs=data[2]
    
    #slice sensors by 10 seconds
    time_step=10*train_config.fs
    print('len:', len(sensors))
    
    sensor_segments=[]
    GT_segments=[]
    for i in range(0,time_step*(len(sensors)//time_step), time_step):
        print(i,i+time_step)
        sensor_segment = sensors[i:i+time_step]
        GT_segment = GTs[i:i+time_step]

        sensor_segments.append(sensor_segment)
        GT_segments.append(GT_segment)

    print('time:', len(times), len(sensors), len(GTs))

    heart_rate_labels=[]
    for i in range(3):
        print('======Fold {0}======'.format(i+1))
        r_peaks, rr_intervals, heart_rates = get_heart_rate(GT_segments[i],train_config)

        # Print results
        print("R-peaks (indices):", r_peaks)
        print("R-R intervals (s):", rr_intervals)
        print("Heart rate (bpm):", heart_rates)
        print('r_peaks:', len(r_peaks), len(heart_rates))
        
        print("Average heart rate:", np.mean(heart_rates))
        print('len:', len(heart_rates))
        heart_rate_labels.append(heart_rates[:13])

    #train ridge regression with input sensor and label heart rates
    clf = Ridge(alpha=0.1)

    
    #train and evaluate
    fold=1
    maes=[]
    preds=[]
    labels=[]
    for i in range(3):        
        train_X=[]
        train_y=[]
        test_X=[]
        test_y=[]
        for j in range(3):
            current_sensor=sensor_segments[j]
            current_heart_rate = heart_rate_labels[j]
            if j==i:
                test_X.append(current_sensor)
                test_y.append(current_heart_rate)
            else:
                train_X.append(current_sensor)
                train_y.append(current_heart_rate)
        
        train_X=np.asarray(train_X)
        train_y=np.asarray(train_y)
        test_X=np.asarray(test_X)
        test_y=np.asarray(test_y)

        labels.append(test_y[0])
        print(train_X.shape, train_y.shape)
        print(test_X.shape, test_y.shape)
        clf.fit(train_X, train_y)
        pred = clf.predict(test_X)
        # pred=pred[0]

        preds.append(pred[0])
        mse = mean_squared_error(test_y, pred)
        # raw_mae=mean_absolute_error(test_y, test_X)
        mae = mean_absolute_error(test_y, pred)
        maes.append(mae)
        print('=====Fold {0}====='.format(i+1))
        print('mse:',mse)
        print('mae:', mae)
        # print('raw mae:', raw_mae)
        
    print('mean mae /std')
    print(round(np.mean(maes),2), round(np.std(maes),2))
    

    
    #plot heart rates
    ecg_signal=GTs[:15000]
    r_peaks, rr_intervals, heart_rates = get_heart_rate(ecg_signal,train_config)
    fig, axes = plt.subplots(1,1, figsize=(13,3))
    t=np.arange(len(heart_rates))
    axes.plot(t, heart_rates)
    plt.xticks([])
    axes.set_ylabel('Heart rates (BPM)')
    plt.tight_layout()
    plt.savefig('hr.png')
    plt.show()
    
    exit()
    
    fig, axes = plt.subplots(1,3, figsize=(13,3))
    
    
    # #Optional: plot ECG with R-peaks
    gap=0.8
    for i in range(len(preds)):
        t=np.arange(13)
        axes[i].plot(t, preds[i], color='#1f77b4', label='Predictions')
        axes[i].plot(t, labels[i], color='#d62728', label='Heart Rates')
        axes[i].set_title(f'Fold {i+1}')

        axes[i].fill_between(t, preds[i] - gap, preds[i] + gap, color='#1f77b4', alpha=0.2)
        axes[i].fill_between(t, labels[i] - gap, labels[i] + gap, color='#d62728', alpha=0.2)
        
    axes[1].set_xlabel('Indices')
    axes[0].set_ylabel('Heart rate (BPM)')
    

    fig.supxlabel('Predictions refer to the predicted hearts rates from Sensor for 10 seconds')
    
    legend_lines = [Line2D([0], [0], color='#1f77b4', lw=5),   # Blue line in legend, thickened
                    Line2D([0], [0], color='#d62728', lw=5)]  # Orange line in legend, thickened

    plt.legend(handles=legend_lines, labels=['Predictions', 'Heart Rates'], loc='upper right', handlelength=3, fontsize=10, bbox_to_anchor=(1.49,1))
    

    plt.tight_layout()
    # plt.tight_layout(rect=[0.01, 0.01, 0.01, 0.95])  # Leave space on the left for y-label
    plt.savefig('fig.png')
    
    print('saved')
    plt.show()

    