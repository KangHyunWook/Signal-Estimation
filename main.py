from filtering import triangle_filter, compute_smoothed_hilbert
from estimators import RR_Estimator
from scipy.signal import find_peaks
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from biosppy.signals import ecg
from create_dataset import IMU, MIC
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import get_config

import numpy as np

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

def cross_domain_evaluate(imu_pattern2, GT_RR_list, GT_RR_predictor):
    imu_pattern2=imu_pattern2[:5000]
    np_imu_pattern2=np.asarray(imu_pattern2)
    np_imu_pattern2=np_imu_pattern2.reshape(-1, 500)

    GT_RR_predictions=GT_RR_predictor.predict(np_imu_pattern2)
    
    rr_estimator=RR_Estimator(train_config, imu_pattern2, GT_RR_predictions)
    rr_estimates = rr_estimator.estimate(train_config.estimator)
    time_points = np.arange(0, 50, 10)

    GT_RR_list=GT_RR_list[:-1]

    mae=mean_absolute_error(GT_RR_list, rr_estimates)
    print('mae:',mae)
    exit()
    print('axis {0} | MSE: {1}'.format(train_config.axis, mse))
    print('r_squared:', r_squared)

    train_len=int(len(rr_estimates)*0.7)
    print('l:', train_len)


    input_len=len(rr_estimates)//3

    target=0
    fold=0
    raw_maes=[]
    maes=[]

    train_data=[]
    train_label=[]
    test_data=[]
    test_label=[]
    clf = Ridge(alpha=0.1)
    fold+=1    
    for j in range(0,len(rr_estimates),input_len):
        
        data=np.asarray(rr_estimates[j:j+input_len])
        label=np.asarray(GT_RR_list[j:j+input_len])

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
    raw_maes.append(raw_mae)
    mse = mean_squared_error(test_label, pred)
    mae = mean_absolute_error(test_label, pred)
    maes.append(mae)
    print('=====Fold {0}====='.format(fold))
    print('raw mse:',raw_mse)
    print('mse:',mse)
    print('raw mae:', raw_mae)
    print('mae:', mae)


    exit()
    with open('results.csv', train_config.w_mode) as f:
        if train_config.w_mode=='w':
            f.write('ma window, long window size, triangle size,  mean mae, mean std\n')
        f.write(str(train_config.ma_window)+','+str(train_config.long_window_size)+','+str(train_config.triangle_w_size)+','+str(mean_mae)+','+str(mean_std)+'\n')

    f.close()

if 'imu' in train_config.dataset_dir.lower():
    
    axis=train_config.axis
    
    imu_pattern1=data[0]
    GT_RR_list=data[1]
    imu_pattern2=data[2]
    gt_rrs_pattern2=data[3]
    
    imu_pattern1_signal=imu_pattern1[axis]
    imu_pattern2_signal=imu_pattern2[axis]

    signal=imu_pattern1_signal
    GT_RR_predictor = Ridge(alpha=0.1)

    signal_np = np.asarray(signal)
    GT_RR_np = np.asarray(GT_RR_list)
    
    signal_np=signal_np.reshape(GT_RR_np.shape[0],-1)
    
    GT_RR_predictor.fit(signal_np, GT_RR_np)    
    
    if train_config.triangle_filter:
        signal = triangle_filter(signal, train_config)
    
    if train_config.pattern2:
        cross_domain_evaluate(imu_pattern2_signal, gt_rrs_pattern2,GT_RR_predictor)
    else:
        #todo: train GT_RR_predictor with inputs to outputs of signal, GT_RR
        print(GT_RR_np.shape)
        print(signal_np.shape)
        GT_RR_predictions = GT_RR_predictor.predict(signal_np)
        
        print(GT_RR_list)
        print(GT_RR_predictions)
        
        if train_config.adaptive_bp:
            arg_GT_RR_list=GT_RR_predictions
        else:
            arg_GT_RR_list=GT_RR_list
         
        # if train_config.estimator=='FFT':
            # rr_estimates=estimate_rr_fft(signal, GT_RR_predictions, train_config)
        # elif train_config.estimator=='Peak':
            # rr_estimates=estimate_rr_peak_detection(signal, GT_RR_predictions, train_config)
        # else:
            # print('No such estimatory')
            # exit()
        
        rr_estimator=RR_Estimator(train_config, signal, GT_RR_predictions)
        rr_estimates = rr_estimator.estimate(train_config.estimator)
        time_points = np.arange(0, 50, 10)

        mse = mean_squared_error(GT_RR_list, rr_estimates)
        r_squared = r_squared(GT_RR_list, rr_estimates)


        print('axis {0} | MSE: {1}'.format(train_config.axis, mse))
        print('r_squared:', r_squared)

        train_len=int(len(rr_estimates)*0.7)
        print('l:', train_len)


        start=0


        input_len=len(rr_estimates)//3

        target=0
        fold=0
        raw_maes=[]
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
                label=np.asarray(GT_RR_list[j:j+input_len])

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
            raw_maes.append(raw_mae)
            mse = mean_squared_error(test_label, pred)
            mae = mean_absolute_error(test_label, pred)
            maes.append(mae)
            print('=====Fold {0}====='.format(fold))
            print('raw mse:',raw_mse)
            print('mse:',mse)
            print('raw mae:', raw_mae)
            print('mae:', mae)

        print('mean mae /std')
        mean_raw_mae = round(np.mean(raw_maes),2)
        raw_mae_std = round(np.std(raw_maes), 2)
        mean_mae= round(np.mean(maes),2)
        mean_std =round(np.std(maes),2)
        print('mean raw mae')
        print(round(np.mean(raw_maes),2), raw_mae_std)
        print('mean mae')
        print(round(np.mean(maes),2), mean_std)


        with open('results.csv', train_config.w_mode) as f:
            if train_config.w_mode=='w':
                f.write('ma window, long window size, triangle size,  mean mae, mean std\n')
            f.write(str(train_config.ma_window)+','+str(train_config.long_window_size)+','+str(train_config.triangle_w_size)+','+str(mean_mae)+','+str(mean_std)+'\n')

        f.close()
    


elif 'mic' in train_config.dataset_dir.lower():

    times=data[0]
    sensors = data[1]
    GTs=data[2]
    
    train_config.fs = 500
    #slice sensors by 10 seconds
    time_step=10*train_config.fs
    print('len:', len(sensors))
    
    sensor_segments=[]
    GT_segments=[]
    
    window_size=2*train_config.fs
    n_iter = len(GTs)//window_size

    hr_gts=[]
    for i in range(n_iter):
        GT_segment = GTs[i*window_size:(i+1)*window_size]
        min_distance = int(train_config.fs * 0.5)
        peaks, _ = find_peaks(GT_segment, distance=min_distance, prominence=0.2)
        hr_bpm = (len(peaks)/window_size) * 60 
        hr_gts.append(hr_bpm)

    in_ear_signal_list=[]
    for i in range(len(hr_gts)):
        in_ear_seg = sensors[i*window_size:(i+1)*window_size]
        # in_ear_seg=compute_smoothed_hilbert(in_ear_seg,train_config.fs)
        in_ear_signal_list.append(in_ear_seg)



    #train ridge regression with input sensor and label heart rates
    clf = Ridge(alpha=0.1)

    
    #train and evaluate
    n_folds=3
    maes=[]
    preds=[]
    labels=[]
    print('len:', len( in_ear_signal_list))
    for i in range(3):        
        print('=========Fold {0}========='.format(i+1))
        train_X=[]
        train_y=[]
        test_X=[]
        test_y=[]
        size = len(in_ear_signal_list)//n_folds

        flag=0
        for j in range(3):
            start_idx = j*size
            end_idx = (j+1)*size
            # print('f:', flag)
            if flag==1:
                start_idx+=1
                end_idx+=1
            
            if j==i:
                end_idx=end_idx+1
                print(start_idx,end_idx)
                current_in_ear_seg = in_ear_signal_list[start_idx:end_idx]
                current_hr_gts = hr_gts[start_idx:end_idx]
                test_X.extend(current_in_ear_seg)
                test_y.extend(current_hr_gts)
                flag=1
            else:
                
                current_in_ear_seg = in_ear_signal_list[start_idx:end_idx]
                current_hr_gts = hr_gts[start_idx:end_idx]
                print(start_idx,end_idx)
                train_X.extend(current_in_ear_seg)
                train_y.extend(current_hr_gts)


        train_X=np.asarray(train_X)
        train_y=np.asarray(train_y)
        test_X=np.asarray(test_X)
        test_y=np.asarray(test_y)
        print('train:', train_X.shape, train_y.shape)
        print('test:', test_X.shape, test_y.shape)


        print(train_X.shape, train_y.shape)
        print(test_X.shape, test_y.shape)
        clf.fit(train_X, train_y)
        preds = clf.predict(test_X)
        
        print(test_y)
        print(preds)
        mse = mean_squared_error(test_y, preds)
        # raw_mae=mean_absolute_error(test_y, test_X)
        mae = mean_absolute_error(test_y, preds)
        maes.append(mae)
        print('mse:',mse)
        print('mae:', mae)
        # print('raw mae:', raw_mae)
    

    print('mean mae /std')
    print(round(np.mean(maes),2), round(np.std(maes),4))
    

    
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

    