import argparse

def get_config(**optional_kwargs):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_dir', default = r"C:\workspace\NYUAD\MIC_1000gain_facingEarCanal.csv")
    parser.add_argument('--sampling_rate', type=int, default=1000)
    parser.add_argument('--mode', default='train')
    
    parser.add_argument('--lowcut', type=float, default=0.1)
    parser.add_argument('--highcut', type=float, default=0.5)
    parser.add_argument('--axis', type=int, default='1', help='0: accel.X 1: accel.Y 2: accel.Z 3: gyro X 4: gyro.Y 5: gyro.Z 6: all')

    #filtering options
    parser.add_argument('--smoothing', action='store_true')
    parser.add_argument('--bp', action='store_true', help='applies Butterworth bandpass filtering')
    parser.add_argument('--triangle_filter', action='store_true')
    parser.add_argument('--ma_window', type=int, default=3)
    parser.add_argument('--long_window_size', type=int, default=25)
    parser.add_argument('--triangle_w_size', type=int, default=2)
    parser.add_argument('--w_mode', type=str, default='w')
    
    return parser.parse_args()