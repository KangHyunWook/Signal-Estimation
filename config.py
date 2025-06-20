import argparse

def get_config(**optional_kwargs):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_dir', default = r"C:\workspace\NYUAD\MIC_1000gain_facingEarCanal.csv")
    parser.add_argument('--sampling_rate', type=int, default=1000)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--smoothing', action='store_true')
    parser.add_argument('--bp', action='store_true', help='applies Butterworth bandpass filtering')
    parser.add_argument('--lowcut', type=float, default=0.1)
    parser.add_argument('--highcut', type=float, default=0.5)
    
    
    return parser.parse_args()