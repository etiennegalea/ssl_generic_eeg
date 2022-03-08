import torch
from datetime import date, datetime
import os
from pathlib import Path
import numpy as np
import mne


class HelperFuncs():

    def __init__(self):
        pass
        
    # print dataset lengths to match dimensions
    def print_dataset_lengths(self, datasets):
        for data in datasets:
            print(type(data))
            try:
                for i in range(100):
                    print(f"({i}) length: {len(data)} | type: {type(data)}")
                    data = data[0]
            except:
                print("------------------------------------------------------------------")

    # compare torch models (skeletons or trained)
    def compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')

    # get date, time, or both
    def get_datetime(self, dateonly=False):
        dt = datetime.now()
        return dt.strftime("%Y_%m_%d") if dateonly else dt.strftime("%Y_%m_%d__%H_%M")

    # attempting to enable GPU processing
    def enable_cuda(self):
        device = 'cpu'
        if torch.cuda.is_available():
            print(':: CUDA enabled - using GPU')
            device = 'cuda'
            torch.backends.cudnn.benchmark = True
        else:
            print(':: CUDA unavailable - using CPU')
            
        return device

    
    # helper functions for loading TUH abnormal raw files from hierarchy
    def get_file_list(self, x):
        return [os.path.join(x, fname) for fname in os.listdir(x)]
    def get_id(self, x):
        return x.split('/')[-1]

    # check for existing folder/s and create them if directory does not exist
    def check_dir(self, dir):
        Path(dir).mkdir(parents=True, exist_ok=True)

    # generate simulated noisy signals (sinusoidal waves w/ noise)
    def generate_noisy_raws(
        self,
        ch_names=['SIM0001', 'SIM0002'],
        ch_types=['eeg']*2,
        sfreq=100.0,
        n_times=5000,
        seed=42,
        wave_hz=50.0,
        stage='normal_noise'
        ):
        
        rng = np.random.RandomState(seed)
        noise = rng.randn(len(ch_names), n_times)

        # Add a specified (50hz) sinusoidal burst to the noise and ramp it.
        t = np.arange(n_times, dtype=np.float64) / sfreq
        signal = np.sin(np.pi * 2. * wave_hz * t)  # wave_hz sinusoid signal
        signal[np.logical_or(t < wave_hz-0.5, t > wave_hz+0.5)] = 0.  # Hard windowing
        on_time = np.logical_and(t >= wave_hz-0.5, t <= wave_hz+0.5)
        signal[on_time] *= np.hanning(on_time.sum())  # Ramping
        data = noise + signal

        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(data/1000000, info)
        raw = raw.set_annotations(mne.Annotations(onset=[0], duration=raw.times.max(), description=[stage]))

        return raw

    # generate simulated white noise signals
    def generate_white_noise_raws(
        self,
        ch_names=['SIM0001', 'SIM0002'],
        ch_types=['eeg']*2,
        sfreq=100.0,
        n_times=500,
        seed=42,
        # wave_hz=100.0,
        stage='white_noise',
        bound=1
    ):

        noise = np.array([
            np.random.uniform(sfreq-bound, sfreq+bound, size=n_times), 
            np.random.uniform(sfreq-bound, sfreq+bound, size=n_times)
        ])

        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(noise/1000000, info)
        raw = raw.set_annotations(mne.Annotations(onset=[0], duration=raw.times.max(), description=[stage]))

        return raw


    def return_space(self, X, logspace=False, static_space=[10,100,1000], raw=False):
        '''
        X: data
        logspace: use np.logspace
        static_space: static examples to start with
        raw: multiply the number of examples (for raw windows (number of channels))
        '''

        n_batches = 17 if raw else 7
        
        static_space = np.array(static_space)
        ssl_log_space = np.logspace(0.0001,1,n_batches)/10 if logspace else np.linspace(0.0001, 1, n_batches)
        ssl_space = (ssl_log_space*len(X)).astype(int)

        space = np.concatenate([static_space, ssl_space])


        return space

    # check factor of 10 for space
    def check_factor_loop(self, n, arr, factor):
        if factor < n:
            arr += [self.check_factor_loop(n, arr, factor*10)]
        else:
            return n
        return factor
        
    def factored_space(self, n):
        arr = []
        arr += [self.check_factor_loop(n, arr, factor=10)]
        return arr[::-1]
