import mne
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pprint
from tqdm import tqdm


pp = pprint.PrettyPrinter(indent=2)


# load mats (for scopolamine dataset)
def load_mats(path, info, classification, dataset, descriptions):
    raws, desc = [], []

    mats = os.listdir(path)
    for i, mat in enumerate(mats):
        print(mat)
        raw = mne.io.read_raw_fif(path+mat)
            
        # subject
        subject = int(mat.split('.')[1][2:])
        # recording (occasion)
        recording = int(mat.split('.')[-2].split('M')[0][1:])
        # treatment period
        treatment_period = int(mat.split('.')[-2].split('M')[-1])

        # if even (not placebo)
        # if not recording&1:
        raws += [raw]
        desc += [{'subject': subject, 'recording': recording, 'treatment_period': treatment_period, 'raw': path+mat}]
            

    dataset += raws
    descriptions += desc

    return dataset, descriptions

# load scopolamine data
def load_scopolamine_data(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples):
    print(':: loading SCOPOLAMINE data')

    # 11 measurements times from 0.5 hrs to 8.5 hrs after Scopolamine (or placebo) administration
    # m01 = 'data/scopolamine/M01/'
    m01 = '/media/maligan/My Passport/msc_thesis/data/scopolamine_converted/M01/'
    # m05 = 'data/scopolamine_converted/M05/'
    m05 = '/media/maligan/My Passport/msc_thesis/data/scopolamine_converted/M05/'
    # m11 = 'data/scopolamine_converted/M11/'
    m11 = '/media/maligan/My Passport/msc_thesis/data/scopolamine_converted/M11/'

    dataset, descriptions = [], []
    info = mne.create_info(ch_names=['Fpz-cz', 'Pz-Oz'], ch_types=['eeg']*2, sfreq=1012)

    dataset, descriptions = load_mats(m01, info, 'm01', dataset, descriptions)
    dataset, descriptions = load_mats(m05, info, 'm05', dataset, descriptions)
    dataset, descriptions = load_mats(m11, info, 'm11', dataset, descriptions)

    return dataset, descriptions


dataset, descriptions = load_scopolamine_data(sfreq=100, low_cut_hz=0.5, high_cut_hz=30, n_jobs=1, window_size_samples=500)

df = pd.DataFrame(descriptions).groupby(['subject','treatment_period']).min().head(15)

# plotting one EEG trace
cols = 3
_, axes = plt.subplots(int(len(df)/cols), 3, figsize=(20, 5))

raw = mne.io.read_raw_fif(df.iloc[0,1])
ax[0] = raw.plot(show=False, duration=5, start=15)
