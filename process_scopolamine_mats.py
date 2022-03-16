import mne
import os
from mat73 import loadmat
from tqdm import tqdm

import pprint

pp = pprint.PrettyPrinter(indent=2)
errors = []

# load mats (for scopolamine dataset)
def load_mats(path, info, classification, dataset, descriptions):
    raws, desc = [], []

    # limiter
    mats = os.listdir(path)
    for i, mat in enumerate(tqdm(mats)):
        print(f'mat file: {mat}')
        # select columns 3 and 4 (Fpz-Cz, and Pz-Oz respectively) and convert to microvolts
        x = loadmat(path + mat)['RawSignal'][:, [2,3]].T / 1000000
        raw = mne.io.RawArray(x, info)

        raw = raw.set_annotations(mne.Annotations(onset=[0], duration=raw.times.max(), description=[classification]))
        # try:
        mat = mat.split('.mat')[0]


        # preprocess
        raw = raw.resample(100)   # resample
        raw = raw.filter(l_freq=0.5, h_freq=30, n_jobs=1)    # filtering


        print(f':: {classification} ~ {mat}')
        to_export = f'/media/maligan/My Passport/msc_thesis/data/scopolamine_preprocessed/{classification}/{mat}.fif'
        print(f':: {to_export}')
        raw.save(to_export)
        # except:
        # errors.append(raw_name)


    dataset += raws
    descriptions += desc

    return dataset, descriptions


print(':: loading SCOPOLAMINE data')

# 11 measurements times from 0.5 hrs to 8.5 hrs after Scopolamine (or placebo) administration
# m01 = 'data/scopolamine/M01/'
m01 = '/media/maligan/My Passport/msc_thesis/data/scopolamine/M01/'
# m05 = 'data/scopolamine/M05/'
m05 = '/media/maligan/My Passport/msc_thesis/data/scopolamine/M05/'
# m11 = 'data/scopolamine/M11/'
m11 = '/media/maligan/My Passport/msc_thesis/data/scopolamine/M11/'

dataset, descriptions = [], []
info = mne.create_info(ch_names=['Fpz-cz', 'Pz-Oz'], ch_types=['eeg']*2, sfreq=1012)

dataset, descriptions = load_mats(m01, info, 'M01', dataset, descriptions)
dataset, descriptions = load_mats(m05, info, 'M05', dataset, descriptions)
dataset, descriptions = load_mats(m11, info, 'M11', dataset, descriptions)

print('Errors while saving:')
print(errors) if errors else print('None!')