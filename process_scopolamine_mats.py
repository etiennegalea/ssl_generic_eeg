import mne
import sys
import os
import importlib

from braindecode.preprocessing.preprocess import preprocess, Preprocessor, zscore
from braindecode.datasets import (create_from_mne_raw, create_from_mne_epochs)
from braindecode.preprocessing.windowers import create_windows_from_events
from braindecode.datasets.sleep_physionet import SleepPhysionet
from braindecode.datasets import BaseDataset, BaseConcatDataset, WindowsDataset
from mne_extras import write_edf

from plot import Plot
from segment import Segmenter

import matplotlib.pyplot as plt

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

        print(f':: {classification} ~ {mat}')
        to_export = f'/media/maligan/My Passport/msc_thesis/data/scopolamine_converted/{classification}/{mat}.fif'
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

dataset, descriptions = load_mats(m01, info, 'm01', dataset, descriptions)
dataset, descriptions = load_mats(m05, info, 'm05', dataset, descriptions)
dataset, descriptions = load_mats(m11, info, 'm11', dataset, descriptions)

print('Errors while saving:')
print(errors) if errors else print('None!')