import mne
from mne_extras import write_edf

import os
from tqdm import tqdm
import pandas as pd
import pprint

from helper_funcs import HelperFuncs as hf



pp = pprint.PrettyPrinter(indent=2)

# data_dir = '/media/maligan/My Passport/msc_thesis/ssl_thesis/data/'
data_dir = '/media/maligan/My Passport/msc_thesis/data/tuar/v2.1.0/edf/01_tcp_ar/'

def get_file_list(x):
    return [os.path.join(x, fname) for fname in os.listdir(x)]

def get_id(x):
    return x.split('/')[-1]

subjects = {}
for subject in hf.get_file_list(data_dir):
    recordings = {}
    for recording in hf.get_file_list(subject):
        dates = {}
        for date in hf.get_file_list(recording):
            for raw_path in hf.get_file_list(date):
                if '_2_channels.fif' in hf.get_id(raw_path):
                    break
                else:
                    pass
            dates[hf.get_id(date)] = raw_path
        recordings[hf.get_id(recording)] = dates
    subjects[hf.get_id(subject)] = recordings



pp.pprint(subjects)


df = pd.json_normalize(subjects, sep='_').T

# read, pick channels, save loop
error = []
for i in tqdm(range(len(df))):
    file = df.iloc[i][0]
    raw = mne.io.read_raw_edf(file)
    to_export = f"{file.split('/')[-1].split('.')[0]}"
    try:
        raw.save(f"{file}_2_channels.fif", picks=['EEG FZ-REF', 'EEG PZ-REF'], overwrite=True)
    except:
        error.append(file)

pp.pprint(error)
