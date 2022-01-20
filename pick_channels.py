import mne
from mne_extras import write_edf

import os
from tqdm import tqdm
import pandas as pd
import pprint



pp = pprint.PrettyPrinter(indent=2)

# data_dir = '/media/maligan/My Passport/msc_thesis/ssl_thesis/data/'
data_dir = '/media/maligan/My Passport/msc_thesis/ssl_thesis/data/tuh_abnormal_data/train/'

def get_file_list(x):
    return [os.path.join(x, fname) for fname in os.listdir(x)]

def get_id(x):
    return x.split('/')[-1]

annotations = {}
for annotation in get_file_list(data_dir):
    subjects = {}
    for subject in get_file_list(annotation):
        recordings = {}
        for recording in get_file_list(subject):
            dates = {}
            for date in get_file_list(recording):
                for raw_path in get_file_list(date):
                    if '.edf' in get_id(raw_path):
                        break
                    else:
                        pass
                dates[get_id(date)] = raw_path
            recordings[get_id(recording)] = dates
        subjects[get_id(subject)] = recordings
    annotations[get_id(annotation)] = subjects



pp.pprint(annotations)


df = pd.json_normalize(annotations, sep='_').T

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
