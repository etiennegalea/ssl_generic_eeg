import mne
from mne_extras import write_edf

import os
from tqdm import tqdm

data_dir = '/media/maligan/My Passport/msc_thesis/ssl_thesis/data/SPACE_BAMBI'
files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]

# read, pick channels, save loop
for file in tqdm(files):
    raw = mne.io.read_raw_fif(f'{file}')
    to_export = f"{file.split('/')[-1]}"
    raw.save(f"./data/SPACE_BAMBI_2channels/{to_export}", picks=['Fpz', 'Pz'], overwrite=True)
