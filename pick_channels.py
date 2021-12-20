import mne
from mne_extras import write_edf

import os


data_dir = '/media/maligan/My Passport/ssl_thesis/data/SPACE_BAMBI'
files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]
print(files)


# read, pick channels, save loop
for file in files:
    raw = mne.io.read_raw_fif(f'{file}')
    raw = raw.pick_channels(['Fpz', 'Pz'])
    # raw.export(file, fmt='edf')

    to_export = f"{file.split('/')[-1].split('.fif')[0]}.edf"
    write_edf(raw, f"./picked_channels/{to_export}")
    print(file)
