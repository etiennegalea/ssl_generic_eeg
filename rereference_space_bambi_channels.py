import mne
import os

from tqdm import tqdm


MONTAGE_MAP = {
    0: 'Fpz-Cz',
    1: 'Pz-Oz'
}

data_dir = '/media/maligan/My Passport/msc_thesis/data/SPACE_BAMBI/'

raws = []
r_raws = []
errors = []

print(f'{len(os.listdir(data_dir))} files found')
for i, path in enumerate(tqdm(os.listdir(data_dir))):
    full_path = os.path.join(data_dir, path)
    raw = mne.io.read_raw_fif(full_path)
    raws.append(raw)
    raw_name = path.split('.fif')[0]

    data = []
    for k, v in MONTAGE_MAP.items():
        ch1, ch2 = v.split('-')
        x = raw[ch1][0] - raw[ch2][0]
        data.append(x[0])

    info = mne.create_info([MONTAGE_MAP[0], MONTAGE_MAP[1]], ch_types=['eeg']*2, sfreq=raw.info['sfreq'])
    raw = mne.io.RawArray(data, info)
    r_raws += [raw.set_annotations(raws[0].annotations)]

    try:
        raw.save(f'/media/maligan/My Passport/msc_thesis/data/SPACE_BAMBI_rereferenced/{raw_name}.fif')
    except:
        errors.append(raw_name)

print('Errors while saving:')
print(errors) if errors else print('None!')
