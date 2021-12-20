import mne
from braindecode.preprocessing.preprocess import preprocess, Preprocessor, zscore
from braindecode.datasets import (create_from_mne_raw, create_from_mne_epochs)
from braindecode.preprocessing.windowers import create_windows_from_events
from braindecode.datasets.sleep_physionet import SleepPhysionet
from braindecode.datasets import BaseDataset, BaseConcatDataset, WindowsDataset

raw_set = [
    '~/mne_data/physionet-sleep-data/SC4012E0-PSG.edf',
    '~/mne_data/physionet-sleep-data/SC4451F0-PSG.edf',
    '~/mne_data/physionet-sleep-data/SC4441E0-PSG.edf',
    '~/mne_data/physionet-sleep-data/SC4431E0-PSG.edf',
    '~/mne_data/physionet-sleep-data/SC4421E0-PSG.edf',
]

# load into raw array
raws = [mne.io.read_raw_edf(x, preload=True) for x in raw_set]
# pick channels
raws = [x.pick_channels(['EEG Fpz-Cz', 'EEG Pz-Oz']) for x in raws]

# raws = [mne.io.Raw.filter(raw, l_freq=None, h_freq=30, n_jobs=1) for raw in raws]
base_datasets = [BaseDataset(raw) for raw in raws]
base_concat_datasets = BaseConcatDataset(base_datasets)

window_size_samples = 500

eegmmidb_windows = create_from_mne_raw(
    base_concat_datasets,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,
    drop_last_window=True,
    accepted_bads_ratio=0.5,
    drop_bad_windows=True,
)
