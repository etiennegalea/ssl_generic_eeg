# %pip install braindecode mne umap-learn skorch==0.10.0
# %pip install umap-learn[plot] pandas matplotlib datashader bokeh holoviews scikit-image colorcet

# imports
import numpy as np
import mne
import matplotlib.pyplot as plt
# %matplotlib inline
import os
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
from torch import nn
from torch.utils.data import DataLoader
from braindecode.datasets.sleep_physionet import SleepPhysionet
from braindecode.datasets import BaseConcatDataset
from braindecode.datautil.preprocess import preprocess, Preprocessor
from braindecode.preprocessing.windowers import create_windows_from_events
from braindecode.util import set_random_seeds
from braindecode.models import SleepStagerChambon2018
from braindecode import EEGClassifier
from braindecode.datautil.preprocess import zscore
from braindecode.samplers.ssl import RelativePositioningSampler
from braindecode.datasets import (create_from_mne_raw, create_from_mne_epochs)

from sklearn.model_selection import train_test_split
from skorch.helper import predefined_split
from skorch.callbacks import Checkpoint, EarlyStopping, EpochScoring
from skorch.utils import to_tensor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# visualizations

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import cm
import umap.umap_ as umap
import umap.plot

from umap import UMAP
import plotly.express as px

# ----

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)


### INIT VARIABLES

random_state = 87
n_jobs = 1

# Preprocessing
sfreq = 160
high_cut_hz = 30

# windowing
window_size_samples = 500

# embedder
n_channels, input_size_samples = 2, 500
emb_size = 100

# Training
lr = 5e-3
batch_size = 512
n_epochs = 12
num_workers = 0 if n_jobs <= 1 else n_jobs


### Load model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True

# Set random seed to be able to reproduce results
set_random_seeds(seed=random_state, cuda=device == 'cuda')


# instantiate classes

emb = SleepStagerChambon2018(
    n_channels=n_channels,
    sfreq=sfreq,
    n_classes=emb_size,
    n_conv_chs=16,
    input_size_s=input_size_samples / sfreq,
    dropout=0,
    apply_batch_norm=True
)


class ContrastiveNet(nn.Module):
    """Contrastive module with linear layer on top of siamese embedder.

    Parameters
    ----------
    emb : nn.Module
        Embedder architecture.
    emb_size : int
        Output size of the embedder.
    dropout : float
        Dropout rate applied to the linear layer of the contrastive module.
    """
    def __init__(self, emb, emb_size, dropout=0.5):
        super().__init__()
        self.emb = emb
        self.clf = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_size, 1)
        )

    def forward(self, x):
        x1, x2 = x
        z1, z2 = self.emb(x1), self.emb(x2)
        return self.clf(torch.abs(z1 - z2)).flatten()


class RelativePositioningDataset(BaseConcatDataset):
    """BaseConcatDataset with __getitem__ that expects 2 indices and a target.
    """
    def __init__(self, list_of_ds):
        super().__init__(list_of_ds)
        self.return_pair = True

    def __getitem__(self, index):
        if self.return_pair:
            ind1, ind2, y = index
            return (super().__getitem__(ind1)[0],
                    super().__getitem__(ind2)[0]), y
        else:
            return super().__getitem__(index)

    @property
    def return_pair(self):
        return self._return_pair

    @return_pair.setter
    def return_pair(self, value):
        self._return_pair = value


# load the model
model = torch.load("models/pretrained/sleep_staging_5s_windows_75_subjects_cpu_15_epochs.model")



def compare_models(model_1, model_2):
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

# compare_models(model.emb, emb)


# DOWNSTREAM TASK - FINE TUNING

''' ANNOTATIONS
T0 corresponds to rest
T1 corresponds to onset of motion (real or imagined) of
    the left fist (in runs 3, 4, 7, 8, 11, and 12)
    both fists (in runs 5, 6, 9, 10, 13, and 14)
T2 corresponds to onset of motion (real or imagined) of
    the right fist (in runs 3, 4, 7, 8, 11, and 12)
    both feet (in runs 5, 6, 9, 10, 13, and 14)
'''

subjects = range(1,10)
event_codes = [
    1, 2, # eyes open, eyes closed (baselines)
    3, 4, 5,
    6, 7, 8, 9, 
    10, 11, 12, 13, 14
]

physionet_paths, descriptions = [], []

for subject_id in subjects:
    physionet_paths += [mne.datasets.eegbci.load_data(subject_id, event_codes, update_path=False)]
    descriptions += [{"event_code": code, "subject": subject_id} for code in event_codes]

# flatten list of paths
physionet_paths = [x for sublist in physionet_paths for x in sublist]

# Select the same channels as sleep staging model
# exclude channels which are not in Sleep Staging dataset
# exclude = list(x for x in eegmmidb[0].ch_names if x not in ['Fpz.', 'Pz..'])
exclude = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..',
    'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.',
    'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fp2.', 'Af7.', 
    'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 
    'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 
    'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 
    'P1..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 
    'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']

# Load each of the files
eegmmidb = [mne.io.read_raw_edf(path, preload=True, stim_channel='auto', exclude=exclude) for path in physionet_paths]


### preprocess
# resample to 100Hz
# high pass filtering of 30Hz

for channel in eegmmidb:
    # mne.io.Raw.resample(channel, sfreq)   # resample
    mne.io.Raw.filter(channel, l_freq=None, h_freq=high_cut_hz, n_jobs=n_jobs)    # high-pass filter

eegmmidb_windows = create_from_mne_raw(
    eegmmidb,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,
    drop_last_window=True,
    descriptions=descriptions,
    # mapping=mapping,
    # preload=True
)

### Fine tune on Sleep staging SSL model

# split by subject

subjects = np.unique(eegmmidb_windows.description['subject'])
subj_train, subj_test = train_test_split(
    subjects, test_size=0.4, random_state=random_state)
subj_valid, subj_test = train_test_split(
    subj_test, test_size=0.5, random_state=random_state)


split_ids = {'train': subj_train, 'valid': subj_valid, 'test': subj_test}
splitted = dict()
for name, values in split_ids.items():
    splitted[name] = RelativePositioningDataset(
        [ds for ds in eegmmidb_windows.datasets
            if ds.description['subject'] in values])


tau_pos, tau_neg = int(sfreq * 60), int(sfreq * 15 * 60)
n_examples_train = 250 * len(splitted['train'].datasets)
n_examples_valid = 250 * len(splitted['valid'].datasets)
n_examples_test = 250 * len(splitted['test'].datasets)


train_sampler = RelativePositioningSampler(
    splitted['train'].get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
    n_examples=n_examples_train, same_rec_neg=True, random_state=random_state)
valid_sampler = RelativePositioningSampler(
    splitted['valid'].get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
    n_examples=n_examples_valid, same_rec_neg=True,
    random_state=random_state)
test_sampler = RelativePositioningSampler(
    splitted['test'].get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
    n_examples=n_examples_test, same_rec_neg=True,
    random_state=random_state)


###############################
# trying w/o sequential layer #
###############################

# model.emb.return_feats = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
# Set random seed to be able to reproduce results
set_random_seeds(seed=random_state, cuda=device == 'cuda')

batch_size = 512
num_workers = 1

# Extract features with the trained embedder
data = dict()
for name, split in splitted.items():
    split.return_pair = False  # Return single windows
    loader = DataLoader(split, batch_size=batch_size, num_workers=num_workers)
    with torch.no_grad():
        feats = [model.emb(batch_x.to(device)).cpu().numpy()
                    for batch_x, _, _ in loader]
    data[name] = (np.concatenate(feats), split.get_metadata()['target'].values)


# Initialize the logistic regression model
log_reg = LogisticRegression(
    penalty='l2', C=1.0, class_weight='balanced', solver='sag',
    multi_class='multinomial', random_state=random_state)
clf_pipe = make_pipeline(StandardScaler(), log_reg)

# Fit and score the logistic regression
clf_pipe.fit(*data['train'])
train_y_pred = clf_pipe.predict(data['train'][0])
valid_y_pred = clf_pipe.predict(data['valid'][0])
test_y_pred = clf_pipe.predict(data['test'][0])
# test_ds_y_pred = clf_pipe.predict(test_ds_data[0])

train_bal_acc = balanced_accuracy_score(data['train'][1], train_y_pred)
valid_bal_acc = balanced_accuracy_score(data['valid'][1], valid_y_pred)
test_bal_acc = balanced_accuracy_score(data['test'][1], test_y_pred)
# test_ds_acc = balanced_accuracy_score(test_ds_data[1], test_ds_y_pred)

print('Sleep staging performance with logistic regression:')
print(f'Train bal acc: {train_bal_acc:0.4f}')
print(f'Valid bal acc: {valid_bal_acc:0.4f}')
print(f'Test bal acc: {test_bal_acc:0.4f}')
# print(f'Test bal acc: {test_ds_acc:0.4f}')

print('Results on test set:')
print(confusion_matrix(data['test'][1], test_y_pred))
print(classification_report(data['test'][1], test_y_pred))





### Visualizing clusters ###

X = np.concatenate([v[0] for k, v in data.items()])
y = np.concatenate([v[1] for k, v in data.items()])

_umap = umap.UMAP(n_neighbors=15)
umap_components = _umap.fit_transform(X)

components = [
    # pca_components,
    # tsne_components,
    umap_components
]
components_titles = [
    'UMAP'
 ]

# SINGLE plot
fig, ax = plt.subplots(1, figsize=(20,20))
colors = cm.get_cmap('viridis', 5)(range(5))
for i, stage in enumerate(['1', '2', '3']):
    mask = y == i
    ax.scatter(umap_components[mask, 0], umap_components[mask, 1], s=100, alpha=0.7, label=stage)
ax.legend()

ax.set_title(components_titles[0])

plt.savefig('UMAP.png')


# UMAP plot with connectivity
# https://umap-learn.readthedocs.io/en/latest/plotting.html


mapping = _umap.fit(X)

# fig = umap.plot.connectivity(mapping, show_points=True)
umap.plot.connectivity(mapping, edge_bundling='hammer') # decreased connectivity (more readable (prettier))

plt.savefig('UMAP_connectivity.png')
# plt.show()

# 3D UMAP plot
umap_3d = UMAP(n_components=3, init='random', random_state=0)
proj_3d = umap_3d.fit_transform(X)
series = pd.DataFrame(y, columns=['annots'])

fig_3d = px.scatter_3d(
    proj_3d, x=0, y=1, z=2,
    color=series.annots, labels={'color': 'annots'}
)

fig_3d.update_layout(
    autosize=False,
    width=850,
    height=850
)
fig_3d.update_traces(marker_size=3)
# fig_3d.show()