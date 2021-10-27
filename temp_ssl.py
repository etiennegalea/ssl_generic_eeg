#!/usr/bin/env python
# coding: utf-8

# In[109]:


# imports
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
from torch import nn
from braindecode.datasets.sleep_physionet import SleepPhysionet
from braindecode.datautil.preprocess import preprocess, Preprocessor
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.util import set_random_seeds
from braindecode.models import SleepStagerChambon2018
from braindecode import EEGClassifier
from braindecode.datautil.preprocess import zscore
from braindecode.samplers.ssl import RelativePositioningSampler

import relative_positioning as rp
import contrastive_net as cn

from sklearn.model_selection import train_test_split
from skorch.helper import predefined_split
from skorch.callbacks import Checkpoint, EarlyStopping, EpochScoring
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import helper_funcs as hf


# In[110]:



# constants
random_state = 87
n_jobs = 1
window_size_s = 30
sfreq = 100

# define pre-training dataset
dataset = SleepPhysionet(
    subject_ids=[0,1,2,3,4,5],
    recording_ids=[1]
)


# sample data for which we are trying to generate predictions of the input data using a part of the SSL pre-trained model
path_to_sample = "/home/maligan/Documents/VU/Year_2/M.Sc._Thesis_[X_400285]/my_thesis/code/braindecode_code/sleep_staging_dataset/"

# input test
X = mne.io.read_raw_fif(path_to_sample+"0-raw.fif")


# In[111]:


# dataset involves mutliple datasets - different subjects
print(len(dataset.datasets[0]))
print(len(X))

X.datasets = X
print(len(X.datasets))


# In[112]:


# preprocessing

preprocessors = [
    Preprocessor(lambda x: x * 1e6),
    Preprocessor('filter', l_freq=None, h_freq=30, n_jobs=n_jobs)
]
preprocess(dataset, preprocessors)


# Create windows
window_size_samples = window_size_s * sfreq

mapping = {  # We merge stages 3 and 4 following AASM standards.
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}

# define windows
windows_dataset = create_windows_from_events(
    dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples, preload=True, mapping=mapping)

# preprocess chanel-wise normalization
preprocess(windows_dataset, [Preprocessor(zscore)])


# In[113]:


# from braindecode.datautil.preprocess import preprocess_raw

### input test

# preprocess(X, preprocessors)
# X_windows_dataset = create_windows_from_events(
#     X, trial_start_offset_samples=0, trial_stop_offset_samples=0,
#     window_size_samples=window_size_samples,
#     window_stride_samples=window_size_samples, preload=True, mapping=mapping)
# preprocess(X_windows_dataset, [Preprocessor(zscore)])

X = SleepPhysionet(
    subject_ids=[0],
    recording_ids=[1]
)
X_windowed = create_windows_from_events(
    X, trial_start_offset_samples=0, trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples
)


# In[114]:


# Splitting train, valid, test sets

subjects = np.unique(windows_dataset.description['subject'])
subj_train, subj_test = train_test_split(
    subjects, test_size=0.4, random_state=random_state)
subj_valid, subj_test = train_test_split(
    subj_test, test_size=0.5, random_state=random_state)


split_ids = {'train': subj_train, 'valid': subj_valid, 'test': subj_test}
splitted = dict()
for name, values in split_ids.items():
    splitted[name] = rp.RelativePositioningDataset(
        [ds for ds in windows_dataset.datasets
         if ds.description['subject'] in values])


# In[115]:


# Sampling
# these samplers will be used to create sample data from the training set for the SSL model

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
    random_state=random_state).presample()

test_sampler = RelativePositioningSampler(
    splitted['test'].get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
    n_examples=n_examples_test, same_rec_neg=True,
    random_state=random_state).presample()


# In[116]:


# Create model

# enable CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
    print("CUDA enabled")
# Set random seed to be able to reproduce results
set_random_seeds(seed=random_state, cuda=device == 'cuda')

# Extract number of channels and time steps from dataset
n_channels, input_size_samples = windows_dataset[0][0].shape
emb_size = 100

emb = SleepStagerChambon2018(
    n_channels,
    sfreq,
    n_classes=emb_size,
    n_conv_chs=16,
    input_size_s=input_size_samples / sfreq,
    dropout=0,
    apply_batch_norm=True
)

model = cn.ContrastiveNet(emb, emb_size).to(device)


# In[117]:



lr = 5e-3
batch_size = 512
# for the sake of testing, reduce epochs to just 3 for now
# n_epochs = 25
n_epochs = 3
num_workers = 0 if n_jobs <= 1 else n_jobs

cp = Checkpoint(dirname='', f_criterion=None, f_optimizer=None, f_history=None)
early_stopping = EarlyStopping(patience=10)
train_acc = EpochScoring(
    scoring='accuracy', on_train=True, name='train_acc', lower_is_better=False)
valid_acc = EpochScoring(
    scoring='accuracy', on_train=False, name='valid_acc',
    lower_is_better=False)
callbacks = [
    ('cp', cp),
    ('patience', early_stopping),
    ('train_acc', train_acc),
    ('valid_acc', valid_acc)
]

clf = EEGClassifier(
    model,
    criterion=torch.nn.BCEWithLogitsLoss,
    optimizer=torch.optim.Adam,
    max_epochs=n_epochs,
    iterator_train__shuffle=False,
    iterator_train__sampler=train_sampler,
    iterator_valid__sampler=valid_sampler,
    iterator_train__num_workers=num_workers,
    iterator_valid__num_workers=num_workers,
    train_split=predefined_split(splitted['valid']),
    optimizer__lr=lr,
    batch_size=batch_size,
    callbacks=callbacks,
    device=device
)
# Model training for a specified number of epochs. `y` is None as it is already
# supplied in the dataset.
clf.fit(splitted['train'], y=None)
clf.load_params(checkpoint=cp)  # Load the model with the lowest valid_loss

# os.remove('./params.pt')  # Delete parameters file


# In[118]:


# PLOTS
# Extract loss and balanced accuracy values for plotting from history object

df = pd.DataFrame(clf.history.to_list())

df['train_acc'] *= 100
df['valid_acc'] *= 100

ys1 = ['train_loss', 'valid_loss']
ys2 = ['train_acc', 'valid_acc']
styles = ['-', ':']
markers = ['.', '.']

plt.style.use('seaborn-talk')

fig, ax1 = plt.subplots(figsize=(8, 3))
ax2 = ax1.twinx()
for y1, y2, style, marker in zip(ys1, ys2, styles, markers):
    ax1.plot(df['epoch'], df[y1], ls=style, marker=marker, ms=7,
             c='tab:blue', label=y1)
    ax2.plot(df['epoch'], df[y2], ls=style, marker=marker, ms=7,
             c='tab:orange', label=y2)

ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylabel('Loss', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.set_ylabel('Accuracy [%]', color='tab:orange')
ax1.set_xlabel('Epoch')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2)

plt.tight_layout()


# In[120]:


# CONFUSION MATRIX
# Switch to the test sampler

clf.iterator_valid__sampler = test_sampler
y_pred = clf.forward(splitted['test'], training=False) > 0
y_true = [y for _, _, y in test_sampler]

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))


# ---
# 
# ## TEST AREA
# 
# ---

# In[13]:


# Get embedder from model and return the features
_model = clf.module_.emb
_model.return_feats = True


# In[14]:


# model


# In[15]:


def print_dataset_lengths(datasets):
    for data in datasets:
        print(type(data))
        try:
            for i in range(100):
                print(f"({i}) length: {len(data)}  | 1st value: {data[0]} | type: {type(data)}")
                data = data[0]
        except:
            print("------------------------------------------------------------------")


print_dataset_lengths([X.datasets, X])



#### TESTING AREA ####


# attempt custom input
X = mne.io.read_raw_fif('sleep_staging_dataset/0-raw.fif')

# preprocessing

preprocessors = [
    Preprocessor(lambda x: x * 1e6),
    Preprocessor('filter', l_freq=None, h_freq=30, n_jobs=n_jobs)
]
preprocess(dataset, preprocessors)

# Create windows
window_size_samples = window_size_s * sfreq

mapping = {  # We merge stages 3 and 4 following AASM standards.
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4
}

from braindecode.datasets.mne import create_from_mne_raw

# define windows
X_windowed = create_from_mne_raw(
    X, trial_start_offset_samples=0, trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,
    drop_last_window=True)

# preprocess chanel-wise normalization
preprocess(windows_dataset, [Preprocessor(zscore)])