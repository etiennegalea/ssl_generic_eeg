# %pip install braindecode mne umap-learn skorch==0.10.0
# %pip install umap-learn[plot] pandas matplotlib datashader bokeh holoviews scikit-image colorcet

# imports
import os
import importlib
from datetime import datetime
import pickle
import numpy as np
import click

import mne
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

# ----


# classes
from helper_funcs import HelperFuncs as hf
from ContrastiveNet import *
from RelativePositioningDataset import *
from plot import Plot


# ----

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)


# load preprocessed windowed data from previous run
def load_windowed_data(preprocessed_data):
    with open(f'data/preprocessed/{preprocessed_data}', 'rb') as f:
        windows_dataset = pickle.load(f)
    f.close()
    return windows_dataset


@click.command()
@click.option('--subject_size', default='sample', help='sample (0-5), some (0-40), all (64)')
@click.option('--random_state', default=87, help='')
@click.option('--n_jobs', default=1, help='')
@click.option('--window_size_s', default=5, help='Window sizes in seconds.')
@click.option('--window_size_samples', default=500, help='Window sizes in milliseconds.')
@click.option('--high_cut_hz', default=30, help='High-pass filter frequency.')
@click.option('--low_cut_hz', default=0, help='Low-pass filter frequency.')
@click.option('--sfreq', default=160, help='Sampling frequency of the input data.')
@click.option('--emb_size', default=160, help='Embedding size of the model (should correspond to sampling frequency).')
@click.option('--lr', default=5e-3, help='Learning rate of the pretrained model.')
@click.option('--batch_size', default=512, help='Batch size of the pretrained model.')
@click.option('--n_epochs', default=12, help='Number of epochs while training the pretrained model.')
@click.option('--preprocessed_data', '-d', default=None, help='Preprocessed windowed data from previous run.')

# https://physionet.org/content/sleep-edfx/1.0.0/
# Electrode locations Fpz-Cz, Pz-Oz
def main(subject_size, random_state, n_jobs, window_size_s, window_size_samples, high_cut_hz, low_cut_hz, sfreq, emb_size, lr, batch_size, n_epochs, preprocessed_data):

    
    # set number of workers for EEGClassifier to the same as n_jobs
    num_workers = n_jobs

    device = hf.enable_cuda()
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=random_state, cuda=device == 'cuda')

    subjects = {
            'sample': [*range(5)],
            'some': [*range(0,40)],
            'all': [*range(0,83)],
        }

    metadata_string = f'sleep_staging_{window_size_s}s_windows_{len(subjects[subject_size])}_subjects_{device}_{n_epochs}_epochs_{sfreq}hz'

    # if no windowed_data is specified, download it and preprocess it
    if preprocessed_data is not None:
        print(':: loading windowed dataset: ', preprocessed_data)
        windows_dataset = load_windowed_data(preprocessed_data)
    else:
        dataset = SleepPhysionet(
            subject_ids=subjects[subject_size],
            # recording_ids=[1],
            crop_wake_mins=30,
            load_eeg_only=True,
            resample=160,
            n_jobs=n_jobs
        )

        preprocessors = [
            Preprocessor(lambda x: x * 1e6), # convert to microvolts
            Preprocessor('filter', l_freq=None, h_freq=high_cut_hz, n_jobs=n_jobs) # high pass filtering
        ]

        # Transform the data
        preprocess(dataset, preprocessors)

        # Extracting windows

        mapping = {  # We merge stages 3 and 4 following AASM standards.
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,
            'Sleep stage R': 4
        }

        windows_dataset = create_windows_from_events(
            dataset, trial_start_offset_samples=0, trial_stop_offset_samples=0,
            window_size_samples=window_size_samples,
            window_stride_samples=window_size_samples, preload=True, mapping=mapping)

        ### Preprocessing windows
        preprocess(windows_dataset, [Preprocessor(zscore)])

        ### save fine-tuned model
        with open(f'data/preprocessed/{hf.get_datetime()}_{metadata_string}.pkl', 'wb+') as f:
            pickle.dump(windows_dataset, f)
        f.close()

    print(':: Preprocessed windowed dataset loaded.')

    ### Splitting dataset into train, valid and test sets

    subjects = np.unique(windows_dataset.description['subject'])
    subj_train, subj_test = train_test_split(
        subjects, test_size=0.4, random_state=random_state)
    subj_valid, subj_test = train_test_split(
        subj_test, test_size=0.5, random_state=random_state)


    split_ids = {'train': subj_train, 'valid': subj_valid, 'test': subj_test}
    splitted = dict()
    for name, values in split_ids.items():
        splitted[name] = RelativePositioningDataset(
            [ds for ds in windows_dataset.datasets
            if ds.description['subject'] in values])


    ### Creating samplers

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


    ### Creating the model

    # Extract number of channels and time steps from dataset
    n_channels, input_size_samples = windows_dataset[0][0].shape


    emb = SleepStagerChambon2018(
        n_channels,
        sfreq,
        n_classes=emb_size,
        n_conv_chs=16,
        input_size_s=input_size_samples / sfreq,
        dropout=0,
        apply_batch_norm=True
    )

    model = ContrastiveNet(emb, emb_size).to(device)


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

    os.remove('./params.pt')  # Delete parameters file

    ## Visualizing the results
    p = Plot(metadata_string)
    p.plot_acc(clf.history.to_list())


    # Switch to the test sampler
    clf.iterator_valid__sampler = test_sampler
    y_pred = clf.forward(splitted['test'], training=False) > 0
    y_true = [y for _, _, y in test_sampler]

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))


    ### Save model
    model_name = f'models/pretrained/{hf.get_datetime()}_{metadata_string}.model'
    torch.save(model, model_name)

    print(f'Model trained ~ {os.path.dirname(os.path.abspath(__file__))}/{model_name}')

if __name__ == '__main__':
    main()
