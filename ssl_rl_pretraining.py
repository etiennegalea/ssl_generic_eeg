# %pip install braindecode mne umap-learn skorch==0.10.0
# %pip install umap-learn[plot] pandas matplotlib datashader bokeh holoviews scikit-image colorcet

# imports
import os
import importlib
from datetime import datetime
import pickle
import numpy as np
import click
from tabulate import tabulate
import pprint
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import mne
import torch
from torch import nn
from torch.utils.data import DataLoader
from braindecode.datasets.sleep_physionet import SleepPhysionet
from braindecode.datasets import BaseConcatDataset, BaseDataset
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
from segment import Segmenter


# ----

# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)


# load preprocessed windowed data from previous run
def load_windowed_data(preprocessed_data):
    with open(f'data/preprocessed/{preprocessed_data}', 'rb') as f:
        windows_dataset = pickle.load(f)
    f.close()
    print(':: Preprocessed windowed data loaded...')
    return windows_dataset

def load_sleep_staging_windowed_dataset(subjects, subject_size, n_jobs, window_size_samples, low_cut_hz, high_cut_hz, sfreq):
    print(f':: loading SLEEP STAGING data...')
    dataset = SleepPhysionet(
        subject_ids=subjects[subject_size],
        recording_ids=[1],
        crop_wake_mins=30,
        load_eeg_only=True,
        sfreq=sfreq,
        n_jobs=n_jobs
    )

    preprocessors = [
        Preprocessor(lambda x: x * 1e6), # convert to microvolts
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz, n_jobs=n_jobs) # high pass filtering
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

    return windows_dataset



def load_space_bambi_raws(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_s):
    print(':: loading SPACE/BAMBI data')

    # space_bambi directory
    # data_dir = './data/SPACE_BAMBI_2channels/'
    data_dir = '/home/maligan/Documents/VU/Year_2/M.Sc._Thesis_[X_400285]/my_thesis/code/ssl_thesis/data/SPACE_BAMBI_2channels/'

    raws = []
    # added = 0

    print(f'{len(os.listdir(data_dir))} files found')
    for i, path in enumerate(os.listdir(data_dir)):
        # limiter
        if i == 25:
            break
            
        full_path = os.path.join(data_dir, path)
        # check whether raw has longer duration than window_size
        raw = mne.io.read_raw_fif(full_path)
        # duration = raw.times.max()
        # print(f':: duration of [{path}]: {duration} / {window_size_samples}')
        # if duration > window_size_samples:
        raws.append(raw)


    # preprocess dataset
    dataset = preprocess_raws(raws, sfreq, low_cut_hz, high_cut_hz, n_jobs)
    event_mapping = {0: 'artifact', 1: 'non-artifact', 2:'ignore'}

    # segment dataset recordings into windows and add descriptions
    raws, descriptions = [], []
    segmenter = Segmenter(window_size=window_size_s, window_overlap=0.5, cutoff_length=0.1)
    for subject_id, raw in enumerate(dataset):
        x = segmenter.segment(raw)
        annot_from_events = mne.annotations_from_events(events=x.events, event_desc=event_mapping, sfreq=x.info['sfreq'])
        duration_per_event = [x.times[-1]+x.times[1]]
        annot_from_events.duration = np.array(duration_per_event * len(x.events))
        raws += [raw.set_annotations(annot_from_events)]
        descriptions += [{"subject": int(subject_id), "recording": raw}]
    

    # create windows from epochs and descriptions
    ds = BaseConcatDataset([BaseDataset(raws[i], descriptions[i]) for i in range(len(descriptions))])
    window_size_samples = window_size_s * sfreq
    mapping = {
        'artifact': 0,
        'non-artifact': 1,
        'ignore': 2
    }

    windows_dataset = create_windows_from_events(
        ds, 
        # trial_start_offset_samples = 0,
        # trial_stop_offset_samples = 0,
        # window_size_samples = window_size_samples,
        # window_stride_samples = window_size_samples,
        mapping = mapping,
        # preload = True,
    )


    # channel-wise zscore normalization
    preprocess(windows_dataset, [Preprocessor(zscore)])

    return windows_dataset



def get_file_list(x):
    return [os.path.join(x, fname) for fname in os.listdir(x)]

def get_id(x):
    return x.split('/')[-1]


def preprocess_raws(raws, sfreq, low_cut_hz, high_cut_hz, n_jobs):
    print(':: PREPROCESSING RAWS')
    print(f'--resample {sfreq}')
    print(f'--high_cut freq {high_cut_hz}')
    print(f'--low_cut freq {low_cut_hz}')

    for raw in raws:
        mne.io.Raw.resample(raw, sfreq)   # resample
        mne.io.Raw.filter(raw, l_freq=low_cut_hz, h_freq=high_cut_hz, n_jobs=n_jobs)    # high-pass filter

    return raws


def create_windows_dataset(raws, window_size_samples, descriptions=None, mapping=None):
    print(f':: Creating windows of size: {window_size_samples}')

    windows_dataset = create_from_mne_raw(
        raws,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        drop_last_window=True,
        accepted_bads_ratio=0.0,
        drop_bad_windows=True,
        on_missing='ignore',
        descriptions=descriptions,
        mapping=mapping,
        # preload=True
    )

    # channel-wise zscore normalization
    preprocess(windows_dataset, [Preprocessor(zscore)])

    return windows_dataset


def load_abnormal_raws(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples):
    print(':: loading TUH abnormal data')

    data_dir = '/media/maligan/My Passport/msc_thesis/ssl_thesis/data/tuh_abnormal_data/train/'

    # build data dictionary
    annotations = {}
    for annotation in get_file_list(data_dir):
        subjects = {}
        for subject in get_file_list(annotation):
            recordings = {}
            for recording in get_file_list(subject):
                dates = {}
                for date in get_file_list(recording):
                    for raw_path in get_file_list(date):
                        if '_2_channels.fif' in get_id(raw_path):
                            break
                        else:
                            pass
                    dates[get_id(date)] = raw_path
                recordings[get_id(recording)] = dates
            subjects[get_id(subject)] = recordings
        annotations[get_id(annotation)] = subjects
    

    df = pd.json_normalize(annotations, sep='_').T

    # paths list
    raw_paths = [df.iloc[i][0] for i in range(len(df))]

    # define abnormal and normal subjects
    
    abnormal_subjects = annotations['abnormal'].keys()
    normal_subjects = annotations['normal'].keys()

    # define descriptions (recoding per subject)
    abnormal_descriptions, normal_descriptions, classification = [], [], []
    for id in abnormal_subjects:
        for recording in annotations['abnormal'][id].values():
            for x in recording.keys():
                abnormal_descriptions += [{'subject': id, 'recording': x}]
                classification += ['abnormal']
    for id in normal_subjects:
        for recording in annotations['normal'][id].values():
            for x in recording.keys():
                normal_descriptions += [{'subject': id, 'recording': x}]
                classification += ['normal']

    descriptions = abnormal_descriptions + normal_descriptions

    # shuffle raw_paths and descriptions
    from sklearn.utils import shuffle
    raw_paths, descriptions, classification = shuffle(raw_paths, descriptions, classification)

    # limiters
    raw_paths = raw_paths[:50]
    descriptions = descriptions[:50]
    classification = classification[:50]

    # load data and set annotations
    dataset = []
    for i, path in enumerate(raw_paths):
        _class = classification[i]
        raw = mne.io.read_raw_fif(path)
        raw = raw.set_annotations(mne.Annotations(onset=[0], duration=raw.times.max(), description=[_class]))
        dataset.append(raw)

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(raw_paths)
    pp.pprint(dataset)

    # preprocess dataset
    dataset = preprocess_raws(dataset, sfreq, low_cut_hz, high_cut_hz, n_jobs)

    mapping = {
        'abnormal': 0,
        'normal': 1
    }

    # create windows
    windows_dataset = create_windows_dataset(dataset, window_size_samples, descriptions, mapping)

    return windows_dataset




@click.command()
@click.option('--dataset_name', '--dataset', '-n', default='space_bambi', help='Dataset to be pretrained.')
@click.option('--subject_size', default='sample', help='sample (0-5), some (0-40), all (83)')
@click.option('--random_state', default=87, help='Set a static random state so that the same result is generated everytime.')
@click.option('--n_jobs', default=1, help='Number of subprocesses to run.')
@click.option('--window_size_s', default=5, help='Window sizes in seconds.')
# @click.option('--window_size_samples', default=3000, help='Window sizes in milliseconds.')
@click.option('--high_cut_hz', default=30, help='High-pass filter frequency.')
@click.option('--low_cut_hz', default=0.5, help='Low-pass filter frequency.')
@click.option('--sfreq', default=100, help='Sampling frequency of the input data.')
# @click.option('--emb_size', default=100, help='Embedding size of the model (should correspond to sampling frequency).')
@click.option('--lr', default=5e-3, help='Learning rate of the pretrained model.')
@click.option('--batch_size', default=512, help='Batch size of the pretrained model.')
@click.option('--n_epochs', default=15, help='Number of epochs while training the pretrained model.')
@click.option('--preprocessed_data', '-d', default=None, help='Preprocessed windowed data from previous run.')
@click.option('--accepted_bads_ratio', default=0, help='Acceptable proportion of trials with inconsistent length in a raw. \
                If the number of trials whose length is exceeded by the window size is \
                smaller than this, then only the corresponding trials are dropped, but \
                the computation continues.')

# https://physionet.org/content/sleep-edfx/1.0.0/
# Electrode locations Fpz-Cz, Pz-Oz
def main(dataset_name, subject_size, random_state, n_jobs, window_size_s, high_cut_hz, low_cut_hz, sfreq, lr, batch_size, n_epochs, preprocessed_data, accepted_bads_ratio):
    # init variables
    window_size_samples = window_size_s * sfreq
    emb_size = sfreq
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
    metadata_string = f'{dataset_name}_{window_size_s}s_windows_{len(subjects[subject_size])}_subjects_{device}_{n_epochs}_epochs_{sfreq}hz'

    # print all parameter vars
    print(tabulate(locals().items(), tablefmt='fancy_grid'))

    # if no windowed_data is specified, download it and preprocess it
    if preprocessed_data is not None:
        print(':: loading PREPROCESSED windowed dataset: ', preprocessed_data)
        windows_dataset = load_windowed_data(preprocessed_data)
    else:
        # windows_dataset = load_sleep_staging_windowed_dataset(subjects, subject_size, n_jobs, window_size_samples, low_cut_hz, high_cut_hz, sfreq)
        windows_dataset = load_space_bambi_raws(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_s)
        # windows_dataset = load_abnormal_raws(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples)

        ### save windows
        dir = '/home/maligan/Documents/VU/Year_2/M.Sc._Thesis_[X_400285]/my_thesis/code/ssl_thesis/data/preprocessed/'
        hf.check_dir(dir)
        with open(f'{dir}{hf.get_datetime()}_{metadata_string}.pkl', 'wb+') as f:
            pickle.dump(windows_dataset, f)
        f.close()
        print(':: Data loaded, preprocessed and windowed.')

    print(':: starting training...')

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
        random_state=random_state)
    test_sampler = RelativePositioningSampler(
        splitted['test'].get_metadata(), tau_pos=tau_pos, tau_neg=tau_neg,
        n_examples=n_examples_test, same_rec_neg=True,
        random_state=random_state)


    ### Creating the model

    # Extract number of channels and time steps from dataset
    n_channels, input_size_samples = windows_dataset[0][0].shape
    print(f':: number of channels: {n_channels}\n:: input size samples: {input_size_samples}')


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
    p = Plot(dataset_name, metadata_string)
    p.plot_acc(clf.history.to_list())


    # Switch to the test sampler
    clf.iterator_valid__sampler = test_sampler
    y_pred = clf.forward(splitted['test'], training=False) > 0
    y_true = [y for _, _, y in test_sampler]

    # confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)
    # save plot
    p.plot_confusion_matrix(conf_matrix)

    # classification report
    class_report = classification_report(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    # save report
    dir = 'classification_reports/pretrained/'
    hf.check_dir(dir)
    with open(f'{dir}{hf.get_datetime()}_class_report_{metadata_string}.txt', "w") as f:
        f.write(pprint.pformat(class_report, indent=4, sort_dicts=False))

    ### Save model
    dir = './models/pretrained/'
    hf.check_dir(dir)
    model_name = f'{dir}{hf.get_datetime()}_{metadata_string}.model'
    torch.save(model, model_name)

    print(f'Model trained ~ {os.path.dirname(os.path.abspath(__file__))}/{model_name}')

if __name__ == '__main__':
    main()
