# %pip install braindecode mne umap-learn skorch==0.10.0
# %pip install umap-learn[plot] pandas matplotlib datashader bokeh holoviews scikit-image colorcet

# imports
from datetime import datetime
import pickle
import numpy as np
import click
import os
from time import sleep
from tabulate import tabulate
import pandas as pd
import pprint

import mne
import torch
from torch import nn
from torch.utils.data import DataLoader
from braindecode.datasets.sleep_physionet import SleepPhysionet
from braindecode.datasets.base import BaseConcatDataset, BaseDataset, WindowsDataset
from braindecode.datautil.preprocess import preprocess, Preprocessor
from braindecode.preprocessing.windowers import create_windows_from_events, create_fixed_length_windows
from braindecode.util import set_random_seeds
from braindecode.datautil.preprocess import zscore
from braindecode.datasets import create_from_mne_raw

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

from  mat73 import loadmat
import matplotlib.pyplot as plt

# classes
from helper_funcs import HelperFuncs as hf
from ContrastiveNet import *
from RelativePositioningDataset import *
from plot import Plot
from segment import Segmenter


# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)


### Load model
@click.command()
@click.option('--dataset_name', '--dataset', '-n', default='white_noise', help='Dataset for downstream task: \
    "space_bambi", "sleep_staging", "tuh_abnormal", "scopolamine", "white_noise", "bci".')
@click.option('--subject_size', default='all', help='sample (0-5), some (0-40), all (83)')
# @click.option('--subject_size', nargs=2, default=[1,10], type=int, help='Number of subjects to be trained - max 110.')
@click.option('--random_state', default=87, help='Set a static random state so that the same result is generated everytime.')
@click.option('--n_jobs', default=2, help='Number of subprocesses to run.')
@click.option('--window_size_s', default=5, help='Window sizes in seconds.')
@click.option('--high_cut_hz', '--hfreq', '-h', default=30, help='High-pass filter frequency.')
@click.option('--low_cut_hz', '--lfreq', '-l', default=0.5, help='Low-pass filter frequency.')
@click.option('--sfreq', default=100, help='Sampling frequency of the input data.')
@click.option('--lr', default=5e-3, help='Learning rate of the pretrained model.')
@click.option('--batch_size', default=256, help='Batch size of the pretrained model.')
@click.option('--n_channels', default=2, help='Number of channels.')
@click.option('--connectivity_plot', default=False, help='Plot UMAP connectivity plot.')
@click.option('--edge_bundling_plot', default=False, help='Plot UMAP connectivity plot with edge bundling (takes a long time).')
@click.option('--plot_heavy', '-p', default=True, help='Plot heavy CPU intensive plots.')
@click.option('--show_plots', '--show', default=False, help='Show plots.')
@click.option('--load_feature_vectors', default=None, help='Load feature vectors passed through SSL model (input name of vector file).')
@click.option('--load_latest_model', default=False, help='Load the latest pretrained model from the ssl_rl_pretraining.py script.')
@click.option('--fully_supervised', default=True, help='Train a fully-supervised model for comparison with the downstream task.')


def main(dataset_name, subject_size, random_state, n_jobs, window_size_s, low_cut_hz, high_cut_hz, sfreq, lr, batch_size, n_channels, connectivity_plot, edge_bundling_plot, plot_heavy, show_plots, load_feature_vectors, load_latest_model, fully_supervised):
    print(':: STARTING MAIN ::')
    
    # print local parameters
    # set device to 'cuda' or 'cpu'
    device = hf.enable_cuda()
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=random_state, cuda=device == 'cuda')

    window_size_samples = window_size_s * sfreq

    # load the pretrained model
    # (load the best model)
    if load_latest_model:
        model_dir = "./models/pretrained/"
        # model_dir = "/home/maligan/Documents/VU/Year_2/M.Sc._Thesis_[X_400285]/my_thesis/code/ssl_thesis/models/pretrained/"
        files = [os.path.join(model_dir, fname) for fname in os.listdir(model_dir)]
        latest = max(files, key=os.path.getmtime).split(model_dir)[1].split('.')[0]
        print(f":: loading the latest pretrained model: {latest}")
        model = torch.load(f"{model_dir}{latest}.model")
    else:
        model = torch.load("models/pretrained/2021_12_16__10_23_49_sleep_staging_5s_windows_83_subjects_cpu_15_epochs_100hz.model")
        # model = torch.load("models/pretrained/2022_01_31__12_02_47_sleep_staging_5s_windows_5_subjects_cpu_15_epochs_100hz.model")

    print(model)

    # DOWNSTREAM TASK - FINE TUNING)
    if load_feature_vectors is None:

        if dataset_name == 'sleep_staging':
            windows_dataset = load_sleep_staging_windowed_dataset(subject_size, n_jobs, window_size_samples, high_cut_hz, sfreq)
            annotations = ['W', 'N1', 'N2', 'N3', 'R']
        elif dataset_name == 'bci':
            windows_dataset = load_bci_data(subject_size, sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples)
            annotations = ['T0', 'T1', 'T2']
        elif dataset_name == 'tuh_abnormal':
            windows_dataset = load_abnormal_raws(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples)
            annotations = ['abnormal', 'normal']
        elif dataset_name == 'space_bambi':
            windows_dataset = load_space_bambi_raws(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_s)
            annotations = ['artifact', 'non-artifact', 'ignored']
        elif dataset_name == 'scopolamine':
            windows_dataset = load_scopolamine_data(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples)
            annotations = ['M01', 'M05', 'M11']
        elif dataset_name == 'white_noise':
            windows_dataset = load_abnormal_noise_raws(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples)
            annotations = ['abnormal', 'normal', 'white_noise']
        elif dataset_name == 'tuar':
            windows_dataset = load_tuar_raws(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples)
            annotations = ['eyem', 'chew', 'shiv', 'musc', 'elec']


        # print all parameter vars
        setup = tabulate(locals().items(), tablefmt='fancy_grid')
        print(setup)

        # split by subject
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

        # init metadata_string for file naming
        metadata_string = f'{dataset_name}_{window_size_s}s_windows_{len(subjects)}_subjects_{device}_{sfreq}hz'

        # write setup to file
        dir = 'setup/downstream/'
        hf.check_dir(dir)
        with open(f'{dir}{hf.get_datetime()}_setup_{metadata_string}.txt', "w") as f:
            f.write(pprint.pformat(setup, indent=4, sort_dicts=False))


        # init plotting object
        p = Plot(dataset_name, metadata_string, show=show_plots)

        ### with (false) or without (true) sequential layer
        model.emb.return_feats = True

        num_workers = n_jobs

        # Extract features with the trained embedder
        data, raw_data = dict(), dict()
        for name, split in splitted.items():
            split.return_pair = False  # Return single windows
            loader = DataLoader(split, batch_size=batch_size, num_workers=num_workers)
            with torch.no_grad():
                feats = [model.emb(batch_x.to(device)).cpu().numpy() for batch_x, _, _ in loader]
                # make a copy of the vectors WITHOUT passing them through the pretrained model
                raw_vectors = [batch_x.to(device).cpu().numpy() for batch_x, _, _ in loader]
            data[name] = (np.concatenate(feats), split.get_metadata()['target'].values)
            # concatenate channels and duplicate labels
            raw_data[name] = (np.concatenate(np.concatenate(raw_vectors)), np.tile(splitted[name].get_metadata()['target'].values, n_channels))


        # combine all vectors (X) and labels (y) from DATA sets
        X = np.concatenate([v[0] for k, v in data.items()])
        y = np.concatenate([v[1] for k, v in data.items()])

        # combine all vectors (X) and labels (y) from RAW_DATA sets
        X_raw = np.concatenate([v[0] for k, v in raw_data.items()])
        y_raw = np.concatenate([v[1] for k, v in raw_data.items()])

        # Initialize the logistic regression model
        log_reg = LogisticRegression(
            penalty='l2', C=1.0, class_weight='balanced', solver='newton-cg',
            multi_class='multinomial', random_state=random_state, max_iter=1000, tol=0.01)
        clf_pipe = make_pipeline(StandardScaler(), log_reg)

        # estimate learning curve specifications
        ssl_train_sizes, ssl_train_scores, ssl_test_scores = learning_curve(
            clf_pipe,
            X=X,
            y=y,
            cv=5,
            scoring='balanced_accuracy',
            n_jobs=-1,
            train_sizes = np.linspace(0.00001,1,20),
            shuffle=True
        )

        # Fit and score the logistic regression
        clf_pipe.fit(*data['train'])
        train_y_pred = clf_pipe.predict(data['train'][0])
        valid_y_pred = clf_pipe.predict(data['valid'][0])
        test_y_pred = clf_pipe.predict(data['test'][0])

        train_bal_acc = balanced_accuracy_score(data['train'][1], train_y_pred)
        valid_bal_acc = balanced_accuracy_score(data['valid'][1], valid_y_pred)
        test_bal_acc = balanced_accuracy_score(data['test'][1], test_y_pred)

        print(f'{dataset_name} performance with logistic regression:')
        print(f'Train bal acc: {train_bal_acc:0.4f}')
        print(f'Valid bal acc: {valid_bal_acc:0.4f}')
        print(f'Test bal acc: {test_bal_acc:0.4f}')

        print('Results on test set:')
        # confusion matrix
        conf_matrix = confusion_matrix(data['test'][1], test_y_pred)
        print(conf_matrix)
        # save plot
        p.plot_confusion_matrix(conf_matrix)

        # classification report
        class_report = classification_report(data['test'][1], test_y_pred)
        print(class_report)
        # save report
        dir = 'classification_reports/downstream/'
        hf.check_dir(dir)
        with open(f'{dir}{hf.get_datetime()}_class_report_{metadata_string}.txt', "w") as f:
            f.write(pprint.pformat(class_report, indent=4, sort_dicts=False))

        ### save fine-tuned model
        dir = 'models/finetuned/'
        hf.check_dir(dir)
        with open(f'{dir}{hf.get_datetime()}_{metadata_string}.pkl', 'wb+') as f:
            pickle.dump(clf_pipe, f)
        f.close()

        ### save feature vectors
        dir = 'data/feature_vectors/'
        hf.check_dir(dir)
        with open(f'{dir}{hf.get_datetime()}_{metadata_string}.pkl', 'wb+') as f:
            pickle.dump([X,y], f)
        f.close()

    else:
        ### load fine-tuned model
        metadata_string = load_feature_vectors
        # with open(f'{metadata_string}.pkl', 'rb') as f:
        #     clf_pipe = pickle.load(f)
        # f.close()

        ### load feature vectors
        print(f':: loading feature vectors: {load_feature_vectors}')
        with open(f'data/feature_vectors/{load_feature_vectors}.pkl', 'rb') as f:
            feature_vectors = pickle.load(f)
        f.close()

        X, y = feature_vectors[0], feature_vectors[1]


    ### Visualizing clusters
    p.plot_PCA(X, y, annotations)
    p.plot_TSNE(X, y, annotations)
    p.plot_UMAP(X, y, annotations)
    if connectivity_plot:
        p.plot_UMAP_connectivity(X)
    if edge_bundling_plot:
        p.plot_UMAP_connectivity(X, edge_bundling=True)
    p.plot_UMAP_3d(X, y)


    # plotting with raw data (not embeddings)
    p_fs = Plot('RAW_'+dataset_name, metadata_string, show=show_plots)
    p_fs.plot_PCA(X_raw, y_raw, annotations)
    p_fs.plot_TSNE(X_raw, y_raw, annotations)
    p_fs.plot_UMAP(X_raw, y_raw, annotations)
    if connectivity_plot:
        p_fs.plot_UMAP_connectivity(X_raw)
    if edge_bundling_plot:
        p_fs.plot_UMAP_connectivity(X_raw, edge_bundling=True)
    p_fs.plot_UMAP_3d(X_raw, y_raw)



    ### Train a fully-supervised logistic regresion for comparison and evaluation
    if fully_supervised:
        print(f':: Performing Fully-Supervised Logistic Regression for {dataset_name}')
        # re-init logistic regression
        log_reg = LogisticRegression(
            penalty='l2', C=1.0, class_weight='balanced', solver='newton-cg',
            multi_class='multinomial', random_state=random_state, max_iter=1000, tol=0.01)
        fs_pipe = make_pipeline(StandardScaler(), log_reg)

        raw_train_sizes, raw_train_scores, raw_test_scores = learning_curve(
            fs_pipe,
            X=X_raw,
            y=y_raw,
            cv=5,
            scoring='balanced_accuracy',
            n_jobs=-1,
            train_sizes = np.linspace(0.00001,1,40),
            shuffle=True
        )

        # Fit and score the logistic regression on raw vectors
        fs_pipe.fit(*raw_data['train'])
        train_y_pred = fs_pipe.predict(raw_data['train'][0])
        valid_y_pred = fs_pipe.predict(raw_data['valid'][0])
        test_y_pred = fs_pipe.predict(raw_data['test'][0])

        train_bal_acc = balanced_accuracy_score(raw_data['train'][1], train_y_pred)
        valid_bal_acc = balanced_accuracy_score(raw_data['valid'][1], valid_y_pred)
        test_bal_acc = balanced_accuracy_score(raw_data['test'][1], test_y_pred)

        print(f'{dataset_name} with logistic regression:')
        print(f'Train bal acc: {train_bal_acc:0.4f}')
        print(f'Valid bal acc: {valid_bal_acc:0.4f}')
        print(f'Test bal acc: {test_bal_acc:0.4f}')

        print('Results on test set:')
        # confusion matrix
        conf_matrix = confusion_matrix(raw_data['test'][1], test_y_pred)
        print(conf_matrix)
        # save plot
        p.plot_confusion_matrix(conf_matrix, 'FS')

        # classification report
        class_report = classification_report(raw_data['test'][1], test_y_pred)
        print(class_report)
        # save report
        dir = 'classification_reports/downstream/FS/'
        hf.check_dir(dir)
        with open(f'{dir}{hf.get_datetime()}_class_report_{metadata_string}.txt', "w") as f:
            f.write(pprint.pformat(class_report, indent=4, sort_dicts=False))

        # plotting learning curves
        p.plot_learning_curves(
            ssl_train_sizes,
            raw_train_sizes,
            ssl_train_scores=ssl_train_scores, 
            ssl_test_scores=ssl_test_scores, 
            raw_train_scores=raw_train_scores, 
            raw_test_scores=raw_test_scores, 
            dataset_name=dataset_name
        )


# ---------------------------------- PROCESSING FUNCTIONS ----------------------------------


def preprocess_raws(raws, sfreq, low_cut_hz, high_cut_hz, n_jobs):
    print(':: PREPROCESSING RAWS')
    print(f'--resample {sfreq}')
    print(f'--high_cut freq {high_cut_hz}')
    print(f'--low_cut freq {low_cut_hz}')

    for raw in raws:
        raw = raw.resample(sfreq)   # resample
        raw = raw.filter(l_freq=low_cut_hz, h_freq=high_cut_hz, n_jobs=n_jobs)    # filtering

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


        
# ---------------------------------- LOADING DATASETS ----------------------------------


# load BCI data
def load_bci_data(subject_size, sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples):
    print(':: loading BCI data')

    '''
    ANNOTATIONS

    T0 corresponds to rest
    T1 corresponds to onset of motion (real or imagined) of
        the left fist (in runs 3, 4, 7, 8, 11, and 12)
        both fists (in runs 5, 6, 9, 10, 13, and 14)
    T2 corresponds to onset of motion (real or imagined) of
        the right fist (in runs 3, 4, 7, 8, 11, and 12)
        both feet (in runs 5, 6, 9, 10, 13, and 14)

    =========  ===================================
    run        task
    =========  ===================================
    1          Baseline, eyes open
    2          Baseline, eyes closed
    3, 7, 11   Motor execution: left vs right hand
    4, 8, 12   Motor imagery: left vs right hand
    5, 9, 13   Motor execution: hands vs feet
    6, 10, 14  Motor imagery: hands vs feet
    =========  ===================================

    '''
    print(subject_size)
    print(type(subject_size))
    subjects = range(subject_size[0], subject_size[1]) # max 110
    event_codes = [
        1, 2,       # eyes open, eyes closed (baselines)
        3, 7, 11,   # Motor execution: left vs right hand
        4, 8, 12,   # Motor imagery: left vs right hand
        5, 9, 13,   # Motor execution: hands vs feet
        6, 10, 14   # Motor imagery: hands vs feet
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

    # preprocess dataset
    dataset = preprocess_raws(eegmmidb, sfreq, low_cut_hz, high_cut_hz, n_jobs)

    # create windows
    windows_dataset = create_windows_dataset(dataset, window_size_samples, descriptions)

    return windows_dataset


def load_sleep_staging_windowed_dataset(subject_size, n_jobs, window_size_samples, high_cut_hz, sfreq):
    print(f':: loading SLEEP STAGING data...')

    subjects = {
        'sample': [*range(5)],
        'some': [*range(0,40)],
        'all': [*range(0,83)],
    }

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

    return windows_dataset


def load_space_bambi_raws(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_s):
    print(':: loading SPACE/BAMBI data')

    # space_bambi directory
    data_dir = './data/SPACE_BAMBI_2channels/'
    # data_dir = '/media/maligan/My Passport/msc_thesis/data/SPACE_BAMBI_2channels/'

    raws = []

    print(f'{len(os.listdir(data_dir))} files found')
    for i, path in enumerate(os.listdir(data_dir)):
        # limiter
        # if i == 5:
        #    break
            
        full_path = os.path.join(data_dir, path)
        raw = mne.io.read_raw_fif(full_path)
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
        mapping = mapping,
    )


    # channel-wise zscore normalization
    preprocess(windows_dataset, [Preprocessor(zscore)])

    return windows_dataset



# load mats (for scopolamine dataset)
def load_mats(path, info, classification, dataset, descriptions):
    raws, desc = [], []

    mats = os.listdir(path)
    for i, mat in enumerate(mats):
        print(mat)
        # select columns 3 and 4 (Fpz-Cz, and Pz-Oz respectively) and convert to microvolts
        x = loadmat(path + mat)['RawSignal'][:, [2,3]].T / 100000
        raw = mne.io.RawArray(x, info)
            
        # subject
        subject = int(mat.split('.')[1][2:])
        # recording (occasion)
        recording = int(mat.split('.')[-2].split('M')[0][1:])
        # treatment period
        treatment_period = int(mat.split('.')[-2].split('M')[-1])

        # if even (not placebo)
        if not recording&1:
            raws += [raw.set_annotations(mne.Annotations(onset=[0], duration=raw.times.max(), description=[classification]))]
            desc += [{'subject': subject, 'recording': recording, 'treatment_period': treatment_period}]
            

    dataset += raws
    descriptions += desc

    return dataset, descriptions

# load scopolamine data
def load_scopolamine_data(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples):
    print(':: loading SCOPOLAMINE data')

    # 11 measurements times from 0.5 hrs to 8.5 hrs after Scopolamine (or placebo) administration
    m01 = 'data/scopolamine/M01/'
    # m01 = '/media/maligan/My Passport/msc_thesis/data/scopolamine/M01/'
    m05 = 'data/scopolamine/M05/'
    # m05 = '/media/maligan/My Passport/msc_thesis/data/scopolamine/M05/'
    m11 = 'data/scopolamine/M11/'
    # m11 = '/media/maligan/My Passport/msc_thesis/data/scopolamine/M11/'

    dataset, descriptions = [], []
    info = mne.create_info(ch_names=['Fpz-cz', 'Pz-Oz'], ch_types=['eeg']*2, sfreq=1012)

    dataset, descriptions = load_mats(m01, info, 'm01', dataset, descriptions)
    dataset, descriptions = load_mats(m05, info, 'm05', dataset, descriptions)
    dataset, descriptions = load_mats(m11, info, 'm11', dataset, descriptions)


    # preprocess dataset
    dataset = preprocess_raws(dataset, sfreq, low_cut_hz, high_cut_hz, n_jobs)

    mapping = {
        'm01': 0,
        'm05': 1,
        'm11': 2
    }

    # shuffle raw_paths and descriptions
    from sklearn.utils import shuffle
    dataset, descriptions = shuffle(dataset, descriptions)


    # create windows
    windows_dataset = create_windows_dataset(dataset, window_size_samples, descriptions, mapping)

    return windows_dataset


# load scopolamine data
def load_scopolamine_test_data(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples):
    print(':: loading SCOPOLAMINE data')

    # 11 measurements times from 0.5 hrs to 8.5 hrs after Scopolamine (or placebo) administration
    m01 = 'data/scopolamine/M01/'
    # m01 = '/media/maligan/My Passport/msc_thesis/data/scopolamine/M01/'
    m11 = 'data/scopolamine/M11/'
    # m11 = '/media/maligan/My Passport/msc_thesis/data/scopolamine/M11/'

    dataset, descriptions = [], []
    info = mne.create_info(ch_names=['Fpz-cz', 'Pz-Oz'], ch_types=['eeg']*2, sfreq=1012)

    dataset, descriptions = load_mats(m01, info, 'm01', dataset, descriptions)
    dataset, descriptions = load_mats(m11, info, 'm11', dataset, descriptions, test=True)
    

    # preprocess dataset
    dataset = preprocess_raws(dataset, sfreq, low_cut_hz, high_cut_hz, n_jobs)

    mapping = {
        'm01': 0,
        'm11': 1,
    }

    # shuffle raw_paths and descriptions
    # from sklearn.utils import shuffle
    # dataset, descriptions = shuffle(dataset, descriptions)


    # create windows
    windows_dataset = create_windows_dataset(dataset, window_size_samples, descriptions, mapping)

    return windows_dataset


def load_abnormal_raws(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples):
    print(':: loading TUH abnormal data')

    data_dir = 'data/tuh_abnormal_data/eval/'
    # data_dir = '/media/maligan/My Passport/msc_thesis/data/tuh_abnormal_data/eval/'

    # build data dictionary
    annotations = {}
    for annotation in hf.get_file_list(data_dir):
        subjects = {}
        for subject in hf.get_file_list(annotation):
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
        annotations[hf.get_id(annotation)] = subjects
    

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
                abnormal_descriptions += [{'subject': int(id), 'recording': x}]
                classification += ['abnormal']
    for id in normal_subjects:
        for recording in annotations['normal'][id].values():
            for x in recording.keys():
                normal_descriptions += [{'subject': int(id), 'recording': x}]
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
        raw = mne.io.read_raw_fif(path, preload=True)
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


# load simulated noisy signals (white noise and sin waves with normal noise) for testing model
def load_generated_noisy_signals(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples):
    print(':: loading TUH abnormal data')

    dataset, descriptions = [], []
    for i in range(50):
        # alternate between generating normal noise and white noise
        dataset += [hf.generate_noisy_raws(n_times=50000) if i&1 else hf.generate_white_noise_raws(n_times=50000)]
        descriptions += [{'subject': i}]
            
    # preprocess dataset
    dataset = preprocess_raws(dataset, sfreq, low_cut_hz, high_cut_hz, n_jobs)

    mapping = {
        'white_noise': 0,
        'normal_noise': 1
    }

    # create windows
    windows_dataset = create_windows_dataset(dataset, window_size_samples, descriptions, mapping)

    return windows_dataset


def load_abnormal_noise_raws(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples):
    print(':: loading TUH abnormal data + white noise')

    data_dir = 'data/tuh_abnormal_data/eval/'
    # data_dir = '/media/maligan/My Passport/msc_thesis/data/tuh_abnormal_data/eval/'

    # build data dictionary
    annotations = {}
    for annotation in hf.get_file_list(data_dir):
        subjects = {}
        for subject in hf.get_file_list(annotation):
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
        annotations[hf.get_id(annotation)] = subjects
    

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
                abnormal_descriptions += [{'subject': int(id), 'recording': x}]
                classification += ['abnormal']
    for id in normal_subjects:
        for recording in annotations['normal'][id].values():
            for x in recording.keys():
                normal_descriptions += [{'subject': int(id), 'recording': x}]
                classification += ['normal']

    descriptions = abnormal_descriptions + normal_descriptions


    # shuffle raw_paths and descriptions
    from sklearn.utils import shuffle
    dataset, descriptions, classification = shuffle(raw_paths, descriptions, classification)


    # limiters
    raw_paths = raw_paths[:50]
    descriptions = descriptions[:50]
    classification = classification[:50]

    # load data and set annotations
    dataset = []
    for i, path in enumerate(raw_paths):
        _class = classification[i]
        raw = mne.io.read_raw_fif(path, preload=True)
        raw = raw.set_annotations(mne.Annotations(onset=[0], duration=raw.times.max(), description=[_class]))
        dataset.append(raw)

    # -------------------- NOISE GEN --------------------------------

    for i in range(len(raw_paths)):
        dataset += [hf.generate_white_noise_raws(n_times=5000)]
        descriptions += [{'subject': i}]

    # ---------------------------------------------------------------

    # reshuffle with white noise added in dataset
    dataset, descriptions = shuffle(dataset, descriptions)



    # preprocess dataset
    dataset = preprocess_raws(dataset, sfreq, low_cut_hz, high_cut_hz, n_jobs)

    mapping = {
        'abnormal': 0,
        'normal': 1,
        'white_noise': 2
    }

    # create windows
    windows_dataset = create_windows_dataset(dataset, window_size_samples, descriptions, mapping)

    return windows_dataset


def load_tuar_raws(sfreq, low_cut_hz, high_cut_hz, n_jobs, window_size_samples):
    print(':: loading TUAR data')

    # data_dir = 'data/tuar/v2.1.0/edf/01_tcp_ar/'
    data_dir = '/media/maligan/My Passport/msc_thesis/data/tuar/v2.1.0/edf/01_tcp_ar/'

    # build data dictionary
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
    

    df = pd.json_normalize(subjects, sep='_').T

    # paths list
    raw_paths = [df.iloc[i][0] for i in range(len(df))]

    # define abnormal and normal subjects
    
    abnormal_subjects = subjects['abnormal'].keys()
    normal_subjects = subjects['normal'].keys()

    # define descriptions (recoding per subject)
    abnormal_descriptions, normal_descriptions, classification = [], [], []
    for id in abnormal_subjects:
        for recording in subjects['abnormal'][id].values():
            for x in recording.keys():
                abnormal_descriptions += [{'subject': int(id), 'recording': x}]
                classification += ['abnormal']
    for id in normal_subjects:
        for recording in subjects['normal'][id].values():
            for x in recording.keys():
                normal_descriptions += [{'subject': int(id), 'recording': x}]
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
        raw = mne.io.read_raw_fif(path, preload=True)
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



if __name__ == '__main__':
    main()
