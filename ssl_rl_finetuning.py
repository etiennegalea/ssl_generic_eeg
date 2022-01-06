# %pip install braindecode mne umap-learn skorch==0.10.0
# %pip install umap-learn[plot] pandas matplotlib datashader bokeh holoviews scikit-image colorcet

# imports
from datetime import datetime
import pickle
import numpy as np
import click
import os
from time import sleep

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


# Authors: Hubert Banville <hubert.jbanville@gmail.com>
#
# License: BSD (3-clause)


### Load model
@click.command()
@click.option('--dataset_name', '--dataset', '-n', default='bci', help='Dataset to be finetuned.')
@click.option('--subject_size', nargs=2, default=[1,10], type=int, help='Number of subjects to be trained - max 110.')
@click.option('--random_state', default=87, help='')
@click.option('--n_jobs', default=1, help='')
# @click.option('--num_workers', default=1, help='')  # same as n_jobs
@click.option('--window_size_s', default=5, help='Window sizes in seconds.')
@click.option('--window_size_samples', default=500, help='Window sizes in milliseconds.')
@click.option('--high_cut_hz', '--hfreq', '-h', default=30, help='High-pass filter frequency.')
@click.option('--low_cut_hz', '--lfreq', '-l', default=0, help='Low-pass filter frequency.')
@click.option('--sfreq', default=160, help='Sampling frequency of the input data.')
@click.option('--emb_size', default=160, help='Embedding size of the model (should correspond to sampling frequency).')
@click.option('--lr', default=5e-3, help='Learning rate of the pretrained model.')
@click.option('--batch_size', default=512, help='Batch size of the pretrained model.')
@click.option('--n_epochs', default=12, help='Number of epochs while training the pretrained model.')
@click.option('--n_channels', default=2, help='Number of epochs while training the pretrained model.')
@click.option('--input_size_samples', default=500, help='Number of epochs while training the pretrained model.')
@click.option('--edge_bundling_plot', default=False, help='Plot UMAP connectivity plot with edge bundling (takes a long time).')
@click.option('--annotations', default=['T0', 'T1', 'T2'], help='Annotations for plotting.')
@click.option('--show_plots', '--show', default=False, help='Show plots.')
@click.option('--load_feature_vectors', default=None, help='Load feature vectors passed through SSL model (input name of vector file).')
@click.option('--load_latest_model', default=True, help='Load the latest pretrained model from the ssl_rl_pretraining.py script.')


def main(dataset_name, subject_size, random_state, n_jobs, window_size_s, window_size_samples, high_cut_hz, low_cut_hz, sfreq, emb_size, lr, batch_size, n_epochs, n_channels, input_size_samples, edge_bundling_plot, annotations, show_plots, load_feature_vectors, load_latest_model):
    print('STARTING MAIN')
    # set device to 'cuda' or 'cpu'
    device = hf.enable_cuda()
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

    # load the pretrained model
    # (load the best model)
    if load_latest_model:
        model_dir = "./models/pretrained/"
        files = [os.path.join(model_dir, fname) for fname in os.listdir(model_dir)]
        latest = max(files, key=os.path.getmtime).split(model_dir)[1].split('.')[0]
        print(f":: loading the latest pretrained model: {latest}")
        model = torch.load(f"{model_dir}{latest}.model")
    else:
        model = torch.load("models/pretrained/2021_12_15__01_34_22_sleep_staging_5s_windows_5_subjects_cpu_12_epochs_160hz.model")

    # compare_models(model.emb, emb)


    # DOWNSTREAM TASK - FINE TUNING)
    if load_feature_vectors is None:
        # data, descriptions = load_bci_data(subject_size)
        # data, descriptions = load_sleep_staging_raws()
        data, descriptions = load_space_bambi_raws()
        data = preprocess_raws(data, sfreq, high_cut_hz, n_jobs)
        windows_dataset = create_windows_dataset(data, window_size_samples, descriptions)

        ### Fine tune on Sleep staging SSL model

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


        ### trying w/o sequential layer
        # model.emb.return_feats = True

        batch_size = 512
        num_workers = n_jobs

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

        metadata_string = f'{dataset_name}_{window_size_s}s_windows_{len(subjects)}_subjects_{device}_{sfreq}hz'

        X = np.concatenate([v[0] for k, v in data.items()])
        y = np.concatenate([v[1] for k, v in data.items()])

        ### save fine-tuned model
        with open(f'models/finetuned/{hf.get_datetime()}_{metadata_string}.pkl', 'wb+') as f:
            pickle.dump(clf_pipe, f)
        f.close()

        ### save feature vectors
        with open(f'data/feature_vectors/{hf.get_datetime()}_{metadata_string}.pkl', 'wb+') as f:
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

    # init plotting object
    p = Plot(metadata_string, show=show_plots)

    p.plot_UMAP(X, y, annotations)
    p.plot_UMAP_connectivity(X)
    if edge_bundling_plot:
        p.plot_UMAP_connectivity(X, edge_bundling=True)
    p.plot_UMAP_3d(X, y)


# load BCI data
def load_bci_data(subject_size):
    print('LOADING BCI DATA')

    ''' ANNOTATIONS
    T0 corresponds to rest
    T1 corresponds to onset of motion (real or imagined) of
        the left fist (in runs 3, 4, 7, 8, 11, and 12)
        both fists (in runs 5, 6, 9, 10, 13, and 14)
    T2 corresponds to onset of motion (real or imagined) of
        the right fist (in runs 3, 4, 7, 8, 11, and 12)
        both feet (in runs 5, 6, 9, 10, 13, and 14)
    '''
    print(subject_size)
    print(type(subject_size))
    subjects = range(subject_size[0], subject_size[1]) # max 110
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

    return eegmmidb, descriptions


def load_sleep_staging_raws():
    print('LOADING SLEEP STAGING RAWS')
    raw_set = [
        '/home/maligan/mne_data/physionet-sleep-data/SC4012E0-PSG.edf'
        '/home/maligan/mne_data/physionet-sleep-data/SC4451F0-PSG.edf'
        '/home/maligan/mne_data/physionet-sleep-data/SC4441E0-PSG.edf'
        '/home/maligan/mne_data/physionet-sleep-data/SC4431E0-PSG.edf'
        '/home/maligan/mne_data/physionet-sleep-data/SC4421E0-PSG.edf'
    ]

    # load into raw array
    raws = [mne.io.read_raw_edf(x) for x in raw_set]
    # mne.io.RawArray(raws)

    return raws, None


def load_space_bambi_raws():
    print('LOADING SPACE/BAMBI DATA')

    # space_bambi directory
    data_dir = './data/SPACE_BAMBI_2channels/'

    raws = []

    print(f'{len(os.listdir(data_dir))} files found')
    for i, path in enumerate(os.listdir(data_dir)):
        if i == 5:
            break
        full_path = os.path.join(data_dir, path)
        raws.append(mne.io.read_raw_fif(full_path, preload=True))

    return raws, None


def preprocess_raws(raws, sfreq, high_cut_hz, n_jobs):
    print('PREPROCESSING RAWS')
    print(f'--resample {sfreq}')
    print(f'--highcut freq {high_cut_hz}')

    for raw in raws:
        mne.io.Raw.resample(raw, sfreq)   # resample
        mne.io.Raw.filter(raw, l_freq=None, h_freq=high_cut_hz, n_jobs=n_jobs)    # high-pass filter

    return raws


def create_windows_dataset(raws, window_size_samples, descriptions=None):
    print(f'Creating windows of size: {window_size_samples}')

    windows_dataset = create_from_mne_raw(
        raws,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        drop_last_window=True,
        descriptions=descriptions,
        accepted_bads_ratio=0.5,
        drop_bad_windows=True,
        on_missing='ignore',
        # mapping=mapping,
        # preload=True
    )

    # channel-wise zscore normalization
    preprocess(windows_dataset, [Preprocessor(zscore)])



if __name__ == '__main__':
    main()
