import os
from pathlib import Path
import pickle

import scipy.signal
from scipy.stats import zscore
from pandas.io.formats.style import color
from scipy.interpolate import interp1d
from scipy.stats import f_oneway, kruskal
from tqdm import tqdm

import copy

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

import mne
from mne.datasets import sample
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test
from mne.viz import Brain

from maestro_analysis.utils import epoching_maestro, select_seeg_channels_from_parcellation, select_parcel
from maestro_analysis.stats import unpaired_t_test

from scipy.stats import ttest_ind, ttest_1samp, pearsonr, spearmanr
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

from pactools import Comodulogram

from sklearn.linear_model import RidgeCV, Ridge, RidgeClassifierCV, LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import seaborn as sns

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def save_comodulograms_meg(data_path, wav_folder, save_path, indiv_trial=False, condition=None, channel_type='mag', n_jobs=-1):
    """Load meg data and saves a pickle containing a similarity matrix for every subject (correlation between every stimulus pair)"""

    os.makedirs(save_path, exist_ok=True)

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    del subject_folders[5]  # no behavior data

    subject_list = []
    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        epochs = epoching_maestro(raw_path, wav_folder, allow_maxshield=False, tmin=-0.5, baseline=True, bin_ecg=False, channel_type=channel_type, remove_responses=True)

        # preprocessing
        fs = 200
        epochs.resample(fs, n_jobs=n_jobs)
        epochs.filter(0.5, None, n_jobs=n_jobs)  # highpass filter
        epochs.crop(tmin=0, tmax=8)  # remove onset and offset
        data = epochs.get_data()

        subject_list.append(subject_path.name)
        stim_name = epochs.metadata['stim_name'].to_list()
        info = epochs.info
        del epochs

        # select epochs
        if condition is not None:
            int_stim_name = [int(name[5:8]) for name in stim_name]
            subj_condition = condition[subject_path.name]
            trials_to_keep = [int_stim_name.index(trial) for trial in int_stim_name if trial in subj_condition]  # keep only trials that were kept in MEG
            data = data[trials_to_keep, :, :]

        # compute comodulogram matrix for each channel
        num_low_freq = 100
        num_high_freq = 90
        low_fq_range = np.linspace(0.5, 10, 100)
        high_fq_range = np.linspace(10, fs / 2.0, 90)
        low_fq_width = 1.0  # Hz
        method = "duprelatour"
        if indiv_trial:
            pac_array = np.zeros((data.shape[0], data.shape[1], num_low_freq, num_high_freq))
            for e in tqdm(range(data.shape[0])):
                def individual_channel_comodulogram(chan):
                    """multiprocessing"""
                    estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                                             low_fq_width=low_fq_width, high_fq_range=high_fq_range, method=method,
                                             progress_bar=False)  # default high freq is np.linspace(max(self.low_fq_range), self.fs / 2.0, 80)
                    estimator.fit(data[e, chan, :])
                    return estimator.comod_

                # multiprocessing chan
                chan_comodulograms = Parallel(n_jobs=-1)(delayed(individual_channel_comodulogram)(c) for c in range(data.shape[1]))
                for c in range(data.shape[1]):
                    pac_array[e, c, :, :] = chan_comodulograms[c]

        else:
            pac_array = np.zeros((data.shape[1], num_low_freq, num_high_freq))
            def individual_channel_comodulogram(chan):
                """multiprocessing"""
                estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                                         low_fq_width=low_fq_width, high_fq_range=high_fq_range, method=method,
                                         progress_bar=False)  # default high freq is np.linspace(max(self.low_fq_range), self.fs / 2.0, 80)
                estimator.fit(data[:, chan, :])
                return estimator.comod_

            # multiprocessing chan
            chan_comodulograms = Parallel(n_jobs=-1)(delayed(individual_channel_comodulogram)(c) for c in range(data.shape[1]))
            for c in range(data.shape[1]):
                pac_array[c, :, :] = chan_comodulograms[c]

        pac_dict = {'info': info, 'stim_name': stim_name, 'comodulogram': pac_array}

        with open(os.path.join(save_path, subject_path.name + "_comodulogram.pkl"), 'wb') as f:
            pickle.dump(pac_dict, f)

    print(f'Done saving in {save_path}')

    return


def save_comodulograms_seeg(epochs_path, save_path, channels=None, fs_subjects_dir=None, indiv_trial=False, epochs_to_keep=None, n_jobs=-1):
    """Load seeg data and save a pickle containing comodulograms"""

    os.makedirs(save_path, exist_ok=True)

    subject_name = Path(epochs_path).name[:6]
    print(subject_name)

    # select epochs
    epochs = mne.read_epochs(epochs_path)
    if epochs_to_keep is not None:
        epochs = epochs[epochs.metadata['stim_name'].isin(epochs_to_keep)]
    # epochs = epochs[~epochs.metadata['stim_name'].str.contains('7.5')]  # TODO: remove
    # epochs = epochs[~epochs.metadata['stim_name'].str.contains('4.5')]  # TODO: remove

    if channels is not None:
        assert fs_subjects_dir is not None
        epochs, channel_parcellations = select_seeg_channels_from_parcellation(epochs, channels, fs_subjects_dir)
    else:
        channel_parcellations = None
    channels_to_keep = epochs.ch_names

    epochs = epochs.apply_baseline((None, -0.1))
    fs = 200
    epochs.resample(fs, n_jobs=n_jobs)
    epochs.filter(0.5, None, n_jobs=-1)
    epochs.crop(tmin=0, tmax=8)  # remove onset and offset
    data = epochs.get_data()

    stim_name = epochs.metadata['stim_name'].to_list()
    info = epochs.info
    del epochs

    # compute comodulogram matrix for each channel
    num_low_freq = 100
    num_high_freq = 90
    low_fq_range = np.linspace(0.5, 10, 100)
    high_fq_range = np.linspace(10, fs / 2.0, 90)
    low_fq_width = 1.0  # Hz
    method = "duprelatour"
    if indiv_trial:
        pac_array = np.zeros((data.shape[0], data.shape[1], num_low_freq, num_high_freq))
        for e in tqdm(range(data.shape[0])):
            def individual_channel_comodulogram(chan):
                """multiprocessing"""
                estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                                         low_fq_width=low_fq_width, high_fq_range=high_fq_range, method=method,
                                         progress_bar=False)  # default high freq is np.linspace(max(self.low_fq_range), self.fs / 2.0, 80)
                estimator.fit(data[e, chan, :])
                return estimator.comod_

            # multiprocessing chan
            chan_comodulograms = Parallel(n_jobs=-1)(delayed(individual_channel_comodulogram)(c) for c in range(data.shape[1]))
            for c in range(data.shape[1]):
                pac_array[e, c, :, :] = chan_comodulograms[c]

    else:
        pac_array = np.zeros((data.shape[1], num_low_freq, num_high_freq))
        def individual_channel_comodulogram(chan):
            """multiprocessing"""
            estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                                     low_fq_width=low_fq_width, high_fq_range=high_fq_range, method=method,
                                     progress_bar=False)  # default high freq is np.linspace(max(self.low_fq_range), self.fs / 2.0, 80)
            estimator.fit(data[:, chan, :])
            return estimator.comod_

        # multiprocessing chan
        chan_comodulograms = Parallel(n_jobs=-1)(delayed(individual_channel_comodulogram)(c) for c in range(data.shape[1]))
        for c in range(data.shape[1]):
            pac_array[c, :, :] = chan_comodulograms[c]

    pac_dict = {'info': info, 'stim_name': stim_name, 'comodulogram': pac_array, 'ch_names': channels_to_keep, 'ch_parcellations': channel_parcellations}

    with open(os.path.join(save_path, subject_name + "_comodulogram.pkl"), 'wb') as f:
        pickle.dump(pac_dict, f)

    print(f'Done saving in {save_path}')

    return


def save_comodulograms_seeg_all_subjects(data_path, save_path, channels=None, fs_subjects_dir=None):

    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    for i, epochs_path in tqdm(enumerate(subject_list)):

        save_comodulograms_seeg(epochs_path, save_path, channels, fs_subjects_dir, indiv_trial=False, epochs_to_keep=None, n_jobs=-1)

    print('Done')

    return


def comodulograms_correct_incorrect(correct_incorrect_path):
    correct_incorrect = np.load(correct_incorrect_path, allow_pickle=True)
    correct = {}
    incorrect = {}
    for subj in correct_incorrect:
        correct[subj] = correct_incorrect[subj][0]
        incorrect[subj] = correct_incorrect[subj][1]

    save_comodulograms_meg('/stockage/data/mathieu/meg_maestro/preprocessed',
                           '/home/mathieu/Data/meg_maestro/exp_stim_meg',
                           '/home/mathieu/Data/meg_maestro/pac_analysis/correct',
                           condition=correct,
                           indiv_trial=False,
                           channel_type='mag',
                           n_jobs=-1)

    save_comodulograms_meg('/stockage/data/mathieu/meg_maestro/preprocessed',
                           '/home/mathieu/Data/meg_maestro/exp_stim_meg',
                           '/home/mathieu/Data/meg_maestro/pac_analysis/incorrect',
                           condition=incorrect,
                           indiv_trial=False,
                           channel_type='mag',
                           n_jobs=-1)
    print('Done')
    return


def comodulograms_correct_incorrect_seeg(data_path, behavior_path, save_path, channels=None, fs_subjects_dir=None):

    behavior_all = pd.read_csv(behavior_path)

    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    for i, epochs_path in tqdm(enumerate(subject_list)):

        subject_name = Path(epochs_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct = behavior[behavior['correct'] == 1]['stim_name'].tolist()
        incorrect = behavior[behavior['correct'] == 0]['stim_name'].tolist()

        save_comodulograms_seeg(epochs_path, os.path.join(save_path, 'correct'), channels, fs_subjects_dir, indiv_trial=False, epochs_to_keep=correct, n_jobs=-1)

        save_comodulograms_seeg(epochs_path, os.path.join(save_path, 'incorrect'), channels, fs_subjects_dir, indiv_trial=False, epochs_to_keep=incorrect, n_jobs=-1)

    print('Done')

    return


def comodulograms_correct_incorrect_encoding_seeg(data_path, behavior_path, save_path, channels=None, fs_subjects_dir=None):

    behavior_all = pd.read_csv(behavior_path)

    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    for i, epochs_path in tqdm(enumerate(subject_list)):

        subject_name = Path(epochs_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()

        save_comodulograms_seeg(epochs_path, os.path.join(save_path, 'encoding', 'correct'), channels, fs_subjects_dir, indiv_trial=False, epochs_to_keep=correct, n_jobs=-1)

        save_comodulograms_seeg(epochs_path, os.path.join(save_path, 'encoding', 'incorrect'), channels, fs_subjects_dir, indiv_trial=False, epochs_to_keep=incorrect, n_jobs=-1)

    print('Done')

    return


def comodulograms_correct_incorrect_retrieval_seeg(data_path, behavior_path, save_path, channels=None, fs_subjects_dir=None):

    behavior_all = pd.read_csv(behavior_path)

    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    for i, epochs_path in tqdm(enumerate(subject_list)):

        subject_name = Path(epochs_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_name'].tolist()

        save_comodulograms_seeg(epochs_path, os.path.join(save_path, 'retrieval', 'correct'), channels, fs_subjects_dir, indiv_trial=False, epochs_to_keep=correct, n_jobs=-1)

        save_comodulograms_seeg(epochs_path, os.path.join(save_path, 'retrieval', 'incorrect'), channels, fs_subjects_dir, indiv_trial=False, epochs_to_keep=incorrect, n_jobs=-1)

    print('Done')

    return


def comodulograms_correct_incorrect_novelty_seeg(data_path, behavior_path, save_path, channels=None, fs_subjects_dir=None):

    behavior_all = pd.read_csv(behavior_path)

    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    for i, epochs_path in tqdm(enumerate(subject_list)):

        subject_name = Path(epochs_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct = behavior[(behavior['rep1'] == 1) & (behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect = behavior[(behavior['rep1'] == 1) & (behavior['correct'] == 0)]['stim_name'].tolist()
        if (len(correct) > 0) & (len(incorrect) > 0):
            save_comodulograms_seeg(epochs_path, os.path.join(save_path, 'novelty_detection', 'correct'), channels, fs_subjects_dir, indiv_trial=False, epochs_to_keep=correct, n_jobs=-1)
            save_comodulograms_seeg(epochs_path, os.path.join(save_path, 'novelty_detection', 'incorrect'), channels, fs_subjects_dir, indiv_trial=False, epochs_to_keep=incorrect, n_jobs=-1)

    print('Done')

    return


def plot_comodulogram_meg(data_path, best_channels_path=None):

    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*comodulogram.pkl')))

    avg_como_list = []
    for i, subject_path in enumerate(tqdm(subject_paths)):
        como_obj = np.load(subject_path, allow_pickle=True)
        comodulogram = como_obj['comodulogram']
        # avg_como_list.append(np.mean(comodulogram, axis=0))
        avg_como_list.append(comodulogram)

    avg_como = np.array(avg_como_list)
    print(avg_como.shape)

    if best_channels_path is not None:
        best_channels = np.load(best_channels_path)
        avg_como = np.mean(np.take_along_axis(avg_como, best_channels[:, :, np.newaxis, np.newaxis], axis=1), axis=1)
        print(avg_como.shape)

    # plt.imshow(np.mean(avg_como, axis=(0, 1)).T)
    # plt.show()

    fs = 200
    low_fq_range = np.linspace(0.5, 10, 100)
    print(low_fq_range)
    high_fq_range = np.linspace(10, fs / 2.0, 90)
    low_fq_width = 1.0  # Hz

    # remove edge effects
    low_fq_range = low_fq_range[16:-11]
    avg_como = avg_como[:, 16:-11, :]
    # avg_como = avg_como[:, :, 16:-11, :]

    method = "duprelatour"
    # method = "ozkurt"
    estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                             low_fq_width=low_fq_width, high_fq_range=high_fq_range, method=method,
                             progress_bar=False)  # default high freq is np.linspace(max(self.low_fq_range), self.fs / 2.0, 80)
    if best_channels_path is not None:
        estimator.comod_ = np.mean(avg_como[:, :, :], axis=0)
    else:
        estimator.comod_ = np.mean(avg_como, axis=(0, 1))
    fig, ax = plt.subplots(1, 1)
    estimator.plot(axs=[ax])
    plt.show()

    # for c in range(avg_como.shape[1]):
    #     estimator.comod_ = np.mean(avg_como[:, c, :, :], axis=0)
    #     fig, ax = plt.subplots(1, 1)
    #     estimator.plot(axs=[ax])
    #     plt.show()
    # for s in range(avg_como.shape[0]):
    #     # estimator.comod_ = np.mean(avg_como[s, :, :, :], axis=0)
    #     estimator.comod_ = avg_como[s, :, :]
    #     fig, ax = plt.subplots(1, 1)
    #     estimator.plot(axs=[ax])
    #     plt.show()

    return


def plot_comodulogram_seeg(como_path, channels=None):

    como_obj = np.load(como_path, allow_pickle=True)
    comodulogram = como_obj['comodulogram']

    print(comodulogram.shape)

    if channels is not None:
        chan_ind = []
        for ch in como_obj['ch_names']:
            for shaft_letter in channels:
                if ch.startswith(shaft_letter):
                    chan_ind.append(como_obj['ch_names'].index(ch))
        comodulogram = comodulogram[chan_ind, :, :]

    # comodulogram = np.mean(comodulogram, axis=0)

    # plt.imshow(np.mean(avg_como, axis=(0, 1)).T)
    # plt.show()

    fs = 200
    low_fq_range = np.linspace(0.5, 10, 100)
    print(low_fq_range)
    high_fq_range = np.linspace(10, fs / 2.0, 90)
    low_fq_width = 1.0  # Hz

    # remove edge effects
    low_fq_range = low_fq_range[16:-11]
    comodulogram = comodulogram[:, 16:-11, :]

    method = "duprelatour"
    estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                             low_fq_width=low_fq_width, high_fq_range=high_fq_range, method=method,
                             progress_bar=False)  # default high freq is np.linspace(max(self.low_fq_range), self.fs / 2.0, 80)
    estimator.comod_ = np.mean(comodulogram, axis=0)
    fig, ax = plt.subplots(1, 1)
    estimator.plot(axs=[ax])
    plt.show()

    vmax = np.max(comodulogram)
    for c in range(comodulogram.shape[0]):
        estimator.comod_ = comodulogram[c, :, :]
        fig, ax = plt.subplots(1, 1)
        estimator.plot(axs=[ax], vmin=0, vmax=vmax)
        plt.show()

    return


def plot_comodulogram_seeg_all_subjects(data_path, channels=None):

    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*comodulogram.pkl')))

    como_list = []
    for i, subject_path in enumerate(tqdm(subject_paths)):
        print(subject_path)
        como_obj = np.load(subject_path, allow_pickle=True)
        comodulogram = como_obj['comodulogram']
        if channels is not None:
            chan_ind = []
            for p, parcel in enumerate(como_obj['ch_parcellations']):
                if any(ch_parcel in channels for ch_parcel in parcel):
                    chan_ind.append(p)
            comodulogram = comodulogram[chan_ind, :, :]
            print(np.array(como_obj['ch_names'])[chan_ind])
        if len(chan_ind) > 0:
            como_list.append(np.mean(comodulogram, axis=0))

    comodulogram = np.array(como_list)
    print(comodulogram.shape)

    fs = 200
    low_fq_range = np.linspace(0.5, 10, 100)
    high_fq_range = np.linspace(10, fs / 2.0, 90)
    low_fq_width = 1.0  # Hz

    # remove edge effects
    low_fq_range = low_fq_range[16:-11]
    comodulogram = comodulogram[:, 16:-11, :]

    method = "duprelatour"
    estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                             low_fq_width=low_fq_width, high_fq_range=high_fq_range, method=method,
                             progress_bar=False)  # default high freq is np.linspace(max(self.low_fq_range), self.fs / 2.0, 80)
    estimator.comod_ = np.mean(comodulogram, axis=0) / (np.std(comodulogram, axis=0) / np.sqrt(comodulogram.shape[0]))  # t-values
    fig, ax = plt.subplots(1, 1)
    estimator.plot(axs=[ax])
    ax.set_title(str(channels) + f' t-values (n = {comodulogram.shape[0]})')
    plt.show()

    vmax = np.max(np.abs(comodulogram))

    c = 0
    while c < comodulogram.shape[0]:
        if c % 25 == 0:
            plt.show()
            fig, axes = plt.subplots(5, 5)
        comodulogram[c, :, :]
        axes.flat[c % 25].imshow(comodulogram[c, :, :].T, origin='lower', vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        c += 1
    plt.show()

    return


def plot_contrast_comodulogram_meg(data_path, data_path2, best_channels_path=None):

    path = Path(data_path)
    path2 = Path(data_path2)
    subject_paths = sorted(list(path.glob('*comodulogram.pkl')))

    avg_como_list = []
    for i, subject_path in enumerate(tqdm(subject_paths)):
        como_obj = np.load(subject_path, allow_pickle=True)
        como_obj2 = np.load(path2 / subject_path.name, allow_pickle=True)
        comodulogram = como_obj['comodulogram'] - como_obj2['comodulogram']
        # avg_como_list.append(np.mean(comodulogram, axis=0))
        avg_como_list.append(comodulogram)

    avg_como = np.array(avg_como_list)
    print(avg_como.shape)

    if best_channels_path is not None:
        best_channels = np.load(best_channels_path)
        print(best_channels.shape)
        mask = np.ones((best_channels.shape[0]), dtype=bool)
        mask[5] = False  # remove subject 5
        best_channels = best_channels[mask, :]
        print(best_channels.shape)
        # print(best_channels)
        avg_como = np.mean(np.take_along_axis(avg_como, best_channels[:, :, np.newaxis, np.newaxis], axis=1), axis=1)
        print(avg_como.shape)

    # plt.imshow(np.mean(avg_como, axis=(0, 1)).T)
    # plt.show()

    fs = 200
    low_fq_range = np.linspace(0.5, 10, 100)
    print(low_fq_range)
    high_fq_range = np.linspace(10, fs / 2.0, 90)
    low_fq_width = 1.0  # Hz

    # remove edge effects
    # low_fq_range = low_fq_range[16:-11]
    # avg_como = avg_como[:, 16:-11, :]
    # avg_como = avg_como[:, :, 16:-11, :]

    method = "duprelatour"
    # method = "ozkurt"
    estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                             low_fq_width=low_fq_width, high_fq_range=high_fq_range, method=method,
                             progress_bar=False)  # default high freq is np.linspace(max(self.low_fq_range), self.fs / 2.0, 80)
    if best_channels_path is not None:
        estimator.comod_ = np.mean(avg_como[:, :, :], axis=0)
    else:
        estimator.comod_ = np.mean(avg_como, axis=(0, 1))
    fig, ax = plt.subplots(1, 1)
    estimator.plot(axs=[ax])
    plt.show()

    # for c in range(avg_como.shape[1]):
    #     estimator.comod_ = np.mean(avg_como[:, c, :, :], axis=0)
    #     fig, ax = plt.subplots(1, 1)
    #     estimator.plot(axs=[ax])
    #     plt.show()
    # for s in range(avg_como.shape[0]):
    #     # estimator.comod_ = np.mean(avg_como[s, :, :, :], axis=0)
    #     estimator.comod_ = avg_como[s, :, :]
    #     fig, ax = plt.subplots(1, 1)
    #     estimator.plot(axs=[ax])
    #     plt.show()

    return


def plot_contrast_comodulogram_seeg(como_path, como_path2, channels=None):

    como_obj = np.load(como_path, allow_pickle=True)
    comodulogram = como_obj['comodulogram']
    como_obj2 = np.load(como_path2, allow_pickle=True)
    comodulogram2 = como_obj2['comodulogram']

    print(comodulogram.shape)
    assert comodulogram.shape == comodulogram2.shape

    comodulogram = comodulogram - comodulogram2

    if channels is not None:
        chan_ind = []
        for ch in como_obj['ch_names']:
            for shaft_letter in channels:
                if ch.startswith(shaft_letter):
                    chan_ind.append(como_obj['ch_names'].index(ch))
        comodulogram = comodulogram[chan_ind, :, :]

    # comodulogram = np.mean(comodulogram, axis=0)

    # plt.imshow(np.mean(avg_como, axis=(0, 1)).T)
    # plt.show()

    fs = 200
    low_fq_range = np.linspace(0.5, 10, 100)
    print(low_fq_range)
    high_fq_range = np.linspace(10, fs / 2.0, 90)
    low_fq_width = 1.0  # Hz

    # remove edge effects
    low_fq_range = low_fq_range[16:-11]
    comodulogram = comodulogram[:, 16:-11, :]

    method = "duprelatour"
    estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                             low_fq_width=low_fq_width, high_fq_range=high_fq_range, method=method,
                             progress_bar=False)  # default high freq is np.linspace(max(self.low_fq_range), self.fs / 2.0, 80)
    estimator.comod_ = np.mean(comodulogram, axis=0)
    fig, ax = plt.subplots(1, 1)
    estimator.plot(axs=[ax])
    plt.show()

    vmax = np.max(np.abs(comodulogram))
    for c in range(comodulogram.shape[0]):
        estimator.comod_ = comodulogram[c, :, :]
        fig, ax = plt.subplots(1, 1)
        # estimator.plot(axs=[ax])
        estimator.plot(axs=[ax], vmin=-vmax, vmax=vmax, cmap='RdBu_r')
        plt.show()

    return


def plot_contrast_comodulogram_seeg_all_subjects(data_path, data_path2, channels=None):

    path = Path(data_path)
    path2 = Path(data_path2)
    subject_paths = sorted(list(path.glob('*comodulogram.pkl')))

    como_list = []
    for i, subject_path in enumerate(tqdm(subject_paths)):
        como_obj = np.load(subject_path, allow_pickle=True)
        como_obj2 = np.load(path2 / subject_path.name, allow_pickle=True)
        assert como_obj['comodulogram'].shape == como_obj2['comodulogram'].shape
        comodulogram = como_obj['comodulogram'] - como_obj2['comodulogram']
        print(comodulogram.shape)
        if channels is not None:
            chan_ind = []
            for p, parcel in enumerate(como_obj['ch_parcellations']):
                if any(ch_parcel in channels for ch_parcel in parcel):
                    chan_ind.append(p)
            comodulogram = comodulogram[chan_ind, :, :]
            print(np.array(como_obj['ch_names'])[chan_ind])
        if len(chan_ind) > 0:
            como_list.append(np.mean(comodulogram, axis=0))

    comodulogram = np.array(como_list)
    print(comodulogram.shape)

    fs = 200
    low_fq_range = np.linspace(0.5, 10, 100)
    high_fq_range = np.linspace(10, fs / 2.0, 90)
    low_fq_width = 1.0  # Hz

    # remove edge effects
    low_fq_range = low_fq_range[16:-11]
    comodulogram = comodulogram[:, 16:-11, :]

    # stat test
    pval = 0.05  # arbitrary
    df = comodulogram.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    thresh = scipy.stats.t.ppf(1 - pval / 2, df=df)  # T distribution, two-tailed
    thresh = dict(start=0, step=0.01)
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(comodulogram,
                                                                     threshold=thresh,
                                                                     n_permutations=1000, tail=0,
                                                                     adjacency=None, n_jobs=-1)
    # Create a mask for significant clusters
    significant_mask = np.zeros_like(t_obs, dtype=bool)
    print(sorted(cluster_pv))
    for i, c in enumerate(clusters):
        if cluster_pv[i] <= 0.05:
            significant_mask[c] = True

    method = "duprelatour"
    estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                             low_fq_width=low_fq_width, high_fq_range=high_fq_range, method=method,
                             progress_bar=False)  # default high freq is np.linspace(max(self.low_fq_range), self.fs / 2.0, 80)
    estimator.comod_ = np.mean(comodulogram, axis=0) / (np.std(comodulogram, axis=0) / np.sqrt(comodulogram.shape[0]))  # t-values
    # estimator.comod_[np.abs(estimator.comod_) < 1.96] = 0
    vmax = np.max(np.abs(estimator.comod_))
    fig, ax = plt.subplots(1, 1)
    # estimator.plot(axs=[ax], vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    estimator.plot(axs=[ax], vmin=-5, vmax=5, cmap='RdBu_r')
    ax.set_title(str(channels) + f' correct - incorrect t-values (n = {comodulogram.shape[0]})')

    X, Y = np.meshgrid(low_fq_range, high_fq_range)
    ax.contour(X, Y, significant_mask.T, colors='red', linewidths=1)

    plt.show()

    # vmax = np.max(np.abs(comodulogram))
    # c = 0
    # while c < comodulogram.shape[0]:
    #     if c % 25 == 0:
    #         plt.show()
    #         fig, axes = plt.subplots(5, 5)
    #     comodulogram[c, :, :]
    #     axes.flat[c % 25].imshow(comodulogram[c, :, :].T, origin='lower', vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    #     c += 1
    # plt.show()

    return


def pac_stim_freq_seeg(data_path, save_path, channels=None, fs_subjects_dir=None):
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        # select epochs
        epochs = mne.read_epochs(epochs_path)

        if channels is not None:
            assert fs_subjects_dir is not None
            epochs, channel_parcellations = select_seeg_channels_from_parcellation(epochs, channels, fs_subjects_dir)
        else:
            channel_parcellations = None
        channels_to_keep = epochs.ch_names

        epochs = epochs.apply_baseline((None, -0.1))
        fs = 200
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        epochs.crop(tmin=0, tmax=8)  # remove onset and offset
        data = epochs.get_data()

        stim_name = epochs.metadata['stim_name'].to_list()
        info = epochs.info

        # compute comodulogram matrix for each channel
        num_high_freq = 81
        high_fq_range = np.linspace(10, 90, num_high_freq, endpoint=True)

        pac_array = np.zeros((data.shape[0], data.shape[1], num_high_freq))
        for e in tqdm(range(data.shape[0])):
            stim_freq = float(stim_name[e][14:17])
            low_f_signal = epochs[e].copy().filter(stim_freq - stim_freq/10, stim_freq + stim_freq/10, n_jobs=-1, verbose=0)  # bandwidth of 1/5 an octave
            phase = np.angle(scipy.signal.hilbert(low_f_signal.get_data()[0, :, :], axis=-1))
            phase = np.exp(1j * phase)
            def individual_freq_pac(freq):
                """multiprocessing"""
                high_f_signal = epochs[e].copy().filter(freq - freq/10, freq + freq/10, n_jobs=-1, verbose=0)  # bandwidth of 1/5 an octave
                amplitude = np.abs(scipy.signal.hilbert(high_f_signal.get_data()[0, :, :], axis=-1))
                pac = np.abs(np.mean(amplitude * phase, axis=-1)) / np.sqrt(np.mean(amplitude**2, axis=-1))
                return pac

            # multiprocessing chan
            freq_pac = Parallel(n_jobs=-1)(delayed(individual_freq_pac)(f) for f in high_fq_range)
            for f in range(num_high_freq):
                pac_array[e, :, f] = freq_pac[f]

        pac_dict = {'info': info, 'stim_name': stim_name, 'freqs': high_fq_range, 'pac': pac_array, 'ch_names': channels_to_keep, 'ch_parcellations': channel_parcellations}

        with open(os.path.join(save_path, subject_name + "_stim_freq_pac.pkl"), 'wb') as f:
            pickle.dump(pac_dict, f)

    print('Done')

    return


def pac_fixed_freq_seeg(data_path, save_path, channels=None, fs_subjects_dir=None, low_freq=4):
    """Same as pac_stim_freq_seeg but the driving frequency (low freq) is fixed"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        # select epochs
        epochs = mne.read_epochs(epochs_path)

        if channels is not None:
            assert fs_subjects_dir is not None
            epochs, channel_parcellations = select_seeg_channels_from_parcellation(epochs, channels, fs_subjects_dir)
        else:
            channel_parcellations = None
        channels_to_keep = epochs.ch_names

        epochs = epochs.apply_baseline((None, -0.1))
        fs = 200
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        epochs.crop(tmin=0, tmax=8)  # remove onset and offset
        data = epochs.get_data()

        stim_name = epochs.metadata['stim_name'].to_list()
        info = epochs.info

        # compute comodulogram matrix for each channel
        num_high_freq = 81
        high_fq_range = np.linspace(10, 90, num_high_freq, endpoint=True)

        pac_array = np.zeros((data.shape[0], data.shape[1], num_high_freq))
        for e in tqdm(range(data.shape[0])):
            # low_f_signal = epochs[e].copy().filter(low_freq - low_freq/10, low_freq + low_freq/10, n_jobs=-1, verbose=0)  # bandwidth of 1/5 an octave
            low_f_signal = epochs[e].copy().filter(low_freq - 0.5, low_freq + 0.5, n_jobs=-1, verbose=0)  # bandwidth of 1Hz
            phase = np.angle(scipy.signal.hilbert(low_f_signal.get_data()[0, :, :], axis=-1))
            phase = np.exp(1j * phase)
            def individual_freq_pac(freq):
                """multiprocessing"""
                # high_f_signal = epochs[e].copy().filter(freq - freq/10, freq + freq/10, n_jobs=-1, verbose=0)  # bandwidth of 1/5 an octave
                high_f_signal = epochs[e].copy().filter(freq - 0.5, freq + 0.5, n_jobs=-1, verbose=0)  # bandwidth of 1Hz
                amplitude = np.abs(scipy.signal.hilbert(high_f_signal.get_data()[0, :, :], axis=-1))
                pac = np.abs(np.mean(amplitude * phase, axis=-1)) / np.sqrt(np.mean(amplitude**2, axis=-1))
                return pac

            # multiprocessing chan
            freq_pac = Parallel(n_jobs=-1)(delayed(individual_freq_pac)(f) for f in high_fq_range)
            for f in range(num_high_freq):
                pac_array[e, :, f] = freq_pac[f]

        pac_dict = {'info': info, 'stim_name': stim_name, 'freqs': high_fq_range, 'pac': pac_array, 'ch_names': channels_to_keep, 'ch_parcellations': channel_parcellations}

        with open(os.path.join(save_path, subject_name + f"_{low_freq}Hz_pac.pkl"), 'wb') as f:
            pickle.dump(pac_dict, f)

    print('Done')

    return


def plot_contrast_pac_freq_seeg(data_path, behavior_path, channels=None, freq='stim'):
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    if freq == 'stim':
        subject_paths = sorted(list(path.glob('*stim_freq_pac.pkl')))
    elif isinstance(freq, int):
        subject_paths = sorted(list(path.glob(f'*{freq}Hz_pac.pkl')))
    else:
        raise ValueError('freq should be "stim" or int')

    average_pac_list = []
    rate_pac_list = [[] for r in range(6)]
    novelty_pac_list = []
    retrieval_pac_list = []
    encoding_pac_list = []
    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()

        pac_obj = np.load(subject_path, allow_pickle=True)
        pac = pac_obj['pac']
        print(pac.shape)
        if channels is not None:
            chan_ind = []
            for p, parcel in enumerate(pac_obj['ch_parcellations']):
                if any(ch_parcel in channels for ch_parcel in parcel):
                    chan_ind.append(p)
            pac = pac[:, chan_ind, :]
            print(np.array(pac_obj['ch_names'])[chan_ind])
        if len(chan_ind) > 0:
            average_pac_list.append(np.mean(pac, axis=(0, 1)))  # average epochs and channels
            for r, rate in enumerate(['2.5', '3.5', '4.5', '5.5', '6.5', '7.5']):
                rate_epochs = [e for e in range(len(pac_obj['stim_name'])) if rate in pac_obj['stim_name'][e]]
                rate_pac_list[r].append(np.mean(pac[rate_epochs, :, :], axis=(0, 1)))
            correct_novelty = [e for e in range(len(pac_obj['stim_name'])) if pac_obj['stim_name'][e] in correct_novelty]
            incorrect_novelty = [e for e in range(len(pac_obj['stim_name'])) if pac_obj['stim_name'][e] in incorrect_novelty]
            if (len(correct_novelty) > 0) & (len(incorrect_novelty) > 0):
                novelty_pac_list.append(np.mean(pac[correct_novelty, :, :], axis=(0, 1)) - np.mean(pac[incorrect_novelty, :, :], axis=(0, 1)))
            correct_retrieval = [e for e in range(len(pac_obj['stim_name'])) if pac_obj['stim_name'][e] in correct_retrieval]
            incorrect_retrieval = [e for e in range(len(pac_obj['stim_name'])) if pac_obj['stim_name'][e] in incorrect_retrieval]
            if (len(correct_retrieval) > 0) & (len(incorrect_retrieval) > 0):
                retrieval_pac_list.append(np.mean(pac[correct_retrieval, :, :], axis=(0, 1)) - np.mean(pac[incorrect_retrieval, :, :], axis=(0, 1)))
            correct_encoding = [e for e in range(len(pac_obj['stim_name'])) if pac_obj['stim_name'][e] in correct_encoding]
            incorrect_encoding = [e for e in range(len(pac_obj['stim_name'])) if pac_obj['stim_name'][e] in incorrect_encoding]
            if (len(correct_encoding) > 0) & (len(incorrect_encoding) > 0):
                encoding_pac_list.append(np.mean(pac[correct_encoding, :, :], axis=(0, 1)) - np.mean(pac[incorrect_encoding, :, :], axis=(0, 1)))

    freqs = pac_obj['freqs']
    average_pac_list = np.array(average_pac_list)
    rate_pac_list = [np.array(rate) for rate in rate_pac_list]
    novelty_pac_list = np.array(novelty_pac_list)
    retrieval_pac_list = np.array(retrieval_pac_list)
    encoding_pac_list = np.array(encoding_pac_list)
    print(average_pac_list.shape)

    fig, ax = plt.subplots(2, 1)

    # plot average PAC
    # ax[0].plot(freqs, np.mean(average_pac_list, axis=0) / (np.std(average_pac_list, axis=0) / np.sqrt(average_pac_list.shape[0])), label='Average')
    ax[0].plot(freqs, np.mean(average_pac_list, axis=0), label='Average')
    sem = np.std(average_pac_list, axis=0) / np.sqrt(average_pac_list.shape[0])
    ax[0].fill_between(freqs, np.mean(average_pac_list, axis=0) - sem, np.mean(average_pac_list, axis=0) + sem, alpha=0.6)
    for r, rate in enumerate(['2.5', '3.5', '4.5', '5.5', '6.5', '7.5']):
        # ax[0].plot(freqs, np.mean(rate_pac_list[r], axis=0) / (np.std(rate_pac_list[r], axis=0) / np.sqrt(rate_pac_list[r].shape[0])), label=f'{rate} Hz')
        ax[0].plot(freqs, np.mean(rate_pac_list[r], axis=0), label=f'{rate} Hz')
        sem = np.std(rate_pac_list[r], axis=0) / np.sqrt(rate_pac_list[r].shape[0])
        ax[0].fill_between(freqs, np.mean(rate_pac_list[r], axis=0) - sem, np.mean(rate_pac_list[r], axis=0) + sem, alpha=0.6)
    if freq == 'stim':
        ax[0].set_title('Average PAC at stimulus frequency')
    else:
        ax[0].set_title(f'Average PAC at {freq}Hz')
    # ax[0].set_ylabel('PAC (t-values)')
    ax[0].set_ylabel('PAC')
    ax[0].legend()

    # stat test
    pval = 0.05  # arbitrary
    significant_mask_list = []
    for pac_list in [novelty_pac_list, retrieval_pac_list, encoding_pac_list]:
        df = pac_list.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
        thresh = scipy.stats.t.ppf(1 - pval / 2, df=df)  # T distribution, two-tailed
        thresh = dict(start=0, step=0.01)
        t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(pac_list, threshold=thresh, n_permutations=1000, tail=0, adjacency=None, n_jobs=-1)
        # Create a mask for significant clusters
        significant_mask = np.zeros_like(t_obs, dtype=bool)
        print(cluster_pv)
        for i, c in enumerate(clusters):
            if cluster_pv[i] <= 0.05:
                significant_mask[c] = True
        significant_mask_list.append(significant_mask)

    ax[1].plot(freqs, np.mean(novelty_pac_list, axis=0) / (np.std(novelty_pac_list, axis=0) / np.sqrt(novelty_pac_list.shape[0])), lw=3, label='Novelty')
    ax[1].fill_between(freqs, np.mean(novelty_pac_list, axis=0) / (np.std(novelty_pac_list, axis=0) / np.sqrt(novelty_pac_list.shape[0])), where=significant_mask_list[0], alpha=0.5)
    ax[1].plot(freqs, np.mean(retrieval_pac_list, axis=0) / (np.std(retrieval_pac_list, axis=0) / np.sqrt(retrieval_pac_list.shape[0])), lw=3, label='Retrieval')
    ax[1].fill_between(freqs, np.mean(retrieval_pac_list, axis=0) / (np.std(retrieval_pac_list, axis=0) / np.sqrt(retrieval_pac_list.shape[0])), where=significant_mask_list[1], alpha=0.5)
    ax[1].plot(freqs, np.mean(encoding_pac_list, axis=0) / (np.std(encoding_pac_list, axis=0) / np.sqrt(encoding_pac_list.shape[0])), lw=3, label='Encoding')
    ax[1].fill_between(freqs, np.mean(encoding_pac_list, axis=0) / (np.std(encoding_pac_list, axis=0) / np.sqrt(encoding_pac_list.shape[0])), where=significant_mask_list[2], alpha=0.5)
    ax[1].axhline(0, ls='--', color='k')
    ax[1].legend()
    ax[1].set_xlabel('High frequency (Hz)')
    ax[1].set_ylabel('PAC correct - PAC incorrect (t-values)')
    if freq == 'stim':
        ax[1].set_title('PAC contrasts (correct - incorrect) at stimulus frequency')
    else:
        ax[1].set_title(f'PAC contrasts (correct - incorrect) at {freq}Hz')

    plt.show()

    return


def epochs_spectrum_seeg(data_path, save_path, channels=None, fs_subjects_dir=None):
    """Compute epoch-level spectrum for every subject, channel"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        epochs = mne.read_epochs(epochs_path)

        assert fs_subjects_dir is not None
        epochs, channel_parcellations = select_seeg_channels_from_parcellation(epochs, channels, fs_subjects_dir)
        if channel_parcellations is not None:
            channel_parcellations_list = channel_parcellations
            channels_to_keep = epochs.ch_names

            epochs = epochs.apply_baseline((None, -0.1))
            fs = 200
            epochs.resample(fs, n_jobs=-1)
            epochs.filter(0.5, None, n_jobs=-1)

            stim_name = epochs.metadata['stim_name'].to_list()

            spectrum = epochs.compute_psd(tmin=0, tmax=8)
            spectrum_array = spectrum._data

            freqs = spectrum.freqs  # keep freqs for plotting

            spectrum_dict = {'stim_name': stim_name, 'freqs': freqs, 'spectrum': spectrum_array, 'ch_names': channels_to_keep, 'ch_parcellations': channel_parcellations_list}

            with open(os.path.join(save_path, subject_name + "_epochs_spectrum.pkl"), 'wb') as f:
                pickle.dump(spectrum_dict, f)

    print('Done')

    return

def epochs_comodulogram_seeg(data_path, save_path, channels=None, fs_subjects_dir=None):
    """Compute epoch-level PAC for every subject, channel, low frequency, high frequency"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    # num_low_freq = 80
    num_low_freq = 40
    # num_high_freq = 81
    num_high_freq = 41
    low_fq_range = np.linspace(2, 9, num_low_freq)
    # high_fq_range = np.linspace(10, 90, num_high_freq)
    # low_fq_range = np.logspace(*np.log10([2, 9]), num_low_freq)
    high_fq_range = np.logspace(*np.log10([10, 90]), num_high_freq)

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        pac_list = []
        channels_to_keep = []
        channel_parcellations_list = []
        for channel in channels:  # separate for memory issues
            # select epochs
            epochs = mne.read_epochs(epochs_path)

            assert fs_subjects_dir is not None
            epochs, channel_parcellations = select_seeg_channels_from_parcellation(epochs, [channel], fs_subjects_dir)
            if channel_parcellations is not None:
                channel_parcellations_list += channel_parcellations
                channels_to_keep += epochs.ch_names

                epochs = epochs.apply_baseline((None, -0.1))
                fs = 200
                epochs.resample(fs, n_jobs=-1)
                epochs.filter(0.5, None, n_jobs=-1)
                epochs.crop(tmin=0, tmax=8)  # remove onset and offset

                stim_name = epochs.metadata['stim_name'].to_list()

                # phase = np.array([epochs.copy().filter(low_freq - 0.05, low_freq + 0.05, n_jobs=-1, verbose=0).get_data() for low_freq in low_fq_range])  # bandwidth of 0.1Hz
                # phase = np.array([epochs.copy().filter(low_freq - low_freq/20, low_freq + low_freq/20, n_jobs=-1, verbose=0).get_data() for low_freq in low_fq_range])  # bandwidth of 1/10 of an octave
                phase = np.array([epochs.copy().filter(low_freq - low_freq/10, low_freq + low_freq/10, n_jobs=-1, verbose=0).get_data() for low_freq in low_fq_range])
                phase = np.angle(scipy.signal.hilbert(phase, axis=-1))
                phase = np.exp(1j * phase)
                # amplitude = np.array([epochs.copy().filter(high_freq - 1, high_freq + 1, n_jobs=-1, verbose=0).get_data() for high_freq in high_fq_range])  # bandwidth of 2Hz
                amplitude = np.array([epochs.copy().filter(high_freq - high_freq/10, high_freq + high_freq/10, n_jobs=-1, verbose=0).get_data() for high_freq in high_fq_range])  # bandwidth of 1/5 of an octave
                del epochs
                amplitude = np.abs(scipy.signal.hilbert(amplitude, axis=-1))

                # compute pac
                pac_array = np.zeros((phase.shape[1], phase.shape[2], num_low_freq, num_high_freq))
                for low in range(num_low_freq):
                    for high in range(num_high_freq):
                        pac_array[:, :, low, high] = np.abs(np.mean(amplitude[high] * phase[low], axis=-1)) / np.sqrt(np.mean(amplitude[high] ** 2, axis=-1))
                pac_list.append(pac_array)

        # concatenate channels
        pac_array = np.concatenate(pac_list, axis=1)
        pac_dict = {'stim_name': stim_name, 'low_freqs': low_fq_range, 'high_freqs': high_fq_range, 'pac': pac_array, 'ch_names': channels_to_keep, 'ch_parcellations': channel_parcellations_list}

        with open(os.path.join(save_path, subject_name + "_epochs_comodulogram.pkl"), 'wb') as f:
            pickle.dump(pac_dict, f)

        del pac_array
        del pac_dict
        del phase
        del amplitude

    print('Done')

    return


def epochs_comodulogram_seeg_all(data_path, save_path, fs_subjects_dir=None):
    """Compute epoch-level PAC for every subject, channel, low frequency, high frequency"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    # num_low_freq = 80
    num_low_freq = 40
    # num_high_freq = 81
    num_high_freq = 41
    low_fq_range = np.linspace(2, 9, num_low_freq)
    # high_fq_range = np.linspace(10, 90, num_high_freq)
    # low_fq_range = np.logspace(*np.log10([2, 9]), num_low_freq)
    high_fq_range = np.logspace(*np.log10([10, 90]), num_high_freq)

    batch_size = 10  # nb of channels in one loop

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        pac_list = []
        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=fs_subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        n_chan = len(epochs.ch_names)
        print('Number of channels:', len(epochs.ch_names))
        for c in range(0, len(epochs.ch_names), batch_size):

            epochs = mne.read_epochs(epochs_path)
            epochs.pick([ch for ch in range(c, min(c + batch_size, n_chan))])
            print(epochs.ch_names)
            epochs = epochs.apply_baseline((None, -0.1))
            fs = 200
            epochs.resample(fs, n_jobs=-1)
            epochs.filter(0.5, None, n_jobs=-1)
            epochs.crop(tmin=0, tmax=8)  # remove onset and offset

            phase = np.array([epochs.copy().filter(low_freq - low_freq/10, low_freq + low_freq/10, n_jobs=-1, verbose=0).get_data() for low_freq in low_fq_range])
            phase = np.angle(scipy.signal.hilbert(phase, axis=-1))
            phase = np.exp(1j * phase)
            amplitude = np.array([epochs.copy().filter(high_freq - high_freq/10, high_freq + high_freq/10, n_jobs=-1, verbose=0).get_data() for high_freq in high_fq_range])  # bandwidth of 1/5 of an octave

            del epochs
            amplitude = np.abs(scipy.signal.hilbert(amplitude, axis=-1))
            # compute pac
            pac_array = np.zeros((phase.shape[1], phase.shape[2], num_low_freq, num_high_freq))
            for low in range(num_low_freq):
                for high in range(num_high_freq):
                    pac_array[:, :, low, high] = np.abs(np.mean(amplitude[high] * phase[low], axis=-1)) / np.sqrt(np.mean(amplitude[high] ** 2, axis=-1))
            pac_list.append(pac_array)

        # concatenate channels
        pac_array = np.concatenate(pac_list, axis=1)
        pac_dict = {'stim_name': stim_name, 'low_freqs': low_fq_range, 'high_freqs': high_fq_range, 'pac': pac_array, 'ch_names': channels_list, 'ch_parcellations': channel_parcellations_list}

        with open(os.path.join(save_path, subject_name + "_epochs_comodulogram.pkl"), 'wb') as f:
            pickle.dump(pac_dict, f)

        del pac_array
        del pac_dict
        del phase
        del amplitude

    print('Done')

    return

def plot_sample_activity_seeg(data_path):
    """Compute epoch-level RMS for every subject and channel"""
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        print('Number of channels:', len(epochs.ch_names))

        epochs = mne.read_epochs(epochs_path)
        print(epochs.ch_names)
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        # epochs.crop(tmin=0, tmax=8)  # remove onset and offset
        start_baseline = epochs.time_as_index(-1.5)[0]
        start = epochs.time_as_index(0)[0]
        end = epochs.time_as_index(8)[0]

        data = epochs.get_data()[:, :, start:end] # mean over time
        for ep in range (10):
            plt.figure()
            plt.plot(data[ep, 0, :], 'k')
        plt.show()

    return


def epochs_rms_seeg_all(data_path, save_path):
    """Compute epoch-level RMS for every subject and channel"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        print('Number of channels:', len(epochs.ch_names))

        epochs = mne.read_epochs(epochs_path)
        print(epochs.ch_names)
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        # epochs.crop(tmin=0, tmax=8)  # remove onset and offset
        start_baseline = epochs.time_as_index(-1.5)[0]
        start = epochs.time_as_index(0)[0]
        end = epochs.time_as_index(8)[0]

        rms_array = np.sqrt(np.mean(epochs.get_data()[:, :, start:end]**2, axis=2))  # mean over time
        rms_baseline = np.sqrt(np.mean(epochs.get_data()[:, :, start_baseline:start]**2, axis=2))
        rms_array = (rms_array - rms_baseline) / rms_baseline * 100  # percent wrt baseline

        pac_dict = {'stim_name': stim_name, 'rms': rms_array, 'ch_names': channels_list, 'ch_parcellations': channel_parcellations_list, 'info': epochs.info}

        with open(os.path.join(save_path, subject_name + "_epochs_rms.pkl"), 'wb') as f:
            pickle.dump(pac_dict, f)

    print('Done')

    return


def epochs_ma_rms_seeg_all(data_path, save_path, fs_subjects_dir=None):
    """Compute epoch-level abs for every subject and channel"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        print('Number of channels:', len(epochs.ch_names))

        epochs = mne.read_epochs(epochs_path)
        print(epochs.ch_names)
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        # epochs.crop(tmin=0, tmax=8)  # remove onset and offset
        start_baseline = epochs.time_as_index(-1.5)[0]
        start = epochs.time_as_index(0)[0]
        end = epochs.time_as_index(8)[0]

        # compute rolling rms
        rms_full = epochs.get_data()**2
        window_size = int(1 * fs)  # seconds rolling mean
        for e in range(rms_full.shape[0]):
            for c in range(rms_full.shape[1]):
                rms_full[e, c, :] = np.convolve(rms_full[e, c, :], np.ones(window_size) / window_size, mode='same')
        rms_full = np.sqrt(rms_full)  # obtain rms

        rms_array = rms_full[:, :, start:end]
        rms_baseline = np.mean(rms_full[:, :, start_baseline:start], axis=2, keepdims=True)
        rms_array = (rms_array - rms_baseline) / rms_baseline * 100  # percent wrt baseline

        pac_dict = {'stim_name': stim_name, 'ma_rms': rms_array, 'ch_names': channels_list, 'ch_parcellations': channel_parcellations_list}

        with open(os.path.join(save_path, subject_name + "_epochs_ma_rms.pkl"), 'wb') as f:
            pickle.dump(pac_dict, f)

    print('Done')

    return

def epochs_rms_time_windows_seeg_all(data_path, save_path):
    """Compute epoch-level RMS for every subject and channel"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)
    # mne.datasets.fetch_fsaverage(subjects_dir=None)
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        print('Number of channels:', len(epochs.ch_names))

        epochs = mne.read_epochs(epochs_path)
        print(epochs.ch_names)
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        # epochs.crop(tmin=0, tmax=8)  # remove onset and offset
        start_baseline = epochs.time_as_index(-1.5)[0]
        start = epochs.time_as_index(0)[0]

        rms_baseline = np.sqrt(np.mean(epochs.get_data()[:, :, start_baseline:start]**2, axis=2))

        # list_arrays = [[], [], []]
        list_arrays = []
        # ends = [1, 2, 3, 4, 5, 6, 7, 8]
        # for end in ends:
        #     end = epochs.time_as_index(end)[0]
        #     rms_array = np.sqrt(np.mean(epochs.get_data()[:, :, start:end]**2, axis=2))  # mean over time
        #     list_arrays[0].append((rms_array - rms_baseline) / rms_baseline * 100)  # percent wrt baseline
        # end = epochs.time_as_index(8)[0]
        # starts = [0, 1, 2, 3, 4, 5, 6, 7]
        # for start in starts:
        #     start = epochs.time_as_index(start)[0]
        #     rms_array = np.sqrt(np.mean(epochs.get_data()[:, :, start:end] ** 2, axis=2))  # mean over time
        #     list_arrays[1].append((rms_array - rms_baseline) / rms_baseline * 100)  # percent wrt baseline
        starts = [0, 1, 2, 3, 4, 5, 6, 7]
        for s in range(len(starts)):
            start = epochs.time_as_index(starts[s])[0]
            end = epochs.time_as_index(starts[s]+1)[0]  # 1 sec window
            rms_array = np.sqrt(np.mean(epochs.get_data()[:, :, start:end] ** 2, axis=2))  # mean over time
            # list_arrays[2].append((rms_array - rms_baseline) / rms_baseline * 100)  # percent wrt baseline
            list_arrays.append((rms_array - rms_baseline) / rms_baseline * 100)  # percent wrt baseline

        pac_dict = {'stim_name': stim_name, 'rms': list_arrays, 'ch_names': channels_list, 'ch_parcellations': channel_parcellations_list}

        with open(os.path.join(save_path, subject_name + "_epochs_rms_windows.pkl"), 'wb') as f:
            pickle.dump(pac_dict, f)

    print('Done')

    return


def epochs_band_power_seeg_all(data_path, save_path, freq_bands=np.arange(50, 151, 10), fs_subjects_dir=None):
    """Compute epoch-level power in a given frequency band for every subject and channel"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)
    # mne.datasets.fetch_fsaverage(subjects_dir=None)
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        print('Number of channels:', len(epochs.ch_names))

        epochs = mne.read_epochs(epochs_path)
        print(epochs.ch_names)
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        start_baseline = epochs.time_as_index(-1.5)[0]
        start = epochs.time_as_index(0)[0]
        end = epochs.time_as_index(8)[0]

        epochs.filter(freq_bands[0], freq_bands[1], n_jobs=-1, verbose=0)
        power = np.abs(scipy.signal.hilbert(epochs.get_data(), axis=-1))**2  # extract power
        baseline = np.mean(power[:, :, start_baseline:start], axis=2)  # average over time
        power = (np.mean(power[:, :, start:end], axis=2) - baseline) / baseline * 100  # average over time, percent wrt baseline

        # data = epochs.get_data()
        # del epochs
        # BLP = np.zeros((data.shape[0], data.shape[1], len(freq_bands)-1, data.shape[2]), dtype=np.float32)
        # for j in tqdm(range(len(freq_bands) - 1)):
        #     lowF = freq_bands[j]
        #     highF = freq_bands[j + 1]
        #     B, A = scipy.signal.butter(4, [lowF, highF], btype='bandpass', fs=fs)
        #     filtered_data = scipy.signal.filtfilt(B, A, data, axis=-1)
        #     BLP[:, :, j, :] = np.abs(scipy.signal.hilbert(filtered_data))**2
        #
        # # Normalize the power by dividing by the baseline mean within each frequency band
        # baseline = np.mean(BLP[:, :, :, start_baseline:start], axis=-1, keepdims=True)
        # BLP = (BLP[:, :, :, start:end] - baseline) / baseline
        #
        # # Average across frequency bands and multiply by 100 to express as percent
        # power = BLP.mean(axis=(2, 3)) * 100  # average across bands and time

        power_dict = {'stim_name': stim_name, 'power': power, 'ch_names': channels_list, 'ch_parcellations': channel_parcellations_list, 'info': epochs.info}

        with open(os.path.join(save_path, subject_name + "_epochs_power.pkl"), 'wb') as f:
            pickle.dump(power_dict, f)

    print('Done')

    return


def epochs_band_power_all_freq_seeg_all(data_path, save_path):
    """Compute epoch-level power in a given frequency band for every subject and channel"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)
    # mne.datasets.fetch_fsaverage(subjects_dir=None)
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        print('Number of channels:', len(epochs.ch_names))

        epochs = mne.read_epochs(epochs_path)
        print(epochs.ch_names)
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        start_baseline = epochs.time_as_index(-1.5)[0]
        start = epochs.time_as_index(0)[0]
        end = epochs.time_as_index(8)[0]

        # freqs = np.logspace(*np.log10([12, 188]), num=100).tolist()
        freqs = np.logspace(*np.log10([1.5, 180]), num=200).tolist()
        # freqs = np.logspace(*np.log10([1, 9]), num=20).tolist()

        power_list = []
        for f, freq in enumerate(tqdm(freqs)):
            filtered_epochs = epochs.copy().filter(freq - freq/10, freq + freq/10, n_jobs=-1, verbose=False)
            # filtered_epochs = epochs.copy().filter(freq - 0.5, freq + 0.5, n_jobs=-1, verbose=False)
            power = np.abs(scipy.signal.hilbert(filtered_epochs.get_data(), axis=-1))**2  # extract power
            # baseline = np.mean(power[:, :, start_baseline:start], axis=2)  # average over time
            baseline = power[:, :, start_baseline:start]  # non averaged baseline
            # power = (np.mean(power[:, :, start:end], axis=2) - baseline) / baseline * 100  # average over time, percent wrt baseline
            power = (np.mean(power[:, :, start:end], axis=2) - np.mean(baseline, axis=2)) / np.std(baseline, axis=2)  # average over time, zscored with baseline
            power_list.append(power)
        power_list = np.array(power_list)

        power_dict = {'stim_name': stim_name, 'power': power_list, 'ch_names': channels_list, 'ch_parcellations': channel_parcellations_list, 'info': epochs.info, 'freq': freqs}

        with open(os.path.join(save_path, subject_name + "_epochs_power_all_freqs.pkl"), 'wb') as f:
            pickle.dump(power_dict, f)

    print('Done')

    return


def epochs_band_power_time_windows_seeg_all(data_path, save_path, freq_bands, fs_subjects_dir=None):
    """Compute epoch-level power for every subject and channel"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)
    # mne.datasets.fetch_fsaverage(subjects_dir=None)
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        print('Number of channels:', len(epochs.ch_names))

        print(epochs.ch_names)
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        # epochs.crop(tmin=0, tmax=8)  # remove onset and offset
        start_baseline = epochs.time_as_index(-1.5)[0]
        start = epochs.time_as_index(0)[0]

        epochs.filter(freq_bands[0], freq_bands[1], n_jobs=-1, verbose=0)
        power = np.abs(scipy.signal.hilbert(epochs.get_data(), axis=-1)) ** 2  # extract power
        baseline = np.mean(power[:, :, start_baseline:start], axis=2)  # average over time

        list_arrays = [[], [], []]
        ends = [1, 2, 3, 4, 5, 6, 7, 8]
        for end in ends:
            end = epochs.time_as_index(end)[0]
            list_arrays[0].append((np.mean(power[:, :, start:end], axis=2) - baseline) / baseline * 100)  # percent wrt baseline
        end = epochs.time_as_index(8)[0]
        starts = [0, 1, 2, 3, 4, 5, 6, 7]
        for start in starts:
            start = epochs.time_as_index(start)[0]
            list_arrays[1].append((np.mean(power[:, :, start:end], axis=2) - baseline) / baseline * 100)  # percent wrt baseline
        starts = [0, 1, 2, 3, 4, 5, 6, 7]
        for s in range(len(starts)):
            start = epochs.time_as_index(starts[s])[0]
            end = epochs.time_as_index(starts[s]+1)[0]  # 1 sec window
            list_arrays[2].append((np.mean(power[:, :, start:end], axis=2) - baseline) / baseline * 100)  # percent wrt baseline

        pac_dict = {'stim_name': stim_name, 'power': list_arrays, 'ch_names': channels_list, 'ch_parcellations': channel_parcellations_list}

        with open(os.path.join(save_path, subject_name + "_epochs_power_windows.pkl"), 'wb') as f:
            pickle.dump(pac_dict, f)

    print('Done')

    return


def compute_epoch_band_power_stim_freq(epochs, e, stim_freq, start_baseline, end_baseline, start, end):
    """Helper function to compute PAC for a single epoch."""
    power = epochs[e].copy().filter(stim_freq - 0.1, stim_freq + 0.1, n_jobs=-1, verbose=0).get_data()
    power = np.abs(scipy.signal.hilbert(power[0, :, :], axis=-1)) ** 2

    baseline = np.mean(power[:, start_baseline:end_baseline], axis=-1)
    power = (np.mean(power[:, start:end], axis=-1) - baseline) / baseline * 100  # average over time, percent wrt baseline

    return power

def epochs_band_power_stim_freq_seeg_all(data_path, save_path, fs_subjects_dir=None):
    """Compute epoch-level PAC at given phase and amplitude frequencies for every subject and channel."""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        start_baseline = epochs.time_as_index(-1.5)[0]
        end_baseline = epochs.time_as_index(0)[0]
        start = epochs.time_as_index(0)[0]
        end = epochs.time_as_index(8)[0]

        # Parallelize PAC computation across epochs
        pac_array = Parallel(n_jobs=-1)(
            delayed(compute_epoch_band_power_stim_freq)(epochs, e, float(stim_name[e][14:17]), start_baseline, end_baseline, start, end)
            for e in range(len(epochs))
        )
        pac_array = np.array(pac_array)

        power_dict = {
            'stim_name': stim_name,
            'power': pac_array,
            'ch_names': channels_list,
            'ch_parcellations': channel_parcellations_list,
            'info': epochs.info,
            'freq': 'stim_freq',
        }

        with open(os.path.join(save_path, subject_name + "_epochs_power_stim_freq.pkl"), 'wb') as f:
            pickle.dump(power_dict, f)

    print('Done')
    return


def epochs_pac_seeg_all(data_path, save_path, low_freq=4, high_freq=80, fs_subjects_dir=None):
    """Compute epoch-level pac at given phase and amplitude frequencies for every subject and channel"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)
    # mne.datasets.fetch_fsaverage(subjects_dir=None)
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        print('Number of channels:', len(epochs.ch_names))

        epochs = mne.read_epochs(epochs_path)
        print(epochs.ch_names)
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        start_baseline = epochs.time_as_index(-1.5)[0]
        start = epochs.time_as_index(0)[0]
        end = epochs.time_as_index(8)[0]

        # phase = epochs.copy().filter(low_freq - low_freq / 10, low_freq + low_freq / 10, n_jobs=-1, verbose=0).get_data()
        # phase = epochs.copy().filter(low_freq - low_freq / 3, low_freq + low_freq / 3, n_jobs=-1, verbose=0).get_data()
        phase = epochs.copy().filter(low_freq - 2, low_freq + 2, n_jobs=-1, verbose=0).get_data()
        # phase = epochs.copy().filter(low_freq - 0.1, low_freq + 0.1, n_jobs=-1, verbose=0).get_data()
        phase = np.angle(scipy.signal.hilbert(phase, axis=-1))
        phase = np.exp(1j * phase)
        # amplitude = epochs.copy().filter(high_freq - high_freq / 10, high_freq + high_freq / 10, n_jobs=-1, verbose=0).get_data()  # bandwidth of 1/5 of an octave
        # amplitude = epochs.copy().filter(high_freq - high_freq / 3, high_freq + high_freq / 3, n_jobs=-1, verbose=0).get_data()  # bandwidth of 1/5 of an octave
        amplitude = epochs.copy().filter(high_freq - 10, high_freq + 10, n_jobs=-1, verbose=0).get_data()  # bandwidth of 1/5 of an octave
        # amplitude = epochs.copy().filter(high_freq - 35, high_freq + 35, n_jobs=-1, verbose=0).get_data()  # bandwidth of 1/5 of an octave
        amplitude = np.abs(scipy.signal.hilbert(amplitude, axis=-1))

        baseline = np.abs(np.mean(amplitude[:, :, start_baseline:start] * phase[:, :, start_baseline:start], axis=-1)) / np.sqrt(np.mean(amplitude[:, :, start_baseline:start] ** 2, axis=-1))
        pac = np.abs(np.mean(amplitude[:, :, start:end] * phase[:, :, start:end], axis=-1)) / np.sqrt(np.mean(amplitude[:, :, start:end] ** 2, axis=-1))
        pac = (pac - baseline) / baseline * 100  # percent wrt baseline

        power_dict = {'stim_name': stim_name, 'pac': pac, 'ch_names': channels_list, 'ch_parcellations': channel_parcellations_list, 'info': epochs.info, 'low_freq': low_freq, 'high_freq': high_freq}

        with open(os.path.join(save_path, subject_name + "_epochs_pac.pkl"), 'wb') as f:
            pickle.dump(power_dict, f)

    print('Done')

    return


def epochs_pac_time_windows_seeg_all(data_path, save_path, low_freq=6, high_freq=130):
    """Compute epoch-level power for every subject and channel"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)
    # mne.datasets.fetch_fsaverage(subjects_dir=None)
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        print('Number of channels:', len(epochs.ch_names))

        print(epochs.ch_names)
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        start_baseline = epochs.time_as_index(-1.5)[0]
        end_baseline = epochs.time_as_index(0)[0]

        phase = epochs.copy().filter(low_freq - 2, low_freq + 2, n_jobs=-1, verbose=0).get_data()
        phase = np.angle(scipy.signal.hilbert(phase, axis=-1))
        phase = np.exp(1j * phase)
        amplitude = epochs.copy().filter(high_freq - 10, high_freq + 10, n_jobs=-1, verbose=0).get_data()
        amplitude = np.abs(scipy.signal.hilbert(amplitude, axis=-1))

        baseline = np.abs(np.mean(amplitude[:, :, start_baseline:end_baseline] * phase[:, :, start_baseline:end_baseline], axis=-1)) / np.sqrt(np.mean(amplitude[:, :, start_baseline:end_baseline] ** 2, axis=-1))

        list_arrays = []
        starts = [0, 1, 2, 3, 4, 5, 6, 7]
        for s in range(len(starts)):
            start = epochs.time_as_index(starts[s])[0]
            end = epochs.time_as_index(starts[s]+1)[0]  # 1 sec window
            pac = np.abs(np.mean(amplitude[:, :, start:end] * phase[:, :, start:end], axis=-1)) / np.sqrt(np.mean(amplitude[:, :, start:end] ** 2, axis=-1))
            pac = (pac - baseline) / baseline * 100  # percent wrt baseline
            list_arrays.append(pac)  # percent wrt baseline

        pac_dict = {'stim_name': stim_name, 'pac': list_arrays, 'ch_names': channels_list, 'ch_parcellations': channel_parcellations_list, 'low_freq': low_freq, 'high_freq': high_freq}

        with open(os.path.join(save_path, subject_name + "_epochs_pac_windows.pkl"), 'wb') as f:
            pickle.dump(pac_dict, f)

    print('Done')

    return


def epochs_pac_theta_bands_time_windows_seeg_all(data_path, save_path, low_freq=[4, 5, 6, 7, 8], high_freq=130):
    """Compute epoch-level power for every subject and channel"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)
    # mne.datasets.fetch_fsaverage(subjects_dir=None)
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        print('Number of channels:', len(epochs.ch_names))

        print(epochs.ch_names)
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        start_baseline = epochs.time_as_index(-1.5)[0]
        end_baseline = epochs.time_as_index(0)[0]

        amplitude = epochs.copy().filter(high_freq - 10, high_freq + 10, n_jobs=-1, verbose=0).get_data()
        amplitude = np.abs(scipy.signal.hilbert(amplitude, axis=-1))
        theta_bands_pac_list = []
        for low_f in low_freq:
            phase = epochs.copy().filter(low_f - 0.1, low_f + 0.1, n_jobs=-1, verbose=0).get_data()
            phase = np.angle(scipy.signal.hilbert(phase, axis=-1))
            phase = np.exp(1j * phase)

            baseline = np.abs(np.mean(amplitude[:, :, start_baseline:end_baseline] * phase[:, :, start_baseline:end_baseline], axis=-1)) / np.sqrt(np.mean(amplitude[:, :, start_baseline:end_baseline] ** 2, axis=-1))

            list_arrays = []
            starts = [0, 1, 2, 3, 4, 5, 6, 7]
            for s in range(len(starts)):
                start = epochs.time_as_index(starts[s])[0]
                end = epochs.time_as_index(starts[s]+1)[0]  # 1 sec window
                pac = np.abs(np.mean(amplitude[:, :, start:end] * phase[:, :, start:end], axis=-1)) / np.sqrt(np.mean(amplitude[:, :, start:end] ** 2, axis=-1))
                pac = (pac - baseline) / baseline * 100  # percent wrt baseline
                list_arrays.append(pac)  # percent wrt baseline
            theta_bands_pac_list.append(list_arrays)

        pac_dict = {'stim_name': stim_name, 'pac': theta_bands_pac_list, 'ch_names': channels_list, 'ch_parcellations': channel_parcellations_list, 'low_freq': low_freq, 'high_freq': high_freq}

        with open(os.path.join(save_path, subject_name + "_epochs_pac_theta_bands_windows.pkl"), 'wb') as f:
            pickle.dump(pac_dict, f)

    print('Done')

    return


def compute_epoch_pac(epochs, e, stim_freq, high_freq, start_baseline, end_baseline, start, end):
    """Helper function to compute PAC for a single epoch."""
    # phase = epochs[e].copy().filter(stim_freq - 0.1, stim_freq + 0.1, n_jobs=-1, verbose=False).get_data()
    phase = epochs[e].copy().filter(stim_freq - 0.5, stim_freq + 0.5, n_jobs=-1, verbose=False).get_data()
    # phase = epochs[e].copy().filter(stim_freq - stim_freq/10, stim_freq + stim_freq/10, n_jobs=-1, verbose=False).get_data()
    phase = np.angle(scipy.signal.hilbert(phase[0, :, :], axis=-1))
    phase = np.exp(1j * phase)

    amplitude = epochs[e].copy().filter(high_freq - 10, high_freq + 10, n_jobs=-1, verbose=False).get_data()
    # amplitude = epochs[e].copy().filter(high_freq - 20, high_freq + 20, n_jobs=-1, verbose=False).get_data()
    # amplitude = epochs[e].copy().filter(high_freq - 30, high_freq + 30, n_jobs=-1, verbose=False).get_data()
    # amplitude = epochs[e].copy().filter(high_freq - 35, high_freq + 35, n_jobs=-1, verbose=False).get_data()
    # amplitude = epochs[e].copy().filter(high_freq - high_freq/10, high_freq + high_freq/10, n_jobs=-1, verbose=False).get_data()
    amplitude = np.abs(scipy.signal.hilbert(amplitude[0, :, :], axis=-1))

    baseline = np.abs(np.mean(amplitude[:, start_baseline:end_baseline] * phase[:, start_baseline:end_baseline], axis=-1)) / np.sqrt(np.mean(amplitude[:, start_baseline:end_baseline] ** 2, axis=-1))
    pac = np.abs(np.mean(amplitude[:, start:end] * phase[:, start:end], axis=-1)) / np.sqrt(np.mean(amplitude[:, start:end] ** 2, axis=-1))
    # pac = (pac - baseline) / baseline * 100  # percent wrt baseline
    pac = (pac - baseline)  # percent wrt baseline
    return pac
    # return baseline

def epochs_pac_stim_freq_seeg_all(data_path, save_path, high_freq=80, fs_subjects_dir=None):
    """Compute epoch-level PAC at given phase and amplitude frequencies for every subject and channel."""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        start_baseline = epochs.time_as_index(-1.5)[0]
        end_baseline = epochs.time_as_index(0)[0]
        start = epochs.time_as_index(0)[0]
        end = epochs.time_as_index(8)[0]

        # Parallelize PAC computation across epochs
        pac_array = Parallel(n_jobs=-1)(
            delayed(compute_epoch_pac)(epochs, e, float(stim_name[e][14:17]), high_freq, start_baseline, end_baseline, start, end)
            for e in range(len(epochs))
        )
        pac_array = np.array(pac_array)

        power_dict = {
            'stim_name': stim_name,
            'pac': pac_array,
            'ch_names': channels_list,
            'ch_parcellations': channel_parcellations_list,
            'info': epochs.info,
            'low_freq': 'stim_freq',
            'high_freq': high_freq
        }

        with open(os.path.join(save_path, subject_name + "_epochs_pac.pkl"), 'wb') as f:
            pickle.dump(power_dict, f)

    print('Done')
    return


def epochs_pac_stim_freq_time_windows_seeg_all(data_path, save_path, high_freq=130):
    """Compute epoch-level power for every subject and channel"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)
    # mne.datasets.fetch_fsaverage(subjects_dir=None)
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        print('Number of channels:', len(epochs.ch_names))

        print(epochs.ch_names)
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        start_baseline = epochs.time_as_index(-1.5)[0]
        end_baseline = epochs.time_as_index(0)[0]

        list_arrays = []
        starts = [0, 1, 2, 3, 4, 5, 6, 7]
        for s in range(len(starts)):
            start = epochs.time_as_index(starts[s])[0]
            end = epochs.time_as_index(starts[s]+1)[0]  # 1 sec window
            pac_array = Parallel(n_jobs=-1)(
                delayed(compute_epoch_pac)(epochs, e, float(stim_name[e][14:17]), high_freq, start_baseline, end_baseline, start, end)
                for e in range(len(epochs))
            )
            pac_array = np.array(pac_array)
            list_arrays.append(pac_array)  # percent wrt baseline

        pac_dict = {'stim_name': stim_name, 'pac': list_arrays, 'ch_names': channels_list, 'ch_parcellations': channel_parcellations_list, 'low_freq': 'stim_freq', 'high_freq': high_freq}

        with open(os.path.join(save_path, subject_name + "_epochs_pac_windows.pkl"), 'wb') as f:
            pickle.dump(pac_dict, f)

    print('Done')

    return


def epochs_pac_stim_freq_and_all_freqs_seeg_all(data_path, save_path):
    """Compute epoch-level PAC at given phase and amplitude frequencies for every subject and channel."""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        start_baseline = epochs.time_as_index(-1.5)[0]
        end_baseline = epochs.time_as_index(0)[0]
        start = epochs.time_as_index(0)[0]
        end = epochs.time_as_index(8)[0]

        freqs = np.logspace(*np.log10([12, 188]), num=100).tolist()

        pac_list = []
        for f, freq in enumerate(tqdm(freqs)):
            # Parallelize PAC computation across epochs
            pac_array = Parallel(n_jobs=-1)(delayed(compute_epoch_pac)(epochs, e, float(stim_name[e][14:17]), freq, start_baseline, end_baseline, start, end)
                for e in range(len(epochs))
            )
            pac_list.append(np.array(pac_array))
        pac_list = np.array(pac_list)

        power_dict = {
            'stim_name': stim_name,
            'pac': pac_list,
            'ch_names': channels_list,
            'ch_parcellations': channel_parcellations_list,
            'info': epochs.info,
            'low_freq': 'stim_freq',
            'high_freq': freqs
        }

        with open(os.path.join(save_path, subject_name + "_epochs_pac_all_freqs.pkl"), 'wb') as f:
            pickle.dump(power_dict, f)

    print('Done')
    return


def epochs_comodulogram_theta_gamma_seeg_all(data_path, save_path):
    """Compute epoch-level PAC at given phase and amplitude frequencies for every subject and channel."""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        start = epochs.time_as_index(0)[0]
        end = epochs.time_as_index(8)[0]

        theta_freqs = np.linspace(2, 9, num=40).tolist()
        gamma_freqs = np.logspace(*np.log10([12, 165]), num=40).tolist()

        all_amplitudes = []
        for gf, gfreq in enumerate(tqdm(gamma_freqs)):
            amplitude = epochs.copy().filter(gfreq - gfreq/5, gfreq + gfreq/5, n_jobs=-1, verbose=False).get_data()
            all_amplitudes.append(np.abs(scipy.signal.hilbert(amplitude, axis=-1)))

        como_list = []
        for f, freq in enumerate(tqdm(theta_freqs)):
            pac_list = []
            phase = epochs.copy().filter(freq - 0.5, freq + 0.5, n_jobs=-1, verbose=False).get_data()
            phase = np.angle(scipy.signal.hilbert(phase, axis=-1))
            phase = np.exp(1j * phase)
            for gf, gfreq in enumerate(tqdm(gamma_freqs)):
                # amplitude = epochs.copy().filter(gfreq - 10, gfreq + 10, n_jobs=-1, verbose=False).get_data()
                # amplitude = np.abs(scipy.signal.hilbert(amplitude, axis=-1))
                amplitude = all_amplitudes[gf]
                pac = np.abs(np.mean(amplitude[:, :, start:end] * phase[:, :, start:end], axis=-1)) / np.sqrt(np.mean(amplitude[:, :, start:end] ** 2, axis=-1))
                pac_list.append(pac)
            como_list.append(np.array(pac_list))
        como_list = np.array(como_list)

        power_dict = {
            'stim_name': stim_name,
            'comodulogram': como_list,
            'ch_names': channels_list,
            'ch_parcellations': channel_parcellations_list,
            'info': epochs.info,
            'low_freq': theta_freqs,
            'high_freq': gamma_freqs
        }

        with open(os.path.join(save_path, subject_name + "_epochs_como_all_freqs.pkl"), 'wb') as f:
            pickle.dump(power_dict, f)

    print('Done')
    return


def compute_epoch_phase_coherence(epochs, e, e2, freq, start, end):
    """Helper function to compute phase coherence for a single epoch and its repetition."""
    if e2 is None:
        empty_coherence = np.zeros((len(epochs.ch_names)))
        empty_coherence[:] = np.nan
        return empty_coherence
    # phase = epochs[e].copy().filter(freq - 0.1, freq + 0.1, n_jobs=-1, verbose=False).get_data()
    # phase = epochs[e].copy().filter(freq - 1, freq + 1, n_jobs=-1, verbose=False).get_data()
    phase = epochs[e].copy().filter(freq - freq/10, freq + freq/10, n_jobs=-1, verbose=False).get_data()
    phase = np.angle(scipy.signal.hilbert(phase[0, :, :], axis=-1))
    # phase2 = epochs[e2].copy().filter(freq - 0.1, freq + 0.1, n_jobs=-1, verbose=False).get_data()
    # phase2 = epochs[e2].copy().filter(freq - 1, freq + 1, n_jobs=-1, verbose=False).get_data()
    phase2 = epochs[e2].copy().filter(freq - freq/10, freq + freq/10, n_jobs=-1, verbose=False).get_data()
    phase2 = np.angle(scipy.signal.hilbert(phase2[0, :, :], axis=-1))
    phase_coherence = np.abs(np.mean(np.exp(1j * (phase[:, start:end] - phase2[:, start:end])), axis=-1))  # mean across time
    return phase_coherence

def epochs_phase_coherence_stim_freq_seeg_all(data_path, save_path):
    """Compute epoch-level phase coherenc at given phase and amplitude frequencies for every subject and channel."""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        start = epochs.time_as_index(0)[0]
        end = epochs.time_as_index(8)[0]

        # pair stims with their other rep
        # Create a lookup table for the other repetition
        stim_to_index = {}
        for idx, stim in enumerate(stim_name):
            base_stim = "_".join(stim.split("_")[2:])  # Remove repetition info
            if base_stim not in stim_to_index:
                stim_to_index[base_stim] = []
            stim_to_index[base_stim].append(idx)
        # Build the list of e and e2, keeping e in order
        epoch_pairs = []
        for e in range(len(epochs)):
            base_stim = "_".join(stim_name[e].split("_")[2:])  # Base stimulus name
            other_indices = stim_to_index[base_stim]  # Get all indices for this stimulus
            e2 = [idx for idx in other_indices if idx != e]  # Find the other repetition
            if e2:  # If the other repetition exists
                epoch_pairs.append((e, e2[0]))
            else:
                epoch_pairs.append((e, None))  # Handle cases where no pair exists

        # Parallelize phase_coherence computation across epochs
        phase_coherence = Parallel(n_jobs=-1)(
            delayed(compute_epoch_phase_coherence)(epochs, e, e2, float(stim_name[e][14:17]), start, end)
            for e, e2 in epoch_pairs
        )
        phase_coherence = np.array(phase_coherence)

        power_dict = {
            'stim_name': stim_name,
            'phase_coherence': phase_coherence,
            'ch_names': channels_list,
            'ch_parcellations': channel_parcellations_list,
            'info': epochs.info,
            'low_freq': 'stim_freq',
        }

        with open(os.path.join(save_path, subject_name + "_epochs_phase_coherence.pkl"), 'wb') as f:
            pickle.dump(power_dict, f)

    print('Done')
    return

def epochs_phase_coherence_stim_freq_time_windows_seeg_all(data_path, save_path):
    """Compute epoch-level power for every subject and channel"""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)
    # mne.datasets.fetch_fsaverage(subjects_dir=None)
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        print('Number of channels:', len(epochs.ch_names))

        print(epochs.ch_names)
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)

        # pair stims with their other rep
        # Create a lookup table for the other repetition
        stim_to_index = {}
        for idx, stim in enumerate(stim_name):
            base_stim = "_".join(stim.split("_")[2:])  # Remove repetition info
            if base_stim not in stim_to_index:
                stim_to_index[base_stim] = []
            stim_to_index[base_stim].append(idx)
        # Build the list of e and e2, keeping e in order
        epoch_pairs = []
        for e in range(len(epochs)):
            base_stim = "_".join(stim_name[e].split("_")[2:])  # Base stimulus name
            other_indices = stim_to_index[base_stim]  # Get all indices for this stimulus
            e2 = [idx for idx in other_indices if idx != e]  # Find the other repetition
            if e2:  # If the other repetition exists
                epoch_pairs.append((e, e2[0]))
            else:
                epoch_pairs.append((e, None))  # Handle cases where no pair exists

        list_arrays = []
        starts = [0, 1, 2, 3, 4, 5, 6, 7]
        for s in range(len(starts)):
            start = epochs.time_as_index(starts[s])[0]
            end = epochs.time_as_index(starts[s]+1)[0]  # 1 sec window

            phase_coherence = Parallel(n_jobs=-1)(
                delayed(compute_epoch_phase_coherence)(epochs, e, e2, float(stim_name[e][14:17]), start, end)
                for e, e2 in epoch_pairs
            )
            phase_coherence = np.array(phase_coherence)
            list_arrays.append(phase_coherence)  # percent wrt baseline

        pac_dict = {'stim_name': stim_name, 'phase_coherence': list_arrays, 'ch_names': channels_list, 'ch_parcellations': channel_parcellations_list}

        with open(os.path.join(save_path, subject_name + "_epochs_phase_coherence_windows.pkl"), 'wb') as f:
            pickle.dump(pac_dict, f)

    print('Done')

    return


def epochs_phase_coherence_all_freq_seeg_all(data_path, save_path):
    """Compute epoch-level phase coherence at given phase and amplitude frequencies for every subject and channel."""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        start = epochs.time_as_index(0)[0]
        end = epochs.time_as_index(8)[0]

        # pair stims with their other rep
        # Create a lookup table for the other repetition
        stim_to_index = {}
        for idx, stim in enumerate(stim_name):
            base_stim = "_".join(stim.split("_")[2:])  # Remove repetition info
            if base_stim not in stim_to_index:
                stim_to_index[base_stim] = []
            stim_to_index[base_stim].append(idx)
        # Build the list of e and e2, keeping e in order
        epoch_pairs = []
        for e in range(len(epochs)):
            base_stim = "_".join(stim_name[e].split("_")[2:])  # Base stimulus name
            other_indices = stim_to_index[base_stim]  # Get all indices for this stimulus
            e2 = [idx for idx in other_indices if idx != e]  # Find the other repetition
            if e2:  # If the other repetition exists
                epoch_pairs.append((e, e2[0]))
            else:
                epoch_pairs.append((e, None))  # Handle cases where no pair exists

        freqs = np.logspace(*np.log10([1, 10]), num=40).tolist() + np.logspace(*np.log10([10, 180]), num=91).tolist()

        phase_coherence_list = []
        for f, freq in enumerate(tqdm(freqs)):
            # Parallelize phase_coherence computation across epochs
            phase_coherence = Parallel(n_jobs=-1)(
                delayed(compute_epoch_phase_coherence)(epochs, e, e2, freq, start, end)
                for e, e2 in epoch_pairs
            )
            phase_coherence_list.append(np.array(phase_coherence))

        phase_coherence_list = np.array(phase_coherence_list)

        power_dict = {
            'stim_name': stim_name,
            'phase_coherence': phase_coherence_list,
            'ch_names': channels_list,
            'ch_parcellations': channel_parcellations_list,
            'info': epochs.info,
            'low_freq': freqs,
        }

        with open(os.path.join(save_path, subject_name + "_epochs_phase_coherence_all_freqs.pkl"), 'wb') as f:
            pickle.dump(power_dict, f)

    print('Done')
    return


def compute_epoch_amplitude_by_phase_histogram(epochs, e, stim_freq, high_freq, start, end, phase_bins):
    """Helper function to compute PAC for a single epoch."""
    phase = epochs[e].copy().filter(stim_freq - 0.1, stim_freq + 0.1, n_jobs=-1, verbose=0).get_data()
    phase = np.angle(scipy.signal.hilbert(phase[0, :, :], axis=-1))
    phase = phase[:, start:end]
    amplitude = epochs[e].copy().filter(high_freq - 10, high_freq + 10, n_jobs=-1, verbose=0).get_data()
    # amplitude = epochs[e].copy().filter(high_freq - 35, high_freq + 35, n_jobs=-1, verbose=0).get_data()
    amplitude = np.abs(scipy.signal.hilbert(amplitude[0, :, :], axis=-1))
    amplitude = amplitude[:, start:end]
    bin_indices = np.digitize(phase, phase_bins)
    histogram = np.zeros((bin_indices.shape[0], len(phase_bins)-1))
    for c in range(bin_indices.shape[0]):
        histogram[c, :] = np.array([amplitude[c, bin_indices[c, :] == i].mean() for i in range(1, len(phase_bins))])
        histogram[c, :] = histogram[c, :] / np.sum(histogram[c, :])  # normalize to sum to 1

    return histogram

def epochs_amplitude_by_phase_histogram_seeg_all(data_path, save_path, high_freq=130):
    """Compute epoch-level histogram of gamma amplitude per stim frequency phase bin for every subject and channel."""
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    subject = 'fsaverage'

    phase_bins = np.linspace(-np.pi, np.pi, 18 + 1)  # 18 bins

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)

        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        stim_name = epochs.metadata['stim_name'].to_list()
        epochs = epochs.apply_baseline((None, -0.1))
        fs = 400
        epochs.resample(fs, n_jobs=-1)
        epochs.filter(0.5, None, n_jobs=-1)
        start = epochs.time_as_index(0)[0]
        end = epochs.time_as_index(8)[0]

        # Parallelize PAC computation across epochs
        histogram_array = Parallel(n_jobs=-1)(
            delayed(compute_epoch_amplitude_by_phase_histogram)(epochs, e, float(stim_name[e][14:17]), high_freq, start, end, phase_bins)
            for e in range(len(epochs))
        )
        histogram_array = np.array(histogram_array)

        power_dict = {
            'stim_name': stim_name,
            'histogram': histogram_array,
            'ch_names': channels_list,
            'ch_parcellations': channel_parcellations_list,
            'info': epochs.info,
            'low_freq': 'stim_freq',
            'high_freq': high_freq
        }

        with open(os.path.join(save_path, subject_name + "_epochs_amplitude_by_phase_histogram.pkl"), 'wb') as f:
            pickle.dump(power_dict, f)

    print('Done')
    return


def plot_electrode_spectrum(data_path, behavior_path, channels=None, subject_subset=None):
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*epochs_spectrum.pkl')))
    if subject_subset is not None:
        subject_paths = np.array(subject_paths)[subject_subset]
    print(subject_paths)

    save_path = data_path
    os.makedirs(os.path.join(save_path, 'signif_figures_encoding'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'signif_figures_novelty'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'signif_figures_retrieval'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'signif_figures_memory'), exist_ok=True)

    conditions = ['encoding', 'novelty', 'retrieval', 'memory']

    p_value_lists = [[], [], [], []]
    subject_lists = [[], [], [], []]
    channel_lists = [[], [], [], []]

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        spectrum_obj = np.load(subject_path, allow_pickle=True)
        spectrum = spectrum_obj['spectrum']
        if channels is not None:
            chan_ind = []
            for p, parcel in enumerate(spectrum_obj['ch_parcellations']):
                if any(ch_parcel in channels for ch_parcel in parcel):
                    chan_ind.append(p)
            spectrum = spectrum[:, chan_ind, :]
            print(np.array(spectrum_obj['ch_names'])[chan_ind])
        if len(chan_ind) > 0:
            for cond, correct_incorrect in enumerate([[correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
                correct = [e for e in range(len(spectrum_obj['stim_name'])) if spectrum_obj['stim_name'][e] in correct_incorrect[0]]
                incorrect = [e for e in range(len(spectrum_obj['stim_name'])) if spectrum_obj['stim_name'][e] in correct_incorrect[1]]
                if (len(correct) > 0) & (len(incorrect) > 0):
                    pooled_std = np.sqrt(((len(correct) - 1) * np.std(spectrum[correct, :, :], axis=0, ddof=1) ** 2 + (
                                len(incorrect) - 1) * np.std(spectrum[incorrect, :, :], axis=0, ddof=1) ** 2) / (
                                                     len(correct) + len(incorrect) - 2))
                    t_values = (np.mean(spectrum[correct, :, :], axis=0) - np.mean(spectrum[incorrect, :, :], axis=0)) / (
                                pooled_std * np.sqrt(1 / len(correct) + 1 / len(incorrect)))
                    for c, c_ind in enumerate(chan_ind):
                        # thresh = dict(start=0, step=0.1)
                        pval = 0.2  # arbitrary
                        df = len(correct) + len(incorrect) - 2  # degrees of freedom denominator
                        thresh = scipy.stats.t.ppf(1 - pval/2, df=df)  # t distribution two-tailed
                        print(thresh)
                        F_obs, clusters, cluster_pv, H0 = permutation_cluster_test([spectrum[correct, c, :], spectrum[incorrect, c, :]], stat_fun=unpaired_t_test, threshold=thresh,
                                                                                         n_permutations=10000, tail=0,
                                                                                         adjacency=None, n_jobs=-1)

                        if len(cluster_pv > 0):
                            p_value_lists[cond].append(np.array(cluster_pv))  # to save for fdr correction
                        else:
                            p_value_lists[cond].append(np.array([1.]))
                        subject_lists[cond].append(subject_name)
                        channel_lists[cond].append(c_ind)

                        if any(cluster_pv <= 0.05):
                            # save plot
                            # Create a mask for significant clusters
                            significant_mask = np.zeros_like(F_obs, dtype=bool)
                            print(sorted(cluster_pv))
                            for j, cl in enumerate(clusters):
                                if cluster_pv[j] <= 0.05:
                                    significant_mask[cl] = True

                            fig = plt.figure()
                            freqs = spectrum_obj['freqs']
                            plt.plot(freqs, t_values[c], lw=1)
                            # plt.xscale('log')
                            plt.xlabel('Frequency (Hz)')
                            if cond == 3:
                                plt.ylabel('Memorized - novel (t-value)')
                            else:
                                plt.ylabel('Correct - incorrect (t-value)')
                            plt.fill_between(freqs, t_values[c], where=significant_mask, color='red', alpha=0.5, lw=0)
                            plt.gca().axhline(0, ls='--', color='k')
                            plt.title(spectrum_obj['ch_parcellations'][c_ind])

                            print(c_ind)
                            # plt.show()
                            fig.savefig(os.path.join(save_path, f'signif_figures_{conditions[cond]}', subject_name + f'_channel_{c_ind}.svg'))

    if subject_subset is None:  # only save with all subjects
        encoding_pval = {'subject': subject_lists[0], 'channel': channel_lists[0], 'pval': p_value_lists[0]}
        with open(os.path.join(save_path, "encoding_pval.pkl"), 'wb') as f:
            pickle.dump(encoding_pval, f)
        novelty_pval = {'subject': subject_lists[1], 'channel': channel_lists[1], 'pval': p_value_lists[1]}
        with open(os.path.join(save_path, "novelty_pval.pkl"), 'wb') as f:
            pickle.dump(novelty_pval, f)
        retrieval_pval = {'subject': subject_lists[2], 'channel': channel_lists[2], 'pval': p_value_lists[2]}
        with open(os.path.join(save_path, "retrieval_pval.pkl"), 'wb') as f:
            pickle.dump(retrieval_pval, f)
        memory_pval = {'subject': subject_lists[3], 'channel': channel_lists[3], 'pval': p_value_lists[3]}
        with open(os.path.join(save_path, "memory_pval.pkl"), 'wb') as f:
            pickle.dump(memory_pval, f)

    return


def fdr_correction_electrode_spectrum(data_path, behavior_path):
    behavior_all = pd.read_csv(behavior_path)

    p_obj_encoding = np.load(os.path.join(data_path, 'encoding_pval.pkl'), allow_pickle=True)
    p_obj_novelty = np.load(os.path.join(data_path, 'novelty_pval.pkl'), allow_pickle=True)
    p_obj_retrieval = np.load(os.path.join(data_path, 'retrieval_pval.pkl'), allow_pickle=True)
    p_obj_memory = np.load(os.path.join(data_path, 'memory_pval.pkl'), allow_pickle=True)

    save_path = data_path
    os.makedirs(os.path.join(save_path, 'fdr_figures_encoding'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'fdr_figures_novelty'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'fdr_figures_retrieval'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'fdr_figures_memory'), exist_ok=True)

    conditions = ['encoding', 'novelty', 'retrieval', 'memory']

    for i, subject_name in enumerate(tqdm(np.unique(p_obj_encoding['subject']))):
        print(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        spectrum_obj = np.load(os.path.join(data_path, f'{subject_name}_epochs_spectrum.pkl'), allow_pickle=True)
        spectrum = spectrum_obj['spectrum']
        print(np.array(spectrum_obj['ch_names']))

        correct_incorrect_lists = [[correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]
        for cond, p_obj in enumerate([p_obj_encoding, p_obj_novelty, p_obj_retrieval, p_obj_memory]):
            # correct pval
            pval = p_obj['pval']
            pval = np.array([np.min(pv) for pv in pval])
            _, pval = mne.stats.fdr_correction(pval, alpha=0.05, method='indep')
            pval_subject = pval[np.array(p_obj['subject']) == subject_name]

            correct = [e for e in range(len(spectrum_obj['stim_name'])) if spectrum_obj['stim_name'][e] in correct_incorrect_lists[cond][0]]
            incorrect = [e for e in range(len(spectrum_obj['stim_name'])) if spectrum_obj['stim_name'][e] in correct_incorrect_lists[cond][1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                assert spectrum.shape[1] == len(pval_subject)

                pooled_std = np.sqrt(((len(correct) - 1) * np.std(spectrum[correct, :, :], axis=0, ddof=1) ** 2 + (
                            len(incorrect) - 1) * np.std(spectrum[incorrect, :, :], axis=0, ddof=1) ** 2) / (
                                                 len(correct) + len(incorrect) - 2))
                t_values = (np.mean(spectrum[correct, :, :], axis=0) - np.mean(spectrum[incorrect, :, :], axis=0)) / (
                            pooled_std * np.sqrt(1 / len(correct) + 1 / len(incorrect)))
                for c in range(spectrum.shape[1]):
                    if pval_subject[c] < 0.05:
                        # thresh = dict(start=0, step=0.1)
                        pval = 0.2  # arbitrary
                        df = len(correct) + len(incorrect) - 2  # degrees of freedom denominator
                        thresh = scipy.stats.t.ppf(1 - pval/2, df=df)  # t distribution two-tailed
                        print(thresh)
                        F_obs, clusters, cluster_pv, H0 = permutation_cluster_test([spectrum[correct, c, :], spectrum[incorrect, c, :]], stat_fun=unpaired_t_test, threshold=thresh,
                                                                                         n_permutations=10000, tail=0,
                                                                                         adjacency=None, n_jobs=-1)

                        # save plot
                        # Create a mask for significant clusters
                        significant_mask = np.zeros_like(F_obs, dtype=bool)
                        print(sorted(cluster_pv))
                        for j, cl in enumerate(clusters):
                            if cluster_pv[j] <= 0.05:
                                significant_mask[cl] = True

                        fig = plt.figure()
                        freqs = spectrum_obj['freqs']
                        plt.plot(freqs, t_values[c], lw=1)
                        # plt.xscale('log')
                        plt.xlabel('Frequency (Hz)')
                        if cond == 3:
                            plt.ylabel('Memorized - novel (t-value)')
                        else:
                            plt.ylabel('Correct - incorrect (t-value)')
                        plt.fill_between(freqs, t_values[c], where=significant_mask, color='red', alpha=0.5, lw=0)
                        plt.gca().axhline(0, ls='--', color='k')
                        plt.title(spectrum_obj['ch_parcellations'][c])

                        fig.savefig(os.path.join(save_path, f'fdr_figures_{conditions[cond]}', subject_name + f'_channel_{c}.svg'))

    return


def memorized_vs_novel_paired_electrode_spectrum(data_path, behavior_path, channels=None, subject_subset=None):
    """paired test on epochs"""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*epochs_spectrum.pkl')))
    if subject_subset is not None:
        subject_paths = np.array(subject_paths)[subject_subset]
    print(subject_paths)

    save_path = data_path
    os.makedirs(os.path.join(save_path, 'paired_figures_memory'), exist_ok=True)

    p_value_lists = []
    subject_lists = []
    channel_lists = []

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        memorized_stim_identity = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_identity'].tolist()
        novel_identity = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_identity'].tolist()

        # keep only stim that were correct in both rep 1 and 2 and sort them to align them
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1) & (behavior['stim_identity'].isin(novel_identity))].sort_values('stim_identity')['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1) & (behavior['stim_identity'].isin(memorized_stim_identity))].sort_values('stim_identity')['stim_name'].tolist()

        assert len(memorized) == len(novel)
        assert [mem[14:] for mem in memorized] == [nov[14:] for nov in novel]

        spectrum_obj = np.load(subject_path, allow_pickle=True)
        spectrum = spectrum_obj['spectrum']
        if channels is not None:
            chan_ind = []
            for p, parcel in enumerate(spectrum_obj['ch_parcellations']):
                if any(ch_parcel in channels for ch_parcel in parcel):
                    chan_ind.append(p)
            spectrum = spectrum[:, chan_ind, :]
            print(np.array(spectrum_obj['ch_names'])[chan_ind])
        if len(chan_ind) > 0:
            memorized = [spectrum_obj['stim_name'].index(stim) for stim in memorized]
            novel = [spectrum_obj['stim_name'].index(stim) for stim in novel]
            if (len(memorized) > 0) & (len(novel) > 0):
                t_values = np.mean(spectrum[memorized, :, :] - spectrum[novel, :, :], axis=0) / (np.std(spectrum[memorized, :, :] - spectrum[novel, :, :], axis=0, ddof=1) / np.sqrt(len(memorized)))
                for c, c_ind in enumerate(chan_ind):
                    # thresh = dict(start=0, step=0.2)
                    pval = 0.2  # arbitrary
                    df = len(memorized) - 1  # degrees of freedom denominator
                    thresh = scipy.stats.t.ppf(1 - pval/2, df=df)  # t distribution two-tailed
                    print(thresh)

                    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(spectrum[memorized, c, :] - spectrum[novel, c, :], threshold=thresh,
                                                                                     n_permutations=10000, tail=0,
                                                                                     adjacency=None, n_jobs=-1)

                    if len(cluster_pv > 0):
                        p_value_lists.append(np.array(cluster_pv))  # to save for fdr correction
                        print(sorted(cluster_pv))
                    else:
                        p_value_lists.append(np.array([1.]))
                    subject_lists.append(subject_name)
                    channel_lists.append(c_ind)

                    # if any(cluster_pv <= 0.05):
                    #     # save plot
                    #     # Create a mask for significant clusters
                    #     significant_mask = np.zeros_like(t_obs, dtype=bool)
                    #     print(sorted(cluster_pv))
                    #     for j, cl in enumerate(clusters):
                    #         if cluster_pv[j] <= 0.05:
                    #             significant_mask[cl] = True
                    #
                    #     fig = plt.figure()
                    #     freqs = spectrum_obj['freqs']
                    #     plt.plot(freqs, t_values[c], lw=1)
                    #     # plt.xscale('log')
                    #     plt.xlabel('Frequency (Hz)')
                    #     plt.ylabel('Memorized - novel (t-value)')
                    #     plt.fill_between(freqs, t_values[c], where=significant_mask, color='red', alpha=0.5, lw=0)
                    #     plt.gca().axhline(0, ls='--', color='k')
                    #     plt.title(spectrum_obj['ch_parcellations'][c_ind])
                    #
                    #     print(c_ind)
                    #     # plt.show()
                    #     fig.savefig(os.path.join(save_path, 'paired_figures_memory', subject_name + f'_channel_{c_ind}.svg'))

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        print(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        memorized_stim_identity = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_identity'].tolist()
        novel_identity = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_identity'].tolist()

        # keep only stim that were correct in both rep 1 and 2 and sort them to align them
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1) & (behavior['stim_identity'].isin(novel_identity))].sort_values('stim_identity')['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1) & (behavior['stim_identity'].isin(memorized_stim_identity))].sort_values('stim_identity')['stim_name'].tolist()

        assert len(memorized) == len(novel)
        assert [mem[14:] for mem in memorized] == [nov[14:] for nov in novel]

        spectrum_obj = np.load(subject_path, allow_pickle=True)
        spectrum = spectrum_obj['spectrum']
        if channels is not None:
            chan_ind = []
            for p, parcel in enumerate(spectrum_obj['ch_parcellations']):
                if any(ch_parcel in channels for ch_parcel in parcel):
                    chan_ind.append(p)
            spectrum = spectrum[:, chan_ind, :]
            print(np.array(spectrum_obj['ch_names'])[chan_ind])
        if len(chan_ind) > 0:
            # correct pval
            pval = p_value_lists
            pval = np.array([np.min(pv) for pv in pval])
            _, pval = mne.stats.fdr_correction(pval, alpha=0.05, method='indep')
            pval_subject = pval[np.array(subject_lists) == subject_name]

            memorized = [spectrum_obj['stim_name'].index(stim) for stim in memorized]
            novel = [spectrum_obj['stim_name'].index(stim) for stim in novel]
            if (len(memorized) > 0) & (len(novel) > 0):
                assert spectrum.shape[1] == len(pval_subject)

                t_values = np.mean(spectrum[memorized, :, :] - spectrum[novel, :, :], axis=0) / (np.std(spectrum[memorized, :, :] - spectrum[novel, :, :], axis=0, ddof=1) / np.sqrt(len(memorized)))
                print(sorted(pval_subject))
                for c in range(spectrum.shape[1]):
                    if pval_subject[c] < 0.05:
                        # thresh = dict(start=0, step=0.2)
                        pval = 0.2  # arbitrary
                        df = len(memorized) - 1  # degrees of freedom denominator
                        thresh = scipy.stats.t.ppf(1 - pval / 2, df=df)  # t distribution two-tailed
                        print(thresh)

                        t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(spectrum[memorized, c, :] - spectrum[novel, c, :], threshold=thresh,
                            n_permutations=10000, tail=0,
                            adjacency=None, n_jobs=-1)

                        # save plot
                        # Create a mask for significant clusters
                        significant_mask = np.zeros_like(t_obs, dtype=bool)
                        print(sorted(cluster_pv))
                        for j, cl in enumerate(clusters):
                            if cluster_pv[j] <= 0.05:
                                significant_mask[cl] = True

                        fig = plt.figure()
                        freqs = spectrum_obj['freqs']
                        plt.plot(freqs, t_values[c], lw=1)
                        # plt.xscale('log')
                        plt.xlabel('Frequency (Hz)')
                        plt.ylabel('Memorized - novel (t-value)')
                        plt.fill_between(freqs, t_values[c], where=significant_mask, color='red', alpha=0.5, lw=0)
                        plt.gca().axhline(0, ls='--', color='k')
                        plt.title(spectrum_obj['ch_parcellations'][c])

                        fig.savefig(os.path.join(save_path, 'paired_figures_memory', subject_name + f'_channel_{c}.svg'))

    return


def plot_electrode_comodulogram(data_path, behavior_path, channels=None, subject_subset=None):
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*epochs_comodulogram.pkl')))
    if subject_subset is not None:
        subject_paths = np.array(subject_paths)[subject_subset]
    print(subject_paths)

    save_path = data_path
    os.makedirs(os.path.join(save_path, 'signif_figures_encoding'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'signif_figures_novelty'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'signif_figures_retrieval'), exist_ok=True)

    signif_channels_encoding = {}
    signif_channels_novelty = {}
    signif_channels_retrieval = {}
    conditions = ['encoding', 'novelty', 'retrieval']
    lowest_p_value_lists = [[], [], []]
    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()

        pac_obj = np.load(subject_path, allow_pickle=True)
        pac = pac_obj['pac']
        if channels is not None:
            chan_ind = []
            for p, parcel in enumerate(pac_obj['ch_parcellations']):
                if any(ch_parcel in channels for ch_parcel in parcel):
                    chan_ind.append(p)
            pac = pac[:, chan_ind, :, :]
            print(np.array(pac_obj['ch_names'])[chan_ind])
        signif_channels_lists = [[], [], []]
        if len(chan_ind) > 0:
            for cond, correct_incorrect in enumerate([[correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
                correct = [e for e in range(len(pac_obj['stim_name'])) if pac_obj['stim_name'][e] in correct_incorrect[0]]
                incorrect = [e for e in range(len(pac_obj['stim_name'])) if pac_obj['stim_name'][e] in correct_incorrect[1]]
                if (len(correct) > 0) & (len(incorrect) > 0):
                    pooled_std = np.sqrt(((len(correct) - 1) * np.std(pac[correct, :, :, :], axis=0, ddof=1) ** 2 + (
                                len(incorrect) - 1) * np.std(pac[incorrect, :, :, :], axis=0, ddof=1) ** 2) / (
                                                     len(correct) + len(incorrect) - 2))
                    t_values = (np.mean(pac[correct, :, :, :], axis=0) - np.mean(pac[incorrect, :, :, :], axis=0)) / (
                                pooled_std * np.sqrt(1 / len(correct) + 1 / len(incorrect)))
                    for c, c_ind in enumerate(chan_ind):
                        # thresh = dict(start=0, step=0.3)
                        pval = 0.1  # arbitrary
                        df = len(correct) + len(incorrect) - 2  # degrees of freedom denominator
                        thresh = scipy.stats.t.ppf(1 - pval/2, df=df)  # t distribution two-tailed
                        print(thresh)
                        F_obs, clusters, cluster_pv, H0 = permutation_cluster_test([pac[correct, c, :, :], pac[incorrect, c, :, :]], stat_fun=unpaired_t_test, threshold=thresh,
                                                                                         n_permutations=10000, tail=0,
                                                                                         adjacency=None, n_jobs=-1)

                        # # Create a mask for significant clusters
                        # significant_mask = np.zeros_like(F_obs, dtype=bool)
                        # print(sorted(cluster_pv))
                        # for j, cl in enumerate(clusters):
                        #     if cluster_pv[j] <= 0.05:
                        #         significant_mask[cl] = True
                        #
                        # fig = plt.figure()
                        # vmax = 4
                        # X, Y = np.meshgrid(pac_obj['low_freqs'], pac_obj['high_freqs'])
                        # plt.pcolormesh(X, Y, t_values[c].T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                        # # plt.xscale('log')
                        # plt.yscale('log')
                        # plt.gca().yaxis.set_major_formatter(ScalarFormatter())
                        # plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
                        # plt.xlabel('Driver frequency (Hz)')
                        # plt.ylabel('Signal frequency (Hz)')
                        # cbar = plt.colorbar()
                        # cbar.set_label('t-values', rotation=270)
                        # plt.contour(X, Y, significant_mask.T, colors='k', linewidths=1)
                        # plt.title(pac_obj['ch_parcellations'][c_ind])
                        #
                        # print(c_ind)
                        # plt.show()

                        if len(cluster_pv > 0):
                            lowest_p_value_lists[cond].append(np.array(cluster_pv))  # to save for fdr correction
                        else:
                            lowest_p_value_lists[cond].append(np.array([]))

                        if any(cluster_pv <= 0.05):
                            signif_channels_lists[cond].append(c_ind)

                            # save plot
                            # Create a mask for significant clusters
                            significant_mask = np.zeros_like(F_obs, dtype=bool)
                            print(sorted(cluster_pv))
                            for j, cl in enumerate(clusters):
                                if cluster_pv[j] <= 0.05:
                                    significant_mask[cl] = True

                            fig = plt.figure()
                            vmax = 4
                            X, Y = np.meshgrid(pac_obj['low_freqs'], pac_obj['high_freqs'])
                            plt.pcolormesh(X, Y, t_values[c].T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                            # plt.xscale('log')
                            plt.yscale('log')
                            plt.gca().yaxis.set_major_formatter(ScalarFormatter())
                            plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
                            plt.xlabel('Driver frequency (Hz)')
                            plt.ylabel('Signal frequency (Hz)')
                            cbar = plt.colorbar()
                            cbar.set_label('t-values', rotation=270)
                            plt.contour(X, Y, significant_mask.T, colors='k', linewidths=1)
                            plt.title(pac_obj['ch_parcellations'][c_ind])

                            print(c_ind)
                            # plt.show()
                            fig.savefig(os.path.join(save_path, f'signif_figures_{conditions[cond]}', subject_name + f'_channel_{c_ind}.svg'))

        signif_channels_encoding[subject_name] = signif_channels_lists[0]
        signif_channels_novelty[subject_name] = signif_channels_lists[1]
        signif_channels_retrieval[subject_name] = signif_channels_lists[2]

    if subject_subset is None:  # only save with all subjects
        with open(os.path.join(save_path, "signif_channels_encoding.pkl"), 'wb') as f:
            pickle.dump(signif_channels_encoding, f)
        with open(os.path.join(save_path, "signif_channels_novelty.pkl"), 'wb') as f:
            pickle.dump(signif_channels_novelty, f)
        with open(os.path.join(save_path, "signif_channels_retrieval.pkl"), 'wb') as f:
            pickle.dump(signif_channels_retrieval, f)
        np.save(os.path.join(save_path, "lowest_p_values_encoding.npy"), lowest_p_value_lists[0])
        np.save(os.path.join(save_path, "lowest_p_values_novelty.npy"), lowest_p_value_lists[1])
        np.save(os.path.join(save_path, "lowest_p_values_retrieval.npy"), lowest_p_value_lists[2])

    return


def stat_test_electrode_comodulogram(data_path, behavior_path, subject_subset=None):
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*epochs_comodulogram.pkl')))
    if subject_subset is not None:
        subject_paths = np.array(subject_paths)[subject_subset]
    print(subject_paths)

    save_path = data_path

    lowest_p_value_dict = {}
    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        pac_obj = np.load(subject_path, allow_pickle=True)
        pac = pac_obj['pac']

        lowest_p_value_lists = [[], [], [], [], []]
        for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
            correct = [e for e in range(len(pac_obj['stim_name'])) if pac_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(pac_obj['stim_name'])) if pac_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c, c_ind in enumerate(range(pac.shape[1])):
                    # thresh = dict(start=0, step=0.3)
                    pval = 0.1  # arbitrary
                    df = len(correct) + len(incorrect) - 2  # degrees of freedom denominator
                    thresh = scipy.stats.t.ppf(1 - pval/2, df=df)  # t distribution two-tailed
                    print(thresh)
                    F_obs, clusters, cluster_pv, H0 = permutation_cluster_test([pac[correct, c, :, :], pac[incorrect, c, :, :]], stat_fun=unpaired_t_test, threshold=thresh,
                                                                                     n_permutations=10000, tail=0,
                                                                                     adjacency=None, n_jobs=-1)

                    if len(cluster_pv > 0):
                        lowest_p_value_lists[cond].append(np.min(cluster_pv))  # to save for fdr correction
                    else:
                        lowest_p_value_lists[cond].append(1.)

        lowest_p_value_dict[subject_name] = lowest_p_value_lists

    if subject_subset is None:  # only save with all subjects
        with open(os.path.join(save_path, "lowest_p_all_channels.pkl"), 'wb') as f:
            pickle.dump(lowest_p_value_dict, f)

    return


def stat_test_electrode_rms(data_path, behavior_path, subject_subset=None):
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*epochs_rms.pkl')))
    if subject_subset is not None:
        subject_paths = np.array(subject_paths)[subject_subset]
    print(subject_paths)

    save_path = data_path

    lowest_p_value_dict = {}
    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        print(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']

        lowest_p_value_lists = [[], [], [], [], []]
        for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c, c_ind in enumerate(range(rms.shape[1])):
                    p_value = ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=True, permutations=None, random_state=None, alternative='two-sided').pvalue
                    lowest_p_value_lists[cond].append(np.min(p_value))  # to save for fdr correction

        lowest_p_value_dict[subject_name] = lowest_p_value_lists

    if subject_subset is None:  # only save with all subjects
        with open(os.path.join(save_path, "lowest_p_all_channels.pkl"), 'wb') as f:
            pickle.dump(lowest_p_value_dict, f)

    return

def count_channels(data_path, channels=None):
    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*epochs_comodulogram.pkl')))

    n = 0
    for i, subject_path in enumerate(subject_paths):
        pac_obj = np.load(subject_path, allow_pickle=True)
        if channels is not None:
            chan_ind = []
            for p, parcel in enumerate(pac_obj['ch_parcellations']):
                if any(ch_parcel in channels for ch_parcel in parcel):
                    chan_ind.append(p)
            n += len(chan_ind)
    print(n)


def print_all_regions(data_path, fs_subjects_dir):
    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*.fif')))

    channel_parcellations = []
    for i, subject_path in enumerate(subject_paths):
        epochs = mne.read_epochs(epochs_path)

        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=fs_subjects_dir,
                                                  aseg='aparc+aseg', dist=2)
        for key, value in labels.items():
            # channel_parcellations.append(value[0])
            channel_parcellations += value
    channel_parcellations = np.unique(channel_parcellations)
    print(channel_parcellations)


def plot_ratio_signif_eletrodes(data_path, lowest_p_path, first_parcel_label=True, regions=None, analysis_name='comodulogram', n_permut=10000):
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))

    lowest_p_value_dict = np.load(lowest_p_path, allow_pickle=True)
    conditions = ['Correct-incorrect', 'Encoding', 'Novelty detection', 'Retrieval', 'Memorized-Novel']
    # conditions = ['Correct-incorrect']
    regions_to_remove = ['Unknown', 'Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter', 'ctx-rh-unknown', 'ctx-lh-unknown', 'CSF']
    # regions_to_remove = []

    # find all unique regions
    unique_parcellations = []
    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        pac_obj = np.load(os.path.join(data_path, subject_name + f"_epochs_{analysis_name}.pkl"), allow_pickle=True)
        parcellations = pac_obj['ch_parcellations']
        if first_parcel_label:
            parcellations = [[parcel[0]] for parcel in parcellations]
        for parcel in parcellations:
            unique_parcellations += parcel
    unique_parcellations = np.unique(unique_parcellations)
    unique_parcellations = unique_parcellations[np.isin(unique_parcellations, regions_to_remove, invert=True)]
    print(unique_parcellations)
    print('Number of regions:', len(unique_parcellations))

    # # select regions, erasing the previous selection based on regions_to_remove
    # if regions is not None:
    #     unique_parcellations = np.array(regions)
    #     print(len(np.unique(unique_parcellations)))
    #     print(len(regions))
    #     assert len(np.unique(unique_parcellations)) == len(regions)
    #     print('Manual region selections')
    #     print(unique_parcellations)
    #     print('Number of regions:', len(unique_parcellations))

    n_signif_channels = [np.zeros_like(unique_parcellations, dtype=int) for c in range(len(conditions))]
    n_total_channels = [np.zeros_like(unique_parcellations, dtype=int) for c in range(len(conditions))]
    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_p = lowest_p_value_dict[subject_name]
        pac_obj = np.load(os.path.join(data_path, subject_name + f"_epochs_{analysis_name}.pkl"), allow_pickle=True)
        parcellations = pac_obj['ch_parcellations']
        if first_parcel_label:
            parcellations = [[parcel[0]] for parcel in parcellations]
        for c in range(len(conditions)):
            if len(subject_p[c]) > 0:
                assert len(subject_p[c]) == len(parcellations)
                for p, parcel in enumerate(parcellations):
                    for region in parcel:
                        if region in unique_parcellations:
                            n_total_channels[c][np.where(unique_parcellations == region)[0][0]] += 1
                            if subject_p[c][p] < 0.05:
                                n_signif_channels[c][np.where(unique_parcellations == region)[0][0]] += 1
    signif_channels_ratios = [n_signif_channels[c] / n_total_channels[c] for c in range(len(n_total_channels))]
    print(signif_channels_ratios)

    # stat test: we permute the significant channels, keeping the same region sizes. We obtain a random distribution of ratios for each region, and keep significant ratios
    ratios_distributions = [np.zeros((n_permut+1, unique_parcellations.shape[0])) for c in range(len(conditions))]  # +1 for true distribution
    for c in range(len(ratios_distributions)):
        ratios_distributions[c][-1, :] = signif_channels_ratios[c]

    lowest_p_value_dict_copy = copy.deepcopy(lowest_p_value_dict)
    all_subj_p_values = [[] for c in range(len(conditions))]
    all_subj_parcellations = [[] for c in range(len(conditions))]
    for i, subject_path in enumerate(subject_paths):
        subject_name = Path(subject_path).name[:6]
        pac_obj = np.load(os.path.join(data_path, subject_name + f"_epochs_{analysis_name}.pkl"), allow_pickle=True)
        parcellations = pac_obj['ch_parcellations']
        if first_parcel_label:
            parcellations = [[parcel[0]] for parcel in parcellations]
        for c in range(len(conditions)):
            if len(lowest_p_value_dict_copy[subject_name][c]) > 0:
                assert len(lowest_p_value_dict_copy[subject_name][c]) == len(parcellations)
                for p, parcel in enumerate(parcellations):
                    if any([region in unique_parcellations for region in parcel]):
                        all_subj_p_values[c].append(lowest_p_value_dict_copy[subject_name][c][p])
                        all_subj_parcellations[c].append(parcel)

    def multiprocess_fct(perm):
        rng = np.random.default_rng()
        n_signif_channels = [np.zeros_like(unique_parcellations, dtype=int) for c in range(len(conditions))]

        # shuffle across electrodes from all subjects
        for c in range(len(conditions)):
            rng.shuffle(all_subj_p_values[c])

        for c in range(len(conditions)):
            for p, parcel in enumerate(all_subj_parcellations[c]):
                for region in parcel:
                    if region in unique_parcellations:
                        if all_subj_p_values[c][p] < 0.05:
                            n_signif_channels[c][np.where(unique_parcellations == region)[0][0]] += 1

        return [n_signif_channels[c] / n_total_channels[c] for c in range(len(n_total_channels))]

    #version shuffle within subjects
    # def multiprocess_fct(perm):
        # rng = np.random.default_rng()
        # n_signif_channels = [np.zeros_like(unique_parcellations, dtype=int) for c in range(len(conditions))]
        # lowest_p_value_dict_copy = copy.deepcopy(lowest_p_value_dict)
        # for i, subject_path in enumerate(subject_paths):
        #     subject_name = Path(subject_path).name[:6]
        #     subject_p = lowest_p_value_dict_copy[subject_name]
        #     for c in range(len(subject_p)):
        #         rng.shuffle(subject_p[c])
        #     pac_obj = np.load(os.path.join(save_path, subject_name + "_epochs_comodulogram.pkl"), allow_pickle=True)
        #     parcellations = pac_obj['ch_parcellations']
        #     for c in range(len(conditions)):
        #         if len(subject_p[c]) > 0:
        #             assert len(subject_p[c]) == len(parcellations)
        #             for p, parcel in enumerate(parcellations):
        #                 for region in parcel:
        #                     if region not in regions_to_remove:
        #                         if subject_p[c][p] < 0.05:
        #                             n_signif_channels[c][np.where(unique_parcellations == region)[0][0]] += 1
        #
        # return [n_signif_channels[c] / n_total_channels[c] for c in range(len(n_total_channels))]

    perm_signif_ratios = Parallel(n_jobs=-1)(delayed(multiprocess_fct)(perm) for perm in tqdm(range(n_permut)))
    for perm in range(n_permut):
        for c in range(len(ratios_distributions)):
            ratios_distributions[c][perm, :] = perm_signif_ratios[perm][c]

    # select channels for visu and fdr correction
    if regions is not None:
        for c in range(len(n_total_channels)):
            for chan, n_electrodes in enumerate(n_total_channels[c]):
                if n_electrodes < 5:
                    if unique_parcellations[chan] in regions:
                        regions.remove(unique_parcellations[chan])
        signif_channels_ratios = [signif_channels_ratios[c][np.isin(unique_parcellations, regions, invert=False)] for c in range(len(signif_channels_ratios))]
        n_total_channels = [n_total_channels[c][np.isin(unique_parcellations, regions, invert=False)] for c in range(len(n_total_channels))]
        ratios_distributions = [ratios_distributions[c][:, np.isin(unique_parcellations, regions, invert=False)] for c in range(len(ratios_distributions))]
        unique_parcellations = unique_parcellations[np.isin(unique_parcellations, regions, invert=False)]
    else:
        # regions_to_remove = ['Unknown', 'Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter', 'ctx-rh-unknown', 'ctx-lh-unknown', 'CSF']
        regions_to_remove = []
        # add regions with less than 5 electrodes
        for c in range(len(n_total_channels)):
            print(conditions[c])
            for chan, n_electrodes in enumerate(n_total_channels[c]):
                print(unique_parcellations[chan])
                print(n_electrodes)
                if n_electrodes < 5:
                    regions_to_remove.append(unique_parcellations[chan])
        regions_to_remove = np.unique(regions_to_remove)
        signif_channels_ratios = [signif_channels_ratios[c][np.isin(unique_parcellations, regions_to_remove, invert=True)] for c in range(len(signif_channels_ratios))]
        n_total_channels = [n_total_channels[c][np.isin(unique_parcellations, regions_to_remove, invert=True)] for c in range(len(n_total_channels))]
        ratios_distributions = [ratios_distributions[c][:, np.isin(unique_parcellations, regions_to_remove, invert=True)] for c in range(len(ratios_distributions))]
        unique_parcellations = unique_parcellations[np.isin(unique_parcellations, regions_to_remove, invert=True)]

    print(ratios_distributions[0].shape)
    signif_regions_list = []
    for c in range(len(ratios_distributions)):
        signif_regions = np.mean(ratios_distributions[c][:, :] > ratios_distributions[c][-1, :], axis=0)
        # _, signif_regions = mne.stats.fdr_correction(signif_regions, alpha=0.05, method='indep')
        print(signif_regions)
        signif_regions = signif_regions < 0.05
        signif_regions_list.append(signif_regions)

    # compute number of electrodes for "error bars"
    channel_ratio = [n_total_channels[c] / np.sum(n_total_channels[c]) for c in range(len(n_total_channels))]  # ratio of number of electrodes

    fig, axes = plt.subplots(1, len(conditions))
    for c, cond in enumerate(conditions):
        print(f'Significant regions {cond}:')
        print(unique_parcellations[signif_regions_list[c]])
        cond_parcellation = unique_parcellations[np.argsort(signif_channels_ratios[c])]
        cond_channel_ratio = channel_ratio[c][np.argsort(signif_channels_ratios[c])]
        cond_signif_regions = signif_regions_list[c][np.argsort(signif_channels_ratios[c])]
        colors = ['red' if signif else 'gray' for signif in cond_signif_regions]
        highest_ratios = np.sort(signif_channels_ratios[c])
        axes[c].barh(np.arange(len(highest_ratios)), highest_ratios, tick_label=cond_parcellation, xerr=cond_channel_ratio, color=colors)
        axes[c].set_xlim(0, 1)
        axes[c].set_xlabel('Ratio of significant electrodes')
        axes[c].set_title(cond)
    fig.suptitle(f'{analysis_name}: ratio of significant channels')
    # fig.subplots_adjust(wspace=0.7, hspace=None)
    plt.show()

    return


def plot_significant_electrode_comodulogram(data_path, behavior_path, lowest_p_path, first_parcel_label=True, regions_list=None):
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*epochs_comodulogram.pkl')))
    print(subject_paths)

    lowest_p_value_dict = np.load(lowest_p_path, allow_pickle=True)

    save_path = data_path
    os.makedirs(os.path.join(save_path, 'signif_figures_correct_incorrect'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'signif_figures_encoding'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'signif_figures_novelty'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'signif_figures_retrieval'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'signif_figures_memory'), exist_ok=True)

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']

    # Initialize storage for average t-values across all significant channels for each condition and region

    avg_t_values = [{region: [] for region in regions_list[c]} for c in range(len(conditions))]
    signif_counts = [{region: 0 for region in regions_list[c]} for c in range(len(conditions))]

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        pac_obj = np.load(subject_path, allow_pickle=True)
        pac = pac_obj['pac']
        parcellations = pac_obj['ch_parcellations']
        if first_parcel_label:
            parcellations = [[parcel[0]] for parcel in parcellations]

        subject_p = lowest_p_value_dict[subject_name]

        for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
            if len(subject_p[cond]) > 0:
                assert len(subject_p[cond]) == len(parcellations)

                chan_ind = []
                for p, parcel in enumerate(parcellations):
                    if any(ch_parcel in regions_list[cond] for ch_parcel in parcel):
                        chan_ind.append(p)
                pac_cond = pac[:, chan_ind, :, :]
                print(np.array(pac_obj['ch_names'])[chan_ind])

                correct = [e for e in range(len(pac_obj['stim_name'])) if pac_obj['stim_name'][e] in correct_incorrect[0]]
                incorrect = [e for e in range(len(pac_obj['stim_name'])) if pac_obj['stim_name'][e] in correct_incorrect[1]]
                if (len(correct) > 0) & (len(incorrect) > 0):
                    pooled_std = np.sqrt(((len(correct) - 1) * np.std(pac_cond[correct, :, :, :], axis=0, ddof=1) ** 2 + (len(incorrect) - 1) * np.std(pac_cond[incorrect, :, :, :], axis=0, ddof=1) ** 2) / (len(correct) + len(incorrect) - 2))
                    t_values = (np.mean(pac_cond[correct, :, :, :], axis=0) - np.mean(pac_cond[incorrect, :, :, :], axis=0)) / (pooled_std * np.sqrt(1 / len(correct) + 1 / len(incorrect)))
                    for c, c_ind in enumerate(chan_ind):
                        if subject_p[cond][c_ind] < 0.05:
                            # thresh = dict(start=0, step=0.3)
                            pval = 0.05  # arbitrary
                            df = len(correct) + len(incorrect) - 2  # degrees of freedom denominator
                            thresh = scipy.stats.t.ppf(1 - pval / 2, df=df)  # t distribution two-tailed
                            print(thresh)
                            F_obs, clusters, cluster_pv, H0 = permutation_cluster_test(
                                [pac_cond[correct, c, :, :], pac_cond[incorrect, c, :, :]], stat_fun=unpaired_t_test,
                                threshold=thresh,
                                n_permutations=10000, tail=0,
                                adjacency=None, n_jobs=-1)

                            # save plot
                            # Create a mask for significant clusters
                            significant_mask = np.zeros_like(F_obs, dtype=bool)
                            print(sorted(cluster_pv))
                            for j, cl in enumerate(clusters):
                                if cluster_pv[j] < 0.05:
                                    significant_mask[cl] = True

                            # Accumulate the significant t-values by region
                            for region in regions_list[cond]:
                                if any(ch_parcel in region for ch_parcel in parcellations[c_ind]):
                                    avg_t_values[cond][region].append(np.mean(pac_cond[correct, c, :, :], axis=0) - np.mean(pac_cond[incorrect, c, :, :], axis=0))
                                    signif_counts[cond][region] += 1

                            fig = plt.figure()
                            vmax = 4
                            X, Y = np.meshgrid(pac_obj['low_freqs'], pac_obj['high_freqs'])
                            plt.pcolormesh(X, Y, t_values[c].T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                            # plt.xscale('log')
                            plt.yscale('log')
                            plt.gca().yaxis.set_major_formatter(ScalarFormatter())
                            plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
                            plt.xlabel('Driver frequency (Hz)')
                            plt.ylabel('Signal frequency (Hz)')
                            cbar = plt.colorbar()
                            cbar.set_label('t-values', rotation=270)
                            plt.contour(X, Y, significant_mask.T, colors='k', linewidths=1)
                            plt.title(parcellations[c_ind])
                            # plt.show()
                            fig.savefig(os.path.join(save_path, f'signif_figures_{conditions[cond]}', subject_name + f'_channel_{c_ind}.svg'))
                            plt.close()

    # Calculate and save average t-values for each condition and region
    for cond in range(len(conditions)):
        for region in regions_list[cond]:
            if signif_counts[cond][region] > 0:
                region_pac = np.array(avg_t_values[cond][region])
                # region_pac = np.mean(region_pac, axis=0) / (np.std(region_pac, axis=0) / np.sqrt(region_pac.shape[0]))  # Calculate the t-values across channels
                region_pac = np.mean(region_pac, axis=0)

                fig = plt.figure()
                vmax = np.max(np.abs(region_pac))
                # vmax = 8
                X, Y = np.meshgrid(pac_obj['low_freqs'], pac_obj['high_freqs'])
                plt.pcolormesh(X, Y, region_pac.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                # plt.pcolormesh(X, Y, avg_t_values[cond][region].T, cmap='RdBu_r')
                plt.yscale('log')
                plt.gca().yaxis.set_major_formatter(ScalarFormatter())
                plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
                plt.xlabel('Driver frequency (Hz)')
                plt.ylabel('Signal frequency (Hz)')
                cbar = plt.colorbar()
                cbar.set_label('t-values', rotation=270)
                plt.title(f'{conditions[cond]} t-values {region}')
                fig.savefig(os.path.join(save_path, f'signif_figures_{conditions[cond]}', f'Average_{region}.svg'))

                region_pac = np.array(avg_t_values[cond][region])
                region_pac = np.mean(region_pac, axis=0) / (np.std(region_pac, axis=0) / np.sqrt(region_pac.shape[0]))  # Calculate the t-values across channels
                fig = plt.figure()
                # vmax = np.max(np.abs(region_pac))
                vmax = 8
                X, Y = np.meshgrid(pac_obj['low_freqs'], pac_obj['high_freqs'])
                plt.pcolormesh(X, Y, region_pac.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                # plt.pcolormesh(X, Y, avg_t_values[cond][region].T, cmap='RdBu_r')
                plt.yscale('log')
                plt.gca().yaxis.set_major_formatter(ScalarFormatter())
                plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
                plt.xlabel('Driver frequency (Hz)')
                plt.ylabel('Signal frequency (Hz)')
                cbar = plt.colorbar()
                cbar.set_label('t-values', rotation=270)
                plt.title(f'{conditions[cond]} t-values {region}')
                fig.savefig(os.path.join(save_path, f'signif_figures_{conditions[cond]}', f'tval_{region}.svg'))

    return


def plot_significant_electrode_rms(data_path, behavior_path, lowest_p_path, first_parcel_label=True, regions_list=None, only_signif_channels=True):
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*epochs_rms.pkl')))
    print(subject_paths)

    lowest_p_value_dict = np.load(lowest_p_path, allow_pickle=True)

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']

    # Initialize storage for average t-values across all significant channels for each condition and region

    avg_rms = [{region: [] for region in regions_list[c]} for c in range(len(conditions))]
    signif_chan = [{region: [] for region in regions_list[c]} for c in range(len(conditions))]

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label:
            parcellations = [[parcel[0]] for parcel in parcellations]

        subject_p = lowest_p_value_dict[subject_name]

        for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
            if len(subject_p[cond]) > 0:
                assert len(subject_p[cond]) == len(parcellations)

                chan_ind = []
                for p, parcel in enumerate(parcellations):
                    if any(ch_parcel in regions_list[cond] for ch_parcel in parcel):
                        chan_ind.append(p)
                rms_cond = rms[:, chan_ind]
                print(np.array(rms_obj['ch_names'])[chan_ind])

                correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
                incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
                if (len(correct) > 0) & (len(incorrect) > 0):
                    for c, c_ind in enumerate(chan_ind):
                        if not only_signif_channels or (subject_p[cond][c_ind] < 0.05):
                            # Accumulate the significant rms by region
                            for region in regions_list[cond]:
                                if any(ch_parcel in region for ch_parcel in parcellations[c_ind]):
                                    avg_rms[cond][region].append(np.mean(rms_cond[correct, c], axis=0) - np.mean(rms_cond[incorrect, c], axis=0))
                                    if subject_p[cond][c_ind] < 0.05:
                                        signif_chan[cond][region].append(True)
                                    else:
                                        signif_chan[cond][region].append(False)

    # Calculate and save average rms for each condition and region
    region_rms_per_condition = []
    error_bars_per_condition = []
    regions_per_condition = []
    for cond in range(len(conditions)):
        condition_rms = []
        condition_error_bars = []
        condition_regions = []
        for region in regions_list[cond]:
            # if signif_counts[cond][region] > 0:
            condition_regions.append(region)
            region_rms = np.array(avg_rms[cond][region])
            mean_rms = np.mean(region_rms, axis=0)
            condition_rms.append(mean_rms)
            error_rms = np.std(region_rms, axis=0) / np.sqrt(region_rms.shape[0])
            condition_error_bars.append(error_rms)
        region_rms_per_condition.append(condition_rms)
        error_bars_per_condition.append(condition_error_bars)
        regions_per_condition.append(condition_regions)

    fig, axs = plt.subplots(1, len(conditions), figsize=(20, 5), sharey=True)
    # Plot each condition's data in a separate subplot
    for i, (condition, regions, region_rms, error_bars) in enumerate(
            zip(conditions, regions_per_condition, region_rms_per_condition, error_bars_per_condition)):
        axs[i].bar(regions, region_rms, yerr=error_bars, capsize=5, color='skyblue')
        axs[i].axhline(0, color='gray', linestyle='--', linewidth=1.5)

        # Add individual sample points as scatter dots
        for j, region in enumerate(regions):
            samples = np.array(avg_rms[i][region])
            signif = signif_chan[i][region]
            colors = ['red' if sign else 'black' for sign in signif]
            # Add scatter plot with jitter on x-axis
            x_values = np.full(samples.shape, j) + np.random.uniform(-0.1, 0.1, samples.shape)  # Jitter for clarity
            axs[i].scatter(x_values, samples, color=colors, alpha=0.7, s=20, label='Samples')

        axs[i].set_title(condition)
        if i == 0:  # Only set ylabel on the first subplot
            axs[i].set_ylabel('Average RMS contrast')

        # Rotate x-axis labels for readability
        axs[i].tick_params(axis='x', rotation=45)

    # Adjust the layout so the plots don't overlap
    plt.tight_layout()
    plt.show()

    return


def plot_all_electrode_scalar(data_path, behavior_path, first_parcel_label=True, analysis_name='rms', regions=None, remove_outliers=False):
    """used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
    del subject_paths[1]
    print(subject_paths)

    # conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']
    conditions = ['novelty', 'retrieval', 'encoding']

    unique_parcellations = []
    for i, subject_path in enumerate(tqdm(subject_paths)):
        rms_obj = np.load(subject_path, allow_pickle=True)
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label==True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label=="advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        unique_parcellations += parcellations
    unique_parcellations = np.unique(unique_parcellations)
    # regions_to_remove = ['Unknown', 'Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter', 'ctx-rh-unknown', 'ctx-lh-unknown', 'CSF']
    # regions_to_remove = []
    regions_to_remove = ['Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter', 'CSF', 'Unknown']
    unique_parcellations = unique_parcellations[np.isin(unique_parcellations, regions_to_remove, invert=True)]
    unique_parcellations = sorted(unique_parcellations, key=lambda x: x[::-1], reverse=True)
    print(unique_parcellations)
    print('Number of regions:', len(unique_parcellations))

    if regions is not None:
        unique_parcellations = [region for region in unique_parcellations if region in regions]
    rms_contrasts = [{region: [] for region in unique_parcellations} for c in range(len(conditions))]

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label==True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label=="advanced":
            parcellations = select_parcel(parcellations, go_next)
        print(np.array(rms_obj['ch_names']))

        # for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [correct_encoding, incorrect_encoding]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    # Accumulate rms by region
                    for region in unique_parcellations:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            rms_contrasts[cond][region].append(np.mean(rms[correct, c], axis=0) - np.mean(rms[incorrect, c], axis=0))

    # Calculate and save average rms for each condition and region
    region_rms_per_condition = []
    error_bars_per_condition = []
    regions_per_condition = []
    pvalue_per_condition = []
    t_per_condition = []
    for cond in range(len(conditions)):
        condition_rms = []
        condition_error_bars = []
        condition_regions = []
        condition_pvalue = []
        condition_t = []
        for region in unique_parcellations:
            region_rms = np.array(rms_contrasts[cond][region])
            # region_rms = np.abs(np.array(rms_contrasts[cond][region]))
            condition_regions.append(region)
            mean_rms = np.mean(region_rms, axis=0)
            condition_rms.append(mean_rms)
            error_rms = np.std(region_rms, axis=0) / np.sqrt(region_rms.shape[0])
            condition_error_bars.append(error_rms)
            ttest = ttest_1samp(region_rms, 0)
            condition_pvalue.append(ttest.pvalue)
            condition_t.append(ttest.statistic)
        region_rms_per_condition.append(condition_rms)
        error_bars_per_condition.append(condition_error_bars)
        regions_per_condition.append(condition_regions)
        pvalue_per_condition.append(condition_pvalue)
        t_per_condition.append(condition_t)

    # FDR correction
    for c in range(len(pvalue_per_condition)):
        _, pval = mne.stats.fdr_correction(pvalue_per_condition[c], alpha=0.05, method='indep')
        pvalue_per_condition[c] = pval

    fig, axs = plt.subplots(1, len(conditions), sharey=True)
    # Plot each condition's data in a separate subplot
    for i, (condition, regions, region_rms, error_bars, pvalues, tvalues) in enumerate(zip(conditions, regions_per_condition, region_rms_per_condition, error_bars_per_condition, pvalue_per_condition, t_per_condition)):
        for r in range(len(regions)):
            print(condition)
            print(regions[r])
            print(tvalues[r])
            print(pvalues[r])
            print(len(rms_contrasts[i][regions[r]]))
        colors = ['orange' if p < 0.05 else 'skyblue' for p in pvalues]
        axs[i].barh(regions, region_rms, xerr=error_bars, capsize=5, color=colors)
        axs[i].axvline(0, color='gray', linestyle='--', linewidth=1.5)

        # Add individual sample points as scatter dots
        for j, region in enumerate(regions):
            samples = np.array(rms_contrasts[i][region])
            if remove_outliers:
                samples = samples[(np.abs(samples - samples.mean())) <= 3 * samples.std()]
            # samples = np.abs(np.array(rms_contrasts[i][region]))
            # Add scatter plot with jitter on x-axis
            y_values = np.full(samples.shape, j) + np.random.uniform(-0.1, 0.1, samples.shape)  # Jitter for clarity
            axs[i].scatter(samples, y_values, color='black', alpha=0.7, s=2, label='Samples')

        axs[i].set_title(condition)
        # if i == 0:  # Only set ylabel on the first subplot
            # axs[i].set_xlabel('Average RMS contrast')
    fig.supxlabel(f'Average contrast of {analysis_name} relative to baseline (percent)')

        # Rotate x-axis labels for readability
        # axs[i].tick_params(axis='x', rotation=90)

    # Adjust the layout so the plots don't overlap
    # plt.tight_layout()
    plt.show()

    return


def plot_all_electrode_scalar_colored_regions(data_path, behavior_path, first_parcel_label=True, analysis_name='rms', regions=None, remove_outliers=False):
    """used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
    del subject_paths[1]
    print(subject_paths)

    # conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']
    conditions = ['novelty', 'retrieval']

    unique_parcellations = []
    for i, subject_path in enumerate(tqdm(subject_paths)):
        rms_obj = np.load(subject_path, allow_pickle=True)
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label==True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label=="advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        unique_parcellations += parcellations
    unique_parcellations = np.unique(unique_parcellations)
    # regions_to_remove = ['Unknown', 'Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter', 'ctx-rh-unknown', 'ctx-lh-unknown', 'CSF']
    # regions_to_remove = []
    regions_to_remove = ['Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter', 'CSF', 'Unknown']
    unique_parcellations = unique_parcellations[np.isin(unique_parcellations, regions_to_remove, invert=True)]
    unique_parcellations = sorted(unique_parcellations, key=lambda x: x[::-1], reverse=True)
    print(unique_parcellations)
    print('Number of regions:', len(unique_parcellations))

    if regions is not None:
        unique_parcellations = [region for region in unique_parcellations if region in regions]
    rms_contrasts = [{region: [] for region in unique_parcellations} for c in range(len(conditions))]

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label==True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label=="advanced":
            parcellations = select_parcel(parcellations, go_next)
        print(np.array(rms_obj['ch_names']))

        # for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    # Accumulate rms by region
                    for region in unique_parcellations:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            rms_contrasts[cond][region].append(np.mean(rms[correct, c], axis=0) - np.mean(rms[incorrect, c], axis=0))

    # Calculate and save average rms for each condition and region
    region_rms_per_condition = []
    error_bars_per_condition = []
    regions_per_condition = []
    pvalue_per_condition = []
    t_per_condition = []
    for cond in range(len(conditions)):
        condition_rms = []
        condition_error_bars = []
        condition_regions = []
        condition_pvalue = []
        condition_t = []
        for region in unique_parcellations:
            region_rms = np.array(rms_contrasts[cond][region])
            # region_rms = np.abs(np.array(rms_contrasts[cond][region]))
            condition_regions.append(region)
            mean_rms = np.mean(region_rms, axis=0)
            condition_rms.append(mean_rms)
            error_rms = np.std(region_rms, axis=0) / np.sqrt(region_rms.shape[0])
            condition_error_bars.append(error_rms)
            ttest = ttest_1samp(region_rms, 0)
            condition_pvalue.append(ttest.pvalue)
            condition_t.append(ttest.statistic)
        region_rms_per_condition.append(condition_rms)
        error_bars_per_condition.append(condition_error_bars)
        regions_per_condition.append(condition_regions)
        pvalue_per_condition.append(condition_pvalue)
        t_per_condition.append(condition_t)

    # FDR correction
    for c in range(len(pvalue_per_condition)):
        _, pval = mne.stats.fdr_correction(pvalue_per_condition[c], alpha=0.05, method='indep')
        pvalue_per_condition[c] = pval

    new_names = {'ctx-lh-transversetemporal': 'L-TT', 'ctx-lh-superiortemporal': 'L-STG',
                 'ctx-rh-transversetemporal': 'R-TT', 'ctx-rh-superiortemporal': 'R-STG',
                 'ctx-lh-middletemporal': 'L-MTG', 'ctx-rh-middletemporal': 'R-MTG',
                 'Left-Hippocampus': 'L-HPC', 'Left-Amygdala': 'L-AMY',
                 'ctx-lh-parahippocampal': 'L-PHC', 'ctx-lh-temporalpole': 'L-TP',
                 'ctx-lh-inferiortemporal': 'L-ITG', 'ctx-lh-entorhinal': 'L-EC',
                 'Right-Hippocampus': 'R-HPC', 'Right-Amygdala': 'R-AMY',
                 'ctx-rh-parahippocampal': 'R-PHC', 'ctx-rh-temporalpole': 'R-TP', 'ctx-rh-inferiortemporal': 'R-ITG',
                 'ctx-rh-entorhinal': 'R-EC'}

    cmap = list(plt.get_cmap('Set3').colors)
    del cmap[8]
    region_colors = {region: cmap[i % int(len(regions) / 2)] for i, region in enumerate(regions)}

    fig, axs = plt.subplots(1, len(conditions), sharex=True, figsize=(10, 5))
    # Plot each condition's data in a separate subplot
    for i, (condition, regions, region_rms, error_bars, pvalues, tvalues) in enumerate(zip(conditions, regions_per_condition, region_rms_per_condition, error_bars_per_condition, pvalue_per_condition, t_per_condition)):
        # fig, axs = plt.subplots(1, 1, figsize=)
        sorted_id = np.argsort(region_rms)
        region_rms = np.array(region_rms)[sorted_id]
        regions = np.array(regions)[sorted_id]
        new_names_regions = [new_names[reg] for reg in regions]
        pvalues = np.array(pvalues)[sorted_id]
        # for r in range(len(regions)):
        #     print(condition)
            # print(regions[r])
            # print(tvalues[r])
            # print(pvalues[r])
            # print(len(rms_contrasts[i][regions[r]]))
        colors = [region_colors[regions[r]] if p < 0.05 else 'w' for r, p in enumerate(pvalues)]
        axs[i].barh(new_names_regions, region_rms, xerr=None, capsize=5, color=colors, edgecolor='k', lw=1)
        axs[i].axvline(0, color='k', linestyle='--', linewidth=1.5)
        axs[i].tick_params(axis='y', labelsize=13)
        axs[i].tick_params(axis='x', labelsize=17)

        # Add individual sample points as scatter dots
        for j, region in enumerate(regions):
            samples = np.array(rms_contrasts[i][region])
            if remove_outliers:
                samples = samples[(np.abs(samples - samples.mean())) <= 2 * samples.std()]
            # samples = np.abs(np.array(rms_contrasts[i][region]))
            # Add scatter plot with jitter on x-axis
            y_values = np.full(samples.shape, j) + np.random.uniform(-0.1, 0.1, samples.shape)  # Jitter for clarity
            axs[i].scatter(samples, y_values, color='black', alpha=0.7, s=2, label='Samples')

        # axs[i].set_title(condition)
        # if i == 0:  # Only set ylabel on the first subplot
        axs[i].set_xlabel('Percent change of activity', fontsize=17)
    # fig.supxlabel(f'Average contrast of {analysis_name} relative to baseline (percent)')

        # Rotate x-axis labels for readability
        # axs[i].tick_params(axis='x', rotation=90)

    # Adjust the layout so the plots don't overlap
    plt.tight_layout()
    plt.show()

    return


def plot_all_electrode_scalar_only_correct_incorrect(data_path, behavior_path, first_parcel_label=True, analysis_name='rms'):
    """used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
    print(subject_paths)

    unique_parcellations = []
    for i, subject_path in enumerate(tqdm(subject_paths)):
        rms_obj = np.load(subject_path, allow_pickle=True)
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label==True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label=="advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        unique_parcellations += parcellations
    unique_parcellations = np.unique(unique_parcellations)
    # regions_to_remove = ['Unknown', 'Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter', 'ctx-rh-unknown', 'ctx-lh-unknown', 'CSF']
    regions_to_remove = ['Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter', 'CSF', 'Unknown']
    # regions_to_remove = []
    unique_parcellations = unique_parcellations[np.isin(unique_parcellations, regions_to_remove, invert=True)]
    unique_parcellations = sorted(unique_parcellations)
    print(unique_parcellations)
    print('Number of regions:', len(unique_parcellations))

    rms_contrasts = {region: [] for region in unique_parcellations}
    rms_contrasts_t = {region: [] for region in unique_parcellations}
    perm_contrasts = {region: [] for region in unique_parcellations}
    channel_pvalues = {region: [] for region in unique_parcellations}

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        # correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        # incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_all = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        # correct_all = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        # incorrect_all = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label==True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label=="advanced":
            parcellations = select_parcel(parcellations, go_next)
        print(np.array(rms_obj['ch_names']))

        correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_all]
        incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in incorrect_all]
        if (len(correct) > 0) & (len(incorrect) > 0):
            for c in range(rms.shape[1]):
                # Accumulate rms by region
                for region in unique_parcellations:
                    if any(ch_parcel in region for ch_parcel in parcellations[c]):
                        # rms_contrasts[region].append(np.mean(rms[correct, c], axis=0) - np.mean(rms[incorrect, c], axis=0))
                        observed_diff, perm_diffs = permute_labels_and_compute_contrast(rms[:, c], correct, incorrect, n_permutations=10000)
                        rms_contrasts[region].append(observed_diff)
                        perm_contrasts[region].append(perm_diffs)
                        ttest = ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False, permutations=10000, random_state=None, alternative='two-sided')
                        channel_pvalues[region].append(ttest.pvalue)
                        rms_contrasts_t[region].append(ttest.statistic)

    # Calculate and save average rms for each condition and region
    condition_rms = []
    condition_error_bars = []
    condition_regions = []
    condition_pvalue = []
    num = 0
    for region in unique_parcellations:
        region_rms = np.array(rms_contrasts[region])
        region_perms = np.array(perm_contrasts[region])

        t_obs = np.mean(region_rms) / (np.std(region_rms, ddof=1) / np.sqrt(len(region_rms)))
        t_perms = np.mean(region_perms, axis=0) / (np.std(region_perms, axis=0, ddof=1) / np.sqrt(region_perms.shape[0]))
        population_p = np.sum(np.abs(t_perms) >= np.abs(t_obs)) / region_perms.shape[1]

        print(region)
        print(len(region_rms))
        if len(region_rms) >= 2:
        # if len(region_rms) >= 5:
            num += len(region_rms)
        # if region in ['Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala']:
            condition_regions.append(region)
            mean_rms = np.mean(region_rms, axis=0)
            condition_rms.append(mean_rms)
            # condition_rms.append(t_obs)
            error_rms = np.std(region_rms, axis=0) / np.sqrt(region_rms.shape[0])
            condition_error_bars.append(error_rms)
            # condition_pvalue.append(ttest_1samp(region_rms, 0).pvalue)
            condition_pvalue.append(population_p)
    print('Number of kept channels:', num)

    # FDR correction
    print(condition_pvalue)
    _, pval = mne.stats.fdr_correction(condition_pvalue, alpha=0.05, method='indep')
    condition_pvalue = pval
    print(condition_pvalue)

    fig, axs = plt.subplots(1, 1, sharey=True)
    # Plot each condition's data in a separate subplot
    colors = ['orange' if p < 0.05 else 'skyblue' for p in condition_pvalue]
    axs.barh(condition_regions, condition_rms, xerr=condition_error_bars, capsize=5, color=colors)
    # axs.barh(condition_regions, condition_rms, capsize=5, color=colors)
    axs.axvline(0, color='gray', linestyle='--', linewidth=1.5)

    # Add individual sample points as scatter dots
    for j, region in enumerate(condition_regions):
        samples = np.array(rms_contrasts[region])
        # samples = np.array(rms_contrasts_t[region])
        pvalues = np.array(channel_pvalues[region])
        # samples = np.abs(np.array(rms_contrasts[i][region]))
        # Add scatter plot with jitter on x-axis
        y_values = np.full(samples.shape, j) + np.random.uniform(-0.1, 0.1, samples.shape)  # Jitter for clarity
        colors = ['red' if p<0.05 else 'black' for p in pvalues]
        axs.scatter(samples, y_values, color=colors, alpha=0.7, s=6, label='Samples')

    # if i == 0:  # Only set ylabel on the first subplot
        # axs[i].set_xlabel('Average RMS contrast')
    axs.set_xlabel(f'Average contrast of {analysis_name} relative to baseline (percent)')

        # Rotate x-axis labels for readability
        # axs[i].tick_params(axis='x', rotation=90)

    # Adjust the layout so the plots don't overlap
    # plt.tight_layout()
    plt.show()

    return


def save_rms_contrasts_stat_individual_electrode(data_path, behavior_path, analysis_name='rms'):
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
    del subject_paths[1]
    print(subject_paths)

    conditions = ['novelty', 'retrieval']

    subjects_tvalues = [[] for c in range(len(conditions))]
    subjects_pvalues = [[] for c in range(len(conditions))]

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]
        print(np.array(rms_obj['ch_names']))

        for cond, correct_incorrect in enumerate(
                [[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                ttest = ttest_ind(rms[correct, :], rms[incorrect, :], axis=0, equal_var=False, permutations=100000,
                                  random_state=None, alternative='two-sided')
                subjects_tvalues[cond].append(ttest.statistic)
                subjects_pvalues[cond].append(ttest.pvalue)
            else:
                subjects_tvalues[cond].append(None)
                subjects_pvalues[cond].append(None)

    contrast_stats = {'tvalues': subjects_tvalues, 'pvalues': subjects_pvalues}
    np.save(os.path.join(data_path, 'rms_contrast_stats'), contrast_stats, allow_pickle=True)


def permute_labels_and_compute_contrast(rms, correct_indices, incorrect_indices, n_permutations=1000):
    """Permute correct and incorrect labels across epochs and compute contrasts."""
    # observed_diff = np.abs(np.mean(rms[correct_indices]) - np.mean(rms[incorrect_indices]))
    observed_diff = np.mean(rms[correct_indices]) - np.mean(rms[incorrect_indices])

    perm_diffs = np.zeros(n_permutations + 1)

    combined_indices = np.concatenate([correct_indices, incorrect_indices])
    n_correct = len(correct_indices)

    for i in range(n_permutations):
        # Shuffle the indices
        shuffled_indices = np.random.permutation(combined_indices)
        # Split shuffled data into 'correct' and 'incorrect' groups
        perm_correct_indices = shuffled_indices[:n_correct]
        perm_incorrect_indices = shuffled_indices[n_correct:]
        # Compute the contrast for the permutation
        # perm_diffs[i] = np.abs(np.mean(rms[perm_correct_indices]) - np.mean(rms[perm_incorrect_indices]))
        perm_diffs[i] = np.mean(rms[perm_correct_indices]) - np.mean(rms[perm_incorrect_indices])
    perm_diffs[-1] = observed_diff
    # Calculate p-value as the proportion of permutations where the difference is greater than or equal to the observed difference
    # p_value = np.sum(perm_diffs >= observed_diff) / n_permutations
    return observed_diff, perm_diffs

def plot_individual_electrode_scalar(data_path, behavior_path, regions, first_parcel_label=True, analysis_name='rms'):
    """electrode-level test (FDR corrected) used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
    del subject_paths[1]
    subject_names = []
    print(subject_paths)

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']

    rms_contrasts = [[] for c in range(len(conditions))]
    perm_contrasts = [[] for c in range(len(conditions))]
    pvalues = [[] for c in range(len(conditions))]
    subject_id = [[] for c in range(len(conditions))]  # used for color

    n_chan = 0

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label=="advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            observed_diff, perm_diffs = permute_labels_and_compute_contrast(rms[:, c], correct, incorrect, n_permutations=1000)
                            rms_contrasts[cond].append(observed_diff)
                            perm_contrasts[cond].append(perm_diffs)
                            pvalues[cond].append(ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False, permutations=1000, random_state=None, alternative='greater').pvalue)  # testing of individual electrodes
                            subject_id[cond].append(i)
                            if cond == 0:
                                n_chan+=1

    print("n_chan:", n_chan)

    # FDR correction
    # for c in range(len(pvalues)):
    #     _, pval = mne.stats.fdr_correction(pvalues[c], alpha=0.05, method='indep')
    #     pvalues[c] = pval

    fig, axs = plt.subplots(1, 1, sharey=True)
    subject_colors = plt.get_cmap('Set3').colors
    # Plot each condition's data in a separate subplot
    for i, (condition, rms, perms, pval, s_id) in enumerate(zip(conditions, rms_contrasts, perm_contrasts, pvalues, subject_id)):
    # for i, (condition, rms, perms, s_id) in enumerate(zip(conditions, rms_contrasts, perm_contrasts, subject_id)):
        # population_p = ttest_1samp(rms, 0).pvalue
        t_obs = np.mean(rms) / (np.std(rms, ddof=1) / np.sqrt(len(rms)))
        t_perms = np.mean(perms, axis=0) / (np.std(perms, axis=0, ddof=1) / np.sqrt(np.array(perms).shape[0]))
        population_p = np.sum(np.abs(t_perms) >= np.abs(t_obs)) / np.array(perms).shape[1]
        # population_p = np.sum(np.mean(perms, axis=0) >= np.mean(rms)) / np.array(perms).shape[1]
        print(population_p)
        axs.bar(i, np.mean(rms), yerr=np.std(rms)/np.sqrt(len(rms)), capsize=5, color=('orange' if population_p < 0.05 else 'skyblue'))
        axs.axhline(0, color='gray', linestyle='--', linewidth=1.5)

        # Add individual sample points as scatter dots
        # colors = ['red' if p < 0.05 else 'black' for p in pval]
        # print("Percent of significant electrodes:", np.sum(np.array(pval) < 0.05) / len(pval) * 100)
        colors = [subject_colors[s] for s in s_id]
        x_values = np.full(len(rms), i) + np.random.uniform(-0.1, 0.1, len(rms))  # Jitter for clarity
        axs.scatter(x_values, rms, color=colors, alpha=1, s=20, label='Samples')
        axs.set_ylabel(f'Average contrast of {analysis_name} relative to baseline (percent)')
    axs.set_xticks([x for x in range(len(conditions))], conditions)
    # Create custom legend entries, one per unique color
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=7) for name, color in zip(subject_names, subject_colors)]
    # plt.legend(handles=legend_elements, loc="best", title="Subjects")

    # Adjust the layout so the plots don't overlap
    # plt.tight_layout()
    plt.show()

    print(len(rms_contrasts[2]))
    print(len(rms_contrasts[3]))
    print(len(subject_id[2]))
    print(len(subject_id[3]))
    print(subject_id[2] == subject_id[3])
    print(np.unique(subject_id[2]))

    # Define subject colors
    subject_colors = plt.get_cmap('Set3').colors
    # subject_colors = [subject_colors[i] for i in range(len(subject_colors)) if i != 1]  # different colors (but same as the ones used for regions)
    subject_colors = ['lightgray' for i in range(len(subject_colors)) if i != 1]  # all the same color

    # Create DataFrame
    data_dict = {
        'Novelty_retrieval': rms_contrasts[2],
        'Retrieval': rms_contrasts[3],
        'SubjectID': subject_id[2]
    }
    model_data = pd.DataFrame(data_dict)

    # Compute mean and std for outlier detection
    mean_novelty = model_data['Novelty_retrieval'].mean()
    std_novelty = model_data['Novelty_retrieval'].std()
    mean_retrieval = model_data['Retrieval'].mean()
    std_retrieval = model_data['Retrieval'].std()

    # Mask to identify outliers
    outlier_mask = (
            (np.abs(model_data['Novelty_retrieval'] - mean_novelty) > 3 * std_novelty) |
            (np.abs(model_data['Retrieval'] - mean_retrieval) > 3 * std_retrieval)
    )

    # Data without outliers (for visualization & plotting regression)
    plot_data = model_data[~outlier_mask]

    # Assign colors after reordering subjects
    colors = [subject_colors[s] for s in plot_data['SubjectID']]

    # Plot scatter (without outliers)
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.scatter(plot_data['Novelty_retrieval'], plot_data['Retrieval'], color=colors, s=12, alpha=0.6)
    axs.set_xlabel('Novelty (% change of activity)', fontsize=17)
    axs.set_ylabel('Recognition (% change of activity)', fontsize=17)
    axs.tick_params(axis='y', labelsize=17)
    axs.tick_params(axis='x', labelsize=17)
    axs.axvline(0, ls='--', color='k')
    axs.axhline(0, ls='--', color='k')
    axs.set_xlim(-15, 60)
    axs.set_ylim(-15, 35)

    # Fit mixed-effects model (using full dataset for stats)
    model_full = MixedLM.from_formula('Retrieval ~ Novelty_retrieval', groups='SubjectID', data=model_data)
    result_full = model_full.fit()
    print(result_full.summary())
    p_value = result_full.pvalues['Novelty_retrieval']

    # Fit a new model for visualization (without outliers)
    model_plot = MixedLM.from_formula('Retrieval ~ Novelty_retrieval', groups='SubjectID', data=plot_data)
    result_plot = model_plot.fit()

    # Plot regression line (without outliers)
    x_vals = np.linspace(min(plot_data['Novelty_retrieval']), max(plot_data['Novelty_retrieval']), 100)
    y_vals = result_plot.params['Intercept'] + result_plot.params['Novelty_retrieval'] * x_vals
    axs.plot(x_vals, y_vals, color='k', linewidth=5, zorder=10)

    # Plot individual subject fits (without outliers)
    unique_subjects = np.unique(plot_data['SubjectID'])
    for subj in unique_subjects:
        mask = plot_data['SubjectID'] == subj
        x_subj = plot_data.loc[mask, 'Novelty_retrieval']
        y_subj = plot_data.loc[mask, 'Retrieval']
        if len(x_subj) > 1:  # Ensure enough points to fit
            slope, intercept = np.polyfit(x_subj, y_subj, 1)
            x_range = np.linspace(min(x_subj), max(x_subj), 100)
            axs.plot(x_range, slope * x_range + intercept, color=subject_colors[subj], lw=3, alpha=1)

    # Annotate with model results from full data
    # axs.text(0, axs.get_ylim()[1],
    #          f"Beta = {round(result_full.params['Novelty_retrieval'], 2)}, p = {round(p_value, 5)}")
    plt.tight_layout()


    # novelty vs. retrieval analysis
    novelty_list = []
    retrieval_list = []
    for i in range(len(subject_paths)):
        subj_channels_novelty = np.array(subject_id[2])==i
        subj_channels_retrieval = np.array(subject_id[3])==i
        if (np.sum(subj_channels_novelty) == np.sum(subj_channels_retrieval)) and (np.sum(subj_channels_novelty) > 1):
            contrast_novelty = np.array(rms_contrasts[2])[subj_channels_novelty]
            contrast_novelty = contrast_novelty - np.mean(contrast_novelty)
            novelty_list += contrast_novelty.tolist()
            contrast_retrieval = np.array(rms_contrasts[3])[subj_channels_retrieval]
            contrast_retrieval = contrast_retrieval - np.mean(contrast_retrieval)
            retrieval_list += contrast_retrieval.tolist()

    # del novelty_list[3]
    # del novelty_list[2]
    # del retrieval_list[3]
    # del retrieval_list[2]

    fig, ax = plt.subplots(1, 1)
    ax.scatter(novelty_list, retrieval_list)
    ax.set_xlabel('Novelty contrast (relative to subject mean)')
    ax.set_ylabel('Retrieval contrast (relative to subject mean)')
    ax.axhline(0, ls='--', color='k')
    ax.axvline(0, ls='--', color='k')

    person_corr = pearsonr(novelty_list, retrieval_list, alternative='less')
    print('Correlation test')
    print(person_corr)
    spearman_corr = spearmanr(novelty_list, retrieval_list, alternative='less')
    print('Spearman')
    print(spearman_corr)
    ax.text(0, ax.get_ylim()[1], f'rho = {round(spearman_corr.statistic, 2)}, p = {round(spearman_corr.pvalue, 5)}')


    # novelty vs. retrieval analysis but absolute plot
    novelty_list = []
    retrieval_list = []
    s_list = []
    linear_fit_list = []
    linear_fit_id = []
    for i in range(len(subject_paths)):
        subj_channels_novelty = np.array(subject_id[2]) == i
        subj_channels_retrieval = np.array(subject_id[3]) == i
        if (np.sum(subj_channels_novelty) == np.sum(subj_channels_retrieval)) and (np.sum(subj_channels_novelty) > 1):
            contrast_novelty = np.array(rms_contrasts[2])[subj_channels_novelty]
            novelty_list += contrast_novelty.tolist()
            contrast_retrieval = np.array(rms_contrasts[3])[subj_channels_retrieval]
            retrieval_list += contrast_retrieval.tolist()
            s_list += [i for _ in range(len(contrast_retrieval))]
            # linear fit per subject
            model = LinearRegression(fit_intercept=True)
            model.fit(contrast_novelty[:, np.newaxis], contrast_retrieval)
            linear_fit_list.append([model.intercept_, model.coef_[0]])
            linear_fit_id.append(i)

    fig, ax = plt.subplots(1, 1)
    colors = [subject_colors[s] for s in s_list]
    ax.scatter(novelty_list, retrieval_list, color=colors)
    ax.set_xlabel('Novelty contrast')
    ax.set_ylabel('Retrieval contrast')
    ax.axhline(0, ls='--', lw=2, color='k')
    ax.axvline(0, ls='--', lw=2, color='k')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for i, params in enumerate(linear_fit_list):
        ax.axline((0, params[0]), slope=params[1], color=subject_colors[linear_fit_id[i]], alpha=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.show()

    return

def plot_individual_subject_scalar(data_path, behavior_path, regions, first_parcel_label=True, analysis_name='rms'):
    """electrode-level test (FDR corrected) used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
    subject_names = []
    print(subject_paths)

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']

    rms_contrasts = [[] for c in range(len(conditions))]
    perm_contrasts = [[] for c in range(len(conditions))]
    subject_id = [[] for c in range(len(conditions))]  # used for color

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label==True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label=="advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                ch_index = []
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            ch_index.append(c)
                observed_diff, perm_diffs = permute_labels_and_compute_contrast(np.mean(rms[:, ch_index], axis=1), correct, incorrect, n_permutations=10000)
                rms_contrasts[cond].append(observed_diff)
                perm_contrasts[cond].append(perm_diffs)
                subject_id[cond].append(i)

    fig, axs = plt.subplots(1, 1, sharey=True)
    subject_colors = plt.get_cmap('Set3').colors
    # Plot each condition's data in a separate subplot
    for i, (condition, rms, perms, s_id) in enumerate(zip(conditions, rms_contrasts, perm_contrasts, subject_id)):
        # population_p = ttest_1samp(rms, 0).pvalue
        t_obs = np.mean(rms) / (np.std(rms, ddof=1) / np.sqrt(len(rms)))
        t_perms = np.mean(perms, axis=0) / (np.std(perms, axis=0, ddof=1) / np.sqrt(np.array(perms).shape[0]))
        population_p = np.sum(np.abs(t_perms) >= np.abs(t_obs)) / np.array(perms).shape[1]
        # population_p = np.sum(np.mean(perms, axis=0) >= np.mean(rms)) / np.array(perms).shape[1]
        print(population_p)
        axs.bar(i, np.mean(rms), yerr=np.std(rms)/np.sqrt(len(rms)), capsize=5, color=('orange' if population_p < 0.05 else 'skyblue'))
        axs.axhline(0, color='gray', linestyle='--', linewidth=1.5)

        # Add individual sample points as scatter dots
        # colors = ['red' if p < 0.05 else 'black' for p in pval]
        colors = [subject_colors[s] for s in s_id]
        x_values = np.full(len(rms), i) + np.random.uniform(-0.1, 0.1, len(rms))  # Jitter for clarity
        axs.scatter(x_values, rms, color=colors, alpha=1, s=20, label='Samples')
        axs.set_ylabel(f'Average contrast of {analysis_name} relative to baseline (percent)')
    axs.set_xticks([x for x in range(len(conditions))], conditions)
    # Create custom legend entries, one per unique color
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=7) for name, color in zip(subject_names, subject_colors)]
    plt.legend(handles=legend_elements, loc="best", title="Subjects")

    # Adjust the layout so the plots don't overlap
    # plt.tight_layout()
    plt.show()

    return

def plot_individual_electrode_scalar_windows(data_path, behavior_path, regions, first_parcel_label=True, analysis_name='rms', remove_outliers=False):
    """electrode-level test (FDR corrected) used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}_windows.pkl')))
    del subject_paths[1]
    print(subject_paths)

    rms_obj = np.load(subject_paths[0], allow_pickle=True)
    rms = rms_obj[analysis_name]
    n_windows = len(rms)
    print(n_windows)

    # conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']
    conditions = ['novelty', 'retrieval']

    fig, axs = plt.subplots(1, len(conditions), sharey=True)
    for window in range(n_windows):
        print('Window', window)
        rms_contrasts = [[] for c in range(len(conditions))]
        perm_contrasts = [[] for c in range(len(conditions))]
        pvalues = [[] for c in range(len(conditions))]

        for i, subject_path in enumerate(tqdm(subject_paths)):
            subject_name = Path(subject_path).name[:6]
            behavior = behavior_all[behavior_all['subject'] == subject_name]
            correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
            incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
            correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
            incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
            correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
            incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
            correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
            incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
            correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
            incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
            memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
            novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

            rms_obj = np.load(subject_path, allow_pickle=True)
            rms = rms_obj[analysis_name][window]
            parcellations = rms_obj['ch_parcellations']
            if first_parcel_label == True:
                parcellations = [[parcel[0]] for parcel in parcellations]
            elif first_parcel_label == "advanced":
                go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
                parcellations = select_parcel(parcellations, go_next)

            # for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
            for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
                correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
                incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
                if (len(correct) > 0) & (len(incorrect) > 0):
                    for c in range(rms.shape[1]):
                        for region in regions:
                            if any(ch_parcel in region for ch_parcel in parcellations[c]):
                                observed_diff, perm_diffs = permute_labels_and_compute_contrast(rms[:, c], correct, incorrect, n_permutations=1000)
                                rms_contrasts[cond].append(observed_diff)
                                perm_contrasts[cond].append(perm_diffs)
                                # pvalues[cond].append(ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False, permutations=1000, random_state=None, alternative='two-sided').pvalue)  # testing of individual electrodes

        # FDR correction
        # for c in range(len(pvalues)):
        #     _, pval = mne.stats.fdr_correction(pvalues[c], alpha=0.05, method='indep')
        #     pvalues[c] = pval

        # Plot each condition's data in a separate subplot
        # for i, (condition, rms, perms, pval) in enumerate(zip(conditions, rms_contrasts, perm_contrasts, pvalues)):
        for i, (condition, rms, perms) in enumerate(zip(conditions, rms_contrasts, perm_contrasts)):
            # population_p = ttest_1samp(rms, 0).pvalue
            t_obs = np.mean(rms) / (np.std(rms, ddof=1) / np.sqrt(len(rms)))
            t_perms = np.mean(perms, axis=0) / (np.std(perms, axis=0, ddof=1) / np.sqrt(np.array(perms).shape[0]))
            population_p = np.sum(np.abs(t_perms) >= np.abs(t_obs)) / np.array(perms).shape[1]
            # population_p = np.sum(np.mean(perms, axis=0) >= np.mean(rms)) / np.array(perms).shape[1]
            axs[i].bar(window, np.mean(rms), yerr=np.std(rms)/np.sqrt(len(rms)), capsize=5, color=('orange' if population_p < 0.05 else 'skyblue'))
            axs[i].axhline(0, color='gray', linestyle='--', linewidth=1.5)

            # Add individual sample points as scatter dots
            # colors = ['red' if p < 0.05 else 'black' for p in pval]
            colors = 'black'
            if remove_outliers:
                rms = np.array(rms)
                rms = rms[(np.abs(rms - rms.mean())) <= 3 * rms.std()]
            x_values = np.full(len(rms), window) + np.random.uniform(-0.1, 0.1, len(rms))  # Jitter for clarity
            axs[i].scatter(x_values, rms, color=colors, alpha=0.5, s=0.5, label='Samples')
            axs[i].set_title(condition)
            axs[i].set_xticks([x for x in range(n_windows)], [x for x in range(n_windows)])
            axs[i].set_xlabel(f'From x to x+1 seconds')
    axs[0].set_ylabel(f'Average contrast of {analysis_name} relative to baseline (percent)')

    # Adjust the layout so the plots don't overlap
    # plt.tight_layout()
    plt.show()

    return

def loop_plot_scalar_window_regions(data_path, behavior_path, regions_lists):
    for regions in regions_lists:
        plot_individual_electrode_scalar_windows(data_path, behavior_path, regions, first_parcel_label='advanced', analysis_name='rms', remove_outliers=True)


def plot_individual_electrode_scalar_windows_3rows(data_path, behavior_path, regions, first_parcel_label=True, analysis_name='rms'):
    """electrode-level test (FDR corrected) used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}_windows.pkl')))
    print(subject_paths)

    rms_obj = np.load(subject_paths[0], allow_pickle=True)
    rms = rms_obj[analysis_name]
    assert len(rms[0]) == len(rms[1])
    n_windows = len(rms[0])
    print(n_windows)

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']

    fig, axs = plt.subplots(3, len(conditions), sharey=True)
    for row in range(3):
        for window in range(n_windows):
            print('Window', window)
            rms_contrasts = [[] for c in range(len(conditions))]
            perm_contrasts = [[] for c in range(len(conditions))]
            pvalues = [[] for c in range(len(conditions))]

            for i, subject_path in enumerate(tqdm(subject_paths)):
                subject_name = Path(subject_path).name[:6]
                behavior = behavior_all[behavior_all['subject'] == subject_name]
                correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
                incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
                correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
                incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
                correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
                incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
                correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
                incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
                correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
                incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
                memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
                novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

                rms_obj = np.load(subject_path, allow_pickle=True)
                rms = rms_obj[analysis_name][row][window]
                parcellations = rms_obj['ch_parcellations']
                if first_parcel_label == True:
                    parcellations = [[parcel[0]] for parcel in parcellations]
                elif first_parcel_label == "advanced":
                    go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
                    parcellations = select_parcel(parcellations, go_next)

                for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
                    correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
                    incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
                    if (len(correct) > 0) & (len(incorrect) > 0):
                        for c in range(rms.shape[1]):
                            for region in regions:
                                if any(ch_parcel in region for ch_parcel in parcellations[c]):
                                    observed_diff, perm_diffs = permute_labels_and_compute_contrast(rms[:, c], correct, incorrect, n_permutations=1000)
                                    rms_contrasts[cond].append(observed_diff)
                                    perm_contrasts[cond].append(perm_diffs)
                                    # pvalues[cond].append(ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False, permutations=1000, random_state=None, alternative='two-sided').pvalue)  # testing of individual electrodes

            # FDR correction
            # for c in range(len(pvalues)):
            #     _, pval = mne.stats.fdr_correction(pvalues[c], alpha=0.05, method='indep')
            #     pvalues[c] = pval

            # Plot each condition's data in a separate subplot
            # for i, (condition, rms, perms, pval) in enumerate(zip(conditions, rms_contrasts, perm_contrasts, pvalues)):
            for i, (condition, rms, perms) in enumerate(zip(conditions, rms_contrasts, perm_contrasts)):
                # population_p = ttest_1samp(rms, 0).pvalue
                t_obs = np.mean(rms) / (np.std(rms, ddof=1) / np.sqrt(len(rms)))
                t_perms = np.mean(perms, axis=0) / (np.std(perms, axis=0, ddof=1) / np.sqrt(np.array(perms).shape[0]))
                population_p = np.sum(np.abs(t_perms) >= np.abs(t_obs)) / np.array(perms).shape[1]
                # population_p = np.sum(np.mean(perms, axis=0) >= np.mean(rms)) / np.array(perms).shape[1]
                axs[row, i].bar(window, np.mean(rms), yerr=np.std(rms)/np.sqrt(len(rms)), capsize=5, color=('orange' if population_p < 0.05 else 'skyblue'))
                axs[row, i].axhline(0, color='gray', linestyle='--', linewidth=1.5)

                # Add individual sample points as scatter dots
                # colors = ['red' if p < 0.05 else 'black' for p in pval]
                colors = 'black'
                x_values = np.full(len(rms), window) + np.random.uniform(-0.1, 0.1, len(rms))  # Jitter for clarity
                axs[row, i].scatter(x_values, rms, color=colors, alpha=0.7, s=5, label='Samples')
                if row == 0:
                    axs[row, i].set_title(condition)
                    axs[row, i].set_xticks([x for x in range(n_windows)], [x + 1 for x in range(n_windows)])
                    axs[row, i].set_xlabel(f'From 0 to x seconds')
                elif row == 1:
                    axs[row, i].set_xticks([x for x in range(n_windows)], [x for x in range(n_windows)])
                    axs[row, i].set_xlabel(f'From x to 8 seconds')
                elif row == 2:
                    axs[row, i].set_xticks([x for x in range(n_windows)], [x for x in range(n_windows)])
                    axs[row, i].set_xlabel(f'From x to x+1 seconds')
    axs[1, 0].set_ylabel(f'Average contrast of {analysis_name} relative to baseline (percent)')

    # Adjust the layout so the plots don't overlap
    # plt.tight_layout()
    plt.show()

    return


def plot_individual_subject_scalar_windows(data_path, behavior_path, regions, first_parcel_label=True, analysis_name='rms'):
    """electrode-level test (FDR corrected) used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}_windows.pkl')))
    print(subject_paths)

    rms_obj = np.load(subject_paths[0], allow_pickle=True)
    rms = rms_obj[analysis_name]
    assert len(rms[0]) == len(rms[1])
    n_windows = len(rms[0])
    print(n_windows)

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']

    fig, axs = plt.subplots(3, len(conditions), sharey=True)
    for row in range(3):
        for window in range(n_windows):
            print('Window', window)
            rms_contrasts = [[] for c in range(len(conditions))]
            perm_contrasts = [[] for c in range(len(conditions))]

            for i, subject_path in enumerate(tqdm(subject_paths)):
                subject_name = Path(subject_path).name[:6]
                behavior = behavior_all[behavior_all['subject'] == subject_name]
                correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
                incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
                correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
                incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
                correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
                incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
                correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
                incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
                correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
                incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
                memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
                novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

                rms_obj = np.load(subject_path, allow_pickle=True)
                rms = rms_obj[analysis_name][row][window]
                parcellations = rms_obj['ch_parcellations']
                if first_parcel_label == True:
                    parcellations = [[parcel[0]] for parcel in parcellations]
                elif first_parcel_label == "advanced":
                    go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
                    parcellations = select_parcel(parcellations, go_next)

                for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
                    correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
                    incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
                    if (len(correct) > 0) & (len(incorrect) > 0):
                        ch_index = []
                        for c in range(rms.shape[1]):
                            for region in regions:
                                if any(ch_parcel in region for ch_parcel in parcellations[c]):
                                    ch_index.append(c)
                        observed_diff, perm_diffs = permute_labels_and_compute_contrast(np.mean(rms[:, ch_index], axis=1), correct, incorrect, n_permutations=1000)
                        rms_contrasts[cond].append(observed_diff)
                        perm_contrasts[cond].append(perm_diffs)

            # Plot each condition's data in a separate subplot
            for i, (condition, rms, perms, ) in enumerate(zip(conditions, rms_contrasts, perm_contrasts)):
                # population_p = ttest_1samp(rms, 0).pvalue
                t_obs = np.mean(rms) / (np.std(rms, ddof=1) / np.sqrt(len(rms)))
                t_perms = np.mean(perms, axis=0) / (np.std(perms, axis=0, ddof=1) / np.sqrt(np.array(perms).shape[0]))
                population_p = np.sum(np.abs(t_perms) >= np.abs(t_obs)) / np.array(perms).shape[1]
                # population_p = np.sum(np.mean(perms, axis=0) >= np.mean(rms)) / np.array(perms).shape[1]
                axs[row, i].bar(window, np.mean(rms), yerr=np.std(rms)/np.sqrt(len(rms)), capsize=5, color=('orange' if population_p < 0.05 else 'skyblue'))
                axs[row, i].axhline(0, color='gray', linestyle='--', linewidth=1.5)

                # Add individual sample points as scatter dots
                x_values = np.full(len(rms), window) + np.random.uniform(-0.1, 0.1, len(rms))  # Jitter for clarity
                axs[row, i].scatter(x_values, rms, color='black', alpha=0.7, s=5, label='Samples')
                if row == 0:
                    axs[row, i].set_title(condition)
                    axs[row, i].set_xticks([x for x in range(n_windows)], [x + 1 for x in range(n_windows)])
                    axs[row, i].set_xlabel(f'From 0 to x seconds')
                elif row == 1:
                    axs[row, i].set_xticks([x for x in range(n_windows)], [x for x in range(n_windows)])
                    axs[row, i].set_xlabel(f'From x to 8 seconds')
                elif row == 2:
                    axs[row, i].set_xticks([x for x in range(n_windows)], [x for x in range(n_windows)])
                    axs[row, i].set_xlabel(f'From x to x+1 seconds')
    axs[1, 0].set_ylabel(f'Average contrast of {analysis_name} relative to baseline (percent)')

    # Adjust the layout so the plots don't overlap
    # plt.tight_layout()
    plt.show()

    return


def plot_individual_electrode_scalar_temporal(data_path, behavior_path, regions, first_parcel_label=True, analysis_name='abs'):
    """electrode-level temporal plot used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
    subject_names = []
    print(subject_paths)

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']
    # conditions = ['encoding', 'novelty', 'retrieval']

    rms_contrasts = [[] for c in range(len(conditions))]
    perm_contrasts = [[] for c in range(len(conditions))]
    pvalues = [[] for c in range(len(conditions))]
    subject_id = [[] for c in range(len(conditions))]  # used for color

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
        # for cond, correct_incorrect in enumerate([[correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            rms_contrasts[cond].append(np.mean(rms[correct, c, :], axis=0) - np.mean(rms[incorrect, c, :], axis=0))
                            # perm_contrasts[cond].append(perm_diffs)
                            # pvalues[cond].append(ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False, permutations=1000, random_state=None, alternative='two-sided').pvalue)  # testing of individual electrodes
                            subject_id[cond].append(i)

    # FDR correction
    # for c in range(len(pvalues)):
    #     _, pval = mne.stats.fdr_correction(pvalues[c], alpha=0.05, method='indep')
    #     pvalues[c] = pval

    # fig, axs = plt.subplots(1, len(conditions), sharex=True, sharey=True)
    fig, axs = plt.subplots(len(conditions), 1, sharex=True, sharey=True)
    condition_colors = plt.get_cmap('Set1').colors
    # Plot each condition's data in a separate subplot
    # for i, (condition, rms, perms, pval, s_id) in enumerate(zip(conditions, rms_contrasts, perm_contrasts, pvalues, subject_id)):
    for i, (condition, rms) in enumerate(zip(conditions, rms_contrasts)):
        # population_p = ttest_1samp(rms, 0).pvalue
        print(np.array(rms).shape)

        # stat test permut cluster
        # thresh = dict(start=0, step=0.1)
        pval = 0.05  # arbitrary
        df = len(rms) - 1  # degrees of freedom denominator
        thresh = scipy.stats.t.ppf(1 - pval / 2, df=df)  # t distribution two-tailed
        # thresh = scipy.stats.t.ppf(1 - pval, df=df)  # t distribution one-tailed
        print(thresh)
        T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(np.array(rms),
                                                                   threshold=thresh,
                                                                   n_permutations=10000, tail=0,
                                                                   adjacency=None, n_jobs=-1)
        # Create a mask for significant clusters
        significant_mask = np.zeros_like(T_obs, dtype=bool)
        print(sorted(cluster_pv))
        for j, cl in enumerate(clusters):
            if cluster_pv[j] <= 0.05:
                significant_mask[cl] = True

        avg_contrast = np.mean(rms, axis=0)
        sem_contrast = np.std(rms, axis=0) / np.sqrt(len(rms))
        times = np.linspace(0, 8, avg_contrast.shape[0])
        axs[i].axhline(0, ls='--', c='k')
        axs[i].plot(times, avg_contrast, color=condition_colors[i], label=condition)
        # axs[i].plot(times, avg_contrast/sem_contrast, color=condition_colors[i], label=condition)
        axs[i].fill_between(times, avg_contrast - sem_contrast, avg_contrast + sem_contrast, color=condition_colors[i], alpha=0.5, lw=0)
        mask = np.ma.masked_where(~significant_mask, (np.min(avg_contrast - sem_contrast) - np.abs(np.min(avg_contrast - sem_contrast)) / 100) * np.ones_like(T_obs))  # we mask what is not significant
        axs[i].plot(times, mask, 'k', linewidth=2)
        axs[i].set_title(condition)
    axs[-1].set_xlabel(f'Time (s)')
    axs[2].set_ylabel(f'Average contrast of {analysis_name} relative to baseline (percent)')
    # axs.legend()
    # Create custom legend entries, one per unique color
    # legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=7) for name, color in zip(subject_names, subject_colors)]
    # plt.legend(handles=legend_elements, loc="best", title="Subjects")

    # Adjust the layout so the plots don't overlap
    plt.tight_layout()
    plt.show()

    return


def plot_individual_electrode_scalar_temporal_regions_comparison(data_path, behavior_path, regions, first_parcel_label=True, analysis_name='abs'):
    """electrode-level temporal plot used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
    subject_names = []
    print(subject_paths)

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']

    rms_contrasts = [[[] for r in range(len(regions))] for c in range(len(conditions))]
    subject_id = [[] for c in range(len(conditions))]  # used for color

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for r, region in enumerate(regions):
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            rms_contrasts[cond][r].append(np.mean(rms[correct, c, :], axis=0) - np.mean(rms[incorrect, c, :], axis=0))
                            subject_id[cond].append(i)

    # fig, axs = plt.subplots(1, len(conditions), sharex=True, sharey=True)
    fig, axs = plt.subplots(len(conditions), 1, sharex=True, sharey=True)
    region_colors = plt.get_cmap('Set1').colors
    # Plot each condition's data in a separate subplot
    for i, (condition, rms) in enumerate(zip(conditions, rms_contrasts)):
        axs[i].axhline(0, ls='--', c='k')
        for r, region_rms in enumerate(rms):
            print(np.array(region_rms).shape)
            avg_contrast = np.mean(region_rms, axis=0)
            sem_contrast = np.std(region_rms, axis=0) / np.sqrt(len(region_rms))
            times = np.linspace(0, 8, avg_contrast.shape[0])
            # axs[i].plot(times, avg_contrast, color=region_colors[r], label=regions[r])
            axs[i].plot(times, avg_contrast/sem_contrast, color=region_colors[r], label=regions[r])
            # axs[i].fill_between(times, avg_contrast - sem_contrast, avg_contrast + sem_contrast, color=region_colors[r], alpha=0.5, lw=0)
        axs[i].set_title(condition)
    axs[0].legend()
    axs[-1].set_xlabel(f'Time (s)')
    axs[2].set_ylabel(f'Average contrast of {analysis_name} relative to baseline (percent)')
    # axs.legend()
    # Create custom legend entries, one per unique color
    # legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=7) for name, color in zip(subject_names, subject_colors)]
    # plt.legend(handles=legend_elements, loc="best", title="Subjects")

    # Adjust the layout so the plots don't overlap
    plt.tight_layout()
    plt.show()

    return

def plot_individual_electrode_scalar_temporal_regions_comparison_one_figure(data_path, behavior_path, regions, first_parcel_label=True, analysis_name='abs'):
    """electrode-level temporal plot used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
    del subject_paths[1]
    subject_names = []
    print(subject_paths)

    conditions = ['novelty', 'retrieval']

    rms_contrasts = [[[] for r in range(len(regions))] for c in range(len(conditions))]
    subject_id = [[] for c in range(len(conditions))]  # used for color

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for r, region in enumerate(regions):
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            rms_contrasts[cond][r].append(np.mean(rms[correct, c, :], axis=0) - np.mean(rms[incorrect, c, :], axis=0))
                            subject_id[cond].append(i)

    # fig, axs = plt.subplots(1, len(conditions), sharex=True, sharey=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    region_colors = plt.get_cmap('Set2').colors
    region_colors = ['#FF1100', '#3B2AF5']
    # region_colors = plt.get_cmap('tab20').colors
    # region_colors = [region_colors[13], region_colors[8], region_colors[12], region_colors[9]]
    # Plot each condition's data in a separate subplot

    # compute cluster stats
    mask_list = [] # Store masks for test between conditions
    mask_list_zero = [[] for _ in conditions]  # Store masks for condition vs. 0
    for r in range(len(rms_contrasts[0])):
        diff = np.array(rms_contrasts[0][r]) - np.array(rms_contrasts[1][r])
        pval = 0.05  # arbitrary
        df = len(diff) - 1  # degrees of freedom denominator
        thresh = scipy.stats.t.ppf(1 - pval / 2, df=df)  # t distribution two-tailed
        # thresh = scipy.stats.t.ppf(1 - pval, df=df)  # t distribution one-tailed
        print(thresh)
        T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(diff,
                                                                         threshold=thresh,
                                                                         n_permutations=10000, tail=0,
                                                                         adjacency=None, n_jobs=-1)
        print(T_obs.shape)
        print(clusters)
        print(cluster_pv)
        # Create a mask for significant clusters
        significant_mask = np.zeros_like(T_obs, dtype=bool)
        print(sorted(cluster_pv))
        for j, cl in enumerate(clusters):
            if cluster_pv[j] <= 0.05:
                significant_mask[cl] = True
        mask_list.append(significant_mask)

    # tests against 0
    for r in range(len(rms_contrasts[0])):
        for cond in range(len(conditions)):
            diff = np.array(rms_contrasts[cond][r])
            pval = 0.05  # arbitrary
            df = len(diff) - 1  # degrees of freedom denominator
            thresh = scipy.stats.t.ppf(1 - pval / 2, df=df)  # t distribution two-tailed
            # thresh = scipy.stats.t.ppf(1 - pval, df=df)  # t distribution one-tailed
            print(thresh)
            T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(diff,
                                                                             threshold=thresh,
                                                                             n_permutations=10000, tail=0,
                                                                             adjacency=None, n_jobs=-1)
            print(T_obs.shape)
            print(clusters)
            print(cluster_pv)
            # Create a mask for significant clusters
            significant_mask = np.zeros_like(T_obs, dtype=bool)
            print(sorted(cluster_pv))
            for j, cl in enumerate(clusters):
                if cluster_pv[j] <= 0.05:
                    significant_mask[cl] = True
            mask_list_zero[cond].append(significant_mask)

    # plot
    current = 0
    for i, (condition, rms) in enumerate(zip(conditions, rms_contrasts)):
        axs[i].axhline(0, ls='--', c='k')
        for r, region_rms in enumerate(rms):
            print(np.array(region_rms).shape)
            avg_contrast = np.mean(region_rms, axis=0)
            sem_contrast = np.std(region_rms, axis=0) / np.sqrt(len(region_rms))
            times = np.linspace(0, 8, avg_contrast.shape[0])
            # axs.plot(times, avg_contrast/sem_contrast, color=region_colors[r], label=regions[r])
            axs[r].plot(times, avg_contrast, color=region_colors[i], label=condition + str(regions[r]), lw=3)
            axs[r].fill_between(times, avg_contrast - sem_contrast, avg_contrast + sem_contrast, color=region_colors[i], alpha=0.5, lw=0)
            current += 1
        axs[i].set_xlabel(f'Time (s)', fontsize=17)
        axs[i].tick_params(axis='y', labelsize=17)
        axs[i].tick_params(axis='x', labelsize=17)
    axs[0].set_ylabel(f'Percent change of activity', fontsize=17)
    axs[0].legend()
    axs[1].legend()
    for r in range(len(rms_contrasts[0])):
        mask = mask_list[r]
        print(mask.shape)
        print(axs[r].get_ylim()[0])
        # mask = np.ma.masked_where(~mask, (axs[r].get_ylim()[0]) * np.ones_like(times))  # we mask what is not significant
        mask = np.ma.masked_where(~mask, -10 * np.ones_like(times))  # we mask what is not significant
        axs[r].plot(times, mask, 'k', linewidth=3)
        for cond in range(len(conditions)):
            mask = mask_list_zero[cond][r]
            # mask = np.ma.masked_where(~mask, (axs[r].get_ylim()[0]) * np.ones_like(times))  # we mask what is not significant
            mask = np.ma.masked_where(~mask, -(7+cond*1.5) * np.ones_like(times))  # we mask what is not significant
            axs[r].plot(times, mask, color=region_colors[cond], linewidth=3)

    # Adjust the layout so the plots don't overlap
    plt.tight_layout()
    plt.show()

    return


def plot_individual_electrode_scalar_temporal_regions_comparison_chatgpt(data_path, behavior_path, regions,
                                                                 first_parcel_label=True, analysis_name='abs'):
    """electrode-level temporal plot used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
    subject_names = []
    print(subject_paths)

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']

    rms_contrasts = [[] for c in range(len(conditions))]
    subject_id = [[] for c in range(len(conditions))]  # used for color

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]

        # Define behavior-based conditions
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))][
            'stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))][
            'stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding],
                                                  [correct_novelty, incorrect_novelty],
                                                  [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            rms_contrasts[cond].append(
                                np.mean(rms[correct, c, :], axis=0) - np.mean(rms[incorrect, c, :], axis=0))
                            subject_id[cond].append(i)

    fig, axs = plt.subplots(len(conditions), 1, sharex=True, sharey=True, figsize=(10, 15))
    condition_colors = plt.get_cmap('Set1').colors

    for i, (condition, rms) in enumerate(zip(conditions, rms_contrasts)):
        times = np.linspace(0, 8, np.array(rms[0]).shape[0])

        # Plot each region's data in a separate trace
        for region_idx, region in enumerate(regions):
            region_traces = [contrast for j, contrast in enumerate(rms) if
                             any(region in parcellations[j] for parcellation in parcellations)]
            if region_traces:
                avg_contrast = np.mean(region_traces, axis=0)
                sem_contrast = np.std(region_traces, axis=0) / np.sqrt(len(region_traces))
                axs[i].plot(times, avg_contrast, label=f'{region}',
                            color=condition_colors[region_idx % len(condition_colors)])
                axs[i].fill_between(times, avg_contrast - sem_contrast, avg_contrast + sem_contrast, alpha=0.3)

        axs[i].set_title(f'{condition} condition')
        axs[i].axhline(0, ls='--', color='black')

    axs[-1].set_xlabel('Time (s)')
    axs[len(conditions) // 2].set_ylabel('Average contrast')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


def plot_individual_electrode_power_stim_freq(data_path, regions, first_parcel_label=True):
    """electrode-level test (FDR corrected) used to plot rms, gamma power or other scalar saved with the same format."""
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_power_stim_freq.pkl')))
    subject_names = []
    print(subject_paths)

    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']

    rms_contrasts = [[] for c in range(len(rates))]
    perm_contrasts = [[] for c in range(len(rates))]
    subject_id = [[] for c in range(len(rates))]  # used for color

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['power']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label==True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label=="advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        for cond, rate in enumerate(rates):
            rate_epochs = [e for e in range(len(rms_obj['stim_name'])) if rate in rms_obj['stim_name'][e]]
            if len(rate_epochs) > 0:
                # ch_index = []
                for c in range(rms.shape[1]):
                    for reg, region in enumerate(regions):
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            # ch_index.append(c)
                            # rate_rms = np.mean(rms[:, c], axis=1)
                            rms_contrasts[cond].append(np.mean(rms[rate_epochs, c]))
                            # perm_contrasts[cond].append(perm_diffs)
                            subject_id[cond].append(i)
                            # subject_id[cond].append(reg)  # color with regions

    fig, axs = plt.subplots(1, 1, sharey=True)
    subject_colors = plt.get_cmap('Set1').colors
    # subject_colors = plt.get_cmap('tab20').colors
    print(len(subject_colors))
    # Plot each condition's data in a separate subplot
    for i, (condition, rms, perms, s_id) in enumerate(zip(rates, rms_contrasts, perm_contrasts, subject_id)):
        # population_p = ttest_1samp(rms, 0).pvalue
        t_obs = np.mean(rms) / (np.std(rms, ddof=1) / np.sqrt(len(rms)))
        # t_perms = np.mean(perms, axis=0) / (np.std(perms, axis=0, ddof=1) / np.sqrt(np.array(perms).shape[0]))
        # population_p = np.sum(np.abs(t_perms) >= np.abs(t_obs)) / np.array(perms).shape[1]
        # population_p = np.sum(np.mean(perms, axis=0) >= np.mean(rms)) / np.array(perms).shape[1]
        # print(population_p)
        # axs.bar(i, np.mean(rms), yerr=np.std(rms)/np.sqrt(len(rms)), capsize=5, color=('orange' if population_p < 0.05 else 'skyblue'))
        axs.bar(i, np.mean(rms), yerr=np.std(rms)/np.sqrt(len(rms)), capsize=5, color='skyblue')
        axs.axhline(0, color='gray', linestyle='--', linewidth=1.5)

        # Add individual sample points as scatter dots
        # colors = ['red' if p < 0.05 else 'black' for p in pval]
        colors = [subject_colors[s] for s in s_id]
        x_values = np.full(len(rms), i) + np.random.uniform(-0.1, 0.1, len(rms))  # Jitter for clarity
        axs.scatter(x_values, rms, color=colors, alpha=1, s=20, label='Samples')
        axs.set_ylabel(f'Average contrast of power at stim freq relative to baseline (percent)')
    axs.set_xticks([x for x in range(len(rates))], rates)
    # Create custom legend entries, one per unique color
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=7) for name, color in zip(subject_names, subject_colors)]
    # legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=7) for name, color in zip(regions, subject_colors)]  # colors with regions
    plt.legend(handles=legend_elements, loc="best", title="Subjects")

    # Adjust the layout so the plots don't overlap
    # plt.tight_layout()
    plt.show()

    return


def save_individual_electrode_scalar_3d(data_path, behavior_path, save_path, analysis_name='rms'):
    """electrode-level test (FDR corrected) used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
    subject_names = []
    print(subject_paths)

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']

    tvalues = [[] for c in range(len(conditions))]
    pvalues = [[] for c in range(len(conditions))]
    infos = [[] for c in range(len(conditions))]  # used for color

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]

        for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                infos[cond].append(rms_obj['info'])
                for c in tqdm(range(rms.shape[1])):
                    test_result = ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False, permutations=1000, random_state=None, alternative='two-sided')  # testing of individual electrodes
                    tvalues[cond].append(test_result.statistic)
                    pvalues[cond].append(test_result.pvalue)

    # FDR correction
    # for c in range(len(pvalues)):
    #     _, pval = mne.stats.fdr_correction(pvalues[c], alpha=0.05, method='indep')
    #     pvalues[c] = pval

    np.save(os.path.join(save_path, f"{analysis_name}_contrast_results.npy"), {'condition': conditions, 'tvalues': tvalues, 'pvalues': pvalues, 'infos': infos})

    return


def save_individual_electrode_scalar_correct_incorrect_3d(data_path, behavior_path, save_path, analysis_name='rms'):
    """electrode-level test used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
    subject_names = []
    print(subject_paths)

    tvalues = []
    pvalues = []
    parcellations = []

    all_chan_id = 0
    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]

        info = rms_obj['info']
        new_ch_names = {}
        for c in range(info['nchan']):
            new_ch_names[info.ch_names[c]] = str(all_chan_id)
            all_chan_id += 1
        montage = info.get_montage()
        montage.rename_channels(new_ch_names)
        if i == 0:
            global_montage = montage
        else:
            global_montage = global_montage + montage


        correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_all]
        incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in incorrect_all]
        if (len(correct) > 0) & (len(incorrect) > 0):
            for c in tqdm(range(rms.shape[1])):
                test_result = ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False, permutations=1000, random_state=None, alternative='two-sided')  # testing of individual electrodes
                tvalues.append(test_result.statistic)
                pvalues.append(test_result.pvalue)
                parcellations.append(rms_obj['ch_parcellations'][c])

    # FDR correction
    # for c in range(len(pvalues)):
    #     _, pval = mne.stats.fdr_correction(pvalues[c], alpha=0.05, method='indep')
    #     pvalues[c] = pval

    # info = mne.create_info(global_montage.ch_names, info['sfreq'], ch_types='seeg')
    # info.set_montage(global_montage)
    global_montage.save(os.path.join(save_path, "global_montage.fif"), overwrite=True)
    np.save(os.path.join(save_path, f"{analysis_name}_contrast_results_correct_incorrect.npy"), {'tvalues': tvalues, 'pvalues': pvalues, 'parcellations': parcellations})

    return


def save_individual_electrode_scalar_novelty_retrieval_3d(data_path, behavior_path, save_path, analysis_name='rms'):
    """electrode-level test used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
    subject_names = []
    print(subject_paths)

    tvalues_diff = []
    parcellations = []

    all_chan_id = 0
    first_montage_done = False
    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]

        correct_novelty_epochs = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_novelty]
        incorrect_novelty_epochs = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in incorrect_novelty]
        correct_retrieval_epochs = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_retrieval]
        incorrect_retrieval_epochs = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in incorrect_retrieval]
        if (len(correct_novelty_epochs) > 0) & (len(incorrect_novelty_epochs) > 0) & (len(correct_retrieval_epochs) > 0) & (len(incorrect_retrieval_epochs) > 0):
            info = rms_obj['info']
            new_ch_names = {}
            for c in range(info['nchan']):
                new_ch_names[info.ch_names[c]] = str(all_chan_id)
                all_chan_id += 1
            montage = info.get_montage()
            montage.rename_channels(new_ch_names)
            if first_montage_done:
                global_montage = global_montage + montage
            else:
                global_montage = montage
                first_montage_done = True

            for c in tqdm(range(rms.shape[1])):
                test_novelty = ttest_ind(rms[correct_novelty_epochs, c], rms[incorrect_novelty_epochs, c], axis=0, equal_var=False, permutations=1000, random_state=None, alternative='two-sided')  # testing of individual electrodes
                test_retrieval = ttest_ind(rms[correct_retrieval_epochs, c], rms[incorrect_retrieval_epochs, c], axis=0, equal_var=False, permutations=1000, random_state=None, alternative='two-sided')  # testing of individual electrodes
                tvalues_diff.append(np.abs(test_novelty.statistic) - np.abs(test_retrieval.statistic))  # we take abs to compare the effect strengths
                parcellations.append(rms_obj['ch_parcellations'][c])

    # FDR correction
    # for c in range(len(pvalues)):
    #     _, pval = mne.stats.fdr_correction(pvalues[c], alpha=0.05, method='indep')
    #     pvalues[c] = pval

    # info = mne.create_info(global_montage.ch_names, info['sfreq'], ch_types='seeg')
    # info.set_montage(global_montage)
    global_montage.save(os.path.join(save_path, "global_montage_novelty_retrieval.fif"), overwrite=True)
    np.save(os.path.join(save_path, f"{analysis_name}_contrast_results_novelty_retrieval.npy"), {'tvalues_diff': tvalues_diff, 'parcellations': parcellations})

    return


def decode_surprise_from_comodulogram(data_path, behavior_path, lowest_p_path, first_parcel_label=True, regions=None):
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*epochs_comodulogram.pkl')))
    print(subject_paths)

    lowest_p_value_dict = np.load(lowest_p_path, allow_pickle=True)

    save_path = data_path
    os.makedirs(os.path.join(save_path, 'decoding'), exist_ok=True)

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        print(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        surprise = behavior['surprise_mean_paulrnn'].to_numpy()
        # surprise = behavior['time_interval'].to_numpy()
        print(surprise.shape)
        surprise_stim_name = behavior['stim_name'].tolist()

        pac_obj = np.load(subject_path, allow_pickle=True)
        pac = pac_obj['pac']
        parcellations = pac_obj['ch_parcellations']
        if first_parcel_label:
            parcellations = [[parcel[0]] for parcel in parcellations]

        subject_p = lowest_p_value_dict[subject_name]
        subject_p = [p_list for p_list in subject_p if len(p_list) > 0]  # keep only existing conditions
        for p_list in subject_p:
            assert len(p_list) == len(parcellations)

        # select electrodes
        chan_ind = []
        for p, parcel in enumerate(parcellations):
            if any(ch_parcel in regions for ch_parcel in parcel):
                chan_ind.append(p)
        pac = pac[:, chan_ind, :, :]
        print(np.array(pac_obj['ch_names'])[chan_ind])

        # reorder epochs
        epochs_ind = []
        surprise_to_keep = []
        for e in range(pac.shape[0]):
            if len(np.where(np.array(pac_obj['stim_name']) == surprise_stim_name[e])[0]) > 0:
                epochs_ind.append(np.where(np.array(pac_obj['stim_name']) == surprise_stim_name[e])[0][0])
                surprise_to_keep.append(e)
        pac = pac[epochs_ind, :, :, :]
        surprise = surprise[surprise_to_keep]

        for c, c_ind in enumerate(chan_ind):
            if any([p_list[c_ind] < 0.05 for p_list in subject_p]):  # if this channel is significant in any memory condition
                y = pac[:, c, :, :].reshape((pac.shape[0], -1))  # flatten driving and signal frequencies
                X = surprise.reshape(-1, 1)

                # # Create a mask where y is not NaN
                # # mask = ~np.isnan(y)
                # mask = ~np.isnan(X)
                # X = X[mask]
                # y = y[mask]

                # regression = RidgeCV(alphas=np.logspace(1, 5, 10),
                #                      fit_intercept=True, scoring='r2', cv=None, alpha_per_target=True
                #                      )
                # regression.fit(X, y)
                # score = regression.best_score_
                # print(regression.alpha_)
                # score = regression.score(X, y)
                # print(score)

                # Ridge on training
                score = np.zeros(y.shape[1])
                for f in range(y.shape[1]):
                    regression = Ridge(alpha=1.0, fit_intercept=True)
                    regression.fit(X, y[:, f])
                    # score[f] = regression.score(X, y[:, f])
                    score[f] = regression.coef_[0]

                fig = plt.figure()
                score = score.reshape((pac.shape[2], pac.shape[3]))
                vmax = np.max(np.abs(score))
                X, Y = np.meshgrid(pac_obj['low_freqs'], pac_obj['high_freqs'])
                plt.pcolormesh(X, Y, score.T, cmap='Oranges', vmin=0, vmax=vmax)
                # plt.xscale('log')
                plt.yscale('log')
                plt.gca().yaxis.set_major_formatter(ScalarFormatter())
                plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
                plt.xlabel('Driver frequency (Hz)')
                plt.ylabel('Signal frequency (Hz)')
                cbar = plt.colorbar()
                cbar.set_label('r2', rotation=270)
                plt.title(parcellations[c_ind])
                plt.show()
                # fig.savefig(os.path.join(save_path, f'signif_figures_{conditions[cond]}', subject_name + f'_channel_{c_ind}.svg'))
                plt.close()





    # # Calculate and save average t-values for each condition and region
    # for cond in range(len(conditions)):
    #     for region in regions_list[cond]:
    #         if signif_counts[cond][region] > 0:
    #             region_pac = np.array(avg_t_values[cond][region])
    #             # region_pac = np.mean(region_pac, axis=0) / (np.std(region_pac, axis=0) / np.sqrt(region_pac.shape[0]))  # Calculate the t-values across channels
    #             region_pac = np.mean(region_pac, axis=0)
    #
    #             fig = plt.figure()
    #             vmax = np.max(np.abs(region_pac))
    #             # vmax = 8
    #             X, Y = np.meshgrid(pac_obj['low_freqs'], pac_obj['high_freqs'])
    #             plt.pcolormesh(X, Y, region_pac.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    #             # plt.pcolormesh(X, Y, avg_t_values[cond][region].T, cmap='RdBu_r')
    #             plt.yscale('log')
    #             plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    #             plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
    #             plt.xlabel('Driver frequency (Hz)')
    #             plt.ylabel('Signal frequency (Hz)')
    #             cbar = plt.colorbar()
    #             cbar.set_label('t-values', rotation=270)
    #             plt.title(f'{conditions[cond]} t-values {region}')
    #             fig.savefig(os.path.join(save_path, f'signif_figures_{conditions[cond]}', f'Average_{region}.svg'))
    #
    #             region_pac = np.array(avg_t_values[cond][region])
    #             region_pac = np.mean(region_pac, axis=0) / (np.std(region_pac, axis=0) / np.sqrt(region_pac.shape[0]))  # Calculate the t-values across channels
    #             fig = plt.figure()
    #             # vmax = np.max(np.abs(region_pac))
    #             vmax = 8
    #             X, Y = np.meshgrid(pac_obj['low_freqs'], pac_obj['high_freqs'])
    #             plt.pcolormesh(X, Y, region_pac.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    #             # plt.pcolormesh(X, Y, avg_t_values[cond][region].T, cmap='RdBu_r')
    #             plt.yscale('log')
    #             plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    #             plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
    #             plt.xlabel('Driver frequency (Hz)')
    #             plt.ylabel('Signal frequency (Hz)')
    #             cbar = plt.colorbar()
    #             cbar.set_label('t-values', rotation=270)
    #             plt.title(f'{conditions[cond]} t-values {region}')
    #             fig.savefig(os.path.join(save_path, f'signif_figures_{conditions[cond]}', f'tval_{region}.svg'))

    return


def encode_scalar_from_surprise_and_save_data(data_path_list, analysis_name_list, behavior_path, first_parcel_label=True, regions=None):
    """Also save for GLMM analysis. data_path_list is a list of folders (e.g, one for RMS, one for gamma etc), analysis_name_list is a list of the same length"""

    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path_list[0])
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name_list[0]}.pkl')))
    print(subject_paths)

    save_path = data_path_list[0]
    os.makedirs(os.path.join(save_path, 'encoding'), exist_ok=True)

    # scores = []
    # linear_coeffs = []
    # quadratic_coeffs = []

    # to save data in long format
    subject_id = []
    electrode_id = []
    electrode_region = []
    surprise_feature = []
    positive_surprise_feature = []
    surprise_windows_feature = [[] for _ in range(8)]
    uncertainty_feature = []
    uncertainty_windows_feature = [[] for _ in range(8)]
    predicted_density_feature = []
    stim_list = []
    note_rate = []
    onset = []
    rep = []
    correct_list = []
    trials_since_rep1_list = []
    rms_target = []
    rms_windows_target = [[] for _ in range(8)]
    theta_target = []
    power_stim_freq_target = []
    high_gamma_target = []
    pac_target = []
    pac_windows_target = [[] for _ in range(8)]
    pac_stim_freq_target = []
    pac_stim_freq_windows_target = [[] for _ in range(8)]
    phase_coherence_stim_freq_target = []
    phase_coherence_stim_freq_windows_target = [[] for _ in range(8)]
    pac4_target = []
    pac4_windows_target = [[] for _ in range(8)]
    pac5_target = []
    pac5_windows_target = [[] for _ in range(8)]
    pac6_target = []
    pac6_windows_target = [[] for _ in range(8)]
    pac7_target = []
    pac7_windows_target = [[] for _ in range(8)]
    pac8_target = []
    pac8_windows_target = [[] for _ in range(8)]

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        print(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        surprise = behavior['surprise_mean_paulrnn'].to_numpy()
        positive_surprise = behavior['surprise_possum_paulrnn'].to_numpy()
        surprise_windows = [behavior[f'surprise_{t}_{t+1}s'].to_numpy() for t in range(8)]
        uncertainty = behavior['entropy_pitch_paulrnn'].to_numpy()
        uncertainty_windows = [behavior[f'uncertainty_{t}_{t+1}s'].to_numpy() for t in range(8)]
        predicted_density = behavior['entropy_timing_paulrnn'].to_numpy()
        correct = behavior['correct'].to_numpy()
        trials_since_rep1 = behavior['trial_since_rep1'].to_numpy()
        print(surprise.shape)
        surprise_stim_name = behavior['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name_list[0]]
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label==True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label=="advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        # select electrodes
        chan_ind = []
        kept_parcels = []
        if regions is not None:
            for p, parcel in enumerate(parcellations):
                if any(ch_parcel in regions for ch_parcel in parcel):
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])
        else:  # with regions=None, we keep all channels but the ones in go_next
            for p, parcel in enumerate(parcellations):
                if parcel[0] not in go_next:
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])

        rms = rms[:, chan_ind]
        print(np.array(rms_obj['ch_names'])[chan_ind])

        rms_obj = np.load(os.path.join(data_path_list[0], subject_name + '_epochs_rms_windows.pkl'), allow_pickle=True)
        rms_windows = rms_obj['rms']
        rms_windows = [win[:, chan_ind] for win in rms_windows]

        # load and select channels for power
        power_obj = np.load(os.path.join(data_path_list[1], subject_name + '_epochs_power.pkl'), allow_pickle=True)
        power = power_obj['power']
        theta = power[:, chan_ind]

        power_obj = np.load(os.path.join(data_path_list[11], subject_name + '_epochs_power_stim_freq.pkl'), allow_pickle=True)
        power = power_obj['power']
        power_stim_freq = power[:, chan_ind]

        power_obj = np.load(os.path.join(data_path_list[2], subject_name + '_epochs_power.pkl'), allow_pickle=True)
        power = power_obj['power']
        high_gamma = power[:, chan_ind]

        pac_obj = np.load(os.path.join(data_path_list[3], subject_name + '_epochs_pac.pkl'), allow_pickle=True)
        pac = pac_obj['pac']
        pac = pac[:, chan_ind]

        pac_obj = np.load(os.path.join(data_path_list[3], subject_name + '_epochs_pac_windows.pkl'), allow_pickle=True)
        pac_windows = pac_obj['pac']
        pac_windows = [win[:, chan_ind] for win in pac_windows]

        pac_obj = np.load(os.path.join(data_path_list[4], subject_name + '_epochs_pac.pkl'), allow_pickle=True)
        pac_stim_freq = pac_obj['pac']
        pac_stim_freq = pac_stim_freq[:, chan_ind]

        pac_obj = np.load(os.path.join(data_path_list[4], subject_name + '_epochs_pac_windows.pkl'), allow_pickle=True)
        pac_stim_freq_windows = pac_obj['pac']
        pac_stim_freq_windows = [win[:, chan_ind] for win in pac_stim_freq_windows]

        phase_coherence_obj = np.load(os.path.join(data_path_list[12], subject_name + '_epochs_phase_coherence.pkl'), allow_pickle=True)
        phase_coherence_stim_freq = phase_coherence_obj['phase_coherence']
        phase_coherence_stim_freq = phase_coherence_stim_freq[:, chan_ind]

        phase_coherence_obj = np.load(os.path.join(data_path_list[12], subject_name + '_epochs_phase_coherence_windows.pkl'), allow_pickle=True)
        phase_coherence_stim_freq_windows = phase_coherence_obj['phase_coherence']
        phase_coherence_stim_freq_windows = [win[:, chan_ind] for win in phase_coherence_stim_freq_windows]

        pac_obj = np.load(os.path.join(data_path_list[5], subject_name + '_epochs_pac.pkl'), allow_pickle=True)
        pac4 = pac_obj['pac']
        pac4 = pac4[:, chan_ind]
        pac_obj = np.load(os.path.join(data_path_list[6], subject_name + '_epochs_pac.pkl'), allow_pickle=True)
        pac5 = pac_obj['pac']
        pac5 = pac5[:, chan_ind]
        pac_obj = np.load(os.path.join(data_path_list[7], subject_name + '_epochs_pac.pkl'), allow_pickle=True)
        pac6 = pac_obj['pac']
        pac6 = pac6[:, chan_ind]
        pac_obj = np.load(os.path.join(data_path_list[8], subject_name + '_epochs_pac.pkl'), allow_pickle=True)
        pac7 = pac_obj['pac']
        pac7 = pac7[:, chan_ind]
        pac_obj = np.load(os.path.join(data_path_list[9], subject_name + '_epochs_pac.pkl'), allow_pickle=True)
        pac8 = pac_obj['pac']
        pac8 = pac8[:, chan_ind]

        pac_obj = np.load(os.path.join(data_path_list[10], subject_name + '_epochs_pac_theta_bands_windows.pkl'), allow_pickle=True)
        pac_theta_bands = pac_obj['pac']
        pac4_windows = [win[:, chan_ind] for win in pac_theta_bands[0]]
        pac5_windows = [win[:, chan_ind] for win in pac_theta_bands[1]]
        pac6_windows = [win[:, chan_ind] for win in pac_theta_bands[2]]
        pac7_windows = [win[:, chan_ind] for win in pac_theta_bands[3]]
        pac8_windows = [win[:, chan_ind] for win in pac_theta_bands[4]]

        if len(chan_ind) > 0:

            # reorder and select epochs
            epochs_ind = []
            stim_names = []
            surprise_to_keep = []
            for e in range(len(surprise_stim_name)):
                if len(np.where(np.array(rms_obj['stim_name']) == surprise_stim_name[e])[0]) > 0:
                    epochs_ind.append(np.where(np.array(rms_obj['stim_name']) == surprise_stim_name[e])[0][0])
                    stim_names.append(surprise_stim_name[e])
                    surprise_to_keep.append(e)
            rms = rms[epochs_ind, :]
            rms_windows = [win[epochs_ind, :] for win in rms_windows]
            theta = theta[epochs_ind, :]
            power_stim_freq = power_stim_freq[epochs_ind, :]
            high_gamma = high_gamma[epochs_ind, :]
            pac = pac[epochs_ind, :]
            pac_windows = [win[epochs_ind, :] for win in pac_windows]
            pac_stim_freq = pac_stim_freq[epochs_ind, :]
            pac_stim_freq_windows = [win[epochs_ind, :] for win in pac_stim_freq_windows]
            phase_coherence_stim_freq = phase_coherence_stim_freq[epochs_ind, :]
            phase_coherence_stim_freq_windows = [win[epochs_ind, :] for win in phase_coherence_stim_freq_windows]
            pac4 = pac4[epochs_ind, :]
            pac5 = pac5[epochs_ind, :]
            pac6 = pac6[epochs_ind, :]
            pac7 = pac7[epochs_ind, :]
            pac8 = pac8[epochs_ind, :]
            pac4_windows = [win[epochs_ind, :] for win in pac4_windows]
            pac5_windows = [win[epochs_ind, :] for win in pac5_windows]
            pac6_windows = [win[epochs_ind, :] for win in pac6_windows]
            pac7_windows = [win[epochs_ind, :] for win in pac7_windows]
            pac8_windows = [win[epochs_ind, :] for win in pac8_windows]
            surprise = surprise[surprise_to_keep]
            positive_surprise = positive_surprise[surprise_to_keep]
            surprise_windows = [surp[surprise_to_keep] for surp in surprise_windows]
            uncertainty = uncertainty[surprise_to_keep]
            uncertainty_windows = [surp[surprise_to_keep] for surp in uncertainty_windows]
            predicted_density = predicted_density[surprise_to_keep]
            correct = correct[surprise_to_keep]
            trials_since_rep1 = trials_since_rep1[surprise_to_keep]

            # X = np.zeros((surprise.shape[0], 2))
            # X[:, 0] = surprise
            # X[:, 1] = surprise**2
            # X = (X - X.mean(axis=0)) / X.std(axis=0)

            for chan in range(rms.shape[1]):
                # save data
                rms_target += rms[:, chan].tolist()
                theta_target += theta[:, chan].tolist()
                power_stim_freq_target += power_stim_freq[:, chan].tolist()
                high_gamma_target += high_gamma[:, chan].tolist()
                pac_target += pac[:, chan].tolist()
                pac_stim_freq_target += pac_stim_freq[:, chan].tolist()
                phase_coherence_stim_freq_target += phase_coherence_stim_freq[:, chan].tolist()
                pac4_target += pac4[:, chan].tolist()
                pac5_target += pac5[:, chan].tolist()
                pac6_target += pac6[:, chan].tolist()
                pac7_target += pac7[:, chan].tolist()
                pac8_target += pac8[:, chan].tolist()
                surprise_feature += surprise.tolist()
                positive_surprise_feature += positive_surprise.tolist()
                for t in range(8):
                    surprise_windows_feature[t] += surprise_windows[t].tolist()
                    uncertainty_windows_feature[t] += uncertainty_windows[t].tolist()
                    pac_windows_target[t] += pac_windows[t][:, chan].tolist()
                    pac_stim_freq_windows_target[t] += pac_stim_freq_windows[t][:, chan].tolist()
                    phase_coherence_stim_freq_windows_target[t] += phase_coherence_stim_freq_windows[t][:, chan].tolist()
                    pac4_windows_target[t] += pac4_windows[t][:, chan].tolist()
                    pac5_windows_target[t] += pac5_windows[t][:, chan].tolist()
                    pac6_windows_target[t] += pac6_windows[t][:, chan].tolist()
                    pac7_windows_target[t] += pac7_windows[t][:, chan].tolist()
                    pac8_windows_target[t] += pac8_windows[t][:, chan].tolist()
                    rms_windows_target[t] += rms_windows[t][:, chan].tolist()
                uncertainty_feature += uncertainty.tolist()
                predicted_density_feature += predicted_density.tolist()
                correct_list += correct.tolist()
                trials_since_rep1_list += trials_since_rep1.tolist()
                stim_list += [stim for stim in stim_names]
                rep += [float(stim.split('_')[1][3]) for stim in stim_names]
                note_rate += [stim.split('_')[2][:5] for stim in stim_names]
                onset += [float(stim.split('_')[4][5]) for stim in stim_names]
                subject_id += [subject_name for _ in range(rms[:, chan].shape[0])]
                electrode_id += [rms_obj['ch_names'][chan_ind[chan]] for _ in range(rms.shape[0])]
                electrode_region += [kept_parcels[chan] for _ in range(rms.shape[0])]

                # y = rms[:, chan]

                # Create a mask where y is not NaN
                # mask = ~np.isnan(y)
                # # mask = ~np.isnan(X)
                # X = X[mask]
                # y = y[mask]

                # y = (y - y.mean(axis=0)) / y.std(axis=0)

                # regression = RidgeCV(alphas=np.logspace(1, 10, 50),
                #                      fit_intercept=True, scoring='r2', cv=None, alpha_per_target=False
                #                      )

                # regression = LinearRegression()
                # y = np.array(['correct' if yi==1 else 'incorrect' for yi in y])
                # regression = RidgeClassifierCV(alphas=np.logspace(-10, 10, 50), fit_intercept=True, class_weight='balanced',
                #                                scoring='accuracy', store_cv_values=False, cv=5)
                # regression.fit(X, y)
                # print(regression.predict(X))
                # print(y)
                # print('accur:', accuracy_score(y, regression.predict(X)))
                # print(regression.cv_values_.mean(axis=0))
                # print("alpha:", regression.alpha_)
                # score = regression.best_score_
                # print("cv score:", score)
                # score = regression.score(X, y)
                # print("score training:", score)
                #
                # scores.append(score)
                # linear_coeffs.append(regression.coef_[0])
                # quadratic_coeffs.append(regression.coef_[1])

                # perm_scores = np.zeros(101)
                # perm_scores[-1] = score
                # for p in tqdm(range(100)):
                #     y = np.random.permutation(y)
                #     regression = RidgeCV(alphas=np.logspace(1, 10, 50),
                #             fit_intercept=True, scoring='r2', cv=None, alpha_per_target=False
                #             )
                #     # regression = LinearRegression()
                #     regression.fit(X, y)
                #     perm_scores[p] = regression.best_score_
                #     # perm_scores[p] = regression.score(X, y)
                # # print("Permutation results:")
                # # print(perm_scores)
                # pval = np.mean(perm_scores >= score)
                # print("p =", pval)

    # save_data
    df = {'subject': subject_id, 'electrode': electrode_id, 'electrode_region': electrode_region, 'stim_name': stim_list, 'surprise': surprise_feature, 'positive_surprise': positive_surprise_feature, 'uncertainty': uncertainty_feature, 'predicted_density': predicted_density_feature, 'rep': rep, 'note_rate': note_rate, 'onset': onset, 'correct': correct_list, 'trials_since_rep1': trials_since_rep1_list, 'rms': rms_target, 'theta': theta_target, 'power_stim_freq':power_stim_freq_target, 'high_gamma': high_gamma_target, 'pac': pac_target, 'pac_stim_freq': pac_stim_freq_target, 'phase_coherence_stim_freq': phase_coherence_stim_freq_target, 'pac4': pac4_target, 'pac5': pac5_target, 'pac6': pac6_target, 'pac7': pac7_target, 'pac8': pac8_target}
    for t in range(8):
        df[f'surprise_{t}_{t+1}s'] = surprise_windows_feature[t]
        df[f'uncertainty_{t}_{t+1}s'] = uncertainty_windows_feature[t]
        df[f'rms_{t}_{t+1}s'] = rms_windows_target[t]
        df[f'pac_{t}_{t+1}s'] = pac_windows_target[t]
        df[f'pac_stim_freq_{t}_{t+1}s'] = pac_stim_freq_windows_target[t]
        df[f'phase_coherence_stim_freq_{t}_{t+1}s'] = phase_coherence_stim_freq_windows_target[t]
    thetas = [4, 5, 6, 7, 8]
    for i, target in enumerate([pac4_windows_target, pac5_windows_target, pac6_windows_target, pac7_windows_target, pac8_windows_target]):
        for t in range(8):
            df[f'pac{thetas[i]}_{t}_{t + 1}s'] = target[t]
    df = pd.DataFrame(df)

    # Create a dictionary that maps each stimulus to the 'correct' value from rep 2
    df['stim_name'] = df['stim_name'].str[14:26]
    correct2_map = df[df['rep'] == 2].set_index(['subject', 'stim_name'])['correct'].to_dict()
    # Add 'correct2' column by mapping based on the stimulus and using the correct2_map for rep 1 rows
    df['correct2'] = df.apply(lambda row: correct2_map.get((row['subject'], row['stim_name'])) if row['rep'] == 1 else row['correct'], axis=1)

    df.to_csv(os.path.join(save_path, "encoding", "encoding_data.csv"), index=False)

    # print(len(scores))
    # print(len(linear_coeffs))
    # print(len(quadratic_coeffs))
    # plt.hist(scores, bins=100)
    # plt.figure()
    # plt.hist(linear_coeffs, bins=100)
    # plt.figure()
    # plt.hist(quadratic_coeffs, bins=100)
    # plt.show()

    return



def save_histogram_with_behavior(data_path, behavior_path, first_parcel_label=True, regions=None):
    """Save the gamma by phase histogram with behavior."""

    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_amplitude_by_phase_histogram.pkl')))
    print(subject_paths)

    save_path = data_path

    # to save data in long format
    subject_id = []
    electrode_id = []
    surprise_feature = []
    uncertainty_feature = []
    predicted_density_feature = []
    stim_list = []
    note_rate = []
    onset = []
    rep = []
    correct_list = []
    histogram_target = [[] for bin in range(18)]
    pac_stim_freq_target = []

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        print(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        surprise = behavior['surprise_possum_paulrnn'].to_numpy()
        uncertainty = behavior['entropy_pitch_paulrnn'].to_numpy()
        predicted_density = behavior['entropy_timing_paulrnn'].to_numpy()
        correct = behavior['correct'].to_numpy()
        print(surprise.shape)
        surprise_stim_name = behavior['stim_name'].tolist()

        histogram_obj = np.load(subject_path, allow_pickle=True)
        histogram = histogram_obj['histogram']
        parcellations = histogram_obj['ch_parcellations']
        if first_parcel_label==True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label=="advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        # select electrodes
        chan_ind = []
        for p, parcel in enumerate(parcellations):
            if any(ch_parcel in regions for ch_parcel in parcel):
                chan_ind.append(p)
        histogram = histogram[:, chan_ind, :]
        print(np.array(histogram_obj['ch_names'])[chan_ind])

        pac_obj = np.load(os.path.join(data_path, subject_name + '_epochs_pac.pkl'), allow_pickle=True)
        pac_stim_freq = pac_obj['pac']
        pac_stim_freq = pac_stim_freq[:, chan_ind]

        if len(chan_ind) > 0:
            # reorder and select epochs
            epochs_ind = []
            stim_names = []
            surprise_to_keep = []
            for e in range(len(surprise_stim_name)):
                if len(np.where(np.array(histogram_obj['stim_name']) == surprise_stim_name[e])[0]) > 0:
                    epochs_ind.append(np.where(np.array(histogram_obj['stim_name']) == surprise_stim_name[e])[0][0])
                    stim_names.append(surprise_stim_name[e])
                    surprise_to_keep.append(e)
            histogram = histogram[epochs_ind, :, :]
            pac_stim_freq = pac_stim_freq[epochs_ind, :]
            surprise = surprise[surprise_to_keep]
            uncertainty = uncertainty[surprise_to_keep]
            predicted_density = predicted_density[surprise_to_keep]
            correct = correct[surprise_to_keep]


            for chan in range(histogram.shape[1]):
                # save data
                assert histogram.shape[2] == 18
                for bin in range(18):
                    histogram_target[bin] += histogram[:, chan, bin].tolist()
                pac_stim_freq_target += pac_stim_freq[:, chan].tolist()
                surprise_feature += surprise.tolist()
                uncertainty_feature += uncertainty.tolist()
                predicted_density_feature += predicted_density.tolist()
                correct_list += correct.tolist()
                stim_list += [stim for stim in stim_names]
                rep += [float(stim.split('_')[1][3]) for stim in stim_names]
                note_rate += [stim.split('_')[2][:5] for stim in stim_names]
                onset += [float(stim.split('_')[4][5]) for stim in stim_names]
                subject_id += [subject_name for _ in range(histogram.shape[0])]
                electrode_id += [histogram_obj['ch_names'][chan_ind[chan]] for _ in range(histogram.shape[0])]
    # save_data
    df = {'subject': subject_id, 'electrode': electrode_id, 'stim_name': stim_list, 'surprise': surprise_feature, 'uncertainty': uncertainty_feature, 'predicted_density': predicted_density_feature, 'rep': rep, 'note_rate': note_rate, 'onset': onset, 'correct': correct_list, 'pac_stim_freq': pac_stim_freq_target}
    for bin in range(18):
        df[f'histogram_{bin}'] = histogram_target[bin]
    df = pd.DataFrame(df)

    # Create a dictionary that maps each stimulus to the 'correct' value from rep 2
    df['stim_name'] = df['stim_name'].str[14:26]
    correct2_map = df[df['rep'] == 2].set_index(['subject', 'stim_name'])['correct'].to_dict()
    # Add 'correct2' column by mapping based on the stimulus and using the correct2_map for rep 1 rows
    df['correct2'] = df.apply(lambda row: correct2_map.get((row['subject'], row['stim_name'])) if row['rep'] == 1 else row['correct'], axis=1)

    os.makedirs(os.path.join(save_path, 'histogram'), exist_ok=True)
    df.to_csv(os.path.join(save_path, "histogram", "histogram.csv"), index=False)

    return



def pac_and_surprise(data_path):
    encoding_data = pd.read_csv(data_path)
    # encoding_data = encoding_data.dropna(subset=['correct', 'correct2'])
    print(encoding_data)
    print(encoding_data.columns)
    print(len(encoding_data))

    print(encoding_data['rep'])
    encoding_data = encoding_data[encoding_data['rep']==1]
    # encoding_data = encoding_data[encoding_data['correct2']==1]
    # encoding_data = encoding_data[encoding_data['pac_stim_freq']>0]
    print(encoding_data)

    print(np.mean(encoding_data['pac_stim_freq'].to_list()))
    print(encoding_data.groupby('subject')['pac_stim_freq'].mean())
    print(encoding_data.groupby('correct2')['pac_stim_freq'].mean())

    num_quantiles = 6
    # num_quantiles = 8
    # encoding_data['surprise_bin'] = pd.cut(encoding_data['positive_surprise'], bins=num_quantiles, labels=False)
    encoding_data['surprise_bin'], quantile_values = pd.qcut(encoding_data['positive_surprise'], q=num_quantiles, labels=False, retbins=True)
    print(quantile_values)
    print(encoding_data['positive_surprise'].min())
    print(encoding_data['positive_surprise'].max())

    bin_means = encoding_data.groupby('surprise_bin')['pac_stim_freq'].mean()

    plt.figure(figsize=(8, 6))
    plt.plot(bin_means.index, bin_means.values, marker='o')
    plt.xlabel('Surprise quantile')
    plt.ylabel('Mean PAC at stim freq')

    bin_correct2 = encoding_data.groupby('surprise_bin')['correct2'].mean()
    plt.figure(figsize=(8, 6))
    plt.plot(bin_correct2.index, bin_correct2.values, marker='o')
    plt.xlabel('Surprise quantile')
    plt.ylabel('Mean correct 2')

    # plot correct2 with rate
    fig, axes = plt.subplots(2, 3)
    axes = axes.flat
    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']
    for r, rate in enumerate(rates):
        rate_data = encoding_data[encoding_data['stim_name'].str.contains(rate)]
        rate_data['surprise_bin'] = pd.qcut(rate_data['positive_surprise'], q=num_quantiles, labels=False)
        rate_bin_means = rate_data.groupby('surprise_bin')['correct2'].mean()
        axes[r].plot(rate_bin_means.index, rate_bin_means.values, marker='o')
        axes[r].set_title(rate)
        axes[r].set_xlabel('Surprise quantile')
        axes[r].set_ylabel('Mean correct 2')

    fig, axes = plt.subplots(2, 3)
    axes = axes.flat
    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']
    for r, rate in enumerate(rates):
        rate_data = encoding_data[encoding_data['stim_name'].str.contains(rate)]
        rate_data['surprise_bin'] = pd.qcut(rate_data['positive_surprise'], q=num_quantiles, labels=False)
        rate_bin_means = rate_data.groupby('surprise_bin')['pac_stim_freq'].mean()
        axes[r].plot(rate_bin_means.index, rate_bin_means.values, marker='o')
        axes[r].set_title(rate)
        axes[r].set_xlabel('Surprise quantile')
        axes[r].set_ylabel('Mean PAC at stim freq')

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=encoding_data,
        x='surprise_bin',
        y='pac_stim_freq',
        # inner=None,  # Do not show internal boxplot
        # color='lightgray',
        alpha=0.7
    )
    # sns.swarmplot(
    #     data=encoding_data,
    #     x='surprise_bin',
    #     y='pac_stim_freq',
    #     color='black',
    #     size=4,  # Marker size
    #     alpha=0.8
    # )

    # plt.show()
    plt.figure(figsize=(8, 6))
    bin_means.plot(kind='bar', figsize=(8, 6), color='skyblue', edgecolor='black')
    plt.xlabel('Surprise quantile')
    plt.ylabel('Mean PAC')
    plt.title('Relationship Between Surprise and PAC')
    plt.xticks(rotation=0)
    # plt.show()

    # Scatter plot with quadratic regression
    sns.lmplot(
        data=encoding_data,
        x='positive_surprise',
        y='pac_stim_freq',
        height=6,
        aspect=1.2,
        scatter_kws={'alpha': 0.7},
        line_kws={'color': 'green'},
        order=2,  # Quadratic fit
    )
    plt.title('Scatter Plot of Surprise vs PAC with Quadratic Fit')
    plt.grid(True)


    plt.figure(figsize=(8, 6))
    bin_means = encoding_data.groupby('surprise_bin')['theta'].mean()
    plt.plot(bin_means.index, bin_means.values, marker='o', label='theta')
    plt.xlabel('Surprise quantile')
    plt.ylabel('Mean')
    bin_means = encoding_data.groupby('surprise_bin')['high_gamma'].mean()
    plt.plot(bin_means.index, bin_means.values, marker='o', label='high_gamma')
    bin_means = encoding_data.groupby('surprise_bin')['power_stim_freq'].mean()
    plt.plot(bin_means.index, bin_means.values, marker='o', label='power_stim_freq')
    bin_means = encoding_data.groupby('surprise_bin')['pac'].mean()
    plt.plot(bin_means.index, bin_means.values, marker='o', label='pac')
    bin_means = encoding_data.groupby('surprise_bin')['pac_stim_freq'].mean()
    plt.plot(bin_means.index, bin_means.values, marker='o', label='pac_stim_freq')
    plt.legend()

    # plt.show()

    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_quantiles))  # Fixed colors for 6 quantiles
    # pac with surprise in time windows
    print(quantile_values)
    quantile_values[0] = 0  # make sure that extreme values are in a bin
    quantile_values[-1] = 100
    print(quantile_values)
    shift = np.linspace(-num_quantiles/2, num_quantiles/2, num_quantiles)
    for window in range(8):
        encoding_data[f'surprise_bin_{window}'], bins = pd.qcut(encoding_data[f'surprise_{window}_{window+1}s'], q=num_quantiles, labels=False, retbins=True)
        # print(bins)
        # print(encoding_data[['surprise_bin', f'surprise_bin_{window}']])
        print(encoding_data)
        quantile_data = encoding_data.groupby([f'surprise_bin_{window}']).mean()
        # print(quantile_data)
        pac = quantile_data[f'pac_stim_freq_{window}_{window + 1}s'].tolist()
        print('windows', window)
        print(encoding_data[f'surprise_{window}_{window+1}s'].min())
        print(encoding_data[f'surprise_{window}_{window+1}s'].max())
        for quantile in range(num_quantiles):
            # pac = encoding_data[(encoding_data[f'surprise_{window}_{window+1}s'] > quantile_values[quantile]) & (encoding_data[f'surprise_{window}_{window+1}s'] <= quantile_values[quantile+1])][f'pac_stim_freq_{window}_{window + 1}s'].mean()
            plt.bar(window + shift[quantile] * 0.1,  # Offset bars slightly for clarity
                    pac[quantile],
                    # pac,
                    width=0.1,
                    # label=f'Quantile {quantile + 1}',
                    color=colors[quantile])

    # plt.xlabel('Time window')
    plt.ylabel('PAC at stim freq', fontsize=15)
    plt.xlabel(f'Trial time window (t to t+1 s)', fontsize=15)
    plt.xticks([x for x in range(8)], [x for x in range(8)], fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend([f'Quantile {q+1}' for q in range(num_quantiles)], title='Surprise quantile', fontsize=12)

    # plt.show()

    fig, axes = plt.subplots(2, 3)
    axes = axes.flat
    colors = plt.cm.viridis(np.linspace(0, 1, num_quantiles))  # Fixed colors for 6 quantiles
    # pac with surprise in time windows
    shift = np.linspace(-num_quantiles / 2, num_quantiles / 2, num_quantiles)
    for r, rate in enumerate(rates):
        rate_data = encoding_data[encoding_data['stim_name'].str.contains(rate)]
        for window in range(8):
            rate_data[f'surprise_bin_{window}'], bins = pd.qcut(rate_data[f'surprise_{window}_{window + 1}s'],
                                                                    q=num_quantiles, labels=False, retbins=True)
            # print(bins)
            # print(encoding_data[['surprise_bin', f'surprise_bin_{window}']])
            print(encoding_data)
            quantile_data = rate_data.groupby([f'surprise_bin_{window}']).mean()
            # print(quantile_data)
            pac = quantile_data[f'pac_stim_freq_{window}_{window + 1}s'].tolist()
            print('windows', window)
            print(encoding_data[f'surprise_{window}_{window + 1}s'].min())
            print(encoding_data[f'surprise_{window}_{window + 1}s'].max())
            for quantile in range(num_quantiles):
                # pac = encoding_data[(encoding_data[f'surprise_{window}_{window+1}s'] > quantile_values[quantile]) & (encoding_data[f'surprise_{window}_{window+1}s'] <= quantile_values[quantile+1])][f'pac_stim_freq_{window}_{window + 1}s'].mean()
                axes[r].bar(window + shift[quantile] * 0.1,  # Offset bars slightly for clarity
                        pac[quantile],
                        # pac,
                        width=0.1,
                        label=f'Quantile {quantile + 1}',
                        color=colors[quantile])

        # plt.xlabel('Time window')
        axes[r].set_ylabel('PAC at stim freq')
        axes[r].set_xlabel(f'From x to x+1 seconds')
        axes[r].set_xticks([x for x in range(8)], [x for x in range(8)])
        # plt.legend(title='Surprise Quantile')

    plt.show()


def phase_coherence_and_surprise(data_path):
    encoding_data = pd.read_csv(data_path)
    # encoding_data = encoding_data.dropna(subset=['correct', 'correct2'])
    encoding_data = encoding_data.dropna(subset=['phase_coherence_stim_freq'])
    print(encoding_data)
    print(encoding_data.columns)
    print(len(encoding_data))

    print(encoding_data['rep'])
    encoding_data = encoding_data[encoding_data['rep']==1]
    # encoding_data = encoding_data[encoding_data['correct2']==1]
    # encoding_data = encoding_data[encoding_data['pac_stim_freq']>0]
    print(encoding_data)

    print(np.mean(encoding_data['phase_coherence_stim_freq'].to_list()))
    print(encoding_data.groupby('subject')['phase_coherence_stim_freq'].mean())
    print(encoding_data.groupby('correct2')['phase_coherence_stim_freq'].mean())

    num_quantiles = 6
    # num_quantiles = 8
    # encoding_data['surprise_bin'] = pd.cut(encoding_data['positive_surprise'], bins=num_quantiles, labels=False)
    encoding_data['surprise_bin'], quantile_values = pd.qcut(encoding_data['positive_surprise'], q=num_quantiles, labels=False, retbins=True)
    print(quantile_values)
    print(encoding_data['positive_surprise'].min())
    print(encoding_data['positive_surprise'].max())

    bin_means = encoding_data.groupby('surprise_bin')['phase_coherence_stim_freq'].mean()

    plt.figure(figsize=(8, 6))
    plt.plot(bin_means.index, bin_means.values, marker='o')
    plt.xlabel('Surprise quantile')
    plt.ylabel('Mean phase coherence at stim freq')

    bin_correct2 = encoding_data.groupby('surprise_bin')['correct2'].mean()
    plt.figure(figsize=(8, 6))
    plt.plot(bin_correct2.index, bin_correct2.values, marker='o')
    plt.xlabel('Surprise quantile')
    plt.ylabel('Mean correct 2')

    # plot correct2 with rate
    fig, axes = plt.subplots(2, 3)
    axes = axes.flat
    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']
    for r, rate in enumerate(rates):
        rate_data = encoding_data[encoding_data['stim_name'].str.contains(rate)]
        rate_data['surprise_bin'] = pd.qcut(rate_data['positive_surprise'], q=num_quantiles, labels=False)
        rate_bin_means = rate_data.groupby('surprise_bin')['correct2'].mean()
        axes[r].plot(rate_bin_means.index, rate_bin_means.values, marker='o')
        axes[r].set_title(rate)
        axes[r].set_xlabel('Surprise quantile')
        axes[r].set_ylabel('Mean correct 2')

    fig, axes = plt.subplots(2, 3)
    axes = axes.flat
    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']
    for r, rate in enumerate(rates):
        rate_data = encoding_data[encoding_data['stim_name'].str.contains(rate)]
        rate_data['surprise_bin'] = pd.qcut(rate_data['positive_surprise'], q=num_quantiles, labels=False)
        rate_bin_means = rate_data.groupby('surprise_bin')['phase_coherence_stim_freq'].mean()
        axes[r].plot(rate_bin_means.index, rate_bin_means.values, marker='o')
        axes[r].set_title(rate)
        axes[r].set_xlabel('Surprise quantile')
        axes[r].set_ylabel('Mean phase coherence at stim freq')

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=encoding_data,
        x='surprise_bin',
        y='phase_coherence_stim_freq',
        # inner=None,  # Do not show internal boxplot
        # color='lightgray',
        alpha=0.7
    )
    # sns.swarmplot(
    #     data=encoding_data,
    #     x='surprise_bin',
    #     y='pac_stim_freq',
    #     color='black',
    #     size=4,  # Marker size
    #     alpha=0.8
    # )

    # plt.show()
    plt.figure(figsize=(8, 6))
    bin_means.plot(kind='bar', figsize=(8, 6), color='skyblue', edgecolor='black')
    plt.xlabel('Surprise quantile')
    plt.ylabel('Mean phase coherence')
    plt.title('Relationship Between Surprise and phase coherence')
    plt.xticks(rotation=0)
    # plt.show()

    # Scatter plot with quadratic regression
    sns.lmplot(
        data=encoding_data,
        x='positive_surprise',
        y='phase_coherence_stim_freq',
        height=6,
        aspect=1.2,
        scatter_kws={'alpha': 0.7},
        line_kws={'color': 'green'},
        order=2,  # Quadratic fit
    )
    plt.title('Scatter Plot of Surprise vs phase coherence with Quadratic Fit')
    plt.grid(True)


    plt.figure(figsize=(8, 6))
    bin_means = encoding_data.groupby('surprise_bin')['theta'].mean()
    plt.plot(bin_means.index, bin_means.values, marker='o', label='theta')
    plt.xlabel('Surprise quantile')
    plt.ylabel('Mean')
    bin_means = encoding_data.groupby('surprise_bin')['high_gamma'].mean()
    plt.plot(bin_means.index, bin_means.values, marker='o', label='high_gamma')
    bin_means = encoding_data.groupby('surprise_bin')['power_stim_freq'].mean()
    plt.plot(bin_means.index, bin_means.values, marker='o', label='power_stim_freq')
    bin_means = encoding_data.groupby('surprise_bin')['pac'].mean()
    plt.plot(bin_means.index, bin_means.values, marker='o', label='pac')
    bin_means = encoding_data.groupby('surprise_bin')['pac_stim_freq'].mean()
    plt.plot(bin_means.index, bin_means.values, marker='o', label='pac_stim_freq')
    bin_means = encoding_data.groupby('surprise_bin')['phase_coherence_stim_freq'].mean()
    plt.plot(bin_means.index, bin_means.values, marker='o', label='phase_coherence_stim_freq')
    plt.legend()

    # plt.show()

    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_quantiles))  # Fixed colors for 6 quantiles
    # pac with surprise in time windows
    print(quantile_values)
    quantile_values[0] = 0  # make sure that extreme values are in a bin
    quantile_values[-1] = 100
    print(quantile_values)
    shift = np.linspace(-num_quantiles/2, num_quantiles/2, num_quantiles)
    for window in range(8):
        encoding_data[f'surprise_bin_{window}'], bins = pd.qcut(encoding_data[f'surprise_{window}_{window+1}s'], q=num_quantiles, labels=False, retbins=True)
        # print(bins)
        # print(encoding_data[['surprise_bin', f'surprise_bin_{window}']])
        print(encoding_data)
        quantile_data = encoding_data.groupby([f'surprise_bin_{window}']).mean()
        # print(quantile_data)
        pac = quantile_data[f'phase_coherence_stim_freq_{window}_{window + 1}s'].tolist()
        print('windows', window)
        print(encoding_data[f'surprise_{window}_{window+1}s'].min())
        print(encoding_data[f'surprise_{window}_{window+1}s'].max())
        for quantile in range(num_quantiles):
            # pac = encoding_data[(encoding_data[f'surprise_{window}_{window+1}s'] > quantile_values[quantile]) & (encoding_data[f'surprise_{window}_{window+1}s'] <= quantile_values[quantile+1])][f'pac_stim_freq_{window}_{window + 1}s'].mean()
            plt.bar(window + shift[quantile] * 0.1,  # Offset bars slightly for clarity
                    pac[quantile],
                    # pac,
                    width=0.1,
                    # label=f'Quantile {quantile + 1}',
                    color=colors[quantile])

    # plt.xlabel('Time window')
    plt.ylabel('Phase coherence at stim freq', fontsize=15)
    plt.xlabel(f'Trial time window (t to t+1 s)', fontsize=15)
    plt.xticks([x for x in range(8)], [x for x in range(8)], fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend([f'Quantile {q+1}' for q in range(num_quantiles)], title='Surprise quantile', fontsize=12)

    # plt.show()

    fig, axes = plt.subplots(2, 3)
    axes = axes.flat
    colors = plt.cm.viridis(np.linspace(0, 1, num_quantiles))  # Fixed colors for 6 quantiles
    # pac with surprise in time windows
    shift = np.linspace(-num_quantiles / 2, num_quantiles / 2, num_quantiles)
    for r, rate in enumerate(rates):
        rate_data = encoding_data[encoding_data['stim_name'].str.contains(rate)]
        for window in range(8):
            rate_data[f'surprise_bin_{window}'], bins = pd.qcut(rate_data[f'surprise_{window}_{window + 1}s'],
                                                                    q=num_quantiles, labels=False, retbins=True)
            # print(bins)
            # print(encoding_data[['surprise_bin', f'surprise_bin_{window}']])
            print(encoding_data)
            quantile_data = rate_data.groupby([f'surprise_bin_{window}']).mean()
            # print(quantile_data)
            pac = quantile_data[f'phase_coherence_stim_freq_{window}_{window + 1}s'].tolist()
            print('windows', window)
            print(encoding_data[f'surprise_{window}_{window + 1}s'].min())
            print(encoding_data[f'surprise_{window}_{window + 1}s'].max())
            for quantile in range(num_quantiles):
                # pac = encoding_data[(encoding_data[f'surprise_{window}_{window+1}s'] > quantile_values[quantile]) & (encoding_data[f'surprise_{window}_{window+1}s'] <= quantile_values[quantile+1])][f'pac_stim_freq_{window}_{window + 1}s'].mean()
                axes[r].bar(window + shift[quantile] * 0.1,  # Offset bars slightly for clarity
                        pac[quantile],
                        # pac,
                        width=0.1,
                        label=f'Quantile {quantile + 1}',
                        color=colors[quantile])

        # plt.xlabel('Time window')
        axes[r].set_ylabel('Phase coherence at stim freq')
        axes[r].set_xlabel(f'From x to x+1 seconds')
        axes[r].set_xticks([x for x in range(8)], [x for x in range(8)])
        # plt.legend(title='Surprise Quantile')

    plt.show()


def phase_coherence_all_frequencies(data_path, regions, first_parcel_label):
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*all_freqs.pkl')))

    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']
    phase_coherence_list = [[] for r in range(len(rates))]

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        phase_coherence_obj = np.load(epochs_path, allow_pickle=True)
        phase_coherence = phase_coherence_obj['phase_coherence']
        parcellations = phase_coherence_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        # select electrodes
        chan_ind = []
        kept_parcels = []
        if regions is not None:
            for p, parcel in enumerate(parcellations):
                if any(ch_parcel in regions for ch_parcel in parcel):
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])
        else:  # with regions=None, we keep all channels but the ones in go_next
            for p, parcel in enumerate(parcellations):
                if parcel[0] not in go_next:
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])

        if len(chan_ind) > 0:
            phase_coherence = phase_coherence[:, :, chan_ind]
            print(np.array(phase_coherence_obj['ch_names'])[chan_ind])
            stim_names = phase_coherence_obj['stim_name']

            for r, rate in enumerate(rates):
                rate_epochs = [e for e in range(len(stim_names)) if rate in stim_names[e]]
                print(phase_coherence.shape)
                rate_phase_coherence = phase_coherence[:, rate_epochs, :]
                valid_indices = ~np.isnan(rate_phase_coherence).any(axis=(0, 2))
                rate_phase_coherence = rate_phase_coherence[:, valid_indices, :]
                phase_coherence_list[r].append(np.mean(rate_phase_coherence, axis=(1, 2)))  # average epochs and channels
                # phase_coherence_list[r].append(np.mean(rate_phase_coherence, axis=(1,2)) / np.std(np.mean(rate_phase_coherence, axis=2), axis=1))  # average epochs and channels
    phase_coherence_list = [np.array(phase_coherence_list[r]) for r in range(len(phase_coherence_list))]

    freqs = np.array(phase_coherence_obj['low_freq'])
    print(freqs)

    # phase coherence by rate on one plot
    color_list = plt.get_cmap('Set2')
    fig, axs = plt.subplots(1, 2, sharey=True)
    for r, rate in enumerate(rates):
        rate_freq = float(rate[:3])
        phase_coherence_rate = phase_coherence_list[r]
        tvalues = np.mean(phase_coherence_rate, axis=0) / (np.std(phase_coherence_rate, axis=0) / np.sqrt(phase_coherence_rate.shape[0]))
        mean = np.mean(phase_coherence_rate, axis=0)
        sem = np.std(phase_coherence_rate, axis=0) / np.sqrt(phase_coherence_rate.shape[0])
        freq_idx = np.argmin(np.abs(freqs - rate_freq))
        # vmax = 5
        axs[0].plot(mean, freqs, label=rate, color=color_list.colors[r], lw=3)
        # axs[0].fill_betweenx(freqs, mean - sem, mean + sem, alpha=0.5, lw=0, color=color_list.colors[r])
        # axs[0].set_xscale('log')
        axs[0].axvline(0, ls='--', c='k')
        axs[0].hlines(freqs[freq_idx], lw=3, color=color_list.colors[r], ls='--', xmin=0, xmax=mean[freq_idx])
        axs[0].scatter(mean[freq_idx], freqs[freq_idx], color=color_list.colors[r], s=100, zorder=3)
        axs[0].set_xlabel('Phase coherence')
        axs[0].set_ylabel('Frequency (Hz)')
        axs[0].set_ylim(freqs[0], freqs[-1])
        axs[0].set_yscale('log')
        # axs[0, r].set_title(rate)
        axs[0].legend()
        axs[1].plot(tvalues, freqs, label=rate, color=color_list.colors[r], lw=3)
        axs[1].axvline(0, ls='--', c='k')
        axs[1].hlines(freqs[freq_idx], lw=3, color=color_list.colors[r], ls='--', xmin=0, xmax=tvalues[freq_idx])
        axs[1].scatter(tvalues[freq_idx], freqs[freq_idx], color=color_list.colors[r], s=100, zorder=3)
        axs[1].set_xlabel('Phase coherence (tval)')
        axs[1].set_ylabel('Frequency (Hz)')
        axs[1].set_ylim(freqs[0], freqs[-1])
        axs[1].set_yscale('log')
        axs[1].legend()
    plt.show()


def pac_all_frequencies(data_path, regions, first_parcel_label):
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*all_freqs.pkl')))

    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']
    phase_coherence_list = [[] for r in range(len(rates))]

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        phase_coherence_obj = np.load(epochs_path, allow_pickle=True)
        phase_coherence = phase_coherence_obj['pac']
        parcellations = phase_coherence_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        # select electrodes
        chan_ind = []
        kept_parcels = []
        if regions is not None:
            for p, parcel in enumerate(parcellations):
                if any(ch_parcel in regions for ch_parcel in parcel):
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])
        else:  # with regions=None, we keep all channels but the ones in go_next
            for p, parcel in enumerate(parcellations):
                if parcel[0] not in go_next:
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])

        if len(chan_ind) > 0:
            phase_coherence = phase_coherence[:, :, chan_ind]
            print(np.array(phase_coherence_obj['ch_names'])[chan_ind])
            stim_names = phase_coherence_obj['stim_name']

            for r, rate in enumerate(rates):
                rate_epochs = [e for e in range(len(stim_names)) if rate in stim_names[e]]
                print(phase_coherence.shape)
                rate_phase_coherence = phase_coherence[:, rate_epochs, :]
                valid_indices = ~np.isnan(rate_phase_coherence).any(axis=(0, 2))
                rate_phase_coherence = rate_phase_coherence[:, valid_indices, :]
                phase_coherence_list[r].append(np.mean(rate_phase_coherence, axis=(1, 2)))  # average epochs and channels
                # phase_coherence_list[r].append(np.mean(rate_phase_coherence, axis=(1,2)) / np.std(np.mean(rate_phase_coherence, axis=2), axis=1))  # average epochs and channels
    phase_coherence_list = [np.array(phase_coherence_list[r]) for r in range(len(phase_coherence_list))]

    freqs = np.array(phase_coherence_obj['high_freq'])
    print(freqs)

    # phase coherence by rate on one plot
    color_list = plt.get_cmap('Set2')
    fig, axs = plt.subplots(1, 2, sharey=True)
    for r, rate in enumerate(rates):
        rate_freq = float(rate[:3])
        phase_coherence_rate = phase_coherence_list[r]
        tvalues = np.mean(phase_coherence_rate, axis=0) / (np.std(phase_coherence_rate, axis=0) / np.sqrt(phase_coherence_rate.shape[0]))
        mean = np.mean(phase_coherence_rate, axis=0)
        sem = np.std(phase_coherence_rate, axis=0) / np.sqrt(phase_coherence_rate.shape[0])
        freq_idx = np.argmin(np.abs(freqs - rate_freq))
        # vmax = 5
        axs[0].plot(mean, freqs, label=rate, color=color_list.colors[r], lw=3)
        # axs[0].fill_betweenx(freqs, mean - sem, mean + sem, alpha=0.5, lw=0, color=color_list.colors[r])
        # axs[0].set_xscale('log')
        axs[0].axvline(0, ls='--', c='k')
        axs[0].set_xlabel('PAC')
        axs[0].set_ylabel('Frequency (Hz)')
        axs[0].set_ylim(freqs[0], freqs[-1])
        axs[0].set_yscale('log')
        # axs[0, r].set_title(rate)
        axs[0].legend()
        axs[1].plot(tvalues, freqs, label=rate, color=color_list.colors[r], lw=3)
        axs[1].axvline(0, ls='--', c='k')
        axs[1].set_xlabel('PAC (tval)')
        axs[1].set_ylabel('Frequency (Hz)')
        axs[1].set_ylim(freqs[0], freqs[-1])
        axs[1].set_yscale('log')
        axs[1].legend()

    # plot grand average
    phase_coherence_rate = np.mean([phase_coherence_list[r] for r in range(len(rates))], axis=0)
    tvalues = np.mean(phase_coherence_rate, axis=0) / (
                np.std(phase_coherence_rate, axis=0) / np.sqrt(phase_coherence_rate.shape[0]))
    mean = np.mean(phase_coherence_rate, axis=0)
    sem = np.std(phase_coherence_rate, axis=0) / np.sqrt(phase_coherence_rate.shape[0])
    freq_idx = np.argmin(np.abs(freqs - rate_freq))
    # vmax = 5
    axs[0].plot(mean, freqs, label='average', color='gray', lw=3)
    # axs[0].fill_betweenx(freqs, mean - sem, mean + sem, alpha=0.5, lw=0, color=color_list.colors[r])
    # axs[0].set_xscale('log')
    axs[0].axvline(0, ls='--', c='k')
    axs[0].set_xlabel('PAC')
    axs[0].set_ylabel('Frequency (Hz)')
    axs[0].set_ylim(freqs[0], freqs[-1])
    axs[0].set_yscale('log')
    # axs[0, r].set_title(rate)
    axs[0].legend()
    axs[1].plot(tvalues, freqs, label='average', color='gray', lw=3)
    axs[1].axvline(0, ls='--', c='k')
    axs[1].set_xlabel('PAC (tval)')
    axs[1].set_ylabel('Frequency (Hz)')
    axs[1].set_ylim(freqs[0], freqs[-1])
    axs[1].set_yscale('log')
    axs[1].legend()

    plt.show()


def plot_all_frequencies_individual_channels(data_path, regions, region_name, first_parcel_label, analysis_name, freqs_name='freqs', bad_epochs_path='/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs/bad_epochs/bad_epochs_hippocampus.npy'):
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*all_freqs.pkl')))

    bad_epochs = np.load(bad_epochs_path, allow_pickle=True).item()

    subject_names = []
    phase_coherence_list = []
    subject_id = []  # used for color

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)
        subject_names.append(subject_name)
        subject_bad_epochs = bad_epochs[subject_name]
        phase_coherence_obj = np.load(epochs_path, allow_pickle=True)
        phase_coherence = phase_coherence_obj[analysis_name]
        parcellations = phase_coherence_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        # select electrodes
        chan_ind = []
        kept_parcels = []
        if regions is not None:
            for p, parcel in enumerate(parcellations):
                if any(ch_parcel in regions for ch_parcel in parcel):
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])
        else:  # with regions=None, we keep all channels but the ones in go_next
            for p, parcel in enumerate(parcellations):
                if parcel[0] not in go_next:
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])

        if len(chan_ind) > 0:
            phase_coherence = phase_coherence[:, :, chan_ind]
            print(np.array(phase_coherence_obj['ch_names'])[chan_ind])
            stim_names = phase_coherence_obj['stim_name']
            good_epochs_mask = np.ones(phase_coherence.shape[1], dtype=bool)
            good_epochs_mask[subject_bad_epochs] = False
            phase_coherence = phase_coherence[:, good_epochs_mask, :]
            valid_indices = ~np.isnan(phase_coherence).any(axis=(0, 2))
            phase_coherence = phase_coherence[:, valid_indices, :]
            phase_coherence = np.mean(phase_coherence, axis=1)  # average epochs
            # phase_coherence = np.mean(phase_coherence, axis=1) / (np.std(phase_coherence, axis=1, ddof=1) / np.sqrt(phase_coherence.shape[1]))  # t-values
            for c in range(phase_coherence.shape[-1]):
                phase_coherence_list.append(phase_coherence[:, c])  # add channels one by one
                subject_id.append(i)
    phase_coherence_list = np.array(phase_coherence_list)
    print(phase_coherence_list.shape)

    freqs = np.array(phase_coherence_obj[freqs_name])
    print(freqs)

    subject_colors = plt.get_cmap('Set1').colors
    colors = [subject_colors[s] for s in subject_id]
    print(colors)
    print(len(colors))

    # phase coherence by channel on one plot
    fig, axs = plt.subplots(1, 1)
    for c in range(phase_coherence_list.shape[0]):
        axs.plot(phase_coherence_list[c, :], freqs, color=colors[c], lw=1)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=7) for name, color in zip(subject_names, subject_colors)]
    plt.legend(handles=legend_elements, loc="best", title="Subjects")
    axs.axvline(0, ls='--', c='k')
    axs.set_xlabel(analysis_name)
    axs.set_ylabel('Frequency (Hz)')
    axs.set_ylim(freqs[0], freqs[-1])
    axs.set_yscale('log')

    # plt.show()
    fig.savefig(os.path.join(data_path, f'{region_name}_{analysis_name}_all_subjects.svg'))
    plt.close()

    # individual subjects
    fig, axs = plt.subplots(1, 1)
    for i, _ in tqdm(enumerate(subject_list)):
        phase_coherence_list_subject = phase_coherence_list[np.array(subject_id) == i, :]
        axs.plot(np.mean(phase_coherence_list_subject, axis=0), freqs, color=subject_colors[i], lw=3, label=subject_names[i])
    axs.axvline(0, ls='--', c='k')
    axs.set_xlabel(analysis_name)
    axs.set_ylabel('Frequency (Hz)')
    axs.set_ylim(freqs[0], freqs[-1])
    axs.set_yscale('log')
    axs.legend()
    fig.savefig(os.path.join(data_path, f'{region_name}_{analysis_name}_all_subjects_average_channels.svg'))
    plt.close()

    # individual subjects
    for i, _ in tqdm(enumerate(subject_list)):
        fig, axs = plt.subplots(1, 1)
        phase_coherence_list_subject = phase_coherence_list[np.array(subject_id) == i, :]
        # for c in range(phase_coherence_list_subject.shape[0]):
        axs.plot(phase_coherence_list_subject.T, freqs, lw=2)
        axs.plot(np.mean(phase_coherence_list_subject, axis=0), freqs, color='k', lw=4)
        axs.axvline(0, ls='--', c='k')
        axs.set_xlabel(analysis_name)
        axs.set_ylabel('Frequency (Hz)')
        axs.set_ylim(freqs[0], freqs[-1])
        axs.set_yscale('log')
        axs.set_title(subject_names[i])
        # plt.show()
        fig.savefig(os.path.join(data_path, f'{region_name}_{analysis_name}_{subject_names[i]}.svg'))
        plt.close()


def plot_all_frequencies_individual_channels_by_rate(data_path, regions, region_name, first_parcel_label, analysis_name, freqs_name='freqs', bad_epochs_path='/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs/bad_epochs/bad_epochs_hippocampus.npy'):
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*all_freqs.pkl')))

    bad_epochs = np.load(bad_epochs_path, allow_pickle=True).item()

    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']
    subject_names = []
    phase_coherence_list = [[] for r in range(len(rates))]
    subject_id = []  # used for color

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)
        subject_names.append(subject_name)
        subject_bad_epochs = bad_epochs[subject_name]
        phase_coherence_obj = np.load(epochs_path, allow_pickle=True)
        phase_coherence = phase_coherence_obj[analysis_name]
        parcellations = phase_coherence_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        # select electrodes
        chan_ind = []
        kept_parcels = []
        if regions is not None:
            for p, parcel in enumerate(parcellations):
                if any(ch_parcel in regions for ch_parcel in parcel):
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])
        else:  # with regions=None, we keep all channels but the ones in go_next
            for p, parcel in enumerate(parcellations):
                if parcel[0] not in go_next:
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])

        if len(chan_ind) > 0:
            phase_coherence = phase_coherence[:, :, chan_ind]
            print(np.array(phase_coherence_obj['ch_names'])[chan_ind])
            good_epochs_mask = np.ones(phase_coherence.shape[1], dtype=bool)
            good_epochs_mask[subject_bad_epochs] = False
            phase_coherence = phase_coherence[:, good_epochs_mask, :]
            stim_names = np.array(phase_coherence_obj['stim_name'])[good_epochs_mask]

            for r, rate in enumerate(rates):
                rate_epochs = [e for e in range(len(stim_names)) if rate in stim_names[e]]
                rate_phase_coherence = phase_coherence[:, rate_epochs, :]
                valid_indices = ~np.isnan(rate_phase_coherence).any(axis=(0, 2))
                rate_phase_coherence = rate_phase_coherence[:, valid_indices, :]
                rate_phase_coherence = np.mean(rate_phase_coherence, axis=1)  # average epochs
                # rate_phase_coherence = np.mean(rate_phase_coherence, axis=1) / (np.std(rate_phase_coherence, axis=1, ddof=1) / np.sqrt(rate_phase_coherence.shape[1]))  # t-values
                for c in range(phase_coherence.shape[-1]):
                    phase_coherence_list[r].append(rate_phase_coherence[:, c])  # add channels one by one
                    if r == 0:
                        subject_id.append(i)
    phase_coherence_list = [np.array(phase_coherence_list[r]) for r in range(len(phase_coherence_list))]

    freqs = np.array(phase_coherence_obj[freqs_name])
    print(freqs)

    subject_colors = plt.get_cmap('Set1').colors
    colors = [subject_colors[s] for s in subject_id]
    print(colors)
    print(len(colors))

    # phase coherence by channel on one plot
    fig, axs = plt.subplots(1, len(rates), sharex=True, sharey=True)
    for r, rate in enumerate(rates):
        rate_freq = float(rate[:3])
        for c in range(phase_coherence_list[r].shape[0]):
            axs[r].plot(phase_coherence_list[r][c, :], freqs, color=colors[c], lw=1)
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=7) for name, color in zip(subject_names, subject_colors)]
        plt.legend(handles=legend_elements, loc="best", title="Subjects")
        axs[r].axvline(0, ls='--', c='k')
        axs[r].axhline(float(rate[:3]), c='blue', lw=0.5)
        axs[r].set_xlabel(analysis_name)
        axs[r].set_ylim(freqs[0], freqs[-1])
        axs[r].set_yscale('log')
        axs[r].set_title(rate)
    axs[0].set_ylabel('Frequency (Hz)')

    # plt.show()
    fig.savefig(os.path.join(data_path, f'{region_name}_by_rate_{analysis_name}_all_subjects.svg'))
    plt.close()

    # individual subjects
    for i, _ in tqdm(enumerate(subject_list)):
        fig, axs = plt.subplots(1, len(rates), sharex=True, sharey=True)
        for r, rate in enumerate(rates):
            phase_coherence_list_subject = phase_coherence_list[r][np.array(subject_id) == i, :]
            axs[r].plot(phase_coherence_list_subject.T, freqs, lw=2)
            axs[r].plot(np.mean(phase_coherence_list_subject, axis=0), freqs, color='k', lw=4)
            axs[r].axvline(0, ls='--', c='k')
            axs[r].axhline(float(rate[:3]), c='blue', lw=0.5)
            axs[r].set_xlabel(analysis_name)
            axs[r].set_ylim(freqs[0], freqs[-1])
            axs[r].set_yscale('log')
            axs[r].set_title(rate)
        axs[0].set_ylabel('Frequency (Hz)')
        plt.suptitle(subject_names[i])
        # plt.show()
        fig.savefig(os.path.join(data_path, f'{region_name}_by_rate_{analysis_name}_{subject_names[i]}.svg'))
        plt.close()


def study_epoch_consistency_one_subject(data_path, regions, first_parcel_label, analysis_name, freqs_name='freqs'):
    os.makedirs(os.path.join(data_path, 'bad_epochs'), exist_ok=True)

    path = Path(data_path)
    subject_list = sorted(list(path.glob('*all_freqs.pkl')))

    bad_epochs_dict = {}

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        phase_coherence_list = []
        phase_coherence_list_tval = []

        phase_coherence_obj = np.load(epochs_path, allow_pickle=True)
        phase_coherence = phase_coherence_obj[analysis_name]
        parcellations = phase_coherence_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        # select electrodes
        chan_ind = []
        kept_parcels = []
        if regions is not None:
            for p, parcel in enumerate(parcellations):
                if any(ch_parcel in regions for ch_parcel in parcel):
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])
        else:  # with regions=None, we keep all channels but the ones in go_next
            for p, parcel in enumerate(parcellations):
                if parcel[0] not in go_next:
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])

        phase_coherence = phase_coherence[:, :, chan_ind]
        print(np.array(phase_coherence_obj['ch_names'])[chan_ind])
        valid_indices = ~np.isnan(phase_coherence).any(axis=(0, 2))
        phase_coherence = phase_coherence[:, valid_indices, :]
        phase_coherence_mean = np.mean(phase_coherence, axis=1)  # average epochs
        phase_coherence_tval = np.mean(phase_coherence, axis=1) / (np.std(phase_coherence, axis=1, ddof=1) / np.sqrt(phase_coherence.shape[1]))  # t-values
        for c in range(phase_coherence_mean.shape[-1]):
            phase_coherence_list.append(phase_coherence_mean[:, c])  # add channels one by one
            phase_coherence_list_tval.append(phase_coherence_tval[:, c])
        phase_coherence_list = np.array(phase_coherence_list)
        print(phase_coherence_list.shape)
        phase_coherence_list_tval = np.array(phase_coherence_list_tval)
        print(phase_coherence_list_tval.shape)

        freqs = np.array(phase_coherence_obj[freqs_name])

        fig, axs = plt.subplots(1, 2)
        axs[0].plot(phase_coherence_list.T, freqs, lw=3)
        axs[0].plot(np.mean(phase_coherence_list, axis=0), freqs, color='k', lw=4)
        axs[1].plot(phase_coherence_list_tval.T, freqs, lw=3)
        axs[1].plot(np.mean(phase_coherence_list_tval, axis=0), freqs, color='k', lw=4)
        for ax in axs:
            ax.axvline(0, ls='--', c='k')
            ax.set_xlabel(analysis_name)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_ylim(freqs[0], freqs[-1])
            ax.set_yscale('log')
        axs[0].set_title('average')
        axs[1].set_title('tval across epochs')
        fig.suptitle(subject_name)

        # plt.show()

        # individual epochs, one channel
        fig, axs = plt.subplots(2, phase_coherence.shape[2] + 1, sharey='row')
        for c in tqdm(range(phase_coherence.shape[2])):
            # fig, axs = plt.subplots(1, 2)
            channel_phase_coherence = phase_coherence[:, :, c]
            axs[0, c].plot(channel_phase_coherence, freqs, lw=2)
            # axs[0, c].plot(np.mean(channel_phase_coherence, axis=1), freqs, color='k', lw=4)
            axs[0, c].axvline(0, ls='--', c='k')
            axs[0, c].set_xlabel(analysis_name)
            axs[0, c].set_ylim(freqs[0], freqs[-1])
            axs[0, c].set_yscale('log')
            axs[0, c].set_title(f'channel {c}')

            # histogram of max zscore
            max_power = np.max(channel_phase_coherence, axis=0)
            std = np.std(max_power)
            print(f"{max_power.shape[0]} epochs, {np.sum(max_power > 3*std)} removed")
            axs[1, c].hist(max_power, bins=100)
            axs[1, c].axvline(x=3*std, color='r', lw=0.5)
        axs[0, 0].set_ylabel('Frequency (Hz)')
        axs[1, 0].set_xlabel('Max power across frequencies')
        axs[1, 0].set_ylabel('Number of epochs')
        fig.suptitle(subject_name)

        # add the average of channels
        channel_phase_coherence = np.mean(phase_coherence, axis=2)
        axs[0, -1].plot(channel_phase_coherence, freqs, lw=2)
        axs[0, -1].axvline(0, ls='--', c='k')
        axs[0, -1].set_xlabel(analysis_name)
        axs[0, -1].set_ylim(freqs[0], freqs[-1])
        axs[0, -1].set_yscale('log')
        axs[0, -1].set_title(f'Average of channels')
        # histogram of max zscore
        max_power = np.max(channel_phase_coherence, axis=0)
        std = np.std(max_power)
        print(f"{max_power.shape[0]} epochs, {np.sum(max_power > 3 * std)} removed")
        axs[1, -1].hist(max_power, bins=100)
        axs[1, -1].axvline(x=3 * std, color='r', lw=0.5)

        # plt.show()
        fig.savefig(os.path.join(data_path, 'bad_epochs', f'epoch_analysis_{subject_name}.svg'))
        plt.close()

        bad_epochs_dict[subject_name] = np.argwhere(max_power > 3 * std)[:, 0]
    print(bad_epochs_dict)
    np.save(os.path.join(data_path, 'bad_epochs', 'bad_epochs_hippocampus'), bad_epochs_dict)


def pac_all_frequencies_contrast(data_path, behavior_path, regions, first_parcel_label=True):
    """"""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_pac_all_freqs.pkl')))
    subject_names = []
    print(subject_paths)

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']

    rms_contrasts = [[] for c in range(len(conditions))]
    perm_contrasts = [[] for c in range(len(conditions))]
    pvalues = [[] for c in range(len(conditions))]
    subject_id = [[] for c in range(len(conditions))]  # used for color

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['pac']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label=="advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[2]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            observed_diff = np.mean(rms[:, correct, c], axis=1) - np.mean(rms[:, incorrect, c], axis=1)
                            rms_contrasts[cond].append(observed_diff)
                            # perm_contrasts[cond].append(perm_diffs)
                            # pvalues[cond].append(ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False, permutations=1000, random_state=None, alternative='two-sided').pvalue)  # testing of individual electrodes
                            subject_id[cond].append(i)

    # # FDR correction
    # for c in range(len(pvalues)):
    #     _, pval = mne.stats.fdr_correction(pvalues[c], alpha=0.05, method='indep')
    #     pvalues[c] = pval

    freqs = np.array(rms_obj['high_freq'])

    fig, axs = plt.subplots(1, len(conditions), sharey=True, sharex=True)
    subject_colors = plt.get_cmap('Set3').colors
    # Plot each condition's data in a separate subplot
    # for i, (condition, rms, perms, pval, s_id) in enumerate(zip(conditions, rms_contrasts, perm_contrasts, pvalues, subject_id)):
    # for i, (condition, rms, perms, s_id) in enumerate(zip(conditions, rms_contrasts, perm_contrasts, subject_id)):
    for i, (condition, rms, s_id) in enumerate(zip(conditions, rms_contrasts, subject_id)):
        # population_p = ttest_1samp(rms, 0).pvalue
        t_obs = np.mean(rms, axis=0) / (np.std(rms, axis=0, ddof=1) / np.sqrt(len(rms)))
        # t_obs = np.mean(rms, axis=0)
        # t_perms = np.mean(perms, axis=0) / (np.std(perms, axis=0, ddof=1) / np.sqrt(np.array(perms).shape[0]))
        # population_p = np.sum(np.abs(t_perms) >= np.abs(t_obs)) / np.array(perms).shape[1]
        # population_p = np.sum(np.mean(perms, axis=0) >= np.mean(rms)) / np.array(perms).shape[1]
        # print(population_p)

        print(t_obs.shape)
        print(freqs.shape)
        axs[i].plot(t_obs, freqs, label='average', color='gray', lw=3)
        # axs[0].fill_betweenx(freqs, mean - sem, mean + sem, alpha=0.5, lw=0, color=color_list.colors[r])
        axs[i].axvline(0, ls='--', c='k')
        axs[i].set_xlabel('PAC contrast')
        axs[i].set_ylabel('Frequency (Hz)')
        axs[i].set_ylim(freqs[0], freqs[-1])
        axs[i].set_yscale('log')
        axs[i].set_title(condition)
        # axs[0].legend()

    # plt.tight_layout()
    plt.show()


def plot_all_frequencies_contrast_individual_channels(data_path, behavior_path, regions, region_name, first_parcel_label, analysis_name, freqs_name='freqs', bad_epochs_path='/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs/bad_epochs/bad_epochs_hippocampus.npy'):
    """"""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*all_freqs.pkl')))
    subject_names = []
    print(subject_paths)

    bad_epochs = np.load(bad_epochs_path, allow_pickle=True).item()

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']

    rms_contrasts = [[] for c in range(len(conditions))]
    subject_id = [[] for c in range(len(conditions))]  # used for color

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        subject_bad_epochs = bad_epochs[subject_name]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label=="advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        good_epochs_mask = np.ones(rms.shape[1], dtype=bool)
        good_epochs_mask[subject_bad_epochs] = False
        rms = rms[:, good_epochs_mask, :]
        stim_names = np.array(rms_obj['stim_name'])[good_epochs_mask]

        for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
            correct = [e for e in range(len(stim_names)) if stim_names[e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(stim_names)) if stim_names[e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[2]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            observed_diff = np.mean(rms[:, correct, c], axis=1) - np.mean(rms[:, incorrect, c], axis=1)

                            # in t values
                            # se_correct = np.std(rms[:, correct, c], axis=1, ddof=1) / np.sqrt(len(correct))
                            # se_incorrect = np.std(rms[:, incorrect, c], axis=1, ddof=1) / np.sqrt(len(incorrect))
                            # observed_diff = observed_diff / np.sqrt(se_correct ** 2 + se_incorrect ** 2)

                            rms_contrasts[cond].append(observed_diff)
                            subject_id[cond].append(i)

    freqs = np.array(rms_obj[freqs_name])

    fig, axs = plt.subplots(1, len(conditions), sharey=True, sharex=True)
    subject_colors = plt.get_cmap('Set1').colors
    # Plot each condition's data in a separate subplot
    os.makedirs(os.path.join(data_path, 'contrasts'), exist_ok=True)
    for i, (condition, rms, s_id) in enumerate(zip(conditions, rms_contrasts, subject_id)):
        colors = [subject_colors[s] for s in s_id]
        rms = np.array(rms)
        print(rms.shape)
        print(freqs.shape)
        for c in range(rms.shape[0]):
            axs[i].plot(rms[c, :], freqs, color=colors[c], lw=1)
        axs[i].axvline(0, ls='--', c='k')
        axs[i].set_xlabel(f'{analysis_name} contrast')
        axs[i].set_ylim(freqs[0], freqs[-1])
        axs[i].set_yscale('log')
        axs[i].set_title(condition)
    axs[0].set_ylabel('Frequency (Hz)')

    # plt.show()
    fig.savefig(os.path.join(data_path, 'contrasts', f'{region_name}_all_subjects.svg'))
    plt.close()

    # by subject
    for s, _ in tqdm(enumerate(subject_paths)):
        fig, axs = plt.subplots(1, len(conditions), sharey=True, sharex=True)
        subject_colors = plt.get_cmap('Set1').colors
        # Plot each condition's data in a separate subplot
        for i, (condition, rms, s_id) in enumerate(zip(conditions, rms_contrasts, subject_id)):
            rms = np.array(rms)
            print(rms.shape)
            rms = rms[np.array(s_id) == s, :]
            axs[i].plot(rms.T, freqs, lw=2)
            axs[i].plot(np.mean(rms, axis=0), freqs, color='k', lw=4)
            axs[i].axvline(0, ls='--', c='k')
            axs[i].set_xlabel(f'{analysis_name} contrast')
            axs[i].set_ylim(freqs[0], freqs[-1])
            axs[i].set_yscale('log')
            axs[i].set_title(condition)
        axs[0].set_ylabel('Frequency (Hz)')
        plt.suptitle(subject_names[s])

        # plt.show()
        fig.savefig(os.path.join(data_path, 'contrasts', f'{region_name}_{subject_names[s]}.svg'))
        plt.close()


def plot_comodulogram_subjects(data_path, regions, region_name, first_parcel_label, bad_epochs_path='/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs/bad_epochs/bad_epochs_hippocampus.npy'):
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*como_all_freqs.pkl')))

    bad_epochs = np.load(bad_epochs_path, allow_pickle=True).item()

    subject_names = []
    comodulogram_list = []
    subject_id = []  # used for color

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)
        subject_names.append(subject_name)
        subject_bad_epochs = bad_epochs[subject_name]
        phase_coherence_obj = np.load(epochs_path, allow_pickle=True)
        phase_coherence = phase_coherence_obj['comodulogram']
        parcellations = phase_coherence_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        # select electrodes
        chan_ind = []
        kept_parcels = []
        if regions is not None:
            for p, parcel in enumerate(parcellations):
                if any(ch_parcel in regions for ch_parcel in parcel):
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])
        else:  # with regions=None, we keep all channels but the ones in go_next
            for p, parcel in enumerate(parcellations):
                if parcel[0] not in go_next:
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])

        if len(chan_ind) > 0:
            phase_coherence = phase_coherence[:, :, :, chan_ind]
            print(np.array(phase_coherence_obj['ch_names'])[chan_ind])
            stim_names = phase_coherence_obj['stim_name']
            good_epochs_mask = np.ones(phase_coherence.shape[2], dtype=bool)
            good_epochs_mask[subject_bad_epochs] = False
            phase_coherence = phase_coherence[:, :, good_epochs_mask, :]
            valid_indices = ~np.isnan(phase_coherence).any(axis=(0, 1, 3))
            phase_coherence = phase_coherence[:, :, valid_indices, :]
            # phase_coherence = np.mean(phase_coherence, axis=2)  # average epochs
            phase_coherence = np.mean(phase_coherence, axis=2) / (np.std(phase_coherence, ddof=1, axis=2) / np.sqrt(phase_coherence.shape[2]))  # average epochs and standardize
            for c in range(phase_coherence.shape[-1]):
                comodulogram_list.append(phase_coherence[:, :, c])  # add channels one by one
                subject_id.append(i)
    comodulogram_list = np.array(comodulogram_list)
    print(comodulogram_list.shape)

    low_freqs = np.array(phase_coherence_obj['low_freq'])
    high_freqs = np.array(phase_coherence_obj['high_freq'])

    # phase coherence average of all channels
    fig, axs = plt.subplots(1, 1)
    im = axs.pcolormesh(low_freqs, high_freqs, np.mean(comodulogram_list, axis=0), shading='nearest', cmap='viridis')
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('PAC', rotation=270)
    axs.set_xlabel('Frequency for phase (Hz)')
    axs.set_ylabel('Frequency for amplitude (Hz)')
    axs.set_yscale('log')
    axs.set_title('All subjects average')

    # plt.show()
    fig.savefig(os.path.join(data_path, f'{region_name}_all_subjects.svg'))
    plt.close()

    # individual subjects
    for i, _ in tqdm(enumerate(subject_list)):
        fig, axs = plt.subplots(1, 1)
        comodulogram_list_subject = comodulogram_list[np.array(subject_id) == i, :, :]
        im = axs.pcolormesh(low_freqs, high_freqs, np.mean(comodulogram_list_subject, axis=0), shading='nearest',
                            cmap='viridis')
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('PAC', rotation=270)
        axs.set_xlabel('Frequency for phase (Hz)')
        axs.set_ylabel('Frequency for amplitude (Hz)')
        axs.set_yscale('log')
        axs.set_title(subject_names[i])
    # plt.show()
        fig.savefig(os.path.join(data_path, f'{region_name}_{subject_names[i]}.svg'))
        plt.close()

    # save individual channels
    os.makedirs(os.path.join(data_path, 'individual_channels'), exist_ok=True)
    channel_colors = plt.get_cmap('tab10').colors
    for i, _ in tqdm(enumerate(subject_list)):
        comodulogram_list_subject = comodulogram_list[np.array(subject_id) == i, :, :]
        for c in range(comodulogram_list_subject.shape[0]):
            fig, axs = plt.subplots(1, 1)
            im = axs.pcolormesh(low_freqs, high_freqs, comodulogram_list_subject[c, :, :], shading='nearest', cmap='viridis')
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('PAC', rotation=270)
            axs.set_xlabel('Frequency for phase (Hz)')
            axs.set_ylabel('Frequency for amplitude (Hz)')
            axs.set_yscale('log')
            axs.set_title(subject_names[i] + f'_channel{c}', color=channel_colors[c%len(channel_colors)])
            fig.savefig(os.path.join(data_path, 'individual_channels', f'{region_name}_{subject_names[i]}_channel{c}.svg'))
            plt.close()


def plot_comodulogram_contrasts_subjects(data_path, behavior_path, regions, region_name, first_parcel_label, bad_epochs_path='/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs/bad_epochs/bad_epochs_hippocampus.npy'):
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*como_all_freqs.pkl')))

    bad_epochs = np.load(bad_epochs_path, allow_pickle=True).item()

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']
    subject_names = []
    comodulogram_contrast = [[] for c in range(len(conditions))]
    subject_id = [[] for c in range(len(conditions))]  # used for color

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)
        subject_names.append(subject_name)
        subject_bad_epochs = bad_epochs[subject_name]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))][
            'stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))][
            'stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        phase_coherence_obj = np.load(epochs_path, allow_pickle=True)
        phase_coherence = phase_coherence_obj['comodulogram']
        parcellations = phase_coherence_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        # select electrodes
        chan_ind = []
        kept_parcels = []
        if regions is not None:
            for p, parcel in enumerate(parcellations):
                if any(ch_parcel in regions for ch_parcel in parcel):
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])
        else:  # with regions=None, we keep all channels but the ones in go_next
            for p, parcel in enumerate(parcellations):
                if parcel[0] not in go_next:
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])

        if len(chan_ind) > 0:
            phase_coherence = phase_coherence[:, :, :, chan_ind]
            good_epochs_mask = np.ones(phase_coherence.shape[2], dtype=bool)
            good_epochs_mask[subject_bad_epochs] = False
            phase_coherence = phase_coherence[:, :, good_epochs_mask, :]
            stim_names = np.array(phase_coherence_obj['stim_name'])[good_epochs_mask]

            for cond, correct_incorrect in enumerate(
                    [[correct_all, incorrect_all], [correct_encoding, incorrect_encoding],
                     [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval],
                     [memorized, novel]]):
                correct = [e for e in range(len(stim_names)) if stim_names[e] in correct_incorrect[0]]
                incorrect = [e for e in range(len(stim_names)) if stim_names[e] in correct_incorrect[1]]
                if (len(correct) > 0) & (len(incorrect) > 0):
                    for c in range(phase_coherence.shape[3]):
                            observed_diff = np.mean(phase_coherence[:, :, correct, c], axis=2) - np.mean(phase_coherence[:, :, incorrect, c], axis=2)

                            # in t values
                            se_correct = np.std(phase_coherence[:, :, correct, c], axis=2, ddof=1) / np.sqrt(len(correct))
                            se_incorrect = np.std(phase_coherence[:, :, incorrect, c], axis=2, ddof=1) / np.sqrt(len(incorrect))
                            observed_diff = observed_diff / np.sqrt(se_correct ** 2 + se_incorrect ** 2)

                            comodulogram_contrast[cond].append(observed_diff)  # add contrast
                            subject_id[cond].append(i)

            # print(np.array(phase_coherence_obj['ch_names'])[chan_ind])
            # stim_names = phase_coherence_obj['stim_name']
            # valid_indices = ~np.isnan(phase_coherence).any(axis=(0, 1, 3))
            # phase_coherence = phase_coherence[:, :, valid_indices, :]
            # # phase_coherence = np.mean(phase_coherence, axis=2)  # average epochs
            # phase_coherence = np.mean(phase_coherence, axis=2) / np.std(phase_coherence, axis=2)  # average epochs and standardize
            # for c in range(phase_coherence.shape[-1]):
            #     comodulogram_list.append(phase_coherence[:, :, c])  # add channels one by one
            #     subject_id.append(i)

    low_freqs = np.array(phase_coherence_obj['low_freq'])
    high_freqs = np.array(phase_coherence_obj['high_freq'])

    for i, (condition, comodulogram_list, s_id) in enumerate(zip(conditions, comodulogram_contrast, subject_id)):
        os.makedirs(os.path.join(data_path, condition, 'individual_channels'), exist_ok=True)
        comodulogram_list = np.array(comodulogram_list)
        print(comodulogram_list.shape)

        # phase coherence average of all channels
        fig, axs = plt.subplots(1, 1)
        vmax = np.max(np.abs(np.mean(comodulogram_list, axis=0)))
        im = axs.pcolormesh(low_freqs, high_freqs, np.mean(comodulogram_list, axis=0), shading='nearest', cmap='Spectral_r', vmin=-vmax, vmax=vmax)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('PAC', rotation=270)
        axs.set_xlabel('Frequency for phase (Hz)')
        axs.set_ylabel('Frequency for amplitude (Hz)')
        axs.set_yscale('log')
        axs.set_title(f'{condition} all subjects average')

        # plt.show()
        fig.savefig(os.path.join(data_path, condition, f'{region_name}_all_subjects.svg'))
        plt.close()

        # individual subjects
        for i, _ in tqdm(enumerate(subject_list)):
            fig, axs = plt.subplots(1, 1)
            comodulogram_list_subject = comodulogram_list[np.array(s_id) == i, :, :]
            vmax = np.max(np.abs(np.mean(comodulogram_list_subject, axis=0)))
            im = axs.pcolormesh(low_freqs, high_freqs, np.mean(comodulogram_list_subject, axis=0), shading='nearest', cmap='Spectral_r', vmin=-vmax, vmax=vmax)
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('PAC', rotation=270)
            axs.set_xlabel('Frequency for phase (Hz)')
            axs.set_ylabel('Frequency for amplitude (Hz)')
            axs.set_yscale('log')
            axs.set_title(condition + '_' + subject_names[i])
        # plt.show()
            fig.savefig(os.path.join(data_path, condition, f'{region_name}_{subject_names[i]}.svg'))
            plt.close()

        # save individual channels
        channel_colors = plt.get_cmap('tab10').colors
        for i, _ in tqdm(enumerate(subject_list)):
            comodulogram_list_subject = comodulogram_list[np.array(s_id) == i, :, :]
            for c in range(comodulogram_list_subject.shape[0]):
                fig, axs = plt.subplots(1, 1)
                vmax = np.max(np.abs(comodulogram_list_subject[c, :, :]))
                im = axs.pcolormesh(low_freqs, high_freqs, comodulogram_list_subject[c, :, :], shading='nearest', cmap='Spectral_r', vmin=-vmax, vmax=vmax)
                cbar = plt.colorbar(im)
                cbar.ax.set_ylabel('PAC', rotation=270)
                axs.set_xlabel('Frequency for phase (Hz)')
                axs.set_ylabel('Frequency for amplitude (Hz)')
                axs.set_yscale('log')
                axs.set_title(condition + '_' + subject_names[i] + f'_channel{c}', color=channel_colors[c%len(channel_colors)])
                fig.savefig(os.path.join(data_path, condition, 'individual_channels', f'{region_name}_{subject_names[i]}_channel{c}.svg'))
                plt.close()


def plot_theta_beta_and_channel_positions(data_path, regions, first_parcel_label, analysis_name, freqs_name='freqs', subject_colors=True, bad_epochs_path='/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs/bad_epochs/bad_epochs_hippocampus.npy'):
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*all_freqs.pkl')))

    bad_epochs = np.load(bad_epochs_path, allow_pickle=True).item()

    subject_names = []
    phase_coherence_list = []
    channel_positions_list = []
    subject_id = []  # used for color

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)
        subject_names.append(subject_name)
        subject_bad_epochs = bad_epochs[subject_name]
        phase_coherence_obj = np.load(epochs_path, allow_pickle=True)
        phase_coherence = phase_coherence_obj[analysis_name]
        parcellations = phase_coherence_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        # select electrodes
        chan_ind = []
        kept_parcels = []
        if regions is not None:
            for p, parcel in enumerate(parcellations):
                if any(ch_parcel in regions for ch_parcel in parcel):
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])
        else:  # with regions=None, we keep all channels but the ones in go_next
            for p, parcel in enumerate(parcellations):
                if parcel[0] not in go_next:
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])

        if len(chan_ind) > 0:
            phase_coherence = phase_coherence[:, :, chan_ind]
            channel_names = np.array(phase_coherence_obj['ch_names'])[chan_ind]
            print(channel_names)
            stim_names = phase_coherence_obj['stim_name']
            good_epochs_mask = np.ones(phase_coherence.shape[1], dtype=bool)
            good_epochs_mask[subject_bad_epochs] = False
            phase_coherence = phase_coherence[:, good_epochs_mask, :]
            valid_indices = ~np.isnan(phase_coherence).any(axis=(0, 2))
            phase_coherence = phase_coherence[:, valid_indices, :]
            # phase_coherence = np.mean(phase_coherence, axis=1)  # average epochs
            phase_coherence = np.mean(phase_coherence, axis=1) / (np.std(phase_coherence, axis=1, ddof=1) / np.sqrt(phase_coherence.shape[1]))  # average epochs
            for c in range(phase_coherence.shape[-1]):
                phase_coherence_list.append(phase_coherence[:, c])  # add channels one by one
                channel_positions_list.append(phase_coherence_obj['info']['chs'][chan_ind[c]]['loc'][:3])
                subject_id.append(i)
    phase_coherence_list = np.array(phase_coherence_list)
    print(phase_coherence_list.shape)

    freqs = np.array(phase_coherence_obj[freqs_name])
    print(freqs)

    theta_indices = (freqs>=3) & (freqs<=9)
    # theta_indices = (freqs>=4) & (freqs<=8)
    # theta_indices = (freqs>=1) & (freqs<=8)
    beta_indices = (freqs>=12) & (freqs<=30)
    theta_freqs = freqs[theta_indices]
    theta = phase_coherence_list[:, theta_indices]
    beta_freqs = freqs[beta_indices]
    beta = phase_coherence_list[:, beta_indices]

    theta_max = np.max(theta, axis=1)
    theta_max_freq = theta_freqs[np.argmax(theta, axis=1)]
    beta_max = np.max(beta, axis=1)
    beta_max_freq = beta_freqs[np.argmax(beta, axis=1)]

    # # remove peaks at the boarders, since they happen when no peak was detected in the range
    # valid_mask = (theta_max_freq != theta_freqs[0]) & (theta_max_freq != theta_freqs[-1])
    # theta_max = theta_max[valid_mask]
    # theta_max_freq = theta_max_freq[valid_mask]

    if subject_colors:
        subject_colors = plt.get_cmap('Set1').colors
        colors = [subject_colors[s] for s in subject_id]
    else:
        colors = 'k'

    # phase coherence by channel on one plot
    fig, axs = plt.subplots(4, 3, sharex="col")
    dimensions = ['left-right (abs: distance from center)', 'posterior-anterior', 'ventral-dorsal']
    conditions = ['Theta peak power', 'Theta peak frequency', 'Beta peak power', 'Beta peak frequency']
    for d, dimension_name in enumerate(dimensions):
        dim = [positions[d] for positions in channel_positions_list]
        if d==0:
            dim = np.abs(dim)  # for left-right, we transform position to distance from center
        for row, (cond, values) in enumerate(zip(conditions, [theta_max, theta_max_freq, beta_max, beta_max_freq])):
            print(cond, dimension_name)
            # if cond in ['Theta peak power', 'Theta peak frequency']:
            if cond in ['No']:
                current_dim = np.array(dim)[valid_mask]  # for theta, remove extremes
                current_colors = np.array(colors)[valid_mask]
                current_subject_id = np.array(subject_id)[valid_mask]
            else:
                current_dim = dim
                current_colors = colors
                current_subject_id = subject_id
            axs[row, d].scatter(current_dim, values, color=current_colors)
            if d==0:
                axs[row, d].set_ylabel(cond)
            if row==0:
                axs[row, d].set_title(dimension_name)

            # Create a DataFrame for the mixed model
            data_dict = {
                'Value': values,
                'Position': current_dim,
                'SubjectID': current_subject_id
            }
            # model_data = pd.DataFrame(data_dict).dropna()  # Remove any NaNs
            model_data = pd.DataFrame(data_dict)
            # Fit the mixed model
            model = MixedLM.from_formula('Value ~ Position', groups='SubjectID', data=model_data)
            result = model.fit()
            p_value = result.pvalues['Position']

            # fitted line
            x_vals = np.linspace(min(current_dim), max(current_dim), 100)
            y_vals = result.params['Intercept'] + result.params['Position'] * x_vals
            axs[row, d].plot(x_vals, y_vals, color='k', linestyle='--', linewidth=1.5)

            axs[row, d].text(
                0.05, 0.95, f'p={p_value:.3g}', transform=axs[row, d].transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

    plt.show()
    # fig.savefig(os.path.join(data_path, f'{regions}_{analysis_name}_all_subjects.svg'))
    # plt.close()


def plot_scalar_and_channel_positions(data_path, behavior_path, regions, first_parcel_label, analysis_name, subject_colors=True, bad_epochs_path='/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs/bad_epochs/bad_epochs_hippocampus.npy'):
    path = Path(data_path)
    subject_list = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))

    behavior_all = pd.read_csv(behavior_path)

    bad_epochs = np.load(bad_epochs_path, allow_pickle=True).item()

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']

    subject_names = []
    scalar_contrasts = [[] for c in range(len(conditions))]
    channel_positions_list = [[] for c in range(len(conditions))]
    subject_id = [[] for c in range(len(conditions))]  # used for color

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)
        subject_names.append(subject_name)
        subject_bad_epochs = bad_epochs[subject_name]

        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))][
            'stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))][
            'stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        scalar_obj = np.load(epochs_path, allow_pickle=True)
        scalar = scalar_obj[analysis_name]
        parcellations = scalar_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        # select electrodes
        chan_ind = []
        kept_parcels = []
        if regions is not None:
            for p, parcel in enumerate(parcellations):
                if any(ch_parcel in regions for ch_parcel in parcel):
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])
        else:  # with regions=None, we keep all channels but the ones in go_next
            for p, parcel in enumerate(parcellations):
                if parcel[0] not in go_next:
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])

        if len(chan_ind) > 0:
            scalar = scalar[:, chan_ind]
            channel_names = np.array(scalar_obj['ch_names'])[chan_ind]
            print(channel_names)
            good_epochs_mask = np.ones(scalar.shape[0], dtype=bool)
            good_epochs_mask[subject_bad_epochs] = False
            scalar = scalar[good_epochs_mask, :]
            stim_names = np.array(scalar_obj['stim_name'])[good_epochs_mask]
            valid_indices = ~np.isnan(scalar).any(axis=1)
            scalar = scalar[valid_indices, :]
            for cond, correct_incorrect in enumerate(
                    [[correct_all, incorrect_all], [correct_encoding, incorrect_encoding],
                     [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval],
                     [memorized, novel]]):
                correct = [e for e in range(len(stim_names)) if stim_names[e] in correct_incorrect[0]]
                incorrect = [e for e in range(len(stim_names)) if stim_names[e] in correct_incorrect[1]]
                if (len(correct) > 0) & (len(incorrect) > 0):
                    for c in range(scalar.shape[-1]):
                        scalar_contrasts[cond].append(np.mean(scalar[correct, c]) - np.mean(scalar[incorrect, c]))  # add channels one by one
                        channel_positions_list[cond].append(scalar_obj['info']['chs'][chan_ind[c]]['loc'][:3])
                        subject_id[cond].append(i)

    fig, axs = plt.subplots(len(conditions), 3, sharex="col")
    dimensions = ['left-right (abs: distance from center)', 'posterior-anterior', 'ventral-dorsal']
    for d, dimension_name in enumerate(dimensions):
        for row, (cond, values, s_id, channel_positions) in enumerate(zip(conditions, scalar_contrasts, subject_id, channel_positions_list)):
            if subject_colors:
                subject_colors = plt.get_cmap('Set1').colors
                colors = [subject_colors[s] for s in s_id]
            else:
                colors = 'k'
            dim = [positions[d] for positions in channel_positions]
            if d == 0:
                dim = np.abs(dim)  # for left-right, we transform position to distance from center
            print(cond, dimension_name)
            current_dim = dim
            axs[row, d].scatter(current_dim, values, color=colors)
            if d==0:
                axs[row, d].set_ylabel(cond)
            if row==0:
                axs[row, d].set_title(dimension_name)

            # Create a DataFrame for the mixed model
            data_dict = {
                'Value': values,
                'Position': current_dim,
                'SubjectID': s_id
            }
            # model_data = pd.DataFrame(data_dict).dropna()  # Remove any NaNs
            model_data = pd.DataFrame(data_dict)
            # Fit the mixed model
            model = MixedLM.from_formula('Value ~ Position', groups='SubjectID', data=model_data)
            result = model.fit()
            p_value = result.pvalues['Position']

            # fitted line
            x_vals = np.linspace(min(current_dim), max(current_dim), 100)
            y_vals = result.params['Intercept'] + result.params['Position'] * x_vals
            axs[row, d].plot(x_vals, y_vals, color='k', linestyle='--', linewidth=1.5)

            axs[row, d].text(
                0.05, 0.95, f'p={p_value:.3g}', transform=axs[row, d].transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

    # novelty vs. retrieval analysis
    retrieval_minus_novelty_list = []
    s_list = []
    reduced_position_list = []
    for i in range(len(subject_list)):
        subj_channels_novelty = np.array(subject_id[2]) == i
        subj_channels_retrieval = np.array(subject_id[3]) == i
        if (np.sum(subj_channels_novelty) == np.sum(subj_channels_retrieval)) and (np.sum(subj_channels_novelty) > 1):
            contrast_novelty = np.array(scalar_contrasts[2])[subj_channels_novelty]
            contrast_retrieval = np.array(scalar_contrasts[3])[subj_channels_retrieval]
            retrieval_minus_novelty = contrast_retrieval - contrast_novelty
            retrieval_minus_novelty_list += retrieval_minus_novelty.tolist()
            s_list += [i for _ in range(len(contrast_retrieval))]
            reduced_position_list += np.array(channel_positions_list[2])[subj_channels_novelty].tolist()

    # novelty vs retrieval
    fig, axs = plt.subplots(1, 3, sharex="col")
    dimensions = ['left-right (abs: distance from center)', 'posterior-anterior', 'ventral-dorsal']
    for d, dimension_name in enumerate(dimensions):
        subject_colors = plt.get_cmap('Set1').colors
        colors = [subject_colors[s] for s in s_list]
        dim = [positions[d] for positions in reduced_position_list]
        if d == 0:
            dim = np.abs(dim)  # for left-right, we transform position to distance from center
        current_dim = dim
        axs[d].scatter(current_dim, retrieval_minus_novelty_list, color=colors)
        axs[d].set_title(dimension_name)

        # Create a DataFrame for the mixed model
        data_dict = {
            'Value': retrieval_minus_novelty_list,
            'Position': current_dim,
            'SubjectID': s_list
        }
        # model_data = pd.DataFrame(data_dict).dropna()  # Remove any NaNs
        model_data = pd.DataFrame(data_dict)
        # Fit the mixed model
        model = MixedLM.from_formula('Value ~ Position', groups='SubjectID', data=model_data)
        result = model.fit()
        p_value = result.pvalues['Position']

        # fitted line
        x_vals = np.linspace(min(current_dim), max(current_dim), 100)
        y_vals = result.params['Intercept'] + result.params['Position'] * x_vals
        axs[d].plot(x_vals, y_vals, color='k', linestyle='--', linewidth=1.5)

        axs[d].text(
            0.05, 0.95, f'p={p_value:.3g}', transform=axs[d].transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        # Plot individual slopes per subject
        unique_subjects = np.unique(s_list)
        for s in unique_subjects:
            # Filter data for the current subject
            subject_data = model_data[model_data['SubjectID'] == s]
            if len(subject_data) > 1:  # Ensure enough points to fit a line
                x_subject = subject_data['Position']
                y_subject = subject_data['Value']
                coeffs = np.polyfit(x_subject, y_subject, 1)  # Linear fit
                # Clip the regression line to the range of current_dim for this subject
                x_clip = np.linspace(min(x_subject), max(x_subject), 100)
                y_clip = coeffs[1] + coeffs[0] * x_clip
                axs[d].plot(
                    x_clip, y_clip, color=subject_colors[s], linestyle='-', linewidth=1, alpha=0.6
                )

    axs[0].set_ylabel("Retrieval contrast - Novelty contrast")

    plt.show()
    # fig.savefig(os.path.join(data_path, f'{regions}_{analysis_name}_all_subjects.svg'))
    # plt.close()


def pac_at_peak_theta(data_path, como_folder, save_folder, regions, region_name, first_parcel_label, bad_epochs_path='/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs/bad_epochs/bad_epochs_hippocampus.npy'):
    """compute peak theta and PAC at the phase of that frequency for each electrode."""
    os.makedirs(save_folder, exist_ok=True)

    path = Path(data_path)
    subject_list = sorted(list(path.glob('*all_freqs.pkl')))

    bad_epochs = np.load(bad_epochs_path, allow_pickle=True).item()

    subject_names = []
    pac_at_indiv_freq_list = []
    subject_id = []  # used for color

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)
        subject_names.append(subject_name)
        subject_bad_epochs = bad_epochs[subject_name]
        power_obj = np.load(epochs_path, allow_pickle=True)
        theta_power = power_obj['power']
        parcellations = power_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        # load comodulogram of that subject
        comodulogram_obj = np.load(os.path.join(como_folder, f'{subject_name}_epochs_como_all_freqs.pkl'), allow_pickle=True)
        comodulogram = comodulogram_obj['comodulogram']

        # select electrodes
        chan_ind = []
        kept_parcels = []
        if regions is not None:
            for p, parcel in enumerate(parcellations):
                if any(ch_parcel in regions for ch_parcel in parcel):
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])
        else:  # with regions=None, we keep all channels but the ones in go_next
            for p, parcel in enumerate(parcellations):
                if parcel[0] not in go_next:
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])

        if len(chan_ind) > 0:
            theta_power = theta_power[:, :, chan_ind]
            comodulogram = comodulogram[:, :, :, chan_ind]
            channel_names = np.array(power_obj['ch_names'])[chan_ind]
            print(channel_names)
            stim_names = power_obj['stim_name']
            good_epochs_mask = np.ones(theta_power.shape[1], dtype=bool)
            good_epochs_mask[subject_bad_epochs] = False
            theta_power = theta_power[:, good_epochs_mask, :]
            valid_indices = ~np.isnan(theta_power).any(axis=(0, 2))
            theta_power = theta_power[:, valid_indices, :]
            theta_power = np.mean(theta_power, axis=1)  # average epochs
            # theta_power = np.mean(theta_power, axis=1) / (np.std(theta_power, axis=1, ddof=1) / np.sqrt(theta_power.shape[1]))  # average epochs

            comodulogram = comodulogram[:, :, good_epochs_mask, :]
            valid_indices = ~np.isnan(comodulogram).any(axis=(0, 1, 3))
            comodulogram = comodulogram[:, :, valid_indices, :]
            comodulogram = np.mean(comodulogram, axis=2)  # average epochs
            # comodulogram = np.mean(comodulogram, axis=2) / (np.std(comodulogram, axis=2, ddof=1) / np.sqrt(comodulogram.shape[2]))  # average epochs

            low_freqs = np.array(comodulogram_obj['low_freq'])

            freqs = np.array(power_obj['freq'])
            # theta_indices = (freqs >= 3) & (freqs <= 9)
            theta_indices = (freqs >= 4) & (freqs <= 8)
            theta_freqs = freqs[theta_indices]
            theta_power = theta_power[theta_indices, :]

            for c in range(theta_power.shape[-1]):
                max_theta_freq = theta_freqs[np.argmax(theta_power[:, c])]
                # max_theta_freq = 2
                phase_frequency = np.argmin(np.abs(low_freqs - max_theta_freq))  # the comodulogram object does not have necessarily the same frequencies
                pac_at_indiv_freq_list.append(comodulogram[phase_frequency, :, c])  # pac at the max theta frequency, channel one by one

                # phase_frequency_low = np.argmin(np.abs(low_freqs - theta_freqs[0]))  # the comodulogram object does not have necessarily the same frequencies
                # phase_frequency_high = np.argmin(np.abs(low_freqs - theta_freqs[-1]))
                # pac_at_indiv_freq_list.append(np.mean(comodulogram[phase_frequency_low:phase_frequency_high+1, :, c], axis=0))  # pac at the max theta frequency, channel one by one

                subject_id.append(i)
    pac_at_indiv_freq_list = np.array(pac_at_indiv_freq_list)
    print(pac_at_indiv_freq_list.shape)

    high_freqs = np.array(comodulogram_obj['high_freq'])
    print(high_freqs)

    subject_colors = plt.get_cmap('Set1').colors
    colors = [subject_colors[s] for s in subject_id]

    # pac by channel on one plot
    fig, axs = plt.subplots(1, 1)
    for c in range(pac_at_indiv_freq_list.shape[0]):
        axs.plot(pac_at_indiv_freq_list[c, :], high_freqs, color=colors[c], lw=1)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=7) for name, color in zip(subject_names, subject_colors)]
    plt.legend(handles=legend_elements, loc="best", title="Subjects")
    axs.axvline(0, ls='--', c='k')
    axs.set_xlabel("PAC at individual channel's phase frequency")
    axs.set_ylabel('Frequency (Hz)')
    axs.set_ylim(high_freqs[0], high_freqs[-1])
    axs.set_yscale('log')

    # plt.show()
    fig.savefig(os.path.join(save_folder, f'{region_name}_pac_individual_phase_all_subjects.svg'))
    plt.close()

    # individual subjects
    fig, axs = plt.subplots(1, 1)
    for i, _ in tqdm(enumerate(subject_list)):
        pac_at_indiv_freq_list_subject = pac_at_indiv_freq_list[np.array(subject_id) == i, :]
        axs.plot(np.mean(pac_at_indiv_freq_list_subject, axis=0), high_freqs, color=subject_colors[i], lw=3, label=subject_names[i])
    axs.axvline(0, ls='--', c='k')
    axs.set_xlabel("PAC at individual channel's phase frequency")
    axs.set_ylabel('Amplitude frequency (Hz)')
    axs.set_ylim(high_freqs[0], high_freqs[-1])
    axs.set_yscale('log')
    axs.legend()
    fig.savefig(os.path.join(save_folder, f'{region_name}_pac_individual_phase_all_subjects_average_channels.svg'))
    plt.close()

    # individual subjects
    for i, _ in tqdm(enumerate(subject_list)):
        fig, axs = plt.subplots(1, 1)
        pac_at_indiv_freq_list_subject = pac_at_indiv_freq_list[np.array(subject_id) == i, :]
        # for c in range(pac_at_indiv_freq_list_subject.shape[0]):
        axs.plot(pac_at_indiv_freq_list_subject.T, high_freqs, lw=2)
        axs.plot(np.mean(pac_at_indiv_freq_list_subject, axis=0), high_freqs, color='k', lw=4)
        axs.axvline(0, ls='--', c='k')
        axs.set_xlabel("PAC at individual channel's phase frequency")
        axs.set_ylabel('Amplitude frequency (Hz)')
        axs.set_ylim(high_freqs[0], high_freqs[-1])
        axs.set_yscale('log')
        axs.set_title(subject_names[i])
        # plt.show()
        fig.savefig(os.path.join(save_folder, f'{region_name}_pac_individual_phase_{subject_names[i]}.svg'))
        plt.close()


def pac_at_peak_theta_contrasts(data_path, como_folder, behavior_path, save_folder, regions, region_name, first_parcel_label, bad_epochs_path='/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs/bad_epochs/bad_epochs_hippocampus.npy'):
    """compute peak theta and PAC at the phase of that frequency for each electrode."""
    os.makedirs(save_folder, exist_ok=True)

    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*all_freqs.pkl')))

    bad_epochs = np.load(bad_epochs_path, allow_pickle=True).item()

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']

    subject_names = []
    pac_at_indiv_freq_list = [[] for c in range(len(conditions))]
    signif_freqs_list = [[] for c in range(len(conditions))]
    subject_id = [[] for c in range(len(conditions))]  # used for color

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)
        subject_names.append(subject_name)
        subject_bad_epochs = bad_epochs[subject_name]
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
        incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
        memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()

        power_obj = np.load(epochs_path, allow_pickle=True)
        theta_power = power_obj['power']
        parcellations = power_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        # load comodulogram of that subject
        comodulogram_obj = np.load(os.path.join(como_folder, f'{subject_name}_epochs_como_all_freqs.pkl'), allow_pickle=True)
        comodulogram = comodulogram_obj['comodulogram']

        good_epochs_mask = np.ones(theta_power.shape[1], dtype=bool)
        good_epochs_mask[subject_bad_epochs] = False
        theta_power = theta_power[:, good_epochs_mask, :]
        comodulogram = comodulogram[:, :, good_epochs_mask, :]
        stim_names = np.array(power_obj['stim_name'])[good_epochs_mask]

        # select electrodes
        chan_ind = []
        kept_parcels = []
        if regions is not None:
            for p, parcel in enumerate(parcellations):
                if any(ch_parcel in regions for ch_parcel in parcel):
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])
        else:  # with regions=None, we keep all channels but the ones in go_next
            for p, parcel in enumerate(parcellations):
                if parcel[0] not in go_next:
                    chan_ind.append(p)
                    kept_parcels.append(parcel[0])

        if len(chan_ind) > 0:
            theta_power = theta_power[:, :, chan_ind]
            comodulogram = comodulogram[:, :, :, chan_ind]
            channel_names = np.array(power_obj['ch_names'])[chan_ind]
            print(channel_names)
            theta_power = np.mean(theta_power, axis=1)  # average epochs
            # theta_power = np.mean(theta_power, axis=1) / (np.std(theta_power, axis=1, ddof=1) / np.sqrt(theta_power.shape[1]))  # average epochs

            low_freqs = np.array(comodulogram_obj['low_freq'])
            high_freqs = np.array(comodulogram_obj['high_freq'])

            # Theta peak is computed on all epochs
            freqs = np.array(power_obj['freq'])
            # theta_indices = (freqs >= 3) & (freqs <= 9)
            theta_indices = (freqs >= 4) & (freqs <= 8)
            theta_freqs = freqs[theta_indices]
            theta_power = theta_power[theta_indices, :]

            for c in range(theta_power.shape[-1]):
                max_theta_freq = theta_freqs[np.argmax(theta_power[:, c])]
                # max_theta_freq = 2
                phase_frequency = np.argmin(np.abs(low_freqs - max_theta_freq))  # the comodulogram object does not have necessarily the same frequencies
                channel_como = comodulogram[phase_frequency, :, :, c]

                # phase_frequency_low = np.argmin(np.abs(low_freqs - theta_freqs[0]))  # the comodulogram object does not have necessarily the same frequencies
                # phase_frequency_high = np.argmin(np.abs(low_freqs - theta_freqs[-1]))
                # comodulogram[phase_frequency_low:phase_frequency_high + 1, :, :, c] = (comodulogram[phase_frequency_low:phase_frequency_high + 1, :, :, c] - np.mean(comodulogram[phase_frequency_low:phase_frequency_high + 1, :, :, c], axis=(1, 2), keepdims=True)) / np.std(comodulogram[phase_frequency_low:phase_frequency_high + 1, :, :, c], axis=(1, 2), keepdims=True)  # z-score by phase frequency
                # channel_como = np.mean(comodulogram[phase_frequency_low:phase_frequency_high + 1, :, :, c], axis=0)

                for cond, correct_incorrect in enumerate(
                        [[correct_all, incorrect_all], [correct_encoding, incorrect_encoding],
                         [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval],
                         [memorized, novel]]):
                    correct = [e for e in range(len(stim_names)) if stim_names[e] in correct_incorrect[0]]
                    incorrect = [e for e in range(len(stim_names)) if stim_names[e] in correct_incorrect[1]]
                    if (len(correct) > 0) & (len(incorrect) > 0):
                        observed_diff = np.mean(channel_como[:, correct], axis=1) - np.mean(channel_como[:, incorrect], axis=1)

                        # in t values
                        # se_correct = np.std(channel_como[:, correct], axis=1, ddof=1) / np.sqrt(len(correct))
                        # se_incorrect = np.std(channel_como[:, incorrect], axis=1, ddof=1) / np.sqrt(len(incorrect))
                        # observed_diff = observed_diff / np.sqrt(se_correct ** 2 + se_incorrect ** 2)

                        pac_at_indiv_freq_list[cond].append(observed_diff)  # pac at the max theta frequency, channel one by one
                        subject_id[cond].append(i)

                        # permutation cluster test
                        data = [channel_como[:, correct].T, channel_como[:, incorrect].T]
                        pval = 0.05
                        df = len(correct) + len(incorrect) - 2
                        thresh = scipy.stats.t.ppf(1 - pval / 2, df=df)  # T distribution, two-tailed
                        T_obs, clusters, cluster_p_vals, _ = permutation_cluster_test(
                            data, n_permutations=1000, tail=0, stat_fun=mne.stats.ttest_ind_no_p, threshold=thresh, n_jobs=-1, seed=42)
                        significant_clusters = np.where(cluster_p_vals < 0.05)[0]
                        signif_freqs_list[cond].append(np.zeros_like(high_freqs, dtype=bool))
                        for cluster_idx in significant_clusters:
                            cluster_freqs = clusters[cluster_idx]
                            signif_freqs_list[cond][-1][cluster_freqs] = True

    os.makedirs(os.path.join(save_folder, 'contrasts'), exist_ok=True)

    subject_colors = plt.get_cmap('Set1').colors
    # pac by channel on one plot
    fig, axs = plt.subplots(1, len(conditions), sharey=True, sharex=True)
    for i, (condition, pac_at_indiv_freq, s_id) in enumerate(zip(conditions, pac_at_indiv_freq_list, subject_id)):
        colors = [subject_colors[s] for s in s_id]
        pac_at_indiv_freq = np.array(pac_at_indiv_freq)
        for c in range(pac_at_indiv_freq.shape[0]):
            axs[i].plot(pac_at_indiv_freq[c, :], high_freqs, color=colors[c], lw=1)
        axs[i].axvline(0, ls='--', c='k')
        axs[i].set_ylim(high_freqs[0], high_freqs[-1])
        axs[i].set_yscale('log')
        axs[i].set_title(condition)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=color, markersize=7) for name, color in zip(subject_names, subject_colors)]
    axs[0].legend(handles=legend_elements, loc="best", title="Subjects")
    axs[int(len(conditions)/2)].set_xlabel("Contrast of PAC at individual channel's phase frequency")
    axs[0].set_ylabel('Amplitude frequency (Hz)')

    # plt.show()
    fig.savefig(os.path.join(save_folder, 'contrasts', f'{region_name}_pac_individual_phase_all_subjects.svg'))
    plt.close()

    colors = plt.get_cmap('Set1').colors
    # individual subjects
    for s, _ in tqdm(enumerate(subject_list)):
        fig, axs = plt.subplots(1, len(conditions), sharey=True, sharex=True)
        for i, (condition, pac_at_indiv_freq, s_id, signif_freqs) in enumerate(zip(conditions, pac_at_indiv_freq_list, subject_id, signif_freqs_list)):
            pac_at_indiv_freq = np.array(pac_at_indiv_freq)
            pac_at_indiv_freq = pac_at_indiv_freq[np.array(s_id) == s, :]
            signif_freqs = np.array(signif_freqs)[np.array(s_id) == s]
            for c in range(pac_at_indiv_freq.shape[0]):
                # line_color = np.array(['gray' for _ in range(pac_at_indiv_freq.shape[1])])
                # line_color[signif_freqs[c]] = colors[c%len(colors)]
                # line_color = [colors[c%len(colors)] if sign else 'gray' for sign in signif_freqs[c]]
                # axs[i].plot(pac_at_indiv_freq[c, :], high_freqs, color=line_color, lw=2)
                for j in range(len(high_freqs) - 1):  # Loop through segments
                    if signif_freqs[c][j]:
                        segment_color = colors[c % len(colors)]
                        width = 2
                    else:
                        segment_color = 'gray'
                        width = 0.5
                    axs[i].plot(pac_at_indiv_freq[c, j-1:j+1], high_freqs[j-1:j+1], color=segment_color, lw=width)
            # axs[i].plot(np.mean(pac_at_indiv_freq, axis=0), high_freqs, color='k', lw=4)
            axs[i].axvline(0, ls='--', c='k')
            axs[i].set_ylim(high_freqs[0], high_freqs[-1])
            axs[i].set_yscale('log')
            axs[i].set_title(condition)
        axs[int(len(conditions)/2)].set_xlabel("Contrast of PAC (tval) at individual channel's phase frequency")
        axs[0].set_ylabel('Amplitude frequency (Hz)')
        plt.suptitle(subject_names[s])
        # plt.show()
        fig.savefig(os.path.join(save_folder, 'contrasts', f'{region_name}_pac_individual_phase_{subject_names[s]}.svg'))
        plt.close()


def pac_ushape_by_region(data_path):
    encoding_data = pd.read_csv(data_path)
    # encoding_data = encoding_data.dropna(subset=['correct', 'correct2'])
    print(encoding_data)
    print(encoding_data.columns)

    encoding_data['surprise_quantile'] = pd.qcut(encoding_data['surprise'], q=6, labels=False) + 1  # Quantiles labeled 1 to 6

    # Group by electrode_region
    regions = encoding_data['electrode_region'].unique()
    region_results = {}

    for region in regions:
        region_data = encoding_data[encoding_data['electrode_region'] == region]
        electrode_results = []

        # Iterate over unique (subject, electrode) combinations
        for (subject, electrode), group in region_data.groupby(['subject', 'electrode']):
            # Quadratic fit: pac ~ surprise + surprise^2
            X = np.vstack([group['surprise_quantile'], group['surprise_quantile'] ** 2]).T
            X = sm.add_constant(X)  # Add intercept
            y = group['pac_stim_freq']

            model = sm.OLS(y, X).fit()
            quadratic_term = model.params[2]  # Coefficient for surprise^2
            electrode_results.append(quadratic_term)

        region_results[region] = electrode_results

    # Prepare data for plotting
    region_means = {region: np.mean(params) for region, params in region_results.items()}
    region_sems = {region: np.std(params) / np.sqrt(len(params)) for region, params in region_results.items()}

    # Plotting
    regions_sorted = sorted(region_results.keys())  # Sort regions alphabetically
    means = [region_means[region] for region in regions_sorted]
    sems = [region_sems[region] for region in regions_sorted]

    plt.figure(figsize=(12, 6))

    # Bar plot for means with SEM as error bars
    plt.bar(range(len(regions_sorted)), means, yerr=sems, alpha=0.7, capsize=5, label='Region Mean')

    # Add individual dots for electrode results
    for i, region in enumerate(regions_sorted):
        electrode_params = region_results[region]
        plt.scatter([i] * len(electrode_params), electrode_params, color='black', s=3,
                    label='Electrode Values' if i == 0 else "")

    # Customize plot
    plt.xticks(range(len(regions_sorted)), regions_sorted, rotation=45, ha='right')
    plt.xlabel('Electrode Region')
    plt.ylabel('Quadratic Coefficient (surprise^2)')
    plt.gca().axhline(y=0, c='black', ls='--')
    plt.title('Quadratic Coefficients by Electrode Region')
    plt.legend()
    plt.tight_layout()
    plt.show()



def save_note_evoked_gamma(trial_epochs_path, note_level_epochs, behavior_path, save_path, regions, first_parcel_label='advanced', tmin=2, tmax=7, min_time_interval=0.050, only_repeated=True, freq_bands=[120, 140]):
    """note_level epochs are used to get to individual notes, but were already filtered so cannot be used for neural data."""
    os.makedirs(save_path, exist_ok=True)
    path = Path(trial_epochs_path)
    subject_list = sorted(list(path.glob('*.fif')))
    behavior_all = pd.read_csv(behavior_path)

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)
    # mne.datasets.fetch_fsaverage(subjects_dir=None)
    subject = 'fsaverage'

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        # extract surprise and trial_time from note level epochs
        epochs = mne.read_epochs(os.path.join(note_level_epochs, f'{subject_name}_epochs_mono_preprocessed_IDA.fif'))
        metadata = epochs.metadata
        # keep only notes with min time interval
        if min_time_interval is not None:
            print('Number of notes:', len(epochs.metadata))
            epochs = epochs[metadata['time_interval'] >= min_time_interval]
            metadata = epochs.metadata
            print('Number of notes after min_time_interval:', len(epochs.metadata))
        epochs.metadata = metadata
        regressors = metadata
        # # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressors = regressors[regressors['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]
        regressors = regressors[regressors['trial_time'].to_numpy() <= tmax]
        # keep only stim that repeat
        if only_repeated:
            rows_to_keep = []
            already_done = []
            for index, row in epochs.metadata.iterrows():
                if '_rep2_' in row['stim_name']:
                    stim_name_base = row['stim_name'][14:]
                    if stim_name_base not in already_done:
                        matching_rows = epochs.metadata[epochs.metadata['stim_name'].str.contains(stim_name_base)]
                        rows_to_keep.extend(matching_rows.index)
                        already_done.append(stim_name_base)
            mask = epochs.metadata.index.isin(rows_to_keep)
            regressors = regressors[mask]
        regressors = regressors[['stim_name', 'trial_time', 'surprise_possum_paulrnn']]
        del epochs

        # epochs data
        channels_list = []
        channel_parcellations_list = []

        epochs = mne.read_epochs(epochs_path)
        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=subjects_dir,
                                                  aseg='aparc+aseg', dist=2)
        for key, value in labels.items():
            channels_list.append(key)
            channel_parcellations_list.append(value)

        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in channel_parcellations_list]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(channel_parcellations_list, go_next)

        # select electrodes
        chan_ind = []
        for p, parcel in enumerate(parcellations):
            if any(ch_parcel in regions for ch_parcel in parcel):
                chan_ind.append(p)
        print('Kept electrodes:', np.array(channels_list)[chan_ind])

        stim_name = epochs.metadata['stim_name'].to_list()
        print('Number of channels:', len(epochs.ch_names))

        trials_to_keep = [s for s in range(len(stim_name)) if stim_name[s] in regressors['stim_name'].to_list()]
        epochs = mne.read_epochs(epochs_path)
        if len(chan_ind) > 0:
            epochs.pick(chan_ind)
            epochs = epochs[trials_to_keep]
            stim_name = epochs.metadata['stim_name'].to_list()
            print(epochs.ch_names)
            epochs = epochs.apply_baseline((None, -0.1))
            fs = 400
            epochs.resample(fs, n_jobs=-1)
            epochs.filter(0.5, None, n_jobs=-1)
            start_baseline = epochs.time_as_index(-1.5)[0]
            start = epochs.time_as_index(0)[0]
            end = epochs.time_as_index(8)[0]

            epochs.filter(freq_bands[0], freq_bands[1], n_jobs=-1, verbose=0)
            gamma = epochs.get_data()
            gamma_power = np.abs(scipy.signal.hilbert(epochs.get_data(), axis=-1)) ** 2  # extract power
            baseline = np.mean(gamma_power[:, :, start_baseline:start], axis=2, keepdims=True)  # average over time
            gamma_power = (gamma_power - baseline) / baseline * 100  # instantaneous power, percent wrt baseline

            # behavior
            behavior = behavior_all[behavior_all['subject'] == subject_name]
            behavior = behavior[behavior['stim_name'].isin(regressors['stim_name'].to_list())]
            behavior['stim_identity'] = behavior['stim_name'].str[14:26]
            behavior['rep'] = behavior.apply(lambda row: float(row['stim_name'].split('_')[1][3]), axis=1)
            correct2_map = behavior[behavior['rep'] == 2].set_index(['stim_identity'])['correct'].to_dict()
            behavior['correct2'] = behavior.apply(lambda row: correct2_map.get(row['stim_identity']) if row['rep'] == 1 else row['correct'], axis=1)
            correct = behavior['correct'].to_list()
            correct2 = behavior['correct2'].to_list()

            trial_time_list = []
            surprise_list = []
            gamma_power_list = []
            gamma_list = []
            correct_list = []
            correct2_list = []
            rep_list = []
            note_rate_list = []
            onset_list = []
            # loop through epochs:
            stim_name_list = np.unique(regressors['stim_name'].to_list())
            for e, stim in enumerate(stim_name_list):
                if stim in stim_name:  # make sure that we have all info for every trial we keep
                    trial_time_list.append(regressors[regressors['stim_name']==stim]['trial_time'].to_list())
                    surprise_list.append(regressors[regressors['stim_name']==stim]['surprise_possum_paulrnn'].to_list())
                    gamma_power_list.append(gamma_power[stim_name.index(stim), :, :])
                    gamma_list.append(gamma[stim_name.index(stim), :, :])
                    correct_list.append(correct[behavior['stim_name'].to_list().index(stim)])
                    correct2_list.append(correct2[behavior['stim_name'].to_list().index(stim)])
                    rep_list.append(float(stim.split('_')[1][3]))
                    note_rate_list.append(stim.split('_')[2][:5])
                    onset_list.append(float(stim.split('_')[4][5]))

            # power_dict = {'stim_name': stim_name_list, 'gamm_power': gamma_power_list, 'ch_names': channels_list,
            #               'ch_parcellations': channel_parcellations_list, 'info': epochs.info}
            power_dict = {'stim_name': stim_name_list, 'gamma_power': gamma_power_list, 'gamma': gamma_list, 'trial_time': trial_time_list, 'surprise': surprise_list, 'correct': correct_list, 'correct2': correct2_list, 'rep': rep_list, 'note_rate': note_rate_list, 'onset': onset_list, 'times': epochs.times}
            with open(os.path.join(save_path, subject_name + "_note_evoked_power.pkl"), 'wb') as f:
                pickle.dump(power_dict, f)

    print('Done')


def note_evoked_gamma_regression(save_path):
    path = Path(save_path)
    subject_paths = sorted(list(path.glob(f'*note_evoked_power.pkl')))
    r2_all_subjects = []
    r2_linear_all_subjects = []
    r2_control_all_subjects = []
    linear_all_subjects = []
    linear_only_all_subjects = []
    quadratic_all_subjects = []
    all_subject_nulls = []
    evoked_gamma_power_all_subjects = []
    n_permutations = 100  # Number of permutations
    for i, subject_path in enumerate(tqdm(subject_paths)):
        note_evoked_obj = np.load(subject_path, allow_pickle=True)
        times = note_evoked_obj['times']
        trial_time = note_evoked_obj['trial_time']
        surprise = note_evoked_obj['surprise']
        note_rate = np.array(note_evoked_obj['note_rate'])
        rep = note_evoked_obj['rep']
        onset = note_evoked_obj['onset']
        gamma_power = np.array(note_evoked_obj['gamma_power'])
        gamma = np.array(note_evoked_obj['gamma'])
        # plt.plot(times, np.mean(gamma_power[0], axis=0))
        # plt.figure()
        # plt.plot(times, np.mean(gamma[0], axis=0))
        # plt.show()
        # for rate in ["2.5Hz", "3.5Hz", "4.5Hz", "5.5Hz", "6.5Hz", "7.5Hz"]:
        #     rate_stims = np.where(note_rate==rate)[0]
        #     print(rate_stims)
        #     rate_power = np.mean(gamma[rate_stims, :, :], axis=1)  # average electrodes
        #     # rate_power = np.mean(rate_power, axis=0)
        #     rate_power = np.mean(rate_power, axis=0) / (np.std(rate_power, axis=0) / np.sqrt(rate_power.shape[0]))
        #     plt.figure()
        #     plt.plot(times, rate_power, label=rate)
        #     plt.legend()
        # plt.show()

        gamma_power = np.mean(gamma_power, axis=1)
        surprise_full = []
        rep_full = []
        onset_full = []
        note_rate_full = []
        evoked_gamma_power = []
        n_samples_before = np.argmin(np.abs(times - 0.2)) - np.argmin(np.abs(times))
        n_samples_after = np.argmin(np.abs(times - 0.5)) - np.argmin(np.abs(times))
        for epoch in range(len(trial_time)):
            # if rep[epoch] == 2:
            surprise_full += surprise[epoch]
            for t in trial_time[epoch]:
                rep_full.append(rep[epoch] - 1)  # to get rep between 0 and 1
                onset_full.append(onset[epoch])
                note_rate_full.append(float(note_rate[epoch][:3]))
                start = np.argmin(np.abs(times - t)) - n_samples_before
                end = np.argmin(np.abs(times - t)) + n_samples_after
                # evoked_gamma_power.append(gamma_power[epoch, start:end])
                # evoked_gamma_power.append(gamma_power[epoch, :, start:end] - np.mean(gamma_power[epoch, :, start:np.argmin(np.abs(times - t))]))  # with baseline
                evoked_gamma_power.append(gamma_power[epoch, start:end] - np.mean(gamma_power[epoch, start:np.argmin(np.abs(times - t))]))  # with baseline
        evoked_gamma_power_all_subjects.append(np.mean(evoked_gamma_power, axis=0))
        print(np.mean(evoked_gamma_power, axis=0).shape)
        X = np.array(surprise_full)
        y = np.array(evoked_gamma_power)
        print(y.shape)

        X_squared = X ** 2
        X = (X - np.mean(X)) / np.std(X)
        X_squared = (X_squared - np.mean(X_squared)) / np.std(X_squared)
        # Combine X and X_squared into a single feature matrix
        # X_linear = X.reshape(-1, 1)
        X_linear = np.column_stack((X, rep_full, onset_full, note_rate_full))  # Shape becomes (5576, 2)
        X_linear = pd.DataFrame(X_linear, columns=["X", "rep", "onset", "note_rate"])
        X_linear = pd.get_dummies(X_linear, columns=["rep", "onset", "note_rate"], drop_first=True)
        # X_combined = np.column_stack((X, X_squared))  # Shape becomes (5576, 2)
        X_combined = np.column_stack((X, X_squared, rep_full, onset_full, note_rate_full))  # Shape becomes (5576, 2)
        X_combined = pd.DataFrame(X_combined, columns=["X", "X_squared", "rep", "onset", "note_rate"])
        X_combined = pd.get_dummies(X_combined, columns=["rep", "onset", "note_rate"], drop_first=True, dtype=float)
        X_control = np.column_stack((rep_full, onset_full, note_rate_full))  # Shape becomes (5576, 2)
        X_control = pd.DataFrame(X_control, columns=["rep", "onset", "note_rate"])
        X_control = pd.get_dummies(X_control, columns=["rep", "onset", "note_rate"], drop_first=True)

        # for c in range(gamma_power.shape[1]):
        # y = np.array(np.array(evoked_gamma_power)[:, c, :])
        # print(y.shape)

        # Initialize an empty list to store R^2 values
        r2_values = []
        r2_linear_values = []
        r2_control_values = []
        null_distribution = np.zeros((n_permutations, y.shape[1]))
        linear_values = []
        linear_only_values = []
        quadratic_values = []

        # Loop through each of the 600 time points
        for t in tqdm(range(y.shape[1])):  # y.shape[1] is 600
            # Extract target values for the current time point
            y_time = y[:, t]

            # Fit linear regression model
            # model = LinearRegression()
            model = RidgeCV(alphas=[10000], scoring=None, cv=5)
            model.fit(X_combined, y_time)
            linear_values.append(model.coef_[0])
            quadratic_values.append(model.coef_[1])

            # Predict and calculate R^2
            # y_pred = model.predict(X_combined)
            # r2 = r2_score(y_time, y_pred)
            # r2_values.append(r2)
            # print(model.best_score_)
            # print(np.mean(model.best_score_))
            # print(np.mean(model.cv_values_))
            r2_values.append(np.mean(model.best_score_))

            # Only linear model
            # model = LinearRegression()
            model = RidgeCV(alphas=[10000], scoring=None, cv=5)
            model.fit(X_linear, y_time)
            linear_only_values.append(model.coef_[0])
            # y_pred = model.predict(X_linear)
            # r2 = r2_score(y_time, y_pred)
            # r2_linear_values.append(r2)
            r2_linear_values.append(np.mean(model.best_score_))

            # control model
            # model = LinearRegression()
            model = RidgeCV(alphas=[10000], scoring=None, cv=5)
            model.fit(X_control, y_time)
            # y_pred = model.predict(X_control)
            # r2 = r2_score(y_time, y_pred)
            # r2_control_values.append(r2)
            r2_control_values.append(np.mean(model.best_score_))

            # Permutation loop
            # for p in range(n_permutations):
            #     y_time_permuted = np.random.permutation(y_time)
            #     model.fit(X_combined, y_time_permuted)
            #     y_pred_perm = model.predict(X_combined)
            #     null_distribution[p, t] = r2_score(y_time_permuted, y_pred_perm)

        # Convert R^2 values to a numpy array for easier handling
        r2_values = np.array(r2_values)
        r2_all_subjects.append(r2_values)
        r2_linear_all_subjects.append(np.array(r2_linear_values))
        r2_control_all_subjects.append(np.array(r2_control_values))
        all_subject_nulls.append(null_distribution)
        linear_all_subjects.append(np.array(linear_values))
        linear_only_all_subjects.append(np.array(linear_only_values))
        quadratic_all_subjects.append(np.array(quadratic_values))

        # Plot R^2 values over time
        evoked_times = np.linspace(-0.2, 0.5, r2_values.shape[0])
        # plt.figure(figsize=(10, 5))
        # plt.plot(evoked_times, r2_values, label='Quadratic model', c='#D55E00', lw=2)
        # plt.plot(evoked_times, r2_linear_values, label='Linear model', c='#0072B2', lw=2)
        # plt.plot(evoked_times, r2_control_values, label='Control model', c='gray', lw=2)
        # plt.axvline(x=0, c='black', linestyle='--')
        # plt.xlabel('Time (s)')
        # plt.ylabel('R^2')
        # plt.title('R^2 Values of Linear Regression Models Over Time (Quadratic Term Included)')
        # plt.legend()
        # plt.show()

    r2_all_subjects = np.array(r2_all_subjects)
    all_subject_nulls = np.array(all_subject_nulls)
    group_null = np.mean(all_subject_nulls, axis=0)
    group_r2 = np.mean(r2_all_subjects, axis=0)

    linear_all_subjects = np.array(linear_all_subjects)
    linear_only_all_subjects = np.array(linear_only_all_subjects)
    quadratic_all_subjects = np.array(quadratic_all_subjects)
    r2_linear_all_subjects = np.array(r2_linear_all_subjects)
    r2_control_all_subjects = np.array(r2_control_all_subjects)

    p_values = np.mean(group_null >= group_r2, axis=0)

    # t_values, p_values = ttest_1samp(r2_all_subjects, popmean=0, axis=0)
    # significant, p_corrected = fdrcorrection(p_values, alpha=0.05)
    # print(p_corrected)
    significant = p_values < 0.05

    sem_r2_all_subjects = np.std(r2_all_subjects, axis=0) / np.sqrt(r2_all_subjects.shape[0])
    sem_r2_linear_all_subjects = np.std(r2_linear_all_subjects, axis=0) / np.sqrt(r2_linear_all_subjects.shape[0])
    sem_r2_control_all_subjects = np.std(r2_control_all_subjects, axis=0) / np.sqrt(r2_control_all_subjects.shape[0])
    r2_all_subjects = np.mean(r2_all_subjects, axis=0)
    r2_linear_all_subjects = np.mean(r2_linear_all_subjects, axis=0)
    r2_control_all_subjects = np.mean(r2_control_all_subjects, axis=0)
    # r2_all_subjects = np.mean(r2_all_subjects, axis=0) / (np.std(r2_all_subjects, axis=0) / np.sqrt(r2_all_subjects.shape[0]))
    # plt.plot(evoked_times, r2_all_subjects - r2_control_all_subjects, label='Quadratic model', c='#D55E00', lw=2)
    # plt.plot(evoked_times, r2_linear_all_subjects - r2_control_all_subjects, label='Linear model', c='#0072B2', lw=2)
    plt.plot(evoked_times, r2_all_subjects / sem_r2_all_subjects, label='Quadratic model', c='#D55E00', lw=2)
    plt.plot(evoked_times, r2_linear_all_subjects / sem_r2_linear_all_subjects, label='Linear model', c='#0072B2', lw=2)
    plt.plot(evoked_times, r2_control_all_subjects / sem_r2_control_all_subjects, label='Control model', c='gray', lw=2)
    plt.axvline(x=0, c='black', linestyle='--')
    plt.axhline(y=0, c='black', linestyle='--')
    plt.xlim(evoked_times[0], evoked_times[-1])
    # plt.ylim(0, plt.ylim()[1])
    # plt.scatter(evoked_times[significant], r2_all_subjects[significant], color='red', label='Significant time points')
    ymin, ymax = plt.ylim()
    bar_height = ymin - 0.005 * (ymax - ymin)
    bar_width = evoked_times[1] - evoked_times[0]
    # for t in evoked_times[significant]:
    #     plt.plot([t - bar_width / 2, t + bar_width / 2], [bar_height, bar_height], color='black', lw=5)
    plt.xlabel('Time (s)')
    plt.ylabel('R² tval')
    plt.legend()

    plt.figure()
    plt.plot(evoked_times, r2_all_subjects, label='Quadratic model', c='#D55E00', lw=2)
    plt.plot(evoked_times, r2_linear_all_subjects, label='Linear model', c='#0072B2', lw=2)
    plt.plot(evoked_times, r2_control_all_subjects, label='Control model', c='gray', lw=2)
    plt.axvline(x=0, c='black', linestyle='--')
    plt.axhline(y=0, c='black', linestyle='--')
    plt.xlim(evoked_times[0], evoked_times[-1])
    # plt.ylim(0, plt.ylim()[1])
    # plt.scatter(evoked_times[significant], r2_all_subjects[significant], color='red', label='Significant time points')
    ymin, ymax = plt.ylim()
    bar_height = ymin - 0.005 * (ymax - ymin)
    bar_width = evoked_times[1] - evoked_times[0]
    # for t in evoked_times[significant]:
    #     plt.plot([t - bar_width / 2, t + bar_width / 2], [bar_height, bar_height], color='black', lw=5)
    plt.xlabel('Time (s)')
    plt.ylabel('$R²')
    # plt.ylabel('$\Delta$R² (compared to control model)')
    plt.legend()

    plt.figure()
    plt.plot(evoked_times, np.mean(linear_all_subjects, axis=0), label='Beta linear')
    # plt.scatter(evoked_times[significant], r2_all_subjects[significant], color='red', label='Significant time points')
    plt.xlim(evoked_times[0], evoked_times[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Surprise beta (linear term)')
    plt.legend()

    plt.figure()
    plt.plot(evoked_times, np.mean(quadratic_all_subjects, axis=0), label='Beta quadratic')
    # plt.scatter(evoked_times[significant], r2_all_subjects[significant], color='red', label='Significant time points')
    plt.xlim(evoked_times[0], evoked_times[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Surprise beta (quadratic term)')
    plt.legend()

    plt.figure()
    plt.plot(evoked_times, np.mean(linear_only_all_subjects, axis=0), label='Beta linear (only linear model)')
    # plt.scatter(evoked_times[significant], r2_all_subjects[significant], color='red', label='Significant time points')
    plt.xlim(evoked_times[0], evoked_times[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Surprise beta (linear term)')
    plt.legend()

    evoked_gamma_power_all_subjects = np.array(evoked_gamma_power_all_subjects)
    mean = np.mean(evoked_gamma_power_all_subjects, axis=0)
    sem = np.std(evoked_gamma_power_all_subjects, axis=0) / np.sqrt(evoked_gamma_power_all_subjects.shape[0])
    print(evoked_gamma_power_all_subjects.shape)
    plt.figure()
    plt.plot(evoked_times, mean, label='Evoked High Gamma', lw=3)
    plt.fill_between(evoked_times, mean-sem, mean+sem, alpha=0.5, lw=0)
    # plt.scatter(evoked_times[significant], r2_all_subjects[significant], color='red', label='Significant time points')
    plt.axvline(x=0, c='black', linestyle='--')
    plt.xlim(evoked_times[0], evoked_times[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Power (percent change relative to baseline)')
    plt.legend()

    plt.show()


    # logistic regression for correct
    r2_all_subjects = []  # r2 is actually AUC now
    linear_all_subjects = []
    all_subject_nulls = []
    n_permutations = 200  # Number of permutations
    for i, subject_path in enumerate(tqdm(subject_paths)):
        note_evoked_obj = np.load(subject_path, allow_pickle=True)
        times = note_evoked_obj['times']
        trial_time = note_evoked_obj['trial_time']
        correct = note_evoked_obj['correct']
        correct2 = note_evoked_obj['correct2']
        note_rate = np.array(note_evoked_obj['note_rate'])
        rep = note_evoked_obj['rep']
        gamma_power = np.array(note_evoked_obj['gamma_power'])
        gamma = np.array(note_evoked_obj['gamma'])

        gamma_power = np.mean(gamma_power, axis=1)  # average channels
        correct2_full = []
        evoked_gamma_power = []
        n_samples_before = np.argmin(np.abs(times - 0.5)) - np.argmin(np.abs(times))
        n_samples_after = np.argmin(np.abs(times - 1)) - np.argmin(np.abs(times))
        for epoch in range(len(trial_time)):
            if (rep[epoch] == 1) & (not np.isnan(correct2[epoch])):
            # if (rep[epoch] == 1):
                for t in trial_time[epoch]:
                    correct2_full.append(correct2[epoch])
                    start = np.argmin(np.abs(times - t)) - n_samples_before
                    end = np.argmin(np.abs(times - t)) + n_samples_after
                    evoked_gamma_power.append(gamma_power[epoch, start:end])
        y = np.array(correct2_full)
        X = np.array(evoked_gamma_power)

        # Combine X and X_squared into a single feature matrix
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Initialize an empty list to store R^2 values
        r2_values = []
        r2_linear_values = []
        null_distribution = np.zeros((n_permutations, X.shape[1]))
        linear_values = []

        # Loop through each of the 600 time points
        for t in tqdm(range(X.shape[1])):  # y.shape[1] is 600
            # Extract target values for the current time point
            X_time = X[:, t]

            # Fit linear regression model
            model = LogisticRegression()
            model.fit(X_time, y)
            linear_values.append(model.coef_[0])

            # Predict and calculate R^2
            prob_pred = model.predict_proba(X_time)[:, 1]
            auc = roc_auc_score(y, prob_pred)
            r2_values.append(auc)

            # Permutation loop
            # for p in range(n_permutations):
            #     y_time_permuted = np.random.permutation(y_time)
            #     model.fit(X_combined, y_time_permuted)
            #     y_pred_perm = model.predict(X_combined)
            #     null_distribution[p, t] = r2_score(y_time_permuted, y_pred_perm)

        # Convert R^2 values to a numpy array for easier handling
        r2_values = np.array(r2_values)
        r2_all_subjects.append(r2_values)
        all_subject_nulls.append(null_distribution)
        linear_all_subjects.append(np.array(linear_values))

        # Plot R^2 values over time
        evoked_times = np.linspace(-0.5, 1, r2_values.shape[0])
        # plt.figure(figsize=(10, 5))
        # plt.plot(evoked_times, r2_values, label='R^2 over time (with quadratic term)')
        # plt.xlabel('Time (s)')
        # plt.ylabel('R^2')
        # plt.title('R^2 Values of Linear Regression Models Over Time (Quadratic Term Included)')
        # plt.legend()
        # plt.show()

    r2_all_subjects = np.array(r2_all_subjects)
    all_subject_nulls = np.array(all_subject_nulls)
    group_null = np.mean(all_subject_nulls, axis=0)
    group_r2 = np.mean(r2_all_subjects, axis=0)

    linear_all_subjects = np.array(linear_all_subjects)

    p_values = np.mean(group_null >= group_r2, axis=0)

    # t_values, p_values = ttest_1samp(r2_all_subjects, popmean=0, axis=0)
    # significant, p_corrected = fdrcorrection(p_values, alpha=0.05)
    # print(p_corrected)
    # significant = p_values < 0.05

    r2_all_subjects = np.mean(r2_all_subjects, axis=0)
    # r2_all_subjects = np.mean(r2_all_subjects, axis=0) / (np.std(r2_all_subjects, axis=0) / np.sqrt(r2_all_subjects.shape[0]))
    plt.plot(evoked_times, r2_all_subjects, label='AUC')
    # plt.scatter(evoked_times[significant], r2_all_subjects[significant], color='red', label='Significant time points')
    plt.xlabel('Time (s)')
    plt.ylabel('AUC')
    plt.legend()

    plt.figure()
    plt.plot(evoked_times, np.mean(linear_all_subjects, axis=0), label='Beta linear')
    # plt.scatter(evoked_times[significant], r2_all_subjects[significant], color='red', label='Significant time points')
    plt.xlabel('Time (s)')
    plt.ylabel('Surprise beta (linear term)')
    plt.legend()

    plt.show()


def note_evoked_gamma_analysis(save_path):
    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']
    # surprise_quantiles = [0, 0.33, 0.66, 1]
    surprise_quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # surprise_quantiles = [i / 6 for i in range(7)]
    # surprise_quantiles = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # surprise_quantiles = [i/20 for i in range(21)]
    # surprise_quantiles = [i/7 for i in range(8)]
    path = Path(save_path)
    subject_paths = sorted(list(path.glob(f'*note_evoked_power.pkl')))
    evoked_gamma_power_all_subjects = []
    evoked_gamma_power_all_subjects_by_rates = [[] for _ in range(len(rates))]
    evoked_gamma_power_all_subjects_by_surprise_quantile = [[] for _ in range(len(surprise_quantiles)-1)]
    evoked_gamma_power_all_subjects_by_surprise_quantile_rate = [[[] for q in range(len(surprise_quantiles)-1)] for r in range(len(rates))]
    for i, subject_path in enumerate(tqdm(subject_paths)):
        note_evoked_obj = np.load(subject_path, allow_pickle=True)
        times = note_evoked_obj['times']
        trial_time = note_evoked_obj['trial_time']
        surprise = note_evoked_obj['surprise']
        note_rate = np.array(note_evoked_obj['note_rate'])
        rep = note_evoked_obj['rep']
        onset = note_evoked_obj['onset']
        gamma_power = np.array(note_evoked_obj['gamma_power'])
        gamma = np.array(note_evoked_obj['gamma'])

        gamma_power = np.mean(gamma_power, axis=1)  # average channels
        surprise_full = []
        rep_full = []
        onset_full = []
        note_rate_full = []
        evoked_gamma_power = []
        n_samples_before = np.argmin(np.abs(times - 0.2)) - np.argmin(np.abs(times))
        n_samples_after = np.argmin(np.abs(times - 0.5)) - np.argmin(np.abs(times))
        evoked_times = np.linspace(-0.2, 0.5, n_samples_after + n_samples_before)
        for epoch in range(len(trial_time)):
            if rep[epoch] == 1:
                surprise_full += surprise[epoch]
                for t in trial_time[epoch]:
                    rep_full.append(rep[epoch] - 1)  # to get rep between 0 and 1
                    onset_full.append(onset[epoch])
                    note_rate_full.append(note_rate[epoch])
                    start = np.argmin(np.abs(times - t)) - n_samples_before
                    end = np.argmin(np.abs(times - t)) + n_samples_after
                    # evoked_gamma_power.append(gamma_power[epoch, start:end])
                    evoked_gamma_power.append((gamma_power[epoch, start:end] - np.mean(gamma_power[epoch, start:end])) / np.std(gamma_power[epoch, start:end]))  # normalized
                    # evoked_gamma_power.append(gamma_power[epoch, start:end] - np.mean(gamma_power[epoch, start:np.argmin(np.abs(times - t))]))
        print(np.array(evoked_gamma_power).shape)
        evoked_gamma_power = np.array(evoked_gamma_power)
        evoked_gamma_power_all_subjects.append(np.mean(evoked_gamma_power, axis=0))

        for r, rate in enumerate(rates):
            rate_notes_indices = [e for e in range(len(note_rate_full)) if note_rate_full[e] == rate]
            evoked_gamma_power_all_subjects_by_rates[r].append(np.mean(evoked_gamma_power[rate_notes_indices, :], axis=0))

        for q in range(len(surprise_quantiles) - 1):
            lower_value = np.quantile(surprise_full, surprise_quantiles[q])
            higher_value = np.quantile(surprise_full, surprise_quantiles[q+1])
            quantile_indices = [e for e in range(len(surprise_full)) if (surprise_full[e] >= lower_value) and (surprise_full[e] <= higher_value)]
            evoked_gamma_power_all_subjects_by_surprise_quantile[q].append(np.mean(evoked_gamma_power[quantile_indices, :], axis=0))

        for r, rate in enumerate(rates):
            rate_notes_indices = [e for e in range(len(note_rate_full)) if note_rate_full[e] == rate]
            for q in range(len(surprise_quantiles) - 1):
                lower_value = np.quantile(np.array(surprise_full), surprise_quantiles[q])
                higher_value = np.quantile(np.array(surprise_full), surprise_quantiles[q + 1])
                quantile_indices = [e for e in range(len(surprise_full)) if (surprise_full[e] >= lower_value) and (surprise_full[e] <= higher_value) and (e in rate_notes_indices)]
                evoked_gamma_power_all_subjects_by_surprise_quantile_rate[r][q].append(np.mean(evoked_gamma_power[quantile_indices, :], axis=0))

        # for r, rate in enumerate(['2.5Hz']):
        #     rate_notes_indices = [e for e in range(len(note_rate_full)) if note_rate_full[e] == rate]
        #     for q in range(len(surprise_quantiles) - 1):
        #         lower_value = np.quantile(np.array(surprise_full), surprise_quantiles[q])
        #         higher_value = np.quantile(np.array(surprise_full), surprise_quantiles[q + 1])
        #         print(higher_value)
        # for r, rate in enumerate(['3.5Hz']):
        #     rate_notes_indices = [e for e in range(len(note_rate_full)) if note_rate_full[e] == rate]
        #     for q in range(len(surprise_quantiles) - 1):
        #         lower_value = np.quantile(np.array(surprise_full), surprise_quantiles[q])
        #         higher_value = np.quantile(np.array(surprise_full), surprise_quantiles[q + 1])
        #         print(higher_value)

        # Plot R^2 values over time
        # plt.figure(figsize=(10, 5))
        # plt.plot(evoked_times, np.mean(evoked_gamma_power, axis=0), lw=2)
        # plt.axvline(x=0, c='black', linestyle='--')
        # plt.axhline(y=0, c='black', linestyle='--')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Power (percent change relative to baseline)')
        #
        # plt.title('Evoked gamma')
        # plt.show()

    evoked_gamma_power_all_subjects = np.array(evoked_gamma_power_all_subjects)
    mean = np.mean(evoked_gamma_power_all_subjects, axis=0)
    sem = np.std(evoked_gamma_power_all_subjects, axis=0) / np.sqrt(evoked_gamma_power_all_subjects.shape[0])
    print(evoked_gamma_power_all_subjects.shape)
    plt.figure()
    plt.plot(evoked_times, mean, label='Evoked High Gamma', lw=3)
    plt.fill_between(evoked_times, mean-sem, mean+sem, alpha=0.5, lw=0)
    # plt.scatter(evoked_times[significant], r2_all_subjects[significant], color='red', label='Significant time points')
    plt.axvline(x=0, c='black', linestyle='--')
    plt.axhline(y=0, c='black', linestyle='--')
    plt.xlim(evoked_times[0], evoked_times[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Power (percent change relative to baseline)')
    plt.legend()

    tval = mean / sem
    print(evoked_gamma_power_all_subjects.shape)
    plt.figure()
    plt.plot(evoked_times, tval, label='Evoked High Gamma', lw=3)
    # plt.scatter(evoked_times[significant], r2_all_subjects[significant], color='red', label='Significant time points')
    plt.axvline(x=0, c='black', linestyle='--')
    plt.axhline(y=0, c='black', linestyle='--')
    plt.xlim(evoked_times[0], evoked_times[-1])
    plt.xlabel('Time (s)')
    plt.ylabel('Tval')
    plt.legend()

    fig, axes = plt.subplots(2, len(rates), sharex=True, sharey='row')
    for r, rate in enumerate(rates):
        evoked_gamma_rate = np.array(evoked_gamma_power_all_subjects_by_rates[r])
        # evoked_gamma_rate = evoked_gamma_rate - np.mean(evoked_gamma_rate)
        mean = np.mean(evoked_gamma_rate, axis=0)
        sem = np.std(evoked_gamma_rate, axis=0) / np.sqrt(evoked_gamma_rate.shape[0])
        axes[0, r].plot(evoked_times, mean, label=rate, lw=3)
        axes[0, r].fill_between(evoked_times, mean - sem, mean + sem, alpha=0.5, lw=0)
        axes[0, r].axvline(x=0, c='black', linestyle='--')
        axes[0, r].axhline(y=0, c='black', linestyle='--')
        axes[0, r].set_xlim(evoked_times[0], evoked_times[-1])
        axes[0, r].set_xlabel('Time (s)')
        if r==0:
            axes[0, r].set_ylabel('Power (percent change relative to baseline)')
        axes[0, r].legend()

        tval = mean / sem
        axes[1, r].plot(evoked_times, tval, label=rate, lw=3)
        axes[1, r].axvline(x=0, c='black', linestyle='--')
        axes[1, r].axhline(y=0, c='black', linestyle='--')
        axes[1, r].set_xlim(evoked_times[0], evoked_times[-1])
        axes[1, r].set_xlabel('Time (s)')
        if r==0:
            axes[1, r].set_ylabel('Tval')
        axes[1, r].legend()

    # by surprise quantile
    fig, axes = plt.subplots(2, 1, sharex=True, sharey='row')
    # cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#ff5a5f', '#6859D9'])
    cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#6859D9', '#ff5a5f'])
    color = [cmap1(i) for i in np.linspace(0, 1, len(surprise_quantiles) - 1)]
    for q in range(len(surprise_quantiles) - 1):
        evoked_gamma_rate = np.array(evoked_gamma_power_all_subjects_by_surprise_quantile[q])
        # evoked_gamma_rate = evoked_gamma_rate - np.mean(evoked_gamma_rate)
        mean = np.mean(evoked_gamma_rate, axis=0)
        sem = np.std(evoked_gamma_rate, axis=0) / np.sqrt(evoked_gamma_rate.shape[0])
        axes[0].plot(evoked_times, mean, label=surprise_quantiles[q+1], c=color[q], lw=2)
        # axes[0].fill_between(evoked_times, mean - sem, mean + sem, alpha=0.5, lw=0)

        tval = mean / sem
        axes[1].plot(evoked_times, tval, label=surprise_quantiles[q+1], c=color[q], lw=2)
    axes[0].axvline(x=0, c='black', linestyle='--')
    axes[0].axhline(y=0, c='black', linestyle='--')
    axes[0].set_xlim(evoked_times[0], evoked_times[-1])
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Power (percent change relative to baseline)')
    axes[0].legend()
    axes[1].axvline(x=0, c='black', linestyle='--')
    axes[1].axhline(y=0, c='black', linestyle='--')
    axes[1].set_xlim(evoked_times[0], evoked_times[-1])
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Tval')
    axes[1].legend()

    # by rate and quantile
    fig, axes = plt.subplots(2, len(rates), sharex=True, sharey='row')
    for r, rate in enumerate(rates):
        evoked_gamma_rate = np.array(evoked_gamma_power_all_subjects_by_surprise_quantile_rate[r])
        for q in range(len(surprise_quantiles) - 1):
            evoked_gamma_quantile = np.array(evoked_gamma_rate[q])
            # evoked_gamma_quantile = evoked_gamma_quantile - np.mean(evoked_gamma_quantile)
            mean = np.mean(evoked_gamma_quantile, axis=0)
            sem = np.std(evoked_gamma_quantile, axis=0) / np.sqrt(evoked_gamma_quantile.shape[0])
            axes[0, r].plot(evoked_times, mean, label=surprise_quantiles[q + 1], c=color[q], lw=2)
            # axes[0].fill_between(evoked_times, mean - sem, mean + sem, alpha=0.5, lw=0)

            tval = mean / sem
            axes[1, r].plot(evoked_times, tval, label=surprise_quantiles[q + 1], c=color[q], lw=2)
        axes[0, r].axvline(x=0, c='black', linestyle='--')
        axes[0, r].axhline(y=0, c='black', linestyle='--')
        axes[0, r].set_xlim(evoked_times[0], evoked_times[-1])
        axes[0, r].set_xlabel('Time (s)')
        if r == 0:
            axes[0, r].set_ylabel('Power (percent change relative to baseline)')
            # axes[0, r].legend()
        axes[0, r].set_title(rate)
        axes[1, r].axvline(x=0, c='black', linestyle='--')
        axes[1, r].axhline(y=0, c='black', linestyle='--')
        axes[1, r].set_xlim(evoked_times[0], evoked_times[-1])
        axes[1, r].set_xlabel('Time (s)')
        if r == 0:
            axes[1, r].set_ylabel('Tval')
            # axes[1, r].legend()

    plt.show()


def note_evoked_gamma_analysis_phase(save_path):
    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']
    # surprise_quantiles = [0, 0.5, 1]
    # surprise_quantiles = [0, 0.33, 0.66, 1]
    surprise_quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # surprise_quantiles = [i / 6 for i in range(7)]
    # surprise_quantiles = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # surprise_quantiles = [i/20 for i in range(21)]
    # surprise_quantiles = [i/7 for i in range(8)]
    path = Path(save_path)
    subject_paths = sorted(list(path.glob(f'*note_evoked_power.pkl')))
    evoked_gamma_power_all_subjects = []
    evoked_gamma_power_all_subjects_by_rates = [[] for _ in range(len(rates))]
    evoked_gamma_power_all_subjects_by_surprise_quantile = [[] for _ in range(len(surprise_quantiles)-1)]
    evoked_gamma_power_all_subjects_by_surprise_quantile_rate = [[[] for q in range(len(surprise_quantiles)-1)] for r in range(len(rates))]
    n_points = 100  # Number of points in the common phase grid
    common_phase_grid = np.linspace(-2 * np.pi, 2 * np.pi, n_points)
    for i, subject_path in enumerate(tqdm(subject_paths)):
        note_evoked_obj = np.load(subject_path, allow_pickle=True)
        times = note_evoked_obj['times']
        trial_time = note_evoked_obj['trial_time']
        surprise = note_evoked_obj['surprise']
        note_rate = np.array(note_evoked_obj['note_rate'])
        rep = note_evoked_obj['rep']
        onset = note_evoked_obj['onset']
        gamma_power = np.array(note_evoked_obj['gamma_power'])
        gamma = np.array(note_evoked_obj['gamma'])

        gamma_power = np.mean(gamma_power, axis=1)  # average channels
        surprise_full = []
        rep_full = []
        onset_full = []
        note_rate_full = []
        evoked_gamma_power = []
        for epoch in range(len(trial_time)):
            if rep[epoch] == 1:
                surprise_full += surprise[epoch]
                for t in trial_time[epoch]:
                    rep_full.append(rep[epoch] - 1)  # to get rep between 0 and 1
                    onset_full.append(onset[epoch])
                    note_rate_full.append(note_rate[epoch])
                    rate = float(note_rate[epoch][:3])
                    n_samples_before = np.argmin(np.abs(times - 1 / rate)) - np.argmin(np.abs(times))
                    start = np.argmin(np.abs(times - t)) - n_samples_before
                    end = np.argmin(np.abs(times - t)) + n_samples_before
                    # evoked_gamma_power.append(gamma_power[epoch, start:end])
                    # evoked_gamma_power.append(gamma_power[epoch, start:end] - np.mean(gamma_power[epoch, start:np.argmin(np.abs(times - t))]))
                    # interpolator = interp1d(np.linspace(-2*np.pi,2*np.pi, end-start), gamma_power[epoch, start:end], kind='linear', bounds_error=False, fill_value=0)
                    interpolator = interp1d(np.linspace(-2*np.pi,2*np.pi, end-start), (gamma_power[epoch, start:end] - np.mean(gamma_power[epoch, start:end])) / np.std(gamma_power[epoch, start:end]), kind='linear', bounds_error=False, fill_value=0)  # with power normalization
                    # interpolator = interp1d(np.linspace(-2*np.pi,2*np.pi, end-start), gamma_power[epoch, start:end] - np.mean(gamma_power[epoch, start:np.argmin(np.abs(times - t))]), kind='linear', bounds_error=False, fill_value=0)
                    evoked_gamma_power.append(interpolator(common_phase_grid))

        evoked_gamma_power = np.array(evoked_gamma_power)
        evoked_gamma_power_all_subjects.append(np.mean(evoked_gamma_power, axis=0))

        for r, rate in enumerate(rates):
            rate_notes_indices = [e for e in range(len(note_rate_full)) if note_rate_full[e] == rate]
            evoked_gamma_power_rate = np.array([evoked_gamma_power[e] for e in rate_notes_indices])  # also works without common phase grid
            evoked_gamma_power_all_subjects_by_rates[r].append(np.mean(evoked_gamma_power_rate, axis=0))

        for q in range(len(surprise_quantiles) - 1):
            lower_value = np.quantile(surprise_full, surprise_quantiles[q])
            higher_value = np.quantile(surprise_full, surprise_quantiles[q+1])
            quantile_indices = [e for e in range(len(surprise_full)) if (surprise_full[e] >= lower_value) and (surprise_full[e] <= higher_value)]
            evoked_gamma_power_all_subjects_by_surprise_quantile[q].append(np.mean(evoked_gamma_power[quantile_indices, :], axis=0))

        for r, rate in enumerate(rates):
            rate_notes_indices = [e for e in range(len(note_rate_full)) if note_rate_full[e] == rate]
            for q in range(len(surprise_quantiles) - 1):
                lower_value = np.quantile(np.array(surprise_full), surprise_quantiles[q])
                higher_value = np.quantile(np.array(surprise_full), surprise_quantiles[q + 1])
                quantile_indices = [e for e in range(len(surprise_full)) if (surprise_full[e] >= lower_value) and (surprise_full[e] <= higher_value) and (e in rate_notes_indices)]
                print(len(quantile_indices))
                evoked_gamma_power_all_subjects_by_surprise_quantile_rate[r][q].append(np.mean(evoked_gamma_power[quantile_indices, :], axis=0))

    evoked_times = common_phase_grid

    evoked_gamma_power_all_subjects = np.array(evoked_gamma_power_all_subjects)
    mean = np.mean(evoked_gamma_power_all_subjects, axis=0)
    sem = np.std(evoked_gamma_power_all_subjects, axis=0) / np.sqrt(evoked_gamma_power_all_subjects.shape[0])
    print(evoked_gamma_power_all_subjects.shape)
    plt.figure()
    plt.plot(evoked_times, mean, label='Evoked High Gamma', lw=3)
    plt.fill_between(evoked_times, mean-sem, mean+sem, alpha=0.5, lw=0)
    # plt.scatter(evoked_times[significant], r2_all_subjects[significant], color='red', label='Significant time points')
    plt.axvline(x=0, c='black', linestyle='--')
    plt.axhline(y=0, c='black', linestyle='--')
    plt.xlim(evoked_times[0], evoked_times[-1])
    plt.xlabel('Phase (rad)')
    plt.ylabel('Power (percent change relative to baseline)')
    plt.legend()

    tval = mean / sem
    print(evoked_gamma_power_all_subjects.shape)
    plt.figure()
    plt.plot(evoked_times, tval, label='Evoked High Gamma', lw=3)
    # plt.scatter(evoked_times[significant], r2_all_subjects[significant], color='red', label='Significant time points')
    plt.axvline(x=0, c='black', linestyle='--')
    plt.axhline(y=0, c='black', linestyle='--')
    plt.xlim(evoked_times[0], evoked_times[-1])
    plt.xlabel('Phase (rad)')
    plt.ylabel('Tval')
    plt.legend()

    # by rate
    fig, axes = plt.subplots(2, len(rates), sharex=True, sharey='row')
    for r, rate in enumerate(rates):
        evoked_gamma_rate = np.array(evoked_gamma_power_all_subjects_by_rates[r])
        # evoked_gamma_rate = evoked_gamma_rate / np.mean(evoked_gamma_rate)  # normalize power (to emphasize PAC)
        mean = np.mean(evoked_gamma_rate, axis=0)
        sem = np.std(evoked_gamma_rate, axis=0) / np.sqrt(evoked_gamma_rate.shape[0])
        # evoked_times = np.linspace(-2 * np.pi, 2 * np.pi, mean.shape[0])
        print(mean.shape[0])
        axes[0, r].plot(evoked_times, mean, label=rate, lw=3)
        axes[0, r].fill_between(evoked_times, mean - sem, mean + sem, alpha=0.5, lw=0)
        axes[0, r].axvline(x=0, c='black', linestyle='--')
        axes[0, r].axhline(y=0, c='black', linestyle='--')
        axes[0, r].set_xlim(evoked_times[0], evoked_times[-1])
        axes[0, r].set_xlabel('Phase (rad)')
        if r==0:
            axes[0, r].set_ylabel('Power (percent change relative to baseline)')
        axes[0, r].legend()

        tval = mean / sem
        axes[1, r].plot(evoked_times, tval, label=rate, lw=3)
        axes[1, r].axvline(x=0, c='black', linestyle='--')
        axes[1, r].axhline(y=0, c='black', linestyle='--')
        axes[1, r].set_xlim(evoked_times[0], evoked_times[-1])
        axes[1, r].set_xlabel('Phase (rad)')
        if r==0:
            axes[1, r].set_ylabel('Tval')
        axes[1, r].legend()

    # by surprise quantile
    fig, axes = plt.subplots(2, 1, sharex=True, sharey='row')
    # cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#ff5a5f', '#6859D9'])
    # cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#6859D9', '#F678FA', '#ff5a5f'])
    cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#6859D9', '#ff5a5f'])
    # cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#0021DB', '#FF002A'])
    color = [cmap1(i) for i in np.linspace(0, 1, len(surprise_quantiles) - 1)]
    for q in range(len(surprise_quantiles) - 1):
        evoked_gamma_rate = np.array(evoked_gamma_power_all_subjects_by_surprise_quantile[q])
        # evoked_gamma_rate = evoked_gamma_rate / np.mean(evoked_gamma_rate)
        mean = np.mean(evoked_gamma_rate, axis=0)
        sem = np.std(evoked_gamma_rate, axis=0) / np.sqrt(evoked_gamma_rate.shape[0])
        axes[0].plot(evoked_times, mean, label=surprise_quantiles[q+1], c=color[q], lw=2)
        # axes[0].fill_between(evoked_times, mean - sem, mean + sem, alpha=0.5, lw=0)

        tval = mean / sem
        axes[1].plot(evoked_times, tval, label=surprise_quantiles[q+1], c=color[q], lw=2)
    axes[0].axvline(x=0, c='black', linestyle='--')
    axes[0].axhline(y=0, c='black', linestyle='--')
    axes[0].set_xlim(evoked_times[0], evoked_times[-1])
    axes[0].set_xlabel('Phase (rad)')
    axes[0].set_ylabel('Power (percent change relative to baseline)')
    axes[0].legend()
    axes[1].axvline(x=0, c='black', linestyle='--')
    axes[1].axhline(y=0, c='black', linestyle='--')
    axes[1].set_xlim(evoked_times[0], evoked_times[-1])
    axes[1].set_xlabel('Phase (rad)')
    axes[1].set_ylabel('Tval')
    axes[1].legend()

    # by rate and quantile
    fig, axes = plt.subplots(2, len(rates), sharex=True, sharey='row')
    for r, rate in enumerate(rates):
        evoked_gamma_rate = np.array(evoked_gamma_power_all_subjects_by_surprise_quantile_rate[r])
        for q in range(len(surprise_quantiles) - 1):
            evoked_gamma_quantile = np.array(evoked_gamma_rate[q])
            # evoked_gamma_quantile = evoked_gamma_quantile / np.mean(evoked_gamma_quantile)
            mean = np.mean(evoked_gamma_quantile, axis=0)
            sem = np.std(evoked_gamma_quantile, axis=0) / np.sqrt(evoked_gamma_quantile.shape[0])
            axes[0, r].plot(evoked_times, mean, label=surprise_quantiles[q + 1], c=color[q], lw=2)
            # axes[0].fill_between(evoked_times, mean - sem, mean + sem, alpha=0.5, lw=0)

            tval = mean / sem
            axes[1, r].plot(evoked_times, tval, label=surprise_quantiles[q + 1], c=color[q], lw=2)
        axes[0, r].axvline(x=0, c='black', linestyle='--')
        axes[0, r].axhline(y=0, c='black', linestyle='--')
        axes[0, r].set_xlim(evoked_times[0], evoked_times[-1])
        axes[0, r].set_xlabel('Phase (rad)')
        if r == 0:
            axes[0, r].set_ylabel('Power (percent change relative to baseline)')
            # axes[0, r].legend()
        axes[0, r].set_title(rate)
        axes[1, r].axvline(x=0, c='black', linestyle='--')
        axes[1, r].axhline(y=0, c='black', linestyle='--')
        axes[1, r].set_xlim(evoked_times[0], evoked_times[-1])
        axes[1, r].set_xlabel('Phase (rad)')
        if r == 0:
            axes[1, r].set_ylabel('Tval')
            # axes[1, r].legend()

    plt.show()


def note_evoked_gamma_analysis_phase_indiv_chan(save_path):
    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']
    # surprise_quantiles = [0, 0.33, 0.66, 1]
    # surprise_quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1]
    surprise_quantiles = [i/6 for i in range(7)]
    # surprise_quantiles = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # surprise_quantiles = [i/20 for i in range(21)]
    # surprise_quantiles = [i/7 for i in range(8)]
    path = Path(save_path)
    subject_paths = sorted(list(path.glob(f'*note_evoked_power.pkl')))
    n_points = 100  # Number of points in the common phase grid
    common_phase_grid = np.linspace(-2 * np.pi, 2 * np.pi, n_points)
    for i, subject_path in enumerate(tqdm(subject_paths)):
        note_evoked_obj = np.load(subject_path, allow_pickle=True)
        times = note_evoked_obj['times']
        trial_time = note_evoked_obj['trial_time']
        surprise = note_evoked_obj['surprise']
        note_rate = np.array(note_evoked_obj['note_rate'])
        rep = note_evoked_obj['rep']
        onset = note_evoked_obj['onset']
        gamma_power = np.array(note_evoked_obj['gamma_power'])
        gamma = np.array(note_evoked_obj['gamma'])

        gamma_power_all_chan = gamma_power.copy()

        for c in range(gamma_power_all_chan.shape[1]):
            evoked_gamma_power_all_subjects = []
            evoked_gamma_power_all_subjects_by_rates = [[] for _ in range(len(rates))]
            evoked_gamma_power_all_subjects_by_surprise_quantile = [[] for _ in range(len(surprise_quantiles) - 1)]
            evoked_gamma_power_all_subjects_by_surprise_quantile_rate = [
                [[] for q in range(len(surprise_quantiles) - 1)] for r in range(len(rates))]
            gamma_power = gamma_power_all_chan[:, c]
            surprise_full = []
            rep_full = []
            onset_full = []
            note_rate_full = []
            evoked_gamma_power = []
            for epoch in range(len(trial_time)):
                # if rep[epoch] == 2:
                surprise_full += surprise[epoch]
                for t in trial_time[epoch]:
                    rep_full.append(rep[epoch] - 1)  # to get rep between 0 and 1
                    onset_full.append(onset[epoch])
                    note_rate_full.append(note_rate[epoch])
                    rate = float(note_rate[epoch][:3])
                    n_samples_before = np.argmin(np.abs(times - 1 / rate)) - np.argmin(np.abs(times))
                    start = np.argmin(np.abs(times - t)) - n_samples_before
                    end = np.argmin(np.abs(times - t)) + n_samples_before
                    # evoked_gamma_power.append(gamma_power[epoch, start:end])
                    # evoked_gamma_power.append(gamma_power[epoch, start:end] - np.mean(gamma_power[epoch, start:np.argmin(np.abs(times - t))]))
                    interpolator = interp1d(np.linspace(-2*np.pi,2*np.pi, end-start), gamma_power[epoch, start:end], kind='linear', bounds_error=False, fill_value=0)
                    # interpolator = interp1d(np.linspace(-2*np.pi,2*np.pi, end-start), gamma_power[epoch, start:end] - np.mean(gamma_power[epoch, start:np.argmin(np.abs(times - t))]), kind='linear', bounds_error=False, fill_value=0)
                    evoked_gamma_power.append(interpolator(common_phase_grid))

            evoked_gamma_power = np.array(evoked_gamma_power)
            evoked_gamma_power_all_subjects.append(np.mean(evoked_gamma_power, axis=0))

            for r, rate in enumerate(rates):
                rate_notes_indices = [e for e in range(len(note_rate_full)) if note_rate_full[e] == rate]
                evoked_gamma_power_rate = np.array([evoked_gamma_power[e] for e in rate_notes_indices])  # also works without common phase grid
                evoked_gamma_power_all_subjects_by_rates[r].append(np.mean(evoked_gamma_power_rate, axis=0))

            for q in range(len(surprise_quantiles) - 1):
                lower_value = np.quantile(surprise_full, surprise_quantiles[q])
                higher_value = np.quantile(surprise_full, surprise_quantiles[q+1])
                quantile_indices = [e for e in range(len(surprise_full)) if (surprise_full[e] >= lower_value) and (surprise_full[e] <= higher_value)]
                evoked_gamma_power_all_subjects_by_surprise_quantile[q].append(np.mean(evoked_gamma_power[quantile_indices, :], axis=0))

            for r, rate in enumerate(rates):
                rate_notes_indices = [e for e in range(len(note_rate_full)) if note_rate_full[e] == rate]
                for q in range(len(surprise_quantiles) - 1):
                    lower_value = np.quantile(np.array(surprise_full), surprise_quantiles[q])
                    higher_value = np.quantile(np.array(surprise_full), surprise_quantiles[q + 1])
                    quantile_indices = [e for e in range(len(surprise_full)) if (surprise_full[e] >= lower_value) and (surprise_full[e] <= higher_value) and (e in rate_notes_indices)]
                    evoked_gamma_power_all_subjects_by_surprise_quantile_rate[r][q].append(np.mean(evoked_gamma_power[quantile_indices, :], axis=0))

            evoked_times = common_phase_grid

            # evoked_gamma_power_all_subjects = np.array(evoked_gamma_power_all_subjects)
            # mean = np.mean(evoked_gamma_power_all_subjects, axis=0)
            # sem = np.std(evoked_gamma_power_all_subjects, axis=0) / np.sqrt(evoked_gamma_power_all_subjects.shape[0])
            # print(evoked_gamma_power_all_subjects.shape)
            # plt.figure()
            # plt.plot(evoked_times, mean, label='Evoked High Gamma', lw=3)
            # plt.fill_between(evoked_times, mean-sem, mean+sem, alpha=0.5, lw=0)
            # # plt.scatter(evoked_times[significant], r2_all_subjects[significant], color='red', label='Significant time points')
            # plt.axvline(x=0, c='black', linestyle='--')
            # plt.axhline(y=0, c='black', linestyle='--')
            # plt.xlim(evoked_times[0], evoked_times[-1])
            # plt.xlabel('Phase (rad)')
            # plt.ylabel('Power (percent change relative to baseline)')
            # plt.legend()
            #
            # tval = mean / sem
            # print(evoked_gamma_power_all_subjects.shape)
            # plt.figure()
            # plt.plot(evoked_times, tval, label='Evoked High Gamma', lw=3)
            # # plt.scatter(evoked_times[significant], r2_all_subjects[significant], color='red', label='Significant time points')
            # plt.axvline(x=0, c='black', linestyle='--')
            # plt.axhline(y=0, c='black', linestyle='--')
            # plt.xlim(evoked_times[0], evoked_times[-1])
            # plt.xlabel('Phase (rad)')
            # plt.ylabel('Tval')
            # plt.legend()
            #
            # # by rate
            # fig, axes = plt.subplots(2, len(rates), sharex=True, sharey='row')
            # for r, rate in enumerate(rates):
            #     evoked_gamma_rate = np.array(evoked_gamma_power_all_subjects_by_rates[r])
            #     mean = np.mean(evoked_gamma_rate, axis=0)
            #     sem = np.std(evoked_gamma_rate, axis=0) / np.sqrt(evoked_gamma_rate.shape[0])
            #     # evoked_times = np.linspace(-2 * np.pi, 2 * np.pi, mean.shape[0])
            #     print(mean.shape[0])
            #     axes[0, r].plot(evoked_times, mean, label=rate, lw=3)
            #     axes[0, r].fill_between(evoked_times, mean - sem, mean + sem, alpha=0.5, lw=0)
            #     axes[0, r].axvline(x=0, c='black', linestyle='--')
            #     axes[0, r].axhline(y=0, c='black', linestyle='--')
            #     axes[0, r].set_xlim(evoked_times[0], evoked_times[-1])
            #     axes[0, r].set_xlabel('Phase (rad)')
            #     if r==0:
            #         axes[0, r].set_ylabel('Power (percent change relative to baseline)')
            #     axes[0, r].legend()
            #
            #     tval = mean / sem
            #     axes[1, r].plot(evoked_times, tval, label=rate, lw=3)
            #     axes[1, r].axvline(x=0, c='black', linestyle='--')
            #     axes[1, r].axhline(y=0, c='black', linestyle='--')
            #     axes[1, r].set_xlim(evoked_times[0], evoked_times[-1])
            #     axes[1, r].set_xlabel('Phase (rad)')
            #     if r==0:
            #         axes[1, r].set_ylabel('Tval')
            #     axes[1, r].legend()

            # by surprise quantile
            fig, axes = plt.subplots(2, 1, sharex=True, sharey='row')
            # cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#ff5a5f', '#6859D9'])
            # cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#6859D9', '#F678FA', '#ff5a5f'])
            cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#6859D9', '#ff5a5f'])
            # cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#0021DB', '#FF002A'])
            color = [cmap1(i) for i in np.linspace(0, 1, len(surprise_quantiles) - 1)]
            for q in range(len(surprise_quantiles) - 1):
                evoked_gamma_rate = np.array(evoked_gamma_power_all_subjects_by_surprise_quantile[q])
                mean = np.mean(evoked_gamma_rate, axis=0)
                sem = np.std(evoked_gamma_rate, axis=0) / np.sqrt(evoked_gamma_rate.shape[0])
                axes[0].plot(evoked_times, mean, label=surprise_quantiles[q+1], c=color[q], lw=2)
                # axes[0].fill_between(evoked_times, mean - sem, mean + sem, alpha=0.5, lw=0)

                tval = mean / sem
                axes[1].plot(evoked_times, tval, label=surprise_quantiles[q+1], c=color[q], lw=2)
            axes[0].axvline(x=0, c='black', linestyle='--')
            axes[0].axhline(y=0, c='black', linestyle='--')
            axes[0].set_xlim(evoked_times[0], evoked_times[-1])
            axes[0].set_xlabel('Phase (rad)')
            axes[0].set_ylabel('Power (percent change relative to baseline)')
            axes[0].legend()
            axes[1].axvline(x=0, c='black', linestyle='--')
            axes[1].axhline(y=0, c='black', linestyle='--')
            axes[1].set_xlim(evoked_times[0], evoked_times[-1])
            axes[1].set_xlabel('Phase (rad)')
            axes[1].set_ylabel('Tval')
            axes[1].legend()

            # by rate and quantile
            fig, axes = plt.subplots(2, len(rates), sharex=True, sharey='row')
            for r, rate in enumerate(rates):
                evoked_gamma_rate = np.array(evoked_gamma_power_all_subjects_by_surprise_quantile_rate[r])
                for q in range(len(surprise_quantiles) - 1):
                    evoked_gamma_quantile = np.array(evoked_gamma_rate[q])
                    mean = np.mean(evoked_gamma_quantile, axis=0)
                    sem = np.std(evoked_gamma_quantile, axis=0) / np.sqrt(evoked_gamma_quantile.shape[0])
                    axes[0, r].plot(evoked_times, mean, label=surprise_quantiles[q + 1], c=color[q], lw=2)
                    # axes[0].fill_between(evoked_times, mean - sem, mean + sem, alpha=0.5, lw=0)

                    tval = mean / sem
                    axes[1, r].plot(evoked_times, tval, label=surprise_quantiles[q + 1], c=color[q], lw=2)
                axes[0, r].axvline(x=0, c='black', linestyle='--')
                axes[0, r].axhline(y=0, c='black', linestyle='--')
                axes[0, r].set_xlim(evoked_times[0], evoked_times[-1])
                axes[0, r].set_xlabel('Phase (rad)')
                if r == 0:
                    axes[0, r].set_ylabel('Power (percent change relative to baseline)')
                    # axes[0, r].legend()
                axes[0, r].set_title(rate)
                axes[1, r].axvline(x=0, c='black', linestyle='--')
                axes[1, r].axhline(y=0, c='black', linestyle='--')
                axes[1, r].set_xlim(evoked_times[0], evoked_times[-1])
                axes[1, r].set_xlabel('Phase (rad)')
                if r == 0:
                    axes[1, r].set_ylabel('Tval')
                    # axes[1, r].legend()

            plt.show()


def amplitude_by_phase_histogram_plot(csv_path):
    """Plot gamma by phase histograms"""
    df = pd.read_csv(csv_path)
    print(len(df))
    df = df.dropna(subset=['correct', 'correct2'])
    print(len(df))
    print(df.columns)
    print(df[['surprise', 'stim_name']])

    # df = df[df['rep'] == 2]
    df = df[df['rep'] == 1]
    # df = df[df['correct2']==1]

    # Sort the histogram columns for each row
    sorted_histograms = df[[f"histogram_{i}" for i in range(18)]].apply(
        lambda row: np.sort(row.values)[::-1], axis=1
    )
    # Replace the original histogram columns with the sorted ones
    for i in range(18):
        df[f"histogram_{i}"] = sorted_histograms.apply(lambda row: row[i])

    bin_columns = [f'histogram_{i}' for i in range(18)]

    electrode_df = df.groupby(
        ['electrode', 'subject', 'stim_name', 'rep', 'note_rate', 'onset', 'surprise', 'uncertainty', 'predicted_density', 'correct',
         'correct2'], as_index=False)[bin_columns].mean()
    print(electrode_df)

    # average electrodes by subject
    # df = df.groupby(
    #     ['subject', 'stim_name', 'rep', 'note_rate', 'onset', 'surprise', 'uncertainty', 'predicted_density', 'correct',
    #      'correct2'], as_index=False)[bin_columns].mean()
    # print(df)
    # print(df.columns)
    # print(df[['surprise', 'stim_name']])

    def calculate_entropy(row, bin_columns):
        P = np.array(row[bin_columns], dtype=float)
        # return -np.sum(P * np.log(P + 1e-9))  # Add small value to avoid log(0)
        return np.std(P)  # Add small value to avoid log(0)

    # Histogram bin columns
    df['std'] = df.apply(calculate_entropy, bin_columns=bin_columns, axis=1)

    # Step 1: Assign quantiles
    # num_quantiles = 5
    # num_quantiles = 8
    num_quantiles = 6
    df['surprise_quantile'] = pd.qcut(df['surprise'], q=num_quantiles, labels=False)
    # df['surprise_quantile'] = pd.qcut(df['pac_stim_freq'], q=num_quantiles, labels=False)

    # Step 2: Compute average histograms for each quantile
    histogram_cols = [f'histogram_{i}' for i in range(18)]
    mean_histograms = df.groupby('surprise_quantile')[histogram_cols].mean()

    # Step 3: Prepare data for plotting
    plot_data = mean_histograms.reset_index().melt(id_vars='surprise_quantile',
                                                   var_name='histogram_bin',
                                                   value_name='mean_value')
    plot_data['histogram_bin'] = plot_data['histogram_bin'].str.extract('(\d+)').astype(int)

    num_bins = len(histogram_cols)
    phase_bins = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)  # Phase values for bins

    # Step 4: Plot
    plt.figure(figsize=(12, 6))
    cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#6859D9', '#ff5a5f'])
    colors = plt.cm.viridis(np.linspace(0, 1, num_quantiles))  # Fixed colors for 6 quantiles
    # colors = cmap1(np.linspace(0, 1, num_quantiles))  # Fixed colors for 6 quantiles

    for quantile in range(num_quantiles):
        subset = plot_data[plot_data['surprise_quantile'] == quantile]
        plt.bar(subset['histogram_bin'] + quantile * 0.1,  # Offset bars slightly for clarity
                subset['mean_value'],
                width=0.1,
                label=f'Quantile {quantile + 1}',
                color=colors[quantile])

    # Customize the plot
    plt.xlabel('Phase bin')
    plt.ylabel('Normalized gamma power')
    plt.title('Average Histogram by Surprise Quantile')
    plt.xticks(range(18))
    plt.legend(title='Surprise Quantile')
    plt.tight_layout()

    plt.figure(figsize=(12, 6))
    plt.bar([i for i in range(18)],
            df[histogram_cols].mean(),
            width=0.3,)

    # by note rate
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True, sharex=True)
    axes = axes.flatten()
    # Iterate over note_rate values
    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']
    for i, note_rate in enumerate(rates):
        ax = axes[i]

        # Filter data for the current note_rate
        filtered_df = df[df['note_rate'] == note_rate]

        # Step 2: Compute average histograms for each quantile
        histogram_cols = [f'histogram_{i}' for i in range(num_bins)]
        ax.bar([i for i in range(18)],
               filtered_df[histogram_cols].mean(),
               width=0.3,
               )
        # Customize subplot
        ax.set_title(f'Note Rate: {note_rate}')
        ax.set_xlabel('Phase bin')
        if i % 3 == 0:  # Label y-axis for the first column
            ax.set_ylabel('Normalized gamma power')

    # by note rate and surprise bin
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True, sharex=True)
    axes = axes.flatten()
    # Iterate over note_rate values
    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']
    for i, note_rate in enumerate(rates):
        ax = axes[i]

        # Filter data for the current note_rate
        filtered_df = df[df['note_rate'] == note_rate]
        filtered_df['surprise_quantile'] = pd.qcut(filtered_df['surprise'], q=num_quantiles, labels=False)

        # Step 2: Compute average histograms for each quantile
        histogram_cols = [f'histogram_{i}' for i in range(num_bins)]
        mean_histograms = filtered_df.groupby('surprise_quantile')[histogram_cols].mean()

        # Step 3: Prepare data for plotting
        plot_data = mean_histograms.reset_index().melt(id_vars='surprise_quantile',
                                                       var_name='histogram_bin',
                                                       value_name='mean_value')
        plot_data['histogram_bin'] = plot_data['histogram_bin'].str.extract('(\d+)').astype(int)

        # Plot
        colors = plt.cm.viridis(np.linspace(0, 1, num_quantiles))  # Colors for quantiles
        for quantile in range(num_quantiles):
            subset = plot_data[plot_data['surprise_quantile'] == quantile]
            ax.bar(subset['histogram_bin'] + quantile * 0.1,  # Offset for clarity
                   subset['mean_value'],
                   width=0.1,  # Bar width proportional to phase spacing
                   label=f'Quantile {quantile + 1}',
                   color=colors[quantile])

        # Customize subplot
        ax.set_title(f'Note Rate: {note_rate}')
        ax.set_xlabel('Phase bin')
        if i % 3 == 0:  # Label y-axis for the first column
            ax.set_ylabel('Normalized gamma power')

    # Adjust layout and add legend
    plt.tight_layout()
    axes[0].legend(title='Surprise Quantile', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.show()

    # Calculate mean entropy per quantile
    # mean_entropy_per_quantile = df.groupby('surprise_quantile')['pac_stim_freq'].mean()
    mean_entropy_per_quantile = df.groupby('surprise_quantile')['std'].mean()
    plt.figure(figsize=(8, 5))
    mean_entropy_per_quantile.plot(kind='bar', color='skyblue')
    plt.xlabel('Surprise Quantile')
    plt.ylabel('Mean PAC Metric (std)')
    plt.title('PAC Metric Across Surprise Quantiles')
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    df_subject_avg = df.groupby(['subject', 'surprise_quantile'])['std'].mean().reset_index()
    sns.boxplot(
        data=df,
        x='surprise_quantile',
        y='std',
        showfliers=False,  # Optionally hide outliers in the boxplot
        color='white'
    )
    # Overlay scatterplot and lines for each subject
    palette = sns.color_palette('tab10',
                                n_colors=len(df_subject_avg['subject'].unique()))  # Generate colors for subjects
    subject_colors = {subject: palette[i] for i, subject in enumerate(df_subject_avg['subject'].unique())}  # Map colors
    # Plot lines and scatter points
    for subject, group in df_subject_avg.groupby('subject'):
        # Plot line for the subject
        plt.plot(
            group['surprise_quantile'],
            group['std'],
            color=subject_colors[subject],
            linewidth=2,
            alpha=0.8
        )
        # Plot scatter points for the subject
        plt.scatter(
            group['surprise_quantile'],
            group['std'],
            color=subject_colors[subject],
            edgecolor='k',
            linewidth=0.5,
            label=subject if group['surprise_quantile'].iloc[0] == group['surprise_quantile'].min() else None
            # Avoid duplicate labels
        )

    quantile_groups = [group['std'].values for _, group in df.groupby('surprise_quantile')]
    anova_result = f_oneway(*quantile_groups)
    print(f"ANOVA p-value: {anova_result.pvalue}")

    plt.show()


    # for subject in np.unique(electrode_df['subject']):
    #     for electrode in np.unique(electrode_df[electrode_df['subject']==subject]['electrode']):
    #         print(subject)
    #         print(electrode)
    #
    #         current_df = electrode_df[(electrode_df['subject']==subject) & (electrode_df['electrode']==electrode)]
    #
    #         current_df['std'] = df.apply(calculate_entropy, bin_columns=bin_columns, axis=1)
    #
    #         # Step 1: Assign quantiles
    #         num_quantiles = 8
    #         current_df['surprise_quantile'] = pd.qcut(current_df['surprise'], q=num_quantiles, labels=False)
    #         # df['surprise_quantile'] = pd.qcut(df['pac_stim_freq'], q=num_quantiles, labels=False)
    #
    #         # Step 2: Compute average histograms for each quantile
    #         histogram_cols = [f'histogram_{i}' for i in range(18)]
    #         mean_histograms = current_df.groupby('surprise_quantile')[histogram_cols].mean()
    #
    #         # Step 3: Prepare data for plotting
    #         plot_data = mean_histograms.reset_index().melt(id_vars='surprise_quantile',
    #                                                        var_name='histogram_bin',
    #                                                        value_name='mean_value')
    #         plot_data['histogram_bin'] = plot_data['histogram_bin'].str.extract('(\d+)').astype(int)
    #
    #         # Step 4: Plot
    #         plt.figure(figsize=(12, 6))
    #         colors = plt.cm.viridis(np.linspace(0, 1, num_quantiles))  # Fixed colors for 6 quantiles
    #
    #         for quantile in range(num_quantiles):
    #             subset = plot_data[plot_data['surprise_quantile'] == quantile]
    #             plt.bar(subset['histogram_bin'] + quantile * 0.1,  # Offset bars slightly for clarity
    #                     subset['mean_value'],
    #                     width=0.1,
    #                     label=f'Quantile {quantile + 1}',
    #                     color=colors[quantile])
    #
    #         # Customize the plot
    #         plt.xlabel('Histogram Bin')
    #         plt.ylabel('Mean Value')
    #         plt.title('Average Histogram by Surprise Quantile')
    #         plt.xticks(range(18))
    #         plt.legend(title='Surprise Quantile')
    #         plt.tight_layout()
    #
    #         # Calculate mean entropy per quantile
    #         mean_entropy_per_quantile = current_df.groupby('surprise_quantile')['std'].mean()
    #         plt.figure(figsize=(8, 5))
    #         mean_entropy_per_quantile.plot(kind='bar', color='skyblue')
    #         plt.xlabel('Surprise Quantile')
    #         plt.ylabel('Mean PAC Metric (std)')
    #         plt.title('PAC Metric Across Surprise Quantiles')
    #         plt.xticks(rotation=0)
    #         plt.tight_layout()
    #
    #         quantile_groups = [group['std'].values for _, group in current_df.groupby('surprise_quantile')]
    #         anova_result = f_oneway(*quantile_groups)
    #         print(f"ANOVA p-value: {anova_result.pvalue}")
    #
    #         plt.show()

    # high surprise trials loop
    df_trials = df.groupby(['stim_name', 'surprise_quantile', 'surprise'])[histogram_cols].mean().reset_index()
    print(df_trials)
    # Define the number of phase bins
    num_bins = len([col for col in df_trials.columns if col.startswith("histogram_")])
    phase_bins = np.linspace(0, 2 * np.pi, num_bins, endpoint=False) + (np.pi / num_bins)

    # Loop through each trial
    for idx, row in df_trials.iterrows():
        if row['surprise_quantile'] == 7:
            # Extract the histogram values for the current trial
            histogram_values = row[[f"histogram_{i}" for i in range(num_bins)]].values

            # Plot the histogram
            plt.figure()
            plt.bar(phase_bins, histogram_values, width=(2 * np.pi / num_bins), color="skyblue", edgecolor="black", align="center")

            # Add title and labels
            plt.title(f"Histogram for Trial {row['stim_name']}, surprise = {row['surprise']}, quantile = {row['surprise_quantile']}")
            plt.xlabel("Phase Bin (rad)")
            plt.ylabel("Normalized Gamma Power")

            # Show the plot
            plt.show()


    return



if __name__ == "__main__":
    """plot_comodulogram_meg"""
    # plot_comodulogram_meg("/home/mathieupvc/These/data/meg_maestro/pac_analysis/all_trials", best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_10channels_100ms.npy')
    # plot_comodulogram_meg("/home/mathieupvc/These/data/meg_maestro/pac_analysis/all_trials", best_channels_path=None)

    """plot_comodulogram_meg"""
    # plot_contrast_comodulogram_meg("/home/mathieupvc/These/data/meg_maestro/pac_analysis/correct",
    #                                "/home/mathieupvc/These/data/meg_maestro/pac_analysis/incorrect",
    #                                # best_channels_path=None)
    #                                best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_10channels_100ms.npy')

    # comodulograms_correct_incorrect('/home/mathieupvc/These/data/meg_maestro/correct_incorrect_all_reps_lists.pkl')

    """save_comodulograms_seeg"""
    epochs_path = '/home/mathieupvc/These/data/meg_maestro/seeg_trials/Gra_Lu_epochs_mono_preprocessed_trials_IDA_nofilter.fif'
    save_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg'
    channels = ["H'", "B'"]
    # save_comodulograms_seeg(epochs_path, save_path, channels, indiv_trial=False, condition=None, n_jobs=-1)

    """save_comodulograms_seeg_all_subjects"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/seeg_trials'
    save_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg'
    # save_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/no_4hz'
    fs_subjects_dir = '/home/mathieupvc/freesurfer/subjects'
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus']
    # save_comodulograms_seeg_all_subjects(data_path, save_path, channels, fs_subjects_dir)

    """plot_comodulogram_seeg"""
    # channels = ["H'"]
    channels = ["B'"]
    # plot_comodulogram_seeg("/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/Bou_Ni_comodulogram.pkl", channels)

    """plot_comodulogram_seeg_all_subjects"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg'
    # data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/no_4hz'
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal']
    # channels = ['Left-Hippocampus', 'Right-Hippocampus']
    # channels = ['Left-Hippocampus']
    # channels = ['Right-Hippocampus']
    # plot_comodulogram_seeg_all_subjects(data_path, channels=channels)

    """comodulograms_correct_incorrect_seeg"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/seeg_trials'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    save_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg'
    # save_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/no_7hz'
    fs_subjects_dir = '/home/mathieupvc/freesurfer/subjects'
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus']
    # comodulograms_correct_incorrect_seeg(data_path, behavior_path, save_path, channels, fs_subjects_dir)

    """comodulograms_correct_incorrect_encoding_seeg"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/seeg_trials'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    save_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg'
    fs_subjects_dir = '/home/mathieupvc/freesurfer/subjects'
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus']
    # comodulograms_correct_incorrect_encoding_seeg(data_path, behavior_path, save_path, channels, fs_subjects_dir)

    """comodulograms_correct_incorrect_retrieval_seeg"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/seeg_trials'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    save_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg'
    fs_subjects_dir = '/home/mathieupvc/freesurfer/subjects'
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus']
    # comodulograms_correct_incorrect_retrieval_seeg(data_path, behavior_path, save_path, channels, fs_subjects_dir)

    """comodulograms_correct_incorrect_novelty_seeg"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/seeg_trials'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    save_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg'
    fs_subjects_dir = '/home/mathieupvc/freesurfer/subjects'
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus']
    # comodulograms_correct_incorrect_novelty_seeg(data_path, behavior_path, save_path, channels, fs_subjects_dir)

    """plot_contrast_comodulogram_seeg"""
    correct = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/correct/Sou_Fa_comodulogram.pkl'
    incorrect = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/incorrect/Sou_Fa_comodulogram.pkl'
    # channels = ["H"]
    channels = ["B"]
    # plot_contrast_comodulogram_seeg(correct, incorrect, channels=channels)

    """plot_contrast_comodulogram_seeg_all_subjects"""
    # data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/correct'
    # data_path2 = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/incorrect'
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/encoding/correct'
    data_path2 = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/encoding/incorrect'
    # data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/retrieval/correct'
    # data_path2 = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/retrieval/incorrect'
    # data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/novelty_detection/correct'
    # data_path2 = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/novelty_detection/incorrect'
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal']
    # channels = ['ctx-lh-superiortemporal']
    # channels = ['ctx-rh-superiortemporal']
    # channels = ['Left-Hippocampus', 'Right-Hippocampus']
    channels = ['Left-Hippocampus']
    # channels = ['Right-Hippocampus']
    # plot_contrast_comodulogram_seeg_all_subjects(data_path, data_path2, channels=channels)

    """pac_stim_freq_seeg"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/seeg_trials'
    save_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg'
    fs_subjects_dir = '/home/mathieupvc/freesurfer/subjects'
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus']
    # pac_stim_freq_seeg(data_path, save_path, channels, fs_subjects_dir)

    """pac_fixed_freq_seeg"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/seeg_trials'
    save_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg'
    fs_subjects_dir = '/home/mathieupvc/freesurfer/subjects'
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus']
    low_freq = 4
    # pac_fixed_freq_seeg(data_path, save_path, channels, fs_subjects_dir, low_freq=low_freq)

    """plot_contrast_pac_stim_freq_seeg"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal']
    # channels = ['Left-Hippocampus', 'Right-Hippocampus']
    # channels = ['Left-Hippocampus']
    # channels = ['Right-Hippocampus']
    # freq = 4
    freq = 'stim'
    # plot_contrast_pac_freq_seeg(data_path, behavior_path, channels=channels, freq=freq)

    """epochs_spectrum_seeg"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/seeg_trials'
    save_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_spectrum'
    fs_subjects_dir = '/home/mathieupvc/freesurfer/subjects'
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus']
    # channels = ['Left-Hippocampus']
    # epochs_spectrum_seeg(data_path, save_path, channels=channels, fs_subjects_dir=fs_subjects_dir)

    """plot_electrode_spectrum"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_spectrum'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal']
    # channels = ['Left-Hippocampus']
    # channels = ['Right-Hippocampus']
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus']
    subject_subset = None
    # subject_subset = [7]
    # plot_electrode_spectrum(data_path, behavior_path, channels=channels, subject_subset=subject_subset)

    """fdr_correction_electrode_spectrum"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_spectrum'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # fdr_correction_electrode_spectrum(data_path, behavior_path)

    """memorized_vs_novel_paired_electrode_spectrum"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_spectrum'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal']
    # channels = ['Left-Hippocampus']
    # channels = ['Right-Hippocampus']
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus']
    subject_subset = None
    # memorized_vs_novel_paired_electrode_spectrum(data_path, behavior_path, channels=channels, subject_subset=subject_subset)

    """epochs_comodulogram_seeg"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/seeg_trials'
    # save_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram'
    save_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram_low_resolution'
    fs_subjects_dir = '/home/mathieupvc/freesurfer/subjects'
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus']
    # channels = ['Left-Hippocampus']
    # epochs_comodulogram_seeg(data_path, save_path, channels=channels, fs_subjects_dir=fs_subjects_dir)

    """epochs_comodulogram_seeg_all"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/seeg_trials'
    save_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram_low_resolution_all_channels'
    fs_subjects_dir = '/home/mathieu/freesurfer/subjects'
    # epochs_comodulogram_seeg_all(data_path, save_path, fs_subjects_dir=fs_subjects_dir)

    """epochs_rms_seeg_all"""
    data_path = '/home/mathieu/These/data/meg_maestro/seeg_trials'
    save_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels'
    # epochs_rms_seeg_all(data_path, save_path)
    # plot_sample_activity_seeg(data_path)
    """epochs_rms_time_windows_seeg_all"""
    # epochs_rms_time_windows_seeg_all(data_path, save_path)

    """epochs_band_power_seeg_all"""
    data_path = '/home/mathieu/These/data/meg_maestro/seeg_trials'
    # save_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_theta_power_all_channels'
    save_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_high_gamma_power_all_channels'
    fs_subjects_dir = '/home/mathieu/freesurfer/subjects'
    # freq_bands = np.arange(4, 9)
    # freq_bands = np.arange(50, 151, 10)
    # freq_bands = [4, 8]
    freq_bands = [120, 140]
    # freq_bands = [80, 150]
    # epochs_band_power_seeg_all(data_path, save_path, freq_bands=freq_bands, fs_subjects_dir=fs_subjects_dir)
    """epochs_band_power_time_windows_seeg_all"""
    # epochs_band_power_time_windows_seeg_all(data_path, save_path, freq_bands=freq_bands, fs_subjects_dir=fs_subjects_dir)
    """epochs_band_power_stim_freq_seeg_all"""
    save_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_stim_freq_all_channels'
    # epochs_band_power_stim_freq_seeg_all(data_path, save_path, fs_subjects_dir=fs_subjects_dir)

    """epochs_band_power_all_freq_seeg_all"""
    save_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs'
    # epochs_band_power_all_freq_seeg_all(data_path, save_path)

    """epochs_pac_seeg_all"""
    data_path = '/home/mathieu/These/data/meg_maestro/seeg_trials'
    save_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_channels_120_140'
    fs_subjects_dir = '/home/mathieu/freesurfer/subjects'
    low_freq = 6
    # high_freq = 130
    # high_freq = 115
    # high_freq = 160
    high_freq = 50
    # epochs_pac_seeg_all(data_path, save_path, low_freq=low_freq, high_freq=high_freq, fs_subjects_dir=fs_subjects_dir)
    # epochs_pac_time_windows_seeg_all(data_path, save_path, low_freq=low_freq, high_freq=high_freq)
    """epochs_pac_theta_bands_time_windows_seeg_all"""
    save_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_theta_bands_all_channels_120_140'
    # epochs_pac_theta_bands_time_windows_seeg_all(data_path, save_path, low_freq=[4,5,6,7,8], high_freq=high_freq)
    """epochs_pac_stim_freq_seeg_all"""
    save_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels'
    # epochs_pac_stim_freq_seeg_all(data_path, save_path, high_freq=high_freq, fs_subjects_dir=fs_subjects_dir)
    # epochs_pac_stim_freq_time_windows_seeg_all(data_path, save_path, high_freq)

    """epochs_phase_coherence_stim_freq_seeg_all"""
    save_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_phase_coherence_stim_freq_all_channels'
    # epochs_phase_coherence_stim_freq_seeg_all(data_path, save_path)
    # epochs_phase_coherence_stim_freq_time_windows_seeg_all(data_path, save_path)
    save_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_phase_coherence_all_freqs'
    # epochs_phase_coherence_all_freq_seeg_all(data_path, save_path)

    """epochs_amplitude_by_phase_histogram_seeg_all"""
    data_path = '/home/mathieu/These/data/meg_maestro/seeg_trials'
    save_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140'
    high_freq = 130
    # epochs_amplitude_by_phase_histogram_seeg_all(data_path, save_path, high_freq=high_freq)

    """epochs_abs_time_course_seeg_all"""
    data_path = '/home/mathieu/These/data/meg_maestro/seeg_trials'
    save_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_ma_rms_all_channels'
    # epochs_ma_rms_seeg_all(data_path, save_path)

    """plot_electrode_comodulogram"""
    # data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram'
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram_low_resolution'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal']
    # channels = ['Left-Hippocampus']
    # channels = ['Right-Hippocampus']
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus']
    # subject_subset = None
    subject_subset = [2]
    # plot_electrode_comodulogram(data_path, behavior_path, channels=channels, subject_subset=subject_subset)

    """count_channels"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram'
    # channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal']
    # channels = ['Left-Hippocampus']
    channels = ['Right-Hippocampus']
    # count_channels(data_path, channels=channels)

    """stat_test_electrode_comodulogram"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram_low_resolution_all_channels'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # stat_test_electrode_comodulogram(data_path, behavior_path, subject_subset=None)

    """stat_test_electrode_comodulogram_correct_incorrect"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram_low_resolution_all_channels'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # stat_test_electrode_comodulogram_correct_incorrect(data_path, behavior_path, subject_subset=None)

    """stat_test_electrode_rms"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # stat_test_electrode_rms(data_path, behavior_path, subject_subset=None)

    """plot_ratio_signif_eletrodes"""
    # data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram_low_resolution_all_channels'
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels'
    # lowest_p_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram_low_resolution_all_channels/lowest_p_all_channels.pkl'
    lowest_p_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/lowest_p_all_channels.pkl'
    regions = None
    # regions = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus']
    # regions = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus',
    #            'Left-Amygdala', 'Right-Amygdala', 'ctx-lh-bankssts', 'ctx-rh-bankssts',
    #            'ctx-lh-caudalanteriorcingulate', 'ctx-rh-caudalanteriorcingulate', 'ctx-lh-caudalmiddlefrontal',
    #            'ctx-rh-caudalmiddlefrontal', 'ctx-lh-entorhinal', 'ctx-rh-entorhinal', 'ctx-lh-fusiform',
    #            'ctx-rh-fusiform', 'ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal', 'ctx-lh-middletemporal',
    #            'ctx-rh-middletemporal', 'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal', 'ctx-lh-posteriorcingulate',
    #            'ctx-rh-posteriorcingulate', 'ctx-lh-precuneus', 'ctx-rh-precuneus', 'ctx-lh-transversetemporal',
    #            'ctx-rh-transversetemporal', 'ctx-lh-superiorfrontal', 'ctx-rh-superiorfrontal'
    #            ]
    first_parcel_label = True
    # analysis_name = 'comodulogram'
    analysis_name = 'rms'
    # plot_ratio_signif_eletrodes(data_path, lowest_p_path, first_parcel_label=first_parcel_label, regions=regions, analysis_name=analysis_name, n_permut=100000)

    """plot_significant_electrode_comodulogram"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram_low_resolution_all_channels'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    lowest_p_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram_low_resolution_all_channels/lowest_p_all_channels.pkl'
    regions_list = [['ctx-lh-parsopercularis'],
                    ['Left-Hippocampus', 'ctx-lh-parsopercularis', 'ctx-rh-inferiorparietal'],
                    ['Left-Amygdala', 'Right-Hippocampus', 'Left-Thalamus-Proper'],
                    ['Right-Thalamus-Proper'],
                    ['Right-Amygdala', 'Right-Hippocampus']]  # one list per condition
    first_parcel_label = True
    # plot_significant_electrode_comodulogram(data_path, behavior_path, lowest_p_path, first_parcel_label=first_parcel_label, regions_list=regions_list)

    """plot_significant_electrode_rms"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    lowest_p_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/lowest_p_all_channels.pkl'
    regions_list = [['Left-Amygdala', 'ctx-rh-precentral', 'Right-Thalamus-Proper'],
                    ['Left-Hippocampus', 'ctx-rh-inferiorparietal'],
                    ['ctx-lh-parsopercularis', 'Left-Hippocampus'],
                    ['Left-Amygdala', 'Left-Thalamus-Proper'],
                    ['Right-Amygdala', 'Right-Hippocampus']]  # one list per condition
    # regions_list = [['Left-Amygdala', 'Right-Amygdala', 'ctx-rh-precentral', 'Right-Thalamus-Proper', 'Left-Thalamus-Proper', 'Left-Hippocampus', 'Right-Hippocampus', 'ctx-rh-inferiorparietal', 'ctx-lh-parsopercularis'],
    #                 ['Left-Amygdala', 'Right-Amygdala', 'ctx-rh-precentral', 'Right-Thalamus-Proper', 'Left-Thalamus-Proper', 'Left-Hippocampus', 'Right-Hippocampus', 'ctx-rh-inferiorparietal', 'ctx-lh-parsopercularis'],
    #                 ['Left-Amygdala', 'Right-Amygdala', 'ctx-rh-precentral', 'Right-Thalamus-Proper', 'Left-Thalamus-Proper', 'Left-Hippocampus', 'Right-Hippocampus', 'ctx-rh-inferiorparietal', 'ctx-lh-parsopercularis'],
    #                 ['Left-Amygdala', 'Right-Amygdala', 'ctx-rh-precentral', 'Right-Thalamus-Proper', 'Left-Thalamus-Proper', 'Left-Hippocampus', 'Right-Hippocampus', 'ctx-rh-inferiorparietal', 'ctx-lh-parsopercularis'],
    #                 ['Left-Amygdala', 'Right-Amygdala', 'ctx-rh-precentral', 'Right-Thalamus-Proper', 'Left-Thalamus-Proper', 'Left-Hippocampus', 'Right-Hippocampus', 'ctx-rh-inferiorparietal', 'ctx-lh-parsopercularis']]  # one list per condition
    # regions_list = [
    #     ['Left-Amygdala', 'Right-Amygdala', 'ctx-rh-precentral', 'Right-Thalamus-Proper', 'Left-Thalamus-Proper', 'Left-Hippocampus', 'Right-Hippocampus', 'ctx-rh-inferiorparietal', 'ctx-lh-inferiorparietal', 'ctx-lh-temporalpole', 'ctx-lh-parahippocampal', 'ctx-lh-parsopercularis'],
    #     ['Left-Amygdala', 'Right-Amygdala', 'ctx-rh-precentral', 'Right-Thalamus-Proper', 'Left-Thalamus-Proper', 'Left-Hippocampus', 'Right-Hippocampus', 'ctx-rh-inferiorparietal', 'ctx-lh-inferiorparietal', 'ctx-lh-temporalpole', 'ctx-lh-parahippocampal', 'ctx-lh-parsopercularis'],
    #     ['Left-Amygdala', 'Right-Amygdala', 'ctx-rh-precentral', 'Right-Thalamus-Proper', 'Left-Thalamus-Proper', 'Left-Hippocampus', 'Right-Hippocampus', 'ctx-rh-inferiorparietal', 'ctx-lh-inferiorparietal', 'ctx-lh-temporalpole', 'ctx-lh-parahippocampal', 'ctx-lh-parsopercularis'],
    #     ['Left-Amygdala', 'Right-Amygdala', 'ctx-rh-precentral', 'Right-Thalamus-Proper', 'Left-Thalamus-Proper', 'Left-Hippocampus', 'Right-Hippocampus', 'ctx-rh-inferiorparietal', 'ctx-lh-inferiorparietal', 'ctx-lh-temporalpole', 'ctx-lh-parahippocampal', 'ctx-lh-parsopercularis'],
    #     ['Left-Amygdala', 'Right-Amygdala', 'ctx-rh-precentral', 'Right-Thalamus-Proper', 'Left-Thalamus-Proper', 'Left-Hippocampus', 'Right-Hippocampus', 'ctx-rh-inferiorparietal', 'ctx-lh-inferiorparietal', 'ctx-lh-temporalpole', 'ctx-lh-parahippocampal', 'ctx-lh-parsopercularis'],
    #     ]  # one list per condition
    first_parcel_label = True
    only_signif_channels = False
    # plot_significant_electrode_rms(data_path, behavior_path, lowest_p_path, first_parcel_label=True, regions_list=regions_list, only_signif_channels=only_signif_channels)

    """plot_all_electrode_scalar"""
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    first_parcel_label = "advanced"
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels'
    analysis_name = 'rms'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_theta_power_all_channels'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_high_gamma_power_all_channels'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_high_gamma_power_all_channels_80_150'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_high_gamma_power_all_channels_120_140'
    # analysis_name = 'power'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_channels_80_150'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_channels_120_140'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_80_150'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140'
    # analysis_name = 'pac'
    # plot_all_electrode_scalar(data_path, behavior_path, first_parcel_label=first_parcel_label, analysis_name=analysis_name)
    # plot_all_electrode_scalar_only_correct_incorrect(data_path, behavior_path, first_parcel_label=first_parcel_label, analysis_name=analysis_name)

    """with selected regions"""
    regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-lh-middletemporal',
               'ctx-lh-inferiortemporal', 'ctx-lh-temporalpole', 'ctx-lh-entorhinal', 'ctx-lh-parahippocampal',
               'Left-Hippocampus', 'Left-Amygdala',
               'ctx-rh-transversetemporal', 'ctx-rh-superiortemporal', 'ctx-rh-middletemporal',
               'ctx-rh-inferiortemporal', 'ctx-rh-temporalpole', 'ctx-rh-entorhinal', 'ctx-rh-parahippocampal',
               'Right-Hippocampus', 'Right-Amygdala']
    # plot_all_electrode_scalar(data_path, behavior_path, first_parcel_label=first_parcel_label, analysis_name=analysis_name, regions=regions, remove_outliers=True)
    # plot_all_electrode_scalar_colored_regions(data_path, behavior_path, first_parcel_label=first_parcel_label, analysis_name=analysis_name, regions=regions, remove_outliers=True)

    """save_rms_contrasts_stat_individual_electrode"""
    # save_rms_contrasts_stat_individual_electrode(data_path, behavior_path, analysis_name='rms')

    """plot_individual_electrode_scalar"""
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_theta_power_all_channels'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_high_gamma_power_all_channels'
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    first_parcel_label = "advanced"
    # first_parcel_label = "True"
    analysis_name = 'rms'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_high_gamma_power_all_channels_80_150'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_high_gamma_power_all_channels_120_140'
    # analysis_name = 'power'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_channels_80_150'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_channels_120_140'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_80_150'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140_narrow_diff_baseline'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140_diff_baseline'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140_scaling_phase'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140_scaling_phase_no_baseline'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140_no_baseline'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_diff_baseline'
    # analysis_name = 'pac'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_phase_coherence_stim_freq_all_channels'
    # analysis_name = 'phase_coherence'
    regions = ['Left-Hippocampus', 'Right-Hippocampus']
    # regions = ['Left-Hippocampus']
    # regions = ['Right-Hippocampus']
    # regions = ['ctx-lh-inferiortemporal']
    # regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal']
    # regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal', 'ctx-rh-superiortemporal']
    # regions = ['Left-Hippocampus', 'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal']
    # regions = ['Left-Hippocampus', 'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal', 'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal']  # medial temporal
    # regions = ['Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal', 'ctx-lh-entorhinal', 'Right-Amygdala', 'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal']
    # regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal', 'ctx-rh-superiortemporal', 'ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus', 'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal', 'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal']  # all temporal lobe
    # regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal',
    #            'ctx-lh-middletemporal', 'Left-Hippocampus',
    #            'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal',
    #            'ctx-lh-entorhinal']  # all temporal lobe left
    # regions = ['ctx-rh-transversetemporal',
    #            'ctx-rh-superiortemporal','ctx-rh-middletemporal',
    #            'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal',
    #            'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal']  # all temporal lobe right
    # regions = ['Left-Hippocampus', 'Left-Amygdala', 'Right-Amygdala', 'Right-Thalamus-Proper', 'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal', 'ctx-rh-lingual', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal', 'ctx-lh-middletemporal', 'ctx-lh-insula']
    regions = ['Left-Amygdala', 'Right-Amygdala']
    # regions = ['ctx-lh-parahippocampal', 'ctx-rh-parahippocampal']
    # regions = ['ctx-lh-middletemporal', 'ctx-rh-middletemporal']
    # regions = ['ctx-lh-temporalpole', 'ctx-rh-temporalpole']
    # regions = ['ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal']
    # regions = ['ctx-lh-transversetemporal']
    # regions = ['ctx-lh-transversetemporal', 'ctx-rh-transversetemporal']
    # regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis', 'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal']
    # regions = ['ctx-rh-rostralmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-rh-parstriangularis', 'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-lateralorbitofrontal']
    # regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis', 'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal',
    #            'ctx-rh-rostralmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-rh-parstriangularis', 'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-lateralorbitofrontal']
    # regions = ['ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus',
    #            'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal',
    #            'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal',
    #            'ctx-rh-inferiortemporal']  # regions of temporal lobe that show a power contrast
    plot_individual_electrode_scalar(data_path, behavior_path, regions, first_parcel_label=first_parcel_label, analysis_name=analysis_name)
    """plot_individual_subject_scalar"""
    # plot_individual_subject_scalar(data_path, behavior_path, regions, first_parcel_label=first_parcel_label, analysis_name=analysis_name)
    """plot_individual_electrode_scalar_windows"""
    # plot_individual_electrode_scalar_windows(data_path, behavior_path, regions, first_parcel_label=first_parcel_label, analysis_name=analysis_name, remove_outliers=True)
    """plot_individual_subject_scalar_windows"""
    # plot_individual_subject_scalar_windows(data_path, behavior_path, regions, first_parcel_label=first_parcel_label, analysis_name=analysis_name)

    """loop_plot_scalar_window_regions"""
    # regions_lists = [
    #                  ['Left-Hippocampus', 'Right-Hippocampus'],
    #                  ['Left-Hippocampus'],
    #                  ['Right-Hippocampus'],
    #                  ['Left-Amygdala', 'Right-Amygdala'],
    #                  ['Left-Amygdala'],
    #                  ['Right-Amygdala'],
    #                  ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal', 'ctx-rh-superiortemporal'],
    #                  ['Left-Hippocampus', 'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal', 'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal'],  # medial temporal
    #                  ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal', 'ctx-rh-superiortemporal', 'ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus', 'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal', 'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal']  # all temporal lobe
    #                  ]
    # significant regions:
    regions_lists = [
        ['Left-Hippocampus', 'Left-Amygdala',
         'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal'],
        ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal', 'ctx-rh-superiortemporal',
         'ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus', 'Left-Amygdala',
         'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal', 'ctx-lh-entorhinal',
         'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal', 'ctx-rh-temporalpole',
         'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal'],
        ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal', 'ctx-rh-superiortemporal',
         'ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus', 'Left-Amygdala',
         'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal', 'ctx-lh-entorhinal',
         'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal', 'ctx-rh-temporalpole',
         'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal'],  # all temporal lobe

    ]
    # loop_plot_scalar_window_regions(data_path, behavior_path, regions_lists)

    """plot_individual_electrode_power_stim_freq"""
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_stim_freq_all_channels'
    first_parcel_label = "advanced"
    regions = ['Left-Hippocampus']
    # regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal']
    # regions = ['ctx-rh-inferiorparietal', 'ctx-lh-inferiorparietal', 'ctx-rh-insula', 'ctx-lh-insula']
    # regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis', 'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal']
    # regions = ['ctx-rh-rostralmiddlefrontal', 'ctx-lh-rostralmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-rh-parstriangularis', 'ctx-lh-parstriangularis', 'ctx-rh-parsopercularis', 'ctx-lh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-lh-parsorbitalis', 'ctx-rh-lateralorbitofrontal', 'ctx-lh-lateralorbitofrontal']
    # plot_individual_electrode_power_stim_freq(data_path, regions, first_parcel_label=first_parcel_label)

    """save_individual_electrode_scalar_3d"""
    # save_individual_electrode_scalar_3d(data_path, behavior_path, data_path, analysis_name='rms')
    # save_individual_electrode_scalar_correct_incorrect_3d(data_path, behavior_path, data_path, analysis_name='rms')
    # save_individual_electrode_scalar_novelty_retrieval_3d(data_path, behavior_path, data_path, analysis_name='rms')

    """plot_individual_electrode_scalar_temporal"""
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_ma_rms_all_channels'
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    first_parcel_label = 'advanced'
    analysis_name = 'ma_rms'
    # regions = ['Left-Hippocampus', 'Right-Hippocampus']
    # regions = ['Right-Hippocampus']
    regions = ['Left-Hippocampus']
    # regions = ['Left-Hippocampus', 'Left-Amygdala', 'ctx-lh-parahippocampal']
    # regions = ['Left-Amygdala']
    # regions = ['ctx-lh-parahippocampal']
    # regions = ['ctx-lh-temporalpole']
    # plot_individual_electrode_scalar_temporal(data_path, behavior_path, regions, first_parcel_label=first_parcel_label, analysis_name=analysis_name)
    # plot_individual_electrode_scalar_temporal_regions_comparison(data_path, behavior_path, regions, first_parcel_label=first_parcel_label, analysis_name=analysis_name)
    regions = [['Left-Hippocampus', 'Right-Hippocampus'], ['Left-Amygdala', 'Right-Amygdala']]
    plot_individual_electrode_scalar_temporal_regions_comparison_one_figure(data_path, behavior_path, regions, first_parcel_label=first_parcel_label, analysis_name=analysis_name)

    """decode_surprise_from_comodulogram"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram_low_resolution_all_channels'
    lowest_p_path = '/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/lowest_p_all_channels.pkl'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # regions = ['Left-Hippocampus', 'ctx-lh-parsopercularis', 'ctx-rh-inferiorparietal', 'Left-Amygdala', 'Right-Hippocampus', 'Left-Thalamus-Proper', 'Right-Thalamus-Proper', 'Right-Amygdala']  # one list for all conditions
    regions = ['Left-Hippocampus']
    # regions = ['Right-Hippocampus']
    first_parcel_label = True
    # decode_surprise_from_comodulogram(data_path, behavior_path, lowest_p_path, first_parcel_label=first_parcel_label, regions=regions)

    """encode_scalar_from_surprise_and_save_data"""
    data_path_list = ['/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels',
                      '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_theta_power_all_channels',
                      '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_high_gamma_power_all_channels_120_140',
                      '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_channels_120_140',

                      # '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140',
                      # '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140_narrow_diff_baseline',
                      # '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140_no_baseline',
                      "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_80_150_diff_baseline",

                      '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_channels_4',
                      '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_channels_5',
                      '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_channels_6',
                      '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_channels_7',
                      '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_channels_8',
                      '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_theta_bands_all_channels_120_140',
                      '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_stim_freq_all_channels',
                      '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_phase_coherence_stim_freq_all_channels'
                      ]
    analysis_name_list = ['rms', 'theta', 'high_gamma', 'pac', 'pac_stim_freq', 'pac4', 'pac5', 'pac6', 'pac7', 'pac8', 'phase_coherence_stim_freq']
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal']
    regions = ['Left-Hippocampus']
    # regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis',
    #            'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal']
    # regions = None
    # regions = ['Right-Hippocampus']
    # regions = ['Left-Hippocampus', 'Right-Hippocampus']
    # regions = ['ctx-rh-transversetemporal', 'ctx-lh-transversetemporal']
    # first_parcel_label = True
    first_parcel_label = "advanced"
    # encode_scalar_from_surprise_and_save_data(data_path_list, analysis_name_list, behavior_path, first_parcel_label=first_parcel_label, regions=regions)

    """save_histogram_with_behavior"""
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140'
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal']
    regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis',
               'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal']
    # regions = ['Left-Hippocampus']
    first_parcel_label = "advanced"
    # save_histogram_with_behavior(data_path, behavior_path, first_parcel_label=first_parcel_label, regions=regions)

    """pac_and_surprise"""
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/encoding/encoding_data_left_hippocampus.csv'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/encoding/encoding_data_left_hippocampus_no_baseline.csv'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/encoding/encoding_data_left_hippocampus_diff_baseline.csv'
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/encoding/encoding_data_left_hippocampus_80_150.csv'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/encoding/encoding_data_heschl.csv'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/encoding/encoding_data_left_prefrontal.csv'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/encoding/encoding_data_left_stg.csv'
    # pac_and_surprise(data_path)
    """phase_coherence_and_surprise"""
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/encoding/encoding_data_left_hippocampus.csv'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/encoding/encoding_data_heschl.csv'
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/encoding/encoding_data_left_hippocampus_80_150.csv'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/encoding/encoding_data_left_prefrontal.csv'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/encoding/encoding_data_left_stg.csv'
    # phase_coherence_and_surprise(data_path)

    """phase_coherence_all_frequencies"""
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_phase_coherence_all_freqs'
    regions = ['Left-Hippocampus']
    # regions = ['Right-Hippocampus']
    # regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis',
    #            'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal']
    # regions = ['ctx-rh-rostralmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-rh-parstriangularis', 'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-lateralorbitofrontal']
    regions = ['ctx-rh-transversetemporal', 'ctx-lh-transversetemporal']
    first_parcel_label = "advanced"
    # phase_coherence_all_frequencies(data_path, regions, first_parcel_label=first_parcel_label)
    # regions = ['Left-Hippocampus']
    # phase_coherence_all_frequencies(data_path, regions, first_parcel_label=first_parcel_label)
    # regions = ['Right-Hippocampus']
    # phase_coherence_all_frequencies(data_path, regions, first_parcel_label=first_parcel_label)
    # regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis',
    #            'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal']
    # phase_coherence_all_frequencies(data_path, regions, first_parcel_label=first_parcel_label)

    '''pac_all_frequencies'''
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_freqs_with_baseline'
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_freqs_fixed_window'
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_freqs_scaling_window'
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_freqs_narrow'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_freqs_narrow_with_baseline'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_freqs_baseline_only'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_freqs_scaling_phase_fixed_window_no_baseline'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_freqs_diff_baseline'
    # pac_all_frequencies(data_path, regions, first_parcel_label=first_parcel_label)

    """pac_all_frequencies_contrast"""
    # pac_all_frequencies_contrast(data_path, behavior_path, regions, first_parcel_label=first_parcel_label)

    """plot_all_frequencies_individual_channels"""
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_freqs_narrow'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_freqs_narrow_with_baseline'
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_freqs_diff_baseline'
    # regions = ['Left-Hippocampus']
    # regions = ['Right-Hippocampus']
    # regions = ['Left-Hippocampus', 'Right-Hippocampus']
    regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis',
               'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal',
               'ctx-rh-rostralmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-rh-parstriangularis',
               'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-lateralorbitofrontal']
    region_name = 'prefrontal'
    # regions = ['ctx-rh-transversetemporal', 'ctx-lh-transversetemporal']
    first_parcel_label = "advanced"
    analysis_name = 'pac'
    # plot_all_frequencies_individual_channels(data_path, regions, region_name, first_parcel_label, analysis_name, freqs_name='high_freq')
    # plot_all_frequencies_individual_channels_by_rate(data_path, regions, region_name, first_parcel_label, analysis_name, freqs_name='high_freq')
    analysis_name = 'power'
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs'
    # plot_all_frequencies_individual_channels(data_path, regions, region_name, first_parcel_label, analysis_name, freqs_name='freq')
    # plot_all_frequencies_individual_channels_by_rate(data_path, regions, region_name, first_parcel_label, analysis_name, freqs_name='freq')
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs_below_10'
    # plot_all_frequencies_individual_channels(data_path, regions, first_parcel_label, analysis_name, freqs_name='freq')
    analysis_name = 'phase_coherence'
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_phase_coherence_all_freqs'
    # plot_all_frequencies_individual_channels(data_path, regions, region_name, first_parcel_label, analysis_name, freqs_name='low_freq')
    # plot_all_frequencies_individual_channels_by_rate(data_path, regions, region_name, first_parcel_label, analysis_name, freqs_name='low_freq')

    """plot_all_frequencies_contrast_individual_channels"""
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_freqs_narrow'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_freqs_narrow_with_baseline'
    # data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_all_freqs_diff_baseline'
    analysis_name = 'pac'
    # plot_all_frequencies_contrast_individual_channels(data_path, behavior_path, regions, region_name, first_parcel_label, analysis_name, freqs_name='high_freq')
    analysis_name = 'power'
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs'
    # plot_all_frequencies_contrast_individual_channels(data_path, behavior_path, regions, region_name, first_parcel_label, analysis_name, freqs_name='freq')
    analysis_name = 'phase_coherence'
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_phase_coherence_all_freqs'
    # plot_all_frequencies_contrast_individual_channels(data_path, behavior_path, regions, region_name, first_parcel_label, analysis_name, freqs_name='low_freq')

    """plot_comodulogram_subjects"""
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram_all_freqs'
    regions = ['Left-Hippocampus', 'Right-Hippocampus']
    regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis',
               'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal',
               'ctx-rh-rostralmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-rh-parstriangularis',
               'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-lateralorbitofrontal']
    # regions = ['ctx-rh-transversetemporal', 'ctx-lh-transversetemporal']
    region_name = 'prefrontal'
    first_parcel_label = "advanced"
    # plot_comodulogram_subjects(data_path, regions, region_name, first_parcel_label)
    """plot_comodulogram_contrasts_subjects"""
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # plot_comodulogram_contrasts_subjects(data_path, behavior_path, regions, region_name, first_parcel_label)

    """plot_theta_beta_and_channel_positions"""
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs'
    regions = ['Left-Hippocampus', 'Right-Hippocampus']
    # regions = ['Left-Hippocampus']
    # regions = ['Right-Hippocampus']
    # regions = ['ctx-rh-transversetemporal', 'ctx-lh-transversetemporal']
    regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis',
               'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal',
               'ctx-rh-rostralmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-rh-parstriangularis',
               'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-lateralorbitofrontal']
    # plot_theta_beta_and_channel_positions(data_path, regions, first_parcel_label, analysis_name='power', freqs_name='freq', subject_colors=True)

    """plot_scalar_and_channel_positions"""
    data_path = "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels"
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    regions = ['Left-Hippocampus']
    regions = ['Right-Hippocampus']
    regions = ['Left-Hippocampus', 'Right-Hippocampus']
    regions = ['Left-Hippocampus', 'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal']
    regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal', 'ctx-rh-superiortemporal', 'ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus', 'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal']  # all temporal lobe
    regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal', 'ctx-rh-superiortemporal']
    # regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis',
    #            'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal',
    #            ]
    regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis',
               'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal',
               'ctx-rh-rostralmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-rh-parstriangularis',
               'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-lateralorbitofrontal']
    first_parcel_label = "advanced"
    analysis_name = 'rms'
    # plot_scalar_and_channel_positions(data_path, behavior_path, regions, first_parcel_label, analysis_name,
    #                                   subject_colors=True,
    #                                   bad_epochs_path='/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs/bad_epochs/bad_epochs_hippocampus.npy')

    """study_epoch_consistency_one_subject"""
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs'
    # study_epoch_consistency_one_subject(data_path, regions, first_parcel_label, analysis_name="power", freqs_name='freq')

    """pac_at_peak_theta"""
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    regions = ['Left-Hippocampus', 'Right-Hippocampus']
    # regions = ['Left-Hippocampus']
    # regions = ['Right-Hippocampus']
    # regions = ['ctx-rh-transversetemporal', 'ctx-lh-transversetemporal']
    regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis',
               'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal',
               'ctx-rh-rostralmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-rh-parstriangularis',
               'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-lateralorbitofrontal']
    region_name = 'prefrontal'
    first_parcel_label = "advanced"
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_power_all_freqs'
    como_folder = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_comodulogram_all_freqs'
    save_folder = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/pac_individual_phase'
    # pac_at_peak_theta(data_path, como_folder, save_folder, regions, region_name, first_parcel_label)
    # pac_at_peak_theta_contrasts(data_path, como_folder, behavior_path, save_folder, regions, region_name, first_parcel_label)

    """pac_ushape_by_region"""
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/encoding/encoding_data_all_channels.csv'
    # pac_ushape_by_region(data_path)

    """print_all_regions"""
    # print_all_regions('/home/mathieupvc/These/data/meg_maestro/seeg_trials', '/home/mathieupvc/freesurfer/subjects')

    """save_note_evoked_gamma"""
    trial_epochs_path = "/home/mathieu/These/data/meg_maestro/seeg_trials"
    note_level_epochs = "/home/mathieu/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta"
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # save_path = "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/note_evoked_power_left_hippocampus"
    save_path = "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/note_evoked_power_left_superior_temporal"
    # regions = ['Left-Hippocampus']
    regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal']
    first_parcel_label = 'advanced'
    # save_note_evoked_gamma(trial_epochs_path, note_level_epochs, behavior_path, save_path, regions,
    #                            first_parcel_label, tmin=2, tmax=7, min_time_interval=0.050, only_repeated=True, freq_bands=[120, 140])

    """note_evoked_gamma_regression"""
    save_path = "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/note_evoked_power_left_hippocampus"
    save_path = "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/note_evoked_power_left_superior_temporal"
    # save_path = "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/note_evoked_power"
    # note_evoked_gamma_regression(save_path)

    """note_evoked_gamma_analysis"""
    save_path = "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/note_evoked_power_left_hippocampus"
    save_path = "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/note_evoked_power_left_superior_temporal"
    # save_path = "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/note_evoked_power"
    # note_evoked_gamma_analysis(save_path)

    """note_evoked_gamma_analysis_phase"""
    save_path = "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/note_evoked_power_left_hippocampus"
    save_path = "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/note_evoked_power_left_superior_temporal"
    # save_path = "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/note_evoked_power"
    # note_evoked_gamma_analysis_phase(save_path)

    """note_evoked_gamma_analysis_phase_indiv_chan"""
    save_path = "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/note_evoked_power_left_hippocampus"
    # save_path = "/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/note_evoked_power_left_superior_temporal"
    # note_evoked_gamma_analysis_phase_indiv_chan(save_path)

    """amplitude_by_phase_histogram_plot"""
    csv_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140/histogram/histogram_left_hippocampus.csv'
    # csv_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140/histogram/histogram_left_prefrontal.csv'
    # csv_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_pac_stim_freq_all_channels_120_140/histogram/histogram_left_stg.csv'
    # amplitude_by_phase_histogram_plot(csv_path)
