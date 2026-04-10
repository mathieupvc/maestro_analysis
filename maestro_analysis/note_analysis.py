import os
import copy
from pathlib import Path
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import scipy.stats
from joblib import Parallel, delayed
import time

import statsmodels.api as sm

from math import ceil, floor

from maestro_analysis.utils import epoching_maestro_note_level, epoching_maestro_note_level_time_frequency, fixed_golden_aspect_ratio
from maestro_analysis.trf_analysis import condition_list_from_metadata
from maestro_analysis.stats import permutation_cluster_test_adapted, permutation_cluster_test_adapted_seeg
from maestro_analysis.signal_functions import butter_lowpass_filter

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression, RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, accuracy_score

import mne
from mne.decoding import LinearModel, Scaler
from mne.stats import spatio_temporal_cluster_1samp_test
from mne.channels import find_ch_adjacency
from mne.preprocessing import ICA, read_ica

import matplotlib
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


def erp_note_all_subjects(data_path, save_path, midi_folder, surprise_dir, entropy_dir, channel_types=None, wav_folder="/home/mathieu/Data/meg_maestro/exp_stim_meg", times=0.100, save_format='png'):

    if channel_types is None:
        channel_types = ['grad']

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    fig1, axes1 = plt.subplots(nrows=6, ncols=5, figsize=(20, 12), dpi=300, sharex=True)
    fig2, axes2 = plt.subplots(nrows=6, ncols=10, figsize=(20, 12), dpi=300, gridspec_kw=dict(width_ratios=[5, 0.5] * 5))

    evoked_list = []

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        epochs = epoching_maestro_note_level(raw_path, wav_folder, midi_folder, surprise_dir, entropy_dir, t_max=0.5, time_bin=0.01, multiple_notes='max')

        epochs.pick(channel_types)
        # epochs.crop(tmin=-0.2, tmax=0.5)
        listening_epochs = epochs['note_on']
        del epochs

        # ERF
        listening_evoked = listening_epochs.average()
        evoked_list.append(listening_evoked)
        listening_evoked.plot(show=False, axes=axes1.flat[i], titles=dict(grad='', mag=''), window_title='', spatial_colors=True, zorder='std')
        listening_evoked.plot_topomap(show=False, axes=[axes2.flat[2*i], axes2.flat[2*i+1]], times=times, colorbar=True)
        axes1.flat[i].set_title(None)
        if i < len(subject_folders) - 5:
            axes1.flat[i].set_xlabel(None)
        if i % 5 != 0:
            axes1.flat[i].set_ylabel(None)

    for i in range(len(subject_folders), 30):
        axes1.flat[i].axis('off')
        axes2.flat[2*i].axis('off')
        axes2.flat[2*i+1].axis('off')

    # fig.tight_layout()
    #plt.show()
    fig1.savefig(os.path.join(save_path, "all_note_erp." + save_format))
    fig2.savefig(os.path.join(save_path, "all_note_erp" + str(times*1000) + "ms_topo." + save_format))

    evoked = mne.combine_evoked(evoked_list, 'equal') # Plot the across subjects topomap
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 12), dpi=300, gridspec_kw=dict(width_ratios=[5, 0.5]))
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelsize'] = 'large'
    evoked.plot_topomap(show=False, axes=[axes[0], axes[1]], times=times, colorbar=True)
    fig.savefig(os.path.join(save_path, "averaged_subject_note_erp" + str(times*1000) + "ms_topo." + save_format))
    fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(20, 12), dpi=300)
    evoked.plot(show=False, axes=axes2, titles=dict(grad='', mag=''), window_title='',
                          spatial_colors=True, zorder='std')
    fig2.savefig(os.path.join(save_path, "averaged_subject_note_erp." +save_format))

    return


def erp_note_all_subjects_by_rate(data_path, save_path, midi_folder, surprise_dir, entropy_dir, channel_types=None, wav_folder="/home/mathieu/Data/meg_maestro/exp_stim_meg", times=0.100, save_format='png'):

    if channel_types is None:
        channel_types = ['grad']

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    conditions_list = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']
    evoked_list = [[] for i in range(len(conditions_list))]

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        epochs = epoching_maestro_note_level(raw_path, wav_folder, midi_folder, surprise_dir, entropy_dir, t_max=0.5, time_bin=0.01, multiple_notes='max')

        epochs.pick(channel_types)
        # epochs.crop(tmin=-0.2, tmax=0.5)
        listening_epochs = epochs['note_on']
        del epochs

        # ERF
        for c, condition in enumerate(conditions_list):
            to_keep = condition_list_from_metadata(listening_epochs.metadata, [condition])
            listening_evoked = listening_epochs[to_keep].average()
            evoked_list[c].append(listening_evoked)

    evoked = [mne.combine_evoked(evoked, 'equal') for evoked in evoked_list] # Plot the across subjects topomap
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 12), dpi=300)
    colors = [matplotlib.cm.get_cmap('GnBu')(alpha) for alpha in [0.3, 0.5, 0.7, 0.85, 0.99]]
    for c, condition in enumerate(conditions_list):
        evoked_array = evoked[c].get_data()
        evoked_array = np.sqrt(np.mean(evoked_array**2, axis=0))  # RMS
        t = np.linspace(-0.2, 0.5, evoked_array.shape[0])
        axes.plot(t, evoked_array, color=colors[c])
    axes.legend(conditions_list)

    fig.savefig(os.path.join(save_path, "averaged_subject_note_erp_by_rate." +save_format))

    return


def erp_all_subjects_from_epochs(epochs_path, tmin=1, min_time_interval=None):
    """
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times, rates)
    :param epochs: used for times and info
    :param tmin:
    :return:
    """
    plt.switch_backend('TkAgg')

    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))

    evoked_list = []
    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # keep only notes with min time interval
        if min_time_interval is not None:
            print('Nombre de notes:', len(epochs.metadata))
            epochs = epochs[epochs.metadata['time_interval'] >= min_time_interval]
            print('Nombre de notes:', len(epochs.metadata))

        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]

        evoked_list.append(epochs.average())

    evoked = mne.combine_evoked(evoked_list, 'equal')

    fig, axes = plt.subplots(2, 1)

    evoked.plot(show=False, axes=axes[0], zorder='std')
    axes[0].axvline(0, linestyle='--', color='k', zorder=0)
    fixed_golden_aspect_ratio(axes[0])
    axes[0].set_title("Note-level ERP")

    evoked.plot(show=False, axes=axes[1], zorder='std', gfp='only')
    axes[1].axvline(0, linestyle='--', color='k', zorder=0)
    fixed_golden_aspect_ratio(axes[1])
    axes[1].set_title("Note-level RMS")

    plt.show()

    return


def erp_all_subjects_from_epochs_by_rate_gfp(epochs_path, rate_list, tmin=1, min_time_interval=None):
    """
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times, rates)
    :param epochs: used for times and info
    :param tmin:
    :return:
    """
    plt.switch_backend('TkAgg')

    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))

    evoked_list = [[] for r in range(len(rate_list))]
    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[epochs.metadata['time_interval'] >= min_time_interval]

        metadata = epochs.metadata
        metadata['rate'] = 0  # add rate from stim name
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'rate'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[2][0:3])

        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressors = metadata[metadata['trial_time'].to_numpy() >= tmin]

        for r, rate in enumerate(rate_list):
            rate_epochs = epochs[regressors['rate'].to_numpy() == rate]
            evoked_list[r].append(rate_epochs.average())

    evoked = [mne.combine_evoked(evoked, 'equal') for evoked in evoked_list]

    fig, axes = plt.subplots(1, 1)
    colors = [matplotlib.cm.get_cmap('YlOrRd')(alpha) for alpha in [0.2, 0.4, 0.6, 0.8, 0.99]]
    for r, rate in enumerate(rate_list):
        evoked_array = evoked[r].get_data()
        axes.plot(evoked[r].times, np.std(evoked_array, axis=0), color=colors[r])

    axes.legend(rate_list)
    axes.axvline(0, linestyle='--', color='k', zorder=0)
    axes.set_xlabel('Time (s)')
    axes.set_title('GFP by note rate')
    fixed_golden_aspect_ratio(axes)

    plt.show()

    return

def erp_all_subjects_from_epochs_by_rate(epochs_path, rate_list, tmin=2, tmax=7, min_time_interval=None):
    """
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times, rates)
    :param epochs: used for times and info
    :param tmin:
    :return:
    """
    plt.switch_backend('TkAgg')

    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))

    evoked_list = [[] for r in range(len(rate_list))]
    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[epochs.metadata['time_interval'] >= min_time_interval]

        metadata = epochs.metadata
        metadata['rate'] = 0  # add rate from stim name
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'rate'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[2][0:3])

        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]

        regressors = epochs.metadata

        for r, rate in enumerate(rate_list):
            rate_epochs = epochs[regressors['rate'].to_numpy() == rate]
            evoked_list[r].append(rate_epochs.average())

    evoked = [mne.combine_evoked(evoked, 'equal') for evoked in evoked_list]

    fig, axes = plt.subplots(2, 3, sharey=True, squeeze=False)
    for r, rate in enumerate(rate_list):

        evoked[r].plot(show=False, axes=axes.flat[r], zorder='std')
        axes.flat[r].axvline(0, linestyle='--', color='k', zorder=0)

        # dashed gray line at expected previous and next notes
        period = 1
        while axes.flat[r].get_xlim()[0] < - period / rate:
            axes.flat[r].axvline(- period / rate, linestyle='--', color='gray', zorder=0, linewidth=0.5)
            period += 1
        period = 1
        while axes.flat[r].get_xlim()[1] > period / rate:
            axes.flat[r].axvline(period / rate, linestyle='--', color='gray', zorder=0, linewidth=0.5)
            period += 1

        fixed_golden_aspect_ratio(axes.flat[r])
        axes.flat[r].set_title(str(rate) + " Hz")

    for ax in range(1, 6 - len(rate_list) + 1):
        axes.flat[-ax].remove()
    fig.suptitle("Note-level ERP")

    plt.show()

    return


def save_erp_by_quantile(epochs_path, feature, shift, save_path, tmin=2, tmax=7, min_time_interval=None):
    """ For a given feature and quantiles of that feature.
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times, rates)
    :param epochs: used for times and info
    :param tmin:
    :return:
    """
    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))

    evoked_array = np.zeros((len(epochs_list), 102, 201, 10))
    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[epochs.metadata['time_interval'] >= min_time_interval]

        # add shifted feature
        metadata = epochs.metadata
        if shift:
            metadata['shifted_' + feature] = 0  # add shifted entropy
            for i_note in range(metadata.shape[0]):
                if (i_note + 1 < len(metadata)) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note + 1], 'stim_name']):
                    metadata.loc[metadata.index[i_note], 'shifted_' + feature] = metadata.loc[metadata.index[i_note + 1], feature]

        if shift:
            regressor = metadata['shifted_' + feature]
        else:
            regressor = metadata[feature]

        # remove onset response
        regressor = regressor[epochs.metadata['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressor = regressor[epochs.metadata['trial_time'].to_numpy() <= tmax]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]

        regressor = regressor.to_numpy()

        # quantiles_list = np.quantile(regressor, [0.2, 0.4, 0.6, 0.8, 1])
        print(np.log(np.linspace(1, 11, 11)) / np.log(11))
        quantiles_list = np.quantile(regressor, (np.log(np.linspace(1, 11, 11)) / np.log(11))[1:])

        for q, quantile_value in enumerate(quantiles_list):
            if q == 0:
                quantile_epochs = epochs[regressor < quantiles_list[q]]
            else:
                quantile_epochs = epochs[(regressor >= quantiles_list[q-1]) & (regressor < quantiles_list[q])]
            evoked_array[i, :, :, q] = quantile_epochs.average().get_data()

    np.save(os.path.join(save_path, f'erp_{feature}_quantiles.npy'), evoked_array)

    return

def erp_all_subjects_from_epochs_by_quantile(epochs_path, feature, shift, quantiles, tmin=2, tmax=7, min_time_interval=None):
    """ For a given feature and quantiles of that feature.
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times, rates)
    :param epochs: used for times and info
    :param tmin:
    :return:
    """
    plt.switch_backend('TkAgg')

    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))

    evoked_list = [[] for _ in range(5)]
    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[epochs.metadata['time_interval'] >= min_time_interval]

        # add shifted feature
        metadata = epochs.metadata
        if shift:
            metadata['shifted_' + feature] = 0  # add shifted entropy
            for i_note in range(metadata.shape[0]):
                if (i_note + 1 < len(metadata)) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note + 1], 'stim_name']):
                    metadata.loc[metadata.index[i_note], 'shifted_' + feature] = metadata.loc[metadata.index[i_note + 1], feature]

        if shift:
            regressor = metadata['shifted_' + feature]
        else:
            regressor = metadata[feature]

        # remove onset response
        regressor = regressor[epochs.metadata['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressor = regressor[epochs.metadata['trial_time'].to_numpy() <= tmax]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]

        regressor = regressor.to_numpy()

        if quantiles:
            quantiles_list = np.quantile(regressor, [0.2, 0.4, 0.6, 0.8, 1])
        else:
            quantiles_list = [regressor.min() + q * (regressor.max() - regressor.min()) / 5 for q in range(1, 6)]
        print(quantiles_list)

        for q, quantile_value in enumerate(quantiles_list):
            if q == 0:
                quantile_epochs = epochs[regressor < quantiles_list[q]]
            else:
                quantile_epochs = epochs[(regressor >= quantiles_list[q-1]) & (regressor < quantiles_list[q])]
            evoked_list[q].append(quantile_epochs.average())

    evoked = [mne.combine_evoked(evoked, 'equal') for evoked in evoked_list]

    fig, axes = plt.subplots(2, 3, sharey=True, sharex=True, squeeze=False)
    for q, quantile_value in enumerate(quantiles_list):

        evoked[q].plot(show=False, axes=axes.flat[q], zorder='std')
        axes.flat[q].axvline(0, linestyle='--', color='k', zorder=0)

        if q == 0:
            axes.flat[q].set_title("[0, " + str(round(quantiles_list[q], 2)) + "]")
        else:
            axes.flat[q].set_title("[" + str(round(quantiles_list[q-1], 2)) + ", " + str(round(quantiles_list[q], 2)) + "]")

        # remove nave added by mne
        if quantiles:
            for child in axes.flat[q].get_children():
                if isinstance(child, matplotlib.text.Annotation):
                    if child.xy == (0, 1):
                        child.remove()

        # fixed_golden_aspect_ratio(axes.flat[q])

    for ax in range(1, 6 - len(quantiles_list) + 1):
        axes.flat[-ax].remove()
    if shift:
        fig.suptitle("Note-level ERP by " + 'shifted_' + str(feature) + " values")
    else:
        fig.suptitle("Note-level ERP by " + str(feature) + " values")

    plt.show()

    return


def save_erp_by_rate(epochs_path, save_path, rate_list, tmin=2, tmax=7, min_time_interval=None):
    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))

    evoked_array = np.zeros((len(epochs_list), 102, 201, len(rate_list)))
    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[epochs.metadata['time_interval'] >= min_time_interval]

        metadata = epochs.metadata
        metadata['rate'] = 0  # add rate from stim name
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'rate'] = float(
                metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[2][0:3])

        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]

        regressors = epochs.metadata

        for r, rate in enumerate(rate_list):
            rate_epochs = epochs[regressors['rate'].to_numpy() == rate]
            evoked_array[i, :, :, r] = rate_epochs.average().get_data()

    np.save(os.path.join(save_path, 'rate_erp_data.npy'), evoked_array)

    return

def erp_by_rate_value_and_lag(erp_path, rate_list, channels):

    plt.switch_backend('TkAgg')

    # color = ['#b6adff', '#bb89c9', '#bf6592', '#c4415c', '#c81d25']
    color = ['#b6adff', '#BC7DB6', '#C24D6E', '#c81d25']

    time = times = np.linspace(-0.5, 1, 151)
    significant_channels = np.load(channels)
    print(significant_channels.shape)

    erp = np.load(erp_path)[:, :, 50:, :]
    print(erp.shape)

    erp = erp[:, :, :, 1:]

    fig, axes = plt.subplots(1, 1)
    for r, rate in enumerate(rate_list):

        rate_erp = erp[:, :, :, r]
        rate_erp = rate_erp[:, significant_channels, :]
        # axes.plot(times, np.mean(np.abs(rate_erp), axis=(0, 1)), lw=3, color=color[r], label=f'{str(rate)} Hz')
        rate_rms = np.sqrt(np.mean(rate_erp**2, axis=1))
        axes.plot(times, np.mean(rate_rms, axis=0), lw=3, color=color[r], label=f'{str(rate)} Hz')

    axes.axvline(0, linestyle='--', color='k', zorder=0)
    # axes.axhline(0, linestyle='--', color='k', zorder=0)
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('Rms ERP')
    axes.legend()

    peak_time = {}
    peak_value = {}
    for r, rate in enumerate(rate_list):
        rate_erp = erp[:, :, :, r]
        rate_erp = np.abs(rate_erp[:, significant_channels, :])
        # rate_r2_diff = np.mean(rate_r2_diff[:, best_channels[:10], :], axis=1)
        # rate_r2_diff = np.take_along_axis(rate_r2_diff, best_channels[:, :, np.newaxis], axis=1)
        # rate_r2_diff = np.mean(r2_diff[:, :, :, r], axis=1)

        # indices = np.argmax(rate_erp[:, :, 65:80], axis=2) + 65
        indices = np.argmax(rate_erp[:, :, 0:50], axis=2) + 0
        print(np.mean(times[indices], axis=1))
        print(np.mean(np.take_along_axis(rate_erp, indices[:, :, np.newaxis], axis=2), axis=(1, 2)))
        peak_time[str(rate) + 'Hz'] = np.mean(times[indices], axis=1)
        peak_value[str(rate) + 'Hz'] = np.mean(np.take_along_axis(rate_erp, indices[:, :, np.newaxis], axis=2), axis=(1, 2))

    df_time = pd.DataFrame(peak_time)
    df_value = pd.DataFrame(peak_value)

    color = ['#b6adff', '#BC7DB6', '#C24D6E', '#c81d25']

    fig1, axes_boxplot = plt.subplots(1, 1, figsize=(10, 11))
    sns.boxplot(data=df_time, ax=axes_boxplot, palette=color, fill=False, linecolor='black', width=0.6, linewidth=6,
                dodge=False, fliersize=0)
    sns.stripplot(data=df_time, color='k', size=8)
    axes_boxplot.set_ylabel('Abs ERP peak time (s)')

    fig2, axes_boxplot2 = plt.subplots(1, 1, figsize=(10, 11))
    sns.boxplot(data=df_value, ax=axes_boxplot2, palette=color, fill=False, linecolor='black', width=0.6, linewidth=6,
                dodge=False, fliersize=0)
    sns.stripplot(data=df_value, color='k', size=8)
    axes_boxplot2.set_ylabel('Abs ERP peak value')

    # fixed_golden_aspect_ratio(axes)

    spearman = []
    for s in range(erp.shape[0]):
        subject_erp = erp[s, :, :, :]
        subject_erp = subject_erp[significant_channels, :, :]
        subject_erp = np.max(np.abs(subject_erp[:, 65:80, :]), axis=1)

        # subject_data = np.zeros((subject_r2_diff.shape[0] * subject_r2_diff.shape[1], 2))
        # subject_data[:, 0] = subject_r2_diff.flatten()
        # subject_data[:, 1] = np.tile([3.5, 4.5, 5.5, 6.5], subject_r2_diff.shape[0])

        subject_data = np.zeros((subject_erp.shape[1], 2))
        subject_data[:, 0] = np.mean(subject_erp, axis=0).flatten()
        subject_data[:, 1] = [3.5, 4.5, 5.5, 6.5]

        # print(subject_data)

        # print('Correlation:', np.corrcoef(subject_data, rowvar=False)[0, 1])
        # print(scipy.stats.spearmanr(subject_data, alternative='two-sided'))
        spearman.append(scipy.stats.spearmanr(subject_data, alternative='two-sided').statistic)

    print(spearman)
    print(np.mean(spearman))
    print(np.std(spearman))

    plt.show()


def save_best_channels(epochs_path, save_folder, tmin=2, tmax=7, min_time_interval=None, n_channels=5, chosen_time=0.100):
    """
    Select channel subsets for each participant based on M100 value and save the channels indices in a numpy array.
    :param epochs_path: note epochs
    :param n_channels: number of channels to pick per participant
    :param chosen_time: best channels are picked based on values at this time
    :return:
    """
    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))

    best_channels = np.empty((len(epochs_list), n_channels), dtype=int)
    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[epochs.metadata['time_interval'] >= min_time_interval]

        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]

        evoked = epochs.average().get_data()
        best_channels[i, :] = np.argsort(np.abs(evoked[:, epochs.times.tolist().index(chosen_time)]))[-n_channels:]

    np.save(os.path.join(save_folder, 'best_channels.npy'), best_channels)

    return


def concatenate_subjects(epochs_path, save_path):
    path = Path(epochs_path)
    epochs_path_list = sorted(list(path.glob('**/*epo.fif')))

    epochs_array_list = []
    for i, subject_epochs in enumerate(epochs_path_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        epochs_array_list.append(epochs.get_data())
        del epochs

    all_subjects_array = np.concatenate(epochs_array_list)
    print(all_subjects_array.shape)
    # big_epochs.save(save_path)

    return

def save_ica_epochs(epochs_path, save_path, tmin=1, min_time_interval=0.05):
    path = Path(epochs_path)
    epochs_path_list = sorted(list(path.glob('**/*epo.fif')))

    for i, subject_epochs in enumerate(epochs_path_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # keep only notes with min time interval
        if min_time_interval is not None:
            print('Number of notes:', len(epochs.metadata))
            epochs = epochs[epochs.metadata['time_interval'] >= min_time_interval]
            print('Number of notes after min_time_interval:', len(epochs.metadata))

        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]

        ica = ICA(n_components=30, max_iter='auto', random_state=97)
        ica.fit(epochs)
        sources = ica.get_sources(epochs)
        ica.save(os.path.join(save_path, subject_epochs.name[:-8] + '-ica.fif'))
        sources.save(os.path.join(save_path, subject_epochs.name[:-8] + '_ica-epo.fif'))

    print('Done')
    return

def create_decoding_input_from_epochs(raw_path, wav_folder, midi_folder, surprise_dir, entropy_dir, save_path=None, sample_min=0, t_max=0.5,
                                      filt_range=None, channel_type='meg'):
    '''
    For 1 subject only. From epochs, output a numpy array containing MEG data in the right format (sample rate,
    duration, shape).
    '''

    if filt_range is None:
        filt_range = [0.3, 8]

    epochs = epoching_maestro_note_level(raw_path, wav_folder, midi_folder, surprise_dir, entropy_dir, t_max=t_max, time_bin=0.01) # Load raw and create epochs
    epochs.pick(channel_type)  # keep channel_type only
    # remove epochs with None pitch_interval
    to_remove = []
    for i in range(len(epochs.metadata)):
        if (epochs.metadata.iloc[i]['pitch_interval'] == None) or (np.isnan(epochs.metadata.iloc[i]['pitch_interval'])):
            to_remove.append(i)
    epochs.drop(to_remove, reason='None_interval')

    epochs.filter(filt_range[0], filt_range[1], n_jobs=-1)  # Test with lowpass filter
    f_s = 100
    epochs.resample(f_s, n_jobs=-1)
    X = epochs.get_data()
    X = X[:, :, sample_min:]
    #scaler = Scaler(info=epochs.info)
    #X = scaler.fit_transform(X)
    metadata = epochs.metadata
    del epochs
    X = np.moveaxis(X, 2, 0)

    if save_path is not None:
        np.save(save_path, X)

    return X, metadata



def run_decoding(X, metadata, save_path=None, condition=None):
    """
    Run decoding on 1 subject.
        condition: list of str
    """

    print(X.shape)
    # if condition is not None:
    #     condition_list = condition_list_from_metadata(metadata, condition)
    #     X = X[:, condition_list, :]
    # X = X.reshape((-1, X.shape[2]))
    # X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))  # scale each channel
    X = (X - np.mean(X, axis=1)) / np.std(X, axis=1)  # scale each channel at each time sample
    # X = (X - np.mean(X)) / np.std(X)  # scale everything the same way

    Y = metadata[['pitch', 'pitch_interval', 'surprise', 'entropy']].to_numpy()
    for i in range(Y.shape[0]):
        Y[i, 0] = Y[i, 0][0]  # pitch are stored as lists

    Y = Y.astype(float) # Y was object

    print(Y.shape)

    Y = np.tile(Y, (X.shape[0], 1, 1))
    print(Y.shape)

    alphas = np.logspace(-3, 10, 14)
    #alphas = np.logspace(5, 9, 5)
    scores = np.zeros_like(alphas)
    models = []

    n_folds = 10
    cv = KFold(n_folds, shuffle=True, random_state=37)

    for ii, alpha in enumerate(alphas):
        print('alpha:', alpha)

        model = Ridge(alpha=alpha)
        #model = LinearModel(Ridge(alpha=alpha))
        score = np.zeros(n_folds)
        train_score = np.zeros((n_folds, Y.shape[2]))
        test_score = np.zeros((n_folds, Y.shape[2]))
        for fold_number, (train, test) in enumerate(cv.split(np.moveaxis(X, 1, 0))):
            model.fit(X[:, train, :].reshape((-1, X.shape[2])), Y[:, train, :].reshape((-1, Y.shape[2])))
            score[fold_number] = model.score(X[:, test, :].reshape((-1, X.shape[2])), Y[:, test, :].reshape((-1, Y.shape[2])))
            train_score[fold_number, :] = r2_score(Y[:, train, :].reshape((-1, Y.shape[2])), model.predict(X[:, train, :].reshape((-1, X.shape[2]))), multioutput='raw_values')
            test_score[fold_number, :] = r2_score(Y[:, test, :].reshape((-1, Y.shape[2])), model.predict(X[:, test, :].reshape((-1, X.shape[2]))), multioutput='raw_values')
        
        print("scores_train:", np.mean(train_score, axis=0))
        print("scores_test:", np.mean(test_score, axis=0))
        scores[ii] = score.mean()
        models.append(model)
        print('score:', scores[ii])

    # Choose the model that performed best on the held out data
    ix_best_alpha = np.argmax(scores)
    print('best score', np.max(scores))
    best_mod = models[ix_best_alpha]
    # if not math.isnan(np.max(scores)):
    #     best_mod.fit(X, Y)

    # compute the temporal scores
    scores = np.zeros((n_folds, X.shape[0], Y.shape[2]))
    for fold_number, (train, test) in enumerate(cv.split(np.moveaxis(X, 1, 0))):
        best_mod.fit(X[:, train, :].reshape((-1, X.shape[2])), Y[:, train, :].reshape((-1, Y.shape[2])))
        for sample in range(X.shape[0]):
            scores[fold_number, sample, :] = r2_score(Y[sample, test, :], best_mod.predict(X[sample, test, :]), multioutput='raw_values')

    average_scores = np.mean(scores, axis=0)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'decoding_scores.npy'), average_scores)

    return average_scores


def run_time_sample_decoding(X, metadata, save_path=None, condition=None):
    """
    Run decoding on 1 subject.
        condition: list of str
    """

    print(X.shape)
    # if condition is not None:
    #     condition_list = condition_list_from_metadata(metadata, condition)
    #     X = X[:, condition_list, :]
    # X = X.reshape((-1, X.shape[2]))
    X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))  # scale each channel
    #X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)  # scale each channel at each time sample
    #X = (X - np.mean(X)) / np.std(X)  # scale everything the same way

    Y = metadata[['pitch', 'pitch_interval', 'surprise', 'entropy']].to_numpy()
    for i in range(Y.shape[0]):
        Y[i, 0] = Y[i, 0][0]  # pitch are stored as lists

    Y = Y.astype(float)  # Y was object

    print(Y.shape)
    Y = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)   # scaling output features

    Y = np.tile(Y, (X.shape[0], 1, 1))
    print(Y.shape)

    #alphas = np.logspace(-5, 10, 16)
    alphas = np.logspace(4, 6, 3)
    scores = np.zeros_like(alphas)
    models = []

    n_folds = 10
    n_models = X.shape[0]
    cv = KFold(n_folds, shuffle=True, random_state=58)

    for ii, alpha in enumerate(alphas):
        print('alpha:', alpha)

        model = [Ridge(alpha=alpha) for _ in range(n_models)]
        score = np.zeros(n_folds)
        for fold_number, (train, test) in enumerate(cv.split(np.moveaxis(X, 1, 0))):
            sample_scores = np.zeros(n_models)
            for t in range(n_models):
                model[t].fit(X[t, train, :], Y[t, train, :])
                sample_scores[t] = model[t].score(X[t, test, :], Y[t, test, :])
            score[fold_number] = sample_scores.mean()

        scores[ii] = score.mean()
        models.append(model)
        print('score:', scores[ii])

    # Choose the model that performed best on the held out data
    ix_best_alpha = np.argmax(scores)
    print('best score', np.max(scores))
    best_mod = models[ix_best_alpha]
    # if not math.isnan(np.max(scores)):
    #     best_mod.fit(X, Y)

    # compute the temporal scores
    scores = np.zeros((n_folds, X.shape[0], Y.shape[2]))
    for fold_number, (train, test) in enumerate(cv.split(np.moveaxis(X, 1, 0))):
        for sample in range(X.shape[0]):
            best_mod[sample].fit(X[sample, train, :], Y[sample, train, :])
            scores[fold_number, sample, :] = r2_score(Y[sample, test, :], best_mod[sample].predict(X[sample, test, :]),
                                                      multioutput='raw_values')

    average_scores = np.mean(scores, axis=0)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'decoding_scores.npy'), average_scores)

    return average_scores



def run_decoding_with_logistic(input, metadata, save_path=None, condition=None):
    """
    Run decoding on 1 subject.
        condition: list of str
    """

    X = input
    print(X.shape)
    # if condition is not None:
    #     condition_list = condition_list_from_metadata(metadata, condition)
    #     X = X[:, condition_list, :]
    # X = X.reshape((-1, X.shape[2]))
    # X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))

    Y = metadata[['pitch', 'pitch_interval', 'surprise', 'entropy']].to_numpy()
    for i in range(Y.shape[0]):
        Y[i, 0] = Y[i, 0][0]  # pitch are stored as lists

    Y = Y.astype(float)  # Y was object

    print(X.shape)
    print(Y.shape)

    Y = np.tile(Y, (X.shape[0], 1, 1))
    print(Y.shape)

    alphas = np.logspace(-3, 7, 11)
    # alphas = np.logspace(4, 8, 5)
    scores = np.zeros_like(alphas)
    models = []

    n_folds = 10
    cv = KFold(n_folds, shuffle=True, random_state=37)

    for ii, alpha in enumerate(alphas):
        print('alpha:', alpha)

        model = Ridge(alpha=alpha)
        logistic_note = LogisticRegression(C=1/alpha)
        logistic_interval = LogisticRegression(C=1/alpha)

        score = np.zeros(n_folds)
        train_score = np.zeros((n_folds, Y.shape[2]))
        test_score = np.zeros((n_folds, Y.shape[2]))
        for fold_number, (train, test) in enumerate(cv.split(np.moveaxis(X, 1, 0))):
            model.fit(X[:, train, :].reshape((-1, X.shape[2])), Y[:, train, :][:, :, [2,3]].reshape((-1, 2)))
            logistic_note.fit(X[:, train, :].reshape((-1, X.shape[2])), Y[:, train, 0].reshape((-1)))
            logistic_interval.fit(X[:, train, :].reshape((-1, X.shape[2])), Y[:, train, 1].reshape((-1)))
            score[fold_number] = model.score(X[:, test, :].reshape((-1, X.shape[2])), Y[:, test, :][:, :, [2,3]].reshape((-1, 2)))
            train_score[fold_number, [2,3]] = r2_score(Y[:, train, :][:, :, [2,3]].reshape((-1, 2)),
                                                   model.predict(X[:, train, :].reshape((-1, X.shape[2]))),
                                                   multioutput='raw_values')
            train_score[fold_number, 0] = accuracy_score(Y[:, train, 1].reshape((-1)), logistic_note.predict(X[:, train, :].reshape((-1, X.shape[2]))))
            train_score[fold_number, 1] = accuracy_score(Y[:, train, 1].reshape((-1)), logistic_interval.predict(X[:, train, :].reshape((-1, X.shape[2]))))
            test_score[fold_number, [2,3]] = r2_score(Y[:, test, :][:, :, [2,3]].reshape((-1, 2)),
                                                  model.predict(X[:, test, :].reshape((-1, X.shape[2]))),
                                                  multioutput='raw_values')
            test_score[fold_number, 0] = accuracy_score(Y[:, test, 1].reshape((-1)), logistic_note.predict(
                X[:, test, :].reshape((-1, X.shape[2]))))
            test_score[fold_number, 1] = accuracy_score(Y[:, test, 1].reshape((-1)), logistic_interval.predict(
                X[:, test, :].reshape((-1, X.shape[2]))))

        print("scores_train:", np.mean(train_score, axis=0))
        print("scores_test:", np.mean(test_score, axis=0))
        scores[ii] = score.mean()
        models.append(model)
        print('score:', scores[ii])

    # Choose the model that performed best on the held out data
    ix_best_alpha = np.argmax(scores)
    print('best score', np.max(scores))
    best_mod = models[ix_best_alpha]
    # if not math.isnan(np.max(scores)):
    #     best_mod.fit(X, Y)

    # compute the temporal scores
    scores = np.zeros((n_folds, X.shape[0], Y.shape[2]))
    for fold_number, (train, test) in enumerate(cv.split(np.moveaxis(X, 1, 0))):
        best_mod.fit(X[:, train, :].reshape((-1, X.shape[2])), Y[:, train, :].reshape((-1, Y.shape[2])))
        for sample in range(X.shape[0]):
            scores[fold_number, sample, :] = r2_score(Y[sample, test, :], best_mod.predict(X[sample, test, :]),
                                                      multioutput='raw_values')

    average_scores = np.mean(scores, axis=0)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'decoding_scores.npy'), average_scores)

    return average_scores


def run_decoding_all_subjects(data_path, wav_folder, midi_folder, surprise_dir, entropy_dir, save_path, sample_min=0, t_max=0.5, channel_type='meg', condition=None):
    '''
    '''

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    scores_all_subj = []
    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]

        X, metadata = create_decoding_input_from_epochs(raw_path, wav_folder, midi_folder, surprise_dir, entropy_dir, save_path=None, sample_min=sample_min, t_max=t_max, channel_type=channel_type)
        scores = run_time_sample_decoding(X, metadata, save_path=os.path.join(save_path, subject_path.name), condition=None)
        del X
        del metadata
        scores_all_subj.append(scores)
        del scores
    scores_all_subj = np.array(scores_all_subj)

    np.save(os.path.join(save_path, "decoding_scores.npy"), scores_all_subj)

    return


def plot_decoding_scores(scores_path, features=None, time_range=None):

    if time_range is None:
        time_range = [-0.2, 0.5]
    if features is None:
        features = ['pitch', 'interval', 'surprise', 'entropy']

    scores = np.load(scores_path)
    print(scores.shape)
    n_subjects = scores.shape[0]
    scores_avg = np.mean(scores, axis=0)
    sem = np.std(scores, axis=0) / np.sqrt(n_subjects)

    times = np.linspace(time_range[0], time_range[1], scores_avg.shape[0])

    for i in range(len(features)):
        plt.plot(times, scores_avg[:, i])
    plt.xlabel('Time (s)')
    plt.ylabel('Decoding score (R²)')
    plt.legend(features)
    for i in range(len(features)):
        plt.fill_between(times, scores_avg[:, i] - sem[:, i], scores_avg[:, i] + sem[:, i], alpha=0.6)
    plt.axvline(0, linestyle='--', color='k', zorder=0)
    plt.axhline(0, linestyle='--', color='k', zorder=0)

    plt.show()

    return

def plot_decoding_scores_indiv_subject(scores_path, features=None, time_range=None):

    if time_range is None:
        time_range = [-0.2, 0.5]
    if features is None:
        features = ['pitch', 'interval', 'surprise', 'entropy']

    scores = np.load(scores_path)
    print(scores.shape)

    times = np.linspace(time_range[0], time_range[1], scores.shape[0])

    for i in range(len(features)):
        plt.plot(times, scores[:, i])
    plt.xlabel('Time (s)')
    plt.ylabel('Decoding score (R²)')
    plt.legend(features)
    plt.axvline(0, linestyle='--', color='k', zorder=0)
    plt.axhline(0, linestyle='--', color='k', zorder=0)

    plt.show()

    return


def save_note_epochs_processed(data_path, wav_folder, midi_folder, acoustic_folder, perfrnn_dir, magenta_perfrnn_dir, paulrnn_dir, save_path, t_min=-1, t_max=1, channel_type='mag', n_jobs=-1):

    os.makedirs(save_path, exist_ok=True)

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        epochs = epoching_maestro_note_level(raw_path, wav_folder, midi_folder, acoustic_folder, perfrnn_dir, magenta_perfrnn_dir, paulrnn_dir, t_min=t_min,
                                             t_max=t_max, channel_type=channel_type, downsample=True,
                                             detrend=True, lowpass=True, multiple_notes='max', n_jobs=n_jobs)

        epochs.save(os.path.join(save_path, subject_path.name + '-epo.fif'), overwrite=True)

        del epochs

    return


def save_note_epochs_tfr(data_path, wav_folder, midi_folder, acoustic_folder, perfrnn_dir, magenta_perfrnn_dir, paulrnn_dir, save_path):

    os.makedirs(save_path, exist_ok=True)

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        tfr_path = list(subject_path.glob('**/*tfr.h5'))[0]
        epochs = epoching_maestro_note_level_time_frequency(tfr_path, wav_folder, midi_folder, acoustic_folder, perfrnn_dir, magenta_perfrnn_dir, paulrnn_dir, multiple_notes='max')

        epochs.save(os.path.join(save_path, subject_path.name + '_epo-tfr.h5'), overwrite=True)

        del epochs

    return


def from_metadata_to_regressors(metadata, multiple_pitches='max', shifted=True):
    """
    The note-level epoch metadata contains only a subset of the features of interest. This function computes/adds the
     other features and outputs a dataframe of predictors ready to use for regressions.
    Surprise is time-locked on current note, whereas entropy is shifted by one sample to be time-locked on previous note
    """

    paulrnn = 'mean_surprise' in metadata.columns  # test if there are features specific to PaulRNN

    rep = []
    rate = []
    onset = []
    pitch = []
    interval = []
    trial_time = []
    entropy = []
    if paulrnn:
        mean_surprise = []
        positive_surprise = []
        pitch_entropy_mean = []
    else:
        surprise = []

    for index, row in metadata.iterrows():
        rep.append(int(row['stim_name'][12]))
        rate.append(float(row['stim_name'][14:17]))
        onset.append(int(row['stim_name'][-5]))
        if multiple_pitches == 'max':
            p = max(row['pitch'])
            pitch.append(np.log2(p))
        if (index > 0) and (row['stim_name'] == metadata.loc[index-1]['stim_name']):
            interval.append(np.abs(pitch[-1] - pitch[-2]))
        else:
            interval.append(0)
        trial_time.append(row['trial_time'])

        if shifted:
            if (index + 1 < len(metadata)) and (row['stim_name'] == metadata.loc[index + 1]['stim_name']):
                entropy.append(metadata.loc[index+1]['entropy'])
                if paulrnn:
                    pitch_entropy_mean.append(metadata.loc[index+1]['pitch_entropy_mean'])
            else:
                entropy.append(0)
                if paulrnn:
                    pitch_entropy_mean.append(0)
        else:
            entropy.append(row['entropy'])  # TODO: go back to this

        # surprise is not shifted in any case
        if paulrnn:
            mean_surprise.append(row['mean_surprise'])
            positive_surprise.append(row['positive_surprise'])
        else:
            surprise.append(row['surprise'])

    if paulrnn:
        features = pd.DataFrame.from_dict(
            {'rep': rep, 'rate': rate, 'onset': onset, 'pitch': pitch, 'interval': interval, 'trial_time': trial_time,
             'mean_surprise': mean_surprise, 'positive_surprise': positive_surprise, 'pitch_entropy_mean': pitch_entropy_mean, 'entropy': entropy})
    else:
        features = pd.DataFrame.from_dict({'rep': rep, 'rate': rate, 'onset': onset, 'pitch': pitch, 'interval': interval,
                                       'trial_time': trial_time, 'surprise': surprise, 'entropy': entropy})

    return features


def transform_metadata(metadata, multiple_pitches='max', add_shifted=True):
    """
    The note-level epoch metadata contains most of the features of interest. This function computes/adds the
     other features and outputs a dataframe of predictors ready to use for regressions.
    Surprise is time-locked to the current note, whereas entropy can be shifted (shifted=True) by one sample to be time-locked to the previous note
    """

    rep = []
    rate = []
    onset = []
    pitch = []
    interval = []

    shifted_entropy_perfrnn = []
    shifted_entropy_pitch_paulrnn = []
    shifted_entropy_timing_paulrnn = []

    for index, row in metadata.iterrows():
        rep.append(int(row['stim_name'][12]))
        rate.append(float(row['stim_name'][14:17]))
        onset.append(int(row['stim_name'][-5]))
        if multiple_pitches == 'max':
            p = max(row['pitch'])
            pitch.append(np.log2(p))
        if (index > 0) and (row['stim_name'] == metadata.loc[index-1]['stim_name']):
            interval.append(np.abs(pitch[-1] - pitch[-2]))
        else:
            interval.append(0)

        if add_shifted:
            if (index + 1 < len(metadata)) and (row['stim_name'] == metadata.loc[index + 1]['stim_name']):
                shifted_entropy_perfrnn.append(metadata.loc[index+1]['entropy_perfrnn'])
                shifted_entropy_pitch_paulrnn.append(metadata.loc[index+1]['entropy_pitch_paulrnn'])
                shifted_entropy_timing_paulrnn.append(metadata.loc[index+1]['entropy_timing_paulrnn'])
            else:
                shifted_entropy_perfrnn.append(0)
                shifted_entropy_pitch_paulrnn.append(0)
                shifted_entropy_timing_paulrnn.append(0)


        # if shifted:
        #     if (index + 1 < len(metadata)) and (row['stim_name'] == metadata.loc[index + 1]['stim_name']):
        #         entropy.append(metadata.loc[index+1]['entropy'])
        #         if paulrnn:
        #             pitch_entropy_mean.append(metadata.loc[index+1]['pitch_entropy_mean'])
        #     else:
        #         entropy.append(0)
        #         if paulrnn:
        #             pitch_entropy_mean.append(0)
        # else:
        #     entropy.append(row['entropy'])  # TODO: go back to this

    if add_shifted:
        feature_names = ['shifted_entropy_perfrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn', 'rep', 'rate', 'onset', 'pitch', 'interval']
        features = [shifted_entropy_perfrnn, shifted_entropy_pitch_paulrnn, shifted_entropy_timing_paulrnn, rep, rate, onset, pitch, interval]
    else:
        feature_names = ['rep', 'rate', 'onset', 'pitch', 'interval']
        features = [rep, rate, onset, pitch, interval]

    for name, feat in zip(feature_names, features):
        metadata[name] = feat

    return metadata


def correlation_matrix(features):

    plt.switch_backend('TkAgg')
    corr = features.corr()
    sns.heatmap(corr, cmap='bwr', vmin=-1, vmax=1)
    plt.show()

    return


def features_histograms(epoch_path):
    plt.switch_backend('TkAgg')

    epochs = mne.read_epochs(epoch_path)
    features = ['surprise_mean_paulrnn', 'surprise_possum_paulrnn', 'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']

    metadata = epochs.metadata

    print('Number of notes:', len(epochs.metadata))
    epochs = epochs[metadata['time_interval'] >= 0.05]
    metadata = epochs.metadata
    print('Number of notes after min_time_interval:', len(epochs.metadata))

    metadata['shifted_entropy_timing_paulrnn'] = np.nan  # add shifted features
    for i_note in range(metadata.shape[0]):
        if (i_note + 1 < len(metadata)) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note + 1], 'stim_name']):
            metadata.loc[metadata.index[i_note], 'shifted_entropy_timing_paulrnn'] = metadata.loc[metadata.index[i_note + 1], 'entropy_timing_paulrnn']

    epochs.metadata = metadata
    epochs = epochs[~np.isnan(metadata['shifted_entropy_timing_paulrnn'])]  # remove the last notes of pieces
    metadata = epochs.metadata

    epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= 2]
    epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= 7]

    fig, axes = plt.subplots(2, 2)
    for i, feat in enumerate(features):
        feat_array = epochs.metadata[feat].to_numpy()
        axes.flat[i].hist(feat_array, bins=100)
        axes.flat[i].axvline(np.mean(feat_array), c='k')
        axes.flat[i].set_title(feat)

    plt.show()

    return


def regression_predictive_features(epochs_path, save_path, confounds='all', model=None, data_type='meg',
                                   signal_type='raw', montage='mono', specific_channels=None, add_shifted=False, remove=None, tmin=2, tmax=7,
                                   n_folds=5, alphas=(1e2, 1e3, 1e4, 1e5, 1e6, 1e7), permutation=False, r2_control=None, min_time_interval=0.050, best_channels_path=None, deviation_quantile_range=None, baseline=False, avg_surprise=False, proportion=None):
    """
    epochs_path: path to folder containing all epochs.fif files
    save_path: path to folder where results will be saved
    confounds: selection of counfounds regressors. If 'all', everything is selected.
    model: 'perfrnn', 'paulrnn' or None. If None, only confounds regressors will be used.
    dataset: dataset to analyze, 'meg' or 'seeg'.
        If 'meg', returns a dictionnary with the ch_names, montage object, and a numpy array with dimensions: n_subjects, n_features, n_channels, n_timepoints
        If 'seeg', returns a dictionnary that contains, for each patient, the ch_names, montage object, and a numpy array with dimensions: n_features, n_channels, n_timepoints
    signal_type: used for seeg only. 'raw' or 'gamma'
    montage: used for seeg only. 'mono' or 'bipo'
    specific_channels: list of channels to pick
    add_shifted: if True, add shifted versions of entropy regressors.
    remove: list of feature names (str) to remove
    tmin: remove notes before a given time (to avoid onset response)
    n_folds: If 1, just fit a linear regression. If > 1, performs n-folds cv with ridge regression.
    alphas: list of alpha value to use in ridge regression.
    permutation: If False, run the analysis normally. If an integer, use it as a seed to shuffle the values of the features of interest.
    r2_control: Numpy array of shape (n_subjects, n_channels, n_times), used only if permutations are done to transform r2 scores to t-values.
    min_time_interval: float, keep only notes that are played at least min_time_interval seconds after the previous one.
    best_channels_path: str, path to npy file specifying which channels to keep for each subject. Should not be used with 'specific_channels' (which picks the same channels for every subject)
    deviation_quantile_range: list of len 2, contains the low and high quantile values used for the time interval deviation analysis.

    Function updated for complete set of regressors
    """

    alphas = np.array(alphas)

    # create save dir
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    #manage random seed for permutations
    if permutation is not False:
        rng = np.random.default_rng(seed=permutation)

    # feature selection
    features = []
    if confounds == 'all':
        features = ['rate', 'onset_type', 'trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                    'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                    'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']
        if add_shifted:
            features = features + ['shifted_trial_time', 'shifted_time_interval', 'shifted_pitch', 'shifted_pitch_interval',
                                'shifted_loudness', 'shifted_brightness','shifted_periodicity', 'shifted_roughness', 'shifted_spectralflux',
                                'shifted_loudness_deriv', 'shifted_brightness_deriv', 'shifted_periodicity_deriv', 'shifted_roughness_deriv', 'shifted_spectralflux_deriv']
            shifted_features = ['trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                    'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                    'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']  # with all shifted controls but rate
    if model == 'perfrnn':
        features = features + ['surprise_perfrnn', 'entropy_perfrnn']
        F_toshuffle = ['surprise_perfrnn', 'entropy_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_surprise_perfrnn', 'shifted_entropy_perfrnn']
            shifted_features = shifted_features + ['surprise_perfrnn', 'entropy_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_surprise_perfrnn', 'shifted_entropy_perfrnn']
    if model == 'magenta_perfrnn':
        features = features + ['surprise_magenta_pitch_perfrnn', 'entropy_magenta_pitch_perfrnn', 'surprise_magenta_timing_perfrnn', 'entropy_magenta_timing_perfrnn'] #
        F_toshuffle = ['surprise_magenta_pitch_perfrnn', 'entropy_magenta_pitch_perfrnn', 'surprise_magenta_timing_perfrnn', 'entropy_magenta_timing_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_surprise_magenta_pitch_perfrnn', 'shifted_surprise_magenta_timing_perfrnn', 'shifted_entropy_magenta_pitch_perfrnn', 'shifted_entropy_magenta_timing_perfrnn']
            shifted_features = shifted_features + ['surprise_magenta_pitch_perfrnn', 'entropy_magenta_pitch_perfrnn', 'surprise_magenta_timing_perfrnn', 'entropy_magenta_timing_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_surprise_magenta_pitch_perfrnn', 'shifted_surprise_magenta_timing_perfrnn', 'shifted_entropy_magenta_pitch_perfrnn', 'shifted_entropy_magenta_timing_perfrnn']
    elif model == 'paulrnn':
        features = features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        if avg_surprise:
            features = features + ['avg_surprise']
        F_toshuffle = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']
            shifted_features = shifted_features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
            F_toshuffle = F_toshuffle + ['shifted_surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']

    # get the correct epochs path depending on the desired dataset
    path = Path(epochs_path)
    if data_type == 'meg':
        epochs_list = sorted(list(path.glob('**/*epo.fif')))
    if data_type == 'seeg':
        if signal_type == 'gamma':
            st = '_gamma'
        elif signal_type == 'raw':
            st = ''
        epochs_list = sorted(list(path.glob('**/*_epochs' + st + '_' + montage + '_preprocessed_IDA.fif')))

    # prepare dict to save output
    if data_type == 'meg':
        output = {'features': features,
                  'specific_channels': specific_channels,
                  'r2': [],
                  'betas': [],
                  'n_folds': n_folds,
                  'alphas': alphas,
                  'permutation': permutation,
                  'min_time_interval': min_time_interval}
    elif data_type == 'seeg':
        output = dict()

    if best_channels_path is not None:
        best_channels = np.load(best_channels_path)

    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        #add more info to dict
        if data_type == 'seeg':
            patient_ID = subject_epochs.name[0:6]
            output[patient_ID] = {'montage_type': montage,
                                  'signal_type': signal_type,
                                  'ch_names': epochs.ch_names,
                                  'epochs_info': epochs.info,
                                  'times': epochs.times,
                                  'specific_channels': specific_channels,
                                  'montage': epochs.get_montage(),
                                  'n_epochs': epochs.__len__(),
                                  'features': features,
                                  'subject_betas': None,
                                  'subject_r2': None,
                                  'n_folds': n_folds,
                                  'alphas': alphas,
                                  'permutation': permutation,
                                  'min_time_interval': min_time_interval}
        elif data_type == 'meg':
            output['times'] = epochs.times
            output['epochs_info'] = epochs.info

        # pick channels
        if specific_channels is not None:
            epochs.pick(specific_channels)

        # TODO: the following should be added as a function since it is used in several functions.
        metadata = epochs.metadata

        # keep only notes with min time interval
        if min_time_interval is not None:
            print('Number of notes:', len(epochs.metadata))
            epochs = epochs[metadata['time_interval'] >= min_time_interval]
            metadata = epochs.metadata
            print('Number of notes after min_time_interval:', len(epochs.metadata))

        # transform metadata to obtain all features
        metadata['pitch'] = metadata['pitch'].apply(lambda x: np.log2(x[0]))  # pitch info was in list format. Apply log transform
        metadata['rate'] = 0  # add rate from stim name
        metadata['onset_type'] = 0  # add onset type from stim name
        metadata['pitch_interval'] = 0  # add pitch interval
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'rate'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[2][0:3])
            metadata.loc[metadata.index[i_note], 'onset_type'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[4][-5])
            if metadata.loc[metadata.index[i_note], 'trial_time'] > 0:
                metadata.loc[metadata.index[i_note], 'pitch_interval'] = np.abs(metadata.loc[metadata.index[i_note-1], 'pitch'] - metadata.loc[metadata.index[i_note], 'pitch'])

        # add shifted features
        if add_shifted:
            for feat in shifted_features:
                metadata['shifted_' + feat] = np.nan  # add shifted features
            for i_note in range(metadata.shape[0]):
                if (i_note + 1 < len(metadata)) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note+1], 'stim_name']):
                    for feat in shifted_features:
                        metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note+1], feat]
            # version to try to shift in the other direction
            # for i_note in range(metadata.shape[0]):
            #     if (i_note - 1 >= 0) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note-1], 'stim_name']):
            #         for feat in shifted_features:
            #             metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note-1], feat]

            epochs.metadata = metadata
            epochs = epochs[~np.isnan(metadata['shifted_' + shifted_features[0]])]  # remove the last notes of pieces
            metadata = epochs.metadata

        # add avg_surprise and interaction
        if avg_surprise:
            metadata['avg_surprise'] = metadata.groupby('stim_name')['surprise_mean_paulrnn'].transform('mean')

        regressors = metadata[features]

        #remove specific features for variance partitioning
        features_to_keep = copy.copy(features)
        if remove is not None:
            if type(remove) == list:
                for f in remove:
                    features_to_keep.remove(f)
                    regressors = regressors[features_to_keep]
            elif type(remove) == str:
                features_to_keep.remove(remove)
                regressors = regressors[features_to_keep]


        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressors = regressors[regressors['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]
        regressors = regressors[regressors['trial_time'].to_numpy() <= tmax]

        print('Number of notes after min/max trial time:', len(epochs.metadata))

        # entropy median
        # # median_entropy = np.median(epochs.metadata['entropy_timing_paulrnn'].to_numpy())
        # quantile_entropy = np.quantile(epochs.metadata['entropy_timing_paulrnn'].to_numpy(), q=0.1)
        # regressors = regressors[epochs.metadata['entropy_timing_paulrnn'].to_numpy() < quantile_entropy]
        # epochs = epochs[epochs.metadata['entropy_timing_paulrnn'].to_numpy() < quantile_entropy]

        if deviation_quantile_range is not None:
            # add mean_interval
            metadata = epochs.metadata
            mean_interval = metadata.groupby('stim_name', as_index=False)['time_interval'].mean().rename(columns={'time_interval': 'mean_interval'})
            epochs.metadata = epochs.metadata.merge(mean_interval, on='stim_name', how='left')

            # early - ontime - late analysis
            low_quantile_value = np.quantile(epochs.metadata['time_interval'].to_numpy() - epochs.metadata['mean_interval'].to_numpy(), q=deviation_quantile_range[0])
            high_quantile_value = np.quantile(epochs.metadata['time_interval'].to_numpy() - epochs.metadata['mean_interval'].to_numpy(), q=deviation_quantile_range[1])
            # low_quantile_value = np.quantile((epochs.metadata['time_interval'].to_numpy() - epochs.metadata['mean_interval'].to_numpy()) / epochs.metadata['mean_interval'].to_numpy(), q=deviation_quantile_range[0])
            # high_quantile_value = np.quantile((epochs.metadata['time_interval'].to_numpy() - epochs.metadata['mean_interval'].to_numpy()) / epochs.metadata['mean_interval'].to_numpy(), q=deviation_quantile_range[1])
            print('quantiles:', low_quantile_value, high_quantile_value)

            # relative
            # regressors = regressors[((epochs.metadata['time_interval'].to_numpy() - (epochs.metadata['mean_interval'].to_numpy())) / (epochs.metadata['mean_interval'].to_numpy()) >= low_quantile_value)
            #                         & ((epochs.metadata['time_interval'].to_numpy() - (epochs.metadata['mean_interval'].to_numpy())) / (epochs.metadata['mean_interval'].to_numpy()) <= high_quantile_value)]
            # epochs = epochs[((epochs.metadata['time_interval'].to_numpy() - (epochs.metadata['mean_interval'].to_numpy())) / (epochs.metadata['mean_interval'].to_numpy()) >= low_quantile_value)
            #                 & ((epochs.metadata['time_interval'].to_numpy() - (epochs.metadata['mean_interval'].to_numpy())) / (epochs.metadata['mean_interval'].to_numpy()) <= high_quantile_value)]
            # absolute
            regressors = regressors[((epochs.metadata['time_interval'].to_numpy() - epochs.metadata['mean_interval'].to_numpy()) >= low_quantile_value)
                                    & ((epochs.metadata['time_interval'].to_numpy() - (epochs.metadata['mean_interval'].to_numpy())) <= high_quantile_value)]
            epochs = epochs[((epochs.metadata['time_interval'].to_numpy() - (epochs.metadata['mean_interval'].to_numpy())) >= low_quantile_value)
                            & ((epochs.metadata['time_interval'].to_numpy() - (epochs.metadata['mean_interval'].to_numpy())) <= high_quantile_value)]

        #shuffle features for permutations
        if permutation is not False:
            for feat in F_toshuffle:
                regressors[feat] = rng.permutation(regressors[feat].values)

        if baseline:
            epochs = epochs.apply_baseline((None, -0.100))

        # load data
        y = epochs.get_data()  # (trials, channels, time)
        if best_channels_path is not None:
            y = y[:, best_channels[i, :], :]
        # x = epochs.metadata[features].to_numpy()
        x = regressors.to_numpy()

        if proportion is not None:
            n_epochs_to_keep = int(x.shape[0] * proportion)
            x = x[:n_epochs_to_keep]
            y = y[:n_epochs_to_keep]

        # standardization of features and meg
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
        y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)  # removes the relative importance of subjects, channels, and time points

        del epochs

        # independent regression for every time point and channel
        subject_betas = np.zeros((x.shape[1], y.shape[1], y.shape[2]))
        subject_r2 = np.zeros((y.shape[1], y.shape[2]))
        if n_folds > 1:
            cv = KFold(n_splits=n_folds)
            cv.get_n_splits(x)
        start = time.time()

        def individual_channel_regression(c_ind):  # TODO: put outside of function
            """function used for multiprocessing"""
            if n_folds == 1:
                regr = LinearRegression().fit(x, y[:, c_ind, :])
                # subject_betas[:, c_ind, :] = regr.coef_.T
                # subject_r2[c_ind, :] = r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
                return regr.coef_.T, r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
            elif n_folds > 1:
                cv_scores = np.full((alphas.shape[0], n_folds, y.shape[2]), np.nan)
                for a, alpha in enumerate(alphas):
                    for j, (train, test) in enumerate(cv.split(x)):
                        regr = Ridge(alpha=alpha)
                        regr.fit(x[train, :], y[train, c_ind, :])
                        pred = regr.predict(x[test, :])
                        cv_scores[a, j, :] = r2_score(y[test, c_ind, :], pred, multioutput='raw_values')
                # store cv scores and best alphas
                cv_scores = np.mean(cv_scores, axis=1)  # mean cross-validated r² across folds
                best_alphas = alphas[np.argmax(cv_scores, axis=0)]  # store best alpha for each timepoint
                cv_scores = np.max(cv_scores, axis=0)  # for each timepoint, keep the highest cross-val r² across alphas
                # subject_r2[c_ind, :] = cv_scores
                # fit again with the best alpha and store betas
                regr = Ridge(best_alphas).fit(x, y[:, c_ind, :])
                # subject_betas[:, c_ind, :] = regr.coef_.T
                return regr.coef_.T, cv_scores

        # multiprocessing channels
        channels_regressions = Parallel(n_jobs=-1)(delayed(individual_channel_regression)(c) for c in range(y.shape[1]))
        # channels_regressions = [individual_channel_regression(c) for c in range(y.shape[1])]
        for c in range(y.shape[1]):
            subject_betas[:, c, :] = channels_regressions[c][0]
            subject_r2[c, :] = channels_regressions[c][1]

        end = time.time()
        print(f'Regression took {end - start} seconds')

        # store regression results
        if data_type == 'meg':
            output['betas'].append(subject_betas)
            output['r2'].append(subject_r2)
        elif data_type == 'seeg':
            output[patient_ID]['subject_betas'] = subject_betas
            output[patient_ID]['subject_r2'] = subject_r2

    if data_type == 'meg':
        output['betas'] = np.array(output['betas'])
        output['r2'] = np.array(output['r2'])

        # if this is a permutation, store only t-values for r2 and betas
        if permutation is not False:
            output['betas'] = np.mean(output['betas'], axis=0, keepdims=True) / (np.std(output['betas'], axis=0, keepdims=True) / np.sqrt(output['betas'].shape[0]))
            output['r2'] = np.mean(output['r2'] - r2_control, axis=0, keepdims=True) / (np.std(output['r2'] - r2_control, axis=0, keepdims=True) / np.sqrt(output['r2'].shape[0]))  # comparison with control

    if model is None:
        model = 'acoustic'
    if remove is None:
        removed_feat = ''
    else:
        removed_feat = '_removed_' + str(remove)
    if permutation is False:
        permutation = ''
    else:
        permutation = '_perm' + str(permutation).zfill(4)
    if proportion is not None:
        prop_name = '_' + str(int(proportion * 100)) + 'percent'
    else:
        prop_name = ''
    if data_type == 'meg':
        with open(os.path.join(save_path, 'regr_meg_' + model + removed_feat + permutation + prop_name + '.pkl'), 'wb') as fp:
            pickle.dump(output, fp)
    elif data_type == 'seeg':
        with open(os.path.join(save_path, 'regr_seeg_' + signal_type + '_' + montage + '_' + model + removed_feat + permutation + prop_name + '.pkl'), 'wb') as fp:
            pickle.dump(output, fp)

    return output


def regression_predictive_features_by_rate(epochs_path, save_path, confounds='all', model=None, data_type='meg',
                                   signal_type='raw', montage='mono', specific_channels=None, add_shifted=False, tmin=2, tmax=7,
                                   n_folds=5, alphas=(1e2, 1e3, 1e4, 1e5, 1e6, 1e7), permutation=False,
                                   r2_control=None, min_time_interval=0.050):
    """
    epochs_path: path to folder containing all epochs.fif files
    save_path: path to folder where results will be saved
    confounds: selection of counfounds regressors. If 'all', everything is selected.
    model: 'perfrnn', 'paulrnn' or None. If None, only confounds regressors will be used.
    dataset: dataset to analyze, 'meg' or 'seeg'.
        If 'meg', returns a dictionnary with the ch_names, montage object, and a numpy array with dimensions: n_subjects, n_features, n_channels, n_timepoints
        If 'seeg', returns a dictionnary that contains, for each patient, the ch_names, montage object, and a numpy array with dimensions: n_features, n_channels, n_timepoints
    signal_type: used for seeg only. 'raw' or 'gamma'
    montage: used for seeg only. 'mono' or 'bipo'
    specific_channels: list of channels to pick
    add_shifted: if True, add shifted versions of entropy regressors.
    tmin: remove notes before a given time (to avoid onset response)
    n_folds: If 1, just fit a linear regression. If > 1, performs n-folds cv with ridge regression.
    alphas: list of alpha value to use in ridge regression.
    permutation: If False, run the analysis normally. If an integer, use it as a seed to shuffle the values of the features of interest.
    r2_control: Numpy array of shape (n_subjects, n_channels, n_times), used only if permutations are done to transform r2 scores to t-values.
    min_time_interval: float, keep only notes that are played at least min_time_interval seconds after the previous one.

    Function updated for complete set of regressors
    """

    if data_type == 'meg':
        rate_list = [2.5, 3.5, 4.5, 5.5, 6.5]
    elif data_type == 'seeg':
        rate_list = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]

    alphas = np.array(alphas)

    # create save dir
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # manage random seed for permutations
    if permutation is not False:
        rng = np.random.default_rng(seed=permutation)

    # feature selection
    if confounds is None:
        features = []
    elif confounds == 'all':
        features = ['rate', 'onset_type', 'trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                    'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                    'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']
        if add_shifted:
            shifted_features = ['trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                        'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                        'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']  # with all shifted controls but rate
            for control in shifted_features:
                features.append('shifted_' + control)  # add shifted controls
        else:
            shifted_features = []
    else:
        features = confounds
    if model == 'perfrnn':
        features = features + ['surprise_perfrnn']  # , 'entropy_perfrnn']  <-- we keep only shifted for now
        shifted_features = ['entropy_perfrnn']
        F_toshuffle = ['surprise_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_entropy_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_perfrnn']
    elif model == 'magenta_perfrnn':
        # features = features + ['surprise_magenta_perfrnn']  # , 'entropy_magenta_perfrnn']  <-- we keep only shifted for now
        features = features + ['surprise_magenta_pitch_perfrnn', 'entropy_magenta_pitch_perfrnn',
                               'surprise_magenta_timing_perfrnn', 'entropy_magenta_timing_perfrnn']
        shifted_features = ['entropy_magenta_pitch_perfrnn', 'entropy_magenta_timing_perfrnn']
        F_toshuffle = ['surprise_magenta_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_entropy_magenta_pitch_perfrnn', 'shifted_entropy_magenta_timing_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_magenta_pitch_perfrnn',
                                         'shifted_entropy_magenta_timing_perfrnn']
    elif model == 'paulrnn':
        # features = features + ['surprise_mean_paulrnn', 'surprise_posres_paulrnn', 'surprise_possum_paulrnn',
        #                        'surprise_negsum_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        # features = features + ['surprise_mean_paulrnn']
        features = features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        # shifted_features = ['entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        shifted_features = shifted_features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn'] + ['trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                    'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                    'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']  # with all shifted controls but rate
        F_toshuffle = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']

    # get the correct epochs path depending on the desired dataset
    path = Path(epochs_path)
    if data_type == 'meg':
        epochs_list = sorted(list(path.glob('**/*epo.fif')))
    if data_type == 'seeg':
        if signal_type == 'gamma':
            st = '_gamma'
        elif signal_type == 'raw':
            st = ''
        epochs_list = sorted(list(path.glob('**/*_epochs' + st + '_' + montage + '_preprocessed_IDA.fif')))

    # prepare dict to save output
    if data_type == 'meg':
        output = {'features': features[1:],  # we remove rate from the features list
                  'specific_channels': specific_channels,
                  'r2': [],
                  'betas': [],
                  'n_folds': n_folds,
                  'alphas': alphas,
                  'permutation': permutation,
                  'min_time_interval': min_time_interval,
                  'rates': rate_list}
    elif data_type == 'seeg':
        output = dict()

    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # add more info to dict
        if data_type == 'seeg':
            patient_ID = subject_epochs.name[0:6]
            output[patient_ID] = {'montage_type': montage,
                                  'signal_type': signal_type,
                                  'ch_names': epochs.ch_names,
                                  'epochs_info': epochs.info,
                                  'times': epochs.times,
                                  'specific_channels': specific_channels,
                                  'montage': epochs.get_montage(),
                                  'n_epochs': epochs.__len__(),
                                  'features': features[1:],  # we remove rate from the features list
                                  'subject_betas': None,
                                  'subject_r2': None,
                                  'n_folds': n_folds,
                                  'alphas': alphas,
                                  'permutation': permutation,
                                  'min_time_interval': min_time_interval,
                                  'rates': rate_list}
        elif data_type == 'meg':
            output['times'] = epochs.times
            output['epochs_info'] = epochs.info

        # pick channels
        if specific_channels is not None:
            epochs.pick(specific_channels)

        metadata = epochs.metadata

        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[metadata['time_interval'] >= min_time_interval]
            metadata = epochs.metadata

        # transform metadata to obtain all features
        metadata['pitch'] = metadata['pitch'].apply(lambda x: np.log2(x[0]))  # pitch info was in list format. Apply log transform
        metadata['rate'] = 0  # add rate from stim name
        metadata['onset_type'] = 0  # add onset type from stim name
        metadata['pitch_interval'] = 0  # add pitch interval
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'rate'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[2][0:3])
            metadata.loc[metadata.index[i_note], 'onset_type'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[4][-5])
            if metadata.loc[metadata.index[i_note], 'trial_time'] > 0:
                metadata.loc[metadata.index[i_note], 'pitch_interval'] = np.abs(metadata.loc[metadata.index[i_note-1], 'pitch'] - metadata.loc[metadata.index[i_note], 'pitch'])

        # add shifted features
        if add_shifted:
            for feat in shifted_features:
                metadata['shifted_' + feat] = np.nan  # add shifted entropy
            for i_note in range(metadata.shape[0]):
                if (i_note + 1 < len(metadata)) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note+1], 'stim_name']):
                    for feat in shifted_features:
                        metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note+1], feat]
            # version to try to shift in the other direction
            # for i_note in range(metadata.shape[0]):
            #     if (i_note - 1 >= 0) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note - 1], 'stim_name']):
            #         for feat in shifted_features:
            #             metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note - 1], feat]

            epochs.metadata = metadata
            epochs = epochs[~np.isnan(metadata['shifted_' + shifted_features[0]])]  # remove the last notes of pieces
            metadata = epochs.metadata

        regressors = metadata[features]

        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressors = regressors[regressors['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]
        regressors = regressors[regressors['trial_time'].to_numpy() <= tmax]

        # shuffle features for permutations
        if permutation is not False:
            for feat in F_toshuffle:
                regressors[feat] = rng.permutation(regressors[feat].values)

        # divide by rate
        subject_betas = np.zeros((len(features) - 1, len(epochs.ch_names), epochs.times.shape[0], len(rate_list)))  # -1 because we will remove rate feature
        subject_r2 = np.zeros((len(epochs.ch_names), epochs.times.shape[0], len(rate_list)))
        for r, rate in enumerate(rate_list):

            rate_epochs = epochs[regressors['rate'].to_numpy() == rate]
            rate_regressors = regressors[regressors['rate'].to_numpy() == rate].drop('rate', axis=1)  # we drop rate column

            # load data
            y = rate_epochs.get_data()  # (trials, channels, time)
            x = rate_regressors.to_numpy()

            # standardization of features and meg
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)  # removes the relative importance of subjects, channels, and time points

            del rate_epochs

            # independent regression for every time point and channel
            if n_folds > 1:
                cv = KFold(n_splits=n_folds)
                cv.get_n_splits(x)

            def individual_channel_regression(c_ind):  # TODO: put outside of function
                """function used for multiprocessing"""
                if n_folds == 1:
                    regr = LinearRegression().fit(x, y[:, c_ind, :])
                    # subject_betas[:, c_ind, :] = regr.coef_.T
                    # subject_r2[c_ind, :] = r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
                    return regr.coef_.T, r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
                elif n_folds > 1:
                    cv_scores = np.full((alphas.shape[0], n_folds, y.shape[2]), np.nan)
                    for a, alpha in enumerate(alphas):
                        for j, (train, test) in enumerate(cv.split(x)):
                            regr = Ridge(alpha=alpha)
                            regr.fit(x[train, :], y[train, c_ind, :])
                            pred = regr.predict(x[test, :])
                            cv_scores[a, j, :] = r2_score(y[test, c_ind, :], pred, multioutput='raw_values')
                    # store cv scores and best alphas
                    cv_scores = np.mean(cv_scores, axis=1)  # mean cross-validated r² across folds
                    best_alphas = alphas[np.argmax(cv_scores, axis=0)]  # store best alpha for each timepoint
                    cv_scores = np.max(cv_scores, axis=0)  # for each timepoint, keep the highest cross-val r² across alphas
                    # subject_r2[c_ind, :] = cv_scores
                    # fit again with the best alpha and store betas
                    regr = Ridge(best_alphas).fit(x, y[:, c_ind, :])
                    # subject_betas[:, c_ind, :] = regr.coef_.T
                    return regr.coef_.T, cv_scores

            # multiprocessing channels
            channels_regressions = Parallel(n_jobs=-1)(delayed(individual_channel_regression)(c) for c in range(y.shape[1]))
            # channels_regressions = [individual_channel_regression(c) for c in range(y.shape[1])]
            for c in range(y.shape[1]):
                subject_betas[:, c, :, r] = channels_regressions[c][0]
                subject_r2[c, :, r] = channels_regressions[c][1]

        # store regression results
        if data_type == 'meg':
            output['betas'].append(subject_betas)
            output['r2'].append(subject_r2)
        elif data_type == 'seeg':
            output[patient_ID]['subject_betas'] = subject_betas
            output[patient_ID]['subject_r2'] = subject_r2

    if data_type == 'meg':
        output['betas'] = np.array(output['betas'])
        output['r2'] = np.array(output['r2'])

        # if this is a permutation, store only t-values for r2 and betas
        if permutation is not False:
            output['betas'] = np.mean(output['betas'], axis=0, keepdims=True) / (
                        np.std(output['betas'], axis=0, keepdims=True) / np.sqrt(output['betas'].shape[0]))
            output['r2'] = np.mean(output['r2'] - r2_control, axis=0, keepdims=True) / (
                        np.std(output['r2'], axis=0, keepdims=True) / np.sqrt(
                    output['r2'].shape[0]))  # comparison with control

    if model is None:
        model = 'acoustic'
    if permutation is False:
        permutation = ''
    else:
        permutation = '_perm' + str(permutation).zfill(4)
    if data_type == 'meg':
        with open(os.path.join(save_path, 'regr_by_rate_meg_' + model + permutation + '.pkl'), 'wb') as fp:
            pickle.dump(output, fp)
    elif data_type == 'seeg':
        with open(os.path.join(save_path,
                               'regr_by_rate_seeg_' + signal_type + '_' + montage + '_' + model + permutation + '.pkl'),
                  'wb') as fp:
            pickle.dump(output, fp)

    return output


def regression_predictive_features_by_rep(epochs_path, save_path, confounds='all', model=None, data_type='meg',
                                   signal_type='raw', montage='mono', specific_channels=None, add_shifted=False, tmin=2, tmax=7,
                                   n_folds=5, alphas=(1e2, 1e3, 1e4, 1e5, 1e6, 1e7), permutation=False,
                                   r2_control=None, min_time_interval=0.050):
    """
    epochs_path: path to folder containing all epochs.fif files
    save_path: path to folder where results will be saved
    confounds: selection of counfounds regressors. If 'all', everything is selected.
    model: 'perfrnn', 'paulrnn' or None. If None, only confounds regressors will be used.
    dataset: dataset to analyze, 'meg' or 'seeg'.
        If 'meg', returns a dictionnary with the ch_names, montage object, and a numpy array with dimensions: n_subjects, n_features, n_channels, n_timepoints
        If 'seeg', returns a dictionnary that contains, for each patient, the ch_names, montage object, and a numpy array with dimensions: n_features, n_channels, n_timepoints
    signal_type: used for seeg only. 'raw' or 'gamma'
    montage: used for seeg only. 'mono' or 'bipo'
    specific_channels: list of channels to pick
    add_shifted: if True, add shifted versions of entropy regressors.
    tmin: remove notes before a given time (to avoid onset response)
    n_folds: If 1, just fit a linear regression. If > 1, performs n-folds cv with ridge regression.
    alphas: list of alpha value to use in ridge regression.
    permutation: If False, run the analysis normally. If an integer, use it as a seed to shuffle the values of the features of interest.
    r2_control: Numpy array of shape (n_subjects, n_channels, n_times), used only if permutations are done to transform r2 scores to t-values.
    min_time_interval: float, keep only notes that are played at least min_time_interval seconds after the previous one.

    Function updated for complete set of regressors
    """

    if data_type == 'meg':
        rep_list = [1, 2, 3]
    elif data_type == 'seeg':
        rep_list = [1, 2]

    alphas = np.array(alphas)

    # create save dir
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # manage random seed for permutations
    if permutation is not False:
        rng = np.random.default_rng(seed=permutation)

    # feature selection
    if confounds is None:
        features = []
    elif confounds == 'all':
        features = ['rep', 'rate', 'onset_type', 'trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                    'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                    'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']
        if add_shifted:
            shifted_features = ['trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                        'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                        'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']  # with all shifted controls but rate
            for control in shifted_features:
                features.append('shifted_' + control)  # add shifted controls
        else:
            shifted_features = []
    else:
        features = confounds
    if model == 'perfrnn':
        features = features + ['surprise_perfrnn']  # , 'entropy_perfrnn']  <-- we keep only shifted for now
        shifted_features = ['entropy_perfrnn']
        F_toshuffle = ['surprise_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_entropy_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_perfrnn']
    elif model == 'magenta_perfrnn':
        # features = features + ['surprise_magenta_perfrnn']  # , 'entropy_magenta_perfrnn']  <-- we keep only shifted for now
        features = features + ['surprise_magenta_pitch_perfrnn', 'entropy_magenta_pitch_perfrnn',
                               'surprise_magenta_timing_perfrnn', 'entropy_magenta_timing_perfrnn']
        shifted_features = ['entropy_magenta_pitch_perfrnn', 'entropy_magenta_timing_perfrnn']
        F_toshuffle = ['surprise_magenta_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_entropy_magenta_pitch_perfrnn', 'shifted_entropy_magenta_timing_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_magenta_pitch_perfrnn',
                                         'shifted_entropy_magenta_timing_perfrnn']
    elif model == 'paulrnn':
        features = features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        # shifted_features = ['entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        shifted_features = shifted_features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        F_toshuffle = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']

    # get the correct epochs path depending on the desired dataset
    path = Path(epochs_path)
    if data_type == 'meg':
        epochs_list = sorted(list(path.glob('**/*epo.fif')))
    if data_type == 'seeg':
        if signal_type == 'gamma':
            st = '_gamma'
        elif signal_type == 'raw':
            st = ''
        epochs_list = sorted(list(path.glob('**/*_epochs' + st + '_' + montage + '_preprocessed_IDA.fif')))

    # prepare dict to save output
    if data_type == 'meg':
        output = {'features': features[1:],
                  'specific_channels': specific_channels,
                  'r2': [],
                  'betas': [],
                  'n_folds': n_folds,
                  'alphas': alphas,
                  'permutation': permutation,
                  'min_time_interval': min_time_interval,
                  'rep': rep_list}
    elif data_type == 'seeg':
        output = dict()

    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # add more info to dict
        if data_type == 'seeg':
            patient_ID = subject_epochs.name[0:6]
            output[patient_ID] = {'montage_type': montage,
                                  'signal_type': signal_type,
                                  'ch_names': epochs.ch_names,
                                  'epochs_info': epochs.info,
                                  'times': epochs.times,
                                  'specific_channels': specific_channels,
                                  'montage': epochs.get_montage(),
                                  'n_epochs': epochs.__len__(),
                                  'features': features[1:],
                                  'subject_betas': None,
                                  'subject_r2': None,
                                  'n_folds': n_folds,
                                  'alphas': alphas,
                                  'permutation': permutation,
                                  'min_time_interval': min_time_interval,
                                  'rep': rep_list}
        elif data_type == 'meg':
            output['times'] = epochs.times
            output['epochs_info'] = epochs.info

        # pick channels
        if specific_channels is not None:
            epochs.pick(specific_channels)

        metadata = epochs.metadata

        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[metadata['time_interval'] >= min_time_interval]
            metadata = epochs.metadata

        # transform metadata to obtain all features
        metadata['pitch'] = metadata['pitch'].apply(lambda x: np.log2(x[0]))  # pitch info was in list format. Apply log transform
        metadata['rate'] = 0  # add rate from stim name
        metadata['rep'] = 0  # add rep from stim name
        metadata['onset_type'] = 0  # add onset type from stim name
        metadata['pitch_interval'] = 0  # add pitch interval
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'rate'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[2][0:3])
            metadata.loc[metadata.index[i_note], 'rep'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[1][3])
            metadata.loc[metadata.index[i_note], 'onset_type'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[4][-5])
            if metadata.loc[metadata.index[i_note], 'trial_time'] > 0:
                metadata.loc[metadata.index[i_note], 'pitch_interval'] = np.abs(metadata.loc[metadata.index[i_note-1], 'pitch'] - metadata.loc[metadata.index[i_note], 'pitch'])

        # add shifted features
        if add_shifted:
            for feat in shifted_features:
                metadata['shifted_' + feat] = np.nan  # add shifted entropy
            for i_note in range(metadata.shape[0]):
                if (i_note + 1 < len(metadata)) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note+1], 'stim_name']):
                    for feat in shifted_features:
                        metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note+1], feat]
            # version to try to shift in the other direction
            # for i_note in range(metadata.shape[0]):
            #     if (i_note - 1 >= 0) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note - 1], 'stim_name']):
            #         for feat in shifted_features:
            #             metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note - 1], feat]

            epochs.metadata = metadata
            epochs = epochs[~np.isnan(metadata['shifted_' + shifted_features[0]])]  # remove the last notes of pieces
            metadata = epochs.metadata

        regressors = metadata[features]

        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressors = regressors[regressors['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]
        regressors = regressors[regressors['trial_time'].to_numpy() <= tmax]

        if data_type == 'seeg':  # keep only stim that repeat
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
            epochs = epochs[mask]
            regressors = regressors[mask]

        # shuffle features for permutations
        if permutation is not False:
            for feat in F_toshuffle:
                regressors[feat] = rng.permutation(regressors[feat].values)

        # divide by rate
        subject_betas = np.zeros((len(features) - 1, len(epochs.ch_names), epochs.times.shape[0], len(rep_list)))  # -1 because we will remove rep feature
        subject_r2 = np.zeros((len(epochs.ch_names), epochs.times.shape[0], len(rep_list)))
        for r, rate in enumerate(rep_list):

            rate_epochs = epochs[regressors['rep'].to_numpy() == rate]
            rate_regressors = regressors[regressors['rep'].to_numpy() == rate].drop('rep', axis=1)  # we drop rate column

            # load data
            y = rate_epochs.get_data()  # (trials, channels, time)
            x = rate_regressors.to_numpy()

            # standardization of features and meg
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)  # removes the relative importance of subjects, channels, and time points

            del rate_epochs

            # independent regression for every time point and channel
            if n_folds > 1:
                cv = KFold(n_splits=n_folds)
                cv.get_n_splits(x)

            def individual_channel_regression(c_ind):  # TODO: put outside of function
                """function used for multiprocessing"""
                if n_folds == 1:
                    regr = LinearRegression().fit(x, y[:, c_ind, :])
                    # subject_betas[:, c_ind, :] = regr.coef_.T
                    # subject_r2[c_ind, :] = r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
                    return regr.coef_.T, r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
                elif n_folds > 1:
                    cv_scores = np.full((alphas.shape[0], n_folds, y.shape[2]), np.nan)
                    for a, alpha in enumerate(alphas):
                        for j, (train, test) in enumerate(cv.split(x)):
                            regr = Ridge(alpha=alpha)
                            regr.fit(x[train, :], y[train, c_ind, :])
                            pred = regr.predict(x[test, :])
                            cv_scores[a, j, :] = r2_score(y[test, c_ind, :], pred, multioutput='raw_values')
                    # store cv scores and best alphas
                    cv_scores = np.mean(cv_scores, axis=1)  # mean cross-validated r² across folds
                    best_alphas = alphas[np.argmax(cv_scores, axis=0)]  # store best alpha for each timepoint
                    cv_scores = np.max(cv_scores, axis=0)  # for each timepoint, keep the highest cross-val r² across alphas
                    # subject_r2[c_ind, :] = cv_scores
                    # fit again with the best alpha and store betas
                    regr = Ridge(best_alphas).fit(x, y[:, c_ind, :])
                    # subject_betas[:, c_ind, :] = regr.coef_.T
                    return regr.coef_.T, cv_scores

            # multiprocessing channels
            channels_regressions = Parallel(n_jobs=-1)(delayed(individual_channel_regression)(c) for c in range(y.shape[1]))
            # channels_regressions = [individual_channel_regression(c) for c in range(y.shape[1])]
            for c in range(y.shape[1]):
                subject_betas[:, c, :, r] = channels_regressions[c][0]
                subject_r2[c, :, r] = channels_regressions[c][1]

        # store regression results
        if data_type == 'meg':
            output['betas'].append(subject_betas)
            output['r2'].append(subject_r2)
        elif data_type == 'seeg':
            output[patient_ID]['subject_betas'] = subject_betas
            output[patient_ID]['subject_r2'] = subject_r2

    if data_type == 'meg':
        output['betas'] = np.array(output['betas'])
        output['r2'] = np.array(output['r2'])

        # if this is a permutation, store only t-values for r2 and betas
        if permutation is not False:
            output['betas'] = np.mean(output['betas'], axis=0, keepdims=True) / (
                        np.std(output['betas'], axis=0, keepdims=True) / np.sqrt(output['betas'].shape[0]))
            output['r2'] = np.mean(output['r2'] - r2_control, axis=0, keepdims=True) / (
                        np.std(output['r2'], axis=0, keepdims=True) / np.sqrt(
                    output['r2'].shape[0]))  # comparison with control

    if model is None:
        model = 'acoustic'
    if permutation is False:
        permutation = ''
    else:
        permutation = '_perm' + str(permutation).zfill(4)
    if data_type == 'meg':
        with open(os.path.join(save_path, 'regr_by_rep_meg_' + model + permutation + '.pkl'), 'wb') as fp:
            pickle.dump(output, fp)
    elif data_type == 'seeg':
        with open(os.path.join(save_path,
                               'regr_by_rep_seeg_' + signal_type + '_' + montage + '_' + model + permutation + '.pkl'),
                  'wb') as fp:
            pickle.dump(output, fp)

    return output


def regression_predictive_features_by_average_surprise(epochs_path, save_path, confounds='all', model=None, data_type='meg',
                                   signal_type='raw', montage='mono', specific_channels=None, add_shifted=False, tmin=2, tmax=7,
                                   n_folds=5, alphas=(1e2, 1e3, 1e4, 1e5, 1e6, 1e7), permutation=False,
                                   r2_control=None, min_time_interval=0.050):
    """
    epochs_path: path to folder containing all epochs.fif files
    save_path: path to folder where results will be saved
    confounds: selection of counfounds regressors. If 'all', everything is selected.
    model: 'perfrnn', 'paulrnn' or None. If None, only confounds regressors will be used.
    dataset: dataset to analyze, 'meg' or 'seeg'.
        If 'meg', returns a dictionnary with the ch_names, montage object, and a numpy array with dimensions: n_subjects, n_features, n_channels, n_timepoints
        If 'seeg', returns a dictionnary that contains, for each patient, the ch_names, montage object, and a numpy array with dimensions: n_features, n_channels, n_timepoints
    signal_type: used for seeg only. 'raw' or 'gamma'
    montage: used for seeg only. 'mono' or 'bipo'
    specific_channels: list of channels to pick
    add_shifted: if True, add shifted versions of entropy regressors.
    tmin: remove notes before a given time (to avoid onset response)
    n_folds: If 1, just fit a linear regression. If > 1, performs n-folds cv with ridge regression.
    alphas: list of alpha value to use in ridge regression.
    permutation: If False, run the analysis normally. If an integer, use it as a seed to shuffle the values of the features of interest.
    r2_control: Numpy array of shape (n_subjects, n_channels, n_times), used only if permutations are done to transform r2 scores to t-values.
    min_time_interval: float, keep only notes that are played at least min_time_interval seconds after the previous one.

    Function updated for complete set of regressors
    """

    # avg_surprise_quantiles = [0.2, 0.4, 0.6, 0.8, 1]
    # avg_surprise_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # avg_surprise_quantiles = [0.125, 0.250, 0.375, 0.5, 0.625, 0.750, 0.875, 1]
    avg_surprise_quantiles = [0.16666667, 0.33333333, 0.5, 0.66666667, 0.83333333, 1]

    alphas = np.array(alphas)

    # create save dir
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # manage random seed for permutations
    if permutation is not False:
        rng = np.random.default_rng(seed=permutation)

    # feature selection
    if confounds is None:
        features = []
    elif confounds == 'all':
        features = ['rate', 'onset_type', 'trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                    'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                    'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']
        if add_shifted:
            shifted_features = ['trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                        'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                        'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']  # with all shifted controls but rate
            for control in shifted_features:
                features.append('shifted_' + control)  # add shifted controls
        else:
            shifted_features = []
    else:
        features = confounds
    if model == 'perfrnn':
        features = features + ['surprise_perfrnn']  # , 'entropy_perfrnn']  <-- we keep only shifted for now
        shifted_features = ['entropy_perfrnn']
        F_toshuffle = ['surprise_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_entropy_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_perfrnn']
    elif model == 'magenta_perfrnn':
        # features = features + ['surprise_magenta_perfrnn']  # , 'entropy_magenta_perfrnn']  <-- we keep only shifted for now
        features = features + ['surprise_magenta_pitch_perfrnn', 'entropy_magenta_pitch_perfrnn',
                               'surprise_magenta_timing_perfrnn', 'entropy_magenta_timing_perfrnn']
        shifted_features = ['entropy_magenta_pitch_perfrnn', 'entropy_magenta_timing_perfrnn']
        F_toshuffle = ['surprise_magenta_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_entropy_magenta_pitch_perfrnn', 'shifted_entropy_magenta_timing_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_magenta_pitch_perfrnn',
                                         'shifted_entropy_magenta_timing_perfrnn']
    elif model == 'paulrnn':
        # features = features + ['surprise_mean_paulrnn', 'surprise_posres_paulrnn', 'surprise_possum_paulrnn',
        #                        'surprise_negsum_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        # features = features + ['surprise_mean_paulrnn']
        features = features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        # shifted_features = ['entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        shifted_features = shifted_features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn'] + ['trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                    'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                    'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']  # with all shifted controls but rate
        F_toshuffle = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']

    # get the correct epochs path depending on the desired dataset
    path = Path(epochs_path)
    if data_type == 'meg':
        epochs_list = sorted(list(path.glob('**/*epo.fif')))
    if data_type == 'seeg':
        if signal_type == 'gamma':
            st = '_gamma'
        elif signal_type == 'raw':
            st = ''
        epochs_list = sorted(list(path.glob('**/*_epochs' + st + '_' + montage + '_preprocessed_IDA.fif')))

    # prepare dict to save output
    if data_type == 'meg':
        output = {'features': features,  # we remove rate from the features list
                  'specific_channels': specific_channels,
                  'r2': [],
                  'betas': [],
                  'n_folds': n_folds,
                  'alphas': alphas,
                  'permutation': permutation,
                  'min_time_interval': min_time_interval,
                  'avg_surprise_quantiles': avg_surprise_quantiles}
    elif data_type == 'seeg':
        output = dict()

    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # add more info to dict
        if data_type == 'seeg':
            patient_ID = subject_epochs.name[0:6]
            output[patient_ID] = {'montage_type': montage,
                                  'signal_type': signal_type,
                                  'ch_names': epochs.ch_names,
                                  'epochs_info': epochs.info,
                                  'times': epochs.times,
                                  'specific_channels': specific_channels,
                                  'montage': epochs.get_montage(),
                                  'n_epochs': epochs.__len__(),
                                  'features': features,  # we remove rate from the features list
                                  'subject_betas': None,
                                  'subject_r2': None,
                                  'n_folds': n_folds,
                                  'alphas': alphas,
                                  'permutation': permutation,
                                  'min_time_interval': min_time_interval,
                                  'avg_surprise_quantiles': avg_surprise_quantiles}
        elif data_type == 'meg':
            output['times'] = epochs.times
            output['epochs_info'] = epochs.info

        # pick channels
        if specific_channels is not None:
            epochs.pick(specific_channels)

        metadata = epochs.metadata

        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[metadata['time_interval'] >= min_time_interval]
            metadata = epochs.metadata

        # transform metadata to obtain all features
        metadata['pitch'] = metadata['pitch'].apply(lambda x: np.log2(x[0]))  # pitch info was in list format. Apply log transform
        metadata['rate'] = 0  # add rate from stim name
        metadata['onset_type'] = 0  # add onset type from stim name
        metadata['pitch_interval'] = 0  # add pitch interval
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'rate'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[2][0:3])
            metadata.loc[metadata.index[i_note], 'onset_type'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[4][-5])
            if metadata.loc[metadata.index[i_note], 'trial_time'] > 0:
                metadata.loc[metadata.index[i_note], 'pitch_interval'] = np.abs(metadata.loc[metadata.index[i_note-1], 'pitch'] - metadata.loc[metadata.index[i_note], 'pitch'])

        # add shifted features
        if add_shifted:
            for feat in shifted_features:
                metadata['shifted_' + feat] = np.nan  # add shifted entropy
            for i_note in range(metadata.shape[0]):
                if (i_note + 1 < len(metadata)) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note+1], 'stim_name']):
                    for feat in shifted_features:
                        metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note+1], feat]
            # version to try to shift in the other direction
            # for i_note in range(metadata.shape[0]):
            #     if (i_note - 1 >= 0) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note - 1], 'stim_name']):
            #         for feat in shifted_features:
            #             metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note - 1], feat]

            epochs.metadata = metadata
            epochs = epochs[~np.isnan(metadata['shifted_' + shifted_features[0]])]  # remove the last notes of pieces
            metadata = epochs.metadata

        # add avg_surprise and interaction
        metadata['avg_surprise'] = metadata.groupby('stim_name')['surprise_mean_paulrnn'].transform('mean')

        regressors = metadata[features]

        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressors = regressors[regressors['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]
        regressors = regressors[regressors['trial_time'].to_numpy() <= tmax]

        # shuffle features for permutations
        if permutation is not False:
            for feat in F_toshuffle:
                regressors[feat] = rng.permutation(regressors[feat].values)

        # divide by rate
        subject_betas = np.zeros((len(features), len(epochs.ch_names), epochs.times.shape[0], len(avg_surprise_quantiles)))
        subject_r2 = np.zeros((len(epochs.ch_names), epochs.times.shape[0], len(avg_surprise_quantiles)))
        last_quantile = 0
        for q, quantile in enumerate(avg_surprise_quantiles):

            quantile_value = np.quantile(epochs.metadata['avg_surprise'].to_numpy(), q=quantile)
            quantile_regressors = regressors[(epochs.metadata['avg_surprise'].to_numpy() <= quantile_value) & (epochs.metadata['avg_surprise'].to_numpy() > last_quantile)]
            quantile_epochs = epochs[(epochs.metadata['avg_surprise'].to_numpy() <= quantile_value) & (epochs.metadata['avg_surprise'].to_numpy() > last_quantile)]
            last_quantile = quantile_value

            # load data
            y = quantile_epochs.get_data()  # (trials, channels, time)
            x = quantile_regressors.to_numpy()

            # standardization of features and meg
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)  # removes the relative importance of subjects, channels, and time points

            del quantile_epochs

            # independent regression for every time point and channel
            if n_folds > 1:
                cv = KFold(n_splits=n_folds)
                cv.get_n_splits(x)

            def individual_channel_regression(c_ind):  # TODO: put outside of function
                """function used for multiprocessing"""
                if n_folds == 1:
                    regr = LinearRegression().fit(x, y[:, c_ind, :])
                    # subject_betas[:, c_ind, :] = regr.coef_.T
                    # subject_r2[c_ind, :] = r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
                    return regr.coef_.T, r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
                elif n_folds > 1:
                    cv_scores = np.full((alphas.shape[0], n_folds, y.shape[2]), np.nan)
                    for a, alpha in enumerate(alphas):
                        for j, (train, test) in enumerate(cv.split(x)):
                            regr = Ridge(alpha=alpha)
                            regr.fit(x[train, :], y[train, c_ind, :])
                            pred = regr.predict(x[test, :])
                            cv_scores[a, j, :] = r2_score(y[test, c_ind, :], pred, multioutput='raw_values')
                    # store cv scores and best alphas
                    cv_scores = np.mean(cv_scores, axis=1)  # mean cross-validated r² across folds
                    best_alphas = alphas[np.argmax(cv_scores, axis=0)]  # store best alpha for each timepoint
                    cv_scores = np.max(cv_scores, axis=0)  # for each timepoint, keep the highest cross-val r² across alphas
                    # subject_r2[c_ind, :] = cv_scores
                    # fit again with the best alpha and store betas
                    regr = Ridge(best_alphas).fit(x, y[:, c_ind, :])
                    # subject_betas[:, c_ind, :] = regr.coef_.T
                    return regr.coef_.T, cv_scores

            # multiprocessing channels
            channels_regressions = Parallel(n_jobs=-1)(delayed(individual_channel_regression)(c) for c in range(y.shape[1]))
            # channels_regressions = [individual_channel_regression(c) for c in range(y.shape[1])]
            for c in range(y.shape[1]):
                subject_betas[:, c, :, q] = channels_regressions[c][0]
                subject_r2[c, :, q] = channels_regressions[c][1]

        # store regression results
        if data_type == 'meg':
            output['betas'].append(subject_betas)
            output['r2'].append(subject_r2)
        elif data_type == 'seeg':
            output[patient_ID]['subject_betas'] = subject_betas
            output[patient_ID]['subject_r2'] = subject_r2

    if data_type == 'meg':
        output['betas'] = np.array(output['betas'])
        output['r2'] = np.array(output['r2'])

        # if this is a permutation, store only t-values for r2 and betas
        if permutation is not False:
            output['betas'] = np.mean(output['betas'], axis=0, keepdims=True) / (
                        np.std(output['betas'], axis=0, keepdims=True) / np.sqrt(output['betas'].shape[0]))
            output['r2'] = np.mean(output['r2'] - r2_control, axis=0, keepdims=True) / (
                        np.std(output['r2'], axis=0, keepdims=True) / np.sqrt(
                    output['r2'].shape[0]))  # comparison with control

    if model is None:
        model = 'acoustic'
    if permutation is False:
        permutation = ''
    else:
        permutation = '_perm' + str(permutation).zfill(4)
    if data_type == 'meg':
        with open(os.path.join(save_path, 'regr_by_avg_surprise_meg_' + model + permutation + '.pkl'), 'wb') as fp:
            pickle.dump(output, fp)
    elif data_type == 'seeg':
        with open(os.path.join(save_path, 'regr_by_avg_surprise_seeg_' + signal_type + '_' + montage + '_' + model + permutation + '.pkl'), 'wb') as fp:
            pickle.dump(output, fp)

    return output


def regression_predictive_features_by_onset_type(epochs_path, save_path, confounds='all', model=None, data_type='meg',
                                   signal_type='raw', montage='mono', specific_channels=None, add_shifted=False, tmin=2, tmax=7,
                                   n_folds=5, alphas=(1e2, 1e3, 1e4, 1e5, 1e6, 1e7), permutation=False,
                                   r2_control=None, min_time_interval=0.050):
    """
    epochs_path: path to folder containing all epochs.fif files
    save_path: path to folder where results will be saved
    confounds: selection of counfounds regressors. If 'all', everything is selected.
    model: 'perfrnn', 'paulrnn' or None. If None, only confounds regressors will be used.
    dataset: dataset to analyze, 'meg' or 'seeg'.
        If 'meg', returns a dictionnary with the ch_names, montage object, and a numpy array with dimensions: n_subjects, n_features, n_channels, n_timepoints
        If 'seeg', returns a dictionnary that contains, for each patient, the ch_names, montage object, and a numpy array with dimensions: n_features, n_channels, n_timepoints
    signal_type: used for seeg only. 'raw' or 'gamma'
    montage: used for seeg only. 'mono' or 'bipo'
    specific_channels: list of channels to pick
    add_shifted: if True, add shifted versions of entropy regressors.
    tmin: remove notes before a given time (to avoid onset response)
    n_folds: If 1, just fit a linear regression. If > 1, performs n-folds cv with ridge regression.
    alphas: list of alpha value to use in ridge regression.
    permutation: If False, run the analysis normally. If an integer, use it as a seed to shuffle the values of the features of interest.
    r2_control: Numpy array of shape (n_subjects, n_channels, n_times), used only if permutations are done to transform r2 scores to t-values.
    min_time_interval: float, keep only notes that are played at least min_time_interval seconds after the previous one.

    Function updated for complete set of regressors
    """

    onset_list = [0, 2]

    alphas = np.array(alphas)

    # create save dir
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # manage random seed for permutations
    if permutation is not False:
        rng = np.random.default_rng(seed=permutation)

    # feature selection
    if confounds is None:
        features = []
    elif confounds == 'all':
        features = ['rate', 'onset_type', 'trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                    'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                    'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']
        if add_shifted:
            shifted_features = ['trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                        'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                        'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']  # with all shifted controls but rate
            for control in shifted_features:
                features.append('shifted_' + control)  # add shifted controls
        else:
            shifted_features = []
    else:
        features = confounds
    if model == 'perfrnn':
        features = features + ['surprise_perfrnn']  # , 'entropy_perfrnn']  <-- we keep only shifted for now
        shifted_features = ['entropy_perfrnn']
        F_toshuffle = ['surprise_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_entropy_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_perfrnn']
    elif model == 'magenta_perfrnn':
        # features = features + ['surprise_magenta_perfrnn']  # , 'entropy_magenta_perfrnn']  <-- we keep only shifted for now
        features = features + ['surprise_magenta_pitch_perfrnn', 'entropy_magenta_pitch_perfrnn',
                               'surprise_magenta_timing_perfrnn', 'entropy_magenta_timing_perfrnn']
        shifted_features = ['entropy_magenta_pitch_perfrnn', 'entropy_magenta_timing_perfrnn']
        F_toshuffle = ['surprise_magenta_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_entropy_magenta_pitch_perfrnn', 'shifted_entropy_magenta_timing_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_magenta_pitch_perfrnn',
                                         'shifted_entropy_magenta_timing_perfrnn']
    elif model == 'paulrnn':
        # features = features + ['surprise_mean_paulrnn', 'surprise_posres_paulrnn', 'surprise_possum_paulrnn',
        #                        'surprise_negsum_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        # features = features + ['surprise_mean_paulrnn']
        features = features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        # shifted_features = ['entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        shifted_features = shifted_features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn'] + ['trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                    'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                    'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']  # with all shifted controls but rate
        F_toshuffle = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']

    # get the correct epochs path depending on the desired dataset
    path = Path(epochs_path)
    if data_type == 'meg':
        epochs_list = sorted(list(path.glob('**/*epo.fif')))
    if data_type == 'seeg':
        if signal_type == 'gamma':
            st = '_gamma'
        elif signal_type == 'raw':
            st = ''
        epochs_list = sorted(list(path.glob('**/*_epochs' + st + '_' + montage + '_preprocessed_IDA.fif')))

    # prepare dict to save output
    if data_type == 'meg':
        output = {'features': [features[i] for i in range(len(features)) if i != 1],  # we remove onset from the features list
                  'specific_channels': specific_channels,
                  'r2': [],
                  'betas': [],
                  'n_folds': n_folds,
                  'alphas': alphas,
                  'permutation': permutation,
                  'min_time_interval': min_time_interval,
                  'onset_list': onset_list}
    elif data_type == 'seeg':
        output = dict()

    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # add more info to dict
        if data_type == 'seeg':
            patient_ID = subject_epochs.name[0:6]
            output[patient_ID] = {'montage_type': montage,
                                  'signal_type': signal_type,
                                  'ch_names': epochs.ch_names,
                                  'epochs_info': epochs.info,
                                  'times': epochs.times,
                                  'specific_channels': specific_channels,
                                  'montage': epochs.get_montage(),
                                  'n_epochs': epochs.__len__(),
                                  'features': [features[i] for i in range(len(features)) if i != 1],  # we remove onset from the features list
                                  'subject_betas': None,
                                  'subject_r2': None,
                                  'n_folds': n_folds,
                                  'alphas': alphas,
                                  'permutation': permutation,
                                  'min_time_interval': min_time_interval,
                                  'onset_list': onset_list}
        elif data_type == 'meg':
            output['times'] = epochs.times
            output['epochs_info'] = epochs.info

        # pick channels
        if specific_channels is not None:
            epochs.pick(specific_channels)

        metadata = epochs.metadata

        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[metadata['time_interval'] >= min_time_interval]
            metadata = epochs.metadata

        # transform metadata to obtain all features
        metadata['pitch'] = metadata['pitch'].apply(lambda x: np.log2(x[0]))  # pitch info was in list format. Apply log transform
        metadata['rate'] = 0  # add rate from stim name
        metadata['onset_type'] = 0  # add onset type from stim name
        metadata['pitch_interval'] = 0  # add pitch interval
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'rate'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[2][0:3])
            metadata.loc[metadata.index[i_note], 'onset_type'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[4][-5])
            if metadata.loc[metadata.index[i_note], 'trial_time'] > 0:
                metadata.loc[metadata.index[i_note], 'pitch_interval'] = np.abs(metadata.loc[metadata.index[i_note-1], 'pitch'] - metadata.loc[metadata.index[i_note], 'pitch'])

        # add shifted features
        if add_shifted:
            for feat in shifted_features:
                metadata['shifted_' + feat] = np.nan  # add shifted entropy
            for i_note in range(metadata.shape[0]):
                if (i_note + 1 < len(metadata)) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note+1], 'stim_name']):
                    for feat in shifted_features:
                        metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note+1], feat]
            # version to try to shift in the other direction
            # for i_note in range(metadata.shape[0]):
            #     if (i_note - 1 >= 0) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note - 1], 'stim_name']):
            #         for feat in shifted_features:
            #             metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note - 1], feat]

            epochs.metadata = metadata
            epochs = epochs[~np.isnan(metadata['shifted_' + shifted_features[0]])]  # remove the last notes of pieces
            metadata = epochs.metadata

        regressors = metadata[features]

        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressors = regressors[regressors['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]
        regressors = regressors[regressors['trial_time'].to_numpy() <= tmax]

        # shuffle features for permutations
        if permutation is not False:
            for feat in F_toshuffle:
                regressors[feat] = rng.permutation(regressors[feat].values)

        # divide by rate
        subject_betas = np.zeros((len(features) - 1, len(epochs.ch_names), epochs.times.shape[0], len(onset_list)))  # -1 because we will remove rate feature
        subject_r2 = np.zeros((len(epochs.ch_names), epochs.times.shape[0], len(onset_list)))
        for r, rate in enumerate(onset_list):

            rate_epochs = epochs[regressors['onset_type'].to_numpy() == rate]
            rate_regressors = regressors[regressors['onset_type'].to_numpy() == rate].drop('onset_type', axis=1)  # we drop rate column

            # load data
            y = rate_epochs.get_data()  # (trials, channels, time)
            x = rate_regressors.to_numpy()

            # standardization of features and meg
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)  # removes the relative importance of subjects, channels, and time points

            del rate_epochs

            # independent regression for every time point and channel
            if n_folds > 1:
                cv = KFold(n_splits=n_folds)
                cv.get_n_splits(x)

            def individual_channel_regression(c_ind):  # TODO: put outside of function
                """function used for multiprocessing"""
                if n_folds == 1:
                    regr = LinearRegression().fit(x, y[:, c_ind, :])
                    # subject_betas[:, c_ind, :] = regr.coef_.T
                    # subject_r2[c_ind, :] = r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
                    return regr.coef_.T, r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
                elif n_folds > 1:
                    cv_scores = np.full((alphas.shape[0], n_folds, y.shape[2]), np.nan)
                    for a, alpha in enumerate(alphas):
                        for j, (train, test) in enumerate(cv.split(x)):
                            regr = Ridge(alpha=alpha)
                            regr.fit(x[train, :], y[train, c_ind, :])
                            pred = regr.predict(x[test, :])
                            cv_scores[a, j, :] = r2_score(y[test, c_ind, :], pred, multioutput='raw_values')
                    # store cv scores and best alphas
                    cv_scores = np.mean(cv_scores, axis=1)  # mean cross-validated r² across folds
                    best_alphas = alphas[np.argmax(cv_scores, axis=0)]  # store best alpha for each timepoint
                    cv_scores = np.max(cv_scores, axis=0)  # for each timepoint, keep the highest cross-val r² across alphas
                    # subject_r2[c_ind, :] = cv_scores
                    # fit again with the best alpha and store betas
                    regr = Ridge(best_alphas).fit(x, y[:, c_ind, :])
                    # subject_betas[:, c_ind, :] = regr.coef_.T
                    return regr.coef_.T, cv_scores

            # multiprocessing channels
            channels_regressions = Parallel(n_jobs=-1)(delayed(individual_channel_regression)(c) for c in range(y.shape[1]))
            # channels_regressions = [individual_channel_regression(c) for c in range(y.shape[1])]
            for c in range(y.shape[1]):
                subject_betas[:, c, :, r] = channels_regressions[c][0]
                subject_r2[c, :, r] = channels_regressions[c][1]

        # store regression results
        if data_type == 'meg':
            output['betas'].append(subject_betas)
            output['r2'].append(subject_r2)
        elif data_type == 'seeg':
            output[patient_ID]['subject_betas'] = subject_betas
            output[patient_ID]['subject_r2'] = subject_r2

    if data_type == 'meg':
        output['betas'] = np.array(output['betas'])
        output['r2'] = np.array(output['r2'])

        # if this is a permutation, store only t-values for r2 and betas
        if permutation is not False:
            output['betas'] = np.mean(output['betas'], axis=0, keepdims=True) / (
                        np.std(output['betas'], axis=0, keepdims=True) / np.sqrt(output['betas'].shape[0]))
            output['r2'] = np.mean(output['r2'] - r2_control, axis=0, keepdims=True) / (
                        np.std(output['r2'], axis=0, keepdims=True) / np.sqrt(
                    output['r2'].shape[0]))  # comparison with control

    if model is None:
        model = 'acoustic'
    if permutation is False:
        permutation = ''
    else:
        permutation = '_perm' + str(permutation).zfill(4)
    if data_type == 'meg':
        with open(os.path.join(save_path, 'regr_by_onset_meg_' + model + permutation + '.pkl'), 'wb') as fp:
            pickle.dump(output, fp)
    elif data_type == 'seeg':
        with open(os.path.join(save_path,
                               'regr_by_onset_seeg_' + signal_type + '_' + montage + '_' + model + permutation + '.pkl'),
                  'wb') as fp:
            pickle.dump(output, fp)

    return output


def regression_predictive_features_by_trial_time(epochs_path, save_path, confounds='all', model=None, data_type='meg',
                                   signal_type='raw', montage='mono', specific_channels=None, add_shifted=False, tmin=2, tmax=7,
                                   n_folds=5, alphas=(1e2, 1e3, 1e4, 1e5, 1e6, 1e7), permutation=False,
                                   r2_control=None, min_time_interval=0.050):
    """
    epochs_path: path to folder containing all epochs.fif files
    save_path: path to folder where results will be saved
    confounds: selection of counfounds regressors. If 'all', everything is selected.
    model: 'perfrnn', 'paulrnn' or None. If None, only confounds regressors will be used.
    dataset: dataset to analyze, 'meg' or 'seeg'.
        If 'meg', returns a dictionnary with the ch_names, montage object, and a numpy array with dimensions: n_subjects, n_features, n_channels, n_timepoints
        If 'seeg', returns a dictionnary that contains, for each patient, the ch_names, montage object, and a numpy array with dimensions: n_features, n_channels, n_timepoints
    signal_type: used for seeg only. 'raw' or 'gamma'
    montage: used for seeg only. 'mono' or 'bipo'
    specific_channels: list of channels to pick
    add_shifted: if True, add shifted versions of entropy regressors.
    tmin: remove notes before a given time (to avoid onset response)
    n_folds: If 1, just fit a linear regression. If > 1, performs n-folds cv with ridge regression.
    alphas: list of alpha value to use in ridge regression.
    permutation: If False, run the analysis normally. If an integer, use it as a seed to shuffle the values of the features of interest.
    r2_control: Numpy array of shape (n_subjects, n_channels, n_times), used only if permutations are done to transform r2 scores to t-values.
    min_time_interval: float, keep only notes that are played at least min_time_interval seconds after the previous one.

    Function updated for complete set of regressors
    """

    trial_time_list = ['first_half', 'second_half']

    alphas = np.array(alphas)

    # create save dir
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # manage random seed for permutations
    if permutation is not False:
        rng = np.random.default_rng(seed=permutation)

    # feature selection
    if confounds is None:
        features = []
    elif confounds == 'all':
        features = ['rate', 'onset_type', 'trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                    'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                    'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']
        if add_shifted:
            shifted_features = ['time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                        'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                        'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']  # with all shifted controls but rate
            for control in shifted_features:
                features.append('shifted_' + control)  # add shifted controls
        else:
            shifted_features = []
    else:
        features = confounds
    if model == 'perfrnn':
        features = features + ['surprise_perfrnn']  # , 'entropy_perfrnn']  <-- we keep only shifted for now
        shifted_features = ['entropy_perfrnn']
        F_toshuffle = ['surprise_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_entropy_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_perfrnn']
    elif model == 'magenta_perfrnn':
        # features = features + ['surprise_magenta_perfrnn']  # , 'entropy_magenta_perfrnn']  <-- we keep only shifted for now
        features = features + ['surprise_magenta_pitch_perfrnn', 'entropy_magenta_pitch_perfrnn',
                               'surprise_magenta_timing_perfrnn', 'entropy_magenta_timing_perfrnn']
        shifted_features = ['entropy_magenta_pitch_perfrnn', 'entropy_magenta_timing_perfrnn']
        F_toshuffle = ['surprise_magenta_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_entropy_magenta_pitch_perfrnn', 'shifted_entropy_magenta_timing_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_magenta_pitch_perfrnn',
                                         'shifted_entropy_magenta_timing_perfrnn']
    elif model == 'paulrnn':
        # features = features + ['surprise_mean_paulrnn', 'surprise_posres_paulrnn', 'surprise_possum_paulrnn',
        #                        'surprise_negsum_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        # features = features + ['surprise_mean_paulrnn']
        features = features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        # shifted_features = ['entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        shifted_features = shifted_features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn'] + ['trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                    'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                    'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']  # with all shifted controls but rate
        F_toshuffle = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']

    # get the correct epochs path depending on the desired dataset
    path = Path(epochs_path)
    if data_type == 'meg':
        epochs_list = sorted(list(path.glob('**/*epo.fif')))
    if data_type == 'seeg':
        if signal_type == 'gamma':
            st = '_gamma'
        elif signal_type == 'raw':
            st = ''
        epochs_list = sorted(list(path.glob('**/*_epochs' + st + '_' + montage + '_preprocessed_IDA.fif')))

    # prepare dict to save output
    if data_type == 'meg':
        output = {'features': [features[i] for i in range(len(features)) if i != 1],  # we remove onset from the features list
                  'specific_channels': specific_channels,
                  'r2': [],
                  'betas': [],
                  'n_folds': n_folds,
                  'alphas': alphas,
                  'permutation': permutation,
                  'min_time_interval': min_time_interval,
                  'trial_time_list': trial_time_list}
    elif data_type == 'seeg':
        output = dict()

    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # add more info to dict
        if data_type == 'seeg':
            patient_ID = subject_epochs.name[0:6]
            output[patient_ID] = {'montage_type': montage,
                                  'signal_type': signal_type,
                                  'ch_names': epochs.ch_names,
                                  'epochs_info': epochs.info,
                                  'times': epochs.times,
                                  'specific_channels': specific_channels,
                                  'montage': epochs.get_montage(),
                                  'n_epochs': epochs.__len__(),
                                  'features': [features[i] for i in range(len(features)) if i != 1],  # we remove onset from the features list
                                  'subject_betas': None,
                                  'subject_r2': None,
                                  'n_folds': n_folds,
                                  'alphas': alphas,
                                  'permutation': permutation,
                                  'min_time_interval': min_time_interval,
                                  'trial_time_list': trial_time_list}
        elif data_type == 'meg':
            output['times'] = epochs.times
            output['epochs_info'] = epochs.info

        # pick channels
        if specific_channels is not None:
            epochs.pick(specific_channels)

        metadata = epochs.metadata

        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[metadata['time_interval'] >= min_time_interval]
            metadata = epochs.metadata

        # transform metadata to obtain all features
        metadata['pitch'] = metadata['pitch'].apply(lambda x: np.log2(x[0]))  # pitch info was in list format. Apply log transform
        metadata['rate'] = 0  # add rate from stim name
        metadata['onset_type'] = 0  # add onset type from stim name
        metadata['pitch_interval'] = 0  # add pitch interval
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'rate'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[2][0:3])
            metadata.loc[metadata.index[i_note], 'onset_type'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[4][-5])
            if metadata.loc[metadata.index[i_note], 'trial_time'] > 0:
                metadata.loc[metadata.index[i_note], 'pitch_interval'] = np.abs(metadata.loc[metadata.index[i_note-1], 'pitch'] - metadata.loc[metadata.index[i_note], 'pitch'])

        # add shifted features
        if add_shifted:
            for feat in shifted_features:
                metadata['shifted_' + feat] = np.nan  # add shifted entropy
            for i_note in range(metadata.shape[0]):
                if (i_note + 1 < len(metadata)) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note+1], 'stim_name']):
                    for feat in shifted_features:
                        metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note+1], feat]
            # version to try to shift in the other direction
            # for i_note in range(metadata.shape[0]):
            #     if (i_note - 1 >= 0) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note - 1], 'stim_name']):
            #         for feat in shifted_features:
            #             metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note - 1], feat]

            epochs.metadata = metadata
            epochs = epochs[~np.isnan(metadata['shifted_' + shifted_features[0]])]  # remove the last notes of pieces
            metadata = epochs.metadata

        regressors = metadata[features]

        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressors = regressors[regressors['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]
        regressors = regressors[regressors['trial_time'].to_numpy() <= tmax]

        # shuffle features for permutations
        if permutation is not False:
            for feat in F_toshuffle:
                regressors[feat] = rng.permutation(regressors[feat].values)

        # divide by rate
        subject_betas = np.zeros((len(features) - 1, len(epochs.ch_names), epochs.times.shape[0], len(trial_time_list)))  # -1 because we will remove rate feature
        subject_r2 = np.zeros((len(epochs.ch_names), epochs.times.shape[0], len(trial_time_list)))
        for r, rate in enumerate(trial_time_list):

            if r == 0:
                rate_epochs = epochs[regressors['trial_time'].to_numpy() < 4.5]
                rate_regressors = regressors[regressors['trial_time'].to_numpy() < 4.5].drop('trial_time', axis=1)  # we drop trial_time column
            if r == 1:
                rate_epochs = epochs[regressors['trial_time'].to_numpy() > 4.5]
                rate_regressors = regressors[regressors['trial_time'].to_numpy() > 4.5].drop('trial_time', axis=1)  # we drop trial_time column

            # load data
            y = rate_epochs.get_data()  # (trials, channels, time)
            x = rate_regressors.to_numpy()

            # standardization of features and meg
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)  # removes the relative importance of subjects, channels, and time points

            del rate_epochs

            # independent regression for every time point and channel
            if n_folds > 1:
                cv = KFold(n_splits=n_folds)
                cv.get_n_splits(x)

            def individual_channel_regression(c_ind):  # TODO: put outside of function
                """function used for multiprocessing"""
                if n_folds == 1:
                    regr = LinearRegression().fit(x, y[:, c_ind, :])
                    # subject_betas[:, c_ind, :] = regr.coef_.T
                    # subject_r2[c_ind, :] = r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
                    return regr.coef_.T, r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
                elif n_folds > 1:
                    cv_scores = np.full((alphas.shape[0], n_folds, y.shape[2]), np.nan)
                    for a, alpha in enumerate(alphas):
                        for j, (train, test) in enumerate(cv.split(x)):
                            regr = Ridge(alpha=alpha)
                            regr.fit(x[train, :], y[train, c_ind, :])
                            pred = regr.predict(x[test, :])
                            cv_scores[a, j, :] = r2_score(y[test, c_ind, :], pred, multioutput='raw_values')
                    # store cv scores and best alphas
                    cv_scores = np.mean(cv_scores, axis=1)  # mean cross-validated r² across folds
                    best_alphas = alphas[np.argmax(cv_scores, axis=0)]  # store best alpha for each timepoint
                    cv_scores = np.max(cv_scores, axis=0)  # for each timepoint, keep the highest cross-val r² across alphas
                    # subject_r2[c_ind, :] = cv_scores
                    # fit again with the best alpha and store betas
                    regr = Ridge(best_alphas).fit(x, y[:, c_ind, :])
                    # subject_betas[:, c_ind, :] = regr.coef_.T
                    return regr.coef_.T, cv_scores

            # multiprocessing channels
            channels_regressions = Parallel(n_jobs=-1)(delayed(individual_channel_regression)(c) for c in range(y.shape[1]))
            # channels_regressions = [individual_channel_regression(c) for c in range(y.shape[1])]
            for c in range(y.shape[1]):
                subject_betas[:, c, :, r] = channels_regressions[c][0]
                subject_r2[c, :, r] = channels_regressions[c][1]

        # store regression results
        if data_type == 'meg':
            output['betas'].append(subject_betas)
            output['r2'].append(subject_r2)
        elif data_type == 'seeg':
            output[patient_ID]['subject_betas'] = subject_betas
            output[patient_ID]['subject_r2'] = subject_r2

    if data_type == 'meg':
        output['betas'] = np.array(output['betas'])
        output['r2'] = np.array(output['r2'])

        # if this is a permutation, store only t-values for r2 and betas
        if permutation is not False:
            output['betas'] = np.mean(output['betas'], axis=0, keepdims=True) / (
                        np.std(output['betas'], axis=0, keepdims=True) / np.sqrt(output['betas'].shape[0]))
            output['r2'] = np.mean(output['r2'] - r2_control, axis=0, keepdims=True) / (
                        np.std(output['r2'], axis=0, keepdims=True) / np.sqrt(
                    output['r2'].shape[0]))  # comparison with control

    if model is None:
        model = 'acoustic'
    if permutation is False:
        permutation = ''
    else:
        permutation = '_perm' + str(permutation).zfill(4)
    if data_type == 'meg':
        with open(os.path.join(save_path, 'regr_by_trial_time_meg_' + model + permutation + '.pkl'), 'wb') as fp:
            pickle.dump(output, fp)
    elif data_type == 'seeg':
        with open(os.path.join(save_path,
                               'regr_by_trial_time_seeg_' + signal_type + '_' + montage + '_' + model + permutation + '.pkl'),
                  'wb') as fp:
            pickle.dump(output, fp)

    return output



# def regression_predictive_features_by_rate(epochs_path, save_path, rate_list, features=None, specific_channels=None, rms=True, tmin=1, shifted=False):
#     """tmin: remove notes before a given time (to avoid onset response)"""
#
#     if features is None:
#         features = ['surprise', 'entropy']
#
#     path = Path(epochs_path)
#     epochs_list = sorted(list(path.glob('**/*epo.fif')))
#
#     betas = []
#     for i, subject_epochs in enumerate(epochs_list):
#         print(subject_epochs.name)
#         epochs = mne.read_epochs(subject_epochs)
#
#         # times = epochs.times
#
#         if specific_channels is not None:
#             epochs.pick(specific_channels)
#
#         # transform metadata to obtain all features
#         regressors = from_metadata_to_regressors(epochs.metadata, multiple_pitches='max', shifted=shifted)  # keep only max pitch if multiple notes
#
#         # remove onset response
#         epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
#         regressors = regressors[regressors['trial_time'].to_numpy() >= tmin]
#
#         if rms:
#             subject_betas = np.zeros((len(features), epochs.times.shape[0], len(rate_list)))
#         else:
#             subject_betas = np.zeros((len(features), len(epochs.ch_names), epochs.times.shape[0], len(rate_list)))
#         for r, rate in enumerate(rate_list):
#
#             rate_epochs = epochs[regressors['rate'].to_numpy() == rate]
#             rate_regressors = regressors[regressors['rate'].to_numpy() == rate]
#
#             y = rate_epochs.get_data()  # (epochs, channels, time)
#             # x = epochs.metadata[features].to_numpy()
#             x = rate_regressors[features].to_numpy()
#
#             del rate_epochs
#
#             # standardization of features and meg
#             x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
#
#             if rms:
#                 if rms == 'mean':
#                     y = np.mean(y, axis=1)  # regression on average channel
#                 else:
#                     y = np.sqrt(np.mean(y**2, axis=1))  # regression on RMS
#                 # y = (y - np.mean(y, axis=1, keepdims=True)) / np.std(y, axis=1, keepdims=True)  # removes the relative importance of subjects and time points
#                 y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)  # removes the relative importance of subjects and time points
#             else:
#                 # y = (y - np.mean(y, axis=(1, 2), keepdims=True)) / np.std(y, axis=(1, 2), keepdims=True)  # removes the relative importance of subjects, channels, and time points
#                 y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)  # removes the relative importance of subjects, channels, and time points
#
#             # independent regression for every time point
#             if rms:
#                 for t in range(y.shape[1]):
#                     regr = LinearRegression().fit(x, y[:, t])
#                     subject_betas[:, t, r] = regr.coef_
#             else:
#                 for t in range(y.shape[2]):
#                     for c in range(y.shape[1]):
#                         regr = LinearRegression().fit(x, y[:, c, t])
#                         subject_betas[:, c, t, r] = regr.coef_
#
#         betas.append(subject_betas)
#
#     betas = np.array(betas)  # shape (subjects, features, channels, times, rates)
#
#     if rms:
#         np.save(os.path.join(save_path, 'regression_betas_predictive_features_by_rate_rms.npy'), betas)
#     else:
#         np.save(os.path.join(save_path, 'regression_betas_predictive_features_by_rate.npy'), betas)
#
#     return


def regression_predictive_features_by_frequency(epochs_path, save_path, confounds='all', model=None, data_type='meg',
                                   signal_type='raw', montage='mono', specific_channels=None, add_shifted=False, tmin=2, tmax=7,
                                   n_folds=5, alphas=(1e2, 1e3, 1e4, 1e5, 1e6, 1e7), permutation=False,
                                   r2_control=None, min_time_interval=0.050):
    """
    epochs_path: path to folder containing all epochs.fif files
    save_path: path to folder where results will be saved
    confounds: selection of counfounds regressors. If 'all', everything is selected.
    model: 'perfrnn', 'paulrnn' or None. If None, only confounds regressors will be used.
    dataset: dataset to analyze, 'meg' or 'seeg'.
        If 'meg', returns a dictionnary with the ch_names, montage object, and a numpy array with dimensions: n_subjects, n_features, n_channels, n_timepoints
        If 'seeg', returns a dictionnary that contains, for each patient, the ch_names, montage object, and a numpy array with dimensions: n_features, n_channels, n_timepoints
    signal_type: used for seeg only. 'raw' or 'gamma'
    montage: used for seeg only. 'mono' or 'bipo'
    specific_channels: list of channels to pick
    add_shifted: if True, add shifted versions of entropy regressors.
    tmin: remove notes before a given time (to avoid onset response)
    n_folds: If 1, just fit a linear regression. If > 1, performs n-folds cv with ridge regression.
    alphas: list of alpha value to use in ridge regression.
    permutation: If False, run the analysis normally. If an integer, use it as a seed to shuffle the values of the features of interest.
    r2_control: Numpy array of shape (n_subjects, n_channels, n_times), used only if permutations are done to transform r2 scores to t-values.
    min_time_interval: float, keep only notes that are played at least min_time_interval seconds after the previous one.

    Function updated for complete set of regressors
    """

    alphas = np.array(alphas)

    # create save dir
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # manage random seed for permutations
    if permutation is not False:
        rng = np.random.default_rng(seed=permutation)

    # feature selection
    if confounds is None:
        features = []
    elif confounds == 'all':
        features = ['rate', 'onset_type', 'trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                    'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                    'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']
        if add_shifted:
            shifted_features = ['trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                        'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                        'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']  # with all shifted controls but rate
            for control in shifted_features:
                features.append('shifted_' + control)  # add shifted controls
        else:
            shifted_features = []
    else:
        features = confounds
    if model == 'perfrnn':
        features = features + ['surprise_perfrnn']  # , 'entropy_perfrnn']  <-- we keep only shifted for now
        shifted_features = ['entropy_perfrnn']
        F_toshuffle = ['surprise_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_entropy_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_perfrnn']
    elif model == 'magenta_perfrnn':
        # features = features + ['surprise_magenta_perfrnn']  # , 'entropy_magenta_perfrnn']  <-- we keep only shifted for now
        features = features + ['surprise_magenta_pitch_perfrnn', 'entropy_magenta_pitch_perfrnn',
                               'surprise_magenta_timing_perfrnn', 'entropy_magenta_timing_perfrnn']
        shifted_features = ['entropy_magenta_pitch_perfrnn', 'entropy_magenta_timing_perfrnn']
        F_toshuffle = ['surprise_magenta_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_entropy_magenta_pitch_perfrnn', 'shifted_entropy_magenta_timing_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_magenta_pitch_perfrnn',
                                         'shifted_entropy_magenta_timing_perfrnn']
    elif model == 'paulrnn':
        # features = features + ['surprise_mean_paulrnn', 'surprise_posres_paulrnn', 'surprise_possum_paulrnn',
        #                        'surprise_negsum_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        # features = features + ['surprise_mean_paulrnn']
        features = features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        # shifted_features = ['entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        shifted_features = shifted_features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn'] + ['trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                    'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                    'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']  # with all shifted controls but rate
        F_toshuffle = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']

    # get the correct epochs path depending on the desired dataset
    path = Path(epochs_path)
    if data_type == 'meg':
        epochs_list = sorted(list(path.glob('**/*epo-tfr.h5')))
    if data_type == 'seeg':
        raise NotImplemented
        if signal_type == 'gamma':
            st = '_gamma'
        elif signal_type == 'raw':
            st = ''
        epochs_list = sorted(list(path.glob('**/*_epochs' + st + '_' + montage + '_preprocessed_IDA.fif')))

    # prepare dict to save output
    if data_type == 'meg':
        output = {'features': features,
                  'specific_channels': specific_channels,
                  'r2': [],
                  'betas': [],
                  'n_folds': n_folds,
                  'alphas': alphas,
                  'permutation': permutation,
                  'min_time_interval': min_time_interval}
    elif data_type == 'seeg':
        output = dict()

    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.time_frequency.read_tfrs(subject_epochs)[0]

        # add more info to dict
        if data_type == 'seeg':
            patient_ID = subject_epochs.name[0:6]
            output[patient_ID] = {'montage_type': montage,
                                  'signal_type': signal_type,
                                  'ch_names': epochs.ch_names,
                                  'epochs_info': epochs.info,
                                  'times': epochs.times,
                                  'specific_channels': specific_channels,
                                  'montage': epochs.get_montage(),
                                  'n_epochs': epochs.__len__(),
                                  'features': features,
                                  'subject_betas': None,
                                  'subject_r2': None,
                                  'n_folds': n_folds,
                                  'alphas': alphas,
                                  'permutation': permutation,
                                  'min_time_interval': min_time_interval,
                                  'rates': rate_list}
        elif data_type == 'meg':
            output['times'] = epochs.times
            output['epochs_info'] = epochs.info
            output['freqs'] = epochs.freqs

        # pick channels
        if specific_channels is not None:
            epochs.pick(specific_channels)

        metadata = epochs.metadata

        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[metadata['time_interval'] >= min_time_interval]
            metadata = epochs.metadata

        # transform metadata to obtain all features
        metadata['pitch'] = metadata['pitch'].apply(lambda x: np.log2(x[0]))  # pitch info was in list format. Apply log transform
        metadata['rate'] = 0  # add rate from stim name
        metadata['onset_type'] = 0  # add onset type from stim name
        metadata['pitch_interval'] = 0  # add pitch interval
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'rate'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[2][0:3])
            metadata.loc[metadata.index[i_note], 'onset_type'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[4][-5])
            if metadata.loc[metadata.index[i_note], 'trial_time'] > 0:
                metadata.loc[metadata.index[i_note], 'pitch_interval'] = np.abs(metadata.loc[metadata.index[i_note-1], 'pitch'] - metadata.loc[metadata.index[i_note], 'pitch'])

        # add shifted features
        if add_shifted:
            for feat in shifted_features:
                metadata['shifted_' + feat] = np.nan  # add shifted entropy
            for i_note in range(metadata.shape[0]):
                if (i_note + 1 < len(metadata)) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note+1], 'stim_name']):
                    for feat in shifted_features:
                        metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note+1], feat]
            # version to try to shift in the other direction
            # for i_note in range(metadata.shape[0]):
            #     if (i_note - 1 >= 0) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note - 1], 'stim_name']):
            #         for feat in shifted_features:
            #             metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note - 1], feat]

            epochs.metadata = metadata
            epochs = epochs[~np.isnan(metadata['shifted_' + shifted_features[0]])]  # remove the last notes of pieces
            metadata = epochs.metadata

        regressors = metadata[features]

        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressors = regressors[regressors['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]
        regressors = regressors[regressors['trial_time'].to_numpy() <= tmax]

        # shuffle features for permutations
        if permutation is not False:
            for feat in F_toshuffle:
                regressors[feat] = rng.permutation(regressors[feat].values)

        # divide by rate
        subject_betas = np.zeros((len(features), len(epochs.ch_names), epochs.times.shape[0], len(epochs.freqs)))
        subject_r2 = np.zeros((len(epochs.ch_names), epochs.times.shape[0], len(epochs.freqs)))
        for r, rate in enumerate(epochs.freqs):

            # load data
            y = epochs.data[:, :, r, :]  # (trials, channels, time)
            x = regressors.to_numpy()

            # standardization of features and meg
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)  # removes the relative importance of subjects, channels, and time points

            # independent regression for every time point and channel
            if n_folds > 1:
                cv = KFold(n_splits=n_folds)
                cv.get_n_splits(x)

            def individual_channel_regression(c_ind):  # TODO: put outside of function
                """function used for multiprocessing"""
                if n_folds == 1:
                    regr = LinearRegression().fit(x, y[:, c_ind, :])
                    # subject_betas[:, c_ind, :] = regr.coef_.T
                    # subject_r2[c_ind, :] = r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
                    return regr.coef_.T, r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
                elif n_folds > 1:
                    cv_scores = np.full((alphas.shape[0], n_folds, y.shape[2]), np.nan)
                    for a, alpha in enumerate(alphas):
                        for j, (train, test) in enumerate(cv.split(x)):
                            regr = Ridge(alpha=alpha)
                            regr.fit(x[train, :], y[train, c_ind, :])
                            pred = regr.predict(x[test, :])
                            cv_scores[a, j, :] = r2_score(y[test, c_ind, :], pred, multioutput='raw_values')
                    # store cv scores and best alphas
                    cv_scores = np.mean(cv_scores, axis=1)  # mean cross-validated r² across folds
                    best_alphas = alphas[np.argmax(cv_scores, axis=0)]  # store best alpha for each timepoint
                    cv_scores = np.max(cv_scores, axis=0)  # for each timepoint, keep the highest cross-val r² across alphas
                    # subject_r2[c_ind, :] = cv_scores
                    # fit again with the best alpha and store betas
                    regr = Ridge(best_alphas).fit(x, y[:, c_ind, :])
                    # subject_betas[:, c_ind, :] = regr.coef_.T
                    return regr.coef_.T, cv_scores

            # multiprocessing channels
            channels_regressions = Parallel(n_jobs=-1)(delayed(individual_channel_regression)(c) for c in range(y.shape[1]))
            # channels_regressions = [individual_channel_regression(c) for c in range(y.shape[1])]
            for c in range(y.shape[1]):
                subject_betas[:, c, :, r] = channels_regressions[c][0]
                subject_r2[c, :, r] = channels_regressions[c][1]

        # store regression results
        if data_type == 'meg':
            output['betas'].append(subject_betas)
            output['r2'].append(subject_r2)
        elif data_type == 'seeg':
            output[patient_ID]['subject_betas'] = subject_betas
            output[patient_ID]['subject_r2'] = subject_r2

    if data_type == 'meg':
        output['betas'] = np.array(output['betas'])
        output['r2'] = np.array(output['r2'])

        # if this is a permutation, store only t-values for r2 and betas
        if permutation is not False:
            output['betas'] = np.mean(output['betas'], axis=0, keepdims=True) / (
                        np.std(output['betas'], axis=0, keepdims=True) / np.sqrt(output['betas'].shape[0]))
            output['r2'] = np.mean(output['r2'] - r2_control, axis=0, keepdims=True) / (
                        np.std(output['r2'], axis=0, keepdims=True) / np.sqrt(
                    output['r2'].shape[0]))  # comparison with control

    if model is None:
        model = 'acoustic'
    if permutation is False:
        permutation = ''
    else:
        permutation = '_perm' + str(permutation).zfill(4)
    if data_type == 'meg':
        with open(os.path.join(save_path, 'regr_by_freq_meg_' + model + permutation + '.pkl'), 'wb') as fp:
            pickle.dump(output, fp)
    elif data_type == 'seeg':
        with open(os.path.join(save_path,
                               'regr_by_freq_seeg_' + signal_type + '_' + montage + '_' + model + permutation + '.pkl'),
                  'wb') as fp:
            pickle.dump(output, fp)

    return output


def regression_shuffled_predictive_features(epochs_path, save_path, n_shuffle=1000):

    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))

    rng = np.random.default_rng()

    betas = []
    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # times = epochs.times

        y = epochs.get_data()
        x = epochs.metadata[['surprise', 'entropy']].to_numpy()

        # standardization
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

        del epochs

        shuffled_betas = np.zeros((n_shuffle, x.shape[1], y.shape[1], y.shape[2]))
        for n in range(n_shuffle):
            print(n)
            # shuffle features values in time
            x_shuffled = rng.permuted(x, axis=0)
            # independent regression for every time point

            for t in range(y.shape[2]):
                for c in range(y.shape[1]):
                    regr = LinearRegression().fit(x_shuffled, y[:, c, t])
                    shuffled_betas[n, :, c, t] = regr.coef_

        betas.append(np.mean(shuffled_betas, axis=0))

    betas = np.array(betas)

    np.save(os.path.join(save_path, 'regression_betas_shuffled_predictive_features.npy'), betas)

    return

def experiment_duration_analysis():
    """Assess the impact of experiment duration on the results, running several fits with varying portions of the data"""
    # proportions = [0.2, 1.]
    proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    for proportion in proportions:
        print('#########################################################################################################')
        print('Proportion:', proportion)
        regression_predictive_features('/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
                                       '/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/experiment_duration_analysis',
                                       confounds='all',
                                       # confounds=['trial_time'],
                                       # model=None,
                                       model='paulrnn',
                                       # model='magenta_perfrnn',
                                       # model='perfrnn',
                                       data_type='meg',
                                       signal_type='raw',
                                       montage='mono',
                                       specific_channels=None,
                                       # specific_channels=['MEG1941', 'MEG1721', 'MEG1641', 'MEG1631', 'MEG1611', 'MEG2421', 'MEG2411', 'MEG2431', 'MEG2441', 'MEG2221'],
                                       # add_shifted=False,
                                       add_shifted=True,
                                       tmin=2,
                                       tmax=7,
                                       n_folds=5,
                                       alphas=np.logspace(2, 7, 6),
                                       permutation=False,
                                       min_time_interval=0.050,
                                       best_channels_path=None,
                                       # best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_channels.npy',
                                       # baseline=True
                                       baseline=False,
                                       avg_surprise=False,
                                       proportion=proportion
                                       )

def plot_duration_analysis(regression_folder):
    """
    :param r_squared_list: list of numpy scores (1 per feature of interest) of shape (n_subjects, n_channels, times)
    :param control_r_squared: numpy scores of control only of shape (n_subjects, n_channels, times)
    :param epochs: used for times and info
    :param features_list: names of the features
    :param tmin:
    :return:
    """
    plt.switch_backend('TkAgg')

    path = Path(regression_folder)
    regr_list = sorted(list(path.glob('*percent.pkl')))

    # remove acoustic
    regr_list = [regr for regr in regr_list if regr.name[9:17] != 'acoustic']
    regr_list = regr_list[1:] + [regr_list[0]]

    # cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#6859D9', '#ff5a5f'])
    cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#ff5a5f', '#6859D9'])
    color = [cmap1(i) for i in np.linspace(0, 1, len(regr_list))]
    fig, axes = plt.subplots(1, 1)
    for i, model_regression_path in enumerate(tqdm(regr_list)):
        with open(model_regression_path, 'rb') as file:
            model_regression = pickle.load(file)
        control_regression_path = model_regression_path
        control_regression_path = control_regression_path.parent / control_regression_path.name.replace("paulrnn", 'acoustic')

        with open(control_regression_path, 'rb') as file:
            control_regression = pickle.load(file)

        control_r_squared = control_regression['r2']
        model_r_squared = model_regression['r2']
        r2_diff = model_r_squared - control_r_squared
        times = model_regression['times']

        # reshape to (n_subjects, times, n_channels)
        r2_diff = np.swapaxes(r2_diff, 1, 2)
        # Perform spatio-temporal cluster test
        threshold = None  # Default threshold
        n_permutations = 1000  # Number of permutations
        ch_adjacency, ch_names = find_ch_adjacency(model_regression['epochs_info'], 'mag')
        T_obs, clusters, cluster_pv, _ = spatio_temporal_cluster_1samp_test(
            r2_diff,
            threshold=threshold,
            n_permutations=n_permutations,
            tail=0,  # Two-tailed test
            n_jobs=-1,  # Parallelize if possible
            adjacency=ch_adjacency,
            # verbose=False
        )
        r2_diff = np.swapaxes(r2_diff, 1, 2)  # back to (n_subjects, n_channels, times)
        # Identify significant times
        significant_clusters = [cluster for (cluster, p_value) in zip(clusters, cluster_pv) if p_value < 0.05]
        # significant_clusters = np.array([cluster for (cluster, p_value) in zip(clusters, cluster_pv) if p_value < 0.05])
        # significant_clusters = significant_clusters[np.argsort([cluster[0][0] for cluster in significant_clusters])]  # sort by time
        masked_times = np.full(r2_diff.shape[2], True)  # non-significant times will be masked
        for c, cluster in enumerate(significant_clusters):
            start_sample, end_sample = cluster[0][0], cluster[0][-1]
            masked_times[start_sample:end_sample] = False  # store significant times for later

        # model_r_squared = np.mean(model_r_squared, axis=1)  # average channels
        model_r_squared = np.mean(r2_diff, axis=1)  # average channels
        # sem = np.std(model_r_squared, axis=0) / np.sqrt(model_r_squared.shape[0])

        if i == len(regr_list) - 1:
            percent = model_regression_path.name[17:20]
        else:
            percent = model_regression_path.name[17:19]
        axes.plot(times, np.mean(model_r_squared, axis=0), color=color[i], label=percent + '%')
        # axes.fill_between(times, np.mean(model_r_squared, axis=0) - sem, np.mean(model_r_squared, axis=0) + sem, alpha=0.6)

        # if lower_lim is None:
        #     lower_lim = axes.get_ylim()[0]
        mask = np.ma.masked_where(masked_times, axes.get_ylim()[0] * np.ones(r2_diff.shape[2]))  # we mask what is not significant
        axes.plot(model_regression['times'], mask, color=color[i], linewidth=2)

    axes.legend()

    axes.axvline(0, linestyle='--', color='k', zorder=0)
    axes.axhline(0, linestyle='--', color='k', zorder=0)

    axes.set_title('Cross-validated score with percent of trials used')
    axes.set_ylabel('Delta R²')
    axes.set_xlabel('Time (s)')

    plt.show()

    return


def plot_duration_analysis_by_subject(regression_folder):
    """
    :param r_squared_list: list of numpy scores (1 per feature of interest) of shape (n_subjects, n_channels, times)
    :param control_r_squared: numpy scores of control only of shape (n_subjects, n_channels, times)
    :param epochs: used for times and info
    :param features_list: names of the features
    :param tmin:
    :return:
    """
    plt.switch_backend('TkAgg')

    path = Path(regression_folder)
    regr_list = sorted(list(path.glob('*percent.pkl')))

    # remove acoustic
    regr_list = [regr for regr in regr_list if regr.name[9:17] != 'acoustic']
    regr_list = regr_list[1:] + [regr_list[0]]

    correlations = []  # Store correlations for each percent
    percent_labels = []  # Store labels for boxplot

    # Load the 100% regression first for reference
    with open(regr_list[-1], 'rb') as file:
        model_regression_100 = pickle.load(file)
    control_regression_path = regr_list[-1].parent / regr_list[-1].name.replace("paulrnn", 'acoustic')
    with open(control_regression_path, 'rb') as file:
        control_regression_100 = pickle.load(file)
    # Compute the 100% difference (model - acoustic)
    r2_diff_100 = np.mean(model_regression_100['r2'] - control_regression_100['r2'], axis=1)

    for i, model_regression_path in enumerate(tqdm(regr_list[:-1])):
        with open(model_regression_path, 'rb') as file:
            model_regression = pickle.load(file)
        control_regression_path = model_regression_path
        control_regression_path = control_regression_path.parent / control_regression_path.name.replace("paulrnn", 'acoustic')
        with open(control_regression_path, 'rb') as file:
            control_regression = pickle.load(file)

        # Compute difference (model - acoustic) for the current percent
        r2_diff = np.mean(model_regression['r2'] - control_regression['r2'], axis=1)

        # Compute correlations for each subject
        percent_corr = [
            scipy.stats.pearsonr(r2_diff_100[subj, 50:], r2_diff[subj, 50:])[0]  # Correlation for each subject
            # scipy.stats.pearsonr(r2_diff_100[subj], r2_diff[subj])[0]  # Correlation for each subject
            for subj in range(r2_diff_100.shape[0])
        ]
        correlations.append(percent_corr)

        percent = model_regression_path.name[17:19]
        percent_labels.append(percent + '%')

    fig, ax = plt.subplots()
    ax.boxplot(correlations, labels=percent_labels, vert=True, showfliers=False)
    ax.set_title('Correlation of ΔR² at different percents with 100%')
    ax.set_ylabel('Pearson Correlation with 100% trials')
    ax.set_xlabel('Percent of Trials Used')

    # Add individual dots
    for i, percent_corr in enumerate(correlations):
        x_positions = np.random.normal(i + 1, 0.04, size=len(percent_corr))  # Add jitter for clarity
        ax.scatter(x_positions, percent_corr, alpha=0.6, color='#ff5a5f', edgecolor='black', zorder=2, s=10)

    ax.axhline(0, linestyle='--', color='k', zorder=0)


    # all time courses per subject
    # cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#6859D9', '#ff5a5f'])
    cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#ff5a5f', '#6859D9'])
    color = [cmap1(i) for i in np.linspace(0, 1, len(regr_list))]
    fig, axes = plt.subplots(6, 5, sharex=True, sharey=True)
    for i, model_regression_path in enumerate(tqdm(regr_list)):
        with open(model_regression_path, 'rb') as file:
            model_regression = pickle.load(file)
        control_regression_path = model_regression_path
        control_regression_path = control_regression_path.parent / control_regression_path.name.replace("paulrnn",
                                                                                                        'acoustic')

        with open(control_regression_path, 'rb') as file:
            control_regression = pickle.load(file)

        control_r_squared = control_regression['r2']
        model_r_squared = model_regression['r2']
        r2_diff = model_r_squared - control_r_squared
        times = model_regression['times']

        model_r_squared = np.mean(r2_diff, axis=1)  # average channels

        if i == len(regr_list) - 1:
            percent = model_regression_path.name[17:20]
        else:
            percent = model_regression_path.name[17:19]

        for s in range(model_r_squared.shape[0]):
            axes.flat[s].plot(times[50:], model_r_squared[s, 50:], color=color[i], label=percent + '%')

    for s in range(model_r_squared.shape[0]):
        axes.flat[s].axvline(0, linestyle='--', color='k', zorder=0)
        axes.flat[s].axhline(0, linestyle='--', color='k', zorder=0)

    # axes.legend()
    # axes.set_title('Cross-validated score with percent of trials used')
    # axes.set_ylabel('Delta R²')
    # axes.set_xlabel('Time (s)')

    plt.show()









    plt.show()

    return


def plot_regression_betas(betas_path, features_path, tmin=-1, tmax=1, multiple_channels=True, features_plot='all', backend = None):
    '''
    betas_path: path to betas files
    features_names: list of name of features used in the regression analysis
    tmin: low boundary of time-window
    tmax: high boundary of time-window
    multiple_channels: 'mean', 'abs' or 'False'. Use if multiple channels are present.
    features_plot: 'all', or list of features to plot
    backend: matplotlib backend to switch to before plotting. If None, the current backend is used. Otherwise, choices are 'TkAgg', 'QtAgg'

    Function updated for complete set of regressors
    '''

    features_names = np.load(features_path)
    if features_plot == 'all':
        features_plot_idx = [_ for _ in range(len(features_names))]
    else:
        features_plot_idx = [features_names.tolist().index(_) for _ in features_plot]

    # loading params
    betas = np.load(betas_path)  # betas of shape (n_subjects, n_features, n_channels, times) or (n_subjects, n_features, times)

    # mean across channels
    if multiple_channels == 'abs':
        # betas = np.mean(betas, axis=2)
        betas = np.mean(np.abs(betas), axis=2)
        # betas = np.sqrt(np.mean(betas**2, axis=2))
    elif multiple_channels == 'mean':
        betas = np.mean(betas, axis=2)
    elif multiple_channels != False:
        raise ValueError("multiple_channels should be in {'mean', 'abs', 'False'}")

    # timings
    times = np.linspace(tmin, tmax, betas.shape[2])

    betas_tscores = np.mean(betas, axis=0) / (np.std(betas, axis=0) / np.sqrt(betas.shape[0]))

    if backend is not None:
        if backend == 'TkAgg':
            plt.switch_backend('TkAgg')
        if backend == 'QtAgg':
            plt.switch_backend('QtAgg')

    fig, axes = plt.subplots(3, 1, sharex=True)

    # t-values
    for f in features_plot_idx:
        axes[0].plot(times, betas_tscores[f, :], linewidth=3)
    # axes[0].plot(times, betas_tscores[1, :], linewidth=3)
    axes[0].axvline(0, linestyle='--', color='k', zorder=0)
    axes[0].axhline(0, linestyle='--', color='k', zorder=0)
    axes[0].set_ylabel('Regression Beta (t-value)')
    axes[0].legend(features_names[features_plot_idx])
    fixed_golden_aspect_ratio(axes[0])


    for f in features_plot_idx:
        axes[1].plot(times, np.mean(betas[:, f, :], axis=0), linewidth=3)
        sem = np.std(betas[:, f, :], axis=0) / np.sqrt(betas.shape[0])
        axes[1].fill_between(times, np.mean(betas[:, f, :], axis=0) - sem, np.mean(betas[:, f, :], axis=0) + sem, alpha=0.6)
    # axes[1].plot(times, np.mean(betas[:, 1, :], axis=0), linewidth=3)
    # sem = np.std(betas[:, 1, :], axis=0) / np.sqrt(betas.shape[0])
    # axes[1].fill_between(times, np.mean(betas[:, 1, :], axis=0) - sem, np.mean(betas[:, 1, :], axis=0) + sem, alpha=0.6)
    axes[1].axvline(0, linestyle='--', color='k', zorder=0)
    axes[1].axhline(0, linestyle='--', color='k', zorder=0)
    axes[1].set_ylabel('Regression Beta')
    fixed_golden_aspect_ratio(axes[1])

    for f in features_plot_idx:
        line = axes[2].plot(times, np.mean(betas[:, f, :], axis=0), linewidth=3, zorder=2)
        axes[2].plot(times, betas[:, f, :].T, line[0].get_color(), linewidth=1, alpha=0.25, zorder=1)
    # axes[2].plot(times, betas[:, 1, :].T, 'navajowhite', linewidth=1)
    # axes[2].plot(times, betas[:, 0, :].T, 'skyblue', linewidth=1)
    # axes[2].plot(times, np.mean(betas[:, 0, :], axis=0), linewidth=3)
    # axes[2].plot(times, np.mean(betas[:, 1, :], axis=0), linewidth=3)
    axes[2].axvline(0, linestyle='--', color='k', zorder=0)
    axes[2].axhline(0, linestyle='--', color='k', zorder=0)
    axes[2].set_xlabel('Times (s)')
    axes[2].set_ylabel('Regression Beta')
    fixed_golden_aspect_ratio(axes[2])

    plt.show()

    return


def plot_regression_main_effects(npy_path, tmin=-0.2, tmax=0.5, individual_subjects=False, multiple_channels=True, features=None):

    if features is None:
        features = ['Surprise', 'Entropy']

    # loading params
    betas = np.load(npy_path)  # betas of shape (n_subjects, n_features, n_channels, times) or (n_subjects, n_features, times)

    # mean across channels
    if multiple_channels == 'abs':
        # betas = np.mean(betas, axis=2)
        betas = np.mean(np.abs(betas), axis=2)
        # betas = np.sqrt(np.mean(betas**2, axis=2))
    elif multiple_channels == 'mean':
        betas = np.mean(betas, axis=2)
    elif multiple_channels != False:
        raise ValueError("multiple_channels should be in {'mean', 'abs', 'False'}")

    # timings
    times = np.linspace(tmin, tmax, betas.shape[2])

    plt.switch_backend('TkAgg')
    fig, axes = plt.subplots(ceil(len(features) / 3), 3, sharex=True, sharey=True)

    for f in range(len(features)):
        if individual_subjects:
            line = axes.flat[f].plot(times, np.mean(betas[:, f, :], axis=0), linewidth=3, zorder=2)
            axes.flat[f].plot(times, betas[:, f, :].T, line[0].get_color(), linewidth=1, alpha=0.25, zorder=1)
        else:
            axes.flat[f].plot(times, np.mean(betas[:, f, :], axis=0), linewidth=3)
            sem = np.std(betas[:, f, :], axis=0) / np.sqrt(betas.shape[0])
            axes.flat[f].fill_between(times, np.mean(betas[:, f, :], axis=0) - sem, np.mean(betas[:, f, :], axis=0) + sem, alpha=0.6)
    # axes[2].plot(times, betas[:, 1, :].T, 'navajowhite', linewidth=1)
    # axes[2].plot(times, betas[:, 0, :].T, 'skyblue', linewidth=1)
    # axes[2].plot(times, np.mean(betas[:, 0, :], axis=0), linewidth=3)
    # axes[2].plot(times, np.mean(betas[:, 1, :], axis=0), linewidth=3)
        axes.flat[f].axvline(0, linestyle='--', color='k', zorder=0)
        axes.flat[f].axhline(0, linestyle='--', color='k', zorder=0)
        axes.flat[f].set_xlabel('Times (s)')
        axes.flat[f].set_ylabel('Regression Beta')
        axes.flat[f].legend([features[f]])
        # fixed_golden_aspect_ratio(axes.flat[f])

    plt.show()

    return


def plot_topo_regression_betas(betas_path, features_path, epochs_path, time=None, features_plot='all', backend = None):
    '''
    Topographical plot of beta values from regression analysis
    betas_path: path to betas files
    features_path: list of name of features used in the regression analysis
    epochs_path: path to epochs
    time: time range (smaller than [-1:1] sec) within which betas values are averaged
    features_plot: 'all', or list of features to plot
    backend: matplotlib backend to switch to before plotting. If None, the current backend is used. Otherwise, choices are 'TkAgg', 'QtAgg'

    '''

    if time is None:
        time = [0.08, 0.13]

    if backend is not None:
        if backend == 'TkAgg':
            plt.switch_backend('TkAgg')
        if backend == 'QtAgg':
            plt.switch_backend('QtAgg')

    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epochs.fif')))

    features_names = np.load(features_path)
    if features_plot == 'all':
        features_plot_idx = [_ for _ in range(len(features_names))]
    else:
        features_plot_idx = [features_names.tolist().index(_) for _ in features_plot]

    #load epochs
    epochs = mne.read_epochs(epochs_list[0])
    time_sample = [np.where(epochs.times == time[0])[0][0], np.where(epochs.times == time[1])[0][0]]

    #load betas
    betas = np.load(betas_path) #dim : n_subject, n_features, n_channels, n_time_points
    betas = betas[:,:,:,time_sample[0]:time_sample[1]] #select betas within the desired time range
    betas = np.mean(betas, axis=3) #average beta values within that time-range

    beta_t_values = np.mean(betas, axis=0) / (np.std(betas, axis=0) / np.sqrt(betas.shape[0]))
    # beta_t_values = np.squeeze(betas[0,:,:]) #just to check one subject

    # Topo plot
    fig, axes = plt.subplots(len(features_plot_idx), 1)
    vmax = np.max(np.abs(beta_t_values))
    for i, f in enumerate(features_plot_idx):
        im, cn = mne.viz.plot_topomap(beta_t_values[f, :], epochs.info, show=False, axes=axes[i], ch_type='mag', vmin=-vmax, vmax=vmax)  # plot t-scores
        clb = fig.colorbar(im, ax=axes[i])
        if f == 0:
            clb.ax.set_title('Beta t-value', fontsize=15)  # title on top of colorbar
        axes[i].set_title(features_names[f] + '['+str(time[0])+', '+str(time[1])+'] s')

    plt.show()

    return


def spatio_temporal_cluster_regression_betas(model_regression, features_to_plot, n_permut=1000, averaging_method='mean', channel_subset=False):
    """
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times)
    :param epochs: used for times and info
    :param feature_list: must be of length n_features
    :param tmin:
    :param channel_subset: set it to True if the regression was not done on all channels. Topomaps cannot be plotted in that case.
    :return:
    """
    plt.switch_backend('TkAgg')

    betas = model_regression['betas']

    if averaging_method == 'mean':
        def averaging_fct(params): return np.mean(params, axis=0)
    elif averaging_method == 't-values':
        def averaging_fct(params): return np.mean(params, axis=0) / (np.std(params, axis=0) / np.sqrt(params.shape[0]))

    feature_indices = [model_regression['features'].index(feat) for feat in features_to_plot]

    fig_all_features, axes_all_features = plt.subplots(3, ceil(len(features_to_plot) / 3), sharex=True, sharey=True, squeeze=False)
    mask_list = []
    mask_y_value = 0

    for plot_i, f in enumerate(feature_indices):
        feature_betas = betas[:, f, :, :]
        if not channel_subset:
            # reshape to (n_subjects, times, n_channels)
            feature_betas = np.swapaxes(feature_betas, 1, 2)

            ch_adjacency, ch_names = find_ch_adjacency(model_regression['epochs_info'], 'mag')
            pval = 0.05  # arbitrary
            df = feature_betas.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
            thresh = scipy.stats.t.ppf(1 - pval / 2, df=df)  # T distribution, two-tailed
            # t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(feature_betas, threshold=dict(start=0, step=0.8), n_permutations=n_permut, tail=0,
            #                                                                      adjacency=ch_adjacency, n_jobs=-1)
            t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(feature_betas,
                                                                                 threshold=thresh,
                                                                                 n_permutations=n_permut, tail=0,
                                                                                 adjacency=ch_adjacency, n_jobs=-1)

            feature_betas = np.swapaxes(feature_betas, 1, 2)  # back to (n_subjects, n_channels, times)

            significant_clusters = np.array([cluster for (cluster, p_value) in zip(clusters, cluster_pv) if p_value < 0.05])
            significant_clusters = significant_clusters[np.argsort([cluster[0][0] for cluster in significant_clusters])]  # sort by time
            n_significant_clusters = len(significant_clusters)
            channels_indices = np.arange(feature_betas.shape[1])
            masked_times = np.full(feature_betas.shape[2], True)  # non-significant times will be masked

            height_ratios = [1 for _ in range(ceil(n_significant_clusters / 5))]
            height_ratios.append(3)
            fig, axes = plt.subplots(ceil(n_significant_clusters / 5) + 1, 5, gridspec_kw={'height_ratios': height_ratios}, squeeze=False)  # +1 row for the temporal plot

            for i, cluster in enumerate(significant_clusters):
                start_sample, end_sample = cluster[0][0], cluster[0][-1]
                masked_times[start_sample:end_sample] = False  # store significant times for later
                feature_betas_topo = np.mean(feature_betas[:, :, start_sample:end_sample], axis=2)
                mne.viz.plot_topomap(averaging_fct(feature_betas_topo), model_regression['epochs_info'], axes=axes.flat[i], show=False, mask=np.isin(channels_indices, cluster[1]))
                axes.flat[i].set_title('[' + str(model_regression['times'][start_sample]) + ', ' + str(model_regression['times'][end_sample]) + '] s')

            print(n_significant_clusters)
            if n_significant_clusters % 5 > 0:
                for remaining_axes in range(n_significant_clusters, n_significant_clusters + 5 - n_significant_clusters % 5):
                    axes.flat[remaining_axes].remove()

            # merge axes of last row
            gs = axes[-1][0].get_gridspec()
            for ax in axes[-1]:
                ax.remove()
            ax_row = fig.add_subplot(gs[-1, :])
            evoked_betas = mne.EvokedArray(averaging_fct(feature_betas), info=model_regression['epochs_info'], tmin=model_regression['times'][0])
            evoked_betas.plot(show=False, axes=ax_row, unit=False, zorder='std', gfp=True, spatial_colors=True)
            ax_row.axvline(0, linestyle='--', color='k', zorder=0)

            mask = np.ma.masked_where(masked_times, ax_row.get_ylim()[0] * np.ones(feature_betas.shape[2]))  # we mask what is not significant
            ax_row.plot(model_regression['times'], mask, 'k', linewidth=2)

            fixed_golden_aspect_ratio(ax_row)

            ax_row.set_title(features_to_plot[plot_i])
            if averaging_method == 'mean':
                ax_row.set_ylabel('Regression parameter (a.u.)')
            elif averaging_method == 't-values':
                ax_row.set_ylabel('t-values')

        # replot in the general figure
        if channel_subset:
            channel_rms = np.sqrt(np.mean(feature_betas**2, axis=1))
            axes_all_features.flat[plot_i].plot(model_regression['times'], averaging_fct(channel_rms), linewidth=3)
        else:
            evoked_betas.plot(show=False, axes=axes_all_features.flat[plot_i], unit=False, zorder='std', gfp=False, spatial_colors=True)
        axes_all_features.flat[plot_i].axvline(0, linestyle='--', color='k', zorder=0)
        axes_all_features.flat[plot_i].set_title(features_to_plot[plot_i])
        if averaging_method == 'mean':
            axes_all_features.flat[plot_i].set_ylabel('Regression parameter (a.u.)')
        elif averaging_method == 't-values':
            axes_all_features.flat[plot_i].set_ylabel('t-values')
        if not channel_subset:
            mask_list.append(masked_times)
            mask_y_value = min(mask_y_value, axes_all_features.flat[plot_i].get_ylim()[0])

    # plot significant lines separately to have them at the same y value
    if not channel_subset:
        for plot_i in range(len(features_to_plot)):
            mask = np.ma.masked_where(mask_list[plot_i], mask_y_value * np.ones(feature_betas.shape[2]))  # we mask what is not significant
            axes_all_features.flat[plot_i].plot(model_regression['times'], mask, 'k', linewidth=2)

    for plot_i in range(len(features_to_plot)):
        fixed_golden_aspect_ratio(axes_all_features.flat[plot_i])

    for remaining_axes in range(axes_all_features.size - len(features_to_plot)):
        axes_all_features.flat[-remaining_axes-1].remove()

    plt.show()

    return


def regression_betas_ica(model_regression, control_regression, ica_path, features_to_plot, averaging_method='mean', nb_components=10):
    """
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times)
    :param epochs: used for times and info
    :param feature_list: must be of length n_features
    :param tmin:
    :param channel_subset: set it to True if the regression was not done on all channels. Topomaps cannot be plotted in that case.
    :return:
    """
    plt.switch_backend('TkAgg')

    times = model_regression['times']
    info = model_regression['epochs_info']
    betas = model_regression['betas']
    r2_diff = model_regression['r2'] - control_regression['r2']

    ica_list = sorted(list(Path(ica_path).glob('**/*ica.fif')))

    if averaging_method == 'mean':
        def averaging_fct(params): return np.mean(params, axis=0)
    elif averaging_method == 't-values':
        def averaging_fct(params): return np.mean(params, axis=0) / (np.std(params, axis=0) / np.sqrt(params.shape[0]))

    feature_indices = [model_regression['features'].index(feat) for feat in features_to_plot]

    for plot_i, f in enumerate(feature_indices):
        feature_betas = betas[:, f, :, :]
        # reshape to (n_subjects, times, n_channels)

        fig, axes = plt.subplots(6, 5, sharex=True, sharey=True, squeeze=False)  # +1 row for the temporal plot

        time_samples = [np.where(times == -1)[0][0],
                        np.where(times == 1)[0][0]]
        # time_samples = [np.where(times == -0.780)[0][0],
        #                 np.where(times == -0.580)[0][0]]

        print(features_to_plot[plot_i])
        betas_list = []
        components_list = []
        for subject in range(feature_betas.shape[0]):
            # sorted_components = np.argsort(np.mean(r2_diff[subject, :, :], axis=1), axis=0)
            sorted_components = np.argsort(np.mean(np.abs(feature_betas[subject, :, time_samples[0]:time_samples[1]]), axis=1), axis=0)
            # sorted_components = np.argsort(np.mean(np.abs(feature_betas[subject, :, :]), axis=1), axis=0)
            axes.flat[subject].plot(times, feature_betas[subject, sorted_components[-nb_components:], :].T)
            axes.flat[subject].axvline(0, linestyle='--', color='k', zorder=0)
            axes.flat[subject].axhline(0, linestyle='--', color='k', zorder=0)

            betas_list.append(np.abs(feature_betas[subject, sorted_components[-nb_components:], :]).T)

            print(f'subject {subject}', sorted_components[-nb_components:])
            # print('r2', np.mean(r2_diff[subject, sorted_components[-nb_components:], time_samples[0]:time_samples[1]], axis=1))

            print(ica_list[subject].name)
            ica = read_ica(ica_list[subject])
            components_list.append(np.abs(ica.get_components()[:, sorted_components[-nb_components:]]))
            # ica.plot_components(picks=sorted_components[-nb_components:], show=False, title=(f'subject {subject}'))

        # print(n_significant_clusters)
        # if n_significant_clusters % 5 > 0:
        #     for remaining_axes in range(n_significant_clusters, n_significant_clusters + 5 - n_significant_clusters % 5):
        #         axes.flat[remaining_axes].remove()

        # fixed_golden_aspect_ratio(ax_row)

        # ax_row.set_title(features_to_plot[plot_i])
        # if averaging_method == 'mean':
        #     ax_row.set_ylabel('Regression parameter (a.u.)')
        # elif averaging_method == 't-values':
        #     ax_row.set_ylabel('t-values')

        for remaining_axes in range(axes.size - len(betas)):
            axes.flat[-remaining_axes-1].remove()

        fig.suptitle(features_to_plot[plot_i])

        betas_list = np.average(np.array(betas_list), axis=0)
        fig, ax = plt.subplots(1, 1)
        ax.plot(times, betas_list)
        ax.axvline(0, linestyle='--', color='k', zorder=0)
        ax.axhline(0, linestyle='--', color='k', zorder=0)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Regression betas (a.u.)')
        fig.suptitle(features_to_plot[plot_i])

        components_list = np.average(np.array(components_list), axis=0)
        fig, axes = plt.subplots(1, nb_components)
        for compo in range(nb_components):
            mne.viz.plot_topomap(components_list[:, nb_components-1-compo], ica.info, axes=axes.flat[compo], show=False)

        plt.show()

    fig, axes = plt.subplots(6, 5, sharex=True, sharey=True, squeeze=False)  # +1 row for the temporal plot
    print('r2')
    for subject in range(feature_betas.shape[0]):
        sorted_components = np.argsort(np.mean(r2_diff[subject, :, :], axis=1), axis=0)
        axes.flat[subject].plot(times, r2_diff[subject, sorted_components[-nb_components:], :].T)
        axes.flat[subject].axvline(0, linestyle='--', color='k', zorder=0)
        axes.flat[subject].axhline(0, linestyle='--', color='k', zorder=0)

        print(f'subject {subject}', sorted_components[-nb_components:])

    for remaining_axes in range(axes.size - len(r2_diff)):
        axes.flat[-remaining_axes-1].remove()
    fig.suptitle('r2 diff')

    plt.show()

    return


def spatio_temporal_cluster_regression_scores(model_regression, control_regression):
    """
    :param r_squared_list: list of numpy scores (1 per feature of interest) of shape (n_subjects, n_channels, times)
    :param control_r_squared: numpy scores of control only of shape (n_subjects, n_channels, times)
    :param epochs: used for times and info
    :param features_list: names of the features
    :param tmin:
    :return:
    """
    plt.switch_backend('TkAgg')

    control_r_squared = control_regression['r2']
    model_r_squared = model_regression['r2']
    r2_diff = model_r_squared - control_r_squared
    times = model_regression['times']

    fig, axes = plt.subplots(2, 2, sharex=True, squeeze=False)  # sharey='col'
    # axes.flat[-1].remove()
    lines = []

    control_r_squared = np.mean(control_r_squared, axis=1)
    control_sem = np.std(control_r_squared, axis=0) / np.sqrt(control_r_squared.shape[0])
    lines.append(axes.flat[1].plot(times, np.mean(control_r_squared, axis=0)))
    axes.flat[1].fill_between(times, np.mean(control_r_squared, axis=0) - control_sem, np.mean(control_r_squared, axis=0) + control_sem, alpha=0.6)

    model_r_squared = np.mean(model_r_squared, axis=1)  # average channels
    sem = np.std(model_r_squared, axis=0) / np.sqrt(model_r_squared.shape[0])
    lines.append(axes.flat[1].plot(times, np.mean(model_r_squared, axis=0)))
    axes.flat[1].fill_between(times, np.mean(model_r_squared, axis=0) - sem, np.mean(model_r_squared, axis=0) + sem, alpha=0.6)
    diff = model_r_squared - control_r_squared
    diff_sem = np.std(diff, axis=0) / np.sqrt(diff.shape[0])
    lines.append(axes.flat[1].plot(times, np.mean(diff, axis=0)))
    axes.flat[1].fill_between(times, np.mean(diff, axis=0) - diff_sem, np.mean(diff, axis=0) + diff_sem, alpha=0.6)

    axes.flat[1].legend([line[0] for line in lines], ['Control features only', 'All features', 'Difference'])

    axes.flat[1].axvline(0, linestyle='--', color='k', zorder=0)
    axes.flat[1].axhline(0, linestyle='--', color='k', zorder=0)

    fixed_golden_aspect_ratio(axes.flat[1])

    axes.flat[1].set_title('Channel average')
    axes.flat[1].set_ylabel('R²')
    axes.flat[1].set_xlabel('Time (s)')

    # plot the difference only
    axes.flat[3].plot(times, np.mean(diff, axis=0), 'g', linewidth=2)
    axes.flat[3].fill_between(times, np.mean(diff, axis=0) - diff_sem, np.mean(diff, axis=0) + diff_sem, color='g', alpha=0.6)
    axes.flat[3].axvline(0, linestyle='--', color='k', zorder=0)
    axes.flat[3].axhline(0, linestyle='--', color='k', zorder=0)
    fixed_golden_aspect_ratio(axes.flat[3])
    axes.flat[3].set_title('Channel average (All features - control)')
    axes.flat[3].set_ylabel('R²')
    axes.flat[3].set_xlabel('Time (s)')

    # plot channel scores individually
    model_r_squared = model_regression['r2']
    evoked_score = mne.EvokedArray(np.mean(model_r_squared, axis=0), info=model_regression['epochs_info'], tmin=model_regression['times'][0])
    evoked_score.plot(show=False, axes=axes.flat[0], unit=False, zorder='std')
    axes.flat[0].axvline(0, linestyle='--', color='k', zorder=0)
    axes.flat[0].set_ylabel('R²')
    axes.flat[0].set_title('All features')
    fixed_golden_aspect_ratio(axes.flat[0])

    evoked_score_diff = mne.EvokedArray(np.mean(r2_diff, axis=0), info=model_regression['epochs_info'], tmin=model_regression['times'][0])
    evoked_score_diff.plot(show=False, axes=axes.flat[2], unit=False, zorder='std')
    axes.flat[2].axvline(0, linestyle='--', color='k', zorder=0)
    axes.flat[2].set_ylabel('R²')
    axes.flat[2].set_title('All features - Control features only')
    fixed_golden_aspect_ratio(axes.flat[2])

    # remove nave added by mne
    for child in axes.flat[0].get_children():
        if isinstance(child, matplotlib.text.Annotation):
            if child.xy == (0, 1):
                child.remove()
    for child in axes.flat[2].get_children():
        if isinstance(child, matplotlib.text.Annotation):
            if child.xy == (0, 1):
                child.remove()

    plt.show()

    return


def spatio_temporal_cluster_regression_betas_by_rate(model_regression, features_to_plot, n_permut=1000):
    """
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times, rates)
    :param epochs: used for times and info
    :param feature_list: must be of length n_features
    :param tmin:
    :return:
    """
    plt.switch_backend('TkAgg')

    fig_list = []

    betas = model_regression['betas']

    feature_indices = [model_regression['features'].index(feat) for feat in features_to_plot]

    for f, feature in enumerate(feature_indices):
        fig, axes = plt.subplots(2, 3, sharey=True, squeeze=False)
        lower_lim = None
        for r, rate in enumerate(model_regression['rates']):

            feature_betas = betas[:, feature, :, :, r]
            # reshape to (n_subjects, times, n_channels)
            feature_betas = np.swapaxes(feature_betas, 1, 2)

            ch_adjacency, ch_names = find_ch_adjacency(model_regression['epochs_info'], 'mag')
            t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(feature_betas, n_permutations=n_permut,
                                                                                 tail=0, adjacency=ch_adjacency, n_jobs=-1)

            feature_betas = np.swapaxes(feature_betas, 1, 2)  # back to (n_subjects, n_channels, times)

            significant_clusters = np.array([cluster for (cluster, p_value) in zip(clusters, cluster_pv) if p_value < 0.05])
            significant_clusters = significant_clusters[np.argsort([cluster[0][0] for cluster in significant_clusters])]  # sort by time
            masked_times = np.full(feature_betas.shape[2], True)  # non-significant times will be masked

            for i, cluster in enumerate(significant_clusters):
                start_sample, end_sample = cluster[0][0], cluster[0][-1]
                masked_times[start_sample:end_sample] = False  # store significant times for later

            evoked_betas = mne.EvokedArray(np.mean(feature_betas, axis=0), info=model_regression['epochs_info'], tmin=model_regression['times'][0])
            evoked_betas.plot(show=False, axes=axes.flat[r], zorder='std')
            axes.flat[r].axvline(0, linestyle='--', color='k', zorder=0)

            # dashed gray line at expected previous and next notes
            period = 1
            while axes.flat[r].get_xlim()[0] < - period / rate:
                axes.flat[r].axvline(- period / rate, linestyle='--', color='gray', zorder=0, linewidth=0.5)
                period += 1
            period = 1
            while axes.flat[r].get_xlim()[1] > period / rate:
                axes.flat[r].axvline(period / rate, linestyle='--', color='gray', zorder=0, linewidth=0.5)
                period += 1

            if lower_lim is None:
                lower_lim = axes.flat[r].get_ylim()[0]
            mask = np.ma.masked_where(masked_times, lower_lim * np.ones(feature_betas.shape[2]))  # we mask what is not significant
            axes.flat[r].plot(model_regression['times'], mask, 'k', linewidth=2)

            fixed_golden_aspect_ratio(axes.flat[r])

            axes.flat[r].set_title(str(rate) + " Hz")
            axes.flat[r].set_ylabel('Regression parameter (a.u.)')

        for ax in range(1, 6 - len(model_regression['rates']) + 1):
            axes.flat[-ax].remove()
        fig.suptitle(features_to_plot[f])
        print(feature)
        fig_list.append(fig)

    plt.show()

    return


def spatio_temporal_cluster_regression_scores_by_rate(model_regression, control_regression):
    """
    :param r_squared_list: list of numpy scores (1 per feature of interest) of shape (n_subjects, n_channels, times)
    :param control_r_squared: numpy scores of control only of shape (n_subjects, n_channels, times)
    :param epochs: used for times and info
    :param features_list: names of the features
    :param tmin:
    :return:
    """
    plt.switch_backend('TkAgg')

    control_r_squared = control_regression['r2']
    model_r_squared = model_regression['r2']
    r2_diff = model_r_squared - control_r_squared
    times = model_regression['times']

    lines = []
    fig, axes = plt.subplots(2, 3, sharey=True, squeeze=False)
    lower_lim = None
    for r, rate in enumerate(model_regression['rates']):
        rate_r2_diff = r2_diff[:, :, :, r]  # select rate and average subjects

        evoked_score_diff = mne.EvokedArray(np.mean(rate_r2_diff, axis=0), info=model_regression['epochs_info'], tmin=model_regression['times'][0])
        evoked_score_diff.plot(show=False, axes=axes.flat[r], unit=False, zorder='std')
        axes.flat[r].axvline(0, linestyle='--', color='k', zorder=1000)
        axes.flat[r].axhline(0, linestyle='--', color='k', zorder=1000)
        axes.flat[r].set_ylabel('R²')
        axes.flat[r].set_title(str(rate) + ' Hz')
        fixed_golden_aspect_ratio(axes.flat[r])

        # dashed gray line at expected previous and next notes
        period = 1
        while axes.flat[r].get_xlim()[0] < - period / rate:
            axes.flat[r].axvline(- period / rate, linestyle='--', color='gray', zorder=0, linewidth=0.5)
            period += 1
        period = 1
        while axes.flat[r].get_xlim()[1] > period / rate:
            axes.flat[r].axvline(period / rate, linestyle='--', color='gray', zorder=0, linewidth=0.5)
            period += 1

    for ax in range(1, 6 - len(model_regression['rates']) + 1):
        axes.flat[-ax].remove()

    fig.suptitle('All features - Control features only')

    # Plot scores all features
    fig, axes = plt.subplots(2, 3, sharey=True, squeeze=False)
    lower_lim = None
    for r, rate in enumerate(model_regression['rates']):
        rate_model_r2 = model_r_squared[:, :, :, r]  # select rate and average subjects

        evoked_score_diff = mne.EvokedArray(np.mean(rate_model_r2, axis=0), info=model_regression['epochs_info'],
                                            tmin=model_regression['times'][0])
        evoked_score_diff.plot(show=False, axes=axes.flat[r], unit=False, zorder='std')
        axes.flat[r].axvline(0, linestyle='--', color='k', zorder=1000)
        axes.flat[r].axhline(0, linestyle='--', color='k', zorder=1000)
        axes.flat[r].set_ylabel('R²')
        axes.flat[r].set_title(str(rate) + ' Hz')
        fixed_golden_aspect_ratio(axes.flat[r])

        # dashed gray line at expected previous and next notes
        period = 1
        while axes.flat[r].get_xlim()[0] < - period / rate:
            axes.flat[r].axvline(- period / rate, linestyle='--', color='gray', zorder=0, linewidth=0.5)
            period += 1
        period = 1
        while axes.flat[r].get_xlim()[1] > period / rate:
            axes.flat[r].axvline(period / rate, linestyle='--', color='gray', zorder=0, linewidth=0.5)
            period += 1

    for ax in range(1, 6 - len(model_regression['rates']) + 1):
        axes.flat[-ax].remove()

    fig.suptitle('All features')
    plt.show()

    return


def regression_betas_by_rep_seeg(model_regression_path, features_to_plot):
    """
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times, rates)
    :param epochs: used for times and info
    :param feature_list: must be of length n_features
    :param tmin:
    :return:
    """
    plt.switch_backend('TkAgg')

    model_regression = np.load(model_regression_path, allow_pickle=True)

    for f, feature in enumerate(features_to_plot):
        betas = [[], []]
        fig, axes = plt.subplots(1, 1, sharey=True, squeeze=True)
        for subject in model_regression:
            subj_regression = model_regression[subject]
            subject_betas = subj_regression['subject_betas']
            feature_index = subj_regression['features'].index(feature)
            for r, rep in enumerate(subj_regression['rep']):

                feature_betas = subject_betas[feature_index, :, :, r]
                # feature_betas = subj_regression['subject_r2'][:, :, r]
                betas[r].append(np.mean(feature_betas, axis=0))

        betas = [np.array(betas[0]), np.array(betas[1])]
        avg_betas = [np.mean(betas[0], axis=0), np.mean(betas[1], axis=0)]
        sem_betas = [np.std(betas[0], axis=0) / np.sqrt(betas[0].shape[0]), np.std(betas[1], axis=0) / np.sqrt(betas[1].shape[0])]

        axes.plot(subj_regression['times'], avg_betas[0], label='rep1')
        axes.fill_between(subj_regression['times'], avg_betas[0]-sem_betas[0], avg_betas[0]+sem_betas[0], alpha=0.5, lw=0)
        axes.plot(subj_regression['times'], avg_betas[1], label='rep2')
        axes.fill_between(subj_regression['times'], avg_betas[1] - sem_betas[1], avg_betas[1] + sem_betas[1], alpha=0.5, lw=0)
        axes.axvline(0, linestyle='--', color='k', zorder=0)
        axes.axhline(0, linestyle='--', color='k', zorder=0)
        axes.legend()
        axes.set_ylabel('Regression parameter (a.u.)')
        axes.set_xlabel('Time (s)')

    plt.show()

    return


def regression_betas_by_frequency(model_regression, features_to_plot, averaging_method='mean', channel_subset=False):
    """
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times)
    :param epochs: used for times and info
    :param feature_list: must be of length n_features
    :param tmin:
    :param channel_subset: set it to True if the regression was not done on all channels. Topomaps cannot be plotted in that case.
    :return:
    """
    plt.switch_backend('TkAgg')

    betas = model_regression['betas']

    if averaging_method == 'mean':
        def averaging_fct(params): return np.mean(params, axis=0)
    elif averaging_method == 't-values':
        def averaging_fct(params): return np.mean(params, axis=0) / (np.std(params, axis=0) / np.sqrt(params.shape[0]))

    # model_regression['features'].insert(0, 'rate')  # TODO: remove
    feature_indices = [model_regression['features'].index(feat) for feat in features_to_plot]

    fig_all_features, axes_all_features = plt.subplots(3, ceil(len(features_to_plot) / 3) + 1, gridspec_kw={'width_ratios': [1 for i in range(ceil(len(features_to_plot) / 3))] + [0.1]}, squeeze=False)

    vmax = 0
    for plot_i, f in enumerate(feature_indices):
        feature_betas = betas[:, f, :, :, :]
        feature_betas = np.mean(feature_betas, axis=(0, 1))  # average subjects and channels
        vmax = max(vmax, np.max(np.abs(feature_betas)))
    for plot_i, f in enumerate(feature_indices):
        feature_betas = betas[:, f, :, :, :]
        plot_coord = [plot_i % 3, floor(plot_i / 3)]
        if not channel_subset:

            feature_betas = np.mean(feature_betas, axis=(0, 1))  # average subjects and channels
            im = axes_all_features[plot_coord[0], plot_coord[1]].pcolormesh(model_regression['times'], model_regression['freqs'], feature_betas.T, shading='nearest', cmap='bwr', vmin=-vmax, vmax=vmax)
            axes_all_features[plot_coord[0], plot_coord[1]].set_yscale('log')
            axes_all_features[plot_coord[0], plot_coord[1]].get_yaxis().set_major_formatter(ticker.ScalarFormatter())
            axes_all_features[plot_coord[0], plot_coord[1]].axvline(0, linestyle='--', color='k')
            axes_all_features[plot_coord[0], plot_coord[1]].set_ylabel('Frequency (Hz)')
            if plot_coord[0] == 2:
                axes_all_features[plot_coord[0], plot_coord[1]].set_xlabel("Time (s)")

            # fixed_golden_aspect_ratio(ax_row)

            axes_all_features[plot_coord[0], plot_coord[1]].set_title(features_to_plot[plot_i])
    fig_all_features.colorbar(im, cax=axes_all_features[0, -1])
    axes_all_features[0, -1].set_title(r'$\beta$')

    plt.show()

    return


def regression_scores_by_frequency(model_regression, control_regression):
    """
    :param r_squared_list: list of numpy scores (1 per feature of interest) of shape (n_subjects, n_channels, times)
    :param control_r_squared: numpy scores of control only of shape (n_subjects, n_channels, times)
    :param epochs: used for times and info
    :param features_list: names of the features
    :param tmin:
    :return:
    """
    plt.switch_backend('TkAgg')

    control_r_squared = control_regression['r2']
    model_r_squared = model_regression['r2']
    r2_diff = model_r_squared - control_r_squared
    r2_diff = np.mean(r2_diff, axis=(0, 1))  # average subjects and channels
    times = model_regression['times']

    r2_diff = np.log(np.maximum(np.zeros_like(r2_diff), r2_diff))
    # r2_diff = np.maximum(np.zeros_like(r2_diff), r2_diff)

    fig, axes = plt.subplots(1, 1)
    # vmax = np.max(np.abs(r2_diff))
    vmax = np.max(r2_diff)
    im = axes.pcolormesh(times, model_regression['freqs'], r2_diff.T, shading='nearest', cmap='Reds')
    axes.set_yscale('log')
    axes.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    axes.axvline(0, linestyle='--', color='k')
    axes.set_ylabel('Frequency (Hz)')
    axes.set_xlabel("Time (s)")
    axes.set_title('R²')
    fig.colorbar(im)

    plt.show()

    return

def regression_scores_all_subjects(model_regression, control_regression):
    """
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times)
    :param epochs: used for times and info
    :param feature_list: must be of length n_features
    :param tmin:
    :return:
    """
    plt.switch_backend('TkAgg')

    control_r_squared = control_regression['r2']
    model_r_squared = model_regression['r2']
    r2_diff = model_r_squared - control_r_squared

    n_subjects = r2_diff.shape[0]

    fig, axes = plt.subplots(ceil(n_subjects / 5), 5, sharex=True, sharey=True, squeeze=False)
    for i in tqdm(range(n_subjects)):
    # for i in tqdm(range(3)):
        evoked_betas = mne.EvokedArray(r2_diff[i, :, :], info=model_regression['epochs_info'], tmin=model_regression['times'][0])
        evoked_betas.plot(show=False, axes=axes.flat[i], unit=False, titles=dict(grad='', mag=''), window_title='', zorder='std')
        axes.flat[i].axvline(0, linestyle='--', color='k', zorder=0)
        axes.flat[i].set_xlabel(None)
        axes.flat[i].set_ylabel(None)
        axes.flat[i].set_title(None)
        for child in axes.flat[i].get_children():
            if isinstance(child, matplotlib.text.Annotation):
                if child.xy == (0, 1):
                    child.remove()

        # fixed_golden_aspect_ratio(axes.flat[i])

    for i in range(n_subjects, 30):
        axes.flat[i].remove()

    fig.suptitle('Delta r2')

    plt.show()

    return

def regression_betas_all_subjects(model_regression, features_to_plot, channel_subset=False):
    """
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times)
    :param epochs: used for times and info
    :param feature_list: must be of length n_features
    :param nb_channels: if None, keep all channels, if int, uses time to select nb_channels best channels
    :return:
    """
    plt.switch_backend('TkAgg')

    betas = model_regression['betas']

    n_subjects = betas.shape[0]
    feature_indices = [model_regression['features'].index(feat) for feat in features_to_plot]

    for plot_i, f in enumerate(feature_indices):
        print(features_to_plot[plot_i])
        feature_betas = betas[:, f, :, :]
        subjects_betas = []
        fig, axes = plt.subplots(ceil(n_subjects / 5), 5, sharex=True, sharey=True, squeeze=False)
        for i in tqdm(range(n_subjects)):
            if channel_subset:
                axes.flat[i].plot(model_regression['times'], np.sqrt(np.mean(feature_betas[i, :, :], axis=0)**2), linewidth=3)
                subjects_betas.append(np.sqrt(np.mean(feature_betas[i, :, :]**2, axis=0)))
            else:
                evoked_betas = mne.EvokedArray(feature_betas[i, :, :], info=model_regression['epochs_info'], tmin=model_regression['times'][0])
                evoked_betas.plot(show=False, axes=axes.flat[i], unit=False, titles=dict(grad='', mag=''), window_title='', zorder='std')
            axes.flat[i].axvline(0, linestyle='--', color='k', zorder=0)
            axes.flat[i].axhline(0, linestyle='--', color='k', zorder=0)
            axes.flat[i].set_xlabel(None)
            axes.flat[i].set_ylabel(None)
            axes.flat[i].set_title(None)
            for child in axes.flat[i].get_children():
                if isinstance(child, matplotlib.text.Annotation):
                    if child.xy == (0, 1):
                        child.remove()
        for i in range(n_subjects, 30):
            axes.flat[i].remove()
        fig.suptitle(features_to_plot[plot_i])

        # one t-values plot
        fig, axes = plt.subplots(1, 1, sharex=True, sharey=True, squeeze=False)
        subjects_betas = np.array(subjects_betas)
        axes.flat[0].plot(model_regression['times'], np.mean(subjects_betas, axis=0) / (np.std(subjects_betas, axis=0) / np.sqrt(subjects_betas.shape[0])), linewidth=3)
        axes.flat[0].axvline(0, linestyle='--', color='k', zorder=0)
        axes.flat[0].axhline(0, linestyle='--', color='k', zorder=0)
        axes.flat[0].set_xlabel('Times (s)')
        axes.flat[0].set_ylabel('t-values')
        axes.flat[0].set_title(features_to_plot[plot_i])
        fixed_golden_aspect_ratio(axes.flat[0])

    plt.show()

    return


def custom_spatio_temporal_cluster_regression_scores(regression_pkl, control_pkl, permutation_folder):
    """
    regression_pkl, control_pkl: paths to pickle files
    permutation_folder: path to folder containing permutation pickle files
    """

    plt.switch_backend('TkAgg')

    with open(regression_pkl, 'rb') as file:
        pkl_obj = pickle.load(file)  # used later for info and times
    r2 = pkl_obj['r2']
    with open(control_pkl, 'rb') as file:
        r2_control = pickle.load(file)['r2']
    permutation_folder = Path(permutation_folder)
    permutation_list = sorted(list(permutation_folder.glob('**/*perm*')))
    r2_permutations = np.empty((len(permutation_list), *r2.shape[1:]))  # already in t-values
    for p, perm in enumerate(permutation_list):
        with open(perm, 'rb') as file:
            r2_permutations[p, :, :] = pickle.load(file)['r2'][0, :, :]

    # we work on the difference between full and control models
    r2_diff = r2 - r2_control
    # reshape to (n_subjects, times, n_channels)
    r2_diff = np.swapaxes(r2_diff, 1, 2)
    # reshape to (n_permut, times, n_channels)
    r2_permutations = np.swapaxes(r2_permutations, 1, 2)

    ch_adjacency, ch_names = find_ch_adjacency(pkl_obj['epochs_info'], 'mag')
    pval = 0.05  # arbitrary
    df = r2.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    thresh = scipy.stats.t.ppf(1 - pval, df=df)  # T distribution, two-tailed
    # t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(feature_betas, threshold=dict(start=0, step=0.8), n_permutations=n_permut, tail=0,
    #                                                                      adjacency=ch_adjacency, n_jobs=-1)
    # t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(feature_betas,
    #                                                                      threshold=thresh,
    #                                                                      n_permutations=n_permut, tail=0,
    #                                                                      adjacency=ch_adjacency, n_jobs=-1)
    print(r2_diff.shape)
    print(r2_permutations.shape)
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_test_adapted(r2_diff, r2_permutations, threshold=thresh, tail=1, adjacency=ch_adjacency)

    # back to (n_subjects, n_channels, times)
    r2_diff = np.swapaxes(r2_diff, 1, 2)  # back to (n_subjects, n_channels, times)

    significant_clusters = np.array([cluster for (cluster, p_value) in zip(clusters, cluster_pv) if p_value < 0.05])
    significant_clusters = significant_clusters[np.argsort([cluster[0][0] for cluster in significant_clusters])]  # sort by time
    n_significant_clusters = len(significant_clusters)
    channels_indices = np.arange(r2_diff.shape[1])
    masked_times = np.full(r2_diff.shape[2], True)  # non-significant times will be masked

    height_ratios = [1 for _ in range(ceil(n_significant_clusters / 5))]
    height_ratios.append(3)
    fig, axes = plt.subplots(ceil(n_significant_clusters / 5) + 1, 5, gridspec_kw={'height_ratios': height_ratios}, squeeze=False)  # +1 row for the temporal plot

    for i, cluster in enumerate(significant_clusters):
        start_sample, end_sample = cluster[0][0], cluster[0][-1]
        masked_times[start_sample:end_sample] = False  # store significant times for later
        mne.viz.plot_topomap(np.mean(r2_diff[:, :, start_sample:end_sample], axis=(0, 2)),
                             pkl_obj['epochs_info'], axes=axes.flat[i], show=False,
                             mask=np.isin(channels_indices, cluster[1]))
        axes.flat[i].set_title('[' + str(pkl_obj['times'][start_sample]) + ', ' + str(pkl_obj['times'][end_sample]) + '] s')

    print(n_significant_clusters)
    if n_significant_clusters % 5 > 0:
        for remaining_axes in range(n_significant_clusters, n_significant_clusters + 5 - n_significant_clusters % 5):
            axes.flat[remaining_axes].remove()

    # merge axes of last row
    gs = axes[-1][0].get_gridspec()
    for ax in axes[-1]:
        ax.remove()
    ax_row = fig.add_subplot(gs[-1, :])
    evoked_betas = mne.EvokedArray(np.mean(r2_diff, axis=0), info=pkl_obj['epochs_info'], tmin=pkl_obj['times'][0])
    evoked_betas.plot(show=False, axes=ax_row, unit=False, zorder='std')
    ax_row.axvline(0, linestyle='--', color='k', zorder=0)

    mask = np.ma.masked_where(masked_times, ax_row.get_ylim()[0] * np.ones(r2_diff.shape[2]))  # we mask what is not significant
    ax_row.plot(pkl_obj['times'], mask, 'k', linewidth=2)

    fixed_golden_aspect_ratio(ax_row)

    # ax_row.set_title(features_to_plot[plot_i])
    ax_row.set_ylabel('r2 full model - r2 control')

    plt.show()

    return

def custom_spatio_temporal_cluster_regression_scores_seeg(model,regression_pkl, control_pkl, permutation_folder, cluster_thresh, out_folder):
    """
    regression_pkl, control_pkl: paths to pickle files
    permutation_folder: path to folder containing permutation pickle files
    """
    if not os.path.exists(out_folder + 'all_r2_'+model+'.pkl'):
        os.makedirs(out_folder, exist_ok=True)

        #load full-model results
        with open(regression_pkl, 'rb') as file:
            pkl_obj = pickle.load(file)  # used later for info and times
        r2 = []
        for sub in pkl_obj.keys():
            r2.append(pkl_obj[sub]['subject_r2'])
        r2 = np.concatenate(r2, axis = 0)

        #load control results
        with open(control_pkl, 'rb') as file:
            pkl_control = pickle.load(file)
        r2_control = []
        for sub in pkl_obj.keys():
            r2_control.append(pkl_control[sub]['subject_r2'])
        r2_control = np.concatenate(r2_control, axis = 0)

        #load permutations
        permutation_folder = Path(permutation_folder)
        permutation_list = sorted(list(permutation_folder.glob('**/*perm*')))
        r2_permutations = np.empty((len(permutation_list), *r2.shape))
        for p, perm in enumerate(permutation_list):
            r2_perm = []
            with open(perm, 'rb') as file:
                pklo_perm = pickle.load(file)
            for sub in pklo_perm.keys():
                r2_perm.append(pklo_perm[sub]['subject_r2'])
            r2_perm = np.concatenate(r2_perm, axis = 0)
            print('loaded permutation n°: ',p, '/', len(permutation_list))
            r2_permutations[p, :, :] = r2_perm

        #save data in a lighter form
        tmp = {'r2': r2, 'r2_control': r2_control, 'r2_permutations': r2_permutations}
        with open(out_folder+'all_r2_'+model+'.pkl', 'wb') as handle:
            pickle.dump(tmp, handle)

    else:
        with open(out_folder+'all_r2_'+model+'.pkl', 'rb') as file:
            pkl_obj = pickle.load(file)  # used later for info and times
            r2 = pkl_obj['r2']
            r2_control = pkl_obj['r2_control']
            r2_permutations = pkl_obj['r2_permutations']

    # # we work on the difference between full and control models
    r2_diff = r2 - r2_control
    for p in range(r2_permutations.shape[0]):
        r2_permutations[p,:,:] = r2_permutations[p,:,:] - r2_control
    # reshape to (n_subjects, times, n_channels)
    r2_diff = np.swapaxes(r2_diff, 0, 1)
    # reshape to (n_permut, times, n_channels)
    r2_permutations = np.swapaxes(r2_permutations, 1, 2)



    ch_adjacency = scipy.sparse.csr_matrix(np.zeros((r2_diff.shape[1], r2_diff.shape[1]))) # Empty adjacency matrix
    print(r2_diff.shape)
    print(r2_permutations.shape)
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_test_adapted_seeg(r2_diff, r2_permutations, threshold=cluster_thresh, tail=1, adjacency=ch_adjacency)

    # back to (n_channels, times)
    r2_diff = np.swapaxes(r2_diff, 0, 1)

    significant_clusters = np.array([cluster for (cluster, p_value) in zip(clusters, cluster_pv) if p_value < 0.05])
    significant_clusters = significant_clusters[np.argsort([cluster[0][0] for cluster in significant_clusters])]  # sort by time
    n_significant_clusters = len(significant_clusters)
    channels_indices = np.arange(r2_diff.shape[0])
    masked_times = np.full(r2_diff.shape[1], True)  # non-significant times will be masked

    #indices of channels with at least one significant cluster
    significant_channels = []
    for c in significant_clusters:
        significant_channels.append(c[1][0])
    significant_channels = np.unique(significant_channels)

    return significant_clusters, significant_channels


def plot_topo_regression_scores(regression_pkl, control_pkl, time_intervals_to_plot=None):
    """
    regression_pkl, control_pkl: paths to pickle files
    permutation_folder: path to folder containing permutation pickle files
    time_intervals_to_plot: list of 2-elements lists specifying start and end of time intervals to plot
    """

    plt.switch_backend('TkAgg')

    with open(regression_pkl, 'rb') as file:
        pkl_obj = pickle.load(file)  # used later for info and times
    r2 = pkl_obj['r2']
    with open(control_pkl, 'rb') as file:
        r2_control = pickle.load(file)['r2']

    if time_intervals_to_plot is None:
        time_intervals_to_plot = [[pkl_obj['times'][0], pkl_obj['times'][-1]]]

    # we work on the difference between full and control models
    r2_diff = r2 - r2_control

    fig, axes = plt.subplots(ceil(len(time_intervals_to_plot) / 4), 4, squeeze=False)

    for i, time_interval in enumerate(time_intervals_to_plot):
        start_sample, end_sample = np.where(pkl_obj['times'] == time_interval[0])[0][0], np.where(pkl_obj['times'] == time_interval[1])[0][0]
        mne.viz.plot_topomap(np.mean(r2_diff[:, :, start_sample:end_sample], axis=(0, 2)), pkl_obj['epochs_info'], axes=axes.flat[i], show=False)
        axes.flat[i].set_title('[' + str(pkl_obj['times'][start_sample]) + ', ' + str(pkl_obj['times'][end_sample]) + '] s')

    if len(time_intervals_to_plot) % 4 > 0:
        for remaining_axes in range(len(time_intervals_to_plot), len(time_intervals_to_plot) + 4 - len(time_intervals_to_plot) % 4):
            axes.flat[remaining_axes].remove()

    plt.suptitle('Topographies of r2 full-control')
    fig.tight_layout()

    plt.show()

    return


def plot_channel_subset_regression_scores(regression_pkl, control_pkl, best_channels_path=None):
    """
    regression_pkl, control_pkl: paths to pickle files
    permutation_folder: path to folder containing permutation pickle files
    time_intervals_to_plot: list of 2-elements lists specifying start and end of time intervals to plot
    """

    plt.switch_backend('TkAgg')

    if best_channels_path is not None:
        best_channels = np.load(best_channels_path)

    with open(regression_pkl, 'rb') as file:
        pkl_obj = pickle.load(file)  # used later for info and times
    r2 = pkl_obj['r2']
    with open(control_pkl, 'rb') as file:
        r2_control = pickle.load(file)['r2']

    # we work on the difference between full and control models
    r2_diff = r2 - r2_control

    if best_channels_path is not None:
        r2_diff = r2_diff[np.arange(r2_diff.shape[0])[:, np.newaxis], best_channels, :]
    r2_diff = np.sqrt(np.mean(r2_diff**2, axis=1))
    sem = np.std(r2_diff, axis=0) / np.sqrt(r2_diff.shape[0])

    fig, axes = plt.subplots(1, 1)
    axes.plot(pkl_obj['times'], np.mean(r2_diff, axis=0))
    axes.fill_between(pkl_obj['times'], np.mean(r2_diff, axis=0) - sem, np.mean(r2_diff, axis=0) + sem, alpha=0.6)

    # for i, time_interval in enumerate(time_intervals_to_plot):
    #     start_sample, end_sample = np.where(pkl_obj['times'] == time_interval[0])[0][0], np.where(pkl_obj['times'] == time_interval[1])[0][0]
    #     mne.viz.plot_topomap(np.mean(r2_diff[:, :, start_sample:end_sample], axis=(0, 2)), pkl_obj['epochs_info'], axes=axes.flat[i], show=False)
    #     axes.flat[i].set_title('[' + str(pkl_obj['times'][start_sample]) + ', ' + str(pkl_obj['times'][end_sample]) + '] s')
    #
    # if len(time_intervals_to_plot) % 4 > 0:
    #     for remaining_axes in range(len(time_intervals_to_plot), len(time_intervals_to_plot) + 4 - len(time_intervals_to_plot) % 4):
    #         axes.flat[remaining_axes].remove()

    plt.suptitle('Topographies of r2 full-control')
    fig.tight_layout()

    plt.show()

    return


def regression_betas_by_rate_average(model_regression, features_list):
    """
    Takes a by rate regression, and averages the betas of the different rates in order to emphasize what is time locked on the current note.
    :return:
    """
    plt.switch_backend('TkAgg')

    betas = model_regression['betas']
    feature_indices = [model_regression['features'].index(feat) for feat in features_list]

    fig, axes = plt.subplots(3, ceil(len(features_list) / 3), sharex=True, sharey=True, squeeze=False)
    for f, feature_index in enumerate(feature_indices):
        # fig, axes = plt.subplots(1, 1, sharey=True, squeeze=False)

        betas_average = np.mean(betas[:, feature_index, :, :, :], axis=(0, -1))  # average subjects and rates

        evoked_betas = mne.EvokedArray(betas_average, info=model_regression['epochs_info'], tmin=model_regression['times'][0])
        evoked_betas.plot(show=False, axes=axes.flat[f], unit=False, zorder='std')
        axes.flat[f].axvline(0, linestyle='--', color='k', zorder=0)

        fixed_golden_aspect_ratio(axes.flat[f])

        axes.flat[f].set_title(features_list[f] + ' rate betas averaged')
        axes.flat[f].set_ylabel('Regression parameter (a.u.)')

    plt.show()

    return


def regression_betas_by_rate_from_next_expected_note(model_regression, features_list, note_nb=1):
    """
    Takes a by rate regression, and averages the betas of the different rates in order to emphasize what is time locked on the next exppected note.
    :return:
    """
    plt.switch_backend('TkAgg')

    betas = model_regression['betas']
    feature_indices = [model_regression['features'].index(feat) for feat in features_list]

    fig, axes = plt.subplots(3, ceil(len(features_list) / 3), sharex=True, sharey=True, squeeze=False)
    for f, feature_index in enumerate(feature_indices):
        betas_list = []
        # fig, axes = plt.subplots(1, 1, sharey=True, squeeze=False)
        for r, rate in enumerate(model_regression['rates']):
            next_expected_note_index = model_regression['times'].tolist().index(round(note_nb/rate - 0.500, 2))  # we take 500ms before next note
            print(next_expected_note_index)
            rate_betas = betas[:, feature_index, :, next_expected_note_index:, r]
            if r > 0:  # set to 2.5Hz length
                rate_betas = rate_betas[:, :, :betas_list[0].shape[2]]
            betas_list.append(rate_betas)

        betas_next_note = np.mean(betas_list, axis=(0, 1))

        evoked_betas = mne.EvokedArray(betas_next_note, info=model_regression['epochs_info'], tmin=-0.500)
        evoked_betas.plot(show=False, axes=axes.flat[f], unit=False, zorder='std')
        axes.flat[f].axvline(0, linestyle='--', color='k', zorder=0)

        fixed_golden_aspect_ratio(axes.flat[f])

        axes.flat[f].set_title(features_list[f] + ' rate betas averaged from next expected note')
        axes.flat[f].set_ylabel('Regression parameter (a.u.)')

    plt.show()

    return


def plot_regression_betas_by_rate_all_features(betas, epochs, feature_list, rate_list, mode='mean_abs'):
    """
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times, rates)
    :param epochs: used for times and info
    :param feature_list: must be of length n_features
    :param tmin:
    :return:
    """
    plt.switch_backend('TkAgg')

    fig, axes = plt.subplots(2, 4, sharey=True, squeeze=False)
    colors = [matplotlib.cm.get_cmap('YlOrRd')(alpha) for alpha in [0.2, 0.4, 0.6, 0.8, 0.99]]
    order = np.arange(1, len(rate_list) + 1)
    for f, feature in enumerate(feature_list):
        for r, rate in enumerate(rate_list):
            if mode == 'mean_abs':
                feature_betas = betas[:, f, :, :, r]
                axes.flat[f].plot(epochs.times, np.mean(np.abs(np.mean(feature_betas, axis=0)), axis=0), color=colors[r], zorder=order[-(r+1)])  # average across subjects then take the mean of absolute value across channels for visualization
            elif mode == 'rms':
                feature_betas = betas[:, f, :, r]
                axes.flat[f].plot(epochs.times, np.mean(feature_betas, axis=0), color=colors[r])

        axes.flat[f].axvline(0, linestyle='--', color='k', zorder=0)

        # fixed_golden_aspect_ratio(axes.flat[f])

        axes.flat[f].set_title(feature)
        axes.flat[f].set_ylabel('Regression parameter (a.u.)')
        axes.flat[f].set_xlabel('Time (s)')

        if f==0:
            axes.flat[f].legend(rate_list)

    for ax in range(1, 8 - len(feature_list) + 1):
        axes.flat[-ax].remove()
        # fixed_golden_aspect_ratio(axes.flat[-ax])

    plt.show()

    return


def regression_noise_ceiling(epochs_path, save_path, tmin=1, n_folds=5, alphas=(1e2, 1e3, 1e4, 1e5, 1e6, 1e7), min_time_interval=None):
    """
    epochs_path: path to folder containing all epochs.fif files
    save_path: path to folder where results will be saved
    tmin: remove notes before a given time (to avoid onset response)
    n_folds: If 1, just fit a linear regression. If > 1, performs n-folds cv with ridge regression.
    
    Compute noise ceiling for MEG dataset.
    ! The regression has to be done exactly the same as in the main regression analysis.
    ! UPDATE THIS FUNCTION IF NECESSARY !
    """
    
    alphas = np.array(alphas)
    
    #create save dir
    Path(save_path).mkdir(parents=True, exist_ok=True)

    #get the correct epochs path depending on the desired dataset
    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))[:3]

    #prepare dict to save output
    output = {'betas': [],
              'r2': [],
              'n_folds': n_folds,
              'alphas': alphas,
              'min_time_interval': min_time_interval
              }
    
    #find shared epochs among all subjects
    metadatas = []
    for i, subject_epochs in enumerate(epochs_list):
        print("Retrieving metadata from ", subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs, preload=False,verbose=0)
        metadata = epochs.metadata

        metadata['pitch'] = metadata['pitch'].apply(lambda x: x[0])   #pitch info was in list format
        metadata['rate'] = np.nan  #add rate from stim name
        metadata['pitch_interval'] = np.nan #add pitch interval
        metadata['epoch_ID'] = None
        for i_note in range(metadata.shape[0]):
            metadata.loc[i_note, 'rate'] = float(metadata['stim_name'][i_note].split('_')[2][0:3])
            if metadata['trial_time'][i_note] > 0:
                metadata.loc[i_note, 'pitch_interval'] = np.abs(metadata.loc[i_note-1, 'pitch'] - metadata.loc[i_note, 'pitch'])
            #create epoch ID based on stim_name and trial_time, used to merge metadatas later
            metadata.loc[i_note, 'epoch_ID'] = int(metadata['stim_name'][i_note][5:8])*1e3 + np.round(metadata['trial_time'][i_note], 6)
        metadatas.append(metadata)
    m_common = metadatas[0].copy()
    idx_drop = []
    for i in range(len(metadatas)):
        m_ = metadatas[i]
        for j in range(m_common.shape[0]):
            if m_common['epoch_ID'][j] not in m_['epoch_ID'].tolist():
                idx_drop.append(j)
    idx_drop = np.unique(idx_drop)

    m_common = m_common.drop(index=idx_drop)

    #main loop for regression
    for i, subject_epochs in enumerate(epochs_list):
        print('Loading data for subject n°', str(i))
        #retrieve brain data for the subject of interest
        epochs = mne.read_epochs(subject_epochs, preload=True, verbose=0)
        output['times'] = epochs.times
        output['ch_names'] = epochs.ch_names
        metadata = metadatas[i]
        #compare metadatas and remove epochs that are not common to all subjects
        idx_drop = []
        for j in range(metadata.shape[0]):
            if metadata['epoch_ID'][j] not in m_common['epoch_ID'].tolist():
                idx_drop.append(j)
        metadata = metadata.drop(index=idx_drop)
        epochs.drop(idx_drop)
        epochs = epochs[metadata['trial_time'].to_numpy() >= tmin]  # remove onset response
        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[epochs.metadata['time_interval'] >= min_time_interval]
        y = epochs.get_data()
        
        #retrieve brain data for all other subjects
        x = np.full(y.shape, 0.0)
        for j, others_epochs in enumerate(epochs_list):
            if j != i:
                epochs = mne.read_epochs(others_epochs, preload=True, verbose=0)
                metadata = metadatas[j]
                #compare metadatas and remove epochs that are not common to all subjects
                idx_drop = []
                for h in range(metadata.shape[0]):
                    if metadata['epoch_ID'][h] not in m_common['epoch_ID'].tolist():
                        idx_drop.append(h)
                metadata = metadata.drop(index=idx_drop)
                epochs.drop(idx_drop)
                epochs = epochs[metadata['trial_time'].to_numpy() >= tmin]  # remove onset response
                # keep only notes with min time interval
                if min_time_interval is not None:
                    epochs = epochs[epochs.metadata['time_interval'] >= min_time_interval]
                x += epochs.get_data()
                del epochs

        # standardization
        x = (x - np.mean(x, axis=0, keepdims=True)) / np.std(x, axis=0, keepdims=True)
        y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)  # removes the relative importance of subjects, channels, and time points
        
        # independent regression for every time point and channel
        print('Running regression for subject n°', str(i))
        subject_r2 = np.zeros((y.shape[1], y.shape[2]))
        if n_folds > 1:
            cv = KFold(n_splits=n_folds)
            cv.get_n_splits(x)
        # for c in range(y.shape[1]):
          # print('Processing channel ', c)
          # for t in range(y.shape[2]):
          #     if n_folds == 1:
          #       regr = LinearRegression().fit(x[:,:,t], y[:, c, t])
          #       subject_r2[c, t] = r2_score(y[:,c,t], regr.predict(x[:,:,t]),multioutput='raw_values')
          #     elif n_folds > 1:
          #       cv_scores = np.full((alphas.shape[0], n_folds), np.nan)
          #       for i, alpha in enumerate(alphas):
          #         for j, (train, test) in enumerate(cv.split(x)):
          #           regr = Ridge(alpha=alpha)
          #           regr.fit(x[train,:,t],y[train, c, t])
          #           pred = regr.predict(x[test,:,t])
          #           cv_scores[i, j] = r2_score(y[test,c,t], pred, multioutput='raw_values')
          #       #store cv scores and best alphas
          #       cv_scores = np.mean(cv_scores, axis=1) #mean cross-validated r² across folds
          #       cv_scores = np.max(cv_scores, axis=0) #for each timepoint, keep the highest cross-val r² across alphas
          #       subject_r2[c, t] = cv_scores

        def individual_channel_regression(c_ind):
            """function used for multiprocessing"""
            print('Processing channel ', c_ind)
            channel_r2 = np.zeros((y.shape[2]))
            for t in range(y.shape[2]):
                if n_folds == 1:
                    regr = LinearRegression().fit(x[:, :, t], y[:, c_ind, t])
                    channel_r2[t] = r2_score(y[:, c_ind, t], regr.predict(x[:, :, t]), multioutput='raw_values')
                elif n_folds > 1:
                    cv_scores = np.full((alphas.shape[0], n_folds), np.nan)
                    for i, alpha in enumerate(alphas):
                        for j, (train, test) in enumerate(cv.split(x)):
                            regr = Ridge(alpha=alpha)
                            regr.fit(x[train, :, t], y[train, c_ind, t])
                            pred = regr.predict(x[test, :, t])
                            cv_scores[i, j] = r2_score(y[test, c_ind, t], pred, multioutput='raw_values')
                    # store cv scores and best alphas
                    cv_scores = np.mean(cv_scores, axis=1)  # mean cross-validated r² across folds
                    cv_scores = np.max(cv_scores, axis=0)  # for each timepoint, keep the highest cross-val r² across alphas
                    channel_r2[t] = cv_scores
            return channel_r2

        # multiprocessing channels
        channels_regressions = Parallel(n_jobs=-1)(delayed(individual_channel_regression)(c) for c in range(y.shape[1]))
        for c in range(y.shape[1]):
            subject_r2[c, :] = channels_regressions[c]
        #store regression results
        # output['betas'].append(subject_betas)
        output['r2'].append(subject_r2)
    # output['betas'] = np.array(output['betas'])
    output['r2'] = np.array(output['r2'])

    with open(save_path + 'NC_meg.pkl', 'wb') as fp:
        pickle.dump(output, fp)

    return output


def impact_time_interval_on_regression(epochs_path, model_regression, feature_of_interest, time_windows_list, tmin=1, specific_channels=None):
    """Analyze the impact of the time interval from current to next note on the performance of the regression. This is done in order to understand why shifted entropy works but non-shifted entropy does not.

    There should be one time window per note_rate."""

    plt.switch_backend('TkAgg')

    min_time_interval = model_regression['min_time_interval']
    rate_list = model_regression['rates']
    betas = model_regression['betas']
    feature_of_interest_index = model_regression['features'].index(feature_of_interest)

    features = ['rate', 'trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']
    features = features + ['surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']

    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))

    delta_se_list = [[] for _ in range(len(rate_list))]
    delta_t_list = [[] for _ in range(len(rate_list))]
    sample_positions_list = [[] for _ in range(len(rate_list))]
    for i, subject_epochs in enumerate(epochs_list):
    # for i, subject_epochs in enumerate(epochs_list[1:5]):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        # pick channels
        if specific_channels is not None:
            epochs.pick(specific_channels)

        metadata = epochs.metadata

        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[metadata['time_interval'] >= min_time_interval]
            metadata = metadata[metadata['time_interval'] >= min_time_interval]

        # transform metadata to obtain all features
        metadata['pitch'] = metadata['pitch'].apply(lambda x: np.log2(x[0]))  # pitch info was in list format. Apply log transform
        metadata['rate'] = 0  # add rate from stim name
        metadata['pitch_interval'] = 0  # add pitch interval
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'rate'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[2][0:3])
            if metadata.loc[metadata.index[i_note], 'trial_time'] > 0:
                metadata.loc[metadata.index[i_note], 'pitch_interval'] = np.abs(metadata.loc[metadata.index[i_note-1], 'pitch'] - metadata.loc[metadata.index[i_note], 'pitch'])

        # add shifted features
        for feat in ['entropy_pitch_paulrnn', 'entropy_timing_paulrnn']:
            metadata['shifted_' + feat] = 0  # add shifted entropy
        for i_note in range(metadata.shape[0]):
            if (i_note + 1 < len(metadata)) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note+1], 'stim_name']):
                for feat in ['entropy_pitch_paulrnn', 'entropy_timing_paulrnn']:
                    metadata.loc[metadata.index[i_note], 'shifted_' + feat] = metadata.loc[metadata.index[i_note + 1], feat]

        # add time to next note (which is the shifted time interval)
        metadata['time_to_next_note'] = -1  # -1 is a flag
        for i_note in range(metadata.shape[0]):
            if (i_note + 1 < len(metadata)) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note+1], 'stim_name']):
                metadata.loc[metadata.index[i_note], 'time_to_next_note'] = metadata.loc[metadata.index[i_note + 1], 'time_interval']

        regressors = metadata[features]
        delta_t = metadata['time_to_next_note']

        # remove onset response
        delta_t = delta_t[epochs.metadata['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressors = regressors[regressors['trial_time'].to_numpy() >= tmin]

        # remove last sample from each piece
        epochs = epochs[delta_t.to_numpy() != -1]
        regressors = regressors[delta_t.to_numpy() != -1]
        delta_t = delta_t[delta_t.to_numpy() != -1]

        # by rate
        for r, rate in enumerate(rate_list):

            rate_epochs = epochs[regressors['rate'].to_numpy() == rate]
            rate_regressors = regressors[regressors['rate'].to_numpy() == rate].drop('rate', axis=1)  # we drop rate column
            rate_delta_t = delta_t[regressors['rate'].to_numpy() == rate]

            # load data
            y = rate_epochs.get_data()  # (trials, channels, time)
            x = rate_regressors.to_numpy()  # (trials, features)
            rate_delta_t = rate_delta_t.to_numpy()  # (trials, delta_t)
            rate_delta_t = rate_delta_t * rate  # ratio, equals 1 when delta_t has expected length

            # standardization of features and meg
            x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
            y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)  # removes the relative importance of subjects, channels, and time points

            del rate_epochs

            # keep the time_window only
            time_samples = [np.where(epochs.times == time_windows_list[r][0])[0][0], np.where(epochs.times == time_windows_list[r][1])[0][0]]
            y = y[:, :, time_samples[0]:time_samples[1]]
            y = y.reshape((y.shape[0], y.shape[1] * y.shape[2]))  # reshape for matrix product

            # compute squared errors
            rate_betas = betas[i, :, :, time_samples[0]:time_samples[1], r]
            if specific_channels is not None:
                rate_betas = rate_betas[:, mne.channel_indices_by_type(epochs.info, picks=specific_channels)['mag'], :]
            n_time_samples = rate_betas.shape[2]
            n_channels = rate_betas.shape[1]
            rate_betas = rate_betas.reshape((rate_betas.shape[0], rate_betas.shape[1] * rate_betas.shape[2]))  # reshape for matrix product
            rate_betas_no_entropy = rate_betas.copy()
            rate_betas_no_entropy[feature_of_interest_index, :] = 0
            predictions = np.matmul(x, rate_betas)
            predictions_no_entropy = np.matmul(x, rate_betas_no_entropy)

            delta_se = (predictions_no_entropy - y)**2 - (predictions - y)**2
            # delta_se = (predictions_no_entropy - y)**2
            # delta_se = np.mean(delta_se.reshape((delta_se.shape[0], n_channels, n_time_samples)), axis=1)  # reshape back and average channels
            delta_se = np.mean(delta_se.reshape((delta_se.shape[0], n_channels, n_time_samples)), axis=(1, 2))  # reshape back and average channels
            # sample_positions = np.tile(np.linspace(0, 1, n_time_samples), delta_se.shape[0])  # used for colormap
            # rate_delta_t = np.repeat(rate_delta_t, delta_se.shape[1])

            delta_se_list[r] = delta_se_list[r] + delta_se.flatten().tolist()
            delta_t_list[r] = delta_t_list[r] + rate_delta_t.tolist()
            # sample_positions_list[r] = sample_positions_list[r] + sample_positions.tolist()

    fig, axes = plt.subplots(2, 3, sharey=True, squeeze=False)
    for r, rate in enumerate(model_regression['rates']):
        # sc = axes.flat[r].scatter(delta_t_list[r], delta_se_list[r], s=20, c=sample_positions_list[r], cmap='coolwarm')
        sc = axes.flat[r].scatter(delta_t_list[r], delta_se_list[r], s=20)
        # cbar = plt.colorbar(sc)
        # cbar.ax.set_ylabel('time sample position between next 2 notes', rotation=270)
        axes.flat[r].set_title(str(rate) + " Hz")
        axes.flat[r].axhline(0, linestyle='--', color='k', zorder=10)
        axes.flat[r].axvline(1, linestyle='--', color='k', zorder=10)
        axes.flat[r].set_ylabel('SE no entropy - SE full')
        axes.flat[r].set_xlabel('Time to next note (time * rate)')
        fixed_golden_aspect_ratio(axes.flat[r])
    for ax in range(1, 6 - len(model_regression['rates']) + 1):
        axes.flat[-ax].remove()

    plt.figure()
    full_delta_t = []
    full_delta_se = []
    full_sample_positions = []
    for r, rate in enumerate(model_regression['rates']):
        full_delta_t += delta_t_list[r]
        full_delta_se += delta_se_list[r]
        # full_sample_positions += sample_positions_list[r]

    # Binarize
    bins = np.arange(min(full_delta_t), max(full_delta_t), 0.05)
    indices_in_bins = np.digitize(full_delta_t, bins)
    average_delta_se = [np.mean(np.array(full_delta_se)[indices_in_bins == bin]) for bin in range(len(bins))]

    # sc = plt.scatter(full_delta_t, full_delta_se, s=20, c=full_sample_positions, cmap='coolwarm')
    # sc = plt.scatter(full_delta_t, full_delta_se, s=20)
    sc = plt.scatter(bins, average_delta_se, s=20)
    plt.axhline(y=sum(full_delta_se)/len(full_delta_se), color='r')
    # plt.plot(bins, average_delta_se)
    # cbar = plt.colorbar(sc)
    # cbar.ax.set_ylabel('time sample position between next 2 notes', rotation=270)
    plt.axhline(0, linestyle='--', color='k', zorder=10)
    plt.axvline(1, linestyle='--', color='k', zorder=10)
    plt.ylabel('SE no entropy - SE full')
    plt.xlabel('Time to next note (time * rate)')

    plt.figure()
    # sc = plt.scatter(full_delta_t, full_delta_se, s=20, c=full_sample_positions, cmap='coolwarm')
    sc = plt.scatter(full_delta_t, full_delta_se, s=20)
    plt.axhline(y=sum(full_delta_se) / len(full_delta_se), color='r')
    # plt.plot(bins, average_delta_se)
    # cbar = plt.colorbar(sc)
    # cbar.ax.set_ylabel('time sample position between next 2 notes', rotation=270)
    plt.axhline(0, linestyle='--', color='k', zorder=10)
    plt.axvline(1, linestyle='--', color='k', zorder=10)
    plt.ylabel('SE no entropy - SE full')
    plt.xlabel('Time to next note (time * rate)')

    plt.show()

    return


def correlate_consecutive_notes(subject_epochs, feature_name='surprise_mean_paulrnn', tmin=1, min_time_interval=None):

    plt.switch_backend('TkAgg')

    epochs = mne.read_epochs(subject_epochs)
    metadata = epochs.metadata

    # keep only notes with min time interval
    if min_time_interval is not None:
        print('Number of notes:', len(epochs.metadata))
        epochs = epochs[metadata['time_interval'] >= min_time_interval]
        print('Number of notes after min_time_interval:', len(epochs.metadata))

    # remove onset response
    epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
    feature = epochs.metadata[feature_name].to_numpy()

    current_features = feature[:-1]
    next_features = feature[1:]
    # current_features = feature[:-2]
    # next_features = feature[2:]

    corr = np.corrcoef(np.concatenate([current_features[np.newaxis, :], next_features[np.newaxis, :]], axis=0))
    print(corr)

    lm = LinearRegression()
    lm.fit(current_features[:, np.newaxis], next_features)
    print(lm.coef_)
    print(lm.intercept_)

    plt.plot(current_features, 'b')
    plt.plot(next_features, 'r')
    plt.show()

    fig, axes = plt.subplots(1, 1)
    axes.scatter(current_features, next_features, c='k', s=2)
    x = np.linspace(current_features.min(), current_features.max(), 100)
    y = lm.coef_ * x + lm.intercept_
    axes.plot(x, x, 'k--', linewidth=0.5)
    axes.plot(x, y, c='r', linewidth=2)
    axes.set_xlabel('Current note')
    axes.set_ylabel('Next note')
    axes.set_title(feature_name)
    axes.text(x.max(), (x.max() + x.min()) / 2, f'rho = {round(corr[0, 1], 2)}', size=15)

    fixed_golden_aspect_ratio(axes)

    plt.show()

    return


def interaction_surprise_deviation_from_expected(epochs_path, save_folder, tmin=1, min_time_interval=None, plot_time=0.210, deviation_method='relative', best_channels_path=None):

    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))

    surprise = []
    deviation = []
    meg_rms = []

    if best_channels_path is not None:
        best_channels = np.load(best_channels_path)

    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        metadata = epochs.metadata
        # keep only notes with min time interval
        if min_time_interval is not None:
            print('Number of notes:', len(epochs.metadata))
            epochs = epochs[metadata['time_interval'] >= min_time_interval]
            metadata = epochs.metadata
            print('Number of notes after min_time_interval:', len(epochs.metadata))

        # transform metadata to obtain all features
        metadata['rate'] = 0  # add rate from stim name
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'rate'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[2][0:3])

        # add deviation column
        metadata['deviation'] = 0
        for i_note in range(metadata.shape[0]):
            if deviation_method == 'relative':
                metadata.loc[metadata.index[i_note], 'deviation'] = (metadata.loc[metadata.index[i_note], 'time_interval'] - (1 / metadata.loc[metadata.index[i_note], 'rate'])) / (1 / metadata.loc[metadata.index[i_note], 'rate'])  # relative
            elif deviation_method == 'absolute':
                metadata.loc[metadata.index[i_note], 'deviation'] = metadata.loc[metadata.index[i_note], 'time_interval'] - (1 / metadata.loc[metadata.index[i_note], 'rate'])  # absolute

        regressors = metadata[['surprise_mean_paulrnn', 'deviation', 'trial_time']]

        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressors = regressors[regressors['trial_time'].to_numpy() >= tmin]

        surprise = surprise + regressors['surprise_mean_paulrnn'].tolist()
        deviation = deviation + regressors['deviation'].tolist()

        # load data
        y = epochs.get_data()  # (trials, channels, time)
        y = y[:, :, epochs.times.tolist().index(plot_time)]  # select time
        if best_channels_path is not None:
            y = y[:, best_channels[i, :]]
        # standardization of meg data
        # y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)  # removes the relative importance of subjects, channels
        y = np.sqrt(np.mean(y**2, axis=1))  # RMS across channels
        meg_rms = meg_rms + y.tolist()

        del epochs

    with open(os.path.join(save_folder, 'deviation_analysis.pkl'), 'wb') as f:
        pickle.dump({'surprise': surprise, 'deviation': deviation, 'meg_rms': meg_rms}, f)

    return


def plot_interaction(pickle_file, num_bins=100):
    plt.switch_backend('TkAgg')

    with open(pickle_file, 'rb') as file:
        d = pickle.load(file)
    df = pd.DataFrame(d)
    df['meg_rms'] = np.log(df['meg_rms'])
    # df['surprise'] = np.log(df['surprise'])
    print(df)

    # y_bins = np.linspace(df['surprise'].min(), df['surprise'].max(), num_bins)
    # x_bins = np.linspace(df['deviation'].min(), df['deviation'].max(), num_bins)

    x_bins = pd.cut(df['deviation'], bins=num_bins)
    y_bins = pd.cut(df['surprise'], bins=num_bins)
    # Compute the center values of the X and Y intervals
    x_bin_centers = x_bins.apply(lambda x: x.mid).astype(float)
    y_bin_centers = y_bins.apply(lambda x: x.mid).astype(float)

    # Use pivot_table to create a 2D grid with the mean of 'Z' values in each bin
    # heatmap_data = df.pivot_table(values='meg_rms', index=pd.cut(df['surprise'], y_bins), columns=pd.cut(df['deviation'], x_bins), aggfunc='mean')
    heatmap_data = df.pivot_table(values='meg_rms', index=y_bin_centers, columns=x_bin_centers, aggfunc='mean')
    heatmap_data = heatmap_data[::-1]  # inverse y axis

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(heatmap_data, ax=ax, cmap='plasma', cbar_kws={'label': 'log(meg_rms)'})
    ax.set_xticks(np.arange(len(heatmap_data.columns)), [round(label, 2) for label in heatmap_data.columns])
    ax.set_yticks(np.arange(len(heatmap_data.index)), [round(label, 2) for label in heatmap_data.index])
    # ax.set_ylabel('log(surprise)')
    ax.set_ylabel('surprise')
    ax.set_xlabel('deviation')
    ax.set_title('surprise and deviation effect on MMN')

    plt.show()

    return


def interval_deviation_analysis(epochs_path, save_path, quantiles_list=None):

    if quantiles_list is None:
        quantiles_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for i in range(len(quantiles_list) - 1):
        output = regression_predictive_features(epochs_path,
                                       save_path,
                                       confounds='all',
                                       model='paulrnn',
                                       data_type='meg',
                                       signal_type='raw',
                                       montage='mono',
                                       specific_channels=None,
                                       add_shifted=True,
                                       tmin=1,
                                       n_folds=5,
                                       alphas=np.logspace(2, 7, 6),
                                       permutation=False,
                                       min_time_interval=0.050,
                                       best_channels_path=None,
                                       # best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_channels.npy'
                                       deviation_quantile_range=[quantiles_list[i], quantiles_list[i+1]]
                                       )

        with open(os.path.join(save_path, 'regr_meg_quantiles_' + str(quantiles_list[i]) + '-' + str(quantiles_list[i+1]) + '.pkl'), 'wb') as fp:
            pickle.dump(output, fp)

    return

def plot_early_late_analysis(model_regression_list, condition_names, features_to_plot, best_channels_path=None):

    plt.switch_backend('TkAgg')
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set2.colors)
    colors = [matplotlib.cm.get_cmap('bwr')(alpha) for alpha in np.linspace(0.1, 0.9, len(model_regression_list))]

    betas_list = [model_regression['betas'] for model_regression in model_regression_list]
    feature_indices = [model_regression_list[0]['features'].index(feat) for feat in features_to_plot]

    if best_channels_path is not None:
        best_channels = np.load(best_channels_path)

    fig_all_features, axes_all_features = plt.subplots(len(features_to_plot), 1, sharex=True, squeeze=False)
    plot_list = []
    for plot_i, f in enumerate(feature_indices):
        for c, condition_betas in enumerate(betas_list):
            feature_betas = condition_betas[:, f, :, :]
            if best_channels_path is not None:
                feature_betas = feature_betas[np.arange(feature_betas.shape[0])[:, np.newaxis], best_channels, :]
            channel_rms = np.sqrt(np.mean(feature_betas ** 2, axis=1))
            # plot_list.append(axes_all_features.flat[plot_i].plot(model_regression_list[0]['times'], butter_lowpass_filter(np.mean(channel_rms, axis=0), 10, 100), linewidth=2, c=colors[c])[0])
            plot_list.append(axes_all_features.flat[plot_i].plot(model_regression_list[0]['times'], np.mean(channel_rms, axis=0), linewidth=2, c=colors[c])[0])
            # sem = np.std(channel_rms, axis=0) / np.sqrt(channel_rms.shape[0])
            # axes_all_features.flat[plot_i].fill_between(model_regression_list[0]['times'], np.mean(channel_rms, axis=0) - sem, np.mean(channel_rms, axis=0) + sem, alpha=0.6)
        axes_all_features.flat[plot_i].legend(plot_list, [str(cond) for cond in condition_names])
        axes_all_features.flat[plot_i].axvline(0, linestyle='--', color='k', zorder=0)
        axes_all_features.flat[plot_i].set_title(features_to_plot[plot_i])
        axes_all_features.flat[plot_i].set_ylabel('Regression parameter RMS')
    axes_all_features.flat[plot_i].set_xlabel('Time (s)')

    # for plot_i in range(len(features_to_plot)):
    #     fixed_golden_aspect_ratio(axes_all_features.flat[plot_i])

    for remaining_axes in range(axes_all_features.size - len(features_to_plot)):
        axes_all_features.flat[-remaining_axes - 1].remove()

    plt.show()

    return


def sustained_activity_per_piece(epochs_path, save_folder, tmin=1, min_time_interval=0.05, features=None, best_channels_path=None, time_interval=(-0.6, -0.4)):

    if features is None:
        features = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']

    if best_channels_path is not None:
        best_channels = np.load(best_channels_path)

    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))

    df_list = []

    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        metadata = epochs.metadata

        # keep only notes with min time interval
        if min_time_interval is not None:
            epochs = epochs[metadata['time_interval'] >= min_time_interval]

        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]

        excerpts = epochs.metadata[('stim_name')].to_numpy()
        excerpt_names = np.unique(excerpts)
        indices = [np.where(excerpts == name)[0] for name in excerpt_names]

        subject_df = epochs.metadata.groupby('stim_name', as_index=False)[features].mean()

        # keep time_interval only
        epochs = epochs.crop(tmin=time_interval[0], tmax=time_interval[1])
        # compute rms per excerpt
        y = epochs.get_data()  # (trials, channels, time)
        if best_channels_path is not None:
            y = y[:, best_channels[i, :], :]
        y = np.sqrt(np.mean(y**2, axis=1))  # rms
        y = np.mean(y, axis=1)  # average time points
        rms_list = [np.mean(y[ind]) for ind in indices]  # average the notes of each exerpt together

        subject_df['rms'] = rms_list
        df_list.append(subject_df)

    with open(os.path.join(save_folder, 'features_rms.pkl'), 'wb') as file:
        pickle.dump(df_list, file)

    return


def plot_rms_vs_features(pickle_file):
    plt.switch_backend('TkAgg')
    with open(pickle_file, 'rb') as file:
        df_list = pickle.load(file)

    features = df_list[0].columns[1:-1]

    # for f in features:
    #     fig, axes = plt.subplots(1, 1)
    #     feat_list = []
    #     rms_list = []
    #     for subject_df in df_list:
    #         feat_list += subject_df[f].tolist()
    #         rms_list += (subject_df['rms'] - subject_df['rms'].mean()).tolist()
    #         # axes.scatter(subject_df[f], subject_df['rms'] - subject_df['rms'].mean())
    #         # regr = LinearRegression().fit(subject_df[f][:, np.newaxis], subject_df['rms'] - subject_df['rms'].mean())
    #         # x = np.linspace(subject_df[f].min(), subject_df[f].max(), 100)
    #         # axes.plot(x, x * regr.coef_ + regr.intercept_)
    #     axes.axhline(0, c='k', linestyle='--')
    #     feat_array = np.array(feat_list)
    #     rms_array = np.array(rms_list)
    #     axes.scatter(feat_array, rms_array)
    #     regr = LinearRegression().fit(feat_array[:, np.newaxis], rms_array)
    #     x = np.linspace(feat_array.min(), feat_array.max(), 100)
    #     axes.plot(x, x * regr.coef_ + regr.intercept_, 'r', linewidth=3)
    #     axes.set_title(f)
    #     axes.set_xlabel('Average feature of the piece')
    #     axes.set_ylabel('Average RMS of the piece')
    #
    #     fig, axes = plt.subplots(1, 1)
    #     axes.axhline(0, c='k', linestyle='--')
    #     bin_edges = np.linspace(min(feat_array), max(feat_array), num=100)
    #     bin_indices = np.digitize(feat_array, bin_edges)
    #     binned_x = []
    #     averaged_y = []
    #     for i in range(1, len(bin_edges)):
    #         mask = bin_indices == i
    #         if np.any(mask):
    #             binned_x.append(np.mean(feat_array[mask]))
    #             averaged_y.append(np.mean(rms_array[mask]))
    #     binned_feat = np.array(binned_x)
    #     averaged_rms = np.array(averaged_y)
    #     axes.scatter(binned_feat, averaged_rms)
    #     regr = LinearRegression().fit(binned_feat[:, np.newaxis], averaged_rms)
    #     x = np.linspace(binned_feat.min(), binned_feat.max(), 50)
    #     axes.plot(x, x * regr.coef_ + regr.intercept_, 'r', linewidth=3)
    #     axes.set_title(f)
    #     axes.set_xlabel('Average feature of the piece')
    #     axes.set_ylabel('Average RMS of the piece')
    #
    #     print(f'Correlation for {f}:', np.corrcoef(feat_array, rms_array)[0, 1])
    #
    #     ols = sm.OLS(rms_array, sm.add_constant(feat_array[:, np.newaxis]))
    #     results = ols.fit()
    #     print(results.summary())

    # remove subject mean
    for subject in range(len(df_list)):
        df_list[subject]['rms'] = df_list[subject]['rms'] - df_list[subject]['rms'].mean()

    df = pd.concat(df_list)
    df = df.groupby('stim_name', as_index=False).mean()
    print(df)
    for f in features:
        fig, axes = plt.subplots(1, 1)
        feat_array = df[f].to_numpy()
        rms_array = df['rms'].to_numpy()
        axes.axhline(0, c='k', linestyle='--')
        axes.scatter(feat_array, rms_array)
        regr = LinearRegression().fit(feat_array[:, np.newaxis], rms_array)
        x = np.linspace(feat_array.min(), feat_array.max(), 100)
        axes.plot(x, x * regr.coef_ + regr.intercept_, 'r', linewidth=3)
        axes.set_title(f)
        axes.set_xlabel('Average feature of the piece')
        axes.set_ylabel('Average RMS of the piece')

        print(f'Correlation for {f}:', np.corrcoef(feat_array, rms_array)[0, 1])
        print(scipy.stats.spearmanr(feat_array, rms_array, alternative='greater'))


        ols = sm.OLS(rms_array, sm.add_constant(feat_array[:, np.newaxis]))
        results = ols.fit()
        print(results.summary())

    plt.show()


def compute_note_density(epoch_path, normalize=True):
    tw = [-0.5, 1]

    # retrieve note density
    epochs = mne.read_epochs(epoch_path, preload=False)
    metadata = epochs.metadata
    metadata = metadata[metadata['trial_time'] < 7]
    metadata = metadata[metadata['trial_time'] > 2]
    TI = metadata[['time_interval']].to_numpy()

    # times = np.arange(-0.5, 1, 0.01)
    times = np.linspace(-0.5, 1, 151)
    lim = [np.abs(times - tw[0]).argmin(), np.abs(times - tw[1]).argmin() + 1]
    times = times[lim[0]:lim[1]]

    note_density = np.zeros(times.shape)
    for idx in range(len(TI)):
        # going backward
        r = 10 if idx > 10 else idx
        t = 0
        for i in range(r):
            t -= TI[idx - i]
            if t >= -0.5:
                t_idx = np.argmin(np.abs(times - t))
                note_density[t_idx] += 1

        # going forward
        r = 10 if len(TI) - idx > 10 else len(TI) - idx
        t = 0
        for i in range(1, r):
            t += TI[idx + i]
            if t <= 1:
                t_idx = np.argmin(np.abs(times - t))
                note_density[t_idx] += 1

    # remove notes at t0
    t_idx = np.argmin(np.abs(times))
    note_density[t_idx] = np.nan

    # normalize
    note_density = note_density / len(TI)

    # normalize for plotting
    if normalize:
        note_density = note_density / np.nanmax(note_density)

    return note_density


if __name__ == "__main__":

    '''plot_decoding_scores'''
    # plot_decoding_scores("/home/mathieupvc/These/data/meg_maestro/note_analysis/decoding/decoding_scores.npy", features=['pitch', 'interval', 'surprise', 'entropy'], time_range=[-0.2, 0.5])
    # plot_decoding_scores("/home/mathieupvc/These/data/meg_maestro/note_analysis/decoding_time_sample/decoding_scores.npy", features=['pitch', 'interval', 'surprise', 'entropy'], time_range=[-0.2, 0.5])
    # plot_decoding_scores_indiv_subject("/home/mathieupvc/These/data/meg_maestro/note_analysis/decoding_time_sample/maestro08_mxnpsq/decoding_scores.npy", features=['pitch', 'interval', 'surprise', 'entropy'], time_range=[-0.2, 0.5])

    '''features_histograms'''
    # features_histograms(epoch_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/maestro08_mxnpsq-epo.fif')

    '''regression_predictive_features'''
    # regression_predictive_features('/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                                '/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/test_25',
    #                                confounds='all',
    #                                # confounds=['trial_time'],
    #                                # model=None,
    #                                model='paulrnn',
    #                                # model='magenta_perfrnn',
    #                                # model='perfrnn',
    #                                data_type='meg',
    #                                signal_type='raw',
    #                                montage='mono',
    #                                specific_channels=None,
    #                                # specific_channels=['MEG1941', 'MEG1721', 'MEG1641', 'MEG1631', 'MEG1611', 'MEG2421', 'MEG2411', 'MEG2431', 'MEG2441', 'MEG2221'],
    #                                # add_shifted=False,
    #                                add_shifted=True,
    #                                tmin=2,
    #                                tmax=7,
    #                                n_folds=5,
    #                                alphas=np.logspace(2, 7, 6),
    #                                permutation=False,
    #                                min_time_interval=0.050,
    #                                best_channels_path=None,
    #                                # best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_channels.npy',
    #                                # baseline=True
    #                                baseline=False,
    #                                avg_surprise=False,
    #                                )

    # Regression on ICA
    # regression_predictive_features('/home/mathieupvc/These/data/meg_maestro/note_analysis/ica_epochs',
    #                                '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/ica_23_10_23',
    #                                confounds='all',
    #                                # confounds=['trial_time'],
    #                                # model=None,
    #                                model='paulrnn',
    #                                # model='magenta_perfrnn',
    #                                # model='perfrnn',
    #                                data_type='meg',
    #                                signal_type='raw',
    #                                montage='mono',
    #                                specific_channels=None,
    #                                # specific_channels=['MEG1941', 'MEG1721', 'MEG1641', 'MEG1631', 'MEG1611', 'MEG2421', 'MEG2411', 'MEG2431', 'MEG2441', 'MEG2221'],
    #                                # add_shifted=False,
    #                                add_shifted=True,
    #                                tmin=2,
    #                                tmax=7,
    #                                n_folds=5,
    #                                alphas=np.logspace(2, 7, 6),
    #                                permutation=False,
    #                                min_time_interval=0.050,
    #                                best_channels_path=None,
    #                                # best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_channels.npy',
    #                                baseline=False
    #                                )

    # sEEG
    # regression_predictive_features('/home/mathieupvc/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta',
    #                                '/home/mathieupvc/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta/regression/31_07',
    #                                confounds='all',
    #                                # confounds=['trial_time'],
    #                                # model=None,
    #                                model='paulrnn',
    #                                # model='magenta_perfrnn',
    #                                # model='perfrnn',
    #                                data_type='seeg',
    #                                signal_type='raw',
    #                                montage='mono',
    #                                specific_channels=None,
    #                                # add_shifted=False,
    #                                add_shifted=True,
    #                                tmin=1,
    #                                n_folds=5,
    #                                alphas=np.logspace(2, 7, 6),
    #                                permutation=False,
    #                                min_time_interval=0.050
    #                                )

    '''variance partitioning loop'''
    feat_to_remove = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'shifted_surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']
    # for feat in feat_to_remove:
    #     regression_predictive_features('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                                    '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23',
    #                                    confounds='all',
    #                                    # confounds=['trial_time'],
    #                                    # model=None,
    #                                    model='paulrnn',
    #                                    # model='magenta_perfrnn',
    #                                    # model='perfrnn',
    #                                    data_type='meg',
    #                                    signal_type='raw',
    #                                    montage='mono',
    #                                    specific_channels=None,
    #                                    # specific_channels=['MEG1941', 'MEG1721', 'MEG1641', 'MEG1631', 'MEG1611', 'MEG2421', 'MEG2411', 'MEG2431', 'MEG2441', 'MEG2221'],
    #                                    # add_shifted=False,
    #                                    add_shifted=True,
    #                                    remove=feat,
    #                                    tmin=2,
    #                                    tmax=7,
    #                                    n_folds=5,
    #                                    alphas=np.logspace(2, 7, 6),
    #                                    permutation=False,
    #                                    min_time_interval=0.050,
    #                                    best_channels_path=None,
    #                                    # best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_channels.npy',
    #                                    # baseline=True
    #                                    baseline=False
    #                                    )

    '''regression_predictive_features_by_rate'''
    # regression_predictive_features_by_rate('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                                        '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/onset_07_12_23',
    #                                        confounds='all',
    #                                        # confounds=['trial_time'],
    #                                        # model=None,
    #                                        model='paulrnn',
    #                                        # model='magenta_perfrnn',
    #                                        # model='perfrnn',
    #                                        data_type='meg',
    #                                        signal_type='raw',
    #                                        montage='mono',
    #                                        specific_channels=None,
    #                                        add_shifted=True,
    #                                        # add_shifted=False,
    #                                        tmin=2,
    #                                        tmax=7,
    #                                        n_folds=5,
    #                                        alphas=np.logspace(2, 7, 6),
    #                                        permutation=False,
    #                                        min_time_interval=0.050
    #                                        )

    # sEEG
    # regression_predictive_features_by_rate('/home/mathieupvc/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta',
    #                                        '/home/mathieupvc/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta/regression/by_rate_05_01_24',
    #                                        confounds='all',
    #                                        # confounds=['trial_time'],
    #                                        # model=None,
    #                                        model='paulrnn',
    #                                        # model='magenta_perfrnn',
    #                                        # model='perfrnn',
    #                                        data_type='seeg',
    #                                        signal_type='raw',
    #                                        montage='mono',
    #                                        specific_channels=None,
    #                                        add_shifted=True,
    #                                        # add_shifted=False,
    #                                        tmin=2,
    #                                        tmax=7,
    #                                        n_folds=5,
    #                                        alphas=np.logspace(2, 7, 6),
    #                                        permutation=False,
    #                                        min_time_interval=0.050
    #                                        )

    # regression_predictive_features_by_rep('/home/mathieu/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta',
    #                                        '/home/mathieu/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta/regression/by_rep_21_10_24',
    #                                        confounds='all',
    #                                        # confounds=['trial_time'],
    #                                        # model=None,
    #                                        model='paulrnn',
    #                                        # model='magenta_perfrnn',
    #                                        # model='perfrnn',
    #                                        data_type='seeg',
    #                                        signal_type='raw',
    #                                        montage='mono',
    #                                        specific_channels=None,
    #                                        add_shifted=True,
    #                                        # add_shifted=False,
    #                                        tmin=2,
    #                                        tmax=7,
    #                                        n_folds=5,
    #                                        alphas=np.logspace(2, 7, 6),
    #                                        permutation=False,
    #                                        min_time_interval=0.050
    #                                        )

    '''regression_predictive_features_by_onset'''
    # regression_predictive_features_by_onset_type('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                                              '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/onset_07_12_23',
    #                                                 confounds='all',
    #                                                 # confounds=['trial_time'],
    #                                                 # model=None,
    #                                                 model='paulrnn',
    #                                                 # model='magenta_perfrnn',
    #                                                 # model='perfrnn',
    #                                                 data_type='meg',
    #                                                 signal_type='raw',
    #                                                 montage='mono',
    #                                                 specific_channels=None,
    #                                                 add_shifted=True,
    #                                                 # add_shifted=False,
    #                                                 tmin=2,
    #                                                 tmax=7,
    #                                                 n_folds=5,
    #                                                 alphas=np.logspace(2, 7, 6),
    #                                                 permutation=False,
    #                                                 min_time_interval=0.050
    #                                              )

    '''regression_predictive_features_by_trial_time'''
    # regression_predictive_features_by_trial_time('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                                              '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/onset_07_12_23',
    #                                                 confounds='all',
    #                                                 # confounds=['trial_time'],
    #                                                 # model=None,
    #                                                 model='paulrnn',
    #                                                 # model='magenta_perfrnn',
    #                                                 # model='perfrnn',
    #                                                 data_type='meg',
    #                                                 signal_type='raw',
    #                                                 montage='mono',
    #                                                 specific_channels=None,
    #                                                 add_shifted=True,
    #                                                 # add_shifted=False,
    #                                                 tmin=2,
    #                                                 tmax=7,
    #                                                 n_folds=5,
    #                                                 alphas=np.logspace(2, 7, 6),
    #                                                 permutation=False,
    #                                                 min_time_interval=0.050
    #                                              )

    '''regression_predictive_features_by_average_surprise'''
    # regression_predictive_features_by_average_surprise('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                                                    '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/onset_07_12_23',
    #                                                    confounds='all',
    #                                                    # confounds=['trial_time'],
    #                                                    # model=None,
    #                                                    model='paulrnn',
    #                                                    # model='magenta_perfrnn',
    #                                                    # model='perfrnn',
    #                                                    data_type='meg',
    #                                                    # data_type='seeg',
    #                                                    signal_type='raw',
    #                                                    montage='mono',
    #                                                    specific_channels=None,
    #                                                    add_shifted=True,
    #                                                    # add_shifted=False,
    #                                                    tmin=2,
    #                                                    tmax=7,
    #                                                    n_folds=5,
    #                                                    alphas=np.logspace(2, 7, 6),
    #                                                    permutation=False,
    #                                                    min_time_interval=0.050,
    #                                                    )

    '''regression_predictive_features_by_frequency'''
    # regression_predictive_features_by_frequency('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_time_freq_10chan',
    #                                                    '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_time_freq_10chan/regression/2Hz',
    #                                                    confounds='all',
    #                                                    # confounds=['trial_time'],
    #                                                    # model=None,
    #                                                    model='paulrnn',
    #                                                    # model='magenta_perfrnn',
    #                                                    # model='perfrnn',
    #                                                    data_type='meg',
    #                                                    # data_type='seeg',
    #                                                    signal_type='raw',
    #                                                    montage='mono',
    #                                                    specific_channels=None,
    #                                                    add_shifted=True,
    #                                                    # add_shifted=False,
    #                                                    tmin=2,
    #                                                    tmax=7,
    #                                                    n_folds=5,
    #                                                    alphas=np.logspace(2, 7, 6),
    #                                                    permutation=False,
    #                                                    min_time_interval=0.050,
    #                                                    )

    '''regression_shuffled_predictive_features'''
    # regression_shuffled_predictive_features('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_5channels',
    #                                         '/home/mathieupvc/These/data/meg_maestro/note_analysis',
    #                                         n_shuffle=100)

    '''regression_noise_ceiling'''
    # regression_noise_ceiling('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                          '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/noise_ceiling_26_09_23',
    #                          tmin=1,
    #                          n_folds=5,
    #                          alphas=np.logspace(3, 7, 5),
    #                          min_time_interval=0.050)

    '''experiment_duration_analysis'''
    # experiment_duration_analysis()
    regression_folder = '/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/experiment_duration_analysis'
    # plot_duration_analysis(regression_folder)
    plot_duration_analysis_by_subject(regression_folder)

    '''plot_regression_betas'''
    # plot_regression_betas('/home/mathieupvc/These/data/meg_maestro/note_analysis/regression_betas_predictive_features.npy', tmin=-0.5, tmax=0.5, multiple_channels='mean', features=['surprise', 'entropy'])

    '''plot_regression_main_effects'''
    # plot_regression_main_effects('/home/mathieupvc/These/data/meg_maestro/note_analysis/regression_betas_predictive_features.npy', tmin=-0.5, tmax=0.5, individual_subjects=False, multiple_channels='mean', features=['surprise', 'entropy', 'rep', 'rate', 'pitch', 'interval', 'trial_time'])

    '''plot_topo_regression_betas'''
    # plot_topo_regression_betas('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs', time=[-0.4, -0.2], features=None, tmin=1)

    '''spatio_temporal_cluster_regression_betas'''
    # # # feature_beta = np.load('/home/mathieupvc/These/data/meg_maestro/note_analysis/regression_betas_predictive_features.npy')
    # feature_beta = np.load('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression_betas_predictive_features.npy')
    # epochs = mne.read_epochs('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/maestro00_hoyzez-epo.fif')
    # # features = ['Mean surprise', 'Positive surprise', 'Negative surprise', 'Pitch entropy mean', 'Entropy', 'Repetition', 'Onset type (bigger is smoother)', 'Note Rate', 'Pitch', 'Interval', 'Trial time']
    # # features = ['surprise_perfrnn', 'entropy_perfrnn', 'mean_surprise_paulrnn', 'surprise_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'shifted_entropy_perfrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn', 'loudness', 'brightness', 'periodicity', 'roughness', 'loudness_deriv', 'brightness_deriv', 'periodicity_deriv', 'roughness_deriv', 'rep', 'onset', 'rate', 'pitch', 'interval', 'time_interval', 'trial_time']
    # features = ['mean_surprise_paulrnn', 'shifted_entropy_pitch_paulrnn', 'loudness', 'brightness', 'periodicity', 'roughness', 'loudness_deriv', 'brightness_deriv', 'periodicity_deriv', 'roughness_deriv', 'rep', 'onset', 'trial_time']
    # spatio_temporal_cluster_regression_betas(feature_beta, epochs, features, tmin=-1, n_permut=1000)

    '''regression_betas_by_rep_seeg'''
    model_regression_path = "/home/mathieu/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta/regression/by_rep_21_10_24/regr_by_rep_seeg_raw_mono_paulrnn.pkl"
    features_to_plot = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
    # regression_betas_by_rep_seeg(model_regression_path, features_to_plot)

    with open('/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/surp_entr_interaction_31_01_24/regr_meg_paulrnn.pkl', 'rb') as file:
        model_regression = pickle.load(file)
    # features_to_plot = ['surprise_possum_paulrnn', 'surprise_negsum_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']
    # features_to_plot = ['surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']
    # features_to_plot = ['surprise_mean_paulrnn']
    # features_to_plot = ['surprise_mean_paulrnn', 'shifted_surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']
    # features_to_plot = ['surprise_mean_paulrnn', 'shifted_surprise_mean_paulrnn', 'avg_surprise', 'avg_surprise*surprise', 'rate', 'rate*surprise']
    features_to_plot = ['surprise_mean_paulrnn', 'surprise*entropy_pitch', 'surprise*entropy_timing', 'surprise*trial_time']
    # features_to_plot = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn']
    # features_to_plot = ['surprise_perfrnn', 'shifted_entropy_perfrnn']
    # features_to_plot = ['surprise_magenta_pitch_perfrnn', 'entropy_magenta_pitch_perfrnn', 'surprise_magenta_timing_perfrnn', 'entropy_magenta_timing_perfrnn']
    # spatio_temporal_cluster_regression_betas(model_regression, features_to_plot, n_permut=1000, averaging_method='mean', channel_subset=False)

    '''regression_betas_ica'''
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/ica_23_10_23/regr_meg_paulrnn.pkl', 'rb') as file:
    #     model_regression = pickle.load(file)
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/ica_23_10_23/regr_meg_acoustic.pkl', 'rb') as file:
    #     control_regression = pickle.load(file)
    #     features_to_plot = ['surprise_possum_paulrnn', 'surprise_negsum_paulrnn', 'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']
    #     # features_to_plot = ['surprise_mean_paulrnn', 'shifted_surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']
    # ica_path = '/home/mathieupvc/These/data/meg_maestro/note_analysis/ica_epochs'
    # regression_betas_ica(model_regression, control_regression, ica_path, features_to_plot, averaging_method='mean', nb_components=3)

    '''spatio_temporal_cluster_regression_betas_by_rate'''
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/fixed_shift_25_10_23/regr_by_rate_meg_paulrnn.pkl', 'rb') as file:
    #     model_regression = pickle.load(file)
    # features_to_plot = ['surprise_mean_paulrnn', 'shifted_surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']
    # features_to_plot = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn']
    # features_to_plot = ['entropy_pitch_paulrnn']
    # spatio_temporal_cluster_regression_betas_by_rate(model_regression, features_to_plot, n_permut=100)

    '''regression_betas_by_frequency'''
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_time_freq_10chan/regression/6Hz/regr_by_freq_meg_paulrnn.pkl', 'rb') as file:
    #     model_regression = pickle.load(file)
    # features_to_plot = ['surprise_mean_paulrnn', 'shifted_surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']
    # regression_betas_by_frequency(model_regression, features_to_plot, averaging_method='mean', channel_subset=False)

    '''regression_scores_by_frequency'''
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_time_freq_10chan/regression/all_rates/regr_by_freq_meg_paulrnn.pkl', 'rb') as file:
    #     model_regression = pickle.load(file)
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_time_freq_10chan/regression/all_rates/regr_by_freq_meg_acoustic.pkl', 'rb') as file:
    #     control_regression = pickle.load(file)
    # regression_scores_by_frequency(model_regression, control_regression)

    '''spatio_temporal_cluster_regression_scores'''
    with open('/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/first_second_08_12_23/regr_meg_paulrnn.pkl', 'rb') as file:
        model_regression = pickle.load(file)
    with open('/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/first_second_08_12_23/regr_meg_acoustic.pkl', 'rb') as file:
        control_regression = pickle.load(file)
    # spatio_temporal_cluster_regression_scores(model_regression, control_regression)

    '''spatio_temporal_cluster_regression_scores_by_rate'''
    with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/fresh_start_27_10_23/regr_by_rate_meg_paulrnn.pkl', 'rb') as file:
        model_regression = pickle.load(file)
    with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/fresh_start_27_10_23/regr_by_rate_meg_acoustic.pkl', 'rb') as file:
        control_regression = pickle.load(file)
    # spatio_temporal_cluster_regression_scores_by_rate(model_regression, control_regression)

    '''regression_betas_by_rate_average'''
    with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/fresh_start_27_10_23/regr_by_rate_meg_paulrnn.pkl', 'rb') as file:
        model_regression = pickle.load(file)
    # features_list = ['surprise_mean_paulrnn', 'shifted_surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']
    features_list = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn']
    # regression_betas_by_rate_average(model_regression, features_list)

    '''regression_betas_by_rate_from_next_expected_note'''
    with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/anti_shifted_06_10_23/regr_by_rate_meg_paulrnn.pkl', 'rb') as file:
        model_regression = pickle.load(file)
    features_list = ['surprise_mean_paulrnn', 'shifted_surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']
    # regression_betas_by_rate_from_next_expected_note(model_regression, features_list, note_nb=0)

    '''custom_spatio_temporal_cluster_regression_scores'''
    regression_pkl = '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_08_08_23/regr_meg_paulrnn.pkl'
    control_pkl = '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_08_08_23/regr_meg_acoustic.pkl'
    permutation_folder = '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_08_08_23'
    # custom_spatio_temporal_cluster_regression_scores(regression_pkl, control_pkl, permutation_folder)

    '''plot_topo_regression_scores'''
    regression_pkl = '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/shifted_controls_20_09_23/regr_meg_paulrnn.pkl'
    control_pkl = '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/shifted_controls_20_09_23/regr_meg_acoustic.pkl'
    # plot_topo_regression_scores(regression_pkl, control_pkl, time_intervals_to_plot=[[-0.6, -0.48], [-0.45, -0.3], [0.07, 0.13], [0.20, 0.25], [0.3, 0.5]])

    '''plot_topo_regression_scores'''
    regression_pkl = '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/shifted_controls_20_09_23/regr_meg_paulrnn.pkl'
    control_pkl = '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/shifted_controls_20_09_23/regr_meg_acoustic.pkl'
    # plot_channel_subset_regression_scores(regression_pkl, control_pkl, best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_channels_210ms.npy')

    '''correlation_matrix'''
    # epochs = mne.read_epochs('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/maestro00_hoyzez-epo.fif')
    # # features = from_metadata_to_regressors(epochs.metadata, multiple_pitches='max')
    # features = transform_metadata(epochs.metadata, add_shifted=True, multiple_pitches='max')
    # correlation_matrix(features)

    '''erp_all_subjects_from_epochs'''
    # erp_all_subjects_from_epochs('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta', tmin=1, min_time_interval=0.050)

    '''erp_all_subjects_from_epochs_by_rate_gfp'''
    # erp_all_subjects_from_epochs_by_rate_gfp('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                                          rate_list=[2.5, 3.5, 4.5, 5.5, 6.5], tmin=1, min_time_interval=0.050)

    '''erp_all_subjects_from_epochs_by_rate'''
    # erp_all_subjects_from_epochs_by_rate('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                                      rate_list=[2.5, 3.5, 4.5, 5.5, 6.5], tmin=2, tmax=7, min_time_interval=0.050)

    '''erp_all_subjects_from_epochs_by_quantile'''
    # erp_all_subjects_from_epochs_by_quantile('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                                          feature='surprise_mean_paulrnn', shift=False,
    #                                          quantiles=True, tmin=1, min_time_interval=0.050)

    '''save_best_channels'''
    # save_best_channels('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                    '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp',
    #                    tmin=2,
    #                    tmax=7,
    #                    min_time_interval=0.050,
    #                    n_channels=5,
    #                    chosen_time=0.100)

    '''regression_predictive_features_by_rate'''
    # regression_predictive_features_by_rate('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_paul_1sec',
    #                                        '/home/mathieupvc/These/data/meg_maestro/note_analysis',
    #                                        rate_list=[2.5, 3.5, 4.5, 5.5, 6.5],
    #                                        # features=['surprise', 'entropy', 'rep', 'onset', 'pitch', 'interval', 'trial_time'],
    #                                        features=['mean_surprise', 'positive_surprise', 'negative_surprise',
    #                                                  'pitch_entropy_mean', 'entropy', 'rep', 'onset', 'pitch',
    #                                                  'interval', 'trial_time'],
    #                                        # specific_channels=['MEG1941', 'MEG1721', 'MEG1641', 'MEG1631', 'MEG1611'],
    #                                        # specific_channels=['MEG2421', 'MEG2411', 'MEG2431', 'MEG2441', 'MEG2221'],
    #                                        specific_channels=None,
    #                                        rms=False,
    #                                        tmin=1, shifted=True
    #                                        )

    '''spatio_temporal_cluster_regression_betas_by_rate'''
    # feature_beta = np.load('/home/mathieupvc/These/data/meg_maestro/note_analysis/regression_betas_predictive_features_by_rate.npy')
    # epochs = mne.read_epochs('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_paul_1sec/maestro00_hoyzez-epo.fif')
    # # features = ['Surprise', 'Entropy', 'Repetition', 'Onset type (bigger is smoother)', 'Pitch', 'Interval', 'Trial time']
    # features = ['Mean surprise', 'Positive surprise', 'Negative surprise', 'Pitch entropy mean', 'Entropy',
    #             'Repetition', 'Onset type (bigger is smoother)', 'Pitch', 'Interval', 'Trial time']
    # spatio_temporal_cluster_regression_betas_by_rate(feature_beta, epochs, features,
    #                                                  rate_list=[2.5, 3.5, 4.5, 5.5, 6.5], tmin=-1, n_permut=1000)

    '''plot_regression_betas_by_rate_all_features'''
    # feature_beta = np.load('/home/mathieupvc/These/data/meg_maestro/note_analysis/regression_betas_predictive_features_by_rate.npy')
    # epochs = mne.read_epochs('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_paul_1sec/maestro00_hoyzez-epo.fif')
    # features = ['Surprise', 'Entropy', 'Repetition', 'Onset type (bigger is smoother)', 'Pitch', 'Interval', 'Trial time']
    # plot_regression_betas_by_rate_all_features(feature_beta, epochs, features, rate_list=[2.5, 3.5, 4.5, 5.5, 6.5], mode='mean_abs')

    '''impact_time_interval_on_regression'''
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/wide_bin_18_07/regr_by_rate_meg_paulrnn.pkl', 'rb') as file:
    #     model_regression = pickle.load(file)
    # impact_time_interval_on_regression(epochs_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                                    model_regression=model_regression,
    #                                    # feature_of_interest='shifted_entropy_pitch_paulrnn',
    #                                    feature_of_interest='surprise_mean_paulrnn',
    #                                    # time_windows_list=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
    #                                    time_windows_list=[[0.54, 0.59], [0.43, 0.52], [0.30, 0.36], [0.27, 0.33], [0.22, 0.27]],
    #                                    # time_windows_list=[[0.4, 0.8], [0.29, 0.57], [0.22, 0.44], [0.18, 0.36], [0.15, 0.31]],
    #                                    tmin=1,
    #                                    # specific_channels=['MEG1941', 'MEG1721', 'MEG1641', 'MEG1631', 'MEG1611', 'MEG2421', 'MEG2411', 'MEG2431', 'MEG2441', 'MEG2221'])
    #                                    specific_channels=['MEG2411'])
    #                                    # specific_channels=None)


    '''regression_betas_all_subjects'''
    with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/5channels_05_10_23/regr_meg_paulrnn.pkl', 'rb') as file:
        model_regression = pickle.load(file)
    features_to_plot = ['surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
    # features_to_plot = ['surprise_mean_paulrnn']
    # regression_betas_all_subjects(model_regression, features_to_plot, channel_subset=True)

    '''regression_scores_all_subjects'''
    with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/shifted_controls_20_09_23/regr_meg_paulrnn.pkl', 'rb') as file:
        model_regression = pickle.load(file)
    with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/shifted_controls_20_09_23/regr_meg_acoustic.pkl', 'rb') as file:
        control_regression = pickle.load(file)
    # regression_scores_all_subjects(model_regression, control_regression)

    '''correlate_consecutive_notes'''
    # correlate_consecutive_notes('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/maestro01_itdheb-epo.fif',
    #                             feature_name='surprise_mean_paulrnn',
    #                             tmin=1,
    #                             min_time_interval=0.050)

    '''interation_surprise_deviation_from_expected'''
    # interaction_surprise_deviation_from_expected('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                                             '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/shifted_controls_20_09_23',
    #                                             tmin=1,
    #                                             min_time_interval=0.050,
    #                                             plot_time=0.210,
    #                                             deviation_method='relative',  # 'absolute'
    #                                             best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_channels_210ms.npy')

    '''plot_interation'''
    # plot_interaction('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/shifted_controls_20_09_23/deviation_analysis_relative_best100ms.pkl', num_bins=100)

    '''interval_deviation_analysis'''
    # interval_deviation_analysis('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                             '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/early_late_09_10_23',
    #                             quantiles_list=None)

    '''plot_early_late_analysis'''
    with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/early_late_09_10_23/regr_meg_paulrnn_early.pkl', 'rb') as file:
        early_regression = pickle.load(file)
    with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/early_late_09_10_23/regr_meg_paulrnn_ontime.pkl', 'rb') as file:
        ontime_regression = pickle.load(file)
    with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/early_late_09_10_23/regr_meg_paulrnn_late.pkl', 'rb') as file:
        late_regression = pickle.load(file)
    # features_list = ['surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
    # features_list = ['surprise_mean_paulrnn']
    features_list = ['entropy_timing_paulrnn']
    # best_channels_path = '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_10channels_100ms.npy'
    best_channels_path = None
    # condition_names = ['Early', 'On time', 'Late']
    quantiles = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    condition_names = [str(quantiles[i]) + '-' + str(quantiles[i+1]) for i in range(len(quantiles) - 1)]
    regression_list = []
    for i in range(len(quantiles) - 1):
        with open(f'/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/early_late_09_10_23/absolute_deviation/regr_meg_quantiles_{quantiles[i]}-{quantiles[i+1]}.pkl', 'rb') as file:
            regression_list.append(pickle.load(file))
    # plot_early_late_analysis(regression_list, condition_names, features_list, best_channels_path=best_channels_path)

    '''sustained_activity_per_piece'''
    features = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
    # sustained_activity_per_piece('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta', '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/excerpts_analysis', tmin=1, min_time_interval=0.05, features=features, best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_10channels_100ms.npy', time_interval=(0.160, 0.260))

    '''plot_rms_vs_features'''
    # plot_rms_vs_features('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/excerpts_analysis/features_rms.pkl')

    '''concatenate_subjects'''
    # concatenate_subjects('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                      '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/concatenated_epochs/all_subjects-epo.fif')

    '''save_ica_epochs'''
    # save_ica_epochs('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                  '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/ica_epochs')

    '''save_erp_by_rate'''
    # save_erp_by_rate('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta', '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp', rate_list=[2.5, 3.5, 4.5, 5.5, 6.5], tmin=2, tmax=7, min_time_interval=0.050)

    '''erp_by_rate_value_and_lag'''
    # erp_by_rate_value_and_lag('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/rate_erp_data.npy', rate_list=[3.5, 4.5, 5.5, 6.5], channels='/home/mathieupvc/These/figures_papier/significant_channels_200ms.npy')

    '''save_erp_by_quantile'''
    # save_erp_by_quantile('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                      'surprise_mean_paulrnn',
    #                      False,
    #                      '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp',
    #                      tmin=2,
    #                      tmax=7,
    #                      min_time_interval=0.05)

    '''save_note_epochs_tfr'''
    # save_note_epochs_tfr('/home/mathieupvc/These/data/meg_maestro/time_frequency_analysis/time_frequency_freqcycles_10chan', '/home/mathieupvc/These/data/meg_maestro/input_features/wav', '/home/mathieupvc/These/data/meg_maestro/input_features/midi', '/home/mathieupvc/These/data/meg_maestro/input_features/acoustic', '/home/mathieupvc/These/data/meg_maestro/input_features/models/perfRNN', '/home/mathieupvc/These/data/meg_maestro/input_features/models/magenta_perfRNN', '/home/mathieupvc/These/data/meg_maestro/input_features/models/paulRNN', '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_time_freq_freqcycles_10chan')
