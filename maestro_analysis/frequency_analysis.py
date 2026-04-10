import os
from pathlib import Path
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

import mne
from mne.datasets import sample
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_1samp_test

from maestro_analysis.utils import epoching_maestro, select_seeg_channels_from_parcellation

from pactools import Comodulogram

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def plot_meg_grand_average(data_path):
    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*-epo.fif')))

    del subject_paths[5]  # no behavior data

    evoked_list = []
    for i, subject_path in enumerate(subject_paths):
        epochs = mne.read_epochs(subject_path)
        evoked_list.append(epochs.average())
    del epochs

    grand_evoked = mne.grand_average(evoked_list)

    # plot
    grand_evoked.plot()
    grand_evoked.plot_image()
    grand_evoked.plot_joint()
    grand_evoked.plot_sensors(show_names=True)
    grand_evoked.plot_topo()
    grand_evoked.plot_topomap()
    grand_evoked.plot_psd()
    grand_evoked.plot_psd(average=True)
    grand_evoked.plot_psd_topo()
    grand_evoked.plot_psd_topomap()
    bands = {'Delta (0-4 Hz)': (0, 4), 'Theta (4-8 Hz)': (4, 8),
             'Alpha (8-12 Hz)': (8, 12), 'Beta (17-26 Hz)': (17, 26),
             'Gamma (30-45 Hz)': (30, 45), 'High-Gamma (60-90 Hz)': (60, 90)}
    grand_evoked.plot_psd_topomap(bands=bands)

    return


def plot_seeg_average(epochs_path, channels):

    epochs = mne.read_epochs(epochs_path)
    epochs = epochs.pick(np.concatenate(channels))
    epochs = epochs.apply_baseline((None, -0.1))
    epochs.filter(0.5, None, n_jobs=-1)
    evoked = epochs.average()
    del epochs

    # plot
    evoked.plot()
    evoked.plot_image()
    evoked.plot_joint()
    evoked.plot_sensors(show_names=True)
    evoked.plot_topo()
    evoked.plot_topomap()
    for chan_list in channels:
        evoked.plot_psd(picks=chan_list)
        evoked.plot_psd(average=True, picks=chan_list)
        evoked.plot_psd_topo(picks=chan_list)
        plt.show()

    return


def plot_condition_psd(data_path, conditions_path, condition_names):
    """conditions_path should contain correct and incorrect or old/new responses"""
    conditions = np.load(conditions_path, allow_pickle=True)
    cond1 = {}
    cond2 = {}
    for subj in conditions:
        cond1[subj] = conditions[subj][0]
        cond2[subj] = conditions[subj][1]

    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*-epo.fif')))

    del subject_paths[5]  # no behavior data

    evoked_list = [[] for i in range(2)]
    for i, subject_path in enumerate(subject_paths):
        subject_name = subject_path.name[:-8]
        epochs = mne.read_epochs(subject_path)
        stim_name = epochs.metadata['stim_name'].to_list()

        for c, cond in enumerate([cond1, cond2]):
            cond_epochs = epochs.copy()
            int_stim_name = [int(name[5:8]) for name in stim_name]
            subj_condition = cond[subject_name]
            trials_to_keep = [int_stim_name.index(trial) for trial in int_stim_name if trial in subj_condition]  # keep only trials that were kept in MEG
            cond_epochs = cond_epochs[trials_to_keep]
            cond_epochs = cond_epochs[~cond_epochs.metadata['stim_name'].str.contains('rep2')]  # remove rep 2
            evoked_list[c].append(cond_epochs.average())
    del epochs

    grand_evoked1 = mne.grand_average(evoked_list[0])
    grand_evoked2 = mne.grand_average(evoked_list[1])
    spectrum1 = grand_evoked1.compute_psd()
    spectrum2 = grand_evoked2.compute_psd()
    difference_spectrum = spectrum1.copy()
    difference_spectrum._data = spectrum1._data - spectrum2._data
    print(spectrum1._data.shape)
    print(spectrum2._data.shape)
    print(difference_spectrum._data.shape)
    print(spectrum1._data.min())
    print(spectrum1._data.max())
    print(spectrum2._data.min())
    print(spectrum2._data.max())
    print(difference_spectrum._data.min())
    print(difference_spectrum._data.max())

    # plot
    # grand_evoked.plot()
    # grand_evoked.plot_image()
    # grand_evoked.plot_joint()
    # grand_evoked.plot_sensors(show_names=True)
    # grand_evoked.plot_topo()
    # grand_evoked.plot_topomap()
    # grand_evoked.plot_psd()
    # grand_evoked.plot_psd(average=True)
    # grand_evoked.plot_psd_topo()
    # grand_evoked.plot_psd_topomap()
    # bands = {'Delta (0-4 Hz)': (0, 4), 'Theta (4-8 Hz)': (4, 8),
    #          'Alpha (8-12 Hz)': (8, 12), 'Beta (17-26 Hz)': (17, 26),
    #          'Gamma (30-45 Hz)': (30, 45), 'High-Gamma (60-90 Hz)': (60, 90)}
    # grand_evoked.plot_psd_topomap(bands=bands)

    fig, ax = plt.subplots(1, 1)
    spectrum1.plot(average=True, color='blue', axes=ax)
    spectrum2.plot(average=True, color='red', axes=ax)
    # difference_spectrum.plot(average=True, color='green', axes=ax)
    ax.legend(condition_names)
    plt.show()

    return


def trial_level_spectrum(data_path, behavior_path, channels=None, fs_subjects_dir=None):

    behavior_all = pd.read_csv(behavior_path)

    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    condition_spectrum = [[] for i in range(6)]
    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject_name]

        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
        incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
        correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
        incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()

        epochs = mne.read_epochs(epochs_path)

        # processing
        if channels is not None:
            assert fs_subjects_dir is not None
            epochs, _ = select_seeg_channels_from_parcellation(epochs, channels, fs_subjects_dir)

        if epochs is not None:
            epochs = epochs.apply_baseline((None, -0.1))
            fs = 200
            epochs.resample(fs, n_jobs=-1)
            epochs.filter(0.5, None, n_jobs=-1)
            epochs.crop(tmin=0, tmax=8)  # remove onset and offset

            # select epochs
            correct_novelty = epochs[epochs.metadata['stim_name'].isin(correct_novelty)]
            incorrect_novelty = epochs[epochs.metadata['stim_name'].isin(incorrect_novelty)]
            correct_retrieval = epochs[epochs.metadata['stim_name'].isin(correct_retrieval)]
            incorrect_retrieval = epochs[epochs.metadata['stim_name'].isin(incorrect_retrieval)]
            correct_encoding = epochs[epochs.metadata['stim_name'].isin(correct_encoding)]
            incorrect_encoding = epochs[epochs.metadata['stim_name'].isin(incorrect_encoding)]
            for m, memory_step in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [correct_encoding, incorrect_encoding]]):
                if (len(memory_step[0]) > 0) & (len(memory_step[1]) > 0):
                    spectrum = memory_step[0].average().compute_psd()
                    condition_spectrum[2*m].append(np.mean(spectrum._data, axis=0))
                    spectrum = memory_step[1].average().compute_psd()
                    condition_spectrum[2*m+1].append(np.mean(spectrum._data, axis=0))
    freqs = spectrum.freqs  # keep freqs for plotting
    condition_spectrum = [np.array(spectr) for spectr in condition_spectrum]

    for m, memory_step in enumerate(['Novelty detection', 'Retrieval', 'Encoding']):
        fig, axes = plt.subplots(2, 1, squeeze=False)

        axes[0, 0].plot(freqs, np.mean(condition_spectrum[2*m], axis=0), label='Correct')
        sem = np.std(condition_spectrum[2*m], axis=0) / np.sqrt(condition_spectrum[2*m].shape[0])
        axes[0, 0].fill_between(freqs, np.mean(condition_spectrum[2*m], axis=0) - sem, np.mean(condition_spectrum[2*m], axis=0) + sem, alpha=0.6, lw=0)
        axes[0, 0].plot(freqs, np.mean(condition_spectrum[2*m+1], axis=0), label='Incorrect')
        sem = np.std(condition_spectrum[2*m+1], axis=0) / np.sqrt(condition_spectrum[2*m+1].shape[0])
        axes[0, 0].fill_between(freqs, np.mean(condition_spectrum[2*m+1], axis=0) - sem, np.mean(condition_spectrum[2*m+1], axis=0) + sem, alpha=0.6, lw=0)
        # axes[0, 0].set_xlabel('Frequency (Hz)')
        axes[0, 0].set_ylabel('Power')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title(memory_step)
        axes[0, 0].legend()

        axes[1, 0].plot(freqs, np.mean(condition_spectrum[2*m] - condition_spectrum[2*m+1], axis=0) / (np.std(condition_spectrum[2*m] - condition_spectrum[2*m+1], axis=0) / np.sqrt(condition_spectrum[2*m].shape[0])))
        axes[1, 0].axhline(0, linestyle='--', linewidth=2, color='k', zorder=0)
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel(f'Correct - Incorrect, t-value (n = {condition_spectrum[2*m].shape[0]})')
        # axes[1, 0].set_yscale('log')
        # axes[1, 0].set_title('Rep 1: old - new')

        fig.suptitle(str(channels))

    plt.show()

    print('Done')

    return


def trial_level_spectrum_vs_baseline(data_path, regions, fs_subjects_dir=None):

    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)
    # mne.datasets.fetch_fsaverage(subjects_dir=None)
    subject = 'fsaverage'

    fig, axes = plt.subplots(1, 1, squeeze=False)
    for region in regions:
        spectrum_diff = []
        for i, epochs_path in tqdm(enumerate(subject_list)):
            epochs = mne.read_epochs(epochs_path)
            # processing
            assert fs_subjects_dir is not None
            epochs, _ = select_seeg_channels_from_parcellation(epochs, region, subjects_dir)

            if epochs is not None:
                epochs = epochs.apply_baseline((None, -0.1))
                fs = 400
                epochs.resample(fs, n_jobs=-1)
                epochs.filter(0.5, None, n_jobs=-1)
                baseline_spectrum = epochs.compute_psd(method='welch', tmin=None, tmax=0, fmin=0, fmax=200)
                spectrum = epochs.compute_psd(method='welch', tmin=0, tmax=8, fmin=0, fmax=200)
                # spectrum_diff.append(np.mean((spectrum._data - baseline_spectrum._data) / baseline_spectrum._data * 100, axis=(0, 1)))  # subject level, percent change
                for c in range(spectrum._data.shape[1]):  # electrode level
                    spectrum_diff.append(np.mean((spectrum._data[:, c, :] - baseline_spectrum._data[:, c, :]) / baseline_spectrum._data[:, c, :] * 100, axis=0))  # percent change
        freqs = spectrum.freqs  # keep freqs for plotting
        spectrum_diff = np.array(spectrum_diff)

        # axes.flat[0].plot(freqs, np.mean(spectrum_diff, axis=0) / (np.std(spectrum_diff, axis=0) / spectrum_diff.shape[0]))
        axes.flat[0].plot(freqs, np.mean(spectrum_diff, axis=0), label=str(region))
        sem = np.std(spectrum_diff, axis=0) / spectrum_diff.shape[0]
        axes.flat[0].fill_between(freqs, np.mean(spectrum_diff, axis=0) - sem, np.mean(spectrum_diff, axis=0) + sem, alpha=0.5)
        axes.flat[0].set_xlabel('Frequency (Hz)')
        axes.flat[0].set_ylabel('Power (% change relative to baseline)')
        axes.flat[0].legend()

    plt.show()

    return


if __name__ == '__main__':
    """plot_meg_grand_average"""
    # plot_meg_grand_average('/home/mathieupvc/These/data/meg_maestro/meg_trials')

    """plot_condition_psd"""
    # conditions_path = '/home/mathieupvc/These/data/meg_maestro/correct_incorrect_all_reps_lists.pkl'
    # condition_names = ['correct', 'incorrect']
    conditions_path = '/home/mathieupvc/These/data/meg_maestro/new_old_lists.pkl'  # response
    condition_names = ['New', 'Old']
    # plot_condition_psd('/home/mathieupvc/These/data/meg_maestro/meg_trials', conditions_path=conditions_path, condition_names=condition_names)

    """plot_seeg_average"""
    channels = [[f"H'{i}" for i in range(1, 15)],
                [f"B'{i}" for i in range(1, 13)],
                [f"SA'{i}" for i in range(1, 12)]]
    # plot_seeg_average('/home/mathieupvc/These/data/meg_maestro/seeg_trials/Pen_Va_epochs_mono_preprocessed_trials_IDA_nofilter.fif', channels=channels)

    '''trial_level_spectrum'''
    data_path = '/home/mathieupvc/These/data/meg_maestro/seeg_trials'
    behavior_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal']
    # channels = ['Left-Hippocampus', 'Right-Hippocampus']
    # channels = ['Left-Hippocampus']
    # channels = ['Right-Hippocampus']
    # channels = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal', 'Left-Hippocampus', 'Right-Hippocampus']
    fs_subjects_dir = '/home/mathieupvc/freesurfer/subjects'
    # trial_level_spectrum(data_path, behavior_path, channels=channels, fs_subjects_dir=fs_subjects_dir)

    '''trial_level_spectrum_vs_baseline'''
    data_path = '/home/mathieu/These/data/meg_maestro/seeg_trials'
    regions = ['Left-Hippocampus']
    # regions = [['Left-Hippocampus', 'Right-Hippocampus'], ['Left-Amygdala', 'Right-Amygdala'], ['ctx-lh-parahippocampal', 'ctx-rh-parahippocampal'], ['ctx-lh-temporalpole', 'ctx-rh-temporalpole'], ['ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal'], ['ctx-lh-parsopercularis', 'ctx-rh-parsopercularis'], ['ctx-lh-inferiorparietal', 'ctx-rh-inferiorparietal'], ['ctx-lh-precentral', 'ctx-rh-precentral'], ['Left-Thalamus-Proper', 'Right-Thalamus-Proper']]
    fs_subjects_dir = '/home/mathieupvc/freesurfer/subjects'
    trial_level_spectrum_vs_baseline(data_path, regions=regions, fs_subjects_dir=fs_subjects_dir)
