import os
from pathlib import Path
import pickle
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from math import ceil, factorial

from tqdm import tqdm

import mne.time_frequency
from mne.time_frequency import tfr_morlet

from maestro_analysis.utils import epoching_maestro, fixed_golden_aspect_ratio, select_seeg_channels_from_parcellation
from maestro_analysis.stats import test_channels_significance
from maestro_analysis.trf_analysis import condition_list_from_metadata

def time_freq_epochs(raw_path, wav_folder, save_path=None, channel_type='meg', best_channels=None, n_jobs=24):
    '''
    saves time frequency representation for one subject
    best_channels: list/array of channel indices
    '''

    epochs = epoching_maestro(raw_path, wav_folder, tmin=-1.5, baseline=False)  # Load raw and create epochs
    epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]  # Select listening conditions only
    epochs.pick(channel_type)  # keep magnetometers and/or gradiometers
    if best_channels is not None:
        epochs.pick(best_channels)
    epochs.filter(0.5, None, n_jobs=n_jobs)  # Detrend
    epochs.resample(200, n_jobs=n_jobs)

    # time_frequency decomposition
    # freqs = np.logspace(*np.log10([0.5, 95]), num=40)
    freqs = np.logspace(*np.log10([1, 100]), num=100)
    # freqs = np.linspace(1, 100, num=100)
    n_cycles = freqs  # different number of cycle per frequency
    # n_cycles = 6
    # n_cycles = np.minimum(freqs * 2, 15 * np.ones_like(freqs)) # 15 is the number of cycles for 7.5Hz. For higher frequencies we do not need to restrict the bandwidth
    power_epochs = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, decim=20,
                              n_jobs=n_jobs, zero_mean=True, average=False, output='complex')
    # power_epochs = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, decim=2,
    #                           n_jobs=n_jobs, zero_mean=True, average=False, output='complex')

    del epochs

    if save_path is not None:
        power_epochs.save(fname=save_path, overwrite=True)
        del power_epochs
        gc.collect()
    else:
        return power_epochs


def time_freq_epochs_all_subjects(data_path, wav_folder, save_folder, channel_type='meg', best_channels_path=None, n_jobs=24):
    """"""

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    if best_channels_path is not None:
        best_channels = np.load(best_channels_path)

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]

        os.makedirs(os.path.join(save_folder, subject_path.name), exist_ok=True)
        if best_channels_path is not None:
            time_freq_epochs(raw_path, wav_folder, save_path=os.path.join(save_folder, subject_path.name, str(subject_path.name) + '-tfr.h5'), channel_type=channel_type, best_channels=best_channels[i, :], n_jobs=n_jobs)
        else:
            time_freq_epochs(raw_path, wav_folder,
                             save_path=os.path.join(save_folder, subject_path.name, str(subject_path.name) + '-tfr.h5'),
                             channel_type=channel_type, best_channels=None, n_jobs=n_jobs)

    return


def average_subject_tfr(tf_folder, save_folder):

    path = Path(tf_folder)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    avg_subject_power = []

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        tfr_path = list(subject_path.glob('**/*tfr.h5'))[0]
        power_epochs = mne.time_frequency.read_tfrs(tfr_path)[0]
        power_epochs.data = np.abs(power_epochs.data)**2  # get energy from complex tfr
        power_epochs = power_epochs.apply_baseline(baseline=(-0.75, -0.25), mode='logratio')  # TODO: logratio?
        power_avg = power_epochs.average()

        if i==0:
            info, times, freqs = power_avg.info, power_avg.times, power_avg.freqs  # keep necessary objects once

        avg_subject_power.append(power_avg.data)

    avg_subject_power = np.mean(np.array(avg_subject_power), axis=0)
    avg_subject_power = mne.time_frequency.AverageTFR(info, avg_subject_power, times, freqs, nave=0)

    avg_subject_power.save(os.path.join(save_folder, "average_subject-tfr.h5"), overwrite=True)

    return


def individual_subject_spectrum(tf_folder, save_folder, save_format='png'):

    path = Path(tf_folder)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(20, 12), dpi=300, sharex=True, sharey=True)

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        tfr_path = list(subject_path.glob('**/*tfr.h5'))[0]
        power_epochs = mne.time_frequency.read_tfrs(tfr_path)[0]
        power_epochs.data = np.abs(power_epochs.data)**2  # get energy from complex tfr
        power_epochs = power_epochs.apply_baseline(baseline=(-0.75, -0.25), mode='logratio')
        power_avg = power_epochs.average()

        if i==0:
            info, times, freqs = power_avg.info, power_avg.times, power_avg.freqs  # keep necessary objects once

        power_avg = power_avg.crop(tmin=0, tmax=8)
        power_avg = np.mean(power_avg.data, axis=(0, 2))
        axes.flat[i].plot(freqs, power_avg)
        axes.flat[i].set_xscale('log')

    fig.savefig(os.path.join(save_folder, "individual_spectrum." + save_format))
    plt.close('all')

    return


def average_subject_tfr_conditions(tf_folder, save_folder, conditions=['rep1', 'rep2', 'rep3']):
    path = Path(tf_folder)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    avg_subject_power = [[] for i in range(len(conditions))]

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        tfr_path = list(subject_path.glob('**/*tfr.h5'))[0]
        power_epochs = mne.time_frequency.read_tfrs(tfr_path)[0]
        power_epochs.data = np.abs(power_epochs.data) ** 2  # get energy from complex tfr
        power_epochs = power_epochs.apply_baseline(baseline=(-0.75, -0.25), mode='logratio')

        if i == 0:
            info, times, freqs= power_epochs.info, power_epochs.times, power_epochs.freqs  # keep necessary objects once

        for c, condition in enumerate(conditions):
            condition_list = condition_list_from_metadata(power_epochs.metadata, [condition])
            power_avg = power_epochs[condition_list].average()
            avg_subject_power[c].append(power_avg.data)

    avg_subject_power = [np.mean(np.array(power_list), axis=0) for power_list in avg_subject_power]
    avg_subject_power = [mne.time_frequency.AverageTFR(info, avg_list, times, freqs, nave=0) for avg_list in avg_subject_power]
    for c in range(len(conditions)):
        avg_subject_power[c].save(os.path.join(save_folder, "average_subject_" + conditions[c] + "-tfr.h5"), overwrite=True)

    return


def average_subject_tfr_response_condition(tf_folder, save_folder, response_path, conditions=None):
    '''
    Condition linked to the behavioral response. Uses response_path to load the responses for the chosen condition
    e.g, correct and incorrect responses or new and old responses.
    '''

    if conditions is None:
        conditions = ['correct', 'incorrect']

    path = Path(tf_folder)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])
    del subject_folders[5]  # subject 5 does not have behavioral data

    with open(response_path, 'rb') as f:  # '/home/mathieu/Data/meg_maestro/correct_incorrect_lists.pkl'
        response_dict = pickle.load(f)

    avg_subject_power = [[] for i in range(len(conditions))]

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        tfr_path = list(subject_path.glob('**/*tfr.h5'))[0]
        power_epochs = mne.time_frequency.read_tfrs(tfr_path)[0]
        power_epochs.data = np.abs(power_epochs.data) ** 2  # get energy from complex tfr
        power_epochs = power_epochs.apply_baseline(baseline=(-0.75, -0.25), mode='logratio')

        if i == 0:
            info, times, freqs= power_epochs.info, power_epochs.times, power_epochs.freqs  # keep necessary objects once

        # convert correct and incorrect lists to strings
        response1 = [str(response_dict[subject_path.name][0][i]).zfill(3) for i in range(len(response_dict[subject_path.name][0]))]
        response2 = [str(response_dict[subject_path.name][1][i]).zfill(3) for i in range(len(response_dict[subject_path.name][1]))]
        response1_response2 = [response1, response2]
        print('response1_response2')
        print(response1_response2)

        for c, condition in enumerate(conditions):
            condition_list = condition_list_from_metadata(power_epochs.metadata, response1_response2[c])
            power_avg = power_epochs[condition_list].average()
            avg_subject_power[c].append(power_avg.data)

    avg_subject_power = [np.mean(np.array(power_list), axis=0) for power_list in avg_subject_power]
    avg_subject_power = [mne.time_frequency.AverageTFR(info, avg_list, times, freqs, nave=0) for avg_list in avg_subject_power]
    for c in range(len(conditions)):
        avg_subject_power[c].save(os.path.join(save_folder, "average_subject_" + conditions[c] + "-tfr.h5"), overwrite=True)

    return


def tfr_condition_difference(condition1_path, condition2_path, save_folder, condition_name):

    tfr_1 = mne.time_frequency.read_tfrs(condition1_path)[0]
    tfr_2 = mne.time_frequency.read_tfrs(condition2_path)[0]

    difference = tfr_1.data - tfr_2.data
    difference = mne.time_frequency.AverageTFR(tfr_1.info, difference, tfr_1.times, tfr_1.freqs, nave=0)
    difference.save(os.path.join(save_folder, "condition_difference_" + condition_name + "-tfr.h5"), overwrite=True)

    return


def spectrum_conditions(tf_folder, save_folder, condition_name='rep', conditions=['rep1', 'rep2', 'rep3'], time_range=None, save_format='png'):

    if time_range is None:
        time_range = [1, 8]  # remove the first second to avoid the onset response

    path = Path(tf_folder)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    avg_subject_power = [[] for i in range(len(conditions))]

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        tfr_path = list(subject_path.glob('**/*tfr.h5'))[0]
        power_epochs = mne.time_frequency.read_tfrs(tfr_path)[0]
        power_epochs.data = np.abs(power_epochs.data) ** 2  # get energy from complex tfr
        power_epochs = power_epochs.apply_baseline(baseline=(-0.75, -0.25), mode='logratio')

        if i == 0:
            freqs = power_epochs.freqs  # keep necessary objects once

        for c, condition in enumerate(conditions):
            condition_list = condition_list_from_metadata(power_epochs.metadata, [condition])
            power_avg = power_epochs[condition_list].average()
            power_avg = power_avg.crop(tmin=time_range[0], tmax=time_range[1])
            power_avg = np.mean(power_avg.data, axis=(0, 2))
            avg_subject_power[c].append(power_avg)

    average_list = [np.mean(np.array(power_list), axis=0) for power_list in avg_subject_power]
    sem_list = [np.std(np.array(power_list), axis=0) / np.sqrt(len(power_list)) for power_list in avg_subject_power]

    lines = []
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    for c in range(len(conditions)):
        lines.append(ax.plot(freqs, average_list[c])[0])
        ax.fill_between(freqs, average_list[c] - sem_list[c], average_list[c] + sem_list[c], alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power relative to baseline')
    ax.legend(lines, conditions)

    fig.savefig(os.path.join(save_folder, 'spectrum_' + condition_name + '.' + save_format))
    plt.close('all')

    return


def spectrum_response_condition(tf_folder, save_folder, response_path, condition_name, conditions=None, time_range=None, save_format='png'):

    if time_range is None:
        time_range = [1, 8]  # remove the first second to avoid the onset response

    if conditions is None:
        conditions = ['correct', 'incorrect']

    path = Path(tf_folder)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])
    del subject_folders[5]  # subject 5 does not have behavioral data

    with open(response_path, 'rb') as f:  # '/home/mathieu/Data/meg_maestro/correct_incorrect_lists.pkl'
        response_dict = pickle.load(f)

    avg_subject_power = [[] for i in range(len(conditions))]

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        tfr_path = list(subject_path.glob('**/*tfr.h5'))[0]
        power_epochs = mne.time_frequency.read_tfrs(tfr_path)[0]
        power_epochs.data = np.abs(power_epochs.data) ** 2  # get energy from complex tfr
        power_epochs = power_epochs.apply_baseline(baseline=(-0.75, -0.25), mode='logratio')

        if i == 0:
            freqs = power_epochs.freqs  # keep necessary objects once

        # convert correct and incorrect lists to strings
        responses = []
        for c in range(len(conditions)):
            response = [str(response_dict[subject_path.name][c][i]).zfill(3) for i in range(len(response_dict[subject_path.name][c]))]
            responses.append(response)

        for c, condition in enumerate(conditions):
            condition_list = condition_list_from_metadata(power_epochs.metadata, responses[c])
            power_avg = power_epochs[condition_list].average()
            power_avg = power_avg.crop(tmin=time_range[0], tmax=time_range[1])
            power_avg = np.mean(power_avg.data, axis=(0, 2))
            avg_subject_power[c].append(power_avg)

    average_list = [np.mean(np.array(power_list), axis=0) for power_list in avg_subject_power]
    sem_list = [np.std(np.array(power_list), axis=0) / np.sqrt(len(power_list)) for power_list in avg_subject_power]

    lines = []
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    for c in range(len(conditions)):
        lines.append(ax.plot(freqs, average_list[c])[0])
        ax.fill_between(freqs, average_list[c] - sem_list[c], average_list[c] + sem_list[c], alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power relative to baseline')
    ax.legend(lines, conditions)

    fig.savefig(os.path.join(save_folder, 'spectrum_' + condition_name + '.' + save_format))
    plt.close('all')

    return


def plot_trf_kernels_frequency_all_channels(tfr_data_path, trf_folder, features=None, ncols=2, save_format='png'):
    '''
    Time-frequency trf kernels
    ['Envelope', 'Envelope derivative', 'Surprise on', 'Surprise off', 'Surprise velocity', 'Surprise shift', 'Entropy on', 'Entropy off', 'Entropy velocity', 'Entropy shift']
    '''

    if features is None:
        features = ['Envelope', 'Envelope derivative', 'Surprise', 'Entropy']

    path = Path(tfr_data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    coef_list = []

    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)

        frequency_path = Path(os.path.join(trf_folder, subject_path.name))
        frequency_folders = sorted([x for x in frequency_path.iterdir() if x.is_dir()], key=lambda y: float(y.name))

        freq_coef_list = []
        for f, freq_path in enumerate(frequency_folders):

            file = open(os.path.join(freq_path, "rf.pickle"), 'rb')
            rf = pickle.load(file)
            file.close()

            freq_coef_list.append(np.sqrt(np.mean(rf.coef_ ** 2, axis=0)))
        freq_coef_list = np.array(freq_coef_list)
        coef_list.append(freq_coef_list)

    coef_list = np.array(coef_list)
    coefs_average = np.mean(coef_list, axis=0)

    # extract frequencies for plotting
    freqs = mne.time_frequency.read_tfrs(list(subject_folders[0].glob('**/*tfr.h5'))[0])[0].freqs
    # ratio = freqs[1] / freqs[0]
    # mesh_freqs = np.concatenate(([freqs[0] / ratio], freqs, freqs[-1] * ratio))  # add lower and upper frequencies
    # mesh_freqs = np.sqrt(mesh_freqs[:-1] * mesh_freqs[1:])  # take middle points of frequencies (geometric mean in linear scale)


    fig, axes = plt.subplots(nrows=ceil(rf.coef_.shape[1] / ncols), ncols=ncols, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):

        im = axes.T.flat[j].pcolormesh(rf.delays_ / rf.sfreq, freqs, coefs_average[:, j, :], shading='nearest', cmap='plasma')
        axes.T.flat[j].set_yscale('log')
        axes.T.flat[j].get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        axes.T.flat[j].axvline(0, linestyle='--', color='k')
        fig.colorbar(im, ax=axes.T.flat[j])
        axes.T.flat[j].set_title(features[j])
        axes.T.flat[j].set_ylabel('Frequency (Hz)')
        axes.T.flat[j].set_xlabel("Time (s)")

    fig.savefig(os.path.join(trf_folder, 'trf_kernels_frequency_all_channels' + '.' + save_format))
    plt.close('all')

    return


def plot_trf_correlation_tscore_topomap_frequency_all_subjects(tfr_data_path, scores_path, trf_path, stat_test='permut', save_format='png'):
    '''
    Same as plot_trf_topomaps_all_subjects_only, but with only one condition (no AM, A and AM-A).
    scores_path: channels scores
    '''

    scores = np.load(scores_path, allow_pickle=True)
    scores = np.arctanh(scores)  # Fisher z-transform

    path = Path(tfr_data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    # Plot average trf topomap
    tfr_path = list(subject_folders[0].glob('**/*tfr.h5'))[0]
    tfr = mne.time_frequency.read_tfrs(tfr_path)[0]  # load the tfr file to get the Info object
    tfr.pick(['meg'])
    # channels_to_keep = mne.channel_indices_by_type(raw.info)[channel_type]
    # channels indices depending on channel type. This is specific to the trf analysis since gradiometers pairs have been combined

    average_scores = np.mean(scores, axis=0)  # Average over participants
    # if channel_type == 'grad':
    #     average_scores = merge_grad_data(average_scores)  # take mean of the two channels scores for each sensor
    mag_info = tfr.copy().pick(['mag']).info  # Take Info of the magnetometers for channels positions
    # raw.pick([channel_type])

    freqs = tfr.freqs
    frequency_bands = {
                    'delta': [i for i in range(len(freqs)) if freqs[i] < 4],
                    'theta': [i for i in range(len(freqs)) if freqs[i] >= 4 and freqs[i] < 8],
                    'alpha': [i for i in range(len(freqs)) if freqs[i] >= 8 and freqs[i] < 14],
                    'beta': [i for i in range(len(freqs)) if freqs[i] >= 14 and freqs[i] < 30],
                    'gamma': [i for i in range(len(freqs)) if freqs[i] >= 30 and freqs[i] < 80],
                    'high-gamma': [i for i in range(len(freqs)) if freqs[i] >= 80],
                    }

    fig, axes = plt.subplots(nrows=1, ncols=len(frequency_bands), figsize=(20, 12), dpi=300)
    for i, band in enumerate(frequency_bands):

        frequencies = frequency_bands[band]

        # Find channels with significant score
        p_value, t_obs = test_channels_significance(np.mean(scores[:, :, frequencies], axis=2), info=mag_info, test=stat_test, channel_type='mag', return_t_obs=True)  # Permutation across subjects

        # define the statistical threshold (alpha)
        alpha = 0.05

        # im, cn = mne.viz.plot_topomap(average_scores[:, 0], mag_info, show=False, axes=axes[0], ch_type='mag', mask=p_value <= alpha)  # plot correlation scores
        im, cn = mne.viz.plot_topomap(t_obs, mag_info, show=False, axes=axes[i], ch_type='mag', mask=p_value <= alpha)  # plot t-scores
        axes[i].set_title(band)
        # clb = fig.colorbar(im, cax=axes[1])
        clb = fig.colorbar(im, ax=axes[i], shrink=1/4)
        clb.ax.set_title("t scores of prediction correlation (r)", fontsize=15)  # title on top of colorbar
        # axes[1].set_box_aspect(5)
        # axes[1].set_anchor('W')
        # axes[0].set_anchor('E')

    fig.savefig(os.path.join(trf_path, 'trf_tscore_topomap_frequency' + '_' + stat_test + '.' + save_format))
    plt.close()

    return


def plot_time_freq_and_spectrum(tf_folder, condition, spectrum_window=(0, 8), freq_zscore=True):

    path = Path(tf_folder)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    # remove bad subjects
    del subject_folders[22]
    del subject_folders[20]
    del subject_folders[14]
    del subject_folders[5]

    avg_subject_power = []
    avg_subject_spectrum = []

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        tfr_path = list(subject_path.glob('**/*tfr.h5'))[0]
        power_epochs = mne.time_frequency.read_tfrs(tfr_path)[0]
        power_epochs.data = np.abs(power_epochs.data) ** 2  # get energy from complex tfr
        power_epochs = power_epochs[10:]  # remove trials without task
        # power_epochs.data = (power_epochs.data - np.mean(power_epochs.data[:, :, :, 2:8], axis=(3), keepdims=True)) / np.mean(power_epochs.data[:, :, :, 2:8], axis=(3), keepdims=True)
        # power_epochs = power_epochs.apply_baseline((-1.2, -0.5), mode='zscore')  # z-scoring with baseline period
        # power_epochs.data = (power_epochs.data - np.mean(power_epochs.data, axis=(0, 3), keepdims=True)) / np.std(power_epochs.data, axis=(0, 3), keepdims=True)  # z-scoring per channel and frequency across trials and time
        spectrum_epochs = power_epochs.copy()
        spectrum_epochs = spectrum_epochs.crop(*spectrum_window)
        if freq_zscore:
            spectrum_epochs.data = (spectrum_epochs.data - np.mean(spectrum_epochs.data, axis=(0, 3), keepdims=True)) / np.std(spectrum_epochs.data, axis=(0, 3), keepdims=True)  # z-scoring per channel and frequency across trials and time
        # else:
            # spectrum_epochs.data = (spectrum_epochs.data - np.mean(spectrum_epochs.data, axis=(0, 2, 3), keepdims=True)) / np.std(spectrum_epochs.data, axis=(0, 2, 3), keepdims=True)  # z-scoring per channel across trials, frequency and time

        if condition is not None:
            condition_list = condition_list_from_metadata(power_epochs.metadata, [condition])
            power_avg = power_epochs[condition_list].average()  # average epochs
            spectrum_avg = spectrum_epochs[condition_list].average()
        else:
            power_avg = power_epochs.average()  # average epochs
            spectrum_avg = spectrum_epochs.average()
        avg_subject_power.append(np.mean(power_avg.data, axis=0))  # average channels
        avg_subject_spectrum.append(np.mean(spectrum_avg.data, axis=(0, 2)))  # average channels and time

        if i == 0:
            info, times, freqs = power_avg.info, power_avg.times, power_avg.freqs  # keep necessary objects once

    # plot with individual subject time-frequency
    plt.switch_backend('TkAgg')
    fig_indiv, axes_indiv = plt.subplots(nrows=5, ncols=5, sharex=True)
    for i, subject_path in enumerate(subject_folders):
        vmax = np.max(np.abs(avg_subject_power[i]))
        im = axes_indiv.flat[i].pcolormesh(times, freqs, avg_subject_power[i], shading='nearest', cmap='viridis', vmax=vmax, vmin=-vmax)
        # im = axes_indiv.flat[i].pcolormesh(times, freqs, avg_subject_power[i], shading='nearest', cmap='viridis')
        # im = axes_indiv.flat[i].pcolormesh(times, freqs, np.log10(avg_subject_power[i]), shading='nearest', cmap='viridis')
        axes_indiv.flat[i].set_yscale('log')
        axes_indiv.flat[i].get_yaxis().set_major_formatter(ticker.ScalarFormatter())
        axes_indiv.flat[i].axvline(0, linestyle='--', color='k')
        axes_indiv.flat[i].set_title(f"{subject_path.name}")
        # fig_indiv.colorbar(im)

    avg_subject_power = np.mean(avg_subject_power, axis=0)
    std_spectrum = np.std(avg_subject_spectrum, axis=0)
    avg_subject_spectrum = np.mean(avg_subject_spectrum, axis=0)

    vmax = np.max(np.abs(avg_subject_power))
    im = axes_indiv.flat[-1].pcolormesh(times, freqs, avg_subject_power, shading='nearest', cmap='viridis', vmax=vmax, vmin=-vmax)
    # im = axes_indiv.flat[-1].pcolormesh(times, freqs, avg_subject_power, shading='nearest', cmap='viridis')
    # im = axes_indiv.flat[-1].pcolormesh(times, freqs, np.log10(avg_subject_power), shading='nearest', cmap='viridis')
    axes_indiv.flat[-1].set_yscale('log')
    axes_indiv.flat[-1].get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    axes_indiv.flat[-1].axvline(0, linestyle='--', color='k')
    axes_indiv.flat[-1].set_title(f"Average")

    fig, axes = plt.subplots(nrows=1, ncols=1)
    # avg_subject_power = avg_subject_power[:, 5:-5]
    # times = times[5:-5]
    vmax = np.max(np.abs(avg_subject_power))
    im = axes.pcolormesh(times, freqs, avg_subject_power, shading='nearest', cmap='viridis', vmax=vmax, vmin=-vmax)
    # im = axes.pcolormesh(times, freqs, avg_subject_power, shading='nearest', cmap='viridis')
    # im = axes.pcolormesh(times, freqs, np.log10(avg_subject_power), shading='nearest', cmap='viridis')
    axes.set_yscale('log')
    axes.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    axes.axvline(0, linestyle='--', color='k')
    axes.set_title(f"{condition}")
    axes.set_ylabel('Frequency (Hz)')
    axes.set_xlabel("Time (s)")
    fig.colorbar(im)

    fig2, axes2 = plt.subplots(nrows=1, ncols=1)
    axes2.plot(freqs, avg_subject_spectrum, label=condition)
    sem = std_spectrum / np.sqrt(len(subject_folders))
    axes2.fill_between(freqs, avg_subject_spectrum - sem, avg_subject_spectrum + sem, alpha=0.5, linewidth=0)
    axes2.set_xscale('log')
    # axes2.set_yscale('log')
    axes2.set_xlabel('Frequency (Hz)', fontsize=20)
    axes2.set_ylabel('Power', fontsize=20)
    axes2.tick_params(axis='x', labelsize=15)
    axes2.tick_params(axis='y', labelsize=15)
    # axes2.set_title(f'Spectrum, time window: ({spectrum_window[0]}s, {spectrum_window[1]}s)')

    plt.show()

    return


def contrast_conditions(tf_folder, conditions, spectrum_window=(0, 8), correct_incorrect_path=None, new_old_path=None, freq_zscore=True):

    path = Path(tf_folder)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])
    if conditions == ['correct', 'incorrect']:
        del subject_folders[5]  # subject 5 does not have behavioral data
        with open(correct_incorrect_path, 'rb') as f:
            response_dict = pickle.load(f)
    if conditions == ['new', 'old']:
        del subject_folders[5]  # subject 5 does not have behavioral data
        with open(new_old_path, 'rb') as f:
            response_dict = pickle.load(f)

    # remove bad subjects (-1 in index to account removed subject 5)
    del subject_folders[21]
    del subject_folders[19]
    del subject_folders[13]

    avg_subject_power = [[] for c in range(len(conditions))]
    avg_subject_spectrum = [[] for c in range(len(conditions))]

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        tfr_path = list(subject_path.glob('**/*tfr.h5'))[0]
        power_epochs = mne.time_frequency.read_tfrs(tfr_path)[0]
        power_epochs.data = np.abs(power_epochs.data) ** 2  # get energy from complex tfr
        power_epochs = power_epochs[10:]  # remove trials without task
        power_epochs = power_epochs.apply_baseline((-1.2, -0.5), mode='zscore')  # z-scoring with baseline period
        spectrum_epochs = power_epochs.copy()
        # power_epochs.data = (power_epochs.data - np.mean(power_epochs.data, axis=(0, 3), keepdims=True)) / np.std(power_epochs.data, axis=(0, 3), keepdims=True)  # z-scoring per channel and frequency across trials and time
        if freq_zscore:
            spectrum_epochs.data = (spectrum_epochs.data - np.mean(spectrum_epochs.data, axis=(0, 3), keepdims=True)) / np.std(spectrum_epochs.data, axis=(0, 3), keepdims=True)  # z-scoring per channel and frequency across trials and time
        # else:
            # spectrum_epochs.data = (spectrum_epochs.data - np.mean(spectrum_epochs.data, axis=(0, 2, 3), keepdims=True)) / np.std(spectrum_epochs.data, axis=(0, 2, 3), keepdims=True)  # z-scoring per channel across trials, frequency and time

        # convert correct and incorrect lists to strings
        if (conditions == ['correct', 'incorrect']) | (conditions == ['new', 'old']):
            responses = []
            for c in range(len(conditions)):
                response = [str(response_dict[subject_path.name][c][i]).zfill(3) for i in range(len(response_dict[subject_path.name][c]))]
                responses.append(response)

        for c, condition in enumerate(conditions):
            if (conditions == ['correct', 'incorrect']) | (conditions == ['new', 'old']):
                condition_list = condition_list_from_metadata(power_epochs.metadata, responses[c])
            else:
                condition_list = condition_list_from_metadata(power_epochs.metadata, [condition])
            power_avg = power_epochs[condition_list].average()  # average epochs
            avg_subject_power[c].append(np.mean(power_avg.data, axis=0))  # average channels
            spectrum_avg = spectrum_epochs[condition_list].average()
            spectrum = spectrum_avg.crop(*spectrum_window)
            avg_subject_spectrum[c].append(np.mean(spectrum.data, axis=(0, 2)))  # average channels and time

        if i == 0:
            info, times, freqs = power_avg.info, power_avg.times, power_avg.freqs  # keep necessary objects once

    avg_subject_power = [np.mean(power_list, axis=0) for power_list in avg_subject_power]
    std_spectrum = [np.std(power_list, axis=0) for power_list in avg_subject_spectrum]
    avg_subject_spectrum = [np.mean(power_list, axis=0) for power_list in avg_subject_spectrum]

    plt.switch_backend('TkAgg')
    n_cols = int(factorial(len(conditions)) / (factorial(2) * factorial(len(conditions)-2)))  # n_conditions choose 2
    fig, axes = plt.subplots(nrows=1, ncols=n_cols + 1, gridspec_kw={'width_ratios': [1 for i in range(n_cols)] + [0.1]})  # +1 for the colorbar
    vmax=0
    for i in range(len(conditions)):
        for j in range(i+1, len(conditions)):
            vmax = max(vmax, np.max(np.abs(avg_subject_power[j] - avg_subject_power[i])))
    plot_i = 0
    for i in range(len(conditions)):
        for j in range(i+1, len(conditions)):
            im = axes[plot_i].pcolormesh(times, freqs, avg_subject_power[j] - avg_subject_power[i], shading='nearest', cmap='viridis', vmin=-vmax, vmax=vmax)
            # im = axes[plot_i].pcolormesh(times, freqs, avg_subject_power[j] - avg_subject_power[i], shading='nearest', cmap='viridis')
            axes[plot_i].set_yscale('log')
            axes[plot_i].get_yaxis().set_major_formatter(ticker.ScalarFormatter())
            axes[plot_i].axvline(0, linestyle='--', color='k')
            axes[plot_i].set_title(f"{conditions[j]} - {conditions[i]}")
            if plot_i == 0:
                axes[plot_i].set_ylabel('Frequency (Hz)')
            axes[plot_i].set_xlabel("Time (s)")
            plot_i += 1
    fig.colorbar(im, cax=axes[-1])
    axes[-1].set_title("z-score")

    # min_spectra = min([np.min(spectr) for spectr in avg_subject_spectrum])
    fig2, axes2 = plt.subplots(nrows=1, ncols=1)
    for c in range(len(conditions)):
        axes2.plot(freqs, avg_subject_spectrum[c], label=conditions[c])
        sem = std_spectrum[c] / np.sqrt(len(subject_folders))
        axes2.fill_between(freqs, avg_subject_spectrum[c] - sem, avg_subject_spectrum[c] + sem, alpha=0.5, linewidth=0)
    axes2.set_xscale('log')
    axes2.set_xlabel('Frequency (Hz)')
    axes2.set_ylabel('Power (z-score)')
    axes2.set_title(f'Spectrum, time window: ({spectrum_window[0]}s, {spectrum_window[1]}s)')
    axes2.legend()

    plt.show()

    return


def spectrum_double_conditions(tf_folder, conditions1, conditions2, spectrum_window=(0, 8), correct_incorrect_path=None, new_old_path=None, freq_zscore=True, only_correct=False):

    path = Path(tf_folder)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])
    del subject_folders[5]  # subject 5 does not have behavioral data

    # remove bad subjects (-1 in index to account removed subject 5)
    del subject_folders[21]
    del subject_folders[19]
    del subject_folders[13]

    flag = False
    if conditions1 == ['correct', 'incorrect']:
        with open(correct_incorrect_path, 'rb') as f:
            response_dict = pickle.load(f)
        flag = True
    if conditions1 == ['new', 'old']:
        with open(new_old_path, 'rb') as f:
            response_dict = pickle.load(f)
        flag = True
    if conditions2 == ['correct', 'incorrect']:
        if flag:
            raise ValueError('2 response_dict created')
        with open(correct_incorrect_path, 'rb') as f:
            response_dict = pickle.load(f)
        flag = True
    if conditions2 == ['new', 'old']:
        if flag:
            raise ValueError('2 response_dict created')
        with open(new_old_path, 'rb') as f:
            response_dict = pickle.load(f)
        flag = True

    if only_correct:
        if flag:
            raise ValueError('only_correct should be without other response_dict')
        with open(correct_incorrect_path, 'rb') as f:
            correct_dict = pickle.load(f)

    avg_subject_spectrum = [[[] for c in range(len(conditions1))] for d in range(len(conditions2))]

    remove_subjects = []
    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        tfr_path = list(subject_path.glob('**/*tfr.h5'))[0]
        power_epochs = mne.time_frequency.read_tfrs(tfr_path)[0]
        power_epochs.data = np.abs(power_epochs.data) ** 2  # get energy from complex tfr
        power_epochs = power_epochs[10:]  # remove trials without task
        power_epochs = power_epochs.apply_baseline((-1.2, -0.5), mode='zscore')  # z-scoring with baseline period
        spectrum_epochs = power_epochs
        if freq_zscore:
            spectrum_epochs.data = (spectrum_epochs.data - np.mean(spectrum_epochs.data, axis=(0, 3), keepdims=True)) / np.std(spectrum_epochs.data, axis=(0, 3), keepdims=True)  # z-scoring per channel and frequency across trials and time
        # else:
        #     spectrum_epochs.data = (spectrum_epochs.data - np.mean(spectrum_epochs.data, axis=(0, 2, 3), keepdims=True)) / np.std(spectrum_epochs.data, axis=(0, 2, 3), keepdims=True)  # z-scoring per channel across trials, frequency and time

        # convert correct and incorrect lists to strings
        if (conditions1 == ['correct', 'incorrect']) | (conditions1 == ['new', 'old']):
            responses = []
            for c in range(len(conditions1)):
                response = [str(response_dict[subject_path.name][c][i]).zfill(3) for i in range(len(response_dict[subject_path.name][c]))]
                responses.append(response)
        if (conditions2 == ['correct', 'incorrect']) | (conditions2 == ['new', 'old']):
            responses = []
            for c in range(len(conditions2)):
                response = [str(response_dict[subject_path.name][c][i]).zfill(3) for i in range(len(response_dict[subject_path.name][c]))]
                responses.append(response)
        if only_correct:
            correct_responses = [str(correct_dict[subject_path.name][0][i]).zfill(3) for i in range(len(correct_dict[subject_path.name][0]))]  # correct trials only
            correct_list = condition_list_from_metadata(spectrum_epochs.metadata, correct_responses)
        for c, cond1 in enumerate(conditions1):
            for d, cond2 in enumerate(conditions2):
                if (conditions1 == ['correct', 'incorrect']) | (conditions1 == ['new', 'old']):
                    condition_list1 = condition_list_from_metadata(spectrum_epochs.metadata, responses[c])
                else:
                    condition_list1 = condition_list_from_metadata(spectrum_epochs.metadata, [cond1])

                if (conditions2 == ['correct', 'incorrect']) | (conditions2 == ['new', 'old']):
                    condition_list2 = condition_list_from_metadata(spectrum_epochs.metadata, responses[d])
                else:
                    condition_list2 = condition_list_from_metadata(spectrum_epochs.metadata, [cond2])

                # keep indices that are in both lists
                condition_list = [ind for ind in condition_list1 if ind in condition_list2]
                if only_correct:
                    condition_list = [ind for ind in condition_list if ind in correct_list]
                if len(condition_list) == 0:
                    remove_subjects.append(i)
                spectrum_avg = spectrum_epochs[condition_list].average()
                spectrum = spectrum_avg.crop(*spectrum_window)
                avg_subject_spectrum[d][c].append(np.mean(spectrum.data, axis=(0, 2)))  # average channels and time

        if i == 0:
            info, times, freqs = spectrum_avg.info, spectrum_avg.times, spectrum_avg.freqs  # keep necessary objects once

    remove_subjects = np.unique(remove_subjects)
    good_subjects = [s for s in range(i+1)]
    for s in remove_subjects:
        good_subjects.remove(s)
    std_spectrum = [[np.std(np.array(power_list)[good_subjects], axis=0) for power_list in avg_subject_spectrum[d]] for d in range(len(conditions2))]
    avg_subject_spectrum = [[np.mean(np.array(power_list)[good_subjects], axis=0) for power_list in avg_subject_spectrum[d]] for d in range(len(conditions2))]

    plt.switch_backend('TkAgg')

    fig2, axes2 = plt.subplots(nrows=len(conditions2), ncols=1, sharex=True)
    for d in range(len(conditions2)):
        for c in range(len(conditions1)):
            axes2[d].plot(freqs, avg_subject_spectrum[d][c], label=conditions1[c])
            sem = std_spectrum[d][c] / np.sqrt(len(good_subjects))
            axes2[d].fill_between(freqs, avg_subject_spectrum[d][c] - sem, avg_subject_spectrum[d][c] + sem, alpha=0.5, linewidth=0)
            axes2[d].set_ylabel('Power (z-score)')
            axes2[d].set_title(conditions2[d])
            axes2[d].legend()
    axes2[d].set_xscale('log')
    axes2[d].set_xlabel('Frequency (Hz)')

    fig1, axes1 = plt.subplots(nrows=1, ncols=1)
    for c in range(len(conditions1)):
        cond_array = np.sum([spectr[c] for spectr in avg_subject_spectrum], axis=0)
        axes1.plot(freqs, cond_array, label=conditions1[c])
    axes1.set_ylabel('Power (z-score)')
    axes1.legend()
    axes1.set_xscale('log')
    axes1.set_xlabel('Frequency (Hz)')

    plt.show()

    return

def extract_trial_power(tf_folder, save_path):

    path = Path(tf_folder)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    # remove bad subjects
    del subject_folders[22]
    del subject_folders[20]
    del subject_folders[14]
    del subject_folders[5]

    subject_names_list = []
    stim_name_list = []
    delta_1s = []
    theta_1s = []
    alpha_1s = []
    beta_1s = []
    high_gamma_1s = []
    delta_7s = []
    theta_7s = []
    alpha_7s = []
    beta_7s = []
    high_gamma_7s = []
    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        tfr_path = list(subject_path.glob('**/*tfr.h5'))[0]
        power_epochs = mne.time_frequency.read_tfrs(tfr_path)[0]
        power_epochs.data = np.abs(power_epochs.data) ** 2  # get energy from complex tfr
        power_epochs = power_epochs[[int(stim_name[5:8]) >= 10 for stim_name in power_epochs.metadata['stim_name']]]
        power_epochs = power_epochs.apply_baseline((-0.8, -0.3), mode='zscore')  # z-scoring with baseline period
        power_1s = np.mean(power_epochs.copy().crop(0, 1).data, axis=(1, 3))  # average channels and time
        power_7s = np.mean(power_epochs.copy().crop(1, 8).data, axis=(1, 3))  # average channels and time

        delta_band = [i for i in range(len(power_epochs.freqs)) if ((power_epochs.freqs[i] >= 0.5) & (power_epochs.freqs[i] <= 4))]
        theta_band = [i for i in range(len(power_epochs.freqs)) if ((power_epochs.freqs[i] >= 4) & (power_epochs.freqs[i] <= 8))]
        alpha_band = [i for i in range(len(power_epochs.freqs)) if ((power_epochs.freqs[i] >= 8) & (power_epochs.freqs[i] <= 13))]
        # beta_band = [i for i in range(len(power_epochs.freqs)) if ((power_epochs.freqs[i] >= 13) & (power_epochs.freqs[i] <= 32))]
        beta_band = [i for i in range(len(power_epochs.freqs)) if ((power_epochs.freqs[i] >= 17) & (power_epochs.freqs[i] <= 23))]
        high_gamma_band = [i for i in range(len(power_epochs.freqs)) if ((power_epochs.freqs[i] >= 70) & (power_epochs.freqs[i] <= 99))]

        for e in range(len(power_epochs)):
            subject_names_list.append(subject_path.name)
            stim_name_list.append(power_epochs[e].metadata['stim_name'].to_numpy()[0])
            delta_1s.append(np.mean(power_1s[e, delta_band]))  # average frequencies
            theta_1s.append(np.mean(power_1s[e, theta_band]))
            alpha_1s.append(np.mean(power_1s[e, alpha_band]))
            beta_1s.append(np.mean(power_1s[e, beta_band]))
            high_gamma_1s.append(np.mean(power_1s[e, high_gamma_band]))
            delta_7s.append(np.mean(power_7s[e, delta_band]))
            theta_7s.append(np.mean(power_7s[e, theta_band]))
            alpha_7s.append(np.mean(power_7s[e, alpha_band]))
            beta_7s.append(np.mean(power_7s[e, beta_band]))
            high_gamma_7s.append(np.mean(power_7s[e, high_gamma_band]))

    df = pd.DataFrame({'subject': subject_names_list, 'stim_name': stim_name_list, 'delta_1s': delta_1s,
                       'theta_1s': theta_1s, 'alpha_1s': alpha_1s, 'beta_1s': beta_1s, 'high_gamma_1s': high_gamma_1s,
                       'delta_7s': delta_7s, 'theta_7s': theta_7s, 'alpha_7s': alpha_7s, 'beta_7s': beta_7s,
                       'high_gamma_7s': high_gamma_7s})

    df.to_csv(os.path.join(save_path, 'average_power_bands.csv'))

    return


def seeg_time_freq_epochs(data_path, save_path, regions, first_parcel_label='advanced', n_jobs=-1):
    os.makedirs(save_path, exist_ok=True)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        epochs = mne.read_epochs(epochs_path)

        epochs, channel_parcellations = select_seeg_channels_from_parcellation(epochs, regions, first_parcel_label=first_parcel_label)

        if epochs is not None:
            epochs = epochs.apply_baseline((-1.5, 0))
            fs = 400
            epochs.resample(fs, n_jobs=-1)
            epochs.filter(0.5, None, n_jobs=-1)

            epochs._data = epochs.get_data().astype(np.float32)  # reduce memory usage

            # time_frequency decomposition
            # freqs = np.logspace(*np.log10([0.5, 95]), num=40)
            # freqs = np.logspace(*np.log10([1, 100]), num=100)
            # freqs = np.linspace(1, 9.5, num=18).tolist() + np.linspace(10, 190, num=91).tolist()
            freqs = np.logspace(*np.log10([1, 10]), num=40).tolist() + np.logspace(*np.log10([10, 190]), num=91).tolist()
            print(freqs)
            # n_cycles = freqs  # different number of cycle per frequency
            n_cycles = 6
            # n_cycles = np.minimum(freqs * 2, 15 * np.ones_like(freqs)) # 15 is the number of cycles for 7.5Hz. For higher frequencies we do not need to restrict the bandwidth
            power_epochs = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, decim=8,
                                      n_jobs=n_jobs, zero_mean=True, average=False, output='complex', verbose=1)
            # power_epochs = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, decim=8,
            #                           n_jobs=n_jobs, zero_mean=True, average=False, output='power', verbose=1)

            del epochs

            power_epochs.save(fname=os.path.join(save_path, subject_name + "-tfr.h5"), overwrite=True)
            del power_epochs
            # gc.collect()
    print("Done")


def plot_tfr(data_path):
    plt.switch_backend('TkAgg')

    path = Path(data_path)
    subject_list = sorted(list(path.glob('*-tfr.h5')))

    tfr = []
    std_tfr = []
    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        power_epochs = mne.time_frequency.read_tfrs(epochs_path)[0]
        power_epochs = power_epochs.apply_baseline(baseline=(-1.5, -0.25), mode='percent')
        print(power_epochs.ch_names)
        print(power_epochs.times)
        print(power_epochs.freqs)
        power_epochs.crop(-1, 8.5)
        # power_epochs.pick("B'1")
        print(power_epochs.data.shape)
        print(power_epochs.metadata)
        print(power_epochs.metadata.iloc[89, 0])

        tfr.append(np.mean(power_epochs.data, axis=(0, 1)))  # average epochs and channels
        std_tfr.append(np.mean(power_epochs.data, axis=(0, 1)) / np.std(np.mean(power_epochs.data, axis=1), axis=0))  # normalized
    # tfr = np.concatenate(tfr, axis=0)  # concatenate channels
    tfr = np.array(tfr)
    std_tfr = np.array(std_tfr)
    print(tfr.shape)

    # plt.pcolormesh(power_epochs.times, power_epochs.freqs, np.mean(tfr, axis=0), shading='nearest', cmap='viridis')
    # plt.gca().axvline(0, ls='--', c='k')
    # cbar = plt.colorbar()
    # plt.show()

    tvalues = np.mean(tfr, axis=0) / (np.std(tfr, axis=0) / np.sqrt(tfr.shape[0]))
    vmax = np.max(np.abs(tvalues))
    # vmax = 7
    plt.pcolormesh(power_epochs.times, power_epochs.freqs, tvalues, shading='nearest', cmap='Spectral_r', vmin=-vmax, vmax=vmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.yscale('log')
    plt.gca().axvline(0, ls='--', c='k')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('tvalue', rotation=270)

    plt.figure()
    tvalues = np.mean(tfr, axis=0)
    # vmax = np.max(np.abs(tvalues))
    vmax = 3
    plt.pcolormesh(power_epochs.times, power_epochs.freqs, tvalues, shading='nearest', cmap='Spectral_r', vmin=-vmax, vmax=vmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.yscale('log')
    plt.gca().axvline(0, ls='--', c='k')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('average power', rotation=270)

    # low frequencies
    plt.figure()
    tvalues = np.mean(tfr, axis=0) / (np.std(tfr, axis=0) / np.sqrt(tfr.shape[0]))
    # vmax = np.max(np.abs(tvalues))
    vmax = 7
    plt.pcolormesh(power_epochs.times, power_epochs.freqs[:17], tvalues[:17, :], shading='nearest', cmap='Spectral_r', vmin=-vmax,
                   vmax=vmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.yscale('log')
    plt.gca().axvline(0, ls='--', c='k')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('tvalue', rotation=270)
    plt.figure()
    tvalues = np.mean(tfr, axis=0)
    # vmax = np.max(np.abs(tvalues))
    vmax = 2
    plt.pcolormesh(power_epochs.times, power_epochs.freqs[:17], tvalues[:17, :], shading='nearest', cmap='Spectral_r', vmin=-vmax,
                   vmax=vmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.yscale('log')
    plt.gca().axvline(0, ls='--', c='k')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('average', rotation=270)

    plt.figure()
    tvalues = np.mean(std_tfr, axis=0) / (np.std(std_tfr, axis=0) / np.sqrt(std_tfr.shape[0]))
    # vmax = np.max(np.abs(tvalues))
    vmax = 15
    plt.pcolormesh(power_epochs.times, power_epochs.freqs, tvalues, shading='nearest', cmap='Spectral_r', vmin=-vmax,
                   vmax=vmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.yscale('log')
    plt.gca().axvline(0, ls='--', c='k')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('tvalue from normalized', rotation=270)

    plt.figure()
    tvalues = np.mean(std_tfr, axis=0)
    # vmax = np.max(np.abs(tvalues))
    vmax = 1
    plt.pcolormesh(power_epochs.times, power_epochs.freqs, tvalues, shading='nearest', cmap='Spectral_r', vmin=-vmax, vmax=vmax)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.yscale('log')
    plt.gca().axvline(0, ls='--', c='k')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('average from normalized', rotation=270)

    plt.show()

    # for c in range(tfr.shape[0]):
    #     plt.pcolormesh(power_epochs.times, power_epochs.freqs, tfr[c, :, :], shading='nearest', cmap='viridis')
    #     plt.gca().axvline(0, ls='--', c='k')
    #     plt.colorbar()
    #     plt.show()


def plot_tfr_by_rate(data_path):
    plt.switch_backend('TkAgg')

    path = Path(data_path)
    subject_list = sorted(list(path.glob('*-tfr.h5')))

    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']
    tfr = [[] for r in range(len(rates))]
    sem_tfr = [[] for r in range(len(rates))]
    power = [[] for r in range(len(rates))]

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
        print(subject_name)

        power_epochs = mne.time_frequency.read_tfrs(epochs_path)[0]
        power_epochs = power_epochs.apply_baseline(baseline=(-1.5, -0.25), mode='percent')
        power_epochs.crop(-1, 8.5)
        power_epochs_cropped = power_epochs.copy().crop(0, 8)
        stim_names = power_epochs.metadata['stim_name'].tolist()

        for r, rate in enumerate(rates):
            rate_epochs = [e for e in range(len(stim_names)) if rate in stim_names[e]]
            # tfr[r].append(np.mean(np.mean(power_epochs.data[rate_epochs, :, :, :], axis=0), axis=0))
            tfr[r].append(np.mean(np.mean(power_epochs.data[rate_epochs, :, :, :], axis=1), axis=0) / np.std(np.mean(power_epochs.data[rate_epochs, :, :, :], axis=1), axis=0))  # normalized
            # sem_tfr[r].append(np.std(np.mean(power_epochs.data[rate_epochs, :, :, :], axis=1), axis=0) / np.sqrt(len(rate_epochs)))  # save sem across trials for subject-level tval
            # sem_tfr[r].append(np.std(np.mean(power_epochs.data[rate_epochs, :, :, :], axis=1), axis=0))  # save std across trials
            # power[r].append(np.mean(np.mean(power_epochs_cropped.data[rate_epochs, :, :, :], axis=(0, 3)), axis=0))
            power[r].append(np.mean((np.mean(power_epochs_cropped.data[rate_epochs, :, :, :], axis=(0, 1)) / np.std(np.mean(power_epochs_cropped.data[rate_epochs, :, :, :], axis=1), axis=0)), axis=-1))  # zscored

    fig, axs = plt.subplots(6, 2, sharey='col', sharex=True)
    for r, rate in enumerate(rates):
        power_rate = np.array(power[r])
        tvalues = np.mean(power_rate, axis=0) / (np.std(power_rate, axis=0) / np.sqrt(power_rate.shape[0]))
        mean = np.mean(power_rate, axis=0)
        sem = np.std(power_rate, axis=0) / np.sqrt(power_rate.shape[0])
        # vmax = 5
        axs[r, 0].plot(power_epochs.freqs, mean, label=rate)
        axs[r, 0].fill_between(power_epochs.freqs, mean-sem, mean+sem, alpha=0.5, lw=0)
        # axs[0].set_xscale('log')
        axs[r, 0].axhline(0, ls='--', c='k')
        axs[r, 0].axvline(float(rate[:3]), c='r', lw=0.5)
        axs[r, 0].set_ylabel('Power')
        axs[r, 0].set_xlabel('Frequency (Hz)')
        axs[r, 0].set_xscale('log')
        # axs[0, r].set_title(rate)
        axs[r, 0].legend()
        axs[r, 1].plot(power_epochs.freqs, tvalues, label=rate)
        axs[r, 1].axhline(0, ls='--', c='k')
        axs[r, 1].axvline(float(rate[:3]), c='r', lw=0.5)
        axs[r, 1].set_ylabel('t value')
        axs[r, 1].set_xlabel('Frequency (Hz)')
        axs[r, 1].set_xscale('log')
        axs[r, 1].legend()
    # plt.show()

    # power spectrum by rate on one plot
    color_list = plt.get_cmap('Set2')
    fig, axs = plt.subplots(1, 2, sharey=True)
    for r, rate in enumerate(rates):
        rate_freq = float(rate[:3])
        power_rate = np.array(power[r])
        tvalues = np.mean(power_rate, axis=0) / (np.std(power_rate, axis=0) / np.sqrt(power_rate.shape[0]))
        mean = np.mean(power_rate, axis=0)
        sem = np.std(power_rate, axis=0) / np.sqrt(power_rate.shape[0])
        freq_idx = np.argmin(np.abs(power_epochs.freqs - rate_freq))
        # vmax = 5
        axs[0].plot(mean, power_epochs.freqs, label=rate, color=color_list.colors[r], lw=3)
        # axs[0].fill_betweenx(power_epochs.freqs, mean - sem, mean + sem, alpha=0.5, lw=0, color=color_list.colors[r])
        # axs[0].set_xscale('log')
        axs[0].axvline(0, ls='--', c='k')
        axs[0].hlines(rate_freq, lw=3, color=color_list.colors[r], ls='--', xmin=0, xmax=mean[freq_idx])
        axs[0].scatter(mean[freq_idx], rate_freq, color=color_list.colors[r], s=100, zorder=3)
        axs[0].set_xlabel('Power')
        axs[0].set_ylabel('Frequency (Hz)')
        axs[0].set_ylim(power_epochs.freqs[0], power_epochs.freqs[-1])
        axs[0].set_yscale('log')
        # axs[0, r].set_title(rate)
        axs[0].legend()
        axs[1].plot(tvalues, power_epochs.freqs, label=rate, color=color_list.colors[r], lw=3)
        axs[1].axvline(0, ls='--', c='k')
        axs[1].hlines(rate_freq, lw=3, color=color_list.colors[r], ls='--', xmin=0, xmax=tvalues[freq_idx])
        axs[1].scatter(tvalues[freq_idx], rate_freq, color=color_list.colors[r], s=100, zorder=3)
        axs[1].set_xlabel('t value')
        axs[1].set_ylabel('Frequency (Hz)')
        axs[1].set_ylim(power_epochs.freqs[0], power_epochs.freqs[-1])
        axs[1].set_yscale('log')
        axs[1].legend()

    fig, axs = plt.subplots(6, 1, sharex=True, sharey=True)
    for r, rate in enumerate(rates):
        rate_notes = np.arange(0, 8.001, 1/float(rate[:3]))
        print(rate_notes)
        # tfr = np.concatenate(tfr, axis=0)  # concatenate channels
        tfr_rate = np.array(tfr[r])
        # mean = np.mean(tfr_rate, axis=0) / (np.std(tfr_rate, axis=0) / np.sqrt(tfr_rate.shape[0]))
        mean = np.mean(tfr_rate, axis=0)
        # vmax = np.max(np.abs(mean))
        # vmax = 7
        # vmax = 3
        vmax = 1
        print(mean.shape)
        im = axs[r].pcolormesh(power_epochs.times, power_epochs.freqs, mean, shading='nearest', cmap='Spectral_r', vmin=-vmax, vmax=vmax)
        # im = axs[r].pcolormesh(power_epochs.times, power_epochs.freqs, mean, shading='nearest', cmap='Spectral_r')
        if r == len(rates) - 1:
            axs[r].set_xlabel('Time (s)')
        axs[r].set_ylabel('Frequency (Hz)')
        axs[r].set_yscale('log')
        axs[r].axvline(0, ls='--', c='k')
        axs[r].set_title(rate)
        axs[r].axhline(float(rate[:3]), c='blue', lw=0.5)
        for note in rate_notes[1:]:
            axs[r].axvline(note, ls='--', c='gray')
        cbar = plt.colorbar(im)
        if r==0:
            cbar.ax.set_ylabel('percent relative to baseline', rotation=270)
    # plt.show()

    fig, axs = plt.subplots(6, 1, sharex=True, sharey=True)
    for r, rate in enumerate(rates):
        rate_notes = np.arange(0, 8.001, 1 / float(rate[:3]))
        print(rate_notes)
        # tfr = np.concatenate(tfr, axis=0)  # concatenate channels
        tfr_rate = np.array(tfr[r])
        mean = np.mean(tfr_rate, axis=0) / (np.std(tfr_rate, axis=0) / np.sqrt(tfr_rate.shape[0]))
        # mean = np.mean(tfr_rate, axis=0)
        # vmax = np.max([np.max(np.abs(tval)) for tval in tvalues])
        vmax = 7
        # vmax = 5
        # vmax = np.max(np.abs(mean))
        print(mean.shape)
        im = axs[r].pcolormesh(power_epochs.times, power_epochs.freqs, mean, shading='nearest', cmap='Spectral_r', vmin=-vmax, vmax=vmax)
        # im = axs[r].pcolormesh(power_epochs.times, power_epochs.freqs, mean, shading='nearest', cmap='Spectral_r')
        if r==len(rates)-1:
            axs[r].set_xlabel('Time (s)')
        axs[r].set_ylabel('Frequency (Hz)')
        axs[r].set_yscale('log')
        axs[r].axvline(0, ls='--', c='k')
        axs[r].set_title(rate)
        axs[r].axhline(float(rate[:3]), c='blue', lw=0.5)
        for note in rate_notes[1:]:
            axs[r].axvline(note, ls='--', c='gray')
        cbar = plt.colorbar(im)
        if r==0:
            cbar.ax.set_ylabel('percent relative to baseline (tval)', rotation=270)

    # same with only low freq
    fig, axs = plt.subplots(6, 1, sharex=True, sharey=True)
    for r, rate in enumerate(rates):
        rate_notes = np.arange(0, 8.001, 1 / float(rate[:3]))
        print(rate_notes)
        # tfr = np.concatenate(tfr, axis=0)  # concatenate channels
        tfr_rate = np.array(tfr[r])
        # mean = np.mean(tfr_rate, axis=0) / (np.std(tfr_rate, axis=0) / np.sqrt(tfr_rate.shape[0]))
        mean = np.mean(tfr_rate, axis=0)
        # vmax = np.max(np.abs(mean))
        # vmax = 7
        vmax = 2
        print(mean.shape)
        im = axs[r].pcolormesh(power_epochs.times, power_epochs.freqs[:17], mean[:17, :], shading='nearest', cmap='Spectral_r',
                               vmin=-vmax, vmax=vmax)
        # im = axs[r].pcolormesh(power_epochs.times, power_epochs.freqs, mean, shading='nearest', cmap='Spectral_r')
        if r == len(rates) - 1:
            axs[r].set_xlabel('Time (s)')
        axs[r].set_ylabel('Frequency (Hz)')
        axs[r].set_yscale('log')
        axs[r].axvline(0, ls='--', c='k')
        axs[r].set_title(rate)
        axs[r].axhline(float(rate[:3]), c='blue', lw=0.5)
        for note in rate_notes[1:]:
            axs[r].axvline(note, ls='--', c='gray')
        cbar = plt.colorbar(im)
        if r == 0:
            cbar.ax.set_ylabel('percent relative to baseline', rotation=270)
    fig, axs = plt.subplots(6, 1, sharex=True, sharey=True)
    for r, rate in enumerate(rates):
        rate_notes = np.arange(0, 8.001, 1 / float(rate[:3]))
        print(rate_notes)
        # tfr = np.concatenate(tfr, axis=0)  # concatenate channels
        tfr_rate = np.array(tfr[r])
        mean = np.mean(tfr_rate, axis=0) / (np.std(tfr_rate, axis=0) / np.sqrt(tfr_rate.shape[0]))
        # mean = np.mean(tfr_rate, axis=0)
        # vmax = np.max([np.max(np.abs(tval)) for tval in tvalues])
        vmax = 7
        # vmax = 5
        # vmax = np.max(np.abs(mean))
        print(mean.shape)
        im = axs[r].pcolormesh(power_epochs.times, power_epochs.freqs[:17], mean[:17, :], shading='nearest', cmap='Spectral_r',
                               vmin=-vmax, vmax=vmax)
        # im = axs[r].pcolormesh(power_epochs.times, power_epochs.freqs, mean, shading='nearest', cmap='Spectral_r')
        if r == len(rates) - 1:
            axs[r].set_xlabel('Time (s)')
        axs[r].set_ylabel('Frequency (Hz)')
        axs[r].set_yscale('log')
        axs[r].axvline(0, ls='--', c='k')
        axs[r].set_title(rate)
        axs[r].axhline(float(rate[:3]), c='blue', lw=0.5)
        for note in rate_notes[1:]:
            axs[r].axvline(note, ls='--', c='gray')
        cbar = plt.colorbar(im)
        if r == 0:
            cbar.ax.set_ylabel('percent relative to baseline (tval)', rotation=270)

    # plot note evoked tfr
    fig, axs = plt.subplots(6, 1, sharex=True)
    for r, rate in enumerate(rates):
        rate_duration = 1/float(rate[:3])
        rate_notes = np.arange(0, 8.001, rate_duration)
        print(rate_notes)
        # tfr = np.concatenate(tfr, axis=0)  # concatenate channels
        tfr_rate = np.array(tfr[r])
        print(tfr_rate.shape)
        mean = np.mean(tfr_rate, axis=0)
        # vmax = np.max([np.max(np.abs(tval)) for tval in tvalues])
        vmax = 5
        # Determine the fixed window size based on the first note
        fixed_window_size = np.argmin(np.abs(power_epochs.times - (rate_notes[0] + rate_duration))) - np.argmin(np.abs(power_epochs.times - rate_notes[0]))
        evoked_tval = []
        for note in rate_notes[:-1]:
            # Find the time indices within the current note window
            start_idx = np.argmin(np.abs(power_epochs.times - note))
            end_idx = start_idx + fixed_window_size
            evoked_tval.append(mean[:, start_idx:end_idx])
        evoked_tval = np.mean(evoked_tval, axis=0)  # average windows
        # evoked_tval = np.mean(evoked_tval, axis=0) / (np.std(evoked_tval, axis=0) / np.sqrt(evoked_tval.shape[0]))
        print(evoked_tval.shape)
        times = np.linspace(0, rate_notes[1], evoked_tval.shape[1])
        print(times)
        # im = axs[i].pcolormesh(times, power_epochs.freqs, evoked_tval, shading='nearest', cmap='bwr', vmin=-vmax, vmax=vmax)
        im = axs[r].pcolormesh(times, power_epochs.freqs, evoked_tval, shading='nearest', cmap='bwr')
        axs[r].set_xlabel('Time (s)')
        axs[r].set_ylabel('Frequency (Hz)')
        axs[r].set_yscale('log')
        axs[r].axvline(0, ls='--', c='k')
        axs[r].set_title(rate)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('percent', rotation=270)
    plt.show()

    # # by subject
    # for s in range(np.array(tfr[0]).shape[0]):
    #     fig, axs = plt.subplots(6, 1, sharex=True, sharey=True)
    #     for r, rate in enumerate(rates):
    #         rate_notes = np.arange(0, 8.001, 1 / float(rate[:3]))
    #         print(rate_notes)
    #         # tfr = np.concatenate(tfr, axis=0)  # concatenate channels
    #         tfr_rate = np.array(tfr[r])[s]
    #         # mean = np.mean(tfr_rate, axis=0) / (np.std(tfr_rate, axis=0) / np.sqrt(tfr_rate.shape[0]))
    #         # mean = np.mean(tfr_rate, axis=0)
    #         # vmax = np.max([np.max(np.abs(tval)) for tval in tvalues])
    #         # vmax = 7
    #         vmax = 1
    #         print(tfr_rate.shape)
    #         im = axs[r].pcolormesh(power_epochs.times, power_epochs.freqs, tfr_rate, shading='nearest', cmap='Spectral_r', vmin=-vmax, vmax=vmax)
    #         # im = axs[r].pcolormesh(power_epochs.times, power_epochs.freqs, tfr_rate, shading='nearest', cmap='Spectral_r')
    #         if r == len(rates) - 1:
    #             axs[r].set_xlabel('Time (s)')
    #         axs[r].set_ylabel('Frequency (Hz)')
    #         axs[r].set_yscale('log')
    #         axs[r].axvline(0, ls='--', c='k')
    #         axs[r].set_title(rate)
    #         for note in rate_notes[1:]:
    #             axs[r].axvline(note, ls='--', c='gray')
    #         cbar = plt.colorbar(im)
    #         cbar.ax.set_ylabel('percent relative to baseline', rotation=270)
    # plt.show()
    #
    # # by subject tval across trials
    # for s in range(np.array(tfr[0]).shape[0]):
    #     fig, axs = plt.subplots(6, 1, sharex=True, sharey=True)
    #     for r, rate in enumerate(rates):
    #         rate_notes = np.arange(0, 8.001, 1 / float(rate[:3]))
    #         print(rate_notes)
    #         # tfr = np.concatenate(tfr, axis=0)  # concatenate channels
    #         tfr_rate = np.array(tfr[r])[s]
    #         sem_tfr_rate = np.array(sem_tfr[r])[s]
    #         tfr_rate = tfr_rate / sem_tfr_rate
    #         # mean = np.mean(tfr_rate, axis=0) / (np.std(tfr_rate, axis=0) / np.sqrt(tfr_rate.shape[0]))
    #         # mean = np.mean(tfr_rate, axis=0)
    #         # vmax = np.max([np.max(np.abs(tval)) for tval in tvalues])
    #         # vmax = 7
    #         vmax = 2
    #         print(tfr_rate.shape)
    #         im = axs[r].pcolormesh(power_epochs.times, power_epochs.freqs, tfr_rate, shading='nearest', cmap='Spectral_r', vmin=-vmax, vmax=vmax)
    #         # im = axs[r].pcolormesh(power_epochs.times, power_epochs.freqs, tfr_rate, shading='nearest', cmap='Spectral_r')
    #         if r == len(rates) - 1:
    #             axs[r].set_xlabel('Time (s)')
    #         axs[r].set_ylabel('Frequency (Hz)')
    #         axs[r].set_yscale('log')
    #         axs[r].axvline(0, ls='--', c='k')
    #         axs[r].set_title(rate)
    #         for note in rate_notes[1:]:
    #             axs[r].axvline(note, ls='--', c='gray')
    #         cbar = plt.colorbar(im)
    #         cbar.ax.set_ylabel('percent relative to baseline', rotation=270)
    # plt.show()


def plot_tfr_individual_trials(data_path):
    plt.switch_backend('TkAgg')

    power_epochs = mne.time_frequency.read_tfrs(data_path)[0]
    power_epochs = power_epochs.apply_baseline(baseline=(-1.5, -0.25), mode='percent')
    power_epochs.crop(-1, 8.5)
    stim_names = power_epochs.metadata['stim_name'].tolist()

    tfr = np.mean(power_epochs.data, axis=1)  # average channels

    for trial in range(tfr.shape[0]):
        print(stim_names[trial])
        rate_notes = np.arange(0, 8.001, 1 / float(stim_names[trial][14:17]))
        # vmax = np.max(np.abs(tfr[trial, :, :]))
        vmax = 10
        plt.pcolormesh(power_epochs.times, power_epochs.freqs, tfr[trial, :, :], shading='nearest', cmap='Spectral_r', vmin=-vmax, vmax=vmax)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.gca().axvline(0, ls='--', c='k')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('average', rotation=270)
        for note in rate_notes[1:]:
            plt.gca().axvline(note, ls='--', c='gray')
        plt.show()


def plot_tfr_one_song_all_subjects(data_path):
    plt.switch_backend('TkAgg')

    path = Path(data_path)
    subject_list = sorted(list(path.glob('*-tfr.h5')))
    print(subject_list)

    power_epochs = mne.time_frequency.read_tfrs(subject_list[0])[0]
    stim_names = power_epochs.metadata['stim_name'].tolist()

    for stim_name in stim_names[6:]:
    # for stim_name in stim_names:
        print(stim_name)

        tfr = []
        std_tfr = []
        phase = []
        for i, epochs_path in tqdm(enumerate(subject_list[:])):
            subject_name = Path(epochs_path).name[:6]
            tfr_epochs = mne.time_frequency.read_tfrs(epochs_path)[0]
            power_epochs = tfr_epochs.copy()
            phase_epochs = tfr_epochs.copy()
            power_epochs.data = np.abs(power_epochs.data**2)
            phase_epochs.data = np.angle(phase_epochs.data)
            # phase_epochs.data = np.real(phase_epochs.data)
            power_epochs = power_epochs.apply_baseline(baseline=(-1.5, -0.25), mode='percent')
            power_epochs.crop(-1, 8.5)
            phase_epochs.crop(-1, 8.5)
            stim_names_subject = power_epochs.metadata['stim_name'].tolist()
            epochs_to_keep = [e for e in range(len(stim_names_subject)) if stim_name[14:] in stim_names_subject[e]]  # keep the 2 reps of the song
            if len(epochs_to_keep) > 0:
                tfr.append(np.mean(power_epochs.data[epochs_to_keep, :, :, :], axis=(0, 1)))  # average epochs and channels
                # tfr.append(np.mean(power_epochs.data[epochs_to_keep, 0, :, :], axis=0))  # average epochs and channels
                std_tfr.append(np.mean(power_epochs.data, axis=(0, 1)) / np.std(np.mean(power_epochs.data, axis=1), axis=0))  # normalized
                phase.append(np.mean(phase_epochs.data[epochs_to_keep, :, :, :], axis=(0, 1)))
                # phase.append(np.mean(phase_epochs.data[epochs_to_keep, 0, :, :], axis=0))
        # tfr = np.concatenate(tfr, axis=0)  # concatenate channels
        tfr = np.array(tfr)
        print(tfr.shape)
        tfr = np.mean(tfr, axis=0)
        phase = np.array(phase)
        phase = np.mean(phase, axis=0)
        print(phase.shape)
        # tfr = np.mean(tfr, axis=0) / (np.std(tfr, axis=0) / np.sqrt(tfr.shape[0]))
        std_tfr = np.array(std_tfr)
        rate_notes = np.arange(0, 8.001, 1 / float(stim_name[14:17]))
        # vmax = np.max(np.abs(tfr))
        # vmax = np.max(np.abs(tfr)) * 0.9
        vmax = 3
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1])
        ax0 = fig.add_subplot(gs[0, 0])
        mesh = ax0.pcolormesh(power_epochs.times, power_epochs.freqs, tfr, shading='auto', cmap='Spectral_r', vmin=-vmax, vmax=vmax)
        ax0.set_xlabel('Time (s)')
        ax0.set_ylabel('Frequency (Hz)')
        ax0.set_yscale('log')
        ax0.axvline(0, ls='--', c='k')
        ax0.axhline(float(stim_name[14:17]), c='blue', lw=0.5)

        cbar_ax = fig.add_subplot(gs[0, 1])
        cbar = plt.colorbar(mesh, ax=cbar_ax)
        cbar.ax.set_ylabel('average', rotation=270)
        # Remove the axis box and ticks around the colorbar
        cbar_ax.tick_params(left=False, labelleft=False, right=False, labelright=False,
                            top=False, labeltop=False, bottom=False, labelbottom=False)
        for spine in cbar_ax.spines.values():
            spine.set_visible(False)

        for note in rate_notes[1:]:
            ax0.axvline(note, ls='--', c='gray')

        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)  # Shared x-axis
        freq = np.argmin(np.abs(phase_epochs.freqs - float(stim_name[14:17])))
        ax1.plot(phase_epochs.times, phase[freq, :])
        ax1.axvline(0, ls='--', c='k')
        for note in rate_notes[1:]:
            ax1.axvline(note, ls='--', c='gray')

        plt.tight_layout()
        plt.show()


def plot_tfr_contrasts(data_path, behavior_path):
    plt.switch_backend('TkAgg')

    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*-tfr.h5')))

    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']
    tfr_contrasts = [[] for c in range(len(conditions))]
    power_contrasts = [[] for c in range(len(conditions))]

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
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

        power_epochs = mne.time_frequency.read_tfrs(epochs_path)[0]
        power_epochs = power_epochs.apply_baseline(baseline=(-1.5, -0.25), mode='zscore')
        power_epochs.crop(-1, 8.5)
        power_epochs_cropped = power_epochs.copy().crop(0, 8)
        stim_names = power_epochs.metadata['stim_name'].tolist()

        for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
            correct = [e for e in range(len(stim_names)) if stim_names[e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(stim_names)) if stim_names[e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                # tfr_contrasts[cond].append(np.mean(np.mean(power_epochs.data[correct, :, :, :], axis=0), axis=0))
                tfr_contrasts[cond].append(np.mean(np.mean(power_epochs.data[correct, :, :, :], axis=0) - np.mean(power_epochs.data[incorrect, :, :, :], axis=0), axis=0))  # average epochs before contrasting and average channels after
                power_contrasts[cond].append(np.mean(np.mean(power_epochs_cropped.data[correct, :, :, :], axis=(0, 3)) - np.mean(power_epochs_cropped.data[incorrect, :, :, :], axis=(0, 3)), axis=0))
                # power_contrasts[cond].append(np.mean(np.mean(power_epochs_cropped.data[correct, :, :, :], axis=(0, 3)), axis=0))

    power_contrasts = [np.array(power_contrasts[c]) for c in range(len(power_contrasts))]
    fig, axs = plt.subplots(1, 2, sharey=False, sharex=True)
    tvalues = [np.mean(contrast, axis=0) / (np.std(contrast, axis=0) / np.sqrt(contrast.shape[0])) for contrast in power_contrasts]
    mean_values = [np.mean(contrast, axis=0) for contrast in power_contrasts]
    sem_values = [np.std(contrast, axis=0) / np.sqrt(contrast.shape[0]) for contrast in power_contrasts]
    # vmax = 5
    for i, (cond, mean, sem) in enumerate(zip(conditions, mean_values, sem_values)):
        print(mean.shape)
        axs[0].plot(power_epochs.freqs, mean, label=cond)
        axs[0].fill_between(power_epochs.freqs, mean-sem, mean+sem, alpha=0.5, lw=0)
    # axs[0].set_xscale('log')
    axs[0].axhline(0, ls='--', c='k')
    axs[0].set_ylabel('Power')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].legend()
    for i, (cond, tval) in enumerate(zip(conditions, tvalues)):
        print(tval.shape)
        axs[1].plot(power_epochs.freqs, tval, label=cond)
    axs[1].axhline(0, ls='--', c='k')
    axs[1].set_ylabel('t value')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].legend()

    # tfr = np.concatenate(tfr, axis=0)  # concatenate channels
    tfr_contrasts = [np.array(tfr_contrasts[c]) for c in range(len(tfr_contrasts))]
    fig, axs = plt.subplots(len(conditions), 1, sharey=True)
    tvalues = [np.mean(contrast, axis=0) / (np.std(contrast, axis=0) / np.sqrt(contrast.shape[0])) for contrast in tfr_contrasts]
    # vmax = np.max([np.max(np.abs(tval)) for tval in tvalues])
    vmax = 5
    for i, (cond, tval) in enumerate(zip(conditions, tvalues)):
        print(tval.shape)
        im = axs[i].pcolormesh(power_epochs.times, power_epochs.freqs, tval, shading='nearest', cmap='bwr', vmin=-vmax, vmax=vmax)
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Frequency (Hz)')
        axs[i].axvline(0, ls='--', c='k')
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('tvalue', rotation=270)
    plt.show()


def plot_tfr_contrasts_by_rate(data_path, behavior_path):
    plt.switch_backend('TkAgg')

    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_list = sorted(list(path.glob('*-tfr.h5')))

    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']
    conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']
    tfr_contrasts = [[[] for c in range(len(conditions))] for r in range(len(rates))]
    power_contrasts = [[[] for c in range(len(conditions))] for r in range(len(rates))]

    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = Path(epochs_path).name[:6]
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

        power_epochs = mne.time_frequency.read_tfrs(epochs_path)[0]
        power_epochs = power_epochs.apply_baseline(baseline=(-1.5, -0.25), mode='zscore')
        power_epochs.crop(-1, 8.5)
        power_epochs_cropped = power_epochs.copy().crop(0, 8)
        stim_names = power_epochs.metadata['stim_name'].tolist()

        for r, rate in enumerate(rates):
            for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
                correct = [e for e in range(len(stim_names)) if (stim_names[e] in correct_incorrect[0]) and (rate in stim_names[e])]
                incorrect = [e for e in range(len(stim_names)) if (stim_names[e] in correct_incorrect[1]) and (rate in stim_names[e])]
                if (len(correct) > 0) & (len(incorrect) > 0):
                    # tfr_contrasts[r][cond].append(np.mean(np.mean(power_epochs.data[correct+incorrect, :, :, :], axis=0), axis=0))
                    tfr_contrasts[r][cond].append(np.mean(np.mean(power_epochs.data[correct, :, :, :], axis=0) - np.mean(power_epochs.data[incorrect, :, :, :], axis=0), axis=0))  # average epochs before contrasting and average channels after
                    power_contrasts[r][cond].append(np.mean(np.mean(power_epochs_cropped.data[correct, :, :, :], axis=(0, 3)) - np.mean(power_epochs_cropped.data[incorrect, :, :, :], axis=(0, 3)), axis=0))
                    # power_contrasts[r][cond].append(np.mean(np.mean(power_epochs_cropped.data[correct+incorrect, :, :, :], axis=(0, 3)), axis=0))

    for r, rate in enumerate(rates):
        power_contrasts_rate = [np.array(power_contrasts[r][c]) for c in range(len(power_contrasts[r]))]
        fig, axs = plt.subplots(1, 2, sharey=False, sharex=True)
        tvalues = [np.mean(contrast, axis=0) / (np.std(contrast, axis=0) / np.sqrt(contrast.shape[0])) for contrast in power_contrasts_rate]
        mean_values = [np.mean(contrast, axis=0) for contrast in power_contrasts_rate]
        sem_values = [np.std(contrast, axis=0) / np.sqrt(contrast.shape[0]) for contrast in power_contrasts_rate]
        # vmax = 5
        for i, (cond, mean, sem) in enumerate(zip(conditions, mean_values, sem_values)):
            print(mean.shape)
            axs[0].plot(power_epochs.freqs, mean, label=cond)
            axs[0].fill_between(power_epochs.freqs, mean-sem, mean+sem, alpha=0.5, lw=0)
        # axs[0].set_xscale('log')
        axs[0].axhline(0, ls='--', c='k')
        axs[0].set_ylabel('Power')
        axs[0].set_xlabel('Frequency (Hz)')
        axs[0].set_title(rate)
        axs[0].legend()
        for i, (cond, tval) in enumerate(zip(conditions, tvalues)):
            print(tval.shape)
            axs[1].plot(power_epochs.freqs, tval, label=cond)
        axs[1].axhline(0, ls='--', c='k')
        axs[1].set_ylabel('t value')
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].legend()
    # plt.show()

    for r, rate in enumerate(rates):
        rate_notes = np.arange(0, 8.001, 1/float(rate[:3]))
        print(rate_notes)
        # tfr = np.concatenate(tfr, axis=0)  # concatenate channels
        tfr_contrasts_rate = [np.array(tfr_contrasts[r][c]) for c in range(len(tfr_contrasts[r]))]
        fig, axs = plt.subplots(len(conditions), 1, sharey=True)
        tvalues = [np.mean(contrast, axis=0) / (np.std(contrast, axis=0) / np.sqrt(contrast.shape[0])) for contrast in tfr_contrasts_rate]
        # vmax = np.max([np.max(np.abs(tval)) for tval in tvalues])
        vmax = 5
        for i, (cond, tval) in enumerate(zip(conditions, tvalues)):
            print(tval.shape)
            im = axs[i].pcolormesh(power_epochs.times, power_epochs.freqs, tval, shading='nearest', cmap='bwr', vmin=-vmax, vmax=vmax)
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel('Frequency (Hz)')
            axs[i].axvline(0, ls='--', c='k')
            for note in rate_notes[1:]:
                axs[i].axvline(note, ls='--', c='gray')
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('tvalue', rotation=270)
        fig.suptitle(rate)
    # plt.show()

    # plot note evoked tfr
    for r, rate in enumerate(rates):
        rate_duration = 1/float(rate[:3])
        rate_notes = np.arange(0, 8.001, rate_duration)
        print(rate_notes)
        # tfr = np.concatenate(tfr, axis=0)  # concatenate channels
        tfr_contrasts_rate = [np.array(tfr_contrasts[r][c]) for c in range(len(tfr_contrasts[r]))]
        fig, axs = plt.subplots(len(conditions), 1, sharey=True)
        print(tfr_contrasts_rate[0].shape)
        tvalues = [np.mean(contrast, axis=0) / (np.std(contrast, axis=0) / np.sqrt(contrast.shape[0])) for contrast in tfr_contrasts_rate]
        # vmax = np.max([np.max(np.abs(tval)) for tval in tvalues])
        vmax = 5
        # Determine the fixed window size based on the first note
        fixed_window_size = np.argmin(np.abs(power_epochs.times - (rate_notes[0] + rate_duration))) - np.argmin(np.abs(power_epochs.times - rate_notes[0]))
        for i, (cond, contrast) in enumerate(zip(conditions, tfr_contrasts_rate)):
            evoked_tval = []
            for note in rate_notes[:-1]:
                # Find the time indices within the current note window
                start_idx = np.argmin(np.abs(power_epochs.times - note))
                end_idx = start_idx + fixed_window_size
                evoked_tval.append(contrast[:, :, start_idx:end_idx])
            evoked_tval = np.mean(evoked_tval, axis=0)  # average windows
            evoked_tval = np.mean(evoked_tval, axis=0) / (np.std(evoked_tval, axis=0) / np.sqrt(evoked_tval.shape[0]))
            print(evoked_tval.shape)
            times = np.linspace(0, rate_notes[1], evoked_tval.shape[1])
            print(times)
            # im = axs[i].pcolormesh(times, power_epochs.freqs, evoked_tval, shading='nearest', cmap='bwr', vmin=-vmax, vmax=vmax)
            im = axs[i].pcolormesh(times, power_epochs.freqs, evoked_tval, shading='nearest', cmap='bwr')
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel('Frequency (Hz)')
            axs[i].axvline(0, ls='--', c='k')
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('tvalue', rotation=270)
        fig.suptitle(rate)
    plt.show()


if __name__ == "__main__":

    '''plot_time_freq_and_spectrum'''
    # # tf_folder = '/home/mathieupvc/These/data/meg_maestro/time_frequency_analysis/time_frequency_freqby4cycles_10chan'
    # tf_folder = '/home/mathieupvc/These/data/meg_maestro/time_frequency_analysis/time_frequency_freqcycles_10chan'
    # # tf_folder = '/home/mathieupvc/These/data/meg_maestro/time_frequency_analysis/time_frequency_10Hz'
    # # condition = ['rep1', 'rep2', 'rep3']
    # # condition = ['piano0', 'piano2']
    # # condition = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']
    # # condition = ['correct', 'incorrect']
    # # condition = ['new', 'old']
    # condition = None
    # # condition = 'piano0'
    # # condition = '2.5Hz'
    # # condition = 'trial008'
    # # condition = '4.5Hz_clip0_'
    # # condition = '4.5Hz_clip10'
    # # condition = '2.5Hz_clip14'
    # spectrum_window = (0, 8)
    # freq_zscore = False
    # plot_time_freq_and_spectrum(tf_folder, condition, spectrum_window, freq_zscore=freq_zscore)

    '''contrast_conditions'''
    # tf_folder = '/home/mathieupvc/These/data/meg_maestro/time_frequency_analysis/time_frequency_freqcycles_10chan'
    # # tf_folder = '/home/mathieupvc/These/data/meg_maestro/time_frequency_analysis/time_frequency_10Hz'
    # # conditions = ['rep1', 'rep2', 'rep3']
    # # conditions = ['piano0', 'piano2']
    # # conditions = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']
    # conditions = ['correct', 'incorrect']
    # # conditions = ['new', 'old']
    # spectrum_window = (0, 8)
    # # correct_incorrect_path = '/home/mathieupvc/These/data/meg_maestro/correct_incorrect_lists.pkl'
    # # correct_incorrect_path = '/home/mathieupvc/These/data/meg_maestro/correct_incorrect_rep1_lists.pkl'
    # correct_incorrect_path = '/home/mathieupvc/These/data/meg_maestro/correct_incorrect_all_reps_lists.pkl'
    # new_old_path = '/home/mathieupvc/These/data/meg_maestro/new_old_lists.pkl'
    # # new_old_path = '/home/mathieupvc/These/data/meg_maestro/new_old_lists_rep3.pkl'
    # freq_zscore = False
    # contrast_conditions(tf_folder, conditions, spectrum_window, correct_incorrect_path, new_old_path, freq_zscore=freq_zscore)

    '''spectrum_double_conditions'''
    # tf_folder = '/home/mathieupvc/These/data/meg_maestro/time_frequency_analysis/time_frequency_freqcycles_10chan'
    # # tf_folder = '/home/mathieupvc/These/data/meg_maestro/time_frequency_analysis/time_frequency_10Hz'
    # # conditions2 = ['rep1', 'rep2', 'rep3']
    # # conditions2 = ['piano0', 'piano2']
    # conditions1 = ['correct', 'incorrect']
    # # conditions1 = ['new', 'old']
    # conditions2 = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']
    # spectrum_window = (0, 8)
    # # correct_incorrect_path = '/home/mathieupvc/These/data/meg_maestro/correct_incorrect_lists.pkl'
    # # correct_incorrect_path = '/home/mathieupvc/These/data/meg_maestro/correct_incorrect_rep1_lists.pkl'
    # correct_incorrect_path = '/home/mathieupvc/These/data/meg_maestro/correct_incorrect_all_reps_lists.pkl'
    # new_old_path = '/home/mathieupvc/These/data/meg_maestro/new_old_lists.pkl'
    # # new_old_path = '/home/mathieupvc/These/data/meg_maestro/new_old_lists_rep3.pkl'
    # freq_zscore = False
    # only_correct = False
    # spectrum_double_conditions(tf_folder, conditions1, conditions2, spectrum_window, correct_incorrect_path, new_old_path, freq_zscore=freq_zscore, only_correct=only_correct)

    '''extract_trial_power'''
    # extract_trial_power(tf_folder='/home/mathieupvc/These/data/meg_maestro/time_frequency_analysis/time_frequency_freqcycles_10chan',
    #                     save_path='/home/mathieupvc/These/data/meg_maestro')

    """seeg_time_freq_epochs"""
    data_path = '/home/mathieu/These/data/meg_maestro/seeg_trials'
    save_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_hippocampus'
    # save_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_right_hippocampus'
    # save_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_stg'
    # save_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_transversetemporal'
    # save_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_prefrontal'
    regions = ['Left-Hippocampus']
    # regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal']
    # regions = ['ctx-rh-inferiorparietal']
    # regions = ['Left-Amygdala']
    # regions = ['ctx-rh-transversetemporal', 'ctx-lh-transversetemporal']
    # regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis',
    #            'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal']
    # regions = ['Right-Hippocampus']
    # regions = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal']
    # seeg_time_freq_epochs(data_path, save_path, regions, first_parcel_label='advanced', n_jobs=-1)

    """plot_tfr"""
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_hippocampus'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_right_hippocampus'
    tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_stg'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_transversetemporal'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_prefrontal'
    # plot_tfr(tfr_path)

    """plot_tfr_by_rate"""
    tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_transversetemporal'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_hippocampus'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_right_hippocampus'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_stg'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_prefrontal'
    # plot_tfr_by_rate(tfr_path)

    """plot_tfr_individual_trials"""
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_hippocampus/Gra_Lu-tfr.h5'
    tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_prefrontal/Bou_Al-tfr.h5'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_transversetemporal/Gra_Lu-tfr.h5'
    # plot_tfr_individual_trials(tfr_path)

    """plot_tfr_one_song_all_subjects"""
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_transversetemporal'
    tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_hippocampus'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_right_hippocampus'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_stg'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_prefrontal'
    plot_tfr_one_song_all_subjects(tfr_path)

    """plot_tfr_contrasts"""
    tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_transversetemporal'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_hippocampus'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_inferiorparietal'
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # plot_tfr_contrasts(tfr_path, behavior_path)

    """plot_tfr_contrasts_by_rate"""
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_transversetemporal'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_inferiorparietal'
    tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_left_hippocampus'
    # tfr_path = '/home/mathieu/These/data/meg_maestro/time_frequency_analysis/seeg/tf_epochs_right_hippocampus'
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # plot_tfr_contrasts_by_rate(tfr_path, behavior_path)


