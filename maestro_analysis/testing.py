from pathlib import Path
import os

from maestro_analysis.utils import epoching_maestro, merge_grad_data
from maestro_analysis.trf_analysis import list_from_metadata, condition_list_from_metadata, find_surp_entr_files
from maestro_analysis.stats import test_channels_significance
from maestro_analysis.signal_functions import OG_envelope, subsample, butter_lowpass_filter

from scipy.io.wavfile import read, write
from scipy.signal import resample_poly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import mne
from mne.decoding import ReceptiveField


def check_wav_order(data_path, wav_folder="/home/mathieu/Data/meg_maestro/exp_stim_meg"):
    '''
    Check that wav files are in the same order as the epochs
    '''

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])
    wav_path = (sorted(list(Path(wav_folder).glob('**/*.wav'))))

    for subject_path in subject_folders[2:]:
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        epochs = epoching_maestro(raw_path, wav_folder)
        epochs.pick(['MISC003'])
        epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]
        epochs.crop(tmin=0, tmax=8)
        #epochs.resample(100, n_jobs=-1)
        meta = epochs.metadata
        list_to_keep = list_from_metadata(meta)
        print(list_to_keep)
        print(len(meta))
        print(epochs.metadata[:40])
        epochs = epochs.get_data()
        print(epochs.shape)

        for i, wav_file in enumerate(np.array(wav_path)[list_to_keep]):
            print(wav_file)
            wav = read(wav_file)
            # set the duration to 8 seconds
            if wav[1].shape[0] > wav[0]*8:
                new_wav = wav[1][:wav[0]*8]
            elif wav[1].shape[0] < wav[0]*8:
                new_wav = np.zeros(wav[0]*8)
                new_wav[:wav[1].shape[0]] = wav[1]

            write("/home/mathieu/Data/meg_maestro/test/" + wav_file.name, wav[0], new_wav)
            write("/home/mathieu/Data/meg_maestro/test/audio_channel3_" + str(i) + ".wav", 1000, epochs[i,0,:])

            #new_wav = resample_poly(new_wav, 100, wav[0])
            
            print(meta.iloc[i])
            print(new_wav.shape)
            fig, axes = plt.subplots(nrows=2, ncols=1)
            axes[0].plot(new_wav)
            axes[1].plot(epochs[i,0,:])
            plt.show()

def check_channel_names(data_path):
    '''check that all subjects have the same channels'''

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    raw_path = list(sorted(subject_folders)[0].glob('**/*raw_sss.fif'))[0]
    raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
    raw.pick(['meg'])
    ch_names = raw.info['ch_names']
    print(ch_names)

    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
        raw.pick(['meg'])
        if raw.info['ch_names'] != ch_names:
            print(raw.info['ch_names'])

    return



def plot_trf_topomaps_one_subject(raw_path, scores_folder, channel_type='grad', stat_test='permut', save_format='png'):
    '''
    scores_path: folder containing epochs scores and epochs scores env
    '''

    scores = np.load(os.path.join(scores_folder, "epochs_channels_score.npy"))
    scores_env = np.load(os.path.join(scores_folder, "epochs_channels_score_env.npy"))

    # define the statistical threshold (alpha)
    if stat_test == 'permut_cluster':
        alpha = 0.05
    else:
        alpha = 0.025  # in case of a 1_tailed t_test, we don't want to increase the sensitivity

    raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
    raw.pick(['meg'])

    channels_to_keep = mne.channel_indices_by_type(raw.info)[channel_type]
    score = np.mean(scores[:, channels_to_keep], axis=0)
    score_env = np.mean(scores_env[:, channels_to_keep], axis=0)
    if channel_type == 'grad':
        score = merge_grad_data(score)
        score_env = merge_grad_data(score_env)
    mag_info = raw.copy().pick(['mag']).info  # Take Info of the magnetometers for channels positions
    raw.pick([channel_type])

    # Find channels with significant score
    p_am = test_channels_significance(scores[:, channels_to_keep], info=raw.info, test=stat_test, channel_type=channel_type)  # Permutation across epochs
    p_a = test_channels_significance(scores_env[:, channels_to_keep], info=raw.info, test=stat_test, channel_type=channel_type)
    p_diff = test_channels_significance(scores[:, channels_to_keep] - scores_env[:, channels_to_keep], info=raw.info, test=stat_test, channel_type=channel_type)

    p_new = [p_am, p_a, p_diff]

    # plot topomap TRF AM
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 12), dpi=300)
    vmax = max(np.max(np.abs(score)), np.max(np.abs(score_env)))
    im, cn = mne.viz.plot_topomap(score, mag_info, show=False, axes=axes[0, 0], ch_type='mag', mask=p_new[0] <= alpha, vmin=0, vmax=vmax)
    axes[0, 0].set_title("AM")
    clb = fig.colorbar(im, cax=axes[0, 1])
    clb.ax.set_title("Prediction Correlation (r)", fontsize=15) # title on top of colorbar
    axes[0, 1].set_box_aspect(5)
    axes[0, 1].set_anchor('W')
    axes[0, 0].set_anchor('E')
    im, cn = mne.viz.plot_topomap(score_env, mag_info, show=False, axes=axes[1, 0], ch_type='mag', mask=p_new[1] <= alpha, vmin=0, vmax=vmax)
    axes[1, 0].set_title("A")
    clb = fig.colorbar(im, cax=axes[1, 1])
    axes[1, 1].set_box_aspect(5)
    axes[1, 1].set_anchor('W')
    axes[1, 0].set_anchor('E')
    im, cn = mne.viz.plot_topomap(score - score_env, mag_info, show=False, axes=axes[2, 0], ch_type='mag', mask=p_new[2] <= alpha)
    axes[2, 0].set_title("AM - A")
    clb = fig.colorbar(im, cax=axes[2, 1])
    axes[2, 1].set_box_aspect(5)
    axes[2, 1].set_anchor('W')
    axes[2, 0].set_anchor('E')

    plt.savefig(os.path.join(scores_folder, 'trf_topomaps_' + channel_type + '.' + save_format))
    plt.close()

    return



def plot_trf_kernels_best_channels_one_subject(raw_path, pickle_folder, features=None, channel_type='grad', acoust_feat=False, save_format='png', nb_channels=1):

    if features is None:
        features = ['Envelope', 'Envelope derivative', 'Surprise', 'Entropy']

    raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
    raw.pick(['meg'])

    scores = np.load(os.path.join(pickle_folder, "epochs_channels_score.npy"))
    scores_env = np.load(os.path.join(pickle_folder, "epochs_channels_score_env.npy"))
    score = np.mean(scores, axis=0)[mne.channel_indices_by_type(raw.info)[channel_type]]  # Average over epochs and select channel type
    score_env = np.mean(scores_env, axis=0)[mne.channel_indices_by_type(raw.info)[channel_type]]
    best_chans = np.flip(np.argsort(score))[:nb_channels]
    best_chans_env = np.flip(np.argsort(score_env))[:nb_channels]

    file = open(os.path.join(pickle_folder, "rf.pickle"), 'rb')
    rf = pickle.load(file)
    file.close()

    fig, axes = plt.subplots(nrows=rf.coef_.shape[1], ncols=1, sharex=True, sharey=False, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        axes[j].plot(rf.delays_ / rf.sfreq, np.mean(rf.coef_[np.array(mne.channel_indices_by_type(raw.info)[channel_type])[best_chans], j, :], axis=0).T)
        axes[j].axvline(0, linestyle='--', color='k')
        axes[j].set_title(features[j])
    plt.xlabel("Time (s)")
    fig.savefig(os.path.join(pickle_folder, 'trf_kernels_' + channel_type + '.' + save_format))
    plt.close()

    if acoust_feat:
        file = open(os.path.join(pickle_folder, "rf_env.pickle"), 'rb')
        rf_env = pickle.load(file)
        file.close()

        fig, axes = plt.subplots(nrows=rf_env.coef_.shape[1], ncols=1, sharex=True, sharey=False, figsize=(20, 12), dpi=100)
        if rf_env.coef_.shape[1] == 1:
            axes.plot(rf_env.delays_ / rf_env.sfreq, rf_env.coef_[np.array(mne.channel_indices_by_type(raw.info)[channel_type])[best_chans_env], 0, :].T)
            axes.axvline(0, linestyle='--', color='k')
            axes.set_title(features[0])
        else:
            for j in range(rf_env.coef_.shape[1]):
                axes[j].plot(rf_env.delays_ / rf_env.sfreq, rf_env.coef_[np.array(mne.channel_indices_by_type(raw.info)[channel_type])[best_chans_env], j, :].T)
                axes[j].axvline(0, linestyle='--', color='k')
                axes[j].set_title(features[j])
        plt.xlabel("Time (s)")
        fig.savefig(os.path.join(pickle_folder, 'trf_env_kernels_' + channel_type + '.' + save_format))
        plt.close()


    return


def create_trf_output_from_epochs_minimal_processing(raw_path, wav_folder, save_path=None, start_sample=100):
    '''
    For 1 subject only. From epochs, output a numpy array containing MEG data in the right format (sample rate,
    duration, shape).
    '''

    epochs = epoching_maestro(raw_path, wav_folder) # Load raw and create epochs
    epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]  # Select listening conditions only
    epochs.pick('mag')  # keep magnetometers and gradiometers only
    epochs.crop(tmin=0, tmax=8)  # keep 8s only
    epochs.resample(100, n_jobs=-1)
    trf_output = epochs.get_data()
    metadata = epochs.metadata
    del epochs

    # Remove onset activity
    trf_output = trf_output[:, :, start_sample:]

    trf_output = np.moveaxis(trf_output, 2, 0)

    if save_path is not None:
        np.save(save_path, trf_output)

    return trf_output, metadata



def run_one_block_trf(wav_folder, stim_indices, raw_path, save_path=None, sample_range=None, env_filter=15):
    """
    Run TRF on 1 subject with only one set of features.
        condition: list of str
        output: str or numpy array, if str, output is a path to a npy file.
    """

    if sample_range is None:
        sample_range = [1000, 8000]

    wav_folder = Path(wav_folder)
    wav_list = sorted(list(wav_folder.glob('**/*.wav')))
    n_epochs = stim_indices[1] - stim_indices[0]
    envelope_array = np.zeros((8000, n_epochs, 2))  # Contains envelope and its derivative
    for i in range(n_epochs):
        wav_file = wav_list[i + stim_indices[0]]
        # Audio input envelope
        print(wav_file)
        wav = read(wav_file)
        # set the duration to 8 seconds
        if wav[1].shape[0] > wav[0] * 8:
            new_wav = wav[1][:wav[0] * 8]
        elif wav[1].shape[0] < wav[0] * 8:
            new_wav = np.zeros(wav[0] * 8)
            new_wav[:wav[1].shape[0]] = wav[1]
        else:
            new_wav = wav[1]

        envelope = OG_envelope(new_wav, wav[0])  # Extract sound envelope
        if env_filter is not None:
            envelope = butter_lowpass_filter(envelope, env_filter, wav[0])  # lowpass filter the envelope
        env_derivative = np.gradient(envelope)
        envelope = resample_poly(envelope, 1000, wav[0])
        env_derivative = resample_poly(env_derivative, 1000, wav[0])
        env_derivative = np.maximum(0, env_derivative)  # Half-wave rectification of the derivative

        envelope_array[:, i, 0] = envelope  # add trial in the global array
        envelope_array[:, i, 1] = env_derivative

    epochs = epoching_maestro(raw_path, wav_folder, allow_maxshield=True)  # Load raw and create epochs
    epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]  # Select listening conditions only
    epochs.pick('mag')  # keep magnetometers only
    epochs.crop(tmin=0, tmax=8)  # keep 8s only
    # epochs.resample(100, n_jobs=-1)
    trf_output = epochs.get_data()
    metadata = epochs.metadata
    del epochs

    # Remove onset activity
    trf_output = trf_output[:, :, sample_range[0]:sample_range[1]]
    # Z-scoring
    # if zscore:
    #     scaler = Scaler(scalings='mean')
    #     trf_output = scaler.fit_transform(trf_output)

    trf_output = np.moveaxis(trf_output, 2, 0)

    X = envelope_array[sample_range[0]:sample_range[1], :, :]
    # list_to_keep = list_from_metadata(metadata)
    # X = X[:, list_to_keep, :]
    X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))

    Y = trf_output

    print(X.shape)
    print(Y.shape)

    # alphas = np.logspace(-1, 7, 9)
    # alphas = np.logspace(2, 9, 8)
    alphas = np.logspace(1, 7, 7)
    coefs = []

    for ii, alpha in enumerate(alphas):
        print('alpha:', alpha)

        rf = ReceptiveField(-0.2, 0.5, 1000, estimator=alpha, scoring='corrcoef', n_jobs=-1)
        rf.fit(X, Y)
        coefs.append(rf.coef_)

    fig, axes = plt.subplots(nrows=len(coefs), ncols=2, figsize=(20, 12), dpi=300)
    for i, coef in enumerate(coefs):
        axes[i, 0].plot(rf.delays_/rf.sfreq, coef[:, 0, :].T)
        axes[i, 1].plot(rf.delays_/rf.sfreq, coef[:, 1, :].T)

    split = raw_path.split('/')
    plt.savefig(os.path.join(save_path, split[-3] + '_' + split[-1] + '_' + str(sample_range[0]) + '-' + str(sample_range[1]) + '.png'))
    plt.close()

    return


def hist_kernel_time(data_path, trf_folder, features=None, time=0):
    '''
    ['Envelope', 'Envelope derivative', 'Surprise on', 'Surprise off', 'Surprise velocity', 'Surprise shift', 'Entropy on', 'Entropy off', 'Entropy velocity', 'Entropy shift']
    '''

    if features is None:
        features = ['Envelope', 'Envelope derivative', 'Surprise', 'Entropy']

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    coef_list = []

    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)

        file = open(os.path.join(trf_folder, subject_path.name, "rf.pickle"), 'rb')
        rf = pickle.load(file)
        file.close()

        coef_list.append(rf.coef_)

    coef_list = np.array(coef_list)
    coefs_subject_average = np.mean(coef_list, axis=0)
    coefs_channel_average = np.sqrt(np.mean(coef_list**2, axis=1))

    sample = int(round(time * rf.sfreq - rf.delays_[0]))

    fig, axes = plt.subplots(nrows=rf.coef_.shape[1], ncols=1, sharex=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        axes[j].hist(coef_list[:, :, j, sample].flat, bins=100)
        axes[j].set_title(features[j])
    fig.suptitle('Subjects and channels', fontsize=16)

    fig, axes = plt.subplots(nrows=rf.coef_.shape[1], ncols=1, sharex=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        axes[j].hist(coefs_subject_average[:, j, sample].flat, bins=100)
        axes[j].set_title(features[j])
    fig.suptitle('Channels', fontsize=16)

    fig, axes = plt.subplots(nrows=rf.coef_.shape[1], ncols=1, sharex=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        axes[j].hist(coefs_channel_average[:, j, sample].flat, bins=100)
        axes[j].set_title(features[j])
        sort_subject = np.argsort(coefs_channel_average[:, j, sample])
        print(sort_subject)
    fig.suptitle('Subjects', fontsize=16)

    plt.switch_backend('QtAgg')
    plt.show()

    return


def meg_data_statistics(raw_path, wav_folder, channel_type='grad', filt_range=None):
    '''
    For 1 subject only. From epochs, output a numpy array containing MEG data in the right format (sample rate,
    duration, shape).
    '''

    if filt_range is None:
        filt_range = [1, 8]

    epochs = epoching_maestro(raw_path, wav_folder) # Load raw and create epochs
    epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]  # Select listening conditions only
    epochs.pick(channel_type)  # keep magnetometers and gradiometers only
    epochs.crop(tmin=0, tmax=8)  # keep 8s only
    epochs.filter(filt_range[0], filt_range[1], n_jobs=-1)  # Test with lowpass filter
    epochs.resample(100, n_jobs=-1)
    trf_output = epochs.get_data()
    metadata = epochs.metadata
    del epochs

    trf_output = np.moveaxis(trf_output, 2, 0)

    mean = np.mean(trf_output)
    std = np.std(trf_output)

    return mean, std


def meg_stat_all_subjects(data_path, channel_type='grad', wav_folder="/home/mathieu/Data/meg_maestro/exp_stim_meg", save_format='png'):
    '''
    Computes 500ms ERF and power spectrum on channels of the given type, averaged over all listening epochs.
        data_path: str, path to a folder containing all subjects preprocessed epochs
        channel_types: list, channel types to plot
    '''

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    mean_list = []
    std_list = []
    for subject_path in subject_folders:
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        mean, std = meg_data_statistics(raw_path, wav_folder, channel_type=channel_type, filt_range=None)
        mean_list.append(mean)
        std_list.append(std)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(20, 12), dpi=100)
    axes[0].hist(mean_list, bins=100)
    axes[0].set_title('mean')
    axes[1].hist(std_list, bins=100)
    axes[1].set_title('std')

    fig.savefig(os.path.join(data_path, 'stats_' + channel_type + '.' + save_format))



def preprocessing_concatenate_blocks_only(data_path, destination_path):

    # Loading
    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    for subject_path in subject_folders:
        print(subject_path)
        subject_folder = list(subject_path.glob('**/*run01.fif'))[0].parent
        if Path(subject_folder).parts[-2] == 'maestro25_qlmdne':  # Subject 25 has two run06 (a and b) that need to be concatenated
            sample_data_raw_file_list = [os.path.join(subject_folder, 'run' + str(i).zfill(2) + '.fif') for i in
                                         range(1, 6)] + [os.path.join(subject_folder, 'run06a.fif'),
                                         os.path.join(subject_folder, 'run06b.fif')] + [os.path.join(subject_folder, 'run' + str(i).zfill(2) + '.fif') for i in
                                         range(7, 10)]
        else:
            sample_data_raw_file_list = [os.path.join(subject_folder, 'run' + str(i).zfill(2) + '.fif') for i in range(1, 10)]

        raw_list = [mne.io.read_raw_fif(sample, allow_maxshield=True) for sample in sample_data_raw_file_list]
        for raw in raw_list[1:]:
            raw.info['dev_head_t'] = raw_list[0].info['dev_head_t']  # force concatenation even if head positions are different
        raw = mne.concatenate_raws(raw_list)

        # Saving
        os.makedirs(os.path.join(destination_path, Path(subject_folder).parts[-2]), exist_ok=True)
        raw.save(os.path.join(destination_path, Path(subject_folder).parts[-2], Path(subject_folder).name + 'raw_sss.fif'), overwrite=True)
        print('Saved in', destination_path)


def corr_audio_meg(raw_path, wav_folder, wav_array, save_path=None):
    '''
    '''

    # wav_folder = Path(wav_folder)
    # n_epochs = len(list(wav_folder.glob('**/*.wav')))
    # wav_array = np.zeros((n_epochs, 8000))  # Contains envelope and its derivative
    # for i, wav_file in enumerate(sorted(list(wav_folder.glob('**/*.wav')))):
    #     # Audio input envelope
    #     print(wav_file)
    #     wav = read(wav_file)
    #     # set the duration to 8 seconds
    #     if wav[1].shape[0] > wav[0]*8:
    #         new_wav = wav[1][:wav[0]*8]
    #     elif wav[1].shape[0] < wav[0]*8:
    #         new_wav = np.zeros(wav[0]*8)
    #         new_wav[:wav[1].shape[0]] = wav[1]
    #     else:
    #         new_wav = wav[1]
    #
    #     envelope = OG_envelope(new_wav, wav[0])
    #     envelope = resample_poly(envelope, 1000, wav[0])
    #     wav_array[i, :] = envelope

    epochs = epoching_maestro(raw_path, wav_folder, allow_maxshield=True) # Load raw and create epochs
    epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]  # Select listening conditions only
    epochs.pick('mag')  # keep magnetometers and gradiometers only
    epochs.crop(tmin=0, tmax=8)  # keep 8s only
    # epochs.resample(1000, n_jobs=-1)
    meg_data = epochs.get_data()
    meg_data = meg_data[:, :, :8000]
    del epochs

    n_epochs = wav_array.shape[0]
    corr_array = np.zeros((n_epochs, meg_data.shape[1]))
    for epoch in range(n_epochs):
        print(epoch)
        X = np.concatenate((wav_array[np.newaxis, epoch, :], meg_data[epoch, :, :]), axis=0)
        R = np.corrcoef(X)
        corr_array[epoch, :] = R[0, 1:]

    np.save(os.path.join(save_path, 'corr_array.npy'), corr_array)

    return


def corr_audio_meg_all_subjects(data_path, wav_folder):

    wav_folder = Path(wav_folder)
    n_epochs = len(list(wav_folder.glob('**/*.wav')))
    wav_array = np.zeros((n_epochs, 8000))  # Contains envelope and its derivative
    for i, wav_file in enumerate(sorted(list(wav_folder.glob('**/*.wav')))):
        # Audio input envelope
        print(wav_file)
        wav = read(wav_file)
        # set the duration to 8 seconds
        if wav[1].shape[0] > wav[0] * 8:
            new_wav = wav[1][:wav[0] * 8]
        elif wav[1].shape[0] < wav[0] * 8:
            new_wav = np.zeros(wav[0] * 8)
            new_wav[:wav[1].shape[0]] = wav[1]
        else:
            new_wav = wav[1]

        envelope = OG_envelope(new_wav, wav[0])
        envelope = resample_poly(envelope, 1000, wav[0])
        wav_array[i, :] = envelope

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    for subject_path in subject_folders:
        print(subject_path)

        if subject_path.name == 'maestro23_shvfbp' or subject_path.name == 'maestro00_hoyzez':
            print('skipping')
        else:
            raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
            corr_audio_meg(raw_path, wav_folder, wav_array, save_path=subject_path)

    return


def cross_corr_audio_meg(raw_path, wav_folder, epoch, channel, save_path=None):
    '''
    '''

    wav_folder = Path(wav_folder)
    n_epochs = len(list(wav_folder.glob('**/*.wav')))
    wav_file = sorted(list(wav_folder.glob('**/*.wav')))[epoch]
    # Audio input envelope
    print(wav_file)
    wav = read(wav_file)
    # set the duration to 8 seconds
    if wav[1].shape[0] > wav[0]*8:
        new_wav = wav[1][:wav[0]*8]
    elif wav[1].shape[0] < wav[0]*8:
        new_wav = np.zeros(wav[0]*8)
        new_wav[:wav[1].shape[0]] = wav[1]
    else:
        new_wav = wav[1]

    envelope = OG_envelope(new_wav, wav[0])
    envelope = resample_poly(envelope, 1000, wav[0])

    epochs = epoching_maestro(raw_path, wav_folder, allow_maxshield=True) # Load raw and create epochs
    epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]  # Select listening conditions only
    epochs.pick('mag')  # keep magnetometers and gradiometers only
    epochs.crop(tmin=0, tmax=8)  # keep 8s only
    # epochs.resample(1000, n_jobs=-1)
    meg_data = epochs.get_data()
    meg_data = meg_data[epoch, channel, :8000]
    del epochs

    cross_corr = np.correlate(envelope, meg_data, mode='same')

    np.save(os.path.join(save_path, 'cross_corr_e'+str(epoch)+'_c'+str(channel)+'.npy'), cross_corr)

    return


def test_trf_input_features_perfRNN(wav_folder, surprise_dir, entropy_dir, save_path=None, pred_feat_list=None, sample_range=None, env_filter=15):
    '''
    Process and concatenate different input features used in the TRF analysis (mne.decoding.ReceptiveField). Complete
    the envelope, surprise and entropy time series so that they last 8s and then resample them to 100Hz. Finally, they are
    concatenated in an array of shape (n_times, n_epochs, n_features).
        wav_folder: str, path to folder containing the wav files used in the Maestro experiment
        start_sample: int, for each epoch, data before start_sample is removed (to avoid onset activity). Make
            sure to use the same value in create_trf_output_from_epochs.
    '''

    plt.switch_backend('QtAgg')

    if pred_feat_list is None:
        pred_feat_list = ['note_on', 'note_off', 'velocity', 'time_shift']
    if sample_range is None:
        sample_range = [0, 800]

    wav_folder = Path(wav_folder)
    n_epochs = len(list(wav_folder.glob('**/*.wav')))
    envelope_array = np.zeros((800, n_epochs, 2))  # Contains envelope and its derivative
    surprise_array = np.zeros((800, n_epochs, len(pred_feat_list)))
    entropy_array = np.zeros((800, n_epochs, len(pred_feat_list)))
    for i, wav_file in enumerate(sorted(list(wav_folder.glob('**/*.wav')))):
        # Audio input envelope
        print(wav_file)
        wav = read(wav_file)
        # set the duration to 8 seconds
        if wav[1].shape[0] > wav[0]*8:
            new_wav = wav[1][:wav[0]*8]
        elif wav[1].shape[0] < wav[0]*8:
            new_wav = np.zeros(wav[0]*8)
            new_wav[:wav[1].shape[0]] = wav[1]
        else:
            new_wav = wav[1]

        envelope = OG_envelope(new_wav, wav[0])  # Extract sound envelope
        if env_filter is not None:
            envelope = butter_lowpass_filter(envelope, env_filter, wav[0]) # lowpass filter the envelope
        env_derivative = np.gradient(envelope)
        envelope = resample_poly(envelope, 100, wav[0])
        env_derivative = resample_poly(env_derivative, 100, wav[0])
        env_derivative = np.maximum(0, env_derivative)  # Half-wave rectification of the derivative

        envelope_array[:, i, 0] = envelope  # add trial in the global array
        envelope_array[:, i, 1] = env_derivative

        # Surprise and entropy
        surprise_file, entropy_file = find_surp_entr_files(wav_file.name, surprise_dir, entropy_dir)
        surprise = pd.read_csv(surprise_file)[pred_feat_list].to_numpy()
        entropy = pd.read_csv(entropy_file)[pred_feat_list].to_numpy()

        # Keep only 8 seconds
        if surprise.shape[0] > 800:
            surprise = surprise[:800, :]
            entropy = entropy[:800, :]
        # Complete to 8 seconds
        if surprise.shape[0] < 800:
            new_surprise = np.zeros((800, len(pred_feat_list)))
            new_surprise[:surprise.shape[0], :] = surprise
            surprise = new_surprise
            new_entropy = np.zeros((800, len(pred_feat_list)))
            new_entropy[:entropy.shape[0], :] = entropy
            entropy = new_entropy
        # surprise = subsample(new_surprise, 100, 64)  # Downsample to 64Hz
        # entropy = subsample(new_entropy, 100, 64)

        surprise_array[:, i, :] = surprise  # add trial in the global array
        entropy_array[:, i, :] = entropy

        fig, axes = plt.subplots(nrows=7, ncols=1, dpi=200, sharex=True)
        X = np.linspace(0, 8, new_wav.shape[0])
        t = np.linspace(0, 8, 800)
        axes[0].plot(X, new_wav)
        axes[1].plot(t, envelope_array[:, i, 0])
        axes[2].plot(t, envelope_array[:, i, 1])
        axes[3].plot(t, surprise_array[:, i, 0])
        axes[4].plot(t, surprise_array[:, i, 1])
        axes[5].plot(t, entropy_array[:, i, 0])
        axes[6].plot(t, entropy_array[:, i, 1])
        axes[6].set_xlabel('Time (s)')
        plt.show()

    # Concatenate all features
    trf_input = np.concatenate((envelope_array, surprise_array, entropy_array), axis=2)

    # Remove onset activity
    trf_input = trf_input[sample_range[0]:sample_range[1], :, :]

    if save_path is not None:
        np.save(save_path, trf_input)

    return trf_input


def sanity_plot_meg(data_path, wav_folder, save_path, tmax=None, channel_type='mag'):

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    std_trials = []
    std_channels = []
    mean_trials = []
    rms_channels = []

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        epochs = epoching_maestro(raw_path, wav_folder)

        epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]
        epochs.pick(channel_type)
        epochs.filter(0.5, None, n_jobs=-1)  # Detrend
        epochs.crop(tmax=tmax)
        epochs.resample(100, n_jobs=-1)
        epochs_array = epochs.get_data()
        del epochs

        std_trials.append(np.std(epochs_array, axis=0))
        std_channels.append(np.std(epochs_array, axis=1))
        mean_trials.append(np.mean(epochs_array, axis=0))
        rms_channels.append(np.sqrt(np.mean(epochs_array**2, axis=1)))

    std_trials_max = [np.min([np.min(subject) for subject in std_trials]), np.max([np.max(subject) for subject in std_trials])]
    std_channels_max = [np.min([np.min(subject) for subject in std_channels]), np.max([np.max(subject) for subject in std_channels])]
    mean_trials_max = np.max([np.max(np.abs(subject)) for subject in mean_trials])
    rms_channels_max = [np.min([np.min(subject) for subject in rms_channels]), np.max([np.max(subject) for subject in rms_channels])]

    for i, subject_path in enumerate(subject_folders):

        fig, axes = plt.subplots(4, 1, sharex=True)
        im = axes[0].imshow(std_trials[i], aspect='auto', vmax=std_trials_max[1], vmin=std_trials_max[0], norm='log')
        axes[0].set_ylabel('channels')
        axes[0].set_title('Std across trials', loc='left')
        axes[0].axvline(50, linestyle='--', color='k')
        fig.colorbar(im, ax=axes[0])
        im = axes[1].imshow(std_channels[i], aspect='auto', vmax=std_channels_max[1], vmin=std_channels_max[0], norm='log')
        axes[1].set_ylabel('trials')
        axes[1].set_title('Std across channels', loc='left')
        axes[1].axvline(50, linestyle='--', color='k')
        fig.colorbar(im, ax=axes[1])
        im = axes[2].imshow(mean_trials[i], aspect='auto', vmax=mean_trials_max, vmin=-mean_trials_max)
        axes[2].set_ylabel('channels')
        axes[2].set_title('Mean across trials', loc='left')
        axes[2].axvline(50, linestyle='--', color='k')
        fig.colorbar(im, ax=axes[2])
        im = axes[3].imshow(rms_channels[i], aspect='auto', vmax=rms_channels_max[1], vmin=rms_channels_max[0], norm='log')
        axes[3].set_xlabel('time sample number (100Hz)')
        axes[3].set_ylabel('trials')
        axes[3].set_title('RMS across channels', loc='left')
        axes[3].axvline(50, linestyle='--', color='k')
        fig.colorbar(im, ax=axes[3])
        plt.tight_layout()

        plt.savefig(os.path.join(save_path, 'subject_' + subject_path.name + '.svg'))
        plt.close()

    return


if __name__ == "__main__":
    # run_one_block_trf(wav_folder='/home/mathieupvc/These/data/meg_maestro/exp_stim_meg',
    #                   stim_indices=[150, 180],
    #                   raw_path='/home/mathieupvc/These/data/meg_maestro/data/maestro26_ymxnvr/211005/run07.fif',
    #                   save_path='/home/mathieupvc/These/data/meg_maestro/trf_test_block',
    #                   sample_range=[1000, 8000],
    #                   env_filter=None)

    '''hist_kernel_time'''
    # hist_kernel_time(data_path='/home/mathieupvc/These/data/meg_maestro/data',
    #                  trf_folder='/home/mathieupvc/These/data/meg_maestro/trf_on_surprise',
    #                  features=['Envelope', 'Envelope derivative', 'Surprise'],
    #                  time=0.14)

    '''test_trf_input_features_perfRNN'''
    # test_trf_input_features_perfRNN(wav_folder="/home/mathieupvc/These/data/meg_maestro/exp_stim_meg",
    #                                 surprise_dir="/home/mathieupvc/These/data/meg_maestro/surprise_entropy_new/surprise",
    #                                 entropy_dir="/home/mathieupvc/These/data/meg_maestro/surprise_entropy_new/entropy",
    #                                 save_path=None, pred_feat_list=['note_on', 'time_shift'],
    #                                 sample_range=[0, 800], env_filter=15)
