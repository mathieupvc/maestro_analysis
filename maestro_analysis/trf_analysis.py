import os.path
import pickle

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
from pathlib import Path

import math
from math import ceil
import numpy as np
import pandas as pd
from scipy.io.wavfile import read
from scipy.signal import decimate, resample_poly
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

import mne
from mne.decoding import ReceptiveField, cross_val_multiscore, Scaler
from mne.stats import permutation_t_test

from maestro_analysis.signal_functions import OG_envelope, subsample, butter_lowpass_filter
from maestro_analysis.utils import cross_val_multidim_score, epoching_maestro, merge_grad_data, merge_grad_epochs_data, best_flip_combination, best_flip_combination_subjects, find_surp_entr_files, find_predictive_features_files
from maestro_analysis.stats import test_channels_significance, get_significant_channels_names, t_test_trf_kernels, one_sample_t_test_trf_kernels

import matplotlib
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False


def music_response_butterfly(data_path, channel_types=None, wav_folder="/home/mathieu/Data/meg_maestro/exp_stim_meg", save_format='png'):
    '''8 seconds butterfly plot averaged across all participants'''

    if channel_types is None:
        channel_types = ['grad']

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    evoked_list = []

    for subject_path in subject_folders:
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        epochs = epoching_maestro(raw_path, wav_folder)

        epochs.pick(channel_types)
        epochs.crop(tmax=8)
        listening_epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]
        del epochs

        # ERF
        listening_evoked = listening_epochs.average()
        del listening_epochs
        evoked_list.append(listening_evoked)

    evoked = mne.combine_evoked(evoked_list, 'equal')
    evoked.plot(show=False, spatial_colors=True, zorder='std')
    plt.savefig(Path(path, 'butterfly_all_subjects_8s.' + save_format))

    return



def plot_m100_psd(data_path, channel_types=None, wav_folder="/home/mathieu/Data/meg_maestro/exp_stim_meg", save_format='png'):
    '''
    Computes 500ms ERF and power spectrum on channels of the given type, averaged over all listening epochs.
        data_path: str, path to a folder containing all subjects preprocessed epochs
        channel_types: list, channel types to plot
    '''

    if channel_types is None:
        channel_types = ['grad']

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    for subject_path in subject_folders:
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        epochs = epoching_maestro(raw_path, wav_folder)

        epochs.pick(channel_types)
        epochs.crop(tmin=-0.2, tmax=0.5)
        listening_epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]
        del epochs

        # PSD
        listening_epochs.plot_psd(show=False, fmax=50)
        plt.savefig(Path(subject_path, 'psd.' + save_format))

        # ERF
        listening_evoked = listening_epochs.average()
        listening_evoked.plot_joint(show=False)
        plt.savefig(Path(subject_path, 'm100.' + save_format))

    return



def plot_m100_all_subjects(data_path, channel_types=None, wav_folder="/home/mathieu/Data/meg_maestro/exp_stim_meg", times=0.100, save_format='png'):

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
        epochs = epoching_maestro(raw_path, wav_folder)

        epochs.pick(channel_types)
        epochs.crop(tmin=-0.2, tmax=0.5)
        listening_epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]
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
    fig1.savefig(os.path.join(data_path, "all_erp." + save_format))
    fig2.savefig(os.path.join(data_path, "all_erp" + str(times*1000) + "ms_topo." + save_format))

    evoked = mne.combine_evoked(evoked_list, 'equal') # Plot the across subjects topomap
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 12), dpi=300, gridspec_kw=dict(width_ratios=[5, 0.5]))
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['axes.labelsize'] = 'large'
    evoked.plot_topomap(show=False, axes=[axes[0], axes[1]], times=times, colorbar=True)
    fig.savefig(os.path.join(data_path, "averaged_subject_erp" + str(times*1000) + "ms_topo." + save_format))
    fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(20, 12), dpi=300)
    evoked.plot(show=False, axes=axes2, titles=dict(grad='', mag=''), window_title='',
                          spatial_colors=True, zorder='std')
    fig2.savefig(os.path.join(data_path, "averaged_subject_erp." +save_format))

    return


def create_trf_input_features_perfRNN(wav_folder, surprise_dir, entropy_dir, save_path=None, pred_feat_list=None, sample_range=None, env_filter=15):
    '''
    Process and concatenate different input features used in the TRF analysis (mne.decoding.ReceptiveField). Complete
    the envelope, surprise and entropy time series so that they last 9s and then resample them to 100Hz. Finally, they are
    concatenated in an array of shape (n_times, n_epochs, n_features).
        wav_folder: str, path to folder containing the wav files used in the Maestro experiment
        start_sample: int, for each epoch, data before start_sample is removed (to avoid onset activity). Make
            sure to use the same value in create_trf_output_from_epochs.
    '''

    if pred_feat_list is None:
        pred_feat_list = ['note_on', 'note_off', 'velocity', 'time_shift']
    if sample_range is None:
        sample_range = [100, 800]

    wav_folder = Path(wav_folder)
    n_epochs = len(list(wav_folder.glob('**/*.wav')))
    envelope_array = np.zeros((900, n_epochs, 2))  # Contains envelope and its derivative
    surprise_array = np.zeros((900, n_epochs, len(pred_feat_list)))
    entropy_array = np.zeros((900, n_epochs, len(pred_feat_list)))
    for i, wav_file in enumerate(sorted(list(wav_folder.glob('**/*.wav')))):
        # Audio input envelope
        print(wav_file)
        wav = read(wav_file)
        # set the duration to 8 seconds
        if wav[1].shape[0] > wav[0]*9:
            new_wav = wav[1][:wav[0]*9]
        elif wav[1].shape[0] < wav[0]*9:
            new_wav = np.zeros(wav[0]*9)
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
        if surprise.shape[0] > 900:
            surprise = surprise[:900, :]
            entropy = entropy[:900, :]
        # Complete to 8 seconds
        if surprise.shape[0] < 900:
            new_surprise = np.zeros((900, len(pred_feat_list)))
            new_surprise[:surprise.shape[0], :] = surprise
            surprise = new_surprise
            new_entropy = np.zeros((900, len(pred_feat_list)))
            new_entropy[:entropy.shape[0], :] = entropy
            entropy = new_entropy
        # surprise = subsample(new_surprise, 100, 64)  # Downsample to 64Hz
        # entropy = subsample(new_entropy, 100, 64)

        surprise_array[:, i, :] = surprise  # add trial in the global array
        entropy_array[:, i, :] = entropy

    # note_onsets control feature, takes notes positions in note_on surprise and set the values to 1
    note_onsets_array = surprise_array[:, :, :1].copy()
    note_onsets_array[note_onsets_array != 0] = 1

    # Concatenate all features
    trf_input = np.concatenate((envelope_array, note_onsets_array, surprise_array, entropy_array), axis=2)

    # Remove onset activity
    trf_input = trf_input[sample_range[0]:sample_range[1], :, :]

    if save_path is not None:
        np.save(save_path, trf_input)

    print('Done')

    return


def create_trf_input_features_paulRNN(wav_folder, pred_feat_dir, save_path=None, pred_feat_list=None, sample_range=None, env_filter=15):
    '''
    Process and concatenate different input features used in the TRF analysis (mne.decoding.ReceptiveField). Complete
    the envelope, surprise and entropy time series so that they last 8s and then resample them to 100Hz. Finally, they are
    concatenated in an array of shape (n_times, n_epochs, n_features).
        wav_folder: str, path to folder containing the wav files used in the Maestro experiment
        start_sample: int, for each epoch, data before start_sample is removed (to avoid onset activity). Make
            sure to use the same value in create_trf_output_from_epochs.
    '''

    if pred_feat_list is None:
        pred_feat_list = ['surprise_mean', 'positive_surprise_rescaled', 'negative_surprise_rescaled', 'uncertainty']
    if sample_range is None:
        sample_range = [100, 800]

    wav_folder = Path(wav_folder)
    n_epochs = len(list(wav_folder.glob('**/*.wav')))
    envelope_array = np.zeros((900, n_epochs, 2))  # Contains envelope and its derivative
    predictive_features_array = np.zeros((900, n_epochs, len(pred_feat_list)))
    for i, wav_file in enumerate(sorted(list(wav_folder.glob('**/*.wav')))):
        # Audio input envelope
        print(wav_file)
        wav = read(wav_file)
        # set the duration to 9 seconds
        if wav[1].shape[0] > wav[0]*9:
            new_wav = wav[1][:wav[0]*9]
        elif wav[1].shape[0] < wav[0]*9:
            new_wav = np.zeros(wav[0]*9)
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
        predictive_features_file = find_predictive_features_files(wav_file.name, pred_feat_dir)
        predictive_features = pd.read_csv(predictive_features_file)
        predictive_features.rename(columns={'Unnamed: 0': 'surprise_mean'}, inplace=True)  # TODO fix this with Paul
        predictive_features.rename(columns={'suprise_mean': 'surprise_mean'}, inplace=True)
        predictive_features = predictive_features[pred_feat_list].to_numpy()

        # Keep only 9 seconds
        if predictive_features.shape[0] > 900:
            predictive_features = predictive_features[:900, :]
        # Complete to 9 seconds
        if predictive_features.shape[0] < 900:
            new_predictive_features = np.zeros((900, len(pred_feat_list)))
            new_predictive_features[:predictive_features.shape[0], :] = predictive_features
            predictive_features = new_predictive_features

        predictive_features_array[:, i, :] = predictive_features  # add trial in the global array

    # note_onsets control feature, takes notes positions in note_on surprise and set the values to 1
    positive_surprise_index = pred_feat_list.index('positive_surprise_rescaled')
    note_onsets_array = predictive_features_array[:, :, positive_surprise_index:positive_surprise_index+1].copy()
    note_onsets_array[note_onsets_array != 0] = 1

    # Concatenate all features
    trf_input = np.concatenate((envelope_array, note_onsets_array, predictive_features_array), axis=2)

    # Remove onset activity
    trf_input = trf_input[sample_range[0]:sample_range[1], :, :]

    if save_path is not None:
        np.save(save_path, trf_input)

    return trf_input



def plot_trf_input_features(trf_input_path, wav_folder, pred_feat_list=None):

    if pred_feat_list is None:
        pred_feat_list = ['note_on', 'time_shift']

    trf_input = np.load(trf_input_path)
    trf_input = (trf_input - np.mean(trf_input, axis=(0, 1))) / np.std(trf_input, axis=(0, 1))
    print(trf_input.shape)

    plt.switch_backend('QtAgg')

    wav_folder = Path(wav_folder)
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

        new_wav = new_wav[wav[0]:] #remove first second

        new_wav = (new_wav - np.mean(new_wav)) / np.std(new_wav)

        fig, axes = plt.subplots(nrows=5, ncols=1, dpi=200, sharex=True)
        X = np.linspace(0, 7, wav[0] * 7)
        t = np.linspace(0, 7, trf_input.shape[0])
        # print(X.shape)
        # print(new_wav.shape)
        # print(new_wav[wav[0]:].shape)
        # axes[0].plot(X, new_wav[wav[0]:])
        axes[0].plot(X, new_wav)
        # axes[0].plot(t, trf_input[:, i, 0])
        axes[1].plot(t, trf_input[:, i, 0])
        axes[2].plot(t, trf_input[:, i, 1])
        axes[3].plot(t, trf_input[:, i, 2])
        axes[4].plot(t, trf_input[:, i, 3])
        axes[4].set_xlabel('Time (s)')
        plt.show()


def create_trf_output_from_epochs(raw_path, wav_folder, save_path=None, sample_range=None, channel_type='grad', filt_range=None, zscore=True):
    '''
    For 1 subject only. From epochs, output a numpy array containing MEG data in the right format (sample rate,
    duration, shape).
    '''

    if sample_range is None:
        sample_range = [100, 800]

    if filt_range is None:
        filt_range = [1, None]

    epochs = epoching_maestro(raw_path, wav_folder) # Load raw and create epochs
    epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]  # Select listening conditions only
    epochs.pick(channel_type)  # keep magnetometers and gradiometers only
    epochs.filter(filt_range[0], filt_range[1], n_jobs=-1)  # Test with lowpass filter
    epochs.resample(100, n_jobs=-1)
    epochs.crop(tmin=0, tmax=8)  # keep 8s only
    trf_output = epochs.get_data()
    metadata = epochs.metadata
    del epochs

    # Combine planar gradiometers pairs
    if channel_type == 'grad':
        trf_output = merge_grad_epochs_data(trf_output, method='rms')

    # Remove onset activity
    trf_output = trf_output[:, :, sample_range[0]:sample_range[1]]
    # Z-scoring
    if zscore:
        trf_output = (trf_output - np.mean(trf_output, axis=(0, 2), keepdims=True)) / np.std(trf_output, axis=(0, 2), keepdims=True)  # zscoring by channel

    trf_output = np.moveaxis(trf_output, 2, 0)

    if save_path is not None:
        np.save(save_path, trf_output)

    return trf_output, metadata


def create_trf_output_from_epochs_channels_subset(raw_path, wav_folder, channel_types=None, channel_nb=20, save_path=None):
    '''
    For 1 subject only. From epochs, output a numpy array containing MEG data in the right format (sample rate,
    duration, shape).
    '''

    if channel_types is None:
        channel_types = ['grad', 'mag']

    epochs = epoching_maestro(raw_path, wav_folder) # Load raw and create epochs
    epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]  # Select listening conditions only
    # epochs.pick(channel_type)  # keep channel_type only
    epochs.crop(tmin=0, tmax=8)  # keep 8s only
    metadata = epochs.metadata
    resampled_epochs = epochs.copy().filter(1, 8, n_jobs=-1)  # Test with lowpass filter
    resampled_epochs = resampled_epochs.resample(100, n_jobs=-1)
    to_concatenate = []

    for channel_type in channel_types:
        # Get the subset of channels with the highest activity
        epochs_copy = epochs.copy()
        epochs_copy.pick(channel_type)
        epochs_copy.crop(tmin=0.050, tmax=0.200)
        evoked = epochs_copy.average()
        evoked = evoked.get_data()
        rms = np.sqrt(np.mean(evoked**2, axis=1))
        channels_to_keep = np.argsort(rms, axis=0, kind=None)[-channel_nb:]
        print(channels_to_keep)
        print([epochs_copy.info['chs'][i]['ch_name'] for i in channels_to_keep])

        # Seperate left and right channels
        layout = mne.channels.find_layout(epochs_copy.info, ch_type=channel_type)
        left_to_keep = []
        right_to_keep = []
        for channel in channels_to_keep:
            if layout.pos[channel, 0] < 0.5:  # based on x coordinate
                left_to_keep.append(channel)
            else:
                right_to_keep.append(channel)
        print([epochs_copy.info['chs'][i]['ch_name'] for i in left_to_keep])
        print([epochs_copy.info['chs'][i]['ch_name'] for i in right_to_keep])
        print('Left channels:', len(left_to_keep))
        print('Right channels:', len(right_to_keep))
        #assert len(left_to_keep) > 0
        #assert len(right_to_keep) > 0

        # Get trf_output
        resampled_epochs_channel_type = resampled_epochs.copy().pick(channel_type).get_data()
        #left_average = np.mean(resampled_epochs_channel_type[:, left_to_keep, :], axis=1, keepdims=True)
        #right_average = np.mean(resampled_epochs_channel_type[:, right_to_keep, :], axis=1, keepdims=True)
        average = np.mean(resampled_epochs_channel_type[:, left_to_keep + right_to_keep, :], axis=1, keepdims=True)
        #to_concatenate.extend([left_average, right_average])
        to_concatenate.extend([average])

    trf_output = np.concatenate(to_concatenate, axis=1)
    del epochs
    trf_output = np.moveaxis(trf_output, 2, 0)

    if save_path is not None:
        np.save(save_path, trf_output)

    return trf_output, metadata


def load_power_for_trf(tfr_path, sample_range=None):
    '''
    Load power from complex TFR
    :param tfr_path: path to TFR (.h5 file), should be sampled at 100Hz
    :param sample_range: start and end samples
    :return:
    '''

    if sample_range is None:
        sample_range = [100, 800]

    power_epochs = mne.time_frequency.read_tfrs(tfr_path)[0]
    power_epochs.data = np.abs(power_epochs.data)**2  # get energy from complex tfr
    power_epochs = power_epochs.apply_baseline(baseline=(-0.75, -0.25), mode='zscore')

    power_epochs.crop(tmin=0, tmax=8)  # keep 8s only
    trf_output = power_epochs.data
    metadata = power_epochs.metadata
    freqs = power_epochs.freqs
    del power_epochs

    # Remove onset activity
    trf_output = trf_output[:, :, :, sample_range[0]:sample_range[1]]

    trf_output = np.moveaxis(trf_output, 3, 0)

    return trf_output, metadata, freqs


def list_from_metadata(metadata):
    """Create a list to select only epochs that were kept"""

    return [int(row['stim_name'].split('_')[0][5:8]) for _, row in metadata.iterrows()]


def condition_list_from_metadata(metadata, condition):
    """
    Create a list to select only epochs of one or several conditions
        condition: list of str
    """

    to_keep = []
    for i in range(len(metadata)):
        keep = False
        for cond in condition:
            if cond in metadata.loc[metadata.index[i], 'stim_name']:
                keep = True
        if keep:
            to_keep.append(i)

    return to_keep


def normalize_input_features(X, sparse_positive_features=None):
    '''

    :param X: numpy array of shape (time_samples, n_epochs, n_features)
    :param sparse_positive_features:
    :return:
    '''

    if sparse_positive_features is None:
        #sparse_positive_features = [x for x in range(2, X.shape[2])]  # Note onsets and predictive features PerfRNN
        sparse_positive_features = [2, 4]  # PaulRNN
    dense_features = [x for x in range(X.shape[2]) if x not in sparse_positive_features]

    X[:, :, dense_features] = (X[:, :, dense_features] - np.mean(X[:, :, dense_features], axis=(0, 1))) / np.std(X[:, :, dense_features], axis=(0, 1))
    X[:, :, sparse_positive_features] = X[:, :, sparse_positive_features] / np.std(X[:, :, sparse_positive_features], axis=(0, 1))

    return X

def run_trf(input_path, output, metadata, save_model_path=None, condition=None, A_features=None):
    """
    Run TRF on 1 subject.
        condition: list of str
        A_features: features to keep in rf_env (TRF version based on acoustic features only)
        output: str or numpy array, if str, output is a path to a npy file.
    """

    if A_features is None:
        A_features = [0, 1]

    X = np.load(input_path)
    list_to_keep = list_from_metadata(metadata)
    X = X[:, list_to_keep, :]
    print(X.shape)
    if condition is not None:
        condition_list = condition_list_from_metadata(metadata, condition)
        X = X[:, condition_list, :]
    print(X.shape)
    # X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))
    X = normalize_input_features(X)

    if isinstance(output, str):
        Y = np.load(output)
    else:
        Y = output
    if condition is not None:
        Y = Y[:, condition_list, :]
    print(X.shape)
    print(Y.shape)
    X_env = X[:, :, A_features]

    feature_names = np.arange(X.shape[2])
    feature_names_env = np.arange(X_env.shape[2])

    alphas = np.logspace(-1, 7, 9)
    # alphas = np.logspace(3, 7, 5)
    scores = np.zeros((alphas.shape[0], Y.shape[2]))
    scores_env = np.zeros((alphas.shape[0], Y.shape[2]))
    models = []
    models_env = []
    
    n_folds = 6
    cv = KFold(n_folds, shuffle=True, random_state=37)

    for ii, alpha in enumerate(alphas):
        print('alpha:', alpha)

        rf = ReceptiveField(-0.2, 0.5, 100, feature_names, estimator=alpha, scoring='corrcoef', n_jobs=-1)
        rf_env = ReceptiveField(-0.2, 0.5, 100, feature_names_env, estimator=alpha, scoring='corrcoef', n_jobs=-1)
        score = np.zeros((n_folds, Y.shape[2]))
        score_env = np.zeros((n_folds, Y.shape[2]))

        for fold_number, (train, test) in enumerate(cv.split(np.moveaxis(X, 1, 0))):
            rf.fit(X[:, train, :], Y[:, train, :])
            score[fold_number, :] = rf.score(X[:, test, :], Y[:, test, :])

            # TRF based on sound envelope only
            rf_env.fit(X_env[:, train, :], Y[:, train, :])
            score_env[fold_number, :] = rf_env.score(X_env[:, test, :], Y[:, test, :])

        scores[ii, :] = np.mean(score, axis=0)
        models.append(rf)
        scores_env[ii, :] = np.mean(score_env, axis=0)
        models_env.append(rf_env)
        print('mean score across channels:', np.mean(scores[ii, :]))
        print('mean score_env across channels:', np.mean(scores_env[ii, :]))

    # Choose the model that performed best on the held out data
    ix_best_alpha = np.argmax(np.mean(scores, axis=1))
    ix_best_alpha_env = np.argmax(np.mean(scores_env, axis=1))
    print('best mean score across channels', np.max(np.mean(scores, axis=1)))
    print('best mean score_env across channels', np.max(np.mean(scores_env, axis=1)))
    best_mod = models[ix_best_alpha]
    best_mod_env = models_env[ix_best_alpha_env]
    if not math.isnan(np.max(scores)):
        best_mod.fit(X, Y)
    if not math.isnan(np.max(scores_env)):
        best_mod_env.fit(X_env, Y)

    if save_model_path is not None:
        os.makedirs(save_model_path, exist_ok=True)
        file_to_store = open(os.path.join(save_model_path, "rf.pickle"), "wb")
        pickle.dump(best_mod, file_to_store)
        file_to_store.close()

        file_to_store = open(os.path.join(save_model_path, "rf_env.pickle"), "wb")
        pickle.dump(best_mod_env, file_to_store)
        file_to_store.close()

    return scores[ix_best_alpha, :], scores_env[ix_best_alpha_env, :]


def run_acoustic_trf(trf_input, output, metadata, save_model_path=None, condition=None, A_features=None, sparse_positive_features=None):
    """
    Run TRF (only acoustic features) on 1 subject.
        condition: list of str
        A_features: features to keep in rf_env (TRF version based on acoustic features only)
        output: str or numpy array, if str, output is a path to a npy file.
    """

    if A_features is None:
        A_features = [0, 1, 2]

    if isinstance(trf_input, str):
        X = np.load(trf_input)
    else:
        X = trf_input
    list_to_keep = list_from_metadata(metadata)
    X = X[:, list_to_keep, :]
    print(X.shape)
    if condition is not None:
        condition_list = condition_list_from_metadata(metadata, condition)
        X = X[:, condition_list, :]
    print(X.shape)
    # X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))
    X = normalize_input_features(X, sparse_positive_features=sparse_positive_features)

    if isinstance(output, str):
        Y = np.load(output)
    else:
        Y = output
    if condition is not None:
        Y = Y[:, condition_list, :]
    print(X.shape)
    print(Y.shape)
    X_env = X[:, :, A_features]

    feature_names_env = np.arange(X_env.shape[2])

    alphas = np.logspace(-1, 7, 9)
    # alphas = np.logspace(3, 7, 5)
    scores_env = np.zeros((alphas.shape[0], Y.shape[2]))
    models_env = []

    n_folds = 6
    cv = KFold(n_folds, shuffle=True, random_state=37)

    for ii, alpha in enumerate(alphas):
        print('alpha:', alpha)

        rf_env = ReceptiveField(-0.2, 0.5, 100, feature_names_env, estimator=alpha, scoring='corrcoef', n_jobs=-1)
        score_env = np.zeros((n_folds, Y.shape[2]))

        for fold_number, (train, test) in enumerate(cv.split(np.moveaxis(X, 1, 0))):
            # TRF based on sound envelope only
            rf_env.fit(X_env[:, train, :], Y[:, train, :])
            score_env[fold_number, :] = rf_env.score(X_env[:, test, :], Y[:, test, :])

        scores_env[ii, :] = np.mean(score_env, axis=0)
        models_env.append(rf_env)
        print('mean score_env across channels:', np.mean(scores_env[ii, :]))

    # Choose the model that performed best on the held out data
    ix_best_alpha_env = np.argmax(np.mean(scores_env, axis=1))
    print('best mean score_env across channels', np.max(np.mean(scores_env, axis=1)))
    best_mod_env = models_env[ix_best_alpha_env]
    if not math.isnan(np.max(scores_env)):
        best_mod_env.fit(X_env, Y)

    if save_model_path is not None:
        os.makedirs(save_model_path, exist_ok=True)
        file_to_store = open(os.path.join(save_model_path, "rf_env.pickle"), "wb")
        pickle.dump(best_mod_env, file_to_store)
        file_to_store.close()

    return scores_env[ix_best_alpha_env, :]


def run_trf_cv_test(input_path, output, metadata, save_model_path=None, condition=None, A_features=None):
    """
    Run TRF on 1 subject.
        condition: list of str
        A_features: features to keep in rf_env (TRF version based on acoustic features only)
        output: str or numpy array, if str, output is a path to a npy file.
    """

    if A_features is None:
        A_features = [0, 1]

    X = np.load(input_path)
    list_to_keep = list_from_metadata(metadata)
    X = X[:, list_to_keep, :]
    print(X.shape)
    if condition is not None:
        condition_list = condition_list_from_metadata(metadata, condition)
        X = X[:, condition_list, :]
    print(X.shape)
    # X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))
    X = normalize_input_features(X)

    if isinstance(output, str):
        Y = np.load(output)
    else:
        Y = output
    if condition is not None:
        Y = Y[:, condition_list, :]
    print(X.shape)
    print(Y.shape)
    X_env = X[:, :, A_features]

    feature_names = np.arange(X.shape[2])
    feature_names_env = np.arange(X_env.shape[2])

    alphas = np.logspace(-1, 7, 9)
    scores = np.zeros_like(alphas)
    scores_env = np.zeros_like(alphas)
    models = []
    models_env = []

    n_folds = 6
    cv = KFold(n_folds, shuffle=True, random_state=37)

    for ii, alpha in enumerate(alphas):
        print('alpha:', alpha)

        rf = ReceptiveField(-0.2, 0.5, 100, feature_names, estimator=alpha, scoring='corrcoef', n_jobs=-1)
        rf_env = ReceptiveField(-0.2, 0.5, 100, feature_names_env, estimator=alpha, scoring='corrcoef', n_jobs=-1)
        score = np.zeros(n_folds)
        score_env = np.zeros(n_folds)

        score = cross_val_multiscore(rf, X, Y, cv=10, n_jobs=-1)
        print(score.shape)

        score_env = cross_val_multiscore(rf_env, X_env, Y, cv=10, n_jobs=-1)
        print(score_env.shape)

        # Now make predictions about the model output, given input stimuli.
        scores[ii] = score.mean()
        models.append(rf)
        scores_env[ii] = score_env.mean()
        models_env.append(rf_env)
        print('score:', scores[ii])
        print('score_env:', scores_env[ii])

    # Choose the model that performed best on the held out data
    ix_best_alpha = np.argmax(scores)
    ix_best_alpha_env = np.argmax(scores_env)
    print('best score', np.max(scores))
    print('best score_env', np.max(scores_env))
    best_mod = models[ix_best_alpha]
    best_mod_env = models_env[ix_best_alpha_env]
    if not math.isnan(np.max(scores)):
        best_mod.fit(X, Y)
    if not math.isnan(np.max(scores_env)):
        best_mod_env.fit(X_env, Y)

    if save_model_path is not None:
        os.makedirs(save_model_path, exist_ok=True)
        file_to_store = open(os.path.join(save_model_path, "rf.pickle"), "wb")
        pickle.dump(best_mod, file_to_store)
        file_to_store.close()

        file_to_store = open(os.path.join(save_model_path, "rf_env.pickle"), "wb")
        pickle.dump(best_mod_env, file_to_store)
        file_to_store.close()

    return np.max(scores), np.max(scores_env)


def run_existing_trf(input_path, output, metadata, pickle_path, save_model_path=None, condition=None):

    X = np.load(input_path)
    list_to_keep = list_from_metadata(metadata)
    X = X[:, list_to_keep, :]
    print(X.shape)
    if condition is not None:
        condition_list = condition_list_from_metadata(metadata, condition)
        X = X[:, condition_list, :]
    print(X.shape)
    # X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))
    X = normalize_input_features(X)

    if isinstance(output, str):
        Y = np.load(output)
    else:
        Y = output
    if condition is not None:
        Y = Y[:, condition_list, :]
    print(X.shape)
    print(Y.shape)

    file = open(os.path.join(pickle_path, "rf.pickle"), 'rb')
    rf = pickle.load(file)
    file.close()

    rf.fit(X, Y)

    if save_model_path is not None:
        os.makedirs(save_model_path, exist_ok=True)
        file_to_store = open(os.path.join(save_model_path, "rf.pickle"), "wb")
        pickle.dump(rf, file_to_store)
        file_to_store.close()

    return


def run_all_alphas_trf(input_path, output, metadata, save_model_path=None, condition=None):

    X = np.load(input_path)
    list_to_keep = list_from_metadata(metadata)
    X = X[:, list_to_keep, :]
    print(X.shape)
    if condition is not None:
        condition_list = condition_list_from_metadata(metadata, condition)
        X = X[:, condition_list, :]
    print(X.shape)
    # X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))
    X = normalize_input_features(X)

    if isinstance(output, str):
        Y = np.load(output)
    else:
        Y = output
    if condition is not None:
        Y = Y[:, condition_list, :]
    print(X.shape)
    print(Y.shape)

    feature_names = np.arange(X.shape[2])
    alphas = np.logspace(-3, 12, 16)

    for ii, alpha in enumerate(alphas):
        print('alpha:', alpha)

        rf = ReceptiveField(-0.2, 0.5, 100, feature_names, estimator=alpha, scoring='corrcoef', n_jobs=-1)
        rf.fit(X, Y)

        if save_model_path is not None:
            os.makedirs(save_model_path, exist_ok=True)
            file_to_store = open(os.path.join(save_model_path, "rf_" + str(alpha) + ".pickle"), "wb")
            pickle.dump(rf, file_to_store)
            file_to_store.close()

    return


def run_residual_trf(trf_input, output, metadata, pickle_path, save_model_path=None, sample_range=None, condition=None, A_features=None, sparse_positive_features=None):
    """
    Run residual TRF on 1 subject. Use an existing TRF and subtract its prediction from the MEG signal before running a new TRF.
    sample range of the input should not be reduced before this function (because of edge effects caused by trf prediction).
    sample_range should be the same as the one used in create_trf_output_from_epochs.
    """

    if A_features is None:
        A_features = [0, 1, 2]

    if sample_range is None:
        sample_range = [100, 800]

    if isinstance(trf_input, str):
        X = np.load(trf_input)
    else:
        X = trf_input
    list_to_keep = list_from_metadata(metadata)
    X = X[:, list_to_keep, :]
    print(X.shape)
    if condition is not None:
        condition_list = condition_list_from_metadata(metadata, condition)
        X = X[:, condition_list, :]
    print(X.shape)
    # X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))
    X = normalize_input_features(X, sparse_positive_features=sparse_positive_features)

    if isinstance(output, str):
        Y = np.load(output)
    else:
        Y = output
    if condition is not None:
        Y = Y[:, condition_list, :]
    print(X.shape)
    print(Y.shape)

    # Compute residuals
    file = open(os.path.join(pickle_path, "rf_env.pickle"), 'rb')
    rf = pickle.load(file)
    file.close()

    Y = Y - rf.predict(X[:, :, A_features])[sample_range[0]:sample_range[1], :, :]

    # Keep only predictive features
    X = X[sample_range[0]:sample_range[1], :, [x for x in range(X.shape[2]) if x not in A_features]]

    # TRF
    feature_names = np.arange(X.shape[2])

    alphas = np.logspace(-1, 7, 9)
    # alphas = np.logspace(3, 7, 5)
    scores = np.zeros((alphas.shape[0], Y.shape[2]))
    models = []

    n_folds = 6
    cv = KFold(n_folds, shuffle=True, random_state=37)

    for ii, alpha in enumerate(alphas):
        print('alpha:', alpha)

        rf = ReceptiveField(-0.2, 0.5, 100, feature_names, estimator=alpha, scoring='corrcoef', n_jobs=-1)
        score = np.zeros((n_folds, Y.shape[2]))

        for fold_number, (train, test) in enumerate(cv.split(np.moveaxis(X, 1, 0))):
            rf.fit(X[:, train, :], Y[:, train, :])
            score[fold_number, :] = rf.score(X[:, test, :], Y[:, test, :])

        scores[ii, :] = np.mean(score, axis=0)
        models.append(rf)
        print('mean score across channels:', np.mean(scores[ii, :]))

    # Choose the model that performed best on the held out data
    ix_best_alpha = np.argmax(np.mean(scores, axis=1))
    print('best mean score across channels', np.max(np.mean(scores, axis=1)))
    best_mod = models[ix_best_alpha]
    if not math.isnan(np.max(scores)):
        best_mod.fit(X, Y)

    if save_model_path is not None:
        os.makedirs(save_model_path, exist_ok=True)
        file_to_store = open(os.path.join(save_model_path, "rf.pickle"), "wb")
        pickle.dump(best_mod, file_to_store)
        file_to_store.close()

    return scores[ix_best_alpha, :]


def run_existing_residual_trf(input_path, output, metadata, pickle_path, save_model_path=None, condition=None, sample_range=None):

    if sample_range is None:
        sample_range = [100, 800]

    X = np.load(input_path)
    list_to_keep = list_from_metadata(metadata)
    X = X[:, list_to_keep, :]
    print(X.shape)
    if condition is not None:
        condition_list = condition_list_from_metadata(metadata, condition)
        X = X[:, condition_list, :]
    print(X.shape)
    # X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))
    X = normalize_input_features(X)

    if isinstance(output, str):
        Y = np.load(output)
    else:
        Y = output
    if condition is not None:
        Y = Y[:, condition_list, :]
    print(X.shape)
    print(Y.shape)

    file = open(os.path.join(pickle_path, "rf_env.pickle"), 'rb')
    rf = pickle.load(file)
    file.close()

    rf.fit(X, Y)

    if save_model_path is not None:
        os.makedirs(save_model_path, exist_ok=True)
        file_to_store = open(os.path.join(save_model_path, "rf_env.pickle"), "wb")
        pickle.dump(rf, file_to_store)
        file_to_store.close()

    return


def run_trf_all_subjects(data_path, trf_input_path, wav_folder, save_model_folder, filt_range=None, A_features=None, sparse_positive_features=None, channel_type='grad', sample_range=None, residual_trf=False):
    '''
    trf_folder: only with condition
    '''

    if A_features is None:
        A_features = [0, 1, 2]
    if sample_range is None:
        sample_range = [100, 800]

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    scores = []
    scores_env = []
    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]

        trf_output, metadata = create_trf_output_from_epochs(raw_path, wav_folder=wav_folder, save_path=None, sample_range=sample_range, channel_type=channel_type, filt_range=filt_range)

        if residual_trf:
            score = run_residual_trf(trf_input_path, trf_output, metadata, pickle_path=os.path.join(save_model_folder, subject_path.name), save_model_path=os.path.join(save_model_folder, 'residual_trf', subject_path.name), sample_range=sample_range, A_features=A_features, sparse_positive_features=sparse_positive_features)
            scores.append(score[:, np.newaxis])  # newaxis is needed to later use scores with plotting functions

        else:
            score, score_env = run_trf(trf_input_path, trf_output, metadata, save_model_path=os.path.join(save_model_folder, subject_path.name), A_features=A_features)
            scores.append(score)
            scores_env.append(score_env)

    scores = np.array(scores)
    if not residual_trf:
        scores_env = np.array(scores_env)
        scores = np.stack((scores, scores_env), axis=-1)
        np.save(os.path.join(save_model_folder, 'scores.npy'), scores)
    else:
        np.save(os.path.join(save_model_folder, 'residual_trf', 'scores.npy'), scores)

    return


def run_acoustic_trf_all_subjects(data_path, trf_input_path, wav_folder, save_model_folder, filt_range=None, A_features=None, sparse_positive_features=None, channel_type='grad', sample_range=None):
    '''
    '''

    if A_features is None:
        A_features = [0, 1, 2]
    if sample_range is None:
        sample_range = [100, 800]

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    scores_env = []
    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]

        trf_output, metadata = create_trf_output_from_epochs(raw_path, wav_folder=wav_folder, save_path=None, sample_range=sample_range, channel_type=channel_type, filt_range=filt_range)

        score_env = run_acoustic_trf(trf_input_path, trf_output, metadata, save_model_path=os.path.join(save_model_folder, subject_path.name), A_features=A_features, sparse_positive_features=sparse_positive_features)
        scores_env.append(score_env[:, np.newaxis])

    scores_env = np.array(scores_env)
    np.save(os.path.join(save_model_folder, 'scores_env.npy'), scores_env)

    return


def run_acoustic_trf_all_subjects_frequencies(frequency_data_path, trf_input_path, save_model_folder, A_features=None, sample_range=None):
    '''
    '''

    if A_features is None:
        A_features = [0, 1, 2]
    if sample_range is None:
        sample_range = [100, 800]

    path = Path(frequency_data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    scores_env = []
    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        tfr_path = list(subject_path.glob('**/*tfr.h5'))[0]

        trf_output, metadata, freqs = load_power_for_trf(tfr_path, sample_range=sample_range)

        subject_score = np.zeros((trf_output.shape[2], trf_output.shape[3]))
        for frequency in range(trf_output.shape[3]):
            score_env = run_acoustic_trf(trf_input_path, trf_output[:, :, :, frequency], metadata, save_model_path=os.path.join(save_model_folder, subject_path.name, str(round(freqs[frequency], 2))), A_features=A_features)
            subject_score[:, frequency] = score_env
        scores_env.append(subject_score)

    scores_env = np.array(scores_env)
    np.save(os.path.join(save_model_folder, 'scores_env.npy'), scores_env)

    return


def run_residual_trf_all_subjects_frequencies(frequency_data_path, trf_input_path, save_model_folder, A_features=None, sample_range=None):
    '''
    '''

    if A_features is None:
        A_features = [0, 1, 2]
    if sample_range is None:
        sample_range = [100, 800]

    path = Path(frequency_data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    scores = []
    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        tfr_path = list(subject_path.glob('**/*tfr.h5'))[0]

        trf_output, metadata, freqs = load_power_for_trf(tfr_path, sample_range=sample_range)

        subject_score = np.zeros((trf_output.shape[2], trf_output.shape[3]))
        for frequency in range(trf_output.shape[3]):
            score = run_residual_trf(trf_input_path, trf_output[:, :, :, frequency], metadata, pickle_path=os.path.join(save_model_folder, subject_path.name, str(round(freqs[frequency], 2))), save_model_path=os.path.join(save_model_folder, 'residual_trf', subject_path.name, str(round(freqs[frequency], 2))), sample_range=sample_range, A_features=A_features)
            subject_score[:, frequency] = score
        scores.append(subject_score)

    scores = np.array(scores)
    np.save(os.path.join(save_model_folder, 'residual_trf', 'scores.npy'), scores)

    return


def trf_all_conditions_all_subjects(data_path, trf_input_path, wav_folder, main_save_model_folder, conditions, filt_range=None, channel_type='grad', sample_range=None, residual_trf=False, A_features=None, sparse_positive_features=None):
    '''Run different TRF for each condition in the list conditions
    example conditions: ['rep1','rep2','rep3','2.5Hz','3.5Hz','4.5Hz','5.5Hz','6.5Hz']
    '''

    if sample_range is None:
        sample_range = [100, 800]

    if A_features is None:
        A_features = [0, 1, 2]

    for condition in conditions:
        if residual_trf:
            os.makedirs(os.path.join(main_save_model_folder, 'residual_trf', condition), exist_ok=True)
        else:
            os.makedirs(os.path.join(main_save_model_folder, condition), exist_ok=True)

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    scores = [[] for _ in range(len(conditions))]
    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]

        trf_output, metadata = create_trf_output_from_epochs(raw_path, wav_folder=wav_folder, save_path=None, sample_range=sample_range, channel_type=channel_type, filt_range=filt_range)

        for j, condition in enumerate(conditions):
            if residual_trf:
                # fit a trf on all epochs of the current condition (will be used for parameters plotting)
                # run_existing_residual_trf(trf_input_path, trf_output, metadata, os.path.join(main_save_model_folder, subject_path.name), save_model_path=os.path.join(main_save_model_folder, 'residual_trf', condition, subject_path.name), condition=[condition], sample_range=sample_range)  # sample_range is used because we use the non-croped trf input when working with residuals
                # run a 6-fold cross-validation to estimate scores
                score = run_residual_trf(trf_input_path, trf_output, metadata,
                                         # pickle_path=os.path.join(main_save_model_folder, 'residual_trf', condition, subject_path.name),
                                         pickle_path=os.path.join(main_save_model_folder, subject_path.name),  # residual computed with trf_env computed on whole data
                                         save_model_path=os.path.join(main_save_model_folder, 'residual_trf', condition, subject_path.name),
                                         sample_range=sample_range,
                                         condition=[condition],
                                         A_features=A_features,
                                         sparse_positive_features=sparse_positive_features)

            else:
                # fit a trf on all epochs of the current condition (will be used for parameters plotting)
                run_existing_trf(trf_input_path, trf_output, metadata, os.path.join(main_save_model_folder, subject_path.name), save_model_path=os.path.join(main_save_model_folder, condition, subject_path.name), condition=[condition])
                # run a 6-fold cross-validation to estimate scores
                score = run_channel_trf(trf_input_path, trf_output, metadata, os.path.join(main_save_model_folder, subject_path.name), save=False, condition=[condition])

            scores[j].append(score)

    for j, condition in enumerate(conditions):
        scores[j] = np.array(scores[j])
        if residual_trf:
            np.save(os.path.join(main_save_model_folder, 'residual_trf', condition, "channels_scores.npy"), scores[j])
        else:
            np.save(os.path.join(main_save_model_folder, condition, "channels_scores.npy"), scores[j])

    return


def trf_residual_all_conditions_all_subjects_all_frequencies(frequency_data_path, trf_input_path, main_save_model_folder, conditions, sample_range=None, A_features=None):
    '''Run different TRF for each condition in the list conditions
    example conditions: ['rep1','rep2','rep3','2.5Hz','3.5Hz','4.5Hz','5.5Hz','6.5Hz']
    '''

    if sample_range is None:
        sample_range = [100, 800]

    if A_features is None:
        A_features = [0, 1, 2]

    for condition in conditions:
        os.makedirs(os.path.join(main_save_model_folder, 'residual_trf', condition), exist_ok=True)

    path = Path(frequency_data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    scores = [[] for _ in range(len(conditions))]
    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        tfr_path = list(subject_path.glob('**/*tfr.h5'))[0]

        trf_output, metadata, freqs = load_power_for_trf(tfr_path, sample_range=sample_range)

        for j, condition in enumerate(conditions):
            subject_score = np.zeros((trf_output.shape[2], trf_output.shape[3]))
            for frequency in range(trf_output.shape[3]):
                score = run_residual_trf(trf_input_path, trf_output[:, :, :, frequency], metadata,
                                         pickle_path=os.path.join(main_save_model_folder, subject_path.name,
                                                                  str(round(freqs[frequency], 2))),
                                         save_model_path=os.path.join(main_save_model_folder, 'residual_trf', condition,
                                                                      subject_path.name,
                                                                      str(round(freqs[frequency], 2))),
                                         sample_range=sample_range, condition=[condition], A_features=A_features)
                subject_score[:, frequency] = score
            scores[j].append(subject_score)

    for j, condition in enumerate(conditions):
        scores[j] = np.array(scores[j])
        np.save(os.path.join(main_save_model_folder, 'residual_trf', condition, "channels_scores.npy"), scores[j])

    return


def trf_correct_incorrect(data_path, trf_input_path, wav_folder, main_save_model_folder, correct_incorrect_dict, filt_range=None, channel_type='grad', sample_range=None):
    '''Run different TRF for each condition in the list conditions
    example conditions: ['rep1','rep2','rep3','2.5Hz','3.5Hz','4.5Hz','5.5Hz','6.5Hz']
    '''

    if sample_range is None:
        sample_range = [100, 800]

    conditions = ['correct', 'incorrect']

    for condition in conditions:
        os.makedirs(os.path.join(main_save_model_folder, condition), exist_ok=True)

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])
    del subject_folders[20]  # subject 20 has only 5 incorrect trials, which is too few for the cross validation
    del subject_folders[5]  # subject 5 does not have behavioral data

    scores = [[] for _ in range(len(conditions))]
    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]

        trf_output, metadata = create_trf_output_from_epochs(raw_path, wav_folder=wav_folder, save_path=None, sample_range=sample_range, channel_type=channel_type, filt_range=filt_range)

        # convert correct and incorrect lists to strings
        correct = [str(correct_incorrect_dict[subject_path.name][0][i]).zfill(3) for i in range(len(correct_incorrect_dict[subject_path.name][0]))]
        incorrect = [str(correct_incorrect_dict[subject_path.name][1][i]).zfill(3) for i in range(len(correct_incorrect_dict[subject_path.name][1]))]
        correct_incorrect = [correct, incorrect]
        print('correct_incorrect')
        print(correct_incorrect)

        for j, condition in enumerate(conditions):
            # fit a trf on all epochs of the current condition (will be used for parameters plotting)
            run_existing_trf(trf_input_path, trf_output, metadata, os.path.join(main_save_model_folder, subject_path.name), save_model_path=os.path.join(main_save_model_folder, condition, subject_path.name), condition=correct_incorrect[j])
            # run a 10-fold cross-validation to estimate scores
            # score, score_env = run_leave_one_out_channel_trf(trf_input_path, trf_output, metadata, os.path.join(main_save_model_folder, subject_path.name), save=False, A_features=A_features, condition=[condition])
            score = run_channel_trf(trf_input_path, trf_output, metadata, os.path.join(main_save_model_folder, subject_path.name), save=False, condition=correct_incorrect[j])
            # scores[j][0].append(score)
            # scores[j][1].append(score_env)
            scores[j].append(score)

    for j, condition in enumerate(conditions):
        scores[j] = np.array(scores[j])
        # np.save(os.path.join(main_save_model_folder, condition, "epochs_channels_scores.npy"), scores[j])
        np.save(os.path.join(main_save_model_folder, condition, "channels_scores.npy"), scores[j])

    return


def run_trf_20channels_all_subjects(data_path, trf_input_path, wav_folder, save_model_folder, channel_types=None):

    raise NotImplementedError('Non functional since run_trf modification')

    if channel_types is None:
        channel_types = ['grad', 'mag']

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    scores = np.zeros((len(subject_folders), 2))
    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]

        # trf_output, metadata = create_trf_output_from_epochs(raw_path, wav_folder=wav_folder, save_path=None)
        trf_output, metadata = create_trf_output_from_epochs_channels_subset(raw_path, wav_folder=wav_folder, channel_types=channel_types, channel_nb=1, save_path=None)
        score, score_env = run_trf(trf_input_path, trf_output, metadata, save_model_path=None)
        scores[i, 0] = score
        scores[i, 1] = score_env

    np.save(os.path.join(save_model_folder, "scores_1channels.npy"), scores)
    #np.save(os.path.join(save_model_folder, "scores_20channels.npy"), scores)

    return


def compare_trfs(scores, individual_channels=False, residual_trf=False):

    scores = np.load(scores)
    print(scores.shape)
    if individual_channels:
        scores = np.mean(scores, axis=1)
    if not residual_trf:
        scores = np.flip(scores, axis=1)

    plt.switch_backend('QtAgg')
    ax = sns.boxplot(data=scores, showmeans=False, width=0.6, linewidth=2.5)
    ax = sns.swarmplot(data=scores, color=".25", size=7)
    ax.set_ylabel("Prediction Correlation (r)", fontsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    if residual_trf:
        ax.set_xticks(plt.xticks()[0], ['Residual TRF (predictive features)'], fontsize=11)
    else:
        ax.set_xticks(plt.xticks()[0], ['A (envelope/derivative/note_onset only)', 'AM (envelope/derivative/note_onset + predictive features)'], fontsize=11)
    plt.show()

    return


def run_channel_trf(input_path, output, metadata, pickle_path, save=True, condition=None):
    """
    Run channel level TRF on 1 subject.
    """

    X = np.load(input_path)
    list_to_keep = list_from_metadata(metadata)
    X = X[:, list_to_keep, :]
    if condition is not None:
        condition_list = condition_list_from_metadata(metadata, condition)
        X = X[:, condition_list, :]
    # X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))
    X = normalize_input_features(X)

    if isinstance(output, str):
        Y = np.load(output)
    else:
        Y = output
    if condition is not None:
        Y = Y[:, condition_list, :]
    print(X.shape)
    print(Y.shape)

    file = open(os.path.join(pickle_path, "rf.pickle"), 'rb')
    rf = pickle.load(file)
    file.close()

    n_folds = 6
    cv = KFold(n_folds, shuffle=True, random_state=39)

    score = np.zeros((n_folds, Y.shape[2]))

    for fold_number, (train, test) in enumerate(cv.split(np.moveaxis(X, 1, 0))):
        rf.fit(X[:, train, :], Y[:, train, :])
        score[fold_number, :] = rf.score(X[:, test, :], Y[:, test, :])

    score = np.mean(score, axis=0)

    if save:
        np.save(os.path.join(pickle_path, "channels_score.npy"), score)

    return score



def run_leave_one_out_channel_trf(input_path, output, metadata, pickle_path, save=True, A_features=None, condition=None):
    """
    Run channel level TRF on 1 subject. Leave-one-out procedure to keep the epochs dimension
    """

    if A_features is None:
        A_features = [0, 1]

    X = np.load(input_path)
    list_to_keep = list_from_metadata(metadata)
    X = X[:, list_to_keep, :]
    if condition is not None:
        condition_list = condition_list_from_metadata(metadata, condition)
        X = X[:, condition_list, :]
    # X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))
    X = normalize_input_features(X)

    if isinstance(output, str):
        Y = np.load(output)
    else:
        Y = output
    if condition is not None:
        Y = Y[:, condition_list, :]
    print(X.shape)
    print(Y.shape)
    X_env = X[:, :, A_features]

    n_epochs = X.shape[1]

    file = open(os.path.join(pickle_path, "rf.pickle"), 'rb')
    rf = pickle.load(file)
    file.close()
    file = open(os.path.join(pickle_path, "rf_env.pickle"), 'rb')
    rf_env = pickle.load(file)
    file.close()

    score = np.zeros((n_epochs, Y.shape[2]))
    score_env = np.zeros((n_epochs, Y.shape[2]))

    for epoch in range(n_epochs):
        mask = np.full(n_epochs, True)
        mask[epoch] = False
        rf.fit(X[:, mask, :], Y[:, mask, :])
        score[epoch, :] = rf.score(X[:, epoch, :], Y[:, epoch, :])

        # TRF based on sound envelope only
        rf_env.fit(X_env[:, mask, :], Y[:, mask, :])
        score_env[epoch, :] = rf_env.score(X_env[:, epoch, :], Y[:, epoch, :])

    if save:
        np.save(os.path.join(pickle_path, "epochs_channels_score.npy"), score)
        np.save(os.path.join(pickle_path, "epochs_channels_score_env.npy"), score_env)

    return score, score_env


def run_channel_trf_individual_alpha(input_path, output, metadata):
    """
    Run channel level TRF on 1 subject with one alpha per channel.
    """

    X = np.load(input_path)
    list_to_keep = list_from_metadata(metadata)
    X = X[:, list_to_keep, :]
    # X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))
    X = normalize_input_features(X)

    if isinstance(output, str):
        Y = np.load(output)
    else:
        Y = output
    print(X.shape)
    print(Y.shape)
    X_env = X[:, :, :2]

    feature_names = np.arange(X.shape[2])
    feature_names_env = np.arange(X_env.shape[2])

    alphas = np.logspace(-3, 8, 12)
    scores = np.zeros((alphas.shape[0], Y.shape[2]))
    scores_env = np.zeros((alphas.shape[0], Y.shape[2]))
    
    n_folds = 6
    cv = KFold(n_folds, shuffle=True, random_state=36)

    for ii, alpha in enumerate(alphas):
        print('alpha:', alpha)

        rf = ReceptiveField(-0.2, 0.5, 100, feature_names, estimator=alpha, scoring='corrcoef', n_jobs=-1)
        rf_env = ReceptiveField(-0.2, 0.5, 100, feature_names_env, estimator=alpha, scoring='corrcoef', n_jobs=-1)
        score = np.zeros((n_folds, Y.shape[2]))
        score_env = np.zeros((n_folds, Y.shape[2]))

        for fold_number, (train, test) in enumerate(cv.split(np.moveaxis(X, 1, 0))):
            rf.fit(X[:, train, :], Y[:, train, :])
            score[fold_number, :] = rf.score(X[:, test, :], Y[:, test, :])

            # TRF based on sound envelope only
            rf_env.fit(X_env[:, train, :], Y[:, train, :])
            score_env[fold_number, :] = rf_env.score(X_env[:, test, :], Y[:, test, :])

        # Now make predictions about the model output, given input stimuli.
        scores[ii, :] = np.mean(score, axis=0)
        scores_env[ii, :] = np.mean(score_env, axis=0)

    return np.max(scores, axis=0), np.max(scores_env, axis=0)



def epochs_channel_trf_all_subjects(data_path, trf_input_path, wav_folder, trf_folder, filt_range=None, A_features=None, channel_type='grad', sample_range=None):

    if A_features is None:
        A_features = [0, 1]
    if sample_range is None:
        sample_range = [100, 800]

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    scores = []
    scores_env = []
    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]

        trf_output, metadata = create_trf_output_from_epochs(raw_path, wav_folder=wav_folder, save_path=None, sample_range=sample_range, channel_type=channel_type, filt_range=filt_range)
        score, score_env = run_leave_one_out_channel_trf(trf_input_path, trf_output, metadata, pickle_path=os.path.join(trf_folder, subject_path.name), save=True, A_features=A_features)
        scores.append(score)
        scores_env.append(score_env)

    scores = np.array(scores)
    scores_env = np.array(scores_env)
    scores = np.stack((scores, scores_env), axis=-1)
    np.save(os.path.join(trf_folder, "epochs_channels_scores.npy"), scores)

    return


def trf_ridge_param_hist(data_path, trf_folder, rf_env=False):

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    alphas_list = []
    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)

        if rf_env:
            file = open(os.path.join(trf_folder, subject_path.name, "rf_env.pickle"), 'rb')
        else:
            file = open(os.path.join(trf_folder, subject_path.name, "rf.pickle"), 'rb')
        rf = pickle.load(file)
        file.close()

        alphas_list.append(rf.estimator)

    print(alphas_list)

    fig = plt.figure(figsize=(20, 12), dpi=300)
    # plt.xscale('log')
    sns.histplot(alphas_list, binwidth=1)
    plt.show()

    return


def plot_trf_topomaps(data_path, scores_path, trf_path, channel_type='grad', stat_test='permut', save_format='png'):
    '''
    scores_path: epochs scores
    '''

    scores = np.load(scores_path, allow_pickle=True)

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    # Plot average trf topomap
    raw_path = list(subject_folders[0].glob('**/*raw_sss.fif'))[0]
    raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
    raw.pick(['meg'])
    # channels_to_keep = mne.channel_indices_by_type(raw.info)[channel_type]
    # channels indices depending on channel type. This is specific to the trf analysis since gradiometers pairs have been combined

    averaged_epochs_scores = np.swapaxes(np.array([[np.mean(scores[i,j], axis=0) for j in range(scores.shape[1])] for i in range(scores.shape[0])]), 1, 2)  # Average over epochs for each participant
    average_scores = np.mean(averaged_epochs_scores, axis=0)  # Average over participants
    # if channel_type == 'grad':
    #     average_scores = merge_grad_data(average_scores)  # take mean of the two channels scores for each sensor
    mag_info = raw.copy().pick(['mag']).info  # Take Info of the magnetometers for channels positions
    # raw.pick([channel_type])

    # Find channels with significant score
    p_am = test_channels_significance(averaged_epochs_scores[:, :, 0], info=mag_info, test=stat_test, channel_type='mag')  # Permutation across subjects
    p_a = test_channels_significance(averaged_epochs_scores[:, :, 1], info=mag_info, test=stat_test, channel_type='mag')
    p_diff = test_channels_significance(averaged_epochs_scores[:, :, 0] - averaged_epochs_scores[:, :, 1], info=mag_info, test=stat_test, channel_type='mag')

    p_new = [p_am, p_a, p_diff]

    # define the statistical threshold (alpha)
    alpha = 0.05

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 12), dpi=300)
    vmax = max(np.max(np.abs(average_scores[:, 0])), np.max(np.abs(average_scores[:, 1])))
    im, cn = mne.viz.plot_topomap(average_scores[:, 0], mag_info, show=False, axes=axes[0, 0], ch_type='mag', mask=p_new[0] <= alpha, vmin=0, vmax=vmax)
    axes[0, 0].set_title("AM")
    clb = fig.colorbar(im, cax=axes[0, 1])
    clb.ax.set_title("Prediction Correlation (r)", fontsize=15)  # title on top of colorbar
    axes[0, 1].set_box_aspect(5)
    axes[0, 1].set_anchor('W')
    axes[0, 0].set_anchor('E')
    im, cn = mne.viz.plot_topomap(average_scores[:, 1], mag_info, show=False, axes=axes[1, 0], ch_type='mag', mask=p_new[1] <= alpha, vmin=0, vmax=vmax)
    axes[1, 0].set_title("A")
    clb = fig.colorbar(im, cax=axes[1, 1])
    axes[1, 1].set_box_aspect(5)
    axes[1, 1].set_anchor('W')
    axes[1, 0].set_anchor('E')
    im, cn = mne.viz.plot_topomap(average_scores[:, 0] - average_scores[:, 1], mag_info, show=False, axes=axes[2, 0],
                                  ch_type='mag', mask=p_new[2] <= alpha)
    axes[2, 0].set_title("AM - A")
    clb = fig.colorbar(im, cax=axes[2, 1])
    axes[2, 1].set_box_aspect(5)
    axes[2, 1].set_anchor('W')
    axes[2, 0].set_anchor('E')

    fig.savefig(os.path.join(trf_path, 'trf_topomaps_average_' + channel_type + '_' + stat_test + '.' + save_format))

    # individual trf topomaps
    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
        raw.pick(['meg'])

        # channels_to_keep = mne.channel_indices_by_type(raw.info)[channel_type]
        score = averaged_epochs_scores[i, :, :]
        # if channel_type == 'grad':
        #     score = merge_grad_data(score)
        mag_info = raw.copy().pick(['mag']).info  # Take Info of the magnetometers for channels positions
        # raw.pick([channel_type])

        # Find channels with significant score
        p_am = test_channels_significance(scores[i, 0], info=mag_info, test=stat_test, channel_type='mag')  # Permutation across epochs
        p_a = test_channels_significance(scores[i, 1], info=mag_info, test=stat_test, channel_type='mag')
        p_diff = test_channels_significance(scores[i, 0] - scores[i, 1], info=mag_info, test=stat_test, channel_type='mag')

        p_new = [p_am, p_a, p_diff]

        # plot topomap TRF AM
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 12), dpi=300)
        vmax = max(np.max(np.abs(average_scores[:, 0])), np.max(np.abs(average_scores[:, 1])))
        im, cn = mne.viz.plot_topomap(score[:, 0], mag_info, show=False, axes=axes[0, 0], ch_type='mag', mask=p_new[0] <= alpha, vmin=0, vmax=vmax)
        axes[0, 0].set_title("AM")
        clb = fig.colorbar(im, cax=axes[0, 1])
        clb.ax.set_title("Prediction Correlation (r)", fontsize=15) # title on top of colorbar
        axes[0, 1].set_box_aspect(5)
        axes[0, 1].set_anchor('W')
        axes[0, 0].set_anchor('E')
        im, cn = mne.viz.plot_topomap(score[:, 1], mag_info, show=False, axes=axes[1, 0], ch_type='mag', mask=p_new[1] <= alpha, vmin=0, vmax=vmax)
        axes[1, 0].set_title("A")
        clb = fig.colorbar(im, cax=axes[1, 1])
        axes[1, 1].set_box_aspect(5)
        axes[1, 1].set_anchor('W')
        axes[1, 0].set_anchor('E')
        im, cn = mne.viz.plot_topomap(score[:, 0] - score[:, 1], mag_info, show=False, axes=axes[2, 0], ch_type='mag', mask=p_new[2] <= alpha)
        axes[2, 0].set_title("AM - A")
        clb = fig.colorbar(im, cax=axes[2, 1])
        axes[2, 1].set_box_aspect(5)
        axes[2, 1].set_anchor('W')
        axes[2, 0].set_anchor('E')

        plt.savefig(os.path.join(trf_path, subject_path.parts[-1], 'trf_topomaps_' + channel_type + '_' + stat_test + '.' + save_format))
        plt.close()

    return


def plot_trf_topomaps_all_subjects_only(data_path, scores_path, trf_path, channel_type='grad', stat_test='permut', save_format='png'):
    '''
    Same as plot_trf_topomaps but only across subjects (no individual subjects analysis), so that the epochs scores are not necessary.
    scores_path: channels scores
    '''

    scores = np.load(scores_path, allow_pickle=True)

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    # Plot average trf topomap
    raw_path = list(subject_folders[0].glob('**/*raw_sss.fif'))[0]
    raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
    raw.pick(['meg'])
    # channels_to_keep = mne.channel_indices_by_type(raw.info)[channel_type]
    # channels indices depending on channel type. This is specific to the trf analysis since gradiometers pairs have been combined

    average_scores = np.mean(scores, axis=0)  # Average over participants
    # if channel_type == 'grad':
    #     average_scores = merge_grad_data(average_scores)  # take mean of the two channels scores for each sensor
    mag_info = raw.copy().pick(['mag']).info  # Take Info of the magnetometers for channels positions
    # raw.pick([channel_type])

    # Find channels with significant score
    p_am = test_channels_significance(scores[:, :, 0], info=mag_info, test=stat_test, channel_type='mag')  # Permutation across subjects
    p_a = test_channels_significance(scores[:, :, 1], info=mag_info, test=stat_test, channel_type='mag')
    p_diff = test_channels_significance(scores[:, :, 0] - scores[:, :, 1], info=mag_info, test=stat_test, channel_type='mag')

    p_new = [p_am, p_a, p_diff]

    # define the statistical threshold (alpha)
    alpha = 0.05

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 12), dpi=300)
    vmax = max(np.max(np.abs(average_scores[:, 0])), np.max(np.abs(average_scores[:, 1])))
    im, cn = mne.viz.plot_topomap(average_scores[:, 0], mag_info, show=False, axes=axes[0, 0], ch_type='mag', mask=p_new[0] <= alpha, vmin=0, vmax=vmax)
    axes[0, 0].set_title("AM")
    clb = fig.colorbar(im, cax=axes[0, 1])
    clb.ax.set_title("Prediction Correlation (r)", fontsize=15)  # title on top of colorbar
    axes[0, 1].set_box_aspect(5)
    axes[0, 1].set_anchor('W')
    axes[0, 0].set_anchor('E')
    im, cn = mne.viz.plot_topomap(average_scores[:, 1], mag_info, show=False, axes=axes[1, 0], ch_type='mag', mask=p_new[1] <= alpha, vmin=0, vmax=vmax)
    axes[1, 0].set_title("A")
    clb = fig.colorbar(im, cax=axes[1, 1])
    axes[1, 1].set_box_aspect(5)
    axes[1, 1].set_anchor('W')
    axes[1, 0].set_anchor('E')
    im, cn = mne.viz.plot_topomap(average_scores[:, 0] - average_scores[:, 1], mag_info, show=False, axes=axes[2, 0],
                                  ch_type='mag', mask=p_new[2] <= alpha)
    axes[2, 0].set_title("AM - A")
    clb = fig.colorbar(im, cax=axes[2, 1])
    axes[2, 1].set_box_aspect(5)
    axes[2, 1].set_anchor('W')
    axes[2, 0].set_anchor('E')

    fig.savefig(os.path.join(trf_path, 'trf_topomaps_average_' + channel_type + '_' + stat_test + '.' + save_format))
    plt.close()

    return

def plot_trf_correlation_tscore_topomap_all_subjects(data_path, scores_path, trf_path, channel_type='grad', stat_test='permut', save_format='png'):
    '''
    Same as plot_trf_topomaps_all_subjects_only, but with only one condition (no AM, A and AM-A).
    scores_path: channels scores
    '''

    scores = np.load(scores_path, allow_pickle=True)
    scores = np.arctanh(scores)  # Fisher z-transform

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    # Plot average trf topomap
    raw_path = list(subject_folders[0].glob('**/*raw_sss.fif'))[0]
    raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
    raw.pick(['meg'])
    # channels_to_keep = mne.channel_indices_by_type(raw.info)[channel_type]
    # channels indices depending on channel type. This is specific to the trf analysis since gradiometers pairs have been combined

    average_scores = np.mean(scores, axis=0)  # Average over participants
    # if channel_type == 'grad':
    #     average_scores = merge_grad_data(average_scores)  # take mean of the two channels scores for each sensor
    mag_info = raw.copy().pick(['mag']).info  # Take Info of the magnetometers for channels positions
    # raw.pick([channel_type])

    # Find channels with significant score
    p_value, t_obs = test_channels_significance(scores[:, :, 0], info=mag_info, test=stat_test, channel_type='mag', return_t_obs=True)  # Permutation across subjects

    # define the statistical threshold (alpha)
    alpha = 0.05

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 12), dpi=300)
    # im, cn = mne.viz.plot_topomap(average_scores[:, 0], mag_info, show=False, axes=axes[0], ch_type='mag', mask=p_value <= alpha)  # plot correlation scores
    im, cn = mne.viz.plot_topomap(t_obs, mag_info, show=False, axes=axes[0], ch_type='mag', mask=p_value <= alpha)  # plot t-scores
    # axes[0, 0].set_title("AM")
    clb = fig.colorbar(im, cax=axes[1])
    clb.ax.set_title("t scores of prediction correlation (r)", fontsize=15)  # title on top of colorbar
    axes[1].set_box_aspect(5)
    axes[1].set_anchor('W')
    axes[0].set_anchor('E')

    fig.savefig(os.path.join(trf_path, 'trf_tscore_topomap_' + channel_type + '_' + stat_test + '.' + save_format))
    plt.close()

    return

def plot_trf_topomaps_condition_difference(data_path, scores_path, scores_path_2, save_path, scores_path_optional=None,
                                           conditions=None, channel_type='grad', stat_test='permut', save_format='png', plot_name='rep'):
    '''
    No need for epochs scores, this functions uses channels_scores
    scores_path_optional: if given, the second condition is the average score between scores_2 and scores_optional
    '''

    if conditions is None:
        conditions = ['Rep 1', 'Rep 3']

    scores = np.load(scores_path, allow_pickle=True)
    scores = np.arctanh(scores)  # Fisher z-transform
    scores_2 = np.load(scores_path_2, allow_pickle=True)  # Fisher z-transform performed later

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    # Plot average trf topomap
    raw_path = list(subject_folders[0].glob('**/*raw_sss.fif'))[0]
    raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
    raw.pick(['meg'])
    # channels_to_keep = mne.channel_indices_by_type(raw.info)[channel_type]

    # averaged_epochs_scores = np.swapaxes(
    #     np.array([[np.mean(scores[i, j], axis=0) for j in range(scores.shape[1])] for i in range(scores.shape[0])]), 1,
    #     2)  # Average over epochs for each participant
    # averaged_epochs_scores_2 = np.swapaxes(
    #     np.array([[np.mean(scores_2[i, j], axis=0) for j in range(scores_2.shape[1])] for i in range(scores_2.shape[0])]), 1,
    #     2)

    # optional 3rd condition
    if scores_path_optional is not None:
        scores_optional = np.load(scores_path_optional, allow_pickle=True)
        # averaged_epochs_scores_optional = np.swapaxes(
        #     np.array([[np.mean(scores_optional[i, j], axis=0) for j in range(scores_optional.shape[1])] for i in range(scores_optional.shape[0])]), 1, 2)
        # averaged_epochs_scores_2 = (averaged_epochs_scores_2 + averaged_epochs_scores_optional) / 2
        scores_2 = (scores_2 + scores_optional) / 2
    scores_2 = np.arctanh(scores_2)  # Fisher z-transform

    # average_scores = np.mean(averaged_epochs_scores[:, channels_to_keep, :], axis=0)  # Average over participants
    # average_scores_2 = np.mean(averaged_epochs_scores_2[:, channels_to_keep, :], axis=0)
    average_scores = np.mean(scores, axis=0)  # Average over participants
    average_scores_2 = np.mean(scores_2, axis=0)
    # if channel_type == 'grad':
    #     average_scores = merge_grad_data(average_scores)  # take mean of the two channels scores for each sensor
    #     average_scores_2 = merge_grad_data(average_scores_2)  # take mean of the two channels scores for each sensor
    mag_info = raw.copy().pick(['mag']).info  # Take Info of the magnetometers for channels positions
    # raw.pick([channel_type])

    # Find channels with significant score
    # p_am = test_channels_significance(averaged_epochs_scores[:, channels_to_keep, 0] - averaged_epochs_scores[:, channels_to_keep, 1], info=raw.info, test=stat_test,
    p_am, t_obs = test_channels_significance(scores, info=mag_info, test=stat_test,
                                      channel_type='mag', tail=1, return_t_obs=True)  # Permutation across subjects
    # p_am_2 = test_channels_significance(averaged_epochs_scores_2[:, channels_to_keep, 0] - averaged_epochs_scores_2[:, channels_to_keep, 1], info=raw.info, test=stat_test,
    p_am_2, t_obs_2 = test_channels_significance(scores_2, info=mag_info, test=stat_test,
                                     channel_type='mag', tail=1, return_t_obs=True)
    p_diff, t_obs_diff = test_channels_significance(
        # (averaged_epochs_scores[:, channels_to_keep, 0] - averaged_epochs_scores[:, channels_to_keep, 1]) -
        # (averaged_epochs_scores_2[:, channels_to_keep, 0] - averaged_epochs_scores_2[:, channels_to_keep, 1]), info=raw.info,
        scores - scores_2, info=mag_info,
        test=stat_test, channel_type='mag', tail=0, return_t_obs=True)  # difference test, so two-tail is used

    p_new = [p_am, p_am_2, p_diff]

    # define the statistical threshold (alpha)
    alpha = 0.05

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 12), dpi=300)
    vmax = max(np.max(np.abs(t_obs)), np.max(np.abs(t_obs_2)))
    # im, cn = mne.viz.plot_topomap(average_scores[:, 0] - average_scores[:, 1], mag_info, show=False, axes=axes[0, 0], ch_type='mag',
    im, cn = mne.viz.plot_topomap(t_obs, mag_info, show=False, axes=axes[0, 0], ch_type='mag',
                                  mask=p_new[0] <= alpha)
    axes[0, 0].set_title(conditions[0])
    clb = fig.colorbar(im, cax=axes[0, 1])
    clb.ax.set_title("t scores of prediction correlation (r)", fontsize=15)  # title on top of colorbar
    axes[0, 1].set_box_aspect(5)
    axes[0, 1].set_anchor('W')
    axes[0, 0].set_anchor('E')
    # im, cn = mne.viz.plot_topomap(average_scores_2[:, 0] - average_scores_2[:, 1], mag_info, show=False, axes=axes[1, 0], ch_type='mag',
    im, cn = mne.viz.plot_topomap(t_obs_2, mag_info, show=False, axes=axes[1, 0], ch_type='mag',
                                  mask=p_new[1] <= alpha)
    axes[1, 0].set_title(conditions[1])
    clb = fig.colorbar(im, cax=axes[1, 1])
    axes[1, 1].set_box_aspect(5)
    axes[1, 1].set_anchor('W')
    axes[1, 0].set_anchor('E')
    # im, cn = mne.viz.plot_topomap((average_scores[:, 0] - average_scores[:, 1]) - (average_scores_2[:, 0] - average_scores_2[:, 1]), mag_info, show=False, axes=axes[2, 0],
    im, cn = mne.viz.plot_topomap(t_obs_diff, mag_info, show=False, axes=axes[2, 0],
                                  ch_type='mag', mask=p_new[2] <= alpha)
    axes[2, 0].set_title(conditions[0] + " - " + conditions[1])
    clb = fig.colorbar(im, cax=axes[2, 1])
    clb.ax.set_title("t scores of difference", fontsize=15)  # title on top of colorbar
    axes[2, 1].set_box_aspect(5)
    axes[2, 1].set_anchor('W')
    axes[2, 0].set_anchor('E')

    # fig.savefig(os.path.join(save_path, 'trf_topomaps_condition_difference_' + channel_type + '.' + save_format))
    fig.savefig(os.path.join(save_path, 'trf_topomaps_' + plot_name + '_' + channel_type + '_' + stat_test + '.' + save_format))
    plt.close()

    return


def plot_trf_topomaps_individual_subjects(data_path, scores_path, trf_path, channel_type='grad', stat_test='permut', save_format='png'):

    scores = np.load(scores_path, allow_pickle=True)
    averaged_epochs_scores = np.swapaxes(np.array([[np.mean(scores[i, j], axis=0) for j in range(scores.shape[1])] for i in range(scores.shape[0])]), 1,2)  # Average over epochs for each participant

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    # define the statistical threshold (alpha)
    alpha = 0.05

    # individual trf topomaps
    fig, axes = plt.subplots(nrows=6, ncols=10, figsize=(20, 12), dpi=300)
    vmax = np.max(np.abs(averaged_epochs_scores[:, :, 0] - averaged_epochs_scores[:, :, 1]))
    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
        raw.pick(['meg'])

        # channels_to_keep = mne.channel_indices_by_type(raw.info)[channel_type]
        score = averaged_epochs_scores[i, :, :]
        # if channel_type == 'grad':
        #     score = merge_grad_data(score)
        mag_info = raw.copy().pick(['mag']).info  # Take Info of the magnetometers for channels positions
        # raw.pick([channel_type])

        # Find channels with significant score
        p_diff = test_channels_significance(scores[i, 0] - scores[i, 1],
                                            info=mag_info, test=stat_test, channel_type='mag')

        # plot topomap TRF AM-A only
        im, cn = mne.viz.plot_topomap(score[:, 0] - score[:, 1], mag_info, show=False, axes=axes.flat[2*i],
                                      ch_type='mag', mask=p_diff <= alpha, vmin=-vmax, vmax=vmax)
        axes.flat[2*i].set_title("AM - A")
        clb = fig.colorbar(im, cax=axes.flat[2*i+1])
        axes.flat[2*i+1].set_box_aspect(5)
        axes.flat[2*i+1].set_anchor('W')
        axes.flat[2*i].set_anchor('E')

    for i in range(len(subject_folders), 30):
        axes.flat[2 * i].axis('off')
        axes.flat[2 * i + 1].axis('off')

    plt.savefig(os.path.join(trf_path, 'trf_topomaps_AM-A_all_subjects_' + channel_type + '_' + stat_test + '.' + save_format))
    plt.close()

    return


def plot_trf_kernels(data_path, trf_folder, features=None, channel_type='grad', acoust_feat=False, save_format='png'):

    if features is None:
        features = ['Envelope', 'Envelope derivative', 'Surprise', 'Entropy']

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    coef_list = []
    coef_list_env = []

    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)

        file = open(os.path.join(trf_folder, subject_path.name, "rf.pickle"), 'rb')
        rf = pickle.load(file)
        file.close()

        fig, axes = plt.subplots(nrows=rf.coef_.shape[1], ncols=1, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
        for j in range(rf.coef_.shape[1]):
            axes[j].plot(rf.delays_ / rf.sfreq, rf.coef_[:, j, :].T)
            axes[j].axvline(0, linestyle='--', color='k')
            axes[j].set_title(features[j])
        plt.xlabel("Time (s)")
        fig.savefig(os.path.join(trf_folder, subject_path.name, 'trf_kernels_' + channel_type + '.' + save_format))
        plt.close()

        coef_list.append(rf.coef_)
        if acoust_feat:
            file = open(os.path.join(trf_folder, subject_path.name, "rf_env.pickle"), 'rb')
            rf_env = pickle.load(file)
            file.close()
            coef_list_env.append(rf_env.coef_)

    coef_list = np.array(coef_list)
    coefs = np.mean(coef_list, axis=0)

    fig, axes = plt.subplots(nrows=rf.coef_.shape[1], ncols=1, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        axes[j].plot(rf.delays_ / rf.sfreq, coefs[:, j, :].T)
        axes[j].axvline(0, linestyle='--')
        axes[j].set_title(features[j])
    plt.xlabel("Time (s)")
    fig.savefig(os.path.join(trf_folder, 'trf_kernels_average_' + channel_type + '.' + save_format))

    if acoust_feat:
        coef_list_env = np.array(coef_list_env)
        coefs_env = np.mean(coef_list_env, axis=0)
        fig, axes = plt.subplots(nrows=rf_env.coef_.shape[1], ncols=1, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
        for j in range(rf_env.coef_.shape[1]):
            axes[j].plot(rf_env.delays_ / rf_env.sfreq, coefs_env[:, j, :].T)
            axes[j].axvline(0, linestyle='--')
            axes[j].set_title(features[j])
        plt.xlabel("Time (s)")
        fig.savefig(os.path.join(trf_folder, 'trf_env_kernels_average_' + channel_type + '.' + save_format))

    return


def plot_significant_trf_kernels(data_path, scores_path, trf_folder, features=None, channel_type='grad', save_format='png'):

    if features is None:
        features = ['Envelope', 'Envelope derivative', 'Surprise', 'Entropy']

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    sign_channels = get_significant_channels_names(data_path, scores_path, channel_type=channel_type, stat_test='permut')

    coef_list = []

    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
        raw.pick(['meg'])
        raw.pick([channel_type])

        file = open(os.path.join(trf_folder, subject_path.name, "rf.pickle"), 'rb')
        rf = pickle.load(file)
        file.close()

        coef_list.append(rf.coef_[mne.channel_indices_by_type(raw.info)[channel_type], :, :][sign_channels, :, :])

    coef_list = np.array(coef_list)
    coefs = np.mean(coef_list, axis=0)

    fig, axes = plt.subplots(nrows=rf.coef_.shape[1], ncols=1, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        axes[j].plot(rf.delays_ / rf.sfreq, coefs[:, j, :].T)
        axes[j].axvline(0, linestyle='--')
        axes[j].set_title(features[j])
    plt.xlabel("Time (s)")
    fig.savefig(os.path.join(trf_folder, 'trf_kernels_average_' + channel_type + '.' + save_format))

    return


def plot_trf_kernels_best_channels(data_path, scores_path, trf_folder, features=None, channel_type='grad', nb_channels=5, acoust_feat=False, indiv_subjects=True, ncols=2, save_format='png'):
    '''
    ['Envelope', 'Envelope derivative', 'Surprise on', 'Surprise off', 'Surprise velocity', 'Surprise shift', 'Entropy on', 'Entropy off', 'Entropy velocity', 'Entropy shift']
    '''

    if features is None:
        features = ['Envelope', 'Envelope derivative', 'Surprise', 'Entropy']

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    scores = np.load(scores_path, allow_pickle=True)

    coef_list = []
    coef_list_env = []

    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)

        file = open(os.path.join(trf_folder, subject_path.name, "rf.pickle"), 'rb')
        rf = pickle.load(file)
        file.close()

        # score = np.mean(scores[i, 0], axis=0)[channels_to_keep]  # Average over epochs and select channel type
        score = scores[i, :, 0]  # select channel type
        best_chans = np.flip(np.argsort(score))[:nb_channels]

        fig, axes = plt.subplots(nrows=ceil(rf.coef_.shape[1] / ncols), ncols=ncols, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
        for j in range(rf.coef_.shape[1]):

            # Flip kernels for maximum correlation
            if nb_channels > 1:
                signs = best_flip_combination(rf.coef_[best_chans, j, :])
                print(signs)
                rf.coef_[best_chans, j, :] = signs[:, np.newaxis] * rf.coef_[best_chans, j, :]

            lines = axes.T.flat[j].plot(rf.delays_ / rf.sfreq, rf.coef_[best_chans, j, :].T, 'lightgray', linewidth=2)
            axes.T.flat[j].axvline(0, linestyle='--', color='k')
            axes.T.flat[j].axhline(0, linestyle='--', color='k')
            line_average = axes.T.flat[j].plot(rf.delays_ / rf.sfreq, np.mean(rf.coef_[best_chans, j, :], axis=0).T, 'dimgrey', linewidth=4)
            if j==0:
                axes.T.flat[j].legend([lines[0], line_average[0]], ['Best channels', 'Average'])
            axes.T.flat[j].set_title(features[j])
        plt.xlabel("Time (s)")
        fig.savefig(os.path.join(trf_folder, subject_path.name, 'trf_kernels_'+str(nb_channels)+'best_channels_' + channel_type + '.' + save_format))
        plt.close()

        coef_list.append(np.mean(rf.coef_[best_chans, :, :], axis=0))

        if acoust_feat:
            file = open(os.path.join(trf_folder, subject_path.name, "rf_env.pickle"), 'rb')
            rf_env = pickle.load(file)
            file.close()
            score_env = np.mean(scores[i, 1], axis=0)  # Average over epochs and select channel type
            best_chans_env = np.flip(np.argsort(score_env))[:nb_channels]
            if nb_channels > 1:
                for j in range(rf_env.coef_.shape[1]):
                    signs = best_flip_combination(rf_env.coef_[best_chans_env, j, :])
                    rf_env.coef_[best_chans_env, j, :] = signs[:, np.newaxis] * rf_env.coef_[best_chans_env, j, :]
            coef_list_env.append(np.mean(rf_env.coef_[best_chans_env, :, :], axis=0))

    coef_list = np.array(coef_list)

    # Flip kernels for maximum correlation across subjects
    for j in range(rf.coef_.shape[1]):
        if coef_list.shape[0] > 1:
            signs = best_flip_combination_subjects(coef_list[:, j, :])
            print(signs)
            coef_list[:, j, :] = signs[:, np.newaxis] * coef_list[:, j, :]

    coefs_average = np.mean(coef_list, axis=0)

    fig, axes = plt.subplots(nrows=ceil(rf.coef_.shape[1] / ncols), ncols=ncols, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        if indiv_subjects:
            lines = axes.T.flat[j].plot(rf.delays_ / rf.sfreq, coef_list[:, j, :].T, 'lightgray', linewidth=2)
        else:
            # plot sem
            sem = np.std(coef_list[:, j, :], axis=0) / np.sqrt(coef_list.shape[0])
            axes.T.flat[j].fill_between(rf.delays_ / rf.sfreq, coefs_average[j, :] - sem, coefs_average[j, :] + sem, color=['lightgray'], alpha=0.6)
        axes.T.flat[j].axvline(0, linestyle='--', color='k')
        axes.T.flat[j].axhline(0, linestyle='--', color='k')
        line_average = axes.T.flat[j].plot(rf.delays_ / rf.sfreq, coefs_average[j, :], 'dimgrey', linewidth=4)
        if j==0:
            if indiv_subjects:
                axes.T.flat[j].legend([lines[0], line_average[0]], ['Subject', 'Average'])
        axes.T.flat[j].set_title(features[j])
    plt.xlabel("Time (s)")
    if indiv_subjects:
        fig.savefig(os.path.join(trf_folder, 'trf_kernels_'+str(nb_channels)+'best_channels_' + channel_type + '.' + save_format))
    else:
        fig.savefig(os.path.join(trf_folder, 'trf_kernels_'+str(nb_channels)+'best_channels_' + channel_type + '_avg' +'.' + save_format))

    if acoust_feat:
        coef_list_env = np.array(coef_list_env)
        coefs_average_env = np.mean(coef_list_env, axis=0)
        fig, axes = plt.subplots(nrows=int(rf_env.coef_.shape[1] / 2), ncols=2, sharex=True, sharey=False, figsize=(20, 12), dpi=100)
        for j in range(rf_env.coef_.shape[1]):
            lines = axes.T.flat[j].plot(rf_env.delays_ / rf_env.sfreq, coef_list_env[:, j, :].T, 'lightgray', linewidth=2)
            axes.T.flat[j].axvline(0, linestyle='--', color='k')
            axes.T.flat[j].axhline(0, linestyle='--', color='k')
            line_average = axes.T.flat[j].plot(rf_env.delays_ / rf_env.sfreq, coefs_average_env[j, :].T, 'dimgrey', linewidth=4)
            axes.T.flat[j].legend([lines[0], line_average[0]], ['Subject', 'Average'])
            axes.T.flat[j].set_title(features[j])
        plt.xlabel("Time (s)")
        fig.savefig(os.path.join(trf_folder, 'trf_env_kernels_'+str(nb_channels)+'best_channels_' + channel_type + '.' + save_format))
    plt.close('all')

    return


def plot_trf_kernels_all_channels(data_path, trf_folder, features=None, channel_type='grad', acoust_feat=False, indiv_subjects=True, ncols=2, zscore=False, save_format='png'):
    '''
    ['Envelope', 'Envelope derivative', 'Surprise on', 'Surprise off', 'Surprise velocity', 'Surprise shift', 'Entropy on', 'Entropy off', 'Entropy velocity', 'Entropy shift']
    '''

    if features is None:
        features = ['Envelope', 'Envelope derivative', 'Surprise', 'Entropy']

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    coef_list = []
    coef_list_env = []

    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)

        file = open(os.path.join(trf_folder, subject_path.name, "rf.pickle"), 'rb')
        rf = pickle.load(file)
        file.close()

        fig, axes = plt.subplots(nrows=ceil(rf.coef_.shape[1] / ncols), ncols=ncols, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
        for j in range(rf.coef_.shape[1]):

            lines = axes.T.flat[j].plot(rf.delays_ / rf.sfreq, np.abs(rf.coef_[:, j, :]).T, 'lightgray', linewidth=2)
            axes.T.flat[j].axvline(0, linestyle='--', color='k')
            # axes.T.flat[j].axhline(0, linestyle='--', color='k')
            line_average = axes.T.flat[j].plot(rf.delays_ / rf.sfreq, np.sqrt(np.mean(rf.coef_[:, j, :]**2, axis=0)).T, 'dimgrey', linewidth=4)
            if j==0:
                axes.T.flat[j].legend([lines[0], line_average[0]], ['All channels', 'Average'])
            axes.T.flat[j].set_title(features[j])
        plt.xlabel("Time (s)")
        fig.savefig(os.path.join(trf_folder, subject_path.name, 'trf_kernels_all_channels_' + channel_type + '.' + save_format))
        plt.close()

        if zscore:
            rf.coef_ = (rf.coef_ - np.mean(rf.coef_)) / np.std(rf.coef_)
        coef_list.append(np.sqrt(np.mean(rf.coef_**2, axis=0)))

        if acoust_feat:
            file = open(os.path.join(trf_folder, subject_path.name, "rf_env.pickle"), 'rb')
            rf_env = pickle.load(file)
            file.close()
            if zscore:
                rf_env.coef_ = (rf_env.coef_ - np.mean(rf_env.coef_)) / np.std(rf_env.coef_)
            coef_list_env.append(np.sqrt(np.mean(rf_env.coef_**2, axis=0)))

    coef_list = np.array(coef_list)
    coefs_average = np.mean(coef_list, axis=0)

    fig, axes = plt.subplots(nrows=ceil(rf.coef_.shape[1] / ncols), ncols=ncols, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        if indiv_subjects:
            lines = axes.T.flat[j].plot(rf.delays_ / rf.sfreq, coef_list[:, j, :].T, 'lightgray', linewidth=2)
        else:
            # plot sem
            sem = np.std(coef_list[:, j, :], axis=0) / np.sqrt(coef_list.shape[0])
            axes.T.flat[j].fill_between(rf.delays_ / rf.sfreq, coefs_average[j, :] - sem, coefs_average[j, :] + sem, color=['lightgray'], alpha=0.6)
        axes.T.flat[j].axvline(0, linestyle='--', color='k')
        # axes.T.flat[j].axhline(0, linestyle='--', color='k')
        line_average = axes.T.flat[j].plot(rf.delays_ / rf.sfreq, coefs_average[j, :], 'dimgrey', linewidth=4)
        if j==0:
            if indiv_subjects:
                axes.T.flat[j].legend([lines[0], line_average[0]], ['Subject', 'Average'])
        axes.T.flat[j].set_title(features[j])
    plt.xlabel("Time (s)")
    if indiv_subjects:
        fig.savefig(os.path.join(trf_folder, 'trf_kernels_all_channels_' + channel_type + '.' + save_format))
    else:
        fig.savefig(os.path.join(trf_folder, 'trf_kernels_all_channels_' + channel_type + '_avg' +'.' + save_format))

    if acoust_feat:
        coef_list_env = np.array(coef_list_env)
        coefs_average_env = np.mean(coef_list_env, axis=0)
        fig, axes = plt.subplots(nrows=int(rf_env.coef_.shape[1] / 2), ncols=2, sharex=True, sharey=False, figsize=(20, 12), dpi=100)
        for j in range(rf_env.coef_.shape[1]):
            lines = axes.T.flat[j].plot(rf_env.delays_ / rf_env.sfreq, coef_list_env[:, j, :].T, 'lightgray', linewidth=2)
            axes.T.flat[j].axvline(0, linestyle='--', color='k')
            axes.T.flat[j].axhline(0, linestyle='--', color='k')
            line_average = axes.T.flat[j].plot(rf_env.delays_ / rf_env.sfreq, coefs_average_env[j, :].T, 'dimgrey', linewidth=4)
            axes.T.flat[j].legend([lines[0], line_average[0]], ['Subject', 'Average'])
            axes.T.flat[j].set_title(features[j])
        plt.xlabel("Time (s)")
        fig.savefig(os.path.join(trf_folder, 'trf_env_kernels_all_channels_' + channel_type + '.' + save_format))
    plt.close('all')

    return


def plot_trf_kernels_significant_channels(data_path, scores_path, trf_folder, features=None, channel_type='grad', indiv_subjects=True, ncols=2, zscore=False, rms=False, stat_test='permut_cluster', save_format='png'):
    '''
    ['Envelope', 'Envelope derivative', 'Note onset', 'Surprise on', 'Surprise off', 'Surprise velocity', 'Surprise shift', 'Entropy on', 'Entropy off', 'Entropy velocity', 'Entropy shift']
    '''

    if features is None:
        features = ['Envelope', 'Envelope derivative', 'Surprise', 'Entropy']

    scores = np.load(scores_path, allow_pickle=True)
    scores = np.arctanh(scores)  # Fisher z-transform

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    raw_path = list(subject_folders[0].glob('**/*raw_sss.fif'))[0]
    raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
    raw.pick(['meg'])
    mag_info = raw.copy().pick(['mag']).info  # Take Info of the magnetometers for channels positions

    # Find channels with significant score
    p_value = test_channels_significance(scores[:, :, 0], info=mag_info, test=stat_test, channel_type='mag',
                                                return_t_obs=False)  # Permutation across subjects

    # define the statistical threshold and significant channels
    alpha = 0.05
    significant_channels = p_value < alpha

    # If no significant channel, do not plot anything
    if np.sum(significant_channels) == 0:
        print('No significant channel')
        return

    coef_list = []

    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)

        file = open(os.path.join(trf_folder, subject_path.name, "rf.pickle"), 'rb')
        rf = pickle.load(file)
        file.close()

        fig, axes = plt.subplots(nrows=ceil(rf.coef_.shape[1] / ncols), ncols=ncols, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
        for j in range(rf.coef_.shape[1]):
            if rms:
                lines = axes.T.flat[j].plot(rf.delays_ / rf.sfreq, np.abs(rf.coef_[significant_channels, j, :]).T, 'lightgray', linewidth=2)  # TODO: lines should be appended, not assigned (same for all functions)
                axes.T.flat[j].axvline(0, linestyle='--', color='k')
                line_average = axes.T.flat[j].plot(rf.delays_ / rf.sfreq, np.sqrt(np.mean(rf.coef_[significant_channels, j, :]**2, axis=0)).T, 'dimgrey', linewidth=4)
            else:
                # Flip kernels for maximum correlation
                signs = best_flip_combination(rf.coef_[significant_channels, j, :])
                rf.coef_[significant_channels, j, :] = signs[:, np.newaxis] * rf.coef_[significant_channels, j, :]
                lines = axes.T.flat[j].plot(rf.delays_ / rf.sfreq, rf.coef_[significant_channels, j, :].T, 'lightgray', linewidth=2)
                axes.T.flat[j].axvline(0, linestyle='--', color='k')
                axes.T.flat[j].axhline(0, linestyle='--', color='k')
                line_average = axes.T.flat[j].plot(rf.delays_ / rf.sfreq, np.mean(rf.coef_[significant_channels, j, :], axis=0).T, 'dimgrey', linewidth=4)
            if j==0:
                axes.T.flat[j].legend([lines[0], line_average[0]], ['Significant channels', 'Average'])
            axes.T.flat[j].set_title(features[j])
        plt.xlabel("Time (s)")
        fig.savefig(os.path.join(trf_folder, subject_path.name, 'trf_kernels_significant_channels_' + channel_type + '.' + save_format))
        plt.close()

        if zscore:
            rf.coef_ = (rf.coef_ - np.mean(rf.coef_)) / np.std(rf.coef_)

        if rms:
            coef_list.append(np.sqrt(np.mean(rf.coef_[significant_channels, :, :]**2, axis=0)))
        else:
            coef_list.append(np.mean(rf.coef_[significant_channels, :, :], axis=0))

    coef_list = np.array(coef_list)
    coefs_average = np.mean(coef_list, axis=0)

    fig, axes = plt.subplots(nrows=ceil(rf.coef_.shape[1] / ncols), ncols=ncols, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        if indiv_subjects:
            lines = axes.T.flat[j].plot(rf.delays_ / rf.sfreq, coef_list[:, j, :].T, 'lightgray', linewidth=2)
        else:
            # plot sem
            sem = np.std(coef_list[:, j, :], axis=0) / np.sqrt(coef_list.shape[0])
            axes.T.flat[j].fill_between(rf.delays_ / rf.sfreq, coefs_average[j, :] - sem, coefs_average[j, :] + sem, color=['lightgray'], alpha=0.6)
        axes.T.flat[j].axvline(0, linestyle='--', color='k')
        if not rms:
            axes.T.flat[j].axhline(0, linestyle='--', color='k')
        line_average = axes.T.flat[j].plot(rf.delays_ / rf.sfreq, coefs_average[j, :], 'dimgrey', linewidth=4)
        if j==0:
            if indiv_subjects:
                axes.T.flat[j].legend([lines[0], line_average[0]], ['Subject', 'Average'])
        axes.T.flat[j].set_title(features[j])
    plt.xlabel("Time (s)")
    if indiv_subjects:
        fig.savefig(os.path.join(trf_folder, 'trf_kernels_significant_channels_' + channel_type + '.' + save_format))
    else:
        fig.savefig(os.path.join(trf_folder, 'trf_kernels_significant_channels_' + channel_type + '_avg' +'.' + save_format))

    plt.close('all')

    return


def plot_trf_kernels_topomap(data_path, trf_folder, features=None, time=0, channel_type='grad', zscore=True, save_format='png'):
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

        sample = int(round(time * rf.sfreq - rf.delays_[0]))

        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
        mag_info = raw.copy().pick(['mag']).info  # Take Info of the magnetometers for channels positions

        fig, axes = plt.subplots(nrows=rf.coef_.shape[1], ncols=2, figsize=(20, 12), dpi=100)
        for j in range(rf.coef_.shape[1]):
            im, cn = mne.viz.plot_topomap(rf.coef_[:, j, sample], mag_info, show=False, axes=axes[j, 0],
                                          ch_type='mag')
            axes[j, 0].set_title(features[j])
            clb = fig.colorbar(im, cax=axes[j, 1])
            clb.ax.set_title("Coefs", fontsize=15)  # title on top of colorbar
            axes[j, 1].set_box_aspect(5)
            axes[j, 1].set_anchor('W')
            axes[j, 0].set_anchor('E')

        fig.savefig(os.path.join(trf_folder, subject_path.name, 'trf_kernels_topomaps_' + channel_type + '_' + str(time) + 's' + '.' + save_format))

        if zscore:
            rf.coef_ = (rf.coef_ - np.mean(rf.coef_)) / np.std(rf.coef_)
        coef_list.append(rf.coef_)

    coef_list = np.array(coef_list)
    coefs_average = np.mean(coef_list, axis=0)

    sample = int(round(time * rf.sfreq - rf.delays_[0]))

    raw_path = list(subject_folders[0].glob('**/*raw_sss.fif'))[0]
    raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
    mag_info = raw.copy().pick(['mag']).info  # Take Info of the magnetometers for channels positions

    fig, axes = plt.subplots(nrows=rf.coef_.shape[1], ncols=2, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        im, cn = mne.viz.plot_topomap(coefs_average[:, j, sample], mag_info, show=False, axes=axes[j, 0], ch_type='mag')
        axes[j, 0].set_title(features[j])
        clb = fig.colorbar(im, cax=axes[j, 1])
        clb.ax.set_title("Coefs", fontsize=15)  # title on top of colorbar
        axes[j, 1].set_box_aspect(5)
        axes[j, 1].set_anchor('W')
        axes[j, 0].set_anchor('E')

    fig.savefig(os.path.join(trf_folder, 'trf_kernels_topomaps_' + channel_type + '_' +str(time) + 's' + '.' + save_format))

    plt.close('all')

    return



def plot_trf_kernels_best_channels_conditions(data_path, main_trf_folder, conditions_list, save_name, features=None, channel_type='grad', nb_channels=5, ncols=2, save_format='png', remove_subjects=None):

    if features is None:
        features = ['Envelope', 'Envelope derivative', 'Surprise', 'Entropy']

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    if remove_subjects is not None:
        for i in sorted(remove_subjects, reverse=True):
            del subject_folders[i]

    scores_list = [np.load(os.path.join(main_trf_folder, condition, "channels_scores.npy"), allow_pickle=True) for condition in conditions_list]

    coef_list = [[] for _ in range(len(conditions_list))]

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        for k, condition in enumerate(conditions_list):
            file = open(os.path.join(main_trf_folder, condition, subject_path.name, "rf.pickle"), 'rb')
            rf = pickle.load(file)
            file.close()

            # score = np.mean(scores_list[k][i, 0], axis=0)[channels_to_keep]  # Average over epochs and select channel type
            score = scores_list[k][i, :]  # select condition and participant
            best_chans = np.flip(np.argsort(score))[:nb_channels]

            for j in range(rf.coef_.shape[1]):
                # Flip kernels for maximum correlation
                if nb_channels > 1:
                    signs = best_flip_combination(rf.coef_[best_chans, j, :])
                    rf.coef_[best_chans, j, :] = signs[:, np.newaxis] * rf.coef_[best_chans, j, :]

            coef_list[k].append(np.mean(rf.coef_[best_chans, :, :], axis=0))

    # Flip kernels for maximum correlation across subjects
    coef_list = [np.array(coef_list[k]) for k in range(len(conditions_list))]
    for k in range(len(conditions_list)):
        for j in range(rf.coef_.shape[1]):
            if coef_list[k].shape[0] > 1:
                signs = best_flip_combination_subjects(coef_list[k][:, j, :])
                print(signs)
                coef_list[k][:, j, :] = signs[:, np.newaxis] * coef_list[k][:, j, :]

    # Statistical testing, diff between each condition and the first one
    significant_range = t_test_trf_kernels(coef_list, stat_test='permut_cluster')

    coefs_average = [np.mean(coef_list[k], axis=0) for k in range(len(conditions_list))]

    if len(conditions_list) == 3 :  # if 3 conditions, take specified colors, otherwise it uses a colormap
        colors_list = [matplotlib.cm.get_cmap('Greys')(0.5), matplotlib.cm.get_cmap('Blues')(0.7), matplotlib.cm.get_cmap('Greens')(0.7)]
        sem_colors_list = [matplotlib.cm.get_cmap('Greys')(0.4), matplotlib.cm.get_cmap('Blues')(0.4), matplotlib.cm.get_cmap('Greens')(0.4)]
    else:
        colors_list = matplotlib.cm.get_cmap('Dark2').colors
        sem_colors_list = matplotlib.cm.get_cmap('Dark2').colors

    fig, axes = plt.subplots(nrows=ceil(rf.coef_.shape[1] / ncols), ncols=ncols, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        axes.T.flat[j].axvline(0, linestyle='--', color='k')
        axes.T.flat[j].axhline(0, linestyle='--', color='k')
        lines = []
        for k in range(len(conditions_list)):
            lines.append(axes.T.flat[j].plot(rf.delays_ / rf.sfreq, coefs_average[k][j, :], color=colors_list[k], linewidth=2)[0])
            sem = np.std(coef_list[k][:, j, :], axis=0) / np.sqrt(coef_list[k].shape[0])
            axes.T.flat[j].fill_between(rf.delays_ / rf.sfreq, coefs_average[k][j, :] - sem, coefs_average[k][j, :] + sem, color=[sem_colors_list[k]], alpha=0.6)
            #axes.flat[j].fill_between(rf.delays_ / rf.sfreq, coefs_average[k][j, :] - sem, coefs_average[k][j, :] + sem, color=[colors_list[k]], alpha=0.3)
        # draw significance lines
        for k in range(1, len(conditions_list)):
            mask = np.ma.masked_where(significant_range[k-1][j, :] >= 0.05, axes.T.flat[j].get_ylim()[0] * np.ones_like(rf.delays_)) # we mask what is not significant
            axes.T.flat[j].plot(rf.delays_ / rf.sfreq, mask, color=colors_list[k], linewidth=1)
        if j == 0:
            axes.T.flat[j].legend(lines, conditions_list)
        axes.T.flat[j].set_title(features[j])
    plt.xlabel("Time (s)")

    fig.savefig(os.path.join(main_trf_folder, 'trf_kernels_' + str(nb_channels) + 'best_channels_' + channel_type + '_' + save_name + '.' + save_format))
    plt.close('all')

    return


def plot_trf_kernels_all_channels_conditions(data_path, main_trf_folder, conditions_list, save_name, features=None, channel_type='grad', ncols=2, zscore=False, save_format='png', remove_subjects=None):

    if features is None:
        features = ['Envelope', 'Envelope derivative', 'Surprise', 'Entropy']

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    if remove_subjects is not None:
        for i in sorted(remove_subjects, reverse=True):
            del subject_folders[i]

    coef_list = [[] for _ in range(len(conditions_list))]

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        for k, condition in enumerate(conditions_list):
            file = open(os.path.join(main_trf_folder, condition, subject_path.name, "rf.pickle"), 'rb')
            rf = pickle.load(file)
            file.close()

            if zscore:
                rf.coef_ = (rf.coef_ - np.mean(rf.coef_)) / np.std(rf.coef_)
            coef_list[k].append(np.sqrt(np.mean(rf.coef_**2, axis=0)))

    # Flip kernels for maximum correlation across subjects
    coef_list = [np.array(coef_list[k]) for k in range(len(conditions_list))]

    # Statistical testing, diff between each condition and the first one
    significant_range = t_test_trf_kernels(coef_list, stat_test='permut_cluster')

    coefs_average = [np.mean(coef_list[k], axis=0) for k in range(len(conditions_list))]

    if len(conditions_list) == 3:  # if 3 conditions, take specified colors, otherwise it uses a colormap
        colors_list = [matplotlib.cm.get_cmap('Greys')(0.5), matplotlib.cm.get_cmap('Blues')(0.7),
                       matplotlib.cm.get_cmap('Greens')(0.7)]
        sem_colors_list = [matplotlib.cm.get_cmap('Greys')(0.4), matplotlib.cm.get_cmap('Blues')(0.4),
                           matplotlib.cm.get_cmap('Greens')(0.4)]
    else:
        colors_list = matplotlib.cm.get_cmap('Dark2').colors
        sem_colors_list = matplotlib.cm.get_cmap('Dark2').colors

    fig, axes = plt.subplots(nrows=ceil(rf.coef_.shape[1] / ncols), ncols=ncols, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        axes.T.flat[j].axvline(0, linestyle='--', color='k')
        # axes.T.flat[j].axhline(0, linestyle='--', color='k')
        lines = []
        for k in range(len(conditions_list)):
            lines.append(axes.T.flat[j].plot(rf.delays_ / rf.sfreq, coefs_average[k][j, :], color=colors_list[k], linewidth=2)[0])
            sem = np.std(coef_list[k][:, j, :], axis=0) / np.sqrt(coef_list[k].shape[0])
            axes.T.flat[j].fill_between(rf.delays_ / rf.sfreq, coefs_average[k][j, :] - sem, coefs_average[k][j, :] + sem, color=[sem_colors_list[k]], alpha=0.6)
            #axes.flat[j].fill_between(rf.delays_ / rf.sfreq, coefs_average[k][j, :] - sem, coefs_average[k][j, :] + sem, color=[colors_list[k]], alpha=0.3)
        # draw significance lines
        for k in range(1, len(conditions_list)):
            mask = np.ma.masked_where(significant_range[k-1][j, :] >= 0.05, axes.T.flat[j].get_ylim()[0] * np.ones_like(rf.delays_)) # we mask what is not significant
            axes.T.flat[j].plot(rf.delays_ / rf.sfreq, mask, color=colors_list[k], linewidth=1)
        if j == 0:
            axes.T.flat[j].legend(lines, conditions_list)
        axes.T.flat[j].set_title(features[j])
    plt.xlabel("Time (s)")

    fig.savefig(os.path.join(main_trf_folder, 'trf_kernels_all_channels_' + channel_type + '_' + save_name + '.' + save_format))
    plt.close('all')

    return


def plot_trf_kernels_significant_channels_conditions(data_path, scores_path, main_trf_folder, conditions_list, save_name, features=None, channel_type='grad', ncols=2, zscore=False, stat_test='permut_cluster', save_format='png', remove_subjects=None):

    if features is None:
        features = ['Envelope', 'Envelope derivative', 'Surprise', 'Entropy']

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    if remove_subjects is not None:
        for i in sorted(remove_subjects, reverse=True):
            del subject_folders[i]

    scores = np.load(scores_path, allow_pickle=True)
    scores = np.arctanh(scores) # Fisher z-transform

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    raw_path = list(subject_folders[0].glob('**/*raw_sss.fif'))[0]
    raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
    raw.pick(['meg'])
    mag_info = raw.copy().pick(['mag']).info  # Take Info of the magnetometers for channels positions

    # Find channels with significant score
    p_value = test_channels_significance(scores[:, :, 0], info=mag_info, test=stat_test, channel_type='mag',
                                               return_t_obs=False)  # Permutation across subjects

    # define the statistical threshold and significant channels
    alpha = 0.05
    significant_channels = p_value < alpha

    # If no significant channel, do not plot anything
    if np.sum(significant_channels) == 0:
        print('No significant channel')
        return

    coef_list = [[] for _ in range(len(conditions_list))]

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        for k, condition in enumerate(conditions_list):
            file = open(os.path.join(main_trf_folder, condition, subject_path.name, "rf.pickle"), 'rb')
            rf = pickle.load(file)
            file.close()

            if zscore:
                rf.coef_ = (rf.coef_ - np.mean(rf.coef_)) / np.std(rf.coef_)

            coef_list[k].append(np.sqrt(np.mean(rf.coef_[significant_channels, :, :]**2, axis=0)))

    # Flip kernels for maximum correlation across subjects
    coef_list = [np.array(coef_list[k]) for k in range(len(conditions_list))]

    # Statistical testing, diff between each condition and the first one
    significant_range = t_test_trf_kernels(coef_list, stat_test='permut_cluster')

    coefs_average = [np.mean(coef_list[k], axis=0) for k in range(len(conditions_list))]

    if len(conditions_list) == 3:  # if 3 conditions, take specified colors, otherwise it uses a colormap
        colors_list = [matplotlib.cm.get_cmap('Greys')(0.5), matplotlib.cm.get_cmap('Blues')(0.7),
                       matplotlib.cm.get_cmap('Greens')(0.7)]
        sem_colors_list = [matplotlib.cm.get_cmap('Greys')(0.4), matplotlib.cm.get_cmap('Blues')(0.4),
                           matplotlib.cm.get_cmap('Greens')(0.4)]
    else:
        colors_list = matplotlib.cm.get_cmap('Dark2').colors
        sem_colors_list = matplotlib.cm.get_cmap('Dark2').colors

    fig, axes = plt.subplots(nrows=ceil(rf.coef_.shape[1] / ncols), ncols=ncols, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        axes.T.flat[j].axvline(0, linestyle='--', color='k')
        # axes.T.flat[j].axhline(0, linestyle='--', color='k')
        lines = []
        for k in range(len(conditions_list)):
            lines.append(axes.T.flat[j].plot(rf.delays_ / rf.sfreq, coefs_average[k][j, :], color=colors_list[k], linewidth=2)[0])
            sem = np.std(coef_list[k][:, j, :], axis=0) / np.sqrt(coef_list[k].shape[0])
            axes.T.flat[j].fill_between(rf.delays_ / rf.sfreq, coefs_average[k][j, :] - sem, coefs_average[k][j, :] + sem, color=[sem_colors_list[k]], alpha=0.6)
            #axes.flat[j].fill_between(rf.delays_ / rf.sfreq, coefs_average[k][j, :] - sem, coefs_average[k][j, :] + sem, color=[colors_list[k]], alpha=0.3)
        # draw significance lines
        for k in range(1, len(conditions_list)):
            mask = np.ma.masked_where(significant_range[k-1][j, :] >= 0.05, axes.T.flat[j].get_ylim()[0] * np.ones_like(rf.delays_)) # we mask what is not significant
            axes.T.flat[j].plot(rf.delays_ / rf.sfreq, mask, color=colors_list[k], linewidth=1)
        if j == 0:
            axes.T.flat[j].legend(lines, conditions_list)
        axes.T.flat[j].set_title(features[j])
    plt.xlabel("Time (s)")

    fig.savefig(os.path.join(main_trf_folder, 'trf_kernels_significant_channels_' + channel_type + '_' + save_name + '.' + save_format))
    plt.close('all')

    return


def plot_trf_regression_beta_conditions(data_path, scores_path, main_trf_folder, conditions_list, save_name, features=None, channel_type='mag', significant_channels_only=True, ncols=2, stat_test='permut_cluster', save_format='png'):

    if features is None:
        features = ['Envelope', 'Envelope derivative', 'Surprise', 'Entropy']

    scores = np.load(scores_path, allow_pickle=True)
    scores = np.arctanh(scores) # Fisher z-transform

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    raw_path = list(subject_folders[0].glob('**/*raw_sss.fif'))[0]
    raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
    raw.pick(['meg'])
    mag_info = raw.copy().pick(['mag']).info  # Take Info of the magnetometers for channels positions

    if significant_channels_only:
        # Find channels with significant score
        p_value = test_channels_significance(scores[:, :, 0], info=mag_info, test=stat_test, channel_type=channel_type,
                                             return_t_obs=False)  # Permutation across subjects

        # define the statistical threshold and significant channels
        alpha = 0.05
        significant_channels = p_value < alpha
    else:
        significant_channels = np.ones(scores.shape[1], dtype='int')  # keep all channels

    # If no significant channel, do not plot anything
    if np.sum(significant_channels) == 0:
        print('No significant channel')
        return

    coef_list = [[] for _ in range(len(conditions_list))]
    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        for k, condition in enumerate(conditions_list):
            file = open(os.path.join(main_trf_folder, condition, subject_path.name, "rf.pickle"), 'rb')
            rf = pickle.load(file)
            file.close()

            coef_list[k].append(np.sqrt(np.mean(rf.coef_[significant_channels, :, :]**2, axis=0)))

    coef_list = [np.array(coef_list[k]) for k in range(len(conditions_list))]

    # Instantaneous 3 points linear regression between repetition conditions
    betas = np.zeros_like(coef_list[0])
    X = np.array([[1], [2], [3]])  # Repetition number condition is used as a linear feature
    for i in range(betas.shape[0]):
        for j in range(betas.shape[1]):
            for t in range(betas.shape[2]):
                y = np.array([coefs[i, j, t] for coefs in coef_list])
                regr = LinearRegression().fit(X, y)
                betas[i, j, t] = regr.coef_[0]

    betas_t_score = np.mean(betas, axis=0) / (np.std(betas, axis=0) / np.sqrt(betas.shape[0]))  # t-scores

    # Statistical testing
    significant_range = [one_sample_t_test_trf_kernels(betas[:, j, :], stat_test='permut_cluster') for j in range(betas.shape[1])]

    # Plot betas
    fig, axes = plt.subplots(nrows=ceil(rf.coef_.shape[1] / ncols), ncols=ncols, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        axes.T.flat[j].axvline(0, linestyle='--', color='k')
        axes.T.flat[j].axhline(0, linestyle='--', color='k')

        axes.T.flat[j].plot(rf.delays_ / rf.sfreq, betas_t_score[j, :], 'dimgrey', linewidth=4)
        # sem = np.std(betas[:, j, :], axis=0) / np.sqrt(betas.shape[0])
        # axes.T.flat[j].fill_between(rf.delays_ / rf.sfreq, betas_average[j, :] - sem, betas_average[j, :] + sem, color=['lightgray'], alpha=0.6)

        # draw significance lines
        mask = np.ma.masked_where(significant_range[j] >= 0.05, axes.T.flat[j].get_ylim()[0] * np.ones_like(rf.delays_)) # we mask what is not significant
        axes.T.flat[j].plot(rf.delays_ / rf.sfreq, mask, color='dimgrey', linewidth=1)

        axes.T.flat[j].set_title(features[j])
        axes.T.flat[j].set_ylabel('Regression coefficient T-score')
    plt.xlabel("Time (s)")

    if significant_channels_only:
        fig.savefig(os.path.join(main_trf_folder, 'trf_regression_beta_significant_channels_' + channel_type + '_' + save_name + '.' + save_format))
    else:
        fig.savefig(os.path.join(main_trf_folder, 'trf_regression_beta_all_channels_' + channel_type + '_' + save_name + '.' + save_format))

    # Topography of regression coefs, all channels
    coef_list = [[] for _ in range(len(conditions_list))]
    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        for k, condition in enumerate(conditions_list):
            file = open(os.path.join(main_trf_folder, condition, subject_path.name, "rf.pickle"), 'rb')
            rf = pickle.load(file)
            file.close()

            coef_list[k].append(rf.coef_)

    coef_list = [np.array(coef_list[k]) for k in range(len(conditions_list))]

    # Instantaneous 3 points linear regression between repetition conditions
    betas = np.zeros_like(coef_list[0])
    X = np.array([[1], [2], [3]])  # Repetition number condition is used as a linear feature
    for i in range(betas.shape[0]):
        print('Subject', i)
        for c in range(betas.shape[1]):
            for j in range(betas.shape[2]):
                for t in range(betas.shape[3]):
                    y = np.array([coefs[i, c, j, t] for coefs in coef_list])
                    regr = LinearRegression().fit(X, y)
                    betas[i, c, j, t] = regr.coef_[0]

    betas_t_score = np.mean(betas, axis=0) / (np.std(betas, axis=0) / np.sqrt(betas.shape[0]))  # t-scores

    fig, axes = plt.subplots(nrows=ceil(rf.coef_.shape[1] / ncols), ncols=ncols, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        # Find first significant cluster  # TODO: ugly method, improve!
        significant_window = significant_range[j] < 0.05
        first_cluster = False
        first_cluster_done = False
        for t in range(significant_window.shape[0]):
            if (first_cluster == False) and (significant_window[t] == 1):
                first_cluster = True
            if (first_cluster == True) and (significant_window[t] == 0):
                first_cluster_done = True
            if first_cluster_done:
                significant_window[t] = 0
        significant_window = [t for t in range(significant_window.shape[0]) if significant_window[t]==1]

        if len(significant_window) > 0:
            im, cn = mne.viz.plot_topomap(np.mean(betas_t_score[:, j, significant_window], axis=1), mag_info, show=False, axes=axes.T.flat[j], ch_type='mag', vmin=-2.5, vmax=2.5)  # plot t-scores  # TODO: change vmin vmax
            clb = fig.colorbar(im, ax=axes.T.flat[j])
            clb.ax.set_title('T-score', fontsize=15)  # title on top of colorbar

            time_window = rf.delays_[significant_window] / rf.sfreq
            axes.T.flat[j].set_title(features[j] + ' time window = [' + str(time_window[0]) + ', '+ str(time_window[-1]) + ']')

    fig.savefig(os.path.join(main_trf_folder, 'trf_regression_beta_topomap_' + channel_type + '_' + save_name + '.' + save_format))
    plt.close('all')

    return


def spatio_temporal_cluster_trf_kernels_conditions(data_path, main_trf_folder, conditions_list, save_name, features=None, channel_type='grad', ncols=2, save_format='png'):

    if features is None:
        features = ['Envelope', 'Envelope derivative', 'Surprise', 'Entropy']

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    coef_list = []  # lenght number of conditions, each element is an array of shape (n_participant, n_trf_delays, n_channels)

    for condition in conditions_list:
        for i, subject_path in enumerate(sorted(subject_folders)):
            file = open(os.path.join(main_trf_folder, condition, subject_path.name, "rf.pickle"), 'rb')
            rf = pickle.load(file)
            file.close()

            coef_list[k].append(np.mean(rf.coef_, axis=0))


    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)

        for k, condition in enumerate(conditions_list):
            file = open(os.path.join(main_trf_folder, condition, subject_path.name, "rf.pickle"), 'rb')
            rf = pickle.load(file)
            file.close()

            coef_list[k].append(np.mean(rf.coef_, axis=0))

    # Flip kernels for maximum correlation across subjects
    coef_list = [np.array(coef_list[k]) for k in range(len(conditions_list))]

    # Statistical testing, diff between each condition and the first one
    significant_range = t_test_trf_kernels(coef_list, stat_test='permut_cluster')

    coefs_average = [np.mean(coef_list[k], axis=0) for k in range(len(conditions_list))]

    colors_list = [matplotlib.cm.get_cmap('Greys')(0.5), matplotlib.cm.get_cmap('Blues')(0.7), matplotlib.cm.get_cmap('Greens')(0.7)]
    sem_colors_list = [matplotlib.cm.get_cmap('Greys')(0.4), matplotlib.cm.get_cmap('Blues')(0.4), matplotlib.cm.get_cmap('Greens')(0.4)]

    fig, axes = plt.subplots(nrows=ceil(rf.coef_.shape[1] / ncols), ncols=ncols, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    for j in range(rf.coef_.shape[1]):
        axes.T.flat[j].axvline(0, linestyle='--', color='k')
        axes.T.flat[j].axhline(0, linestyle='--', color='k')
        lines = []
        for k in range(len(conditions_list)):
            lines.append(axes.T.flat[j].plot(rf.delays_ / rf.sfreq, coefs_average[k][j, :], color=colors_list[k], linewidth=2)[0])
            sem = np.std(coef_list[k][:, j, :], axis=0) / np.sqrt(coef_list[k].shape[0])
            axes.T.flat[j].fill_between(rf.delays_ / rf.sfreq, coefs_average[k][j, :] - sem, coefs_average[k][j, :] + sem, color=[sem_colors_list[k]], alpha=0.6)
            #axes.flat[j].fill_between(rf.delays_ / rf.sfreq, coefs_average[k][j, :] - sem, coefs_average[k][j, :] + sem, color=[colors_list[k]], alpha=0.3)
        # draw significance lines
        for k in range(1, len(conditions_list)):
            mask = np.ma.masked_where(significant_range[k-1][j, :] >= 0.05, axes.T.flat[j].get_ylim()[0] * np.ones_like(rf.delays_)) # we mask what is not significant
            axes.T.flat[j].plot(rf.delays_ / rf.sfreq, mask, color=colors_list[k], linewidth=1)
        if j == 0:
            axes.T.flat[j].legend(lines, conditions_list)
        axes.T.flat[j].set_title(features[j])
    plt.xlabel("Time (s)")

    fig.savefig(os.path.join(main_trf_folder, 'trf_kernels_all_channels_' + channel_type + '_' + save_name + '.' + save_format))
    plt.close('all')

    return


def plot_topo_best_channels(data_path, scores_path, save_path, channel_type='grad', nb_channels=5, save_format='png'):

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    scores = np.load(scores_path, allow_pickle=True)

    best_chans_list = []
    best_chans_env_list = []

    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
        raw.pick(['meg'])
        mag_info = raw.copy().pick(['mag']).info  # Take Info of the magnetometers for channels positions

        score = np.mean(scores[i, 0], axis=0)[mne.channel_indices_by_type(raw.info)[channel_type]]  # Average over epochs and select channel type
        chans = np.zeros_like(score)
        chans[np.argsort(score)[-nb_channels:]] = 1
        best_chans_list.append(chans)

        # acoustic features only
        score = np.mean(scores[i, 1], axis=0)[mne.channel_indices_by_type(raw.info)[channel_type]]  # Average over epochs and select channel type
        chans = np.zeros_like(score)
        chans[np.argsort(score)[-nb_channels:]] = 1
        best_chans_env_list.append(chans)

    best_chans_list = np.sum(np.array(best_chans_list), axis=0)
    best_chans_env_list = np.sum(np.array(best_chans_env_list), axis=0)
    if channel_type == 'grad':
        best_chans_list = merge_grad_data(best_chans_list, method='sum')  # for grad data, method='sum' in case both channels at a given location are selected
        best_chans_env_list = merge_grad_data(best_chans_env_list, method='sum')

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12), dpi=300)
    im, cn = mne.viz.plot_topomap(best_chans_list, mag_info, show=False, axes=axes[0, 0], ch_type='mag')
    axes[0, 0].set_title("AM")
    clb = fig.colorbar(im, cax=axes[0, 1])
    clb.ax.set_title("Number of best channels", fontsize=15)  # title on top of colorbar
    axes[0, 1].set_box_aspect(5)
    axes[0, 1].set_anchor('W')
    axes[0, 0].set_anchor('E')
    im, cn = mne.viz.plot_topomap(best_chans_env_list, mag_info, show=False, axes=axes[1, 0], ch_type='mag')
    axes[1, 0].set_title("A")
    clb = fig.colorbar(im, cax=axes[1, 1])
    axes[1, 1].set_box_aspect(5)
    axes[1, 1].set_anchor('W')
    axes[1, 0].set_anchor('E')

    fig.savefig(os.path.join(save_path, 'best_channels_topography_' + channel_type + '.' + save_format))


    return


def plot_all_alphas_trf_kernels(raw_path, trf_folder, features=None, channel_type='grad', save_format='png'):

    if features is None:
        features = ['Envelope', 'Envelope derivative', 'Surprise', 'Entropy']

    raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
    raw.pick(['meg'])

    alphas = np.logspace(-3, 12, 16)

    for ii, alpha in enumerate(alphas):
        print('alpha:', alpha)

        file = open(os.path.join(trf_folder, "rf_" + str(alpha) + ".pickle"), 'rb')
        rf = pickle.load(file)
        file.close()

        fig, axes = plt.subplots(nrows=rf.coef_.shape[1], ncols=1, sharex=True, sharey=False, figsize=(20, 12), dpi=100)
        if rf.coef_.shape[1] == 1:
            axes.plot(rf.delays_ / rf.sfreq, rf.coef_[mne.channel_indices_by_type(raw.info)[channel_type], 0, :].T)
            axes.axvline(0, linestyle='--', color='k')
            axes.set_title(features[0])
        else:
            for j in range(rf.coef_.shape[1]):
                axes[j].plot(rf.delays_ / rf.sfreq, rf.coef_[mne.channel_indices_by_type(raw.info)[channel_type], j, :].T)
                axes[j].axvline(0, linestyle='--', color='k')
                axes[j].set_title(features[j])
        plt.xlabel("Time (s)")
        fig.savefig(os.path.join(trf_folder, 'trf_kernels_' + channel_type + '_' + str(alpha) + '.' + save_format))
        plt.close()

    return


def plot_one_feature_all_alphas_trf_kernels(raw_path, trf_folder, features=None, channel_type='grad', save_format='png'):

    if features is None:
        features = ['Envelope']

    raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
    raw.pick(['meg'])

    alphas = np.logspace(0, 7, 8)
    fig, axes = plt.subplots(nrows=8, ncols=1, sharex=True, sharey=True, figsize=(20, 12), dpi=100)
    
    for ii, alpha in enumerate(alphas):
        print('alpha:', alpha)

        file = open(os.path.join(trf_folder, "rf_" + str(alpha) + ".pickle"), 'rb')
        rf = pickle.load(file)
        file.close()

        axes[ii].plot(rf.delays_ / rf.sfreq, rf.coef_[mne.channel_indices_by_type(raw.info)[channel_type], 0, :].T)
        axes[ii].axvline(0, linestyle='--', color='k')
        axes[ii].set_title("alpha="+str(alpha))
    plt.xlabel("Time (s)")
    fig.savefig(os.path.join(trf_folder, 'trf_kernels_' + channel_type + '_all_alphas' + '.' + save_format))
    plt.close()

    return


def plot_trf_prediction(input_path, raw_path, wav_folder, pickle_path, channel_type='grad', sample_range=None):
    """
    """

    if sample_range is None:
        sample_range = [100, 800]

    Y, metadata = create_trf_output_from_epochs(raw_path, wav_folder ,save_path=None, sample_range=sample_range, filt_range=[1, 8], channel_type=channel_type)

    X = np.load(input_path)
    list_to_keep = list_from_metadata(metadata)
    X = X[:, list_to_keep, :]
    print(X.shape)
    X = (X - np.mean(X, axis=(0, 1))) / np.std(X, axis=(0, 1))

    print(Y.shape)

    feature_names = np.arange(X.shape[2])

    file = open(os.path.join(pickle_path, "rf.pickle"), 'rb')
    rf = pickle.load(file)
    file.close()

    pred = rf.predict(X[:, 100, :]) * 8  # predictions values are very low

    t = np.linspace(0, 7, X.shape[0]) / rf.sfreq
    for channel in range(Y.shape[2]):
        plt.plot(t, Y[:, 100, channel])
        plt.plot(t, pred[:, channel])
        plt.xlabel("Time (s)")
        plt.legend(['Data', 'Prediction'])
        plt.show()

    return


def barplots_score_diffs(main_trf_folder, conditions, nb_best_channels):
    '''Looks for scores of name epochs_channels_scores.npy'''

    diffs_matrix = np.zeros((27, len(conditions)))
    for i, condition in enumerate(conditions):
        scores = np.load(os.path.join(main_trf_folder, condition, "epochs_channels_scores.npy"), allow_pickle=True)
        print(scores.shape)
        print(scores[0,0].shape)
        averaged_epochs_scores = np.swapaxes(np.array([[np.mean(scores[i, j], axis=0) for j in range(scores.shape[1])] for i in range(scores.shape[0])]), 1, 2)  # Average over epochs for each participant
        print(averaged_epochs_scores.shape)

        for subject in range(averaged_epochs_scores.shape[0]):
            diffs_matrix[subject, i] = np.mean(np.flip(np.sort(averaged_epochs_scores[subject, :, 0] - averaged_epochs_scores[subject, :, 1]))[:nb_best_channels])

    diffs_matrix = pd.DataFrame(diffs_matrix)
    diffs_matrix.columns = conditions
    print(diffs_matrix)

    ax = sns.boxplot(data=diffs_matrix, showmeans=False, width=0.6, linewidth=2.5)
    ax = sns.swarmplot(data=diffs_matrix, color=".25", size=6)
    ax.set_ylabel("d' score", fontsize=17)
    ax.yaxis.set_tick_params(labelsize=13)
    ax.xaxis.set_tick_params(labelsize=15)

    plt.show()

    return


if __name__ == "__main__":

    '''plot_m100_psd'''
    # plot_m100_psd('/home/mathieupvc/These/data/meg_maestro/preprocessed', channel_types=['meg'])

    '''plot_m100_all_subjects'''
    # plot_m100_all_subjects('/home/mathieupvc/These/data/meg_maestro/preprocessed', channel_types=['grad'], wav_folder="/home/mathieupvc/These/data/meg_maestro/exp_stim_meg")

    '''create_trf_input_features_perfRNN'''
    # create_trf_input_features_perfRNN(wav_folder='/home/mathieupvc/These/data/meg_maestro/exp_stim_meg',
    #                           surprise_dir='/home/mathieupvc/These/data/meg_maestro/surprise_entropy/surprise',
    #                           entropy_dir='/home/mathieupvc/These/data/meg_maestro/surprise_entropy/entropy',
    #                           save_path='/home/mathieupvc/These/data/meg_maestro/trf_input_note_on.npy',
    #                           pred_feat_list=['note_on'],  # ['note_on', 'note_off', 'velocity', 'time_shift']
    #                           sample_range=[100, 800],
    #                           env_filter=15
    #                           )

    # create_trf_input_features_perfRNN(wav_folder='/home/mathieupvc/These/data/meg_maestro/exp_stim_meg',
    #                                   surprise_dir='/home/mathieupvc/These/data/meg_maestro/surprise_entropy/surprise',
    #                                   entropy_dir='/home/mathieupvc/These/data/meg_maestro/surprise_entropy/entropy',
    #                                   save_path='/home/mathieupvc/These/data/meg_maestro/trf_input_fullrange_surprise_on.npy',
    #                                   pred_feat_list=['note_on'],
    #                                   # ['note_on', 'note_off', 'velocity', 'time_shift']
    #                                   sample_range=[0, 900],
    #                                   env_filter=15
    #                                   )

    '''create_trf_input_features_paulRNN'''
    # create_trf_input_features_paulRNN(wav_folder='/home/mathieupvc/These/data/meg_maestro/exp_stim_meg',
    #                           pred_feat_dir='/home/mathieupvc/These/data/meg_maestro/model_paul_output',
    #                           save_path='/home/mathieupvc/These/data/meg_maestro/trf_input_model_paul.npy',
    #                           pred_feat_list=['surprise_mean', 'positive_surprise_rescaled', 'pitch_entropy_mean', 'uncertainty'],
    #                           sample_range=[0, 900],
    #                           env_filter=15
    #                           )

    '''create_trf_output_from_epochs'''
    # create_trf_output_from_epochs(
    #     epoch_path='/home/mathieupvc/These/data/meg_maestro/preprocessed/maestro01_itdheb/210629_epo.fif',
    #     wav_folder=
    #     save_path='/home/mathieupvc/These/data/meg_maestro/trf_output.npy'
    # )

    '''run_trf'''
    # run_trf(input_path='/home/mathieupvc/These/data/meg_maestro/trf_input.npy',
    #         output='/home/mathieupvc/These/data/meg_maestro/trf_output.npy',
    #         save_model_path='/home/mathieupvc/These/data/meg_maestro/untest')

    '''compare_trfs'''
    # compare_trfs("/home/mathieupvc/These/data/meg_maestro/trf/scores.npy")
    # compare_trfs("/home/mathieupvc/These/data/meg_maestro/trf/scores_20channels.npy")
    # compare_trfs("/home/mathieupvc/These/data/meg_maestro/trf_on_shift/scores.npy")
    # compare_trfs("/home/mathieupvc/These/data/meg_maestro/trf_mag/scores.npy", individual_channels=True)
    # compare_trfs('/home/mathieupvc/These/data/meg_maestro/trf_6_channels/residual_trf/scores.npy', individual_channels=True, residual_trf=True)
    # compare_trfs('/home/mathieupvc/These/data/meg_maestro/trf_6_channels/scores.npy', individual_channels=True, residual_trf=False)

    '''plot_trf_kernels'''
    # plot_trf_kernels("/home/mathieupvc/These/data/meg_maestro/preprocessed", "/home/mathieupvc/These/data/meg_maestro/trf", ['Envelope', 'Surprise', 'Entropy'], channel_type='grad', save_format='png')

    '''trf_ridge_param_hist'''
    # trf_ridge_param_hist("/home/mathieupvc/These/data/meg_maestro/preprocessed", "/home/mathieupvc/These/data/meg_maestro/trf")
    # trf_ridge_param_hist("/home/mathieupvc/These/data/meg_maestro/preprocessed", "/home/mathieupvc/These/data/meg_maestro/trf_residual_mag")
    # trf_ridge_param_hist("/home/mathieupvc/These/data/meg_maestro/preprocessed", "/home/mathieupvc/These/data/meg_maestro/trf_mag_hp_23", rf_env=True)
    # trf_ridge_param_hist("/home/mathieupvc/These/data/meg_maestro/preprocessed", "/home/mathieupvc/These/data/meg_maestro/trf_mag_lp_paul/residual_trf", rf_env=False)

    '''plot_trf_input_features'''
    # plot_trf_input_features('/home/mathieupvc/These/data/meg_maestro/trf_input_plotting.npy', '/home/mathieupvc/These/data/meg_maestro/exp_stim_meg', pred_feat_list=['note_on', 'time_shift'])
    # plot_trf_input_features('/home/mathieupvc/These/data/meg_maestro/trf_input_on_new.npy', '/home/mathieupvc/These/data/meg_maestro/exp_stim_meg', pred_feat_list=['note_on', 'time_shift'])

    '''barplots_score_diffs'''
    # barplots_score_diffs("/home/mathieupvc/These/data/meg_maestro/trf_on_shift_new", ['rep1', 'rep2', 'rep3'], nb_best_channels=100)
