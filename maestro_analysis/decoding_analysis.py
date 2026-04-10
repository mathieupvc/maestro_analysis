import os
from pathlib import Path
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from joblib import Parallel, delayed
import time

import statsmodels.api as sm

from math import ceil

from maestro_analysis.utils import epoching_maestro_note_level, fixed_golden_aspect_ratio
from maestro_analysis.trf_analysis import condition_list_from_metadata
from maestro_analysis.stats import permutation_cluster_test_adapted
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


def decoding_predictive_features(epochs_path, save_path, model=None, feature_list=None, data_type='meg',
                                   signal_type='raw', montage='mono', specific_channels=None, add_shifted=True, tmin=1, tmax=7,
                                   n_folds=5, alphas=(1e2, 1e3, 1e4, 1e5, 1e6, 1e7), permutation=False, r2_control=None, min_time_interval=None, best_channels_path=None, deviation_quantile_range=None, baseline=False):
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
    best_channels_path: str, path to npy file specifying which channels to keep for each subject. Should not be used with 'specific_channels' (which picks the same channels for every subject)
    deviation_quantile_range: list of len 2, contains the low and high quantile values used for the time interval deviation analysis.

    Function updated for complete set of regressors
    """

    alphas = np.array(alphas)

    # create save dir
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # manage random seed for permutations
    if permutation is not False:
        rng = np.random.default_rng(seed=permutation)

    features = []
    shifted_features = []
    if model == 'perfrnn':
        features = features + ['surprise_perfrnn']  # , 'entropy_perfrnn']  <-- we keep only shifted for now
        shifted_features = ['entropy_perfrnn']
        F_toshuffle = ['surprise_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_entropy_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_perfrnn']
    elif model == 'magenta_perfrnn':
        # features = features + ['surprise_magenta_perfrnn']  # , 'entropy_magenta_perfrnn']  <-- we keep only shifted for now
        features = features + ['surprise_magenta_pitch_perfrnn', 'entropy_magenta_pitch_perfrnn', 'surprise_magenta_timing_perfrnn', 'entropy_magenta_timing_perfrnn']
        # shifted_features = ['surprise_magenta_pitch_perfrnn', 'entropy_magenta_pitch_perfrnn', 'surprise_magenta_timing_perfrnn', 'entropy_magenta_timing_perfrnn'] + ['trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
        #                 'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
        #                 'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']
        shifted_features = []
        F_toshuffle = ['surprise_magenta_perfrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            # features = features + ['shifted_surprise_magenta_pitch_perfrnn', 'shifted_surprise_magenta_timing_perfrnn', 'shifted_entropy_magenta_pitch_perfrnn', 'shifted_entropy_magenta_timing_perfrnn']
            F_toshuffle = F_toshuffle + ['shifted_entropy_magenta_pitch_perfrnn', 'shifted_entropy_magenta_timing_perfrnn']
    elif model == 'paulrnn':
        features = features + ['surprise_possum_paulrnn', 'surprise_negsum_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        # features = features + ['surprise_mean_paulrnn']
        # features = features + ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        shifted_features = ['entropy_pitch_paulrnn', 'entropy_timing_paulrnn']  # chosen one without shifted controls
        # shifted_features = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
        F_toshuffle = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']  # keep track of the features that will be shuffled in permutations
        if add_shifted:
            features = features + ['shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']
            F_toshuffle = F_toshuffle + ['shifted_surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'shifted_entropy_timing_paulrnn']

    if feature_list is not None:
        features += feature_list

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

    best_alphas_list = []

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
        metadata['pitch_interval'] = 0  # add pitch interval
        metadata['onset_type'] = 0  # add onset type
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'rate'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[2][0:3])
            metadata.loc[metadata.index[i_note], 'onset_type'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[4][-5])
            if metadata.loc[metadata.index[i_note], 'trial_time'] > 0:
                metadata.loc[metadata.index[i_note], 'pitch_interval'] = np.abs(metadata.loc[metadata.index[i_note - 1], 'pitch'] - metadata.loc[metadata.index[i_note], 'pitch'])

        # add shifted features
        if add_shifted:
            for feat in shifted_features:
                metadata['shifted_' + feat] = np.nan  # add shifted features
            for i_note in range(metadata.shape[0]):
                if (i_note + 1 < len(metadata)) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note + 1], 'stim_name']):
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
        regressors = metadata[features]

        # remove onset response
        regressors = regressors[epochs.metadata['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressors = regressors[epochs.metadata['trial_time'].to_numpy() <= tmax]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]

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

        # shuffle features for permutations
        if permutation is not False:
            for feat in F_toshuffle:
                regressors[feat] = rng.permutation(regressors[feat].values)

        # print(len(epochs.metadata))
        # regressors = regressors[epochs.metadata['rate'] != 6.5]
        # epochs = epochs[epochs.metadata['rate'] != 6.5]
        # regressors = regressors[epochs.metadata['rate'] != 5.5]
        # epochs = epochs[epochs.metadata['rate'] != 5.5]
        # regressors = regressors[epochs.metadata['rate'] != 4.5]
        # epochs = epochs[epochs.metadata['rate'] != 4.5]
        # regressors = regressors[epochs.metadata['rate'] != 3.5]
        # epochs = epochs[epochs.metadata['rate'] != 3.5]
        # print(len(epochs.metadata))

        # plt.switch_backend('TkAgg')
        # plt.hist(epochs.metadata['rate'], bins=50)
        # plt.show()

        if baseline:
            epochs = epochs.apply_baseline((None, -0.100))

        # load data
        x = epochs.get_data()  # (trials, channels, time)
        if best_channels_path is not None:
            x = x[:, best_channels[i, :], :]
        y = regressors.to_numpy()

        # standardization of features and meg
        y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)
        x = (x - np.mean(x, axis=0, keepdims=True)) / np.std(x, axis=0, keepdims=True)  # removes the relative importance of subjects, channels, and time points

        # print(x.shape)
        # print(y.shape)
        # print(features)
        # plt.switch_backend('TkAgg')
        # fig, axes = plt.subplots(y.shape[1], 1, sharex=True, sharey=True)
        # for feat in range(y.shape[1]):
        #     axes[feat].plot(y[:, feat])

        plt.show()


        del epochs

        # independent regression for every time point and channel
        subject_betas = np.zeros((x.shape[1], y.shape[1], x.shape[2]))
        subject_r2 = np.zeros((y.shape[1], x.shape[2]))
        if n_folds > 1:
            cv = KFold(n_splits=n_folds)
            cv.get_n_splits(x)
        start = time.time()

        def individual_channel_regression(t):  # TODO: put outside of function
            """function used for multiprocessing."""
            if n_folds == 1:
                raise NotImplementedError()
                regr = LinearRegression().fit(x, y[:, c_ind, :])
                # subject_betas[:, c_ind, :] = regr.coef_.T
                # subject_r2[c_ind, :] = r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
                return regr.coef_.T, r2_score(y[:, c_ind, :], regr.predict(x), multioutput='raw_values')
            elif n_folds > 1:
                cv_scores = np.full((alphas.shape[0], n_folds, y.shape[1]), np.nan)
                for a, alpha in enumerate(alphas):
                    for j, (train, test) in enumerate(cv.split(x)):
                        regr = Ridge(alpha=alpha)
                        regr.fit(x[train, :, t], y[train, :])
                        pred = regr.predict(x[test, :, t])
                        cv_scores[a, j, :] = r2_score(y[test, :], pred, multioutput='raw_values')
                # store cv scores and best alphas
                cv_scores = np.mean(cv_scores, axis=1)  # mean cross-validated r² across folds
                best_alphas = alphas[np.argmax(cv_scores, axis=0)]  # store best alpha for each timepoint
                cv_scores = np.max(cv_scores, axis=0)  # for each timepoint, keep the highest cross-val r² across alphas
                # fit again with the best alpha and store betas
                regr = Ridge(best_alphas).fit(x[:, :, t], y)
                # subject_betas[:, c_ind, :] = regr.coef_.T
                return regr.coef_.T, cv_scores, best_alphas.tolist()

        # multiprocessing channels
        channels_regressions = Parallel(n_jobs=-1)(delayed(individual_channel_regression)(t) for t in range(x.shape[2]))
        # channels_regressions = [individual_channel_regression(c) for c in range(y.shape[1])]
        for t in range(x.shape[2]):
            subject_betas[:, :, t] = channels_regressions[t][0]
            subject_r2[:, t] = channels_regressions[t][1]
            best_alphas_list += channels_regressions[t][2]

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
            raise NotImplementedError()
            output['betas'] = np.mean(output['betas'], axis=0, keepdims=True) / (np.std(output['betas'], axis=0, keepdims=True) / np.sqrt(output['betas'].shape[0]))
            output['r2'] = np.mean(output['r2'] - r2_control, axis=0, keepdims=True) / (np.std(output['r2'] - r2_control, axis=0, keepdims=True) / np.sqrt(output['r2'].shape[0]))  # comparison with control

    if model is None:
        model = 'acoustic'
    if permutation is False:
        permutation = ''
    else:
        permutation = '_perm' + str(permutation).zfill(4)
    if data_type == 'meg':
        with open(os.path.join(save_path, 'regr_meg_' + model + permutation + '.pkl'), 'wb') as fp:
            pickle.dump(output, fp)
    elif data_type == 'seeg':
        with open(os.path.join(save_path,
                               'regr_seeg_' + signal_type + '_' + montage + '_' + model + permutation + '.pkl'),
                  'wb') as fp:
            pickle.dump(output, fp)

    np.save(os.path.join(save_path, 'best_alphas.npy'), np.array(best_alphas_list))

    return output


def spatio_temporal_cluster_regression_scores(model_regression, features_to_plot):
    """
    :param r_squared_list: list of numpy scores (1 per feature of interest) of shape (n_subjects, n_channels, times)
    :param control_r_squared: numpy scores of control only of shape (n_subjects, n_channels, times)
    :param epochs: used for times and info
    :param features_list: names of the features
    :param tmin:
    :return:
    """
    plt.switch_backend('TkAgg')

    r2 = model_regression['r2']
    n_subjects = r2.shape[0]
    times = model_regression['times']
    feature_indices = [model_regression['features'].index(feat) for feat in features_to_plot]

    fig, axes = plt.subplots(1, 1)
    lines = []

    for plot_i, f in enumerate(feature_indices):
        feature_r2 = r2[:, f, :]
        sem = np.std(feature_r2) / np.sqrt(feature_r2.shape[0])
        lines.append(axes.plot(times, np.mean(feature_r2, axis=0)))
        axes.fill_between(times, np.mean(feature_r2, axis=0) - sem, np.mean(feature_r2, axis=0) + sem, alpha=0.6)

    axes.legend([line[0] for line in lines], features_to_plot)

    axes.axvline(0, linestyle='--', color='k', zorder=0)
    axes.axhline(0, linestyle='--', color='k', zorder=0)
    axes.set_title('Decoding score')
    axes.set_ylabel('Decoding r²')
    axes.set_xlabel('Time (s)')

    # fixed_golden_aspect_ratio(axes)

    fig, axes = plt.subplots(ceil(n_subjects / 5), 5, sharex=True, sharey=True, squeeze=False)
    for i in range(n_subjects):
        axes.flat[i].plot(times, r2[i, feature_indices, :].T)
        axes.flat[i].axvline(0, linestyle='--', color='k', zorder=0)
        axes.flat[i].axhline(0, linestyle='--', color='k', zorder=0)
        # axes.flat[i].set_xlabel(None)
        # axes.flat[i].set_ylabel(None)
        # axes.flat[i].set_title(None)

        # fixed_golden_aspect_ratio(axes.flat[i])

    for i in range(n_subjects, 30):
        axes.flat[i].remove()

    plt.show()

    return


if __name__ == '__main__':

    '''decoding_predictive_features'''
    # decoding_predictive_features('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                              '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/decoding/decoding_05_12_23',
    #                               model=None,
    #                               # model='paulrnn',
    #                               # model='magenta_perfrnn',
    #                               # model='perfrnn',
    #                               feature_list=['rate', 'trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
    #                 'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
    #                 'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv', 'onset_type'],
    #                               data_type='meg',
    #                               signal_type='raw',
    #                               montage='mono',
    #                               specific_channels=None,
    #                               # specific_channels=['MEG1941', 'MEG1721', 'MEG1641', 'MEG1631', 'MEG1611', 'MEG2421', 'MEG2411', 'MEG2431', 'MEG2441', 'MEG2221'],
    #                               add_shifted=False,
    #                               # add_shifted=True,
    #                               tmin=2,
    #                               tmax=7,
    #                               n_folds=5,
    #                               alphas=np.logspace(2, 7, 6),
    #                               permutation=False,
    #                               min_time_interval=0.050,
    #                               best_channels_path=None,
    #                               # best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_10channels_100ms.npy',
    #                               baseline=False
    #                              )

    decoding_predictive_features('/home/mathieupvc/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta',
                                 '/home/mathieupvc/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta/decoding/decoding_05_12_23',
                                 model=None,
                                 # model='paulrnn',
                                 # model='magenta_perfrnn',
                                 # model='perfrnn',
                                 feature_list=['rate', 'trial_time', 'time_interval', 'pitch', 'pitch_interval',
                                               'loudness', 'brightness',
                                               'periodicity', 'roughness', 'spectralflux', 'loudness_deriv',
                                               'brightness_deriv',
                                               'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv',
                                               'onset_type'],
                                 data_type='seeg',
                                 signal_type='raw',
                                 montage='mono',
                                 specific_channels=None,
                                 # specific_channels=['MEG1941', 'MEG1721', 'MEG1641', 'MEG1631', 'MEG1611', 'MEG2421', 'MEG2411', 'MEG2431', 'MEG2441', 'MEG2221'],
                                 add_shifted=False,
                                 # add_shifted=True,
                                 tmin=2,
                                 tmax=7,
                                 n_folds=5,
                                 alphas=np.logspace(2, 7, 6),
                                 permutation=False,
                                 min_time_interval=0.050,
                                 best_channels_path=None,
                                 # best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_10channels_100ms.npy',
                                 baseline=False
                                 )

    '''spatio_temporal_cluster_regression_scores'''
    with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/decoding/decoding_05_12_23/regr_meg_acoustic.pkl', 'rb') as file:
        model_regression = pickle.load(file)
    # features_to_plot = ['surprise_mean_paulrnn']
    # features_to_plot = ['surprise_mean_paulrnn', 'shifted_entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
    # features_to_plot = ['surprise_mean_paulrnn', 'shifted_surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']
    # features_to_plot = ['surprise_possum_paulrnn', 'surprise_negsum_paulrnn', 'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']
    features_to_plot = ['rate', 'trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness',
                    'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                    'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv', 'onset_type']
    # features_to_plot = ['rate', 'time_interval', 'pitch_interval', 'loudness_deriv', 'brightness_deriv',
    #                     'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv', 'onset_type']
    spatio_temporal_cluster_regression_scores(model_regression, features_to_plot)
