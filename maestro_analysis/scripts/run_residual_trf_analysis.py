import argparse
import os

import numpy as np
from math import ceil

from maestro_analysis.trf_analysis import run_trf_all_subjects, run_acoustic_trf_all_subjects, epochs_channel_trf_all_subjects, trf_all_conditions_all_subjects, plot_trf_topomaps, plot_trf_correlation_tscore_topomap_all_subjects, plot_trf_topomaps_condition_difference, plot_trf_topomaps_individual_subjects, plot_trf_kernels_significant_channels, plot_trf_kernels_significant_channels_conditions, plot_trf_kernels_all_channels, plot_trf_kernels_all_channels_conditions, plot_trf_regression_beta_conditions

# ========================================================================
'''Argument parser'''
# ========================================================================

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data_path", type=str, help="folder containing preprocessed .fif data in individual subject folders")
parser.add_argument("-i", "--trf_input_path", type=str, help="path to .npy file containing the chosen trf input features. Should be the fullrange version (longer than sample_range so that edge effects can be removed after taking the residuals)")
parser.add_argument("-il", "--trf_input_list", nargs='+', type=str, default=None, help="list of input features names. Default is 'Envelope' 'Envelope derivative' 'Note onset' 'Surprise on' 'Surprise shift' 'Entropy on' 'Entropy shift'")
parser.add_argument("-sr", "--sample_range", nargs='+', type=int, default=None, help="list containing start and end samples used. Default is 100 800")
parser.add_argument("-af", "--acoustic_features", nargs='+', type=int, default=None, help="list containing indices of acoustic features. Default is [0, 1, 2]")
parser.add_argument("-spf", "--sparse_positive_features", nargs='+', type=int, default=None, help="list containing indices of sparse positive features. Default is [2, 4] (for Paul's RNN)")
parser.add_argument("-w", "--wav_folder", type=str, help="folder containing the .wav stimuli (ordered by trial number)")
parser.add_argument("-s", "--save_model_folder", type=str, help="output folder for saving")
parser.add_argument("-c", "--conditions", nargs='+', type=str, default=None, help="conditions, default is: 'rep1' 'rep2' 'rep3' ")
parser.add_argument("-cn", "--condition_name", type=str, default='rep', help="name of condition plots, default: 'rep'")
parser.add_argument("-ct", "--channel_type", nargs='+', type=str, default=['mag'], help="channel type to use in the analysis or list of specific channels. Default 'mag' for MEG")
parser.add_argument("-st", "--statistical_test", type=str, default='permut_cluster', help="statistical test to use: 'permut' or 'permut_cluster'")
parser.add_argument("-f", "--save_format", type=str, default='svg', help="format of the saved figures")
parser.add_argument("-fr", "--filt_range", nargs='+', type=int, default=None, help="Takes two values, high-pass and low-pass frequencies. Default is high-pass at 1Hz only")
parser.add_argument("-nat", "--no_acoustic_training", type=bool, default=False, help="does not run initial acoustic trf training")
parser.add_argument("-nt", "--no_topomap", type=bool, default=False, help="does not plot topomaps (useful when using only a subset of sensors)")
parser.add_argument("-p", "--pipeline", type=str, default='both', help="which steps to run: 'training', 'plotting', or 'both'")

args = parser.parse_args()

data_path = args.data_path
trf_input_path = args.trf_input_path
trf_input_list = args.trf_input_list
if trf_input_list is None:
    trf_input_list = ['Envelope', 'Envelope derivative', 'Note onset', 'Surprise on', 'Surprise shift', 'Entropy on', 'Entropy shift']
sample_range = args.sample_range
if sample_range is None:
    sample_range = [100, 800]
acoustic_features = args.acoustic_features
if acoustic_features is None:
    acoustic_features = [0, 1, 2]
sparse_positive_features = args.sparse_positive_features
wav_folder = args.wav_folder
save_model_folder = args.save_model_folder
conditions = args.conditions
if conditions is None:
    conditions = ['rep1', 'rep2', 'rep3']
condition_name = args.condition_name
channel_type = args.channel_type
statistical_test = args.statistical_test
save_format = args.save_format
filt_range = args.filt_range
if filt_range is None:
    filt_range = None
no_acoustic_training = args.no_acoustic_training
no_topomap = args.no_topomap
pipeline = args.pipeline
print('data_path:', data_path)
print('trf_input_path:', trf_input_path)
print('trf_input_list:', trf_input_list)
print('sample_range:', sample_range)
print('acoustic_features:', acoustic_features)
print('sparse_positive_features:', sparse_positive_features)
print('wav_folder:', wav_folder)
print('save_model_folder:', save_model_folder)
print('conditions:', conditions)
print('condition_name:', condition_name)
print('channel_type:', channel_type)
print('statistical_test:', statistical_test)
print('save_format:', save_format)
print('filt_range:', filt_range)
print('no_initial_training:', no_acoustic_training)
print('no_topomap:', no_topomap)
print('pipeline:', pipeline)
print('')


# ========================================================================
'''Write params to summary file'''
# ========================================================================
os.makedirs(save_model_folder, exist_ok=True)
file = open(os.path.join(save_model_folder, 'summary.txt'), 'w')

file.write('trf_input_list: ' + str(trf_input_list))
file.write('\nsample_range: ' + str(sample_range))
file.write('\nacoustic_features: ' + str(acoustic_features))
file.write('\nsparse_positive_features: ' + str(sparse_positive_features))
file.write('\nconditions: ' + str(conditions))
file.write('\ncondition_name: ' + str(condition_name))
file.write('\nchannel_type: ' + str(channel_type))
file.write('\nstatistical_test: ' + str(statistical_test))
file.write('\nfilt_range: ' + str(filt_range))

file.close()

# ========================================================================
'''TRF analysis'''
# ========================================================================

if pipeline=='training' or pipeline=='both':

    '''The TRF input has to be created before running this script because it is model dependent. For instance, 
    mne_analysis.maestro_analysis.create_trf_input_features_perfRNN can be used with PerformanceRNN and 
    mne_analysis.maestro_analysis.create_trf_input_features_paulRNN can be used Paul's RNN model.'''

    # TODO: the problem with this script is that it runs several trf trainings, going through all participants each time. Time consuming processing steps such as filtering
    #  and resampling are thus performed several time which is highly inefficient!

    # load trf input features and keep a longer version (used to avoid edge effects when computing residuals)
    trf_input_fullrange = np.load(trf_input_path)
    trf_input_cut = trf_input_fullrange[sample_range[0]:sample_range[1], :, :]

    if not no_acoustic_training:
        # Acoustic trf fitting, with 6-folds cross-validation and ridge parameter tuning
        run_acoustic_trf_all_subjects(data_path, trf_input_cut, wav_folder, save_model_folder, filt_range=filt_range,
                                      A_features=acoustic_features, sparse_positive_features=sparse_positive_features, channel_type=channel_type, sample_range=sample_range)

    # Residual trf
    run_trf_all_subjects(data_path, trf_input_fullrange, wav_folder, save_model_folder, filt_range=filt_range, A_features=acoustic_features,
                             sparse_positive_features=sparse_positive_features, channel_type=channel_type, sample_range=sample_range, residual_trf=True)

    # Run separate trf analyses for each condition
    trf_all_conditions_all_subjects(data_path, trf_input_fullrange, wav_folder, save_model_folder, conditions,
                                    filt_range=filt_range, channel_type=channel_type, sample_range=sample_range, residual_trf=True,
                                    A_features=acoustic_features, sparse_positive_features=sparse_positive_features)

if pipeline=='plotting' or pipeline=='both':

    # Plot trf scores topographies
    residual_folder = os.path.join(save_model_folder, 'residual_trf')
    scores_path = os.path.join(residual_folder, "scores.npy")

    if not no_topomap:
        channel_type = channel_type[0]  # the next functions do not take list
        plot_trf_correlation_tscore_topomap_all_subjects(data_path, scores_path, residual_folder, channel_type=channel_type,
                                                         stat_test=statistical_test, save_format=save_format)

        # Plot topographies of score differences between conditions
        plot_trf_topomaps_condition_difference(data_path,
                                               os.path.join(residual_folder, 'rep1', "channels_scores.npy"),
                                               os.path.join(residual_folder, 'rep3', "channels_scores.npy"),
                                               residual_folder,
                                               scores_path_optional=None,
                                               conditions=['Rep1', 'Rep3'],
                                               channel_type=channel_type,
                                               stat_test=statistical_test,
                                               save_format=save_format)

    # Plot trf kernels averaged over significant channels
    predictive_features = [x for i, x in enumerate(trf_input_list) if i not in acoustic_features]
    ncols = ceil(len(predictive_features) / 2)

    if ncols < 2:
        ncols = 2  # TODO: clean this (if only one ax, .T bugs in the following functions)

    if no_topomap:
        channel_type = 'specific_channels'

    plot_trf_kernels_significant_channels(data_path, scores_path, residual_folder, features=predictive_features, channel_type=channel_type,
                                          indiv_subjects=True, ncols=ncols, zscore=False, rms=True, stat_test=statistical_test,
                                          save_format=save_format)
    plot_trf_kernels_significant_channels(data_path, scores_path, residual_folder, features=predictive_features,
                                          channel_type=channel_type,
                                          indiv_subjects=False, ncols=ncols, zscore=False, rms=True, stat_test=statistical_test,
                                          save_format=save_format)

    # Plot trf kernels averaged over all channels
    plot_trf_kernels_all_channels(data_path, residual_folder, features=predictive_features,
                                   channel_type=channel_type, acoust_feat=False,
                                   indiv_subjects=True, ncols=ncols, save_format=save_format)
    plot_trf_kernels_all_channels(data_path, residual_folder, features=predictive_features,
                                   channel_type=channel_type, acoust_feat=False,
                                   indiv_subjects=False, ncols=ncols, save_format=save_format)

    # Plot trf kernels averaged over all channels for each condition
    plot_trf_kernels_all_channels_conditions(data_path, residual_folder, conditions, condition_name,
                                             features=predictive_features, channel_type=channel_type, ncols=ncols,
                                             save_format=save_format)
    # Plot trf kernels averaged on a given number of best channels for each condition
    plot_trf_kernels_significant_channels_conditions(data_path, scores_path, residual_folder, conditions, condition_name,
                                                     features=predictive_features, channel_type=channel_type, ncols=ncols,
                                                     stat_test=statistical_test, save_format=save_format)

    # Linear regression on conditions
    plot_trf_regression_beta_conditions(data_path, scores_path, residual_folder, conditions, condition_name,
                                        features=predictive_features, channel_type=channel_type,
                                        significant_channels_only=True, ncols=ncols, stat_test=statistical_test,
                                        save_format=save_format)
    plot_trf_regression_beta_conditions(data_path, scores_path, residual_folder, conditions, condition_name,
                                        features=predictive_features, channel_type=channel_type,
                                        significant_channels_only=False, ncols=ncols, stat_test=statistical_test,
                                        save_format=save_format)
