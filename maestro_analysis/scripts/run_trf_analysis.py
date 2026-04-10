import argparse
import os

from math import ceil

from maestro_analysis.trf_analysis import run_trf_all_subjects, epochs_channel_trf_all_subjects, trf_all_conditions_all_subjects, plot_trf_topomaps, plot_trf_topomaps_all_subjects_only, plot_trf_topomaps_condition_difference, plot_trf_topomaps_individual_subjects, plot_trf_kernels_best_channels, plot_trf_kernels_best_channels_conditions, plot_trf_kernels_all_channels, plot_trf_kernels_all_channels_conditions, plot_trf_kernels_significant_channels, plot_trf_kernels_significant_channels_conditions

# ========================================================================
'''Argument parser'''
# ========================================================================

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data_path", type=str, help="folder containing preprocessed .fif data in individual subject folders")
parser.add_argument("-i", "--trf_input_path", type=str, help="path to .npy file containing the chosen trf input features")
parser.add_argument("-il", "--trf_input_list", nargs='+', type=str, default=None, help="list of input features names. Default is 'Envelope' 'Envelope derivative' 'Note onset 'Surprise on' 'Surprise shift' 'Entropy on' 'Entropy shift'")
parser.add_argument("-sr", "--sample_range", nargs='+', type=int, default=None, help="list containing start and end samples used. Default is 100 800")
parser.add_argument("-af", "--acoustic_features", nargs='+', type=int, default=None, help="list containing indices of acoustic features. Default is [0, 1, 2]")
parser.add_argument("-w", "--wav_folder", type=str, help="folder containing the .wav stimuli (ordered by trial number)")
parser.add_argument("-s", "--save_model_folder", type=str, help="output folder for saving")
parser.add_argument("-c", "--conditions", nargs='+', type=str, default=None, help="conditions, default is: 'rep1' 'rep2' 'rep3' ")
parser.add_argument("-cn", "--condition_name", type=str, default='rep', help="name of condition plots, default: 'rep'")
parser.add_argument("-ct", "--channel_type", nargs='+', type=str, default='mag', help="channel type to use in the analysis or list of specific channels. Default 'grad' for MEG")
parser.add_argument("-st", "--statistical_test", type=str, default='permut_cluster', help="statistical test to use: 'permut' or 'permut_cluster'")
parser.add_argument("-nc", "--nb_channels", type=int, default=5, help="number of best channels to average to obtain trf kernels")
parser.add_argument("-f", "--save_format", type=str, default='svg', help="format of the saved figures")
parser.add_argument("-fr", "--filt_range", nargs='+', type=int, default=None, help="Takes two values, high-pass and low-pass frequencies. Default is high-pass at 1Hz only")
parser.add_argument("-nit", "--no_initial_training", type=bool, default=False, help="does not run initial trf training")
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
wav_folder = args.wav_folder
save_model_folder = args.save_model_folder
conditions = args.conditions
if conditions is None:
    conditions = ['rep1', 'rep2', 'rep3']
condition_name = args.condition_name
channel_type = args.channel_type
statistical_test = args.statistical_test
nb_channels = args.nb_channels
save_format = args.save_format
filt_range = args.filt_range
if filt_range is None:
    filt_range = None
no_initial_training = args.no_initial_training
no_topomap = args.no_topomap
pipeline = args.pipeline
print('data_path:', data_path)
print('trf_input_path:', trf_input_path)
print('trf_input_list:', trf_input_list)
print('sample_range:', sample_range)
print('acoustic_features:', acoustic_features)
print('wav_folder:', wav_folder)
print('save_model_folder:', save_model_folder)
print('conditions:', conditions)
print('condition_name:', condition_name)
print('channel_type:', channel_type)
print('statistical_test:', statistical_test)
print('nb_channels:', nb_channels)
print('save_format:', save_format)
print('filt_range:', filt_range)
print('no_initial_training:', no_initial_training)
print('no_topomap:', no_topomap)
print('pipeline:', pipeline)
print('')


# ========================================================================
'''TRF analysis'''
# ========================================================================

if pipeline=='training' or pipeline=='both':

    '''The TRF input has to be created before running this script because it is model dependent. For instance, 
    mne_analysis.maestro_analysis.create_trf_input_features_perfRNN can be used with PerformanceRNN and 
    mne_analysis.maestro_analysis.create_trf_input_features_paulRNN can be used Paul's RNN model.'''

    # TODO: the problem with this script is that it runs several trf trainings, going through all participants each time. Time consuming processing steps such as filtering
    #  and resampling are thus performed several time which is highly inefficient!

    if not no_initial_training:
        # First trf fitting, with 10-folds cross-validation and ridge parameter tuning
        run_trf_all_subjects(data_path, trf_input_path, wav_folder, save_model_folder, filt_range=filt_range, A_features=acoustic_features, channel_type=channel_type, sample_range=sample_range)

    # Second trf fitting, with leave-one-out cross-validation which enables individual epoch score estimation (used later in statistical testing)
    # epochs_channel_trf_all_subjects(data_path, trf_input_path, wav_folder, save_model_folder, filt_range=None, A_features=None)

    # Run separate trf analyses for each condition
    trf_all_conditions_all_subjects(data_path, trf_input_path, wav_folder, save_model_folder, conditions, filt_range=filt_range, channel_type=channel_type, sample_range=sample_range)


if pipeline=='plotting' or pipeline=='both':

    # Plot trf scores topographies
    scores_path = os.path.join(save_model_folder, "scores.npy")

    if not no_topomap:
        if isinstance(channel_type, list):  # the next functions do not take list
            channel_type = channel_type[0]

        plot_trf_topomaps_all_subjects_only(data_path, scores_path, save_model_folder, channel_type=channel_type, stat_test=statistical_test, save_format=save_format)

        # Plot topographies of score differences between conditions
        plot_trf_topomaps_condition_difference(data_path,
                                               os.path.join(save_model_folder, 'rep1', "channels_scores.npy"),
                                               os.path.join(save_model_folder, 'rep2', "channels_scores.npy"),
                                               save_model_folder,
                                               scores_path_optional=os.path.join(save_model_folder, 'rep3', "channels_scores.npy"),
                                               conditions=['Rep1', 'Avg(Rep2, Rep3)'],
                                               channel_type=channel_type,
                                               stat_test=statistical_test,
                                               save_format=save_format)

        # Plot AM-A topographies for all subjects
        # plot_trf_topomaps_individual_subjects(data_path, scores_path, save_model_folder, channel_type=channel_type, stat_test=statistical_test, save_format=save_format)

    # Plot trf kernels averaged on a given number of best channels
    ncols = ceil(len(trf_input_list) / 2)

    if no_topomap:
        channel_type = 'specific_channels'

    # plot_trf_kernels_best_channels(data_path, scores_path, save_model_folder, features=trf_input_list,
    #                                channel_type=channel_type, nb_channels=nb_channels, acoust_feat=False,
    #                                indiv_subjects=True, ncols=ncols, save_format=save_format)
    plot_trf_kernels_significant_channels(data_path, scores_path, save_model_folder, features=trf_input_list,
                                          channel_type=channel_type, indiv_subjects=True, ncols=ncols, zscore=False,
                                          rms=True, stat_test=statistical_test, save_format=save_format)
    # plot_trf_kernels_best_channels(data_path, scores_path, save_model_folder, features=trf_input_list,
    #                                channel_type=channel_type, nb_channels=nb_channels, acoust_feat=False,
    #                                indiv_subjects=False, ncols=ncols, save_format=save_format)
    plot_trf_kernels_significant_channels(data_path, scores_path, save_model_folder, features=trf_input_list,
                                          channel_type=channel_type, indiv_subjects=False, ncols=ncols, zscore=False,
                                          rms=True, stat_test=statistical_test, save_format=save_format)

    # Plot trf kernels averaged over all channels
    plot_trf_kernels_all_channels(data_path, save_model_folder, features=trf_input_list,
                                   channel_type=channel_type, acoust_feat=False,
                                   indiv_subjects=True, ncols=ncols, save_format=save_format)
    plot_trf_kernels_all_channels(data_path, save_model_folder, features=trf_input_list,
                                   channel_type=channel_type, acoust_feat=False,
                                   indiv_subjects=False, ncols=ncols, save_format=save_format)

    # Plot trf kernels averaged on a given number of best channels for each condition
    # plot_trf_kernels_best_channels_conditions(data_path, save_model_folder, conditions, condition_name,
    #                                           features=trf_input_list, channel_type=channel_type,
    #                                           nb_channels=nb_channels, ncols=ncols, save_format=save_format)
    plot_trf_kernels_significant_channels_conditions(data_path, scores_path, save_model_folder, conditions,
                                                     condition_name, features=trf_input_list, channel_type=channel_type,
                                                     ncols=ncols, stat_test=statistical_test, save_format=save_format)

    # Plot trf kernels averaged over all channels for each condition
    plot_trf_kernels_all_channels_conditions(data_path, save_model_folder, conditions, condition_name,
                                             features=trf_input_list, channel_type=channel_type, ncols=ncols,
                                             save_format=save_format)
