import os
import pickle

from math import ceil

from maestro_analysis.trf_analysis import run_trf_all_subjects, trf_correct_incorrect, plot_trf_topomaps_condition_difference, plot_trf_kernels_best_channels_conditions, plot_trf_kernels_all_channels_conditions


data_path = '/home/mathieu/Data/meg_maestro/preprocessed'
trf_input_path = '/home/mathieu/Data/meg_maestro/trf_input_on_shift_new.npy'
wav_folder = '/home/mathieu/Data/meg_maestro/exp_stim_meg'
save_model_folder = '/home/mathieu/Data/meg_maestro/trf_mag_correct_time'
filt_range = [1,8]
channel_type = 'mag'
sample_range = [100, 800]
with open('/home/mathieu/Data/meg_maestro/correct_incorrect_lists.pkl', 'rb') as f:
    correct_incorrect_dict = pickle.load(f)
statistical_test = 'permut_cluster'
save_format = 'svg'
trf_input_list = ['Envelope', 'Envelope derivative', 'Surprise on', 'Surprise shift', 'Entropy on', 'Entropy shift']
nb_channels = 5
conditions = ['correct', 'incorrect']
condition_name = 'correct_incorrect'
remove_subjects = [5,20]

#run_trf_all_subjects(data_path, trf_input_path, wav_folder, save_model_folder, filt_range=filt_range, A_features=None, channel_type=channel_type, sample_range=sample_range)

# Run separate trf analyses for each condition
trf_correct_incorrect(data_path, trf_input_path, wav_folder, save_model_folder, correct_incorrect_dict, filt_range=filt_range, channel_type=channel_type, sample_range=sample_range)

# Plot trf scores topographies
# scores_path = os.path.join(save_model_folder, "scores.npy")

# Plot topographies of score differences between conditions
plot_trf_topomaps_condition_difference(data_path,
                                       os.path.join(save_model_folder, 'correct', "channels_scores.npy"),
                                       os.path.join(save_model_folder, 'incorrect', "channels_scores.npy"),
                                       save_model_folder,
                                       conditions=['correct', 'incorrect'],
                                       channel_type=channel_type,
                                       stat_test=statistical_test,
                                       save_format=save_format,
                                       plot_name='correct_incorrect')

# Plot trf kernels averaged on a given number of best channels
ncols = ceil(len(trf_input_list) / 2)

# Plot trf kernels averaged on a given number of best channels for each condition
plot_trf_kernels_best_channels_conditions(data_path, save_model_folder, conditions, condition_name,
                                          features=trf_input_list, channel_type=channel_type,
                                          nb_channels=nb_channels, ncols=ncols, save_format=save_format, remove_subjects=remove_subjects)

# Plot trf kernels averaged over all channels for each condition
plot_trf_kernels_all_channels_conditions(data_path, save_model_folder, conditions, condition_name,
                                         features=trf_input_list, channel_type=channel_type, ncols=ncols,
                                         save_format=save_format, remove_subjects=remove_subjects)
