import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mne
from mne.preprocessing import find_bad_channels_maxwell, ICA, create_eog_epochs, create_ecg_epochs

from maestro_analysis.preprocessing_functions import bad_channels_visu, concatenate_and_Maxfilter_blocks


subject_folder = '/home/mathieupvc/These/data/meg_maestro/data/maestro01_itdheb/210629'

'''Load only one block'''
sample_data_raw_file = os.path.join(subject_folder, 'run01.fif')
print(sample_data_raw_file)
raw = mne.io.read_raw_fif(sample_data_raw_file, allow_maxshield=True)

'''Load all blocks and apply SSS'''
sample_data_raw_file_list = [os.path.join(subject_folder, 'run0' + str(i) + '.fif') for i in range(1,3)]
fine_cal_file = os.path.join(subject_folder, 'sss_config', 'sss_cal_3101_160108.dat')
crosstalk_file = os.path.join(subject_folder, 'sss_config', 'ct_sparse.fif')

raw = concatenate_and_Maxfilter_blocks(sample_data_raw_file_list, fine_cal_file, crosstalk_file, bads=['MEG0341'],
                                       channel_types=['mag', 'grad'])
                                       # channel_types=None)

raw.plot(duration=25, n_channels=30)
plt.show()

# raw_list = [mne.io.read_raw_fif(sample, allow_maxshield=True) for sample in sample_data_raw_file_list]

# # sss_list = [mne.preprocessing.maxwell_filter(raw, cross_talk=crosstalk_file, calibration=fine_cal_file, verbose=True,
# #                                              destination=sample_data_raw_file_list[0])
# #             for raw in raw_list]
#
# sss_list = []
# fine_cal_file = os.path.join(subject_folder, 'sss_config', 'sss_cal_3101_160108.dat')
# crosstalk_file = os.path.join(subject_folder, 'sss_config', 'ct_sparse.fif')
# for raw in raw_list:
#
#     '''Since the MEG recording was maxshielded, we need to apply maxwell_filter()'''
#     # First we look for bad channels
#     raw_check = raw.copy()
#     auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
#         raw_check, cross_talk=crosstalk_file, calibration=fine_cal_file,
#         return_scores=True, verbose=True)
#     print('Noisy channels:', auto_noisy_chs)
#     print('Flat channels:', auto_flat_chs)
#     raw.info['bads'] = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
#
#     '''Inspection of noisy and flat channels'''
#     # raw.plot(duration=5, n_channels=30)
#     # bad_channels_visu(['mag', 'grad'], auto_scores)
#     raw.info['bads'] = ['MEG0341']
#
#     sss_raw = mne.preprocessing.maxwell_filter(raw, cross_talk=crosstalk_file, calibration=fine_cal_file, verbose=True,
#                                      destination=sample_data_raw_file_list[0])
#
#     sss_list.append(sss_raw)
# del raw_check
#
# raw = mne.concatenate_raws(sss_list)
# del raw_list, sss_list


print(raw.info)
print(raw.info['ch_names'])
print(len(raw.info['ch_names']))
print(raw.info['dev_head_t'])


'''Since the MEG recording was maxshielded, we need to apply maxwell_filter()'''
fine_cal_file = os.path.join(subject_folder, 'sss_config', 'sss_cal_3101_160108.dat')
crosstalk_file = os.path.join(subject_folder, 'sss_config', 'ct_sparse.fif')
# First we look for bad channels
raw_check = raw.copy()
auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
    raw_check, cross_talk=crosstalk_file, calibration=fine_cal_file,
    return_scores=True, verbose=True)
print(auto_noisy_chs)
print(auto_flat_chs)
raw.info['bads'] = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
print(raw.info['bads'])



'''Add bad channels after visual inspection (can also be done manually when plotting the signals)'''
raw.plot(duration=5, n_channels=30)
plt.show()
# # raw.plot_psd(fmax=350)
# # raw.info['bads'] += ['MEG0341']
# print(raw.info['bads'])
#
#
# # Actual Maxwell filter
raw_sss = mne.preprocessing.maxwell_filter(raw, cross_talk=crosstalk_file, calibration=fine_cal_file, verbose=True)

raw_sss.plot(duration=5, n_channels=30)
plt.show()

# print(raw.info)
#
# bio = raw.copy().pick(['BIO003'])
# bio.plot(duration=5, n_channels=30, scalings=500e-5)
# plt.show()
#
# # Print MEG and PSD
# # raw.plot(duration=5, n_channels=30)
# raw.plot_psd(fmax=350)
# # raw_sss.plot(duration=5, n_channels=30)
# # raw_sss.plot_psd(fmax=350)

'''ICA'''
# # First, we visualize artifacts
# ecg_epochs = create_ecg_epochs(raw, ch_name='BIO003')
# ecg_epochs.plot_image(combine='mean')
# del ecg_epochs
#
# ecg_evoked = create_ecg_epochs(raw).average()
# ecg_evoked.apply_baseline(baseline=(None, -0.2))
# ecg_evoked.plot_joint()
# del ecg_evoked
#
# eog_epochs = create_eog_epochs(raw, ch_name=['BIO001', 'BIO002'])
# eog_epochs.plot_image(combine='mean')
# del eog_epochs
#
#
# # High-pass filter to remove slow drifts
# filt_raw = raw.copy().filter(l_freq=1., h_freq=None)
#
# # set up and fit the ICA
# ica = ICA(n_components=30, max_iter='auto', random_state=97)
# ica.fit(filt_raw)
# del filt_raw
# print(ica)
#
# # Examine ICs
# raw.load_data()
# ica.plot_sources(raw, show_scrollbars=True)
# ica.plot_components() # plots 20 components maximum at a time
# plt.show()
#
# # remove 1st component (blinks)
# # print(ica.exclude)
# # ica.plot_overlay(raw_sss, exclude=[0], picks='mag', stop=20.)
# # plt.show()
# # ica.plot_properties(raw_sss, picks=[0,1])
# # ica.exclude = [0] # We choose the components to exclude based on the previous figures
#
# # ica.apply() changes the Raw object in-place, so let's make a copy first:
# reconst_raw = raw.copy()
# ica.apply(reconst_raw)
#
# raw.plot(n_channels=30, show_scrollbars=True)
# reconst_raw.plot(n_channels=30, show_scrollbars=True)
# plt.show()
#
# del raw
# # del raw_sss
#
# reconst_raw.plot_psd(fmax=350)

'''Triggers'''
# Detecting triggers
# events = mne.find_events(reconst_raw, stim_channel='STI101')
events = mne.find_events(raw, stim_channel='STI101', shortest_event=1)
# print(events[:30])  # show the first 5
# print(events[events[:,2]==1])
events = events[events[:,2]!=1]
# print(events[:30])  # show the first 5

# event_dict = {'block_start': 1, '2.5Hz': 2, '3.5Hz': 4,
#               '4.5Hz': 8, '5.5Hz': 16, '6.5Hz': 32, 'resp_new': 64, 'resp_old':65}
event_dict = {'2.5Hz': 2, '3.5Hz': 4,
              '4.5Hz': 8, '5.5Hz': 16, '6.5Hz': 32, 'resp_new': 64, 'resp_old':65}

# fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=reconst_raw.info['sfreq'], first_samp=reconst_raw.first_samp)
# fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw.info['sfreq'], first_samp=raw.first_samp)


'''Annotations'''
print(raw.annotations)
for ann in raw.annotations:
    descr = ann['description']
    start = ann['onset']
    end = ann['onset'] + ann['duration']
    print("'{}' goes from {} to {}".format(descr, start, end))


'''Epoching'''
reject_criteria = dict(mag=4000e-15,     # 4000 fT
                       grad=4000e-13,)   # 4000 fT/cm
                       # eeg=150e-6,     # 150 µV
                       # eog=250e-6)     # 250 µV

epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=10, preload=True)  # reject=reject_criteria,
print(epochs.drop_log)
print(epochs.selection)
#
# epochs['2.5Hz'].plot_image()
# epochs['3.5Hz'].plot_image()
# epochs['4.5Hz'].plot_image()


'''Load epochs'''
# epochs = mne.read_epochs('/home/mathieupvc/These/data/meg_maestro/preprocessed/maestro01_itdheb/210629_epo.fif')
#
# print(epochs)
# epochs.pick(['meg', 'eeg'])
# epochs.crop(tmax=0.5)
# listening_epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]
# response_epochs = epochs[['resp_new', 'resp_old']]
# del epochs
#
# # epochs.plot(n_epochs=10)
# # epochs['2.5Hz'].plot(n_epochs=10)
# # epochs['resp_new'].plot(n_epochs=10)
# # plt.show()
#
# print(listening_epochs)
# listening_evoked = listening_epochs.average()
# print(listening_evoked)
# listening_evoked.plot_joint()
# # plt.show()