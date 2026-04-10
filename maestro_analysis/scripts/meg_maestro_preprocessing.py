import argparse
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import mne
from mne.preprocessing import ICA

from maestro_analysis.preprocessing_functions import concatenate_and_Maxfilter_blocks


# ========================================================================
'''Argument parser'''
# ========================================================================

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--subject_folder", type=str, help="folder containing .fif files")
parser.add_argument("-f", "--fine_cal_file", type=str)
parser.add_argument("-ct", "--crosstalk_file", type=str)

def none_or_list(value):
    if value == 'None':
        return None
    return value
parser.add_argument("-ch", "--channel_types", type=none_or_list, default=['mag', 'grad'],
                    help="set it to None if you do not want to manually select bad channels")

parser.add_argument("-b", "--bad_channels", nargs='+', type=str, default=['MEG0341'])
parser.add_argument("-d", "--destination_path", type=str, help="destination folder")
parser.add_argument("-w", "--wav_folder", type=str, help="folder containing the .wav stimuli (ordered by trial number)")
args = parser.parse_args()

subject_folder = args.subject_folder
fine_cal_file = args.fine_cal_file
crosstalk_file = args.crosstalk_file
channel_types = args.channel_types
bad_channels = args.bad_channels
destination_path = args.destination_path
wav_folder = args.wav_folder
print('subject_folder:', subject_folder)
print('fine_cal_file:', fine_cal_file)
print('crosstalk_file:', crosstalk_file)
print('channel_types:', channel_types)
print('bad_channels:', bad_channels)
print('destination_path:', destination_path)
print('wav_folder:', wav_folder)
print('')


# ========================================================================
'''Preprocessing'''
# ========================================================================

# Loading
if pathlib.Path(subject_folder).parts[-2] == 'maestro25_qlmdne':  # Subject 25 has two run06 (a and b) that need to be concatenated
    sample_data_raw_file_list = [os.path.join(subject_folder, 'run' + str(i).zfill(2) + '.fif') for i in range(1,6)] + \
                                [os.path.join(subject_folder, 'run06a.fif'), os.path.join(subject_folder, 'run06b.fif')] + \
                                [os.path.join(subject_folder, 'run' + str(i).zfill(2) + '.fif') for i in range(7,10)]
else:
    sample_data_raw_file_list = [os.path.join(subject_folder, 'run' + str(i).zfill(2) + '.fif') for i in range(1,10)]

# Maxwell filter and concatenation of 9 blocks
raw = concatenate_and_Maxfilter_blocks(sample_data_raw_file_list, fine_cal_file, crosstalk_file, bads=bad_channels,
                                       channel_types=channel_types)

print(raw.info)
print('Channel names:', raw.info['ch_names'])

# Notch filter
raw.notch_filter(np.arange(50, 251, 50), picks='meg', n_jobs=-1)

# Inspection for annotations
raw.plot(duration=30, n_channels=30)
plt.show()

# ICA
# High-pass filter to remove slow drifts
filt_raw = raw.copy().filter(l_freq=1., h_freq=None)

# set up and fit the ICA
ica = ICA(n_components=30, max_iter='auto', random_state=97)
ica.fit(filt_raw)
del filt_raw
print(ica)

# # Examine ICs
# raw.load_data()
# ica.plot_sources(raw, show_scrollbars=True)
# ica.plot_components() # plots 20 components maximum at a time
# plt.show()

# Automatic component selection
# Eye movements
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['BIO001', 'BIO002'])
ica.exclude.extend(eog_indices)

# Heart beats
ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='BIO003', method='correlation', threshold='auto')
ica.exclude.extend(ecg_indices)

# Saving IC figure
ica.plot_sources(raw, show_scrollbars=False, show=False)
plt.savefig(os.path.join(destination_path, 'ica_components.png'))
ica.plot_components(picks=[i for i in range(20)], show=False) # plots 20 components maximum at a time
plt.savefig(os.path.join(destination_path, 'ica_topographies.png'))

# ica.apply() changes the Raw object in-place, so let's make a copy first:
reconst_raw = raw.copy()
ica.apply(reconst_raw)
del raw

# Saving
path = pathlib.PurePath(subject_folder)
reconst_raw.save(os.path.join(destination_path, path.name + 'raw_sss.fif'), overwrite=True)
print('Saved in', destination_path)


# # ========================================================================
# '''Epoching'''
# # ========================================================================
#
# # Detecting triggers
# events = mne.find_events(reconst_raw, stim_channel='STI101', shortest_event=1)
# events = events[events[:,2]!=1]  # Remove event 1 (useless and not consistent across subjects)
#
# if pathlib.Path(subject_folder).parts[-2] == 'maestro00_hoyzez':
#     event_dict = {'2.5Hz': 2, '3.5Hz': 4,
#               '4.5Hz': 8, '5.5Hz': 16, '6.5Hz': 32}
# else:
#     event_dict = {'2.5Hz': 2, '3.5Hz': 4,
#               '4.5Hz': 8, '5.5Hz': 16, '6.5Hz': 32, 'resp_new': 64, 'resp_old':65}
#
# # fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=reconst_raw.info['sfreq'], first_samp=reconst_raw.first_samp)
#
#
# # Metadata
# metadata = pd.DataFrame("", index=[i for i in range(events.shape[0])], columns=['stim_name'])
# wav_folder = (sorted(list(pathlib.Path(wav_folder).glob('**/*.wav'))))
# if pathlib.Path(subject_folder).parts[-2] == 'maestro23_shvfbp':
#     print(wav_folder[30])
#     del wav_folder[30]
# if pathlib.Path(subject_folder).parts[-2] == 'maestro00_hoyzez':
#     print(wav_folder[10:12])
#     del wav_folder[10:12]
#
# stim_index = 0
# for i in range(len(metadata)):
#     if events[i, 2] in [2, 4, 8, 16, 32]:
#         metadata.at[i, 'stim_name'] = wav_folder[stim_index].name
#         stim_index += 1
#
#
# '''Epoching'''
# reject_criteria = dict(mag=4000e-15,     # 4000 fT
#                        grad=4000e-13,)   # 4000 fT/cm
#                        # eeg=150e-6,     # 150 µV
#                        # eog=250e-6)     # 250 µV
#
# epochs = mne.Epochs(reconst_raw, events, event_id=event_dict, tmin=-0.2, tmax=10, preload=True, metadata=metadata)  # reject=reject_criteria,
#
# # Saving
# path = pathlib.PurePath(subject_folder)
# epochs.save(os.path.join(destination_path, path.name + '_epo.fif'), overwrite=True)
# print('Saved in', destination_path)
