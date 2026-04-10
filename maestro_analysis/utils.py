import pathlib
import os
import copy
import mat73
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import itertools

from mido import MidiFile

from sklearn.model_selection import KFold

import mne





def cross_val_multidim_score(rf, alpha, X, Y, n_splits=10):
    '''Custom cross-validation of the RF model that enables working on multidimensional score output'''

    cv = KFold(n_splits)
    for train, test in cv.split(X, Y):
        print(train)
        print(test)

    return


def epoching_maestro(raw_path, wav_folder, allow_maxshield=False, tmin=-0.5, baseline=True, bin_ecg=False, channel_type=None, remove_responses=True):
    """
    From raw_path to a .fif raw file, load the raw data and return the epoched data. wav_folder is used to create
    metadata
    tmin: start time of epochs, default is -0.5
    """
    raw = mne.io.read_raw_fif(raw_path, preload=True, allow_maxshield=allow_maxshield)

    # Detecting triggers
    events = mne.find_events(raw, stim_channel='STI101', shortest_event=1)
    events = events[events[:, 2] != 1]  # Remove event 1 (useless and not consistent across subjects)
    if pathlib.Path(raw_path).parts[-2] == 'maestro00_hoyzez':
        event_dict = {'2.5Hz': 2, '3.5Hz': 4,
                  '4.5Hz': 8, '5.5Hz': 16, '6.5Hz': 32}
    else:
        event_dict = {'2.5Hz': 2, '3.5Hz': 4,
                  '4.5Hz': 8, '5.5Hz': 16, '6.5Hz': 32, 'resp_new': 64, 'resp_old':65}
    if remove_responses:
        events = events[events[:, 2] != 64]  # Remove subjects response
        events = events[events[:, 2] != 65]
        event_dict = {'2.5Hz': 2, '3.5Hz': 4,
                      '4.5Hz': 8, '5.5Hz': 16, '6.5Hz': 32}

    # fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=reconst_raw.info['sfreq'], first_samp=reconst_raw.first_samp)

    if bin_ecg:
        raw.pick('BIO003')
        ecg_events, ch_ecg, average_pulse, ecg = mne.preprocessing.find_ecg_events(raw, ch_name='BIO003', return_ecg=True, reject_by_annotation=False)
        bin_ecg = np.zeros_like(ecg)
        bin_ecg[0, ecg_events[:, 0] - raw.first_samp] = 1
        ch_names = ['bin_ecg']
        ch_types = ['ecg']
        info = mne.create_info(ch_names, raw.info['sfreq'], ch_types)
        new_raw = mne.io.RawArray(bin_ecg, info, raw.first_samp)
        raw = raw.add_channels([new_raw], force_update_info=True)

    if channel_type is not None:
        raw.pick(channel_type)

    # Metadata
    metadata = pd.DataFrame("", index=[i for i in range(events.shape[0])], columns=['stim_name'])
    wav_folder = (sorted(list(pathlib.Path(wav_folder).glob('**/*.wav'))))
    if pathlib.Path(raw_path).parts[-2] == 'maestro23_shvfbp':
        print(wav_folder[30])
        del wav_folder[30]
    if pathlib.Path(raw_path).parts[-2] == 'maestro00_hoyzez':
        print(wav_folder[10:12])
        del wav_folder[10:12]

    stim_index = 0
    for i in range(len(metadata)):
        if events[i, 2] in [2, 4, 8, 16, 32]:
            metadata.at[i, 'stim_name'] = wav_folder[stim_index].name
            stim_index += 1


    '''Epoching'''
    reject_criteria = dict(mag=4000e-15,     # 4000 fT
                           grad=4000e-13,)   # 4000 fT/cm
                           # eeg=150e-6,     # 150 µV
                           # eog=250e-6)     # 250 µV

    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=tmin, tmax=10, baseline=None, preload=True, metadata=metadata)  # reject=reject_criteria,
    epochs.shift_time(-0.010)  # removes the 10ms offset of the triggers
    epochs.crop(tmin=tmin)
    if baseline:
        epochs.apply_baseline((None, -0.1))

    return epochs


def save_meg_epochs(data_path, wav_folder, save_path, channel_type='mag', n_jobs=-1):
    os.makedirs(save_path, exist_ok=True)

    path = pathlib.Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        epochs = epoching_maestro(raw_path, wav_folder, allow_maxshield=False, tmin=-0.5, baseline=True, bin_ecg=False,
                                  channel_type=channel_type, remove_responses=True)

        # preprocessing
        fs = 200
        epochs.resample(fs, n_jobs=n_jobs)
        epochs.filter(0.5, None, n_jobs=n_jobs)  # highpass filter

        epochs.save(os.path.join(save_path, subject_path.name + "-epo.fif"))
        del epochs

    return


def find_surp_entr_files(wav_name, surprise_dir, entropy_dir, add_mid=True):
    '''
    From wav_name, find the corresponding surprise and entropy files paths
    '''

    split = wav_name.split('_')
    if add_mid:
        surprise_file = os.path.join(surprise_dir, split[2], split[3] + '.mid.csv')
        entropy_file = os.path.join(entropy_dir, split[2], split[3] + '.mid.csv')
    else:
        surprise_file = os.path.join(surprise_dir, split[2], split[3] + '.csv')
        entropy_file = os.path.join(entropy_dir, split[2], split[3] + '.csv')

    return surprise_file, entropy_file

def find_predictive_features_files(wav_name, pred_feat_dir):
    '''
    From wav_name, find the corresponding surprise and entropy files paths
    '''

    split = wav_name.split('_')
    predictive_features_file = os.path.join(pred_feat_dir, split[2], split[3] + '.csv')

    return predictive_features_file


def find_midi_file(wav_name, midi_folder):
    '''
    From wav_name, find the corresponding midi clip file paths
    '''

    split = wav_name.split('_')
    midi_file = os.path.join(midi_folder, split[2], split[3] + '.mid')

    return midi_file


def convert_to_pitch(note_list):
    '''
    Convert midi notes to the range [0, 11].
    :param note_list:
    :return: pitch_list
    '''

    pitch_list = []
    for note in note_list:
        pitch_list.append(note % 12)

    return pitch_list


def convert_to_frequency(note_list):
    '''
    Convert midi notes to frequency.
    :param note_list:
    :return: pitch_list
    '''

    pitch_list = []
    for note in note_list:
        pitch_list.append(2**((note-69)/12) * 440)

    return pitch_list


def retrieve_acoustic_regressors(wav_folder, acoustic_folder):
    '''
    :param path: Path to a folder containing "midi", "wav" and "acoustic" subfolders.
    :return: Matrix X of dimensions n_timepoints * n_trials * n_features, a list of regressor names
    '''
    acoustic_regressors = []  # TODO: remove?
    stim_dur = 8  # TODO: parameterize
    fs = 100
    wav_folder = (sorted(list(pathlib.Path(wav_folder).glob('**/*.wav'))))  # TODO: remove?

    #acoustic regressors from Bruno
    auddim1 = mat73.loadmat(os.path.join(acoustic_folder, 'AuditoryDimensions_Clean_batch01.mat'))
    auddim2 = mat73.loadmat(os.path.join(acoustic_folder, 'AuditoryDimensions_Clean_batch02.mat'))
    X = np.concatenate((auddim1['X'], auddim2['X']), axis=2)
    dims = copy.deepcopy(auddim1['names'])
    X = np.moveaxis(X, [0, 1, 2], [0, 2, 1]) #into dims : time * trials * features
    X = np.delete(X, obj=2, axis=2) #remove pitch
    dims.remove('pitch')
    X = mne.filter.resample(X, down=auddim1["fs"]/fs, npad='auto', axis=0) #downsample
    X = X[0:(stim_dur+1)*fs, :, :] #crop the end

    X_SF = np.load(os.path.join(acoustic_folder, 'spectral_flux.npy'))
    X = np.concatenate((X, X_SF), axis=2)

    #compute derivatives
    D = np.diff(X, axis=0, prepend = np.zeros((1, X.shape[1], X.shape[2])))
    D[:, :, 0] = np.where(D[:, :, 0] < 0, 0, D[:, :, 0])
    X = np.concatenate((X, D), axis=2)

    acoustic_regressors = copy.deepcopy(dims)
    for d in dims:
        acoustic_regressors.append(str(d+'_deriv'))

    return X, acoustic_regressors


def epoching_maestro_note_level(raw_path, wav_folder, midi_folder, acoustic_folder, perfrnn_dir, magenta_perfrnn_dir, paulrnn_dir, t_min=-1, t_max=1, channel_type='mag', downsample=True, detrend=True, lowpass=True, multiple_notes='max', n_jobs=-1):
    '''
    :param raw_path:
    :param wav_folder:
    :param midi_folder:
    :param perfrnn_dir: path to folder than contains 'surprise' and 'uncertainty' subfolders
    :param paulrnn_dir:
    :param t_min: minimum t for time-window
    :param t_max: maximum t for time-window
    :return: epoch object
    '''

    #parameters
    stim_dur = 8  # TODO: parameterize
    time_bin = 0.01
    fs = int(1/time_bin)

    #preprocessing
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    if downsample:
        raw.resample(100, n_jobs=n_jobs)
    hp, lp = None, None
    if detrend:
        hp = 0.5
    if lowpass:
        lp = 12
    if detrend or lowpass:
        raw.filter(hp, lp, n_jobs=n_jobs)  # highpass filter

    # Detecting triggers
    events = mne.find_events(raw, stim_channel='STI101', shortest_event=1)
    events = events[events[:, 2] != 1]  # Remove event 1 (useless and not consistent across subjects)
    events = events[events[:, 2] != 64]  # Remove subjects response
    events = events[events[:, 2] != 65]

    # select channels
    raw.pick(channel_type)

    # TODO: the following is not used for now, make it optional or remove
    # find baseline mean of trials
    trial_epochs = mne.Epochs(raw, events, tmin=min(-0.5 + t_min, -0.5), tmax=8+t_max, baseline=None, preload=True)  # tmin and tmax are chosen to check for bad epochs before going to the notes
    trial_epochs.shift_time(-0.010)  # removes the 10ms offset of the triggers
    kept_epochs = trial_epochs.selection  # epochs good for their whole duration
    trial_epochs.crop(tmin=-0.5, tmax=-0.1)  # keep only baseline period
    baseline_means = trial_epochs.get_data()
    baseline_means = np.mean(baseline_means, axis=2)

    # load acoustic regressors
    X, acoustic_regressors = retrieve_acoustic_regressors(wav_folder, acoustic_folder)

    # get stim path
    wav_folder = (sorted(list(pathlib.Path(wav_folder).glob('**/*.wav'))))
    if pathlib.Path(raw_path).parts[-2] == 'maestro23_shvfbp':
        print(wav_folder[30])
        del wav_folder[30]
    if pathlib.Path(raw_path).parts[-2] == 'maestro00_hoyzez':
        print(wav_folder[10:12])
        del wav_folder[10:12]

    #retrieve note informations
    metadata = pd.DataFrame(columns=['stim_name', 'pitch', 'time_interval', 'trial_time',
                                     'surprise_perfrnn', 'entropy_perfrnn', 'surprise_magenta_pitch_perfrnn',
                                     'entropy_magenta_pitch_perfrnn', 'surprise_magenta_timing_perfrnn',
                                     'entropy_magenta_timing_perfrnn', 'surprise_mean_paulrnn',
                                     'surprise_posres_paulrnn', 'surprise_possum_paulrnn', 'surprise_negsum_paulrnn',
                                     'entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'loudness', 'brightness',
                                     'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                                     'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv'
                                     ]
                            )

    raw_fs = raw.info['sfreq']
    note_events = []
    for stim_nb in kept_epochs:  # trials annotated as bad are entirely rejected (even if only one note was bad)
        # Surprise and entropy from both models
        print(wav_folder[stim_nb])
        note_rate = wav_folder[stim_nb].name.split('_')[2]
        clip = wav_folder[stim_nb].name.split('_')[3]

        surprise_file = os.path.join(perfrnn_dir, 'surprise', note_rate, clip + '.mid.csv')
        entropy_file = os.path.join(perfrnn_dir, 'entropy', note_rate, clip + '.mid.csv')
        surprise_perfrnn = pd.read_csv(surprise_file)['note_on']
        entropy_perfrnn = pd.read_csv(entropy_file)['note_on']
        surprise_perfrnn = surprise_perfrnn.tolist()
        entropy_perfrnn = entropy_perfrnn.tolist()

        surprise_magenta_pitch_perfrnn = pd.read_csv(os.path.join(magenta_perfrnn_dir, 'note_on', note_rate, clip + '.csv'))['surprise'].tolist()
        entropy_magenta_pitch_perfrnn = pd.read_csv(os.path.join(magenta_perfrnn_dir, 'note_on', note_rate, clip + '.csv'))['entropy'].tolist()
        surprise_magenta_timing_perfrnn = pd.read_csv(os.path.join(magenta_perfrnn_dir, 'time_shift', note_rate, clip + '.csv'))['surprise'].tolist()
        entropy_magenta_timing_perfrnn = pd.read_csv(os.path.join(magenta_perfrnn_dir, 'time_shift', note_rate, clip + '.csv'))['entropy'].tolist()
        if len(surprise_magenta_pitch_perfrnn) < stim_dur*fs:
            surprise_magenta_pitch_perfrnn += [0 for _ in range(stim_dur*fs-len(surprise_magenta_pitch_perfrnn))]
        if len(entropy_magenta_pitch_perfrnn) < stim_dur*fs:
            entropy_magenta_pitch_perfrnn += [0 for _ in range(stim_dur*fs-len(entropy_magenta_pitch_perfrnn))]
        if len(surprise_magenta_timing_perfrnn) < stim_dur*fs:
            surprise_magenta_timing_perfrnn += [0 for _ in range(stim_dur*fs-len(surprise_magenta_timing_perfrnn))]
        if len(entropy_magenta_timing_perfrnn) < stim_dur*fs:
            entropy_magenta_timing_perfrnn += [0 for _ in range(stim_dur*fs-len(entropy_magenta_timing_perfrnn))]

        surprise_posres_paulrnn = pd.read_csv(os.path.join(paulrnn_dir, note_rate, clip + '.csv'))['positive_surprise_rescaled'].tolist()
        surprise_possum_paulrnn = pd.read_csv(os.path.join(paulrnn_dir, note_rate, clip + '.csv'))['positive_surprise_sum'].tolist()
        surprise_negsum_paulrnn = pd.read_csv(os.path.join(paulrnn_dir, note_rate, clip + '.csv'))['negative_surprise_sum '].tolist()
        surprise_mean_paulrnn = pd.read_csv(os.path.join(paulrnn_dir, note_rate, clip + '.csv'))['suprise_mean'].tolist()
        entropy_pitch_paulrnn = pd.read_csv(os.path.join(paulrnn_dir, note_rate, clip + '.csv'))['uncertainty'].tolist()
        entropy_timing_paulrnn = pd.read_csv(os.path.join(paulrnn_dir, note_rate, clip + '.csv'))['pitch_entropy_mean'].tolist()
        if len(surprise_posres_paulrnn) < stim_dur*fs:
            surprise_posres_paulrnn += [0 for _ in range(stim_dur*fs-len(surprise_posres_paulrnn))]
        if len(surprise_possum_paulrnn) < stim_dur*fs:
            surprise_possum_paulrnn += [0 for _ in range(stim_dur*fs-len(surprise_possum_paulrnn))]
        if len(surprise_negsum_paulrnn) < stim_dur*fs:
            surprise_negsum_paulrnn += [0 for _ in range(stim_dur*fs-len(surprise_negsum_paulrnn))]
        if len(surprise_mean_paulrnn) < stim_dur*fs:
            surprise_mean_paulrnn += [0 for _ in range(stim_dur*fs-len(surprise_mean_paulrnn))]
        if len(entropy_pitch_paulrnn) < stim_dur*fs:
            entropy_pitch_paulrnn += [entropy_pitch_paulrnn[-1] for i in range(stim_dur*fs-len(entropy_pitch_paulrnn))]
        if len(entropy_timing_paulrnn) < stim_dur*fs:
            entropy_timing_paulrnn += [entropy_timing_paulrnn[-1] for i in range(stim_dur*fs-len(entropy_timing_paulrnn))]

        # midi
        midi_file = os.path.join(midi_folder, note_rate, clip + '.mid')
        midi = MidiFile(midi_file, clip=True)
        time = events[stim_nb, 0]
        tick_duration = midi.tracks[0][0].tempo / midi.ticks_per_beat * 10**-6  # conversion from microseconds to seconds

        note_list = []
        time_sec = 0
        time_interval = 0
        for i in range(len(midi.tracks[1])):
            previous_time_sec = time_sec
            time_sec += midi.tracks[1][i].time * tick_duration
            previous_time = time
            time = events[stim_nb, 0] + round(time_sec * raw_fs)
            if time > previous_time:  # if time advances enough to get to the next sample
                if len(note_list) > 0:
                    if multiple_notes == 'max':
                        note_list = [max(note_list)]  # keep the highest note only
                    elif multiple_notes == 'min':
                        note_list = [min(note_list)]  # keep the lowest note only
                    elif multiple_notes != 'all':
                        raise ValueError("multiple_notes should be in {'all', 'min', 'max'}")
                    # note_list = convert_to_frequency(note_list)  # TODO: remove
                    note_list = [2**((note-69)/12) * 440 for note in note_list]
                    note_index = round(previous_time_sec / time_bin)
                    x_note = np.mean(X[note_index:note_index+150, stim_nb, :], axis=0) # take the mean of continuous acoutisc regressors in [0;150ms] after note onset
                    if metadata.shape[0] > 0:
                        time_interval = np.max([previous_time_sec - metadata['trial_time'].iloc[-1], 0])

                    # retrieve surprise value from paulrnn
                    if surprise_posres_paulrnn[note_index] != 0:  # if PE info, store it
                        s_pr = surprise_posres_paulrnn[note_index]
                        s_ps = surprise_possum_paulrnn[note_index]
                        s_ns = surprise_negsum_paulrnn[note_index]
                        s_m = surprise_mean_paulrnn[note_index]
                        ep_p = entropy_pitch_paulrnn[note_index-1]
                        et_p = entropy_timing_paulrnn[note_index-1]
                    else:  # else, look at neighboring idx, from past to future
                        search_size = 5  # in bins
                        dt_min = min([search_size, note_index])
                        dt_max = min([search_size, stim_dur*fs-note_index-1])
                        for dt in range(-dt_min, dt_max+1):
                            if surprise_posres_paulrnn[note_index+dt] != 0:
                                s_pr = surprise_posres_paulrnn[note_index+dt]
                                s_ps = surprise_possum_paulrnn[note_index+dt]
                                s_ns = surprise_negsum_paulrnn[note_index+dt]
                                s_m = surprise_mean_paulrnn[note_index+dt]
                                ep_p = entropy_pitch_paulrnn[note_index+dt-1]
                                et_p = entropy_timing_paulrnn[note_index+dt-1]
                                break
                    if s_pr == 0: print('warning : surprise = 0')

                    metadata = metadata.append({'stim_name': wav_folder[stim_nb].name, 'pitch': note_list, 'time_interval': time_interval, 'trial_time': previous_time_sec,
                                                'surprise_perfrnn': surprise_perfrnn[note_index], 'entropy_perfrnn': entropy_perfrnn[note_index],
                                                'surprise_magenta_pitch_perfrnn': surprise_magenta_pitch_perfrnn[note_index], 'entropy_magenta_pitch_perfrnn': entropy_magenta_pitch_perfrnn[note_index],
                                                'surprise_magenta_timing_perfrnn': surprise_magenta_timing_perfrnn[note_index], 'entropy_magenta_timing_perfrnn': entropy_magenta_timing_perfrnn[note_index],
                                                'surprise_mean_paulrnn': s_m, 'surprise_posres_paulrnn': s_pr, 'surprise_possum_paulrnn': s_ps, 'surprise_negsum_paulrnn': s_ns,
                                                'entropy_pitch_paulrnn': ep_p, 'entropy_timing_paulrnn': et_p,
                                                'loudness': x_note[0],
                                                'brightness': x_note[1],
                                                'periodicity': x_note[2],
                                                'roughness': x_note[3],
                                                'spectralflux': x_note[4],
                                                'loudness_deriv': x_note[5],
                                                'brightness_deriv': x_note[6],
                                                'periodicity_deriv': x_note[7],
                                                'roughness_deriv': x_note[8],
                                                'spectralflux_deriv': x_note[9],
                                                }
                                               , ignore_index=True)
                    note_list = []

            if (midi.tracks[1][i].type == 'note_on') and (midi.tracks[1][i].velocity != 0):
                if len(note_list) == 0:
                    note_events.append([time, 0, 1]) #<-- ? time or previous_time
                note_list.append(midi.tracks[1][i].note)

        if len(note_list) > 0:  # TODO: clean to avoid repetition of the same code
            if multiple_notes == 'max':
                note_list = [max(note_list)]  # keep the highest note only
            elif multiple_notes == 'min':
                note_list = [min(note_list)]  # keep the lowest note only
            elif multiple_notes != 'all':
                raise ValueError("multiple_notes should be in {'all', 'min', 'max'}")
            note_list = [2**((note-69)/12) * 440 for note in note_list]
            note_index = round(previous_time_sec / time_bin)
            x_note = np.mean(X[note_index:note_index+150, stim_nb, :], axis=0)
            if metadata.shape[0] > 0:
                time_interval = np.max([previous_time_sec - metadata['trial_time'].iloc[-1], 0])

            # retrieve surprise value from paulrnn
            if surprise_posres_paulrnn[note_index] != 0:  # if PE info, store it
                s_pr = surprise_posres_paulrnn[note_index]
                s_ps = surprise_possum_paulrnn[note_index]
                s_ns = surprise_negsum_paulrnn[note_index]
                s_m = surprise_mean_paulrnn[note_index]
                ep_p = entropy_pitch_paulrnn[note_index-1]
                et_p = entropy_timing_paulrnn[note_index-1]
            else:  # else, look at neighboring idx, from past to future
                search_size = 5  # in bins
                dt_min = min([search_size, note_index])
                dt_max = min([search_size, stim_dur*fs-note_index-1])
                for dt in range(-dt_min, dt_max+1):
                    if surprise_posres_paulrnn[note_index+dt] != 0:
                        s_pr = surprise_posres_paulrnn[note_index+dt]
                        s_ps = surprise_possum_paulrnn[note_index+dt]
                        s_ns = surprise_negsum_paulrnn[note_index+dt]
                        s_m = surprise_mean_paulrnn[note_index+dt]
                        ep_p = entropy_pitch_paulrnn[note_index+dt-1]
                        et_p = entropy_timing_paulrnn[note_index+dt-1]
                        break
            if s_pr == 0: print('warning : surprise = 0')

            metadata = metadata.append({'stim_name': wav_folder[stim_nb].name, 'pitch': note_list, 'time_interval': time_interval, 'trial_time': time_sec,
                                        'surprise_perfrnn': surprise_perfrnn[note_index], 'entropy_perfrnn': entropy_perfrnn[note_index],
                                        'surprise_magenta_pitch_perfrnn': surprise_magenta_pitch_perfrnn[note_index], 'entropy_magenta_pitch_perfrnn': entropy_magenta_pitch_perfrnn[note_index],
                                        'surprise_magenta_timing_perfrnn': surprise_magenta_timing_perfrnn[note_index], 'entropy_magenta_timing_perfrnn': entropy_magenta_timing_perfrnn[note_index],
                                        'surprise_mean_paulrnn': s_m, 'surprise_posres_paulrnn': s_pr, 'surprise_possum_paulrnn': s_ps, 'surprise_negsum_paulrnn': s_ns,
                                        'entropy_pitch_paulrnn': ep_p, 'entropy_timing_paulrnn': et_p,
                                        'loudness': x_note[0],
                                        'brightness': x_note[1],
                                        'periodicity': x_note[2],
                                        'roughness': x_note[3],
                                        'spectralflux': x_note[4],
                                        'loudness_deriv': x_note[5],
                                        'brightness_deriv': x_note[6],
                                        'periodicity_deriv': x_note[7],
                                        'roughness_deriv': x_note[8],
                                        'spectralflux_deriv': x_note[9],
                                        }
                                       , ignore_index=True)

    # transformations  # TODO: remove, the log transform is done at the linear regression stage
    # for i in range(len(metadata)):
    #     metadata['pitch'].iloc[i] = [np.log2(metadata['pitch'].iloc[i][0])]
    #     if metadata['pitch_interval'].iloc[i] is not None:
    #         metadata['pitch_interval'].iloc[i] = np.abs(np.log2(metadata['pitch_interval'].iloc[i]))

    note_events = np.array(note_events)

    '''Epoching'''

    event_dict = {'note_on': 1}
    epochs = mne.Epochs(raw, note_events, event_id=event_dict, tmin=t_min, tmax=t_max, baseline=None, preload=True, metadata=metadata, event_repeated='error')  # reject=reject_criteria,
    # epochs = mne.Epochs(raw, note_events, event_id=event_dict, tmin=t_min, tmax=t_max, baseline=None, preload=True, metadata=metadata, event_repeated='drop')
    # epochs.apply_baseline((None, -0.1))

    # Apply baseline by trial (all note of a given piece are baselined with the same value)
    # for i, stim_nb in enumerate(kept_epochs):  # TODO: check
    #     note_list = metadata[metadata['stim_name'] == wav_folder[stim_nb].name].index
    #     epochs._data[note_list, :, :] -= baseline_means[i, :, np.newaxis]
    return epochs


def epoching_maestro_note_level_time_frequency(tfr_path, wav_folder, midi_folder, acoustic_folder, perfrnn_dir, magenta_perfrnn_dir, paulrnn_dir, multiple_notes='max'):
    '''
    Takes time-frequency data epoched on excerpts and re-epoch it to the note-level.
    :param raw_path:
    :param wav_folder:
    :param midi_folder:
    :param perfrnn_dir: path to folder than contains 'surprise' and 'uncertainty' subfolders
    :param paulrnn_dir:
    :param t_min: minimum t for time-window
    :param t_max: maximum t for time-window
    :return: epoch object
    '''

    #parameters
    stim_dur = 8  # TODO: parameterize
    time_bin = 0.01
    fs = int(1/time_bin)

    #preprocessing
    tfr_epochs = mne.time_frequency.read_tfrs(tfr_path)[0]
    tfr_epochs.data = np.abs(tfr_epochs.data) ** 2  # keep power
    tfr_epochs.data = np.mean(tfr_epochs.data, axis=1, keepdims=True)  # average channels
    tfr_epochs.pick(0)

    tfr_epochs_times = np.round(tfr_epochs.times, decimals=2)  # precision issue

    # Get triggers
    events = tfr_epochs.events

    # load acoustic regressors
    X, acoustic_regressors = retrieve_acoustic_regressors(wav_folder, acoustic_folder)

    # get stim path
    wav_folder = (sorted(list(pathlib.Path(wav_folder).glob('**/*.wav'))))

    # No need to remove stim for subjects 0 and 23, as it is already accounted for (as opposed to epoching_maestro_note_level)

    #retrieve note informations
    metadata = pd.DataFrame(columns=['stim_name', 'pitch', 'time_interval', 'trial_time',
                                     'surprise_perfrnn', 'entropy_perfrnn', 'surprise_magenta_pitch_perfrnn',
                                     'entropy_magenta_pitch_perfrnn', 'surprise_magenta_timing_perfrnn',
                                     'entropy_magenta_timing_perfrnn', 'surprise_mean_paulrnn',
                                     'surprise_posres_paulrnn', 'surprise_possum_paulrnn', 'surprise_negsum_paulrnn',
                                     'entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'loudness', 'brightness',
                                     'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv',
                                     'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv'
                                     ]
                            )

    raw_fs = tfr_epochs.info['sfreq']
    note_events = []
    data = []
    for e in tqdm(range(len(tfr_epochs))):
        # Surprise and entropy from both models
        stim_nb = int(tfr_epochs.metadata.iloc[e, 0][5:8])  # finds the stim_nb in metadata
        note_rate = wav_folder[stim_nb].name.split('_')[2]
        clip = wav_folder[stim_nb].name.split('_')[3]

        surprise_file = os.path.join(perfrnn_dir, 'surprise', note_rate, clip + '.mid.csv')
        entropy_file = os.path.join(perfrnn_dir, 'entropy', note_rate, clip + '.mid.csv')
        surprise_perfrnn = pd.read_csv(surprise_file)['note_on']
        entropy_perfrnn = pd.read_csv(entropy_file)['note_on']
        surprise_perfrnn = surprise_perfrnn.tolist()
        entropy_perfrnn = entropy_perfrnn.tolist()

        surprise_magenta_pitch_perfrnn = pd.read_csv(os.path.join(magenta_perfrnn_dir, 'note_on', note_rate, clip + '.csv'))['surprise'].tolist()
        entropy_magenta_pitch_perfrnn = pd.read_csv(os.path.join(magenta_perfrnn_dir, 'note_on', note_rate, clip + '.csv'))['entropy'].tolist()
        surprise_magenta_timing_perfrnn = pd.read_csv(os.path.join(magenta_perfrnn_dir, 'time_shift', note_rate, clip + '.csv'))['surprise'].tolist()
        entropy_magenta_timing_perfrnn = pd.read_csv(os.path.join(magenta_perfrnn_dir, 'time_shift', note_rate, clip + '.csv'))['entropy'].tolist()
        if len(surprise_magenta_pitch_perfrnn) < stim_dur*fs:
            surprise_magenta_pitch_perfrnn += [0 for _ in range(stim_dur*fs-len(surprise_magenta_pitch_perfrnn))]
        if len(entropy_magenta_pitch_perfrnn) < stim_dur*fs:
            entropy_magenta_pitch_perfrnn += [0 for _ in range(stim_dur*fs-len(entropy_magenta_pitch_perfrnn))]
        if len(surprise_magenta_timing_perfrnn) < stim_dur*fs:
            surprise_magenta_timing_perfrnn += [0 for _ in range(stim_dur*fs-len(surprise_magenta_timing_perfrnn))]
        if len(entropy_magenta_timing_perfrnn) < stim_dur*fs:
            entropy_magenta_timing_perfrnn += [0 for _ in range(stim_dur*fs-len(entropy_magenta_timing_perfrnn))]

        surprise_posres_paulrnn = pd.read_csv(os.path.join(paulrnn_dir, note_rate, clip + '.csv'))['positive_surprise_rescaled'].tolist()
        surprise_possum_paulrnn = pd.read_csv(os.path.join(paulrnn_dir, note_rate, clip + '.csv'))['positive_surprise_sum'].tolist()
        surprise_negsum_paulrnn = pd.read_csv(os.path.join(paulrnn_dir, note_rate, clip + '.csv'))['negative_surprise_sum '].tolist()
        surprise_mean_paulrnn = pd.read_csv(os.path.join(paulrnn_dir, note_rate, clip + '.csv'))['suprise_mean'].tolist()
        entropy_pitch_paulrnn = pd.read_csv(os.path.join(paulrnn_dir, note_rate, clip + '.csv'))['uncertainty'].tolist()
        entropy_timing_paulrnn = pd.read_csv(os.path.join(paulrnn_dir, note_rate, clip + '.csv'))['pitch_entropy_mean'].tolist()
        if len(surprise_posres_paulrnn) < stim_dur*fs:
            surprise_posres_paulrnn += [0 for _ in range(stim_dur*fs-len(surprise_posres_paulrnn))]
        if len(surprise_possum_paulrnn) < stim_dur*fs:
            surprise_possum_paulrnn += [0 for _ in range(stim_dur*fs-len(surprise_possum_paulrnn))]
        if len(surprise_negsum_paulrnn) < stim_dur*fs:
            surprise_negsum_paulrnn += [0 for _ in range(stim_dur*fs-len(surprise_negsum_paulrnn))]
        if len(surprise_mean_paulrnn) < stim_dur*fs:
            surprise_mean_paulrnn += [0 for _ in range(stim_dur*fs-len(surprise_mean_paulrnn))]
        if len(entropy_pitch_paulrnn) < stim_dur*fs:
            entropy_pitch_paulrnn += [entropy_pitch_paulrnn[-1] for i in range(stim_dur*fs-len(entropy_pitch_paulrnn))]
        if len(entropy_timing_paulrnn) < stim_dur*fs:
            entropy_timing_paulrnn += [entropy_timing_paulrnn[-1] for i in range(stim_dur*fs-len(entropy_timing_paulrnn))]

        # midi
        midi_file = os.path.join(midi_folder, note_rate, clip + '.mid')
        midi = MidiFile(midi_file, clip=True)
        time = events[e, 0]
        tick_duration = midi.tracks[0][0].tempo / midi.ticks_per_beat * 10**-6  # conversion from microseconds to seconds

        note_list = []
        time_sec = 0
        time_interval = 0
        for i in range(len(midi.tracks[1])):
            previous_time_sec = time_sec
            time_sec += midi.tracks[1][i].time * tick_duration
            previous_time = time
            time = events[e, 0] + round(time_sec * raw_fs)
            if time > previous_time:  # if time advances enough to get to the next sample
                if len(note_list) > 0:
                    if multiple_notes == 'max':
                        note_list = [max(note_list)]  # keep the highest note only
                    elif multiple_notes == 'min':
                        note_list = [min(note_list)]  # keep the lowest note only
                    elif multiple_notes != 'all':
                        raise ValueError("multiple_notes should be in {'all', 'min', 'max'}")
                    # note_list = convert_to_frequency(note_list)  # TODO: remove
                    note_list = [2**((note-69)/12) * 440 for note in note_list]
                    note_index = round(previous_time_sec / time_bin)
                    x_note = np.mean(X[note_index:note_index+150, stim_nb, :], axis=0) # take the mean of continuous acoutisc regressors in [0;150ms] after note onset
                    if metadata.shape[0] > 0:
                        time_interval = np.max([previous_time_sec - metadata['trial_time'].iloc[-1], 0])

                    # retrieve surprise value from paulrnn
                    if surprise_posres_paulrnn[note_index] != 0:  # if PE info, store it
                        s_pr = surprise_posres_paulrnn[note_index]
                        s_ps = surprise_possum_paulrnn[note_index]
                        s_ns = surprise_negsum_paulrnn[note_index]
                        s_m = surprise_mean_paulrnn[note_index]
                        ep_p = entropy_pitch_paulrnn[note_index-1]
                        et_p = entropy_timing_paulrnn[note_index-1]
                    else:  # else, look at neighboring idx, from past to future
                        search_size = 5  # in bins
                        dt_min = min([search_size, note_index])
                        dt_max = min([search_size, stim_dur*fs-note_index-1])
                        for dt in range(-dt_min, dt_max+1):
                            if surprise_posres_paulrnn[note_index+dt] != 0:
                                s_pr = surprise_posres_paulrnn[note_index+dt]
                                s_ps = surprise_possum_paulrnn[note_index+dt]
                                s_ns = surprise_negsum_paulrnn[note_index+dt]
                                s_m = surprise_mean_paulrnn[note_index+dt]
                                ep_p = entropy_pitch_paulrnn[note_index+dt-1]
                                et_p = entropy_timing_paulrnn[note_index+dt-1]
                                break
                    if s_pr == 0: print('warning : surprise = 0')

                    metadata = metadata.append({'stim_name': wav_folder[stim_nb].name, 'pitch': note_list, 'time_interval': time_interval, 'trial_time': previous_time_sec,
                                                'surprise_perfrnn': surprise_perfrnn[note_index], 'entropy_perfrnn': entropy_perfrnn[note_index],
                                                'surprise_magenta_pitch_perfrnn': surprise_magenta_pitch_perfrnn[note_index], 'entropy_magenta_pitch_perfrnn': entropy_magenta_pitch_perfrnn[note_index],
                                                'surprise_magenta_timing_perfrnn': surprise_magenta_timing_perfrnn[note_index], 'entropy_magenta_timing_perfrnn': entropy_magenta_timing_perfrnn[note_index],
                                                'surprise_mean_paulrnn': s_m, 'surprise_posres_paulrnn': s_pr, 'surprise_possum_paulrnn': s_ps, 'surprise_negsum_paulrnn': s_ns,
                                                'entropy_pitch_paulrnn': ep_p, 'entropy_timing_paulrnn': et_p,
                                                'loudness': x_note[0],
                                                'brightness': x_note[1],
                                                'periodicity': x_note[2],
                                                'roughness': x_note[3],
                                                'spectralflux': x_note[4],
                                                'loudness_deriv': x_note[5],
                                                'brightness_deriv': x_note[6],
                                                'periodicity_deriv': x_note[7],
                                                'roughness_deriv': x_note[8],
                                                'spectralflux_deriv': x_note[9],
                                                }
                                               , ignore_index=True)
                    note_list = []

            if (midi.tracks[1][i].type == 'note_on') and (midi.tracks[1][i].velocity != 0):
                if len(note_list) == 0:
                    note_events.append([time, 0, 1]) #<-- ? time or previous_time
                    sample = np.where(tfr_epochs_times == round(time_sec - 1, 2))[0][0]  # 1 sec before and after. TODO: should be dependent on sfreq
                    data.append(tfr_epochs.data[e, :, :, sample:sample+201])  # add tfr data
                note_list.append(midi.tracks[1][i].note)

        if len(note_list) > 0:  # TODO: clean to avoid repetition of the same code
            if multiple_notes == 'max':
                note_list = [max(note_list)]  # keep the highest note only
            elif multiple_notes == 'min':
                note_list = [min(note_list)]  # keep the lowest note only
            elif multiple_notes != 'all':
                raise ValueError("multiple_notes should be in {'all', 'min', 'max'}")
            note_list = [2**((note-69)/12) * 440 for note in note_list]
            note_index = round(previous_time_sec / time_bin)
            x_note = np.mean(X[note_index:note_index+150, stim_nb, :], axis=0)
            if metadata.shape[0] > 0:
                time_interval = np.max([previous_time_sec - metadata['trial_time'].iloc[-1], 0])

            # retrieve surprise value from paulrnn
            if surprise_posres_paulrnn[note_index] != 0:  # if PE info, store it
                s_pr = surprise_posres_paulrnn[note_index]
                s_ps = surprise_possum_paulrnn[note_index]
                s_ns = surprise_negsum_paulrnn[note_index]
                s_m = surprise_mean_paulrnn[note_index]
                ep_p = entropy_pitch_paulrnn[note_index-1]
                et_p = entropy_timing_paulrnn[note_index-1]
            else:  # else, look at neighboring idx, from past to future
                search_size = 5  # in bins
                dt_min = min([search_size, note_index])
                dt_max = min([search_size, stim_dur*fs-note_index-1])
                for dt in range(-dt_min, dt_max+1):
                    if surprise_posres_paulrnn[note_index+dt] != 0:
                        s_pr = surprise_posres_paulrnn[note_index+dt]
                        s_ps = surprise_possum_paulrnn[note_index+dt]
                        s_ns = surprise_negsum_paulrnn[note_index+dt]
                        s_m = surprise_mean_paulrnn[note_index+dt]
                        ep_p = entropy_pitch_paulrnn[note_index+dt-1]
                        et_p = entropy_timing_paulrnn[note_index+dt-1]
                        break
            if s_pr == 0: print('warning : surprise = 0')

            metadata = metadata.append({'stim_name': wav_folder[stim_nb].name, 'pitch': note_list, 'time_interval': time_interval, 'trial_time': time_sec,
                                        'surprise_perfrnn': surprise_perfrnn[note_index], 'entropy_perfrnn': entropy_perfrnn[note_index],
                                        'surprise_magenta_pitch_perfrnn': surprise_magenta_pitch_perfrnn[note_index], 'entropy_magenta_pitch_perfrnn': entropy_magenta_pitch_perfrnn[note_index],
                                        'surprise_magenta_timing_perfrnn': surprise_magenta_timing_perfrnn[note_index], 'entropy_magenta_timing_perfrnn': entropy_magenta_timing_perfrnn[note_index],
                                        'surprise_mean_paulrnn': s_m, 'surprise_posres_paulrnn': s_pr, 'surprise_possum_paulrnn': s_ps, 'surprise_negsum_paulrnn': s_ns,
                                        'entropy_pitch_paulrnn': ep_p, 'entropy_timing_paulrnn': et_p,
                                        'loudness': x_note[0],
                                        'brightness': x_note[1],
                                        'periodicity': x_note[2],
                                        'roughness': x_note[3],
                                        'spectralflux': x_note[4],
                                        'loudness_deriv': x_note[5],
                                        'brightness_deriv': x_note[6],
                                        'periodicity_deriv': x_note[7],
                                        'roughness_deriv': x_note[8],
                                        'spectralflux_deriv': x_note[9],
                                        }
                                       , ignore_index=True)

    note_events = np.array(note_events)
    data = np.array(data)
    times = np.linspace(-1, 1, 201)  # TODO: should be dependent on sfreq

    '''Epoching'''
    event_dict = {'note_on': 1}
    epochs = mne.time_frequency.EpochsTFR(tfr_epochs.info, data, times, tfr_epochs.freqs, events=note_events, event_id=event_dict, metadata=metadata)
    return epochs


def merge_grad_epochs_data(data, method='rms'):
    '''
    Combine planar gradiometers pairs to obtain one gradient value per location.
    data: numpy array of shape (n_epochs, n_channels, n_times)
    '''

    data = data.reshape((data.shape[0], data.shape[1] // 2, 2, -1))
    if method == 'rms':
        data = np.sqrt(np.mean(data**2, axis=2))
    else:
        raise ValueError("method should be in {'rms'}")

    return data


def merge_grad_data(data, method='mean'):
    '''
    Useful when plotting grad data that are not grad signals (for instance correlation scores) so that it is not only
    positive (mean instead of RMS used in mne.viz.plot_topomap)
    '''

    data, orig_shape = data.reshape((len(data) // 2, 2, -1)), data.shape
    if method == 'mean':
        data = np.mean(data, axis=1)
    elif method == 'sum':
        data = np.sum(data, axis=1)
    elif method == 'max':
        data = np.max(data, axis=1)

    return data.reshape(data.shape[:1] + orig_shape[1:])


def best_flip_combination(coefs, negative_sample=20):
    '''
    Flip the kernels sign to obtain the highest average pairwise correlation (used when plotting trf kernels to make sure they are not inverted)
    coefs: numpy of shape (nb_channels, time)
    negative_sample: forces the first kernel to have negative value at the this sample (arbitrary). Default is 20 which is time 0s in our case.
    :return: signs
    '''

    if coefs[0, negative_sample] < 0:
        first_sign = 1
    else:
        first_sign = -1

    signs_combinations = [np.array(tuple([first_sign])+combi) for combi in itertools.product([1,-1], repeat=coefs.shape[0]-1)]
    sums_corr = []
    for signs in signs_combinations:
        corr_matrix = np.corrcoef(coefs * signs[:, np.newaxis])
        sums_corr.append(np.sum(np.triu(corr_matrix, k=1)))  # averaging is not necessary since we then take the max value
    best_combi = sums_corr.index(max(sums_corr))

    return signs_combinations[best_combi]


def best_flip_combination_subjects(coefs):
    '''
    Flip the kernels sign to obtain the highest correlation between the first subjects and each other ones. Not as
    accurate as best_flip_combination but less expensive computationally.
    coefs: numpy of shape (nb_channels, time)
    :return: signs
    '''

    signs = [1]
    for i in range(1, coefs.shape[0]):
        corr_pos = np.corrcoef(coefs[[0, i], :])[0,1]
        corr_neg = np.corrcoef(np.stack((coefs[0, :], -1 * coefs[i, :]), axis=0))[0,1]
        if corr_pos >= corr_neg:
            signs.append(1)
        else:
            signs.append(-1)

    return np.array(signs)


def fixed_golden_aspect_ratio(ax, golden_factor=1):
    '''
    Set a fixed aspect ratio on matplotlib plots
    regardless of axis units
    '''

    xvals, yvals = ax.get_xlim(), ax.get_ylim()

    xrange = xvals[1] - xvals[0]
    yrange = yvals[1] - yvals[0]
    golden_ratio = golden_factor * (np.sqrt(5) + 1) / 2
    ax.set_aspect(1/golden_ratio * (xrange / yrange), adjustable='box')

    return

def plot_seeg(scores, scoring, mask, vlim, montages, title, fs_path, save_path, epochs_path, colormap):
  """
  scores: list of arrays (n_channels,)
  scoring: name of scoring to use on plot
  mask: binary matrix for channel selection
  vlim: limits of colorbar, [vmin, vmax]
  montages: list of montage
  title: title of fig 
  fs_path: path to fsaverage files
  save_path: path and name of .png output
  epochs_path: path to epochs folder. TO BE REMOVED
  colormap: 'custom' for darkblue to yellow, or 'diverging' for blue to red
  """

  #azimuth, elevation, roll, distance
  aerd = [[180,90,0,0.4],
          [0,90,0,0.4],
          [180,0,0,0.4],
          [90,90,0,0.4]]
  xy_list = [[] for i in range(len(aerd))]
  scores_list = [[] for i in range(len(aerd))]
  mask_list = [[] for i in range(len(aerd))]

  epochs_path = pathlib.Path(epochs_path)
  epochs_list = sorted(list(epochs_path.glob('**/*_epochs_mono_preprocessed_IDA.fif')))
  for i_sub in range(len(montages)):
    sub_epochs_path = epochs_list[i_sub]

    #load original data
    epochs = mne.read_epochs(sub_epochs_path,preload=False)

    # get montage for whole brain and each hemisphere
    montage = montages[i_sub]
    ch_L, ch_R = [], []
    dig_L, dig_R = [], []
    for i_ch, ch in enumerate(montage.ch_names):
      if "'" in ch:
        ch_L.append(ch)
        dig_L.append(montage.dig[i_ch])
      else:
        ch_R.append(ch)
        dig_R.append(montage.dig[i_ch])
    montage_L = mne.channels.DigMontage(dig=dig_L, ch_names=ch_L)
    montage_R = mne.channels.DigMontage(dig=dig_R, ch_names=ch_R)

    #get
    head_fs = mne.transforms.Transform('head', 'mri',trans=None)
    fig = mne.viz.plot_alignment(epochs.info, trans=head_fs, subject= 'fsaverage', coord_frame = 'mri',
                          subjects_dir=fs_path, surfaces=dict(pial=0.95))
    fig.plotter.set_background(color=[1.0, 1.0, 1.0])

    fig2, axes = plt.subplots(1,len(aerd),figsize=(16, 8))
    for i_pos in range(len(aerd)):
      mne.viz.set_3d_view(figure=fig, azimuth=aerd[i_pos][0], elevation=aerd[i_pos][1],roll=aerd[i_pos][2],distance=aerd[i_pos][3])
      if i_pos == 0:
        M = montage_L
      elif i_pos == 1:
        M = montage_R
      else:
        M = montage
      xy, im = mne.viz.snapshot_brain_montage(fig, M,hide_sensors=True)

      #get values according to montage
      s_tmp = []
      m_tmp = []
      for i_chan in range(len(montage.ch_names)):
        if montage.ch_names[i_chan] in M.ch_names:
          s_tmp.append(scores[i_sub][i_chan])
          m_tmp.append(mask[i_sub][i_chan])

      # Store position for 2D projection, and score, for each point
      xy_pts = np.vstack([xy[ch] for ch in M.ch_names])
      xy_list[i_pos].append(xy_pts.T)
      scores_list[i_pos].append(np.array(s_tmp))
      mask_list[i_pos].append(np.array(m_tmp))
    fig.plotter.close()

  #parameters for final plot
  if colormap == 'custom':
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_gradient', (
        # Edit this gradient at https://eltos.github.io/gradient/#0:000659-0.6:4B1980-5.2:862734-10.2:A71D2E-16.4:B23F12-31.9:BE811D-54:E0B83A-100:FFFFFF
        (0.000, (0.000, 0.024, 0.349)),
        (0.006, (0.294, 0.098, 0.502)),
        (0.052, (0.525, 0.153, 0.204)),
        (0.102, (0.655, 0.114, 0.180)),
        (0.164, (0.698, 0.247, 0.071)),
        (0.319, (0.745, 0.506, 0.114)),
        (0.540, (0.878, 0.722, 0.227)),
        (1.000, (1.000, 1.000, 1.000))))
  elif colormap == 'diverging':
    cmap = 'bwr' #coolwarm, bwr, seismic  

  #get brain image in background
  brain = mne.viz.Brain(subjects_dir=fs_path, subject_id = 'fsaverage',
                        surf='pial', #'pial', 'inflated', 'white',...
                        background='white',
                        alpha = 0.95,
                        cortex='classic',
                        offscreen=True)

  #for each orientation
  fig, axes = plt.subplots(1,len(aerd),figsize=(16, 8), dpi = 300)
  for i_pos in range(len(aerd)):
    #adjust brain background
    brain.show_view(azimuth=aerd[i_pos][0], elevation=aerd[i_pos][1],roll=aerd[i_pos][2],distance=aerd[i_pos][3]*1000)
    im = brain.screenshot()

    # draw 2D scatter plots
    axes[i_pos].imshow(im)
    for i_sub in range(len(montages)):
      xy = xy_list[i_pos][i_sub]
      values = scores_list[i_pos][i_sub]
      mask = mask_list[i_pos][i_sub]

    #   xy_ = xy[:,np.where(values>s_thresh)[0]]
    #   values = values[values>s_thresh]
      xy_ = xy[:,mask]
      values = values[mask]

      scatt_1 = axes[i_pos].scatter(*xy, c='black', s=1, alpha=0.3,marker='.') #s=1.2
      scatt_2 = axes[i_pos].scatter(*xy_, c=values, s=2.5, alpha=1,marker='o', #s=1.2
                                    cmap=cmap, vmin = vlim[0], vmax = vlim[1])
    # cmap_contour = axes[i_pos].scatter(*xy_pts[np.where(scores>thresh)[0],:].T, edgecolors='black',facecolors='none', s=50, alpha=1,marker='o')
    axes[i_pos].set_axis_off()

  #final plot parameters
  cbar = fig.colorbar(scatt_2, fraction=0.025, pad=-0.08) #fraction=0.046, pad=0.04
  cbar.ax.set_ylabel(scoring)
  fig.suptitle(title,y=0.75)
  fig.savefig(save_path)
  plt.show()
  brain.plotter.close()

  return fig

def plot_seeg_gridspec(scores, scoring, mask, vlim, montages, title, fs_path, save_path, epochs_path, colormap):
  """
  scores: list of arrays (n_channels,)
  scoring: name of scoring to use on plot
  mask: binary matrix for channel selection
  vlim: limits of colorbar, [vmin, vmax]
  montages: list of montage
  title: title of fig 
  fs_path: path to fsaverage files
  save_path: path and name of .png output
  epochs_path: path to epochs folder. TO BE REMOVED
  colormap: 'custom' for darkblue to yellow, or 'diverging' for blue to red
  """

  # plt.switch_backend('TkAgg')

  #azimuth, elevation, roll, distance
  aerd = [[180,0,0,0.4],
          [180,90,0,0.4],
          [0,90,0,0.4]]
  xy_list = [[] for i in range(len(aerd))]
  scores_list = [[] for i in range(len(aerd))]
  mask_list = [[] for i in range(len(aerd))]

  # some parameters
  # fig_width_ratio = [(np.sqrt(5) + 1) / 2, 1]
  fig_width_ratio = [1, 1]
  point_size = 2.4
  dpi = 600

  epochs_path = pathlib.Path(epochs_path)
  epochs_list = sorted(list(epochs_path.glob('**/*_epochs_mono_preprocessed_IDA.fif')))
  for i_sub in range(len(montages)):
    sub_epochs_path = epochs_list[i_sub]

    #load original data
    epochs = mne.read_epochs(sub_epochs_path,preload=False)

    # get montage for whole brain and each hemisphere
    montage = montages[i_sub]
    ch_L, ch_R = [], []
    dig_L, dig_R = [], []
    for i_ch, ch in enumerate(montage.ch_names):
      if "'" in ch:
        ch_L.append(ch)
        dig_L.append(montage.dig[i_ch])
      else:
        ch_R.append(ch)
        dig_R.append(montage.dig[i_ch])
    montage_L = mne.channels.DigMontage(dig=dig_L, ch_names=ch_L)
    montage_R = mne.channels.DigMontage(dig=dig_R, ch_names=ch_R)

    #get
    head_fs = mne.transforms.Transform('head', 'mri',trans=None)
    fig = mne.viz.plot_alignment(epochs.info, trans=head_fs, subject= 'fsaverage', coord_frame = 'mri',
                          subjects_dir=fs_path, surfaces=dict(pial=0.95))
    fig.plotter.set_background(color=[1.0, 1.0, 1.0])

    fig2, axes = plt.subplots(1,len(aerd),figsize=(16, 8))
    for i_pos in range(len(aerd)):
      mne.viz.set_3d_view(figure=fig, azimuth=aerd[i_pos][0], elevation=aerd[i_pos][1],roll=aerd[i_pos][2],distance=aerd[i_pos][3])
      if i_pos == 1:
        M = montage_L
      elif i_pos == 2:
        M = montage_R
      else:
        M = montage
      xy, im = mne.viz.snapshot_brain_montage(fig, M,hide_sensors=True)

      #get values according to montage
      s_tmp = []
      m_tmp = []
      for i_chan in range(len(montage.ch_names)):
        if montage.ch_names[i_chan] in M.ch_names:
          s_tmp.append(scores[i_sub][i_chan])
          m_tmp.append(mask[i_sub][i_chan])

      # Store position for 2D projection, and score, for each point
      xy_pts = np.vstack([xy[ch] for ch in M.ch_names])
      xy_list[i_pos].append(xy_pts.T)
      scores_list[i_pos].append(np.array(s_tmp))
      mask_list[i_pos].append(np.array(m_tmp))
    fig.plotter.close()

  #parameters for final plot
  if colormap == 'custom':
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_gradient', (
        # Edit this gradient at https://eltos.github.io/gradient/#0:000659-0.6:4B1980-5.2:862734-10.2:A71D2E-16.4:B23F12-31.9:BE811D-54:E0B83A-100:FFFFFF
        (0.000, (0.000, 0.024, 0.349)),
        (0.006, (0.294, 0.098, 0.502)),
        (0.052, (0.525, 0.153, 0.204)),
        (0.102, (0.655, 0.114, 0.180)),
        (0.164, (0.698, 0.247, 0.071)),
        (0.319, (0.745, 0.506, 0.114)),
        (0.540, (0.878, 0.722, 0.227)),
        (1.000, (1.000, 1.000, 1.000))))
  elif colormap == 'diverging':
    cmap = 'bwr' #coolwarm, bwr, seismic  

  #get brain image in background
  brain = mne.viz.Brain(subjects_dir=fs_path, subject_id = 'fsaverage',
                        surf='pial', #'pial', 'inflated', 'white',...
                        background='white',
                        alpha = 0.95,
                        cortex='classic',
                        offscreen=True)

  #for each orientation
  fig, axes = plt.subplots(2, 2, gridspec_kw={'width_ratios': fig_width_ratio}, figsize=(16, 8), dpi = dpi)
  gs = axes[0, 0].get_gridspec()
  # remove the underlying axes
  for ax in axes[:, 0]:
      ax.remove()
  axbig = fig.add_subplot(gs[:, 0])
  for i_pos in range(len(aerd)):
    if i_pos == 0:
      ax = axbig
    else:
      print(i_pos, axes.shape)
      ax = axes[i_pos-1,1]

    #adjust brain background
    brain.show_view(azimuth=aerd[i_pos][0], elevation=aerd[i_pos][1],roll=aerd[i_pos][2],distance=aerd[i_pos][3]*1000)
    im = brain.screenshot()

    # draw 2D scatter plots
    ax.imshow(im)
    for i_sub in range(len(montages)):
      xy = xy_list[i_pos][i_sub]
      values = scores_list[i_pos][i_sub]
      mask = mask_list[i_pos][i_sub]
      xy_ = xy[:,mask]
      values = values[mask]

      if i_pos == 0:
        s = (point_size * fig_width_ratio[0]*1.5)**2
      else:
        s = point_size
      scatt_1 = ax.scatter(*xy, c='black', s=s, alpha=0.3,marker='.')
      scatt_2 = ax.scatter(*xy_, c=values, s=s, alpha=1,marker='o',
                                    cmap=cmap, vmin = vlim[0], vmax = vlim[1])
    ax.set_axis_off()

  #final plot parameters
  fig.subplots_adjust(bottom=0.7)
  cbar_ax = fig.add_axes([0.3, 0.065, 0.4, 0.025])
  cbar = fig.colorbar(scatt_2, cax=cbar_ax, orientation = 'horizontal')
  cbar.set_label(label=scoring,size=16)
  fig.suptitle(title,y=0.95, fontsize = 10)
  fig.tight_layout()
  fig.savefig(save_path)
  plt.show()
  brain.plotter.close()

  return fig


def gmsi_array_extraction(gmsi_path='/home/mathieupvc/These/data/meg_maestro/gmsi_processed.csv'):

    df = pd.read_csv(gmsi_path)
    df = df.iloc[:27, :]

    return df['FG (General Sophistication)'].to_numpy()


def trial_rms(data_path, wav_folder, save_path, channel_type):

    path = pathlib.Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    subject_names_list = []
    rms_1st_sec_list = []
    rms_1_8_sec_list = []
    stim_name_list = []
    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]

        epochs = epoching_maestro(raw_path, wav_folder, tmin=-1, baseline=False)  # Load raw and create epochs
        epochs = epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]  # Select listening conditions only
        epochs.pick(channel_type)  # keep magnetometers and/or gradiometers
        epochs = epochs[[int(stim_name[5:8]) >= 10 for stim_name in epochs.metadata['stim_name']]]
        epochs.filter(0.5, None, n_jobs=-1)  # Detrend
        epochs.resample(200, n_jobs=-1)
        for e in range(len(epochs)):
            rms_1st_sec_list.append(np.mean(np.sqrt(np.mean(epochs.copy().crop(0, 1).get_data()**2, axis=1)), axis=1)[e])  # average RMS over time
            rms_1_8_sec_list.append(np.mean(np.sqrt(np.mean(epochs.copy().crop(1, 8).get_data()**2, axis=1)), axis=1)[e])  # average RMS over time
            subject_names_list.append(subject_path.name)
            stim_name_list.append(epochs[e].metadata['stim_name'])

    df = pd.DataFrame({'subject': subject_names_list, 'stim_name': stim_name_list, 'rms_1s': rms_1st_sec_list, 'rms_7s': rms_1_8_sec_list})

    df.to_csv(os.path.join(save_path, 'average_rms.csv'))

    return


def subjects_info_meg(epochs_path):
    path = pathlib.Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))

    age = []
    sex = []
    hand = []
    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)
        info = epochs.info

        age.append(2021 - info['subject_info']['birthday'][0])
        sex.append(info['subject_info']['sex'])
        hand.append(info['subject_info']['hand'])

    print('sex:', sex)
    print('hand:', hand)
    print('age:', age)
    print('mean age', np.mean(age))
    print('std age', np.std(age))

    return


def select_seeg_channels_from_parcellation(epochs, regions, first_parcel_label='advanced'):
    """channels: list of labels to keep"""

    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"
    print(subjects_dir)
    # mne.datasets.fetch_fsaverage(subjects_dir=None)
    subject = 'fsaverage'

    montage = epochs.get_montage()
    head_fs = mne.transforms.Transform('head', 'mri', trans=None)
    montage.apply_trans(head_fs)
    labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                   subjects_dir=subjects_dir,
                                                   aseg='aparc+aseg', dist=2)
    channel_parcellations = []
    for key, value in labels.items():
        channel_parcellations.append(value)

    if first_parcel_label == True:
        channel_parcellations = [[parcel[0]] for parcel in channel_parcellations]
    elif first_parcel_label == "advanced":
        go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
        channel_parcellations = select_parcel(channel_parcellations, go_next)

    channels_to_keep = []
    for c, channels in enumerate(channel_parcellations):
        if any(area in regions for area in channels):
            channels_to_keep.append(c)

    if len(channels_to_keep) == 0:
        print('No matching channel, returning None')
        return None, None
    else:
        print('Kept channels:', channels_to_keep)
        return epochs.pick(channels_to_keep), channel_parcellations


def list_of_regions_seeg(data_path, fs_subjects_dir):
    path = pathlib.Path(data_path)
    subject_list = sorted(list(path.glob('*.fif')))

    regions = []
    n_channels = 0
    for i, epochs_path in tqdm(enumerate(subject_list)):
        subject_name = epochs_path.name[:6]
        print(subject_name)

        epochs = mne.read_epochs(epochs_path)

        n_channels += len(epochs.ch_names)

        montage = epochs.get_montage()
        head_fs = mne.transforms.Transform('head', 'mri', trans=None)
        montage.apply_trans(head_fs)
        labels, _ = mne.get_montage_volume_labels(montage, 'fsaverage',
                                                  subjects_dir=fs_subjects_dir,
                                                  aseg='aparc+aseg', dist=2)
        for key, value in labels.items():
            regions += value
    regions = np.unique(regions)
    filtered_regions = []
    for region in regions:
        if ('Unknown' not in region) & ('unknown' not in region):
            filtered_regions.append(region)

    print(filtered_regions)
    print('Number of regions:', len(filtered_regions))
    print('Number of channels:', n_channels)

    return

def select_parcel(parcellations, go_next):
    """for each list of parcellations, select the first parcellation that is not in go_next."""
    new_parcellations = []
    for parcels in parcellations:
        added = False
        for parcel in parcels:
            if parcel not in go_next:
                new_parcellations.append([parcel])
                added = True
                break
        if not added:
            new_parcellations.append([parcels[0]])  # if all parcels were in go_next, add the first one
    assert len(parcellations) == len(new_parcellations)
    return new_parcellations


if __name__ == '__main__':

    # subjects_info_meg('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta')

    """list_of_regions_seeg"""
    data_path = '/home/mathieupvc/These/data/meg_maestro/seeg_trials'
    fs_subjects_dir = '/home/mathieupvc/freesurfer/subjects'
    list_of_regions_seeg(data_path, fs_subjects_dir=fs_subjects_dir)
