from pathlib import Path
import os
import copy

import numpy as np
import pandas as pd

import scipy.stats

import mne

import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.anova import AnovaRM

from statannotations.Annotator import Annotator

from maestro_analysis.utils import gmsi_array_extraction, fixed_golden_aspect_ratio
from maestro_analysis.maestro_behavior import seeg_behavior_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def create_data_matrix(epochs_path, behavior_path, ecg_path, gmsi_path, power_bands_path, rms_path, save_path, data_type='meg',
                       signal_type='raw', montage='mono', remove=None, tmin=2, tmax=7,
                       min_time_interval=0.050, best_channels_path=None):

    # create save dir
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # get the correct epochs path depending on the desired dataset
    path = Path(epochs_path)
    if data_type == 'meg':
        epochs_list = sorted(list(path.glob('**/*epo.fif')))
        del epochs_list[22]  # remove bad subjects
        del epochs_list[20]
        del epochs_list[14]
        del epochs_list[5]
        gmsi_array = gmsi_array_extraction(gmsi_path=gmsi_path)
        exclude = [5, 14, 20, 22]
        gmsi_array = gmsi_array[[i for i in range(27) if i not in exclude]]
    if data_type == 'seeg':
        raise NotImplementedError
        if signal_type == 'gamma':
            st = '_gamma'
        elif signal_type == 'raw':
            st = ''
        epochs_list = sorted(list(path.glob('**/*_epochs' + st + '_' + montage + '_preprocessed_IDA.fif')))

    if best_channels_path is not None:
        best_channels = np.load(best_channels_path)

    df_list = []
    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)

        subject_path = Path(os.path.join(behavior_path, subject_epochs.name[:-8]))
        path_to_csv = subject_path / ('P_' + subject_path.name.split('_')[-1] + '/P_' + subject_path.name.split('_')[-1] + '.csv')
        behavior = pd.read_csv(path_to_csv)

        ecg_epochs = mne.read_epochs(os.path.join(ecg_path, subject_epochs.name[:-8] + "_ecg-epo.fif"))
        ecg_epochs = ecg_epochs[['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']]
        if 'maestro08_mxnpsq' in subject_epochs.name:
            ecg_epochs = ecg_epochs[ecg_epochs.metadata['stim_name'] != 'trial150_rep2_4.5Hz_clip5_piano2.wav']  # this trial is removed in note level epochs (it may have been marked as bad)
        ecg_epochs = ecg_epochs.crop(0, 8)
        heart_pulses = ecg_epochs.get_data(picks='bin_ecg')
        # compute bpm
        bpm = np.zeros(len(heart_pulses))
        hrv = np.zeros(len(heart_pulses))
        for e in range(len(heart_pulses)):
            intervals = []
            start = None
            for t in range(heart_pulses.shape[2]):
                if heart_pulses[e, 0, t] == 1:
                    if start is not None:
                        intervals.append(t - start)
                    start = t
            # if len(intervals) == 0:
            #     print(e)
            #     print(np.sum(heart_pulses[e, 0, :]))
            bpm[e] = 60 * ecg_epochs.info['sfreq'] / np.mean(intervals)
            intervals = np.array(intervals)/ecg_epochs.info['sfreq']
            hrv[e] = np.sqrt(np.mean(intervals**2))
        # heart_pulses = np.sum(heart_pulses[:, 0, :], axis=1)
        heart_pulses = bpm

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
        metadata['onset_type'] = 0  # add onset type from stim name
        metadata['pitch_interval'] = 0  # add pitch interval
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'onset_type'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[4][-5])
            if metadata.loc[metadata.index[i_note], 'trial_time'] > 0:
                metadata.loc[metadata.index[i_note], 'pitch_interval'] = np.abs(metadata.loc[metadata.index[i_note - 1], 'pitch'] - metadata.loc[metadata.index[i_note], 'pitch'])

        epochs.metadata = metadata
        regressors = metadata

        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressors = regressors[regressors['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]
        regressors = regressors[regressors['trial_time'].to_numpy() <= tmax]

        print('Number of notes after min/max trial time:', len(epochs.metadata))

        del epochs

        max_surprise = regressors[['stim_name', 'surprise_mean_paulrnn']].groupby(['stim_name'], as_index=False).max()
        std_surprise = regressors[['stim_name', 'surprise_mean_paulrnn']].groupby(['stim_name'], as_index=False).std()
        regressors = regressors.groupby(['stim_name'], as_index=False).mean()
        regressors['max_surprise'] = max_surprise['surprise_mean_paulrnn']
        regressors['std_surprise'] = std_surprise['surprise_mean_paulrnn']

        # add behavior
        regressors['subject'] = ''
        regressors['stim_identity'] = ''
        regressors['block'] = 0.0
        regressors['position_in_block'] = 0.0
        regressors['rep1'] = 0.0  # one-hot encoding
        regressors['rep2'] = 0.0
        regressors['rep3'] = 0.0
        regressors['2.5Hz'] = 0.0
        regressors['3.5Hz'] = 0.0
        regressors['4.5Hz'] = 0.0
        regressors['5.5Hz'] = 0.0
        regressors['6.5Hz'] = 0.0
        regressors['rs'] = 0.0  # reaction speed
        regressors['resp'] = 0.0
        regressors['correct'] = 0.0
        regressors['gmsi'] = gmsi_array[i]
        regressors['heart_pulses'] = heart_pulses
        regressors['hrv'] = hrv

        assert len(heart_pulses) == len(regressors)
        assert len(hrv) == len(regressors)

        for trial in range(regressors.shape[0]):
            regressors.at[regressors.index[trial], 'subject'] = subject_epochs.name[:-8]
            stim_nb = int(regressors.at[regressors.index[trial], 'stim_name'][5:8])  # finds the stim_nb
            regressors.at[regressors.index[trial], 'stim_identity'] = regressors.at[regressors.index[trial], 'stim_name'].split('_')[2] + '_' + regressors.at[regressors.index[trial], 'stim_name'].split('_')[3]
            regressors.at[regressors.index[trial], 'block'] = behavior.at[behavior.index[stim_nb], 'block']
            regressors.at[regressors.index[trial], 'position_in_block'] = behavior.at[behavior.index[stim_nb], 'trial']
            for rep in [1, 2, 3]:
                if behavior.at[behavior.index[stim_nb], 'rep'] == rep:
                    regressors.at[regressors.index[trial], 'rep' + str(rep)] = 1.0
            for rate in [2.5, 3.5, 4.5, 5.5, 6.5]:
                if behavior.at[behavior.index[stim_nb], 'freq'] == rate:
                    regressors.at[regressors.index[trial], str(rate) + 'Hz'] = 1.0
            regressors.at[regressors.index[trial], 'rs'] = 1 / behavior.at[behavior.index[stim_nb], 'rt']
            regressors.at[regressors.index[trial], 'resp'] = behavior.at[behavior.index[stim_nb], 'resp']
            regressors.at[regressors.index[trial], 'correct'] = float(((behavior.at[behavior.index[stim_nb], 'resp'] == 1) & (behavior.at[behavior.index[stim_nb], 'rep'] == 1)) | ((behavior.at[behavior.index[stim_nb], 'resp'] == 2) & (behavior.at[behavior.index[stim_nb], 'rep'] > 1)))

        df_list.append(regressors[regressors['block'] > 1])  # remove first block

    data_matrix = pd.concat(df_list, axis=0)

    # add rms
    rms = pd.read_csv(rms_path)
    rms = rms[((rms['subject'] != 'maestro05_jzlenu') & (rms['subject'] != 'maestro14_pkemhw') & (rms['subject'] != 'maestro20_dddftu') & (rms['subject'] != 'maestro22_lyzsqv'))]
    rms = rms[~((rms['stim_name'] == '300    trial150_rep2_4.5Hz_clip5_piano2.wav\nName: stim_name, dtype: object') & (rms['subject'] == 'maestro08_mxnpsq'))]  # this trial is removed in note level epochs (it may have been marked as bad)
    rms['rms_1s'] = rms['rms_1s'] / rms['rms_1s'].std()
    rms['rms_7s'] = rms['rms_7s'] / rms['rms_7s'].std()
    rms = rms.iloc[:, 3:]

    # add power bands
    power_bands = pd.read_csv(power_bands_path)
    # power_bands = power_bands[~((power_bands['stim_name'] == 'trial150_rep2_4.5Hz_clip5_piano2.wav') & (power_bands['subject'] == 'maestro08_mxnpsq'))]  # this trial is removed in note level epochs (it may have been marked as bad)
    power_bands = power_bands.iloc[:, 3:]  # remove unnecessary columns
    data_matrix = pd.concat([data_matrix.reset_index(drop=True), power_bands.reset_index(drop=True), rms.reset_index(drop=True)], axis=1, ignore_index=False)
    # data_matrix = pd.concat([data_matrix, power_bands], axis=1, ignore_index=False)
    data_matrix.to_csv(os.path.join(save_path, 'behavior_data_matrix.csv'))

    return


def create_seeg_data_matrix(epochs_path, behavior_path, stim_path, save_path,
                       signal_type='raw', montage='mono', remove=None, tmin=2, tmax=7,
                       min_time_interval=0.050, only_repeated=True):

    behavior_all = seeg_behavior_matrix(behavior_path, stim_path)
    print(behavior_all)

    # create save dir
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # get the correct epochs path depending on the desired dataset
    path = Path(epochs_path)

    epochs_list = sorted(list(path.glob('**/*_epochs' + '_' + montage + '_preprocessed_IDA.fif')))

    df_list = []
    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs)
        subject = subject_epochs.name[:6]
        behavior = behavior_all[behavior_all['subject'] == subject]
        assert len(behavior) > 1

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
        metadata['onset_type'] = 0  # add onset type from stim name
        metadata['pitch_interval'] = 0  # add pitch interval
        for i_note in range(metadata.shape[0]):
            metadata.loc[metadata.index[i_note], 'onset_type'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[4][-5])
            if metadata.loc[metadata.index[i_note], 'trial_time'] > 0:
                metadata.loc[metadata.index[i_note], 'pitch_interval'] = np.abs(metadata.loc[metadata.index[i_note - 1], 'pitch'] - metadata.loc[metadata.index[i_note], 'pitch'])

        epochs.metadata = metadata
        regressors = metadata

        # remove onset response
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
        regressors = regressors[regressors['trial_time'].to_numpy() >= tmin]
        epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]
        regressors = regressors[regressors['trial_time'].to_numpy() <= tmax]

        # keep only stim that repeat
        if only_repeated:
            rows_to_keep = []
            already_done = []
            for index, row in epochs.metadata.iterrows():
                if '_rep2_' in row['stim_name']:
                    stim_name_base = row['stim_name'][14:]
                    if stim_name_base not in already_done:
                        matching_rows = epochs.metadata[epochs.metadata['stim_name'].str.contains(stim_name_base)]
                        rows_to_keep.extend(matching_rows.index)
                        already_done.append(stim_name_base)
            mask = epochs.metadata.index.isin(rows_to_keep)
            epochs = epochs[mask]
            regressors = regressors[mask]

        print('Number of notes after min/max trial time:', len(epochs.metadata))

        del epochs

        # Calculate mean surprise and uncertainty in 1-second windows based on actual trial_time before groupby
        for sec in range(8):
            window_start = sec
            window_end = sec + 1
            column_name = f'surprise_{window_start}_{window_end}s'
            # For each row, calculate the mean surprise for the trial within the time window
            regressors[column_name] = regressors.apply(lambda row: regressors[
                (regressors['trial_time'] >= window_start) &
                (regressors['trial_time'] < window_end) &
                (regressors['stim_name'] == row['stim_name'])  # ensure it's the same trial
                ]['surprise_possum_paulrnn'].mean(), axis=1)
        for sec in range(8):
            window_start = sec
            window_end = sec + 1
            column_name = f'uncertainty_{window_start}_{window_end}s'
            # For each row, calculate the mean surprise for the trial within the time window
            regressors[column_name] = regressors.apply(lambda row: regressors[
                (regressors['trial_time'] >= window_start) &
                (regressors['trial_time'] < window_end) &
                (regressors['stim_name'] == row['stim_name'])  # ensure it's the same trial
                ]['entropy_pitch_paulrnn'].mean(), axis=1)

        # # Calculate min and max surprise in 1-second windows based on actual trial_time before groupby
        # for sec in range(8):
        #     window_start = sec
        #     window_end = sec + 1
        #     column_name = f'max_surprise_{window_start}_{window_end}s'
        #     # For each row, calculate the mean surprise for the trial within the time window
        #     regressors[column_name] = regressors.apply(lambda row: regressors[
        #         (regressors['trial_time'] >= window_start) &
        #         (regressors['trial_time'] < window_end) &
        #         (regressors['stim_name'] == row['stim_name'])  # ensure it's the same trial
        #         ]['surprise_mean_paulrnn'].max(), axis=1)
        # for sec in range(8):
        #     window_start = sec
        #     window_end = sec + 1
        #     column_name = f'min_surprise_{window_start}_{window_end}s'
        #     # For each row, calculate the mean surprise for the trial within the time window
        #     regressors[column_name] = regressors.apply(lambda row: regressors[
        #         (regressors['trial_time'] >= window_start) &
        #         (regressors['trial_time'] < window_end) &
        #         (regressors['stim_name'] == row['stim_name'])  # ensure it's the same trial
        #         ]['surprise_mean_paulrnn'].min(), axis=1)

        max_surprise = regressors[['stim_name', 'surprise_mean_paulrnn']].groupby(['stim_name'], as_index=False).max()
        min_surprise = regressors[['stim_name', 'surprise_mean_paulrnn']].groupby(['stim_name'], as_index=False).min()
        std_surprise = regressors[['stim_name', 'surprise_mean_paulrnn']].groupby(['stim_name'], as_index=False).std()
        regressors = regressors.groupby(['stim_name'], as_index=False).mean()
        regressors['max_surprise'] = max_surprise['surprise_mean_paulrnn']
        regressors['min_surprise'] = min_surprise['surprise_mean_paulrnn']
        regressors['std_surprise'] = std_surprise['surprise_mean_paulrnn']

        # add behavior
        regressors['subject'] = ''
        regressors['stim_identity'] = ''
        regressors['seeg_block'] = 0.0
        regressors['position_in_block'] = 0.0
        regressors['rep1'] = 0.0  # one-hot encoding
        regressors['rep2'] = 0.0
        regressors['2.5Hz'] = 0.0
        regressors['3.5Hz'] = 0.0
        regressors['4.5Hz'] = 0.0
        regressors['5.5Hz'] = 0.0
        regressors['6.5Hz'] = 0.0
        regressors['7.5Hz'] = 0.0
        regressors['resp'] = 0.0
        regressors['correct'] = 0.0

        for trial in range(regressors.shape[0]):
            regressors.at[regressors.index[trial], 'subject'] = subject_epochs.name[:6]
            stim_name = regressors.at[regressors.index[trial], 'stim_name']  # finds the stim_nb
            regressors.at[regressors.index[trial], 'stim_identity'] = regressors.at[regressors.index[trial], 'stim_name'].split('_')[2] + '_' + regressors.at[regressors.index[trial], 'stim_name'].split('_')[3]
            regressors.at[regressors.index[trial], 'seeg_block'] = behavior[behavior['stim_name'] == stim_name]['seeg_block']
            regressors.at[regressors.index[trial], 'position_in_block'] = behavior[behavior['stim_name'] == stim_name]['trial']
            for rep in [1, 2]:
                if behavior[behavior['stim_name'] == stim_name]['rep'].values[0] == rep:
                    regressors.at[regressors.index[trial], 'rep' + str(rep)] = 1.0
            for rate in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]:
                if behavior[behavior['stim_name'] == stim_name]['rate'].values[0] == rate:
                    regressors.at[regressors.index[trial], str(rate) + 'Hz'] = 1.0
            regressors.at[regressors.index[trial], 'resp'] = behavior[behavior['stim_name'] == stim_name]['resp']
            regressors.at[regressors.index[trial], 'correct'] = behavior[behavior['stim_name'] == stim_name]['correct']
            # regressors.at[regressors.index[trial], 'correct'] = float(behavior[behavior['stim_name'] == stim_name]['resp'].values[0] == behavior[behavior['stim_name'] == stim_name]['rep'].values[0])

        # add trials since first rep
        regressors['trial_since_rep1'] = 0
        stim_groups = regressors.groupby('stim_identity')
        for stim, group in stim_groups:
            # Check if both rep1 and rep2 exist for this stimulus
            has_rep1 = (group['rep1'] == 1).any()
            has_rep2 = (group['rep2'] == 1).any()
            if not has_rep1 or not has_rep2:
                # Remove all rows for this stimulus if one of the repetitions is missing
                regressors = regressors[regressors['stim_identity'] != stim]
                continue
            # Extract the trial number for each row based on stim_name
            group['trial_number'] = group['stim_name'].apply(lambda x: int(x[5:8]))
            # Get the trial number for the first rep=1
            rep1_trial_number = group.loc[group['rep1'] == 1, 'trial_number'].min()
            # Compute 'trial_since_rep1' for rep=2 trials
            for idx in group.index:
                if group.at[idx, 'rep2'] == 1:  # Process only rep=2 trials
                    regressors.at[idx, 'trial_since_rep1'] = group.at[idx, 'trial_number'] - rep1_trial_number

        df_list.append(regressors)
    data_matrix = pd.concat(df_list, axis=0)

    data_matrix.to_csv(os.path.join(save_path, 'behavior_seeg_data_matrix.csv'))

    return

def correlation_matrix(data_matrix_path):

    data_matrix = pd.read_csv(data_matrix_path).iloc[:, 2:]
    print(data_matrix)

    # plt.switch_backend('TkAgg')
    corr = data_matrix.corr()
    sns.heatmap(corr, cmap='bwr', vmin=-1, vmax=1, xticklabels=1, yticklabels=1)
    plt.show()

    return

def plot_histograms(data_matrix_path, feature_list):

    data_matrix = pd.read_csv(data_matrix_path)
    fig, axes = plt.subplots(len(feature_list) // 4 + 1, 4)
    for f, feat in enumerate(feature_list):
        axes.flat[f].hist(data_matrix[feat], bins=75)
        axes.flat[f].set_title(feat)

    plt.show()
    return

def lme(data_matrix_path, independent_vars, interactions=[], dependent_var='correct', group='subject', rep=None):

    data_matrix = pd.read_csv(data_matrix_path)
    print(data_matrix)

    if rep is not None:
        data_matrix = data_matrix.loc[data_matrix[rep] == 1, :]
    print(data_matrix)

    # indep = sm.add_constant((data_matrix[independent_vars] - data_matrix[independent_vars].mean()) / data_matrix[independent_vars].std())
    # dep = (data_matrix[dependent_var] - data_matrix[dependent_var].mean()) / data_matrix[dependent_var].std()

    print(data_matrix[independent_vars].mean())
    indep = sm.add_constant(data_matrix[independent_vars] - data_matrix[independent_vars].mean())
    # indep = sm.add_constant(data_matrix[independent_vars])
    dep = data_matrix[dependent_var]

    # add interactions
    independent_vars = independent_vars + interactions
    for inter in interactions:
        var1 = inter.split('*')[0]
        var2 = inter.split('*')[1]
        indep[inter] = indep[var1] * indep[var2]

    print('Multicollinearity check ##############')

    for i in range(len(independent_vars)):
        print(f"{independent_vars[i]}:", variance_inflation_factor(indep, i))

    print('Linear Regression ##############')

    model = sm.OLS(dep, indep)
    result = model.fit()
    print(result.summary())

    # sm.graphics.plot_regress_exog(result, 'surprise_mean_paulrnn')
    # plt.show()

    # if not dependent_var == 'rs':
    #     print('Logistic Regression ##############')
    #     if dependent_var == 'resp':
    #         dep_logit = data_matrix[dependent_var] - 1
    #     else:
    #         dep_logit = data_matrix[dependent_var]  # no z-scoring for categorical target
    #     model = sm.Logit(dep_logit, indep)
    #     result = model.fit()
    #     print(result.summary())

    print('Linear Mixed Effects Model ##############')

    # model = MixedLM(endog=dep, exog=indep, groups=data_matrix[group], exog_re=indep)
    model = MixedLM(endog=dep, exog=indep, groups=data_matrix[group])
    result = model.fit()
    print(result.summary())

    # sm.graphics.plot_partregress_grid(result, ['surprise_mean_paulrnn', 'pitch_interval'])
    # # sm.graphics.plot_ccpr_grid(result, ['surprise_mean_paulrnn', 'pitch_interval'])
    # plt.show()

    return


def barplot_on_behavior_matrix(data_matrix_path):

    data_matrix = pd.read_csv(data_matrix_path)

    matrix = data_matrix[['correct', 'surprise_mean_paulrnn']]
    matrix.loc[matrix['correct'] == 1, 'correct'] = 'correct'
    matrix.loc[matrix['correct'] == 0, 'correct'] = 'incorrect'

    custom_palette = sns.color_palette('Set3')
    # ax = sns.violinplot(data=matrix, x='correct', y='surprise_mean_paulrnn', palette=custom_palette)
    ax = sns.boxplot(data=matrix, x='correct', y='surprise_mean_paulrnn', showmeans=True, palette=custom_palette)
    ax = sns.swarmplot(data=matrix, x='correct', y='surprise_mean_paulrnn', size=2, color='k')
    # ax.set_ylabel("d' score (individual mean removed)", fontsize=17)
    # ax.yaxis.set_tick_params(labelsize=13)
    # ax.xaxis.set_tick_params(labelsize=15)

    # posthocs
    pairs = [('correct', 'incorrect')]
    annotator = Annotator(ax, pairs, data=matrix, x='correct', y='surprise_mean_paulrnn')
    annotator.configure(test='t-test_ind', text_format='star', loc='outside', hide_non_significant=True)
    annotator.apply_and_annotate()

    # t test
    # surprise_correct = matrix[matrix['correct'] == 1]['surprise_mean_paulrnn'].to_numpy()
    # surprise_incorrect = matrix[matrix['correct'] == 0]['surprise_mean_paulrnn'].to_numpy()
    # ttest = scipy.stats.ttest_ind(surprise_correct, surprise_incorrect)
    # print('ttest')
    # print(ttest)
    # ax.text(0, ax.get_ylim()[1], f't-test p = {round(ttest.pvalue, 5)}')

    plt.show()


def trial_analysis(data_matrix_path, save_path):
    data_matrix = pd.read_csv(data_matrix_path)
    print(data_matrix)
    data_matrix['resp'] = data_matrix['resp'] - 1

    stim_list = np.unique(data_matrix['stim_identity'].to_numpy())
    print(stim_list)
    print(stim_list.shape)

    stim_list_to_keep = []
    for stim in stim_list:
        x = []
        y = []
        for rep in [1, 2, 3]:
            if len(data_matrix[(data_matrix['stim_identity']==stim) & (data_matrix['rep'+str(rep)]==1)]) > 0:
                x.append(rep)
                y.append(data_matrix[(data_matrix['stim_identity']==stim) & (data_matrix['rep'+str(rep)]==1)]['resp'].mean())
        if len(x) == 3:
            stim_list_to_keep.append(stim)
        plt.plot(x, y, color='k')
    plt.xlabel('rep')
    plt.ylabel('response (average across participants)')
    plt.show()

    to_remove_by_subject = []
    for i, subject in enumerate(np.unique(data_matrix['subject'].to_numpy())):
        to_remove_by_subject.append([])
        for stim in stim_list_to_keep:
            for rep in [1, 2, 3]:
                if len(data_matrix[(data_matrix['subject'] == subject) & (data_matrix['stim_identity'] == stim) & (
                            data_matrix['rep'+str(rep)] == 1)]) == 0:
                    print(subject, stim, rep)
                    to_remove_by_subject[i].append(stim)

    subject_list = []
    stim_list = []
    onset_list = []
    block_list = []
    rate_list = []
    resp1_list = []
    resp2_list = []
    resp3_list = []
    surprise_list = []
    for i, subject in enumerate(np.unique(data_matrix['subject'].to_numpy())):
        for stim in stim_list_to_keep:
            if stim not in to_remove_by_subject[i]:
                subject_list.append(subject)
                stim_list.append(stim)
                onset_list.append(data_matrix[(data_matrix['subject']==subject) & (data_matrix['stim_identity']==stim) & (data_matrix['rep1']==1)].iloc[0]['onset_type'])
                block_list.append(data_matrix[(data_matrix['subject']==subject) & (data_matrix['stim_identity']==stim) & (data_matrix['rep1']==1)].iloc[0]['block'])
                rate_list.append(float(data_matrix[(data_matrix['subject']==subject) & (data_matrix['stim_identity']==stim) & (data_matrix['rep1']==1)].iloc[0]['stim_name'].split('_')[2][0:3]))
                resp1_list.append(data_matrix[(data_matrix['subject']==subject) & (data_matrix['stim_identity']==stim) & (data_matrix['rep1']==1)].iloc[0]['resp'])
                resp2_list.append(data_matrix[(data_matrix['subject']==subject) & (data_matrix['stim_identity']==stim) & (data_matrix['rep2']==1)].iloc[0]['resp'])
                resp3_list.append(data_matrix[(data_matrix['subject']==subject) & (data_matrix['stim_identity']==stim) & (data_matrix['rep3']==1)].iloc[0]['resp'])
                surprise_list.append(data_matrix[(data_matrix['subject']==subject) & (data_matrix['stim_identity']==stim) & (data_matrix['rep1']==1)].iloc[0]['surprise_mean_paulrnn'])

    stim_df = pd.DataFrame({'subject': subject_list, 'stim_identity': stim_list, 'onset_type': onset_list, 'block_1st_rep': block_list, 'note_rate': rate_list, 'resp1': resp1_list, 'resp2': resp2_list, 'resp3': resp3_list, 'surprise': surprise_list})

    stim_df.to_csv(os.path.join(save_path, 'stim_behavior_matrix.csv'))

    return

def trial_analysis_all_rep1(data_matrix_path, save_path):
    data_matrix = pd.read_csv(data_matrix_path)
    print(data_matrix)
    data_matrix['resp'] = data_matrix['resp'] - 1

    stim_list = np.unique(data_matrix['stim_identity'].to_numpy())
    print(stim_list)
    print(stim_list.shape)

    subject_list = []
    stim_list_df = []
    onset_list = []
    block_list = []
    rate_list = []
    resp1_list = []
    rs1_list = []
    for i, subject in enumerate(np.unique(data_matrix['subject'].to_numpy())):
        print(subject)
        for stim in stim_list:
            if len(data_matrix[(data_matrix['subject']==subject) & (data_matrix['stim_identity']==stim) & (data_matrix['rep1']==1)]) > 0:
                subject_list.append(subject)
                stim_list_df.append(stim)
                onset_list.append(data_matrix[(data_matrix['subject']==subject) & (data_matrix['stim_identity']==stim) & (data_matrix['rep1']==1)].iloc[0]['onset_type'])
                block_list.append(data_matrix[(data_matrix['subject']==subject) & (data_matrix['stim_identity']==stim) & (data_matrix['rep1']==1)].iloc[0]['block'])
                rate_list.append(float(data_matrix[(data_matrix['subject']==subject) & (data_matrix['stim_identity']==stim) & (data_matrix['rep1']==1)].iloc[0]['stim_name'].split('_')[2][0:3]))
                resp1_list.append(data_matrix[(data_matrix['subject']==subject) & (data_matrix['stim_identity']==stim) & (data_matrix['rep1']==1)].iloc[0]['resp'])
                rs1_list.append(data_matrix[(data_matrix['subject']==subject) & (data_matrix['stim_identity']==stim) & (data_matrix['rep1']==1)].iloc[0]['rs'])

    stim_df = pd.DataFrame({'subject': subject_list, 'stim_identity': stim_list_df, 'onset_type': onset_list, 'block_1st_rep': block_list, 'note_rate': rate_list, 'resp1': resp1_list, 'rs1': rs1_list})

    stim_df.to_csv(os.path.join(save_path, 'stim_behavior_matrix_rep1.csv'))

    return


def bias_evolution(stim_behavior_path):
    stim_behavior_matrix = pd.read_csv(stim_behavior_path)
    # stim_behavior_matrix = stim_behavior_matrix[stim_behavior_matrix['block_1st_rep'] > 2]
    print(stim_behavior_matrix)

    n_subjects = len(np.unique(stim_behavior_matrix['subject']))
    print('n subjects:', n_subjects)

    block_groups = stim_behavior_matrix.groupby(['block_1st_rep'], as_index=False).mean(numeric_only=True)[['resp1', 'resp2', 'resp3']]
    print(block_groups)
    block_groups_std = stim_behavior_matrix.groupby(['block_1st_rep'], as_index=False).std(numeric_only=True)[['resp1', 'resp2', 'resp3']]

    onset_groups = stim_behavior_matrix.groupby(['block_1st_rep', 'onset_type'], as_index=False).mean(numeric_only=True)
    print(onset_groups)
    onset_groups_std = stim_behavior_matrix.groupby(['block_1st_rep', 'onset_type'], as_index=False).std(numeric_only=True)

    fig, axes = plt.subplots(2, 2, sharex=False)

    color_list = ['#087e8b', '#0b3954', '#B6ADFF', '#6859D9', '#ff5a5f', '#c81d25']
    x = onset_groups[onset_groups['onset_type']==0]['block_1st_rep']
    # axes[0, 0].plot(x, np.mean(block_groups, axis=1), color='k', ls='--', label='full average')
    for r, rep in enumerate([1, 2, 3]):
        y = onset_groups[onset_groups['onset_type'] == 0][f'resp{rep}']
        axes[0, 0].plot(x+r, y, color=color_list[2*r], label=f'sharp - rep {rep}')
        sem = onset_groups_std[onset_groups_std['onset_type'] == 0][f'resp{rep}'] / np.sqrt(n_subjects)
        axes[0, 0].fill_between(x+r, y - sem, y + sem, color=color_list[2*r], alpha=0.6)
        y = onset_groups[onset_groups['onset_type'] == 2][f'resp{rep}']
        axes[0, 0].plot(x+r, y, color=color_list[2*r+1], label=f'smooth - rep {rep}')
        sem = onset_groups_std[onset_groups_std['onset_type'] == 2][f'resp{rep}'] / np.sqrt(n_subjects)
        axes[0, 0].fill_between(x + r, y - sem, y + sem, color=color_list[2*r+1], alpha=0.6)
        axes[0, 0].plot(x+r, block_groups[f'resp{rep}'], color=color_list[2*r], ls='--', label=f'average - rep {rep}')

    # axes[0, 0].set_xlabel('Block in which the stim was first played')
    axes[0, 0].set_xlabel('Block')
    axes[0, 0].set_ylabel('Average response (1 is old)')
    # axes[0, 0].set_ylim(0,1)
    axes[0, 0].legend()
    axes[0, 0].set_title('onset types')

    # rate
    rate_groups = stim_behavior_matrix.groupby(['block_1st_rep', 'note_rate'], as_index=False).mean(numeric_only=True)
    print(rate_groups)

    pick_colors = []
    i = 0
    while len(pick_colors) < 15:
        if not i % 3 == 0:
            pick_colors.append(i)
        i += 1
    color_list = [plt.cm.get_cmap('Set1')(x) for x in range(5)]
    x = rate_groups[rate_groups['note_rate'] == 3.5]['block_1st_rep']
    # fig, axes = plt.subplots(3, 1)
    for i, rep in enumerate([1, 2, 3]):
        axes.flat[i+1].plot(x+i, block_groups['resp'+str(rep)], color='k', ls='--', label='average')
        axes.flat[i+1].plot(x+i, rate_groups[rate_groups['note_rate'] == 2.5]['resp' + str(rep)], color=color_list[0], label='2.5 Hz')
        axes.flat[i+1].plot(x+i, rate_groups[rate_groups['note_rate'] == 3.5]['resp' + str(rep)], color=color_list[1], label='3.5 Hz')
        axes.flat[i+1].plot(x+i, rate_groups[rate_groups['note_rate'] == 4.5]['resp' + str(rep)], color=color_list[2], label='4.5 Hz')
        axes.flat[i+1].plot(x+i, rate_groups[rate_groups['note_rate'] == 5.5]['resp' + str(rep)], color=color_list[3], label='5.5 Hz')
        axes.flat[i+1].plot(x+i, rate_groups[rate_groups['note_rate'] == 6.5]['resp' + str(rep)], color=color_list[4], label='6.5 Hz')
        # axes.flat[i+1].set_xlabel('Block in which the stim was first played')
        axes.flat[i+1].set_xlabel('Block')
        axes.flat[i+1].set_ylabel('Average response (1 is old)')
        axes.flat[i+1].set_title('rep ' + str(rep))
        axes.flat[i+1].legend()

    # individual slopes onset
    onset_groups = stim_behavior_matrix.groupby(['subject', 'block_1st_rep', 'onset_type'], as_index=False).mean(numeric_only=True)
    print(onset_groups)

    fig, axes = plt.subplots(1, 1, squeeze=False)
    for subject in np.unique(onset_groups['subject']):
        print(subject)
        axes[0, 0].plot(x, onset_groups[(onset_groups['subject'] == subject) & (onset_groups['onset_type'] == 0)]['resp1'], color=color_list[0])
        axes[0, 0].plot(x, onset_groups[(onset_groups['subject'] == subject) & (onset_groups['onset_type'] == 2)]['resp1'], color=color_list[1])
    axes[0, 0].set_xlabel('Block')
    axes[0, 0].set_ylabel('Average response (1 is old)')
    axes[0, 0].legend(['Sharp', 'Smooth'])
    axes[0, 0].set_title('onset types rep 1')

    fig, axes = plt.subplots(1, 1, squeeze=False)
    axes[0, 0].set_xlim(2, 7)
    axes[0, 0].set_ylim(0, 1)
    for subject in np.unique(onset_groups['subject']):
        m, b = np.polyfit(x, onset_groups[(onset_groups['subject'] == subject) & (onset_groups['onset_type'] == 0)]['resp1'], deg=1)
        axes[0, 0].axline(xy1=(4.5, 0.5), slope=m, color=color_list[0])
        # axes[0, 0].axline(xy1=(0, b), slope=m, color=color_list[0])
        m, b = np.polyfit(x, onset_groups[(onset_groups['subject'] == subject) & (onset_groups['onset_type'] == 2)]['resp1'], deg=1)
        axes[0, 0].axline(xy1=(4.5, 0.5), slope=m, color=color_list[1])
        # axes[0, 0].axline(xy1=(0, b), slope=m, color=color_list[1])
    axes[0, 0].set_xlabel('Block')
    axes[0, 0].set_ylabel('Average response (1 is old)')
    axes[0, 0].legend(['Sharp', 'Smooth'])
    axes[0, 0].set_title('Individual slopes onset types rep 1')

    # Slope smooth vs slope sharp
    fig, axes = plt.subplots(1, 1, squeeze=False)
    # axes[0, 0].set_xlim(2, 7)
    # axes[0, 0].set_ylim(0, 1)
    for subject in np.unique(onset_groups['subject']):
        m_sharp, b_sharp = np.polyfit(x, onset_groups[(onset_groups['subject'] == subject) & (onset_groups['onset_type'] == 0)]['resp1'], deg=1)
        m_smooth, b_smooth = np.polyfit(x, onset_groups[(onset_groups['subject'] == subject) & (onset_groups['onset_type'] == 2)]['resp1'], deg=1)
        axes[0, 0].scatter(m_sharp, m_smooth, c='b')
    axes[0, 0].plot(axes[0, 0].get_xlim(), axes[0, 0].get_xlim(), 'k--')
    axes[0, 0].set_xlabel('Slope sharp')
    axes[0, 0].set_ylabel('Slope smooth')
    axes[0, 0].set_title('Individual slopes comparison')

    plt.figure()
    subject_list = []
    onset_list = []
    slope_list = []
    for subject in np.unique(onset_groups['subject']):
        for o, onset in enumerate([0, 2]):
            m, b = np.polyfit(x, onset_groups[(onset_groups['subject'] == subject) & (onset_groups['onset_type'] == onset)]['resp1'], deg=1)
            subject_list.append(subject)
            onset_list.append('onset_' + str(onset))
            slope_list.append(m)
    slope_df = pd.DataFrame({'subject': subject_list, 'onset': onset_list, 'slope': slope_list})

    ax = sns.boxplot(data=slope_df, x='onset', y='slope', showmeans=True, palette='Set3')
    ax = sns.swarmplot(data=slope_df, x='onset', y='slope', color='k', size=4)
    ax.set_ylabel("Individual slope (evolution of response with block)", fontsize=17)
    ax.set_title("Individual slope rep 1", fontsize=17)
    ax.yaxis.set_tick_params(labelsize=13)
    ax.xaxis.set_tick_params(labelsize=15)
    anova = AnovaRM(data=slope_df, depvar="slope", subject='subject', within=['onset'])
    anova = anova.fit()
    print('Anova onset slope')
    print(anova)
    ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

    # individual slopes rate
    rate_groups = stim_behavior_matrix.groupby(['subject', 'block_1st_rep', 'note_rate'], as_index=False).mean(numeric_only=True)
    print(rate_groups)

    fig, axes = plt.subplots(1, 1, squeeze=False)
    axes[0, 0].set_xlim(2, 7)
    axes[0, 0].set_ylim(0, 1)
    for subject in np.unique(rate_groups['subject']):
        if not subject == 'maestro08_mxnpsq':  # this subject misses one condition
            for r, rate in enumerate([2.5, 3.5, 4.5, 5.5, 6.5]):
                m, b = np.polyfit(x, rate_groups[(rate_groups['subject'] == subject) & (rate_groups['note_rate'] == rate)]['resp1'], deg=1)
                # axes[0, 0].axline(xy1=(0, b), slope=m, color=color_list[r])
                axes[0, 0].axline(xy1=(4.5, 0.5), slope=m, color=color_list[r])
    axes[0, 0].set_xlabel('Block')
    axes[0, 0].set_ylabel('Average response (1 is old)')
    axes[0, 0].legend(['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz'])
    axes[0, 0].set_title('Individual slopes note rate rep 1')

    plt.figure()
    subject_list = []
    rate_list = []
    slope_list = []
    for subject in np.unique(rate_groups['subject']):
        if not subject == 'maestro08_mxnpsq':  # this subject misses one condition
            for r, rate in enumerate([2.5, 3.5, 4.5, 5.5, 6.5]):
                m, b = np.polyfit(x, rate_groups[(rate_groups['subject'] == subject) & (rate_groups['note_rate'] == rate)]['resp1'], deg=1)
                subject_list.append(subject)
                rate_list.append(str(rate) + 'Hz')
                slope_list.append(m)
    slope_df = pd.DataFrame({'subject': subject_list, 'rate': rate_list, 'slope': slope_list})

    ax = sns.boxplot(data=slope_df, x='rate', y='slope', showmeans=True, palette='Set3', width=0.8, linewidth=3.5)
    ax = sns.swarmplot(data=slope_df, x='rate', y='slope', color='k', size=9)
    ax.set_ylabel("Individual slope (evolution of response with block)", fontsize=17)
    ax.set_title("Individual slope rep 1", fontsize=17)
    ax.yaxis.set_tick_params(labelsize=13)
    ax.xaxis.set_tick_params(labelsize=15)
    anova = AnovaRM(data=slope_df, depvar="slope", subject='subject', within=['rate'])
    anova = anova.fit()
    print('Anova rate slope')
    print(anova)
    ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')
    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']
    pairs = []
    # for var1 in range(len(rates)):
    #     for var2 in range(var1 + 1, len(rates)):
    #         pairs.append((rates[var1], rates[var2]))
    # annotator = Annotator(ax, pairs, data=slope_df, x='rate', y='slope')
    # annotator.configure(test='t-test_paired', text_format='star', loc='inside', hide_non_significant=True)
    # annotator.apply_and_annotate()

    # Boxplot per block
    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
    fig2, ax = plt.subplots(1, 1)
    rates = [2.5, 3.5, 4.5, 5.5, 6.5]
    block_groups = {
        '2-4': [2, 3, 4],
        '5-7': [5, 6, 7]
    }
    block_groups = {
        '2-3': [2, 3],
        '4-5': [4, 5],
        '6-7': [6, 7],
    }
    block_groups = {
        '2': [2],
        '3': [3],
        '4': [4],
        '5': [5],
        '6': [6],
        '7': [7],
    }
    color_list = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(block_groups)))  # one color per block
    for b, (group_name, blocks) in enumerate(block_groups.items()):
        rate_list = []
        response_list = []
        subject_list = []
        for subject in np.unique(rate_groups['subject']):
            if subject != 'maestro08_mxnpsq':  # this subject misses one condition
                for r, rate in enumerate(rates):
                    response_list.append(rate_groups.loc[(rate_groups['subject'] == subject) & (
                            rate_groups['note_rate'] == rate) & (rate_groups['block_1st_rep'].isin(blocks))]['resp1'].values.mean())
                    rate_list.append(str(rate) + 'Hz')
                    subject_list.append(subject)
        response_df = pd.DataFrame({'subject': subject_list, 'rate': rate_list, 'response': response_list})
        sns.boxplot(data=response_df, x='rate', y='response', showmeans=True, palette='Set3', ax=axes.flat[b])
        sns.swarmplot(data=response_df, x='rate', y='response', color='k', size=4, ax=axes.flat[b])
        axes.flat[b].set_ylabel("Average response")
        axes.flat[b].set_title(f"Block {group_name}")
        fig.suptitle('Repetition 1 only')

        mean = response_df.groupby(['rate'], as_index=False).mean(numeric_only=True)['response']
        sem = response_df.groupby(['rate'], as_index=False).std(numeric_only=True)['response'] / np.sqrt(
            len(np.unique(rate_groups['subject'])) - 1)  # -1 because we removed one subject
        ax.plot(rates, mean, color=color_list[b], lw=6, label=f"Block {group_name}")
        ax.fill_between(rates, mean - sem, mean + sem, color=color_list[b], alpha=0.3, lw=0)
    ax.set_xlabel('Note rate (Hz)')
    ax.set_ylabel('Average response')
    ax.legend()

    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
    fig2, ax = plt.subplots(1, 1)
    for b, (group_name, blocks) in enumerate(block_groups.items()):
        onset_list = []
        response_list = []
        subject_list = []
        for subject in np.unique(onset_groups['subject']):
            for o, onset in enumerate([0, 2]):
                response_list.append(onset_groups.loc[(onset_groups['subject'] == subject) & (
                        onset_groups['onset_type'] == onset) & (onset_groups['block_1st_rep'].isin(blocks))]['resp1'].values.mean())
                onset_list.append('onset_' + str(onset))
                subject_list.append(subject)
        response_df = pd.DataFrame({'subject': subject_list, 'onset': onset_list, 'response': response_list})
        sns.boxplot(data=response_df, x='onset', y='response', showmeans=True, palette='Set3', ax=axes.flat[b])
        sns.swarmplot(data=response_df, x='onset', y='response', color='k', size=4, ax=axes.flat[b])
        axes.flat[b].set_ylabel("Average response")
        axes.flat[b].set_title(f"Block {group_name}")
        fig.suptitle('Repetition 1 only')

        mean = response_df.groupby(['onset'], as_index=False).mean(numeric_only=True)['response']
        sem = response_df.groupby(['onset'], as_index=False).std(numeric_only=True)['response'] / np.sqrt(
            len(np.unique(onset_groups['subject'])))
        ax.plot([0, 2], mean, color=color_list[b], lw=6, label=f"Block {group_name}")
        ax.fill_between([0, 2], mean - sem, mean + sem, color=color_list[b], alpha=0.3, lw=0)
    ax.set_xlabel('Onset (0 is sharp, 2 is smooth)')
    ax.set_ylabel('Average response')
    ax.legend()

    plt.show()


def bias_evolution_rep1(stim_behavior_rep1_path, variable='resp1'):
    stim_behavior_matrix = pd.read_csv(stim_behavior_rep1_path)
    # stim_behavior_matrix = stim_behavior_matrix[stim_behavior_matrix['block_1st_rep'] > 2]
    print(stim_behavior_matrix)

    n_subjects = len(np.unique(stim_behavior_matrix['subject']))
    print('n subjects:', n_subjects)

    onset_groups = stim_behavior_matrix.groupby(['block_1st_rep', 'onset_type'], as_index=False).mean(numeric_only=True)
    x = onset_groups[onset_groups['onset_type'] == 0]['block_1st_rep']

    # individual slopes onset
    onset_groups = stim_behavior_matrix.groupby(['subject', 'block_1st_rep', 'onset_type'], as_index=False).mean(numeric_only=True)
    print(onset_groups)
    color_list = ['#087e8b', '#0b3954', '#B6ADFF', '#6859D9', '#ff5a5f', '#c81d25']

    fig, axes = plt.subplots(1, 1, squeeze=False)
    axes[0, 0].set_xlim(2, 7)
    axes[0, 0].set_ylim(0, 1)
    for subject in np.unique(onset_groups['subject']):
        m, b = np.polyfit(x, onset_groups[(onset_groups['subject'] == subject) & (onset_groups['onset_type'] == 0)][
            variable], deg=1)
        axes[0, 0].axline(xy1=(4.5, 0.5), slope=m, color=color_list[0])
        # axes[0, 0].axline(xy1=(0, b), slope=m, color=color_list[0])
        m, b = np.polyfit(x, onset_groups[(onset_groups['subject'] == subject) & (onset_groups['onset_type'] == 2)][
            variable], deg=1)
        axes[0, 0].axline(xy1=(4.5, 0.5), slope=m, color=color_list[1])
        # axes[0, 0].axline(xy1=(0, b), slope=m, color=color_list[1])
    axes[0, 0].set_xlabel('Block')
    axes[0, 0].set_ylabel(variable)
    axes[0, 0].legend(['Sharp', 'Smooth'])
    axes[0, 0].set_title('Individual slopes onset types rep 1')

    # Slope smooth vs slope sharp
    fig, axes = plt.subplots(1, 1, squeeze=False)
    # axes[0, 0].set_xlim(2, 7)
    # axes[0, 0].set_ylim(0, 1)
    for subject in np.unique(onset_groups['subject']):
        m_sharp, b_sharp = np.polyfit(x, onset_groups[
            (onset_groups['subject'] == subject) & (onset_groups['onset_type'] == 0)][variable], deg=1)
        m_smooth, b_smooth = np.polyfit(x, onset_groups[
            (onset_groups['subject'] == subject) & (onset_groups['onset_type'] == 2)][variable], deg=1)
        axes[0, 0].scatter(m_sharp, m_smooth, c='b')
    axes[0, 0].plot(axes[0, 0].get_xlim(), axes[0, 0].get_xlim(), 'k--')
    axes[0, 0].set_xlabel('Slope sharp')
    axes[0, 0].set_ylabel('Slope smooth')
    axes[0, 0].set_title('Individual slopes comparison')

    plt.figure()
    subject_list = []
    onset_list = []
    slope_list = []
    for subject in np.unique(onset_groups['subject']):
        for o, onset in enumerate([0, 2]):
            m, b = np.polyfit(x, onset_groups[
                (onset_groups['subject'] == subject) & (onset_groups['onset_type'] == onset)][variable], deg=1)
            subject_list.append(subject)
            onset_list.append('onset_' + str(onset))
            slope_list.append(m)
    slope_df = pd.DataFrame({'subject': subject_list, 'onset': onset_list, 'slope': slope_list})

    ax = sns.boxplot(data=slope_df, x='onset', y='slope', showmeans=True, palette='Set3')
    ax = sns.swarmplot(data=slope_df, x='onset', y='slope', color='k', size=4)
    ax.set_ylabel(f"Individual slope (evolution of {variable} with block)", fontsize=17)
    ax.set_title("Individual slope rep 1", fontsize=17)
    ax.yaxis.set_tick_params(labelsize=13)
    ax.xaxis.set_tick_params(labelsize=15)
    anova = AnovaRM(data=slope_df, depvar="slope", subject='subject', within=['onset'])
    anova = anova.fit()
    print('Anova onset slope')
    print(anova)
    ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

    # rate
    rate_groups = stim_behavior_matrix.groupby(['block_1st_rep', 'note_rate'], as_index=False).mean(numeric_only=True)
    print(rate_groups)
    x = rate_groups[rate_groups['note_rate'] == 3.5]['block_1st_rep']

    # individual slopes rate
    rate_groups = stim_behavior_matrix.groupby(['subject', 'block_1st_rep', 'note_rate'], as_index=False).mean(numeric_only=True)
    color_list = [plt.cm.get_cmap('Set1')(x) for x in range(5)]
    print(rate_groups)

    fig, axes = plt.subplots(1, 1, squeeze=False)
    axes[0, 0].set_xlim(2, 7)
    axes[0, 0].set_ylim(0, 1)
    for subject in np.unique(rate_groups['subject']):
        if not subject == 'maestro08_mxnpsq':  # this subject misses one condition
            for r, rate in enumerate([2.5, 3.5, 4.5, 5.5, 6.5]):
                m, b = np.polyfit(x,
                                  rate_groups[(rate_groups['subject'] == subject) & (rate_groups['note_rate'] == rate)][
                                      variable], deg=1)
                # axes[0, 0].axline(xy1=(0, b), slope=m, color=color_list[r])
                axes[0, 0].axline(xy1=(4.5, 0.5), slope=m, color=color_list[r])
    axes[0, 0].set_xlabel('Block')
    axes[0, 0].set_ylabel(variable)
    axes[0, 0].legend(['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz'])
    axes[0, 0].set_title('Individual slopes note rate rep 1')

    plt.figure()
    subject_list = []
    rate_list = []
    slope_list = []
    for subject in np.unique(rate_groups['subject']):
        if not subject == 'maestro08_mxnpsq':  # this subject misses one condition
            for r, rate in enumerate([2.5, 3.5, 4.5, 5.5, 6.5]):
                m, b = np.polyfit(x,
                                  rate_groups[(rate_groups['subject'] == subject) & (rate_groups['note_rate'] == rate)][
                                      variable], deg=1)
                subject_list.append(subject)
                rate_list.append(str(rate) + 'Hz')
                slope_list.append(m)
    slope_df = pd.DataFrame({'subject': subject_list, 'rate': rate_list, 'slope': slope_list})

    ax = sns.boxplot(data=slope_df, x='rate', y='slope', showmeans=True, palette='Set3')
    ax = sns.swarmplot(data=slope_df, x='rate', y='slope', color='k', size=4)
    ax.set_ylabel(f"Individual slope (evolution of {variable} with block)", fontsize=17)
    ax.set_title("Individual slope rep 1", fontsize=17)
    ax.yaxis.set_tick_params(labelsize=13)
    ax.xaxis.set_tick_params(labelsize=15)
    anova = AnovaRM(data=slope_df, depvar="slope", subject='subject', within=['rate'])
    anova = anova.fit()
    print('Anova rate slope')
    print(anova)
    ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')
    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']
    pairs = []
    for var1 in range(len(rates)):
        for var2 in range(var1 + 1, len(rates)):
            pairs.append((rates[var1], rates[var2]))
    annotator = Annotator(ax, pairs, data=slope_df, x='rate', y='slope')
    annotator.configure(test='t-test_paired', text_format='star', loc='inside', hide_non_significant=True)
    annotator.apply_and_annotate()

    # Boxplot per block
    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
    fig2, ax = plt.subplots(1, 1)
    rates = [2.5, 3.5, 4.5, 5.5, 6.5]
    block_groups = {
        '2-5': [2, 3, 4, 5],
        '6-9': [6, 7, 8, 9]
    }
    block_groups = {
        '2-3': [2, 3],
        '4-5': [4, 5],
        '6-7': [6, 7],
        '8-9': [8, 9],
    }
    block_groups = {
        '2': [2],
        '3': [3],
        '4': [4],
        '5': [5],
        '6': [6],
        '7': [7],
        '8': [8],
        '9': [9],
    }
    color_list = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(block_groups)))  # one color per block
    for b, (group_name, blocks) in enumerate(block_groups.items()):
        rate_list = []
        response_list = []
        subject_list = []
        for subject in np.unique(rate_groups['subject']):
            if subject != 'maestro08_mxnpsq':  # this subject misses one condition
                for r, rate in enumerate(rates):
                    response_list.append((rate_groups.loc[(rate_groups['subject'] == subject) & (
                                rate_groups['note_rate'] == rate) & (rate_groups['block_1st_rep'].isin(blocks))][
                                             variable].values.mean()))
                    rate_list.append(str(rate) + 'Hz')
                    subject_list.append(subject)
        response_df = pd.DataFrame({'subject': subject_list, 'rate': rate_list, 'response': response_list})
        sns.boxplot(data=response_df, x='rate', y='response', showmeans=True, palette='Set3', ax=axes.flat[b])
        sns.swarmplot(data=response_df, x='rate', y='response', color='k', size=4, ax=axes.flat[b])
        axes.flat[b].set_ylabel(variable)
        axes.flat[b].set_title(f"Block {group_name}")
        fig.suptitle('Repetition 1 only')

        mean = response_df.groupby(['rate'], as_index=False).mean(numeric_only=True)['response']
        sem = response_df.groupby(['rate'], as_index=False).std(numeric_only=True)['response'] / np.sqrt(len(np.unique(rate_groups['subject'])) - 1)  # -1 because we removed one subject
        ax.plot(rates, mean, color=color_list[b], lw=6, label=f"Block {group_name}")
        ax.fill_between(rates, mean - sem, mean + sem, color=color_list[b], alpha=0.3, lw=0)
    ax.set_xlabel('Note rate (Hz)')
    ax.set_ylabel(variable)
    ax.legend()

    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
    fig2, ax = plt.subplots(1, 1)
    for b, (group_name, blocks) in enumerate(block_groups.items()):
        onset_list = []
        response_list = []
        subject_list = []
        for subject in np.unique(onset_groups['subject']):
            for o, onset in enumerate([0, 2]):
                response_list.append(onset_groups.loc[(onset_groups['subject'] == subject) & (
                        onset_groups['onset_type'] == onset) & (onset_groups['block_1st_rep'].isin(blocks))][variable].values.mean())
                onset_list.append('onset_' + str(onset))
                subject_list.append(subject)
        response_df = pd.DataFrame({'subject': subject_list, 'onset': onset_list, 'response': response_list})
        sns.boxplot(data=response_df, x='onset', y='response', showmeans=True, palette='Set3', ax=axes.flat[b])
        sns.swarmplot(data=response_df, x='onset', y='response', color='k', size=4, ax=axes.flat[b])
        axes.flat[b].set_ylabel(variable)
        axes.flat[b].set_title(f"Block {group_name}")
        fig.suptitle('Repetition 1 only')

        mean = response_df.groupby(['onset'], as_index=False).mean(numeric_only=True)['response']
        sem = response_df.groupby(['onset'], as_index=False).std(numeric_only=True)['response'] / np.sqrt(len(np.unique(onset_groups['subject'])))
        ax.plot([0, 2], mean, color=color_list[b], lw=6, label=f"Block {group_name}")
        ax.fill_between([0, 2], mean - sem, mean + sem, color=color_list[b], alpha=0.3, lw=0)
    ax.set_xlabel('Onset (0 is sharp, 2 is smooth)')
    ax.set_ylabel(variable)
    ax.legend()

    plt.show()


def bias_evolution_surprise(stim_behavior_path):
    stim_behavior_matrix = pd.read_csv(stim_behavior_path)
    # stim_behavior_matrix = stim_behavior_matrix[stim_behavior_matrix['block_1st_rep'] > 2]
    print(stim_behavior_matrix)

    n_subjects = len(np.unique(stim_behavior_matrix['subject']))
    print('n subjects:', n_subjects)

    # Boxplot per block
    fig, axes = plt.subplots(1, 3)
    block_groups = {
        '2-4': [2, 3, 4],
        '5-7': [5, 6, 7]
    }
    block_groups = {
        '2-3': [2, 3],
        '4-5': [4, 5],
        '6-7': [6, 7],
    }
    block_groups = {
        '2': [2],
        '3': [3],
        '4': [4],
        '5': [5],
        '6': [6],
        '7': [7],
    }
    color_list = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(block_groups)))  # one color per block
    stim_behavior_matrix['resp1'] = 1 - stim_behavior_matrix['resp1']  # transform resp into correct
    for r, resp in enumerate(['resp1', 'resp2', 'resp3']):
        for b, (group_name, blocks) in enumerate(block_groups.items()):
            block_behavior = stim_behavior_matrix[stim_behavior_matrix['block_1st_rep'].isin(blocks)][['surprise', resp]]
            print(block_behavior)

            # df_avg = block_behavior.groupby('surprise', as_index=False)[resp].mean()
            # axes[r].plot(df_avg['surprise'], df_avg[resp], lw=2, label=f"Block {group_name}", color=color_list[b])

            quadratic, slope, intercept = np.polyfit(block_behavior['surprise'], block_behavior[resp], 2)
            x_range = np.linspace(min(block_behavior['surprise']), max(block_behavior['surprise']), 100)
            axes[r].plot(x_range, quadratic * x_range**2 + slope * x_range + intercept, lw=3, label=f"Block {group_name}", color=color_list[b])

        axes[r].set_xlabel('Surprise')
        # axes[r].set_ylabel('Average response')
        axes[r].set_ylabel('Average correct')
        axes[r].set_title(f'Repetition {r+1}')
        axes[r].legend()

    plt.show()


def plot_glmm_coefficients(csv_path_list, model_name_list):

    # Create a figure and a set of subplots
    # colors = ['#ff5a5f', '#ff5a5f', '#878787', '#087e8b', '#087e8b', '#087e8b']
    colors = ['#878787', '#878787', '#3998a2', '#3998a2', '#3998a2']
    same_colors = ['#878787', '#878787', '#087e8b', '#087e8b', '#087e8b']
    # colors_2 = ['#FF8F5E', '#FF8F5E', '#BDBDBD', '#06AEC1', '#06AEC1', '#06AEC1']
    colors_2 = ['#AA2024', '#AA2024', '#424242', '#3c6176', '#3c6176', '#3c6176']
    color_lists = [colors, colors_2, colors]
    fig, ax = plt.subplots(1, len(csv_path_list), sharey=True)
    # indiv_fig, indiv_ax = plt.subplots(1, 1)
    # indiv_ax.axvline(0, color='k', ls='--')
    shift_list = [-0.1, 0.1]
    for c, csv_path in enumerate(csv_path_list):
        # Load the CSV file
        df = pd.read_csv(csv_path)
        print(df)

        df = df[df['term'] != '(Intercept)']
        df = df[df['term'] != 'reprep3'].reset_index()
        df = df[df['term'] != 'block'].reset_index()

        # Number of coefficients
        num_coefs = len(df)

        # Y positions for each coefficient
        y_pos = range(num_coefs)

        ax[c].axvline(0, color='k', ls='--')

        # Plot lines for each coefficient
        for i in y_pos:
            term = df.loc[i, 'term']
            estimate = df.loc[i, 'estimate']
            std_error = df.loc[i, 'std_error']
            p_value = df.loc[i, 'p_value']

            # Determine the edge color based on significance
            edge_color = 'red' if p_value < 0.05 else 'black'

            # Draw a horizontal line at the y position
            ax[c].plot([estimate - std_error, estimate + std_error], [i, i], '-', lw=10, color=same_colors[i])
            # indiv_ax.plot([estimate - std_error, estimate + std_error], [i + shift_list[c], i + shift_list[c]], '-', lw=10, color=color_lists[c][i])

            # Draw an empty circle at the estimate
            ax[c].plot(estimate, i, 'o', markersize=15, markerfacecolor='white', markeredgecolor=same_colors[i], markeredgewidth=5)
            # indiv_ax.plot(estimate, i + shift_list[c], 'o', markersize=15, markerfacecolor='white', markeredgecolor=color_lists[c][i], markeredgewidth=5)

            # Add a star on top of the circle if the coefficient is significant
            if p_value < 0.05:
                ax[c].plot(estimate, y_pos[i] - 0.15, '*', color='k', markersize=7)
                # indiv_ax.plot(estimate, y_pos[i] - 0.15 + shift_list[c], '*', color='k', markersize=7)

        xabs_max = abs(max(ax[c].get_xlim()))
        ax[c].set_xlim(xmin=-xabs_max, xmax=xabs_max)
        ax[c].set_yticks(y_pos)
        ax[c].set_yticklabels(df['term'])
        ax[c].set_xlabel('Coefficient value (a.u.)')
        ax[c].set_title(model_name_list[c])
    # indiv_ax.set_yticks(y_pos)
    # indiv_ax.set_yticklabels(df['term'])
    # indiv_ax.set_xlabel('Coefficient Value')

    # color_legend = ['#ff5a5f', '#AA2024', '#878787', '#424242', '#3998a2', '#3c6176']
    # custom_lines = [Line2D([0], [0], color=color, lw=8) for color in color_legend]
    # indiv_ax.legend(custom_lines, ['Acoustic old/new', 'Acoustic correct/incorrect', 'Block old/new', 'Block correct/incorrect', 'Statistical old/new', 'Statistical correct/incorrect'])

    color_legend = ['#878787', '#087e8b']
    custom_lines = [Line2D([0], [0], color=color, lw=8) for color in color_legend]
    ax[0].legend(custom_lines, ['Acoustic features', 'Statistical features'])

    ax[0].invert_yaxis()  # Invert y-axis to have the first term at the top
    # indiv_ax.invert_yaxis()  # Invert y-axis to have the first term at the top

    plt.show()


def save_pred_feat_csv(data_matrix_path, save_path):
    data_matrix = pd.read_csv(data_matrix_path)

    stims = np.unique(data_matrix['stim_name'])
    print(stims)
    surprise_list = []
    pitch_entropy_list = []
    timing_predictability_list = []
    for stim in stims:
        surprise_list.append(data_matrix[data_matrix['stim_name'] == stim]['surprise_mean_paulrnn'].iloc[0])
        pitch_entropy_list.append(data_matrix[data_matrix['stim_name'] == stim]['entropy_pitch_paulrnn'].iloc[0])
        timing_predictability_list.append(data_matrix[data_matrix['stim_name'] == stim]['entropy_timing_paulrnn'].iloc[0])

    df = pd.DataFrame({'stim_names': stims, 'surprise': surprise_list, 'pitch_entropy': pitch_entropy_list, 'timing_predictability': timing_predictability_list})

    df.to_csv(os.path.join(save_path, 'stim_predictive_features.csv'))


if __name__ == '__main__':

    '''create_data_matrix'''
    # create_data_matrix(epochs_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta',
    #                    behavior_path='/home/mathieupvc/These/data/meg_maestro/data',
    #                    ecg_path='/home/mathieupvc/These/data/meg_maestro/ecg_epochs',
    #                    gmsi_path='/home/mathieupvc/These/data/meg_maestro/gmsi_processed.csv',
    #                    power_bands_path='/home/mathieupvc/These/data/meg_maestro/average_power_bands.csv',
    #                    rms_path='/home/mathieupvc/These/data/meg_maestro/average_rms.csv',
    #                    save_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis',
    #                    data_type='meg',
    #                    signal_type='raw',
    #                    montage='mono',
    #                    remove=None,
    #                    # tmin=4,
    #                    tmin=0,
    #                    tmax=8,
    #                    # tmax=4,
    #                    min_time_interval=0.050,
    #                    best_channels_path=None,
    #                    )

    '''create_seeg_data_matrix'''
    create_seeg_data_matrix("/home/mathieu/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta",
                            '/home/mathieu/These/data/meg_maestro/seeg_behavior_data',
                            '/home/mathieu/These/data/meg_maestro/exp_stim_seeg',
                            # '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior',
                            '/home/mathieu/seeg_behavior_test',
                            signal_type='raw', montage='mono', remove=None, tmin=0, tmax=8,
                            min_time_interval=0.050, only_repeated=False)

    '''correlation_matrix'''
    # correlation_matrix(data_matrix_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis/behavior_data_matrix.csv')

    '''plot_histograms'''
    # feature_list = ['surprise_mean_paulrnn',
    #                 'surprise_posres_paulrnn', 'surprise_possum_paulrnn', 'surprise_negsum_paulrnn',
    #                 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
    # plot_histograms(data_matrix_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis/behavior_data_matrix.csv', feature_list=feature_list)

    '''lme'''
    # independent_vars = ['pitch', 'time_interval',
    #                     'surprise_mean_paulrnn',
    #                     'entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'loudness', 'brightness', 'periodicity',
    #                     'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv', 'periodicity_deriv',
    #                     'roughness_deriv', 'spectralflux_deriv', 'onset_type', 'pitch_interval', 'block',
    #                     'position_in_block', 'gmsi', '2.5Hz', '3.5Hz', '5.5Hz', '6.5Hz', 'delta_1s',
    #                     'theta_1s', 'alpha_1s', 'beta_1s', 'high_gamma_1s', 'delta_7s', 'theta_7s', 'alpha_7s',
    #                     'beta_7s', 'high_gamma_7s', 'rms_1s', 'rms_7s']
    # lme(data_matrix_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis/behavior_data_matrix.csv',
    #     independent_vars=independent_vars,
    #     interactions=['surprise_mean_paulrnn*entropy_pitch_paulrnn', 'surprise_mean_paulrnn*entropy_timing_paulrnn',
    #                   'entropy_pitch_paulrnn*entropy_timing_paulrnn',
    #                   'entropy_pitch_paulrnn*onset_type',
    #                   'entropy_timing_paulrnn*onset_type', 'surprise_mean_paulrnn*onset_type', 'surprise_mean_paulrnn*gmsi',
    #                   'entropy_pitch_paulrnn*gmsi', 'entropy_timing_paulrnn*gmsi',
    #                   'surprise_mean_paulrnn*2.5Hz', 'surprise_mean_paulrnn*3.5Hz', 'surprise_mean_paulrnn*5.5Hz', 'surprise_mean_paulrnn*6.5Hz',
    #                   'entropy_pitch_paulrnn*2.5Hz', 'entropy_pitch_paulrnn*3.5Hz', 'entropy_pitch_paulrnn*5.5Hz', 'entropy_pitch_paulrnn*6.5Hz',
    #                   'entropy_timing_paulrnn*2.5Hz', 'entropy_timing_paulrnn*3.5Hz', 'entropy_timing_paulrnn*5.5Hz', 'entropy_timing_paulrnn*6.5Hz',
    #                   'theta_1s*high_gamma_1s', 'delta_1s*high_gamma_1s', 'theta_1s*beta_1s', 'delta_1s*beta_1s',
    #                   'theta_7s*high_gamma_7s', 'delta_7s*high_gamma_7s', 'theta_7s*beta_7s', 'delta_7s*beta_7s',
    #                   'surprise_mean_paulrnn*beta_7s', 'entropy_pitch_paulrnn*beta_7s', 'entropy_timing_paulrnn*beta_7s'
    #                   ],
    #     dependent_var='correct',
    #     group='subject',
    #     rep='rep2'
    #     )

    # independent_vars = ['pitch',
    #                     'surprise_mean_paulrnn',
    #                     'entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'loudness', 'brightness', 'periodicity',
    #                     'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv', 'periodicity_deriv',
    #                     'roughness_deriv', 'spectralflux_deriv', 'onset_type', 'pitch_interval', 'block',
    #                     'position_in_block', 'gmsi', '2.5Hz', '3.5Hz', '5.5Hz', '6.5Hz',
    #                     'delta_7s', 'theta_7s', 'alpha_7s',
    #                     'beta_7s', 'high_gamma_7s', 'rms_7s']
    # lme(data_matrix_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis/behavior_data_matrix.csv',
    #     independent_vars=independent_vars,
    #     interactions=['surprise_mean_paulrnn*entropy_pitch_paulrnn', 'surprise_mean_paulrnn*entropy_timing_paulrnn',
    #                   'entropy_pitch_paulrnn*entropy_timing_paulrnn',
    #                   'entropy_pitch_paulrnn*onset_type',
    #                   'entropy_timing_paulrnn*onset_type', 'surprise_mean_paulrnn*onset_type', 'surprise_mean_paulrnn*gmsi',
    #                   'entropy_pitch_paulrnn*gmsi', 'entropy_timing_paulrnn*gmsi',
    #                   'surprise_mean_paulrnn*2.5Hz', 'surprise_mean_paulrnn*3.5Hz', 'surprise_mean_paulrnn*5.5Hz', 'surprise_mean_paulrnn*6.5Hz',
    #                   'entropy_pitch_paulrnn*2.5Hz', 'entropy_pitch_paulrnn*3.5Hz', 'entropy_pitch_paulrnn*5.5Hz', 'entropy_pitch_paulrnn*6.5Hz',
    #                   'entropy_timing_paulrnn*2.5Hz', 'entropy_timing_paulrnn*3.5Hz', 'entropy_timing_paulrnn*5.5Hz', 'entropy_timing_paulrnn*6.5Hz',
    #                   'theta_7s*high_gamma_7s', 'delta_7s*high_gamma_7s', 'theta_7s*beta_7s', 'delta_7s*beta_7s',
    #                   'surprise_mean_paulrnn*beta_7s', 'entropy_pitch_paulrnn*beta_7s', 'entropy_timing_paulrnn*beta_7s'
    #                   ],
    #     dependent_var='correct',
    #     group='subject',
    #     rep='rep2'
    #     )

    #### dep var: rms

    # independent_vars = ['pitch',
    #                     'surprise_mean_paulrnn',
    #                     'entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'loudness', 'brightness', 'periodicity',
    #                     'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv', 'periodicity_deriv',
    #                     'roughness_deriv', 'spectralflux_deriv', 'onset_type', 'pitch_interval', 'block',
    #                     'position_in_block', 'gmsi', '2.5Hz', '3.5Hz', '5.5Hz', '6.5Hz', 'correct']
    # lme(data_matrix_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis/behavior_data_matrix.csv',
    #     independent_vars=independent_vars,
    #     interactions=['surprise_mean_paulrnn*entropy_pitch_paulrnn', 'surprise_mean_paulrnn*entropy_timing_paulrnn',
    #                   'entropy_pitch_paulrnn*entropy_timing_paulrnn',
    #                   'entropy_pitch_paulrnn*onset_type',
    #                   'entropy_timing_paulrnn*onset_type', 'surprise_mean_paulrnn*onset_type',
    #                   'surprise_mean_paulrnn*gmsi',
    #                   'entropy_pitch_paulrnn*gmsi', 'entropy_timing_paulrnn*gmsi',
    #                   'surprise_mean_paulrnn*2.5Hz', 'surprise_mean_paulrnn*3.5Hz', 'surprise_mean_paulrnn*5.5Hz',
    #                   'surprise_mean_paulrnn*6.5Hz',
    #                   'entropy_pitch_paulrnn*2.5Hz', 'entropy_pitch_paulrnn*3.5Hz', 'entropy_pitch_paulrnn*5.5Hz',
    #                   'entropy_pitch_paulrnn*6.5Hz',
    #                   'entropy_timing_paulrnn*2.5Hz', 'entropy_timing_paulrnn*3.5Hz', 'entropy_timing_paulrnn*5.5Hz',
    #                   'entropy_timing_paulrnn*6.5Hz',
    #                   ],
    #     dependent_var='beta_7s',
    #     group='subject',
    #     rep='rep2'
    #     )

    # independent_vars = ['pitch', 'time_interval',
    #                     'surprise_mean_paulrnn',
    #                     'entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'loudness', 'brightness', 'periodicity',
    #                     'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv', 'periodicity_deriv',
    #                     'roughness_deriv', 'spectralflux_deriv', 'onset_type', 'pitch_interval', 'block',
    #                     'position_in_block', 'gmsi', 'correct']
    # lme(data_matrix_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis/behavior_data_matrix.csv',
    #     independent_vars=independent_vars,
    #     interactions=['surprise_mean_paulrnn*entropy_pitch_paulrnn', 'surprise_mean_paulrnn*entropy_timing_paulrnn',
    #                   'entropy_pitch_paulrnn*entropy_timing_paulrnn',
    #                   'entropy_pitch_paulrnn*onset_type',
    #                   'entropy_timing_paulrnn*onset_type', 'surprise_mean_paulrnn*onset_type',
    #                   'surprise_mean_paulrnn*gmsi',
    #                   'entropy_pitch_paulrnn*gmsi', 'entropy_timing_paulrnn*gmsi',
    #                   'surprise_mean_paulrnn*time_interval', 'entropy_pitch_paulrnn*time_interval', 'entropy_timing_paulrnn*time_interval',
    #                   ],
    #     dependent_var='beta_7s',
    #     group='subject',
    #     rep='rep2'
    #     )

    # independent_vars = ['rep2', 'rep3']
    # lme(data_matrix_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis/behavior_data_matrix.csv',
    #     independent_vars=independent_vars,
    #     interactions=[
    #                   ],
    #     dependent_var='beta_7s',
    #     group='subject',
    #     rep=None
    #     )

    #### dep var: heart pulses

    # independent_vars = ['pitch',
    #                     'surprise_mean_paulrnn',
    #                     'entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'loudness', 'brightness', 'periodicity',
    #                     'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv', 'periodicity_deriv',
    #                     'roughness_deriv', 'spectralflux_deriv', 'onset_type', 'pitch_interval', 'block',
    #                     'position_in_block', 'gmsi', 'rep2', 'rep3', '2.5Hz', '3.5Hz', '5.5Hz', '6.5Hz']
    # lme(data_matrix_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis/behavior_data_matrix.csv',
    #     independent_vars=independent_vars,
    #     interactions=['surprise_mean_paulrnn*entropy_pitch_paulrnn', 'surprise_mean_paulrnn*entropy_timing_paulrnn',
    #                                     'entropy_pitch_paulrnn*entropy_timing_paulrnn',
    #                                     'entropy_pitch_paulrnn*onset_type',
    #                                     'entropy_timing_paulrnn*onset_type', 'surprise_mean_paulrnn*onset_type', 'surprise_mean_paulrnn*gmsi',
    #                                     'entropy_pitch_paulrnn*gmsi', 'entropy_timing_paulrnn*gmsi',
    #                                     'surprise_mean_paulrnn*2.5Hz', 'surprise_mean_paulrnn*3.5Hz', 'surprise_mean_paulrnn*5.5Hz', 'surprise_mean_paulrnn*6.5Hz',
    #                                     'entropy_pitch_paulrnn*2.5Hz', 'entropy_pitch_paulrnn*3.5Hz', 'entropy_pitch_paulrnn*5.5Hz', 'entropy_pitch_paulrnn*6.5Hz',
    #                                     'entropy_timing_paulrnn*2.5Hz', 'entropy_timing_paulrnn*3.5Hz', 'entropy_timing_paulrnn*5.5Hz', 'entropy_timing_paulrnn*6.5Hz',
    #                                     'rep2*2.5Hz', 'rep2*3.5Hz', 'rep2*5.5Hz', 'rep2*6.5Hz',
    #                                     'rep3*2.5Hz', 'rep3*3.5Hz', 'rep3*5.5Hz', 'rep3*6.5Hz',
    #                                     'surprise_mean_paulrnn*rep2', 'surprise_mean_paulrnn*rep3',
    #                                     'entropy_pitch_paulrnn*rep2', 'entropy_pitch_paulrnn*rep3',
    #                                     'entropy_timing_paulrnn*rep2', 'entropy_timing_paulrnn*rep3',
    #                                     ],
    #     dependent_var='heart_pulses',
    #     group='subject',
    #     rep=None
    #     )

    # independent_vars = ['2.5Hz', '3.5Hz', '5.5Hz', '6.5Hz']
    # lme(data_matrix_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis/behavior_data_matrix.csv',
    #     independent_vars=independent_vars,
    #     # interactions=['surprise_mean_paulrnn*surprise_posres_paulrnn', 'surprise_mean_paulrnn*surprise_possum_paulrnn', 'surprise_mean_paulrnn*surprise_negsum_paulrnn',
    #     #               'surprise_mean_paulrnn*entropy_pitch_paulrnn', 'surprise_mean_paulrnn*entropy_timing_paulrnn', 'surprise_posres_paulrnn*surprise_possum_paulrnn',
    #     #               'surprise_posres_paulrnn*surprise_negsum_paulrnn', 'surprise_posres_paulrnn*entropy_pitch_paulrnn', 'surprise_posres_paulrnn*entropy_timing_paulrnn',
    #     #               'surprise_possum_paulrnn*surprise_negsum_paulrnn', 'surprise_possum_paulrnn*entropy_pitch_paulrnn', 'surprise_possum_paulrnn*entropy_timing_paulrnn',
    #     #               'surprise_negsum_paulrnn*entropy_pitch_paulrnn', 'surprise_negsum_paulrnn*entropy_timing_paulrnn', 'entropy_pitch_paulrnn*entropy_timing_paulrnn',
    #     #               'surprise_mean_paulrnn*onset_type', 'surprise_posres_paulrnn*onset_type', 'surprise_possum_paulrnn*onset_type',
    #     #               'surprise_negsum_paulrnn*onset_type', 'entropy_pitch_paulrnn*onset_type', 'entropy_timing_paulrnn*onset_type',
    #     #               'surprise_mean_paulrnn*gmsi', 'surprise_posres_paulrnn*gmsi', 'surprise_possum_paulrnn*gmsi',
    #     #               'surprise_negsum_paulrnn*gmsi', 'entropy_pitch_paulrnn*gmsi', 'entropy_timing_paulrnn*gmsi'
    #     #               ],
    #     # interactions=['surprise_mean_paulrnn*entropy_pitch_paulrnn', 'surprise_mean_paulrnn*entropy_timing_paulrnn',
    #     #               'entropy_pitch_paulrnn*entropy_timing_paulrnn',
    #     #               'entropy_pitch_paulrnn*onset_type',
    #     #               'entropy_timing_paulrnn*onset_type', 'surprise_mean_paulrnn*onset_type',
    #     #               'surprise_mean_paulrnn*gmsi',
    #     #               'entropy_pitch_paulrnn*gmsi', 'entropy_timing_paulrnn*gmsi',
    #     #               'rep2*2.5Hz', 'rep2*3.5Hz', 'rep2*5.5Hz', 'rep2*6.5Hz',
    #     #               'rep3*2.5Hz', 'rep3*3.5Hz', 'rep3*5.5Hz', 'rep3*6.5Hz',
    #     #               'surprise_mean_paulrnn*rep2', 'surprise_mean_paulrnn*rep3',
    #     #               'entropy_pitch_paulrnn*rep2', 'entropy_pitch_paulrnn*rep3',
    #     #               'entropy_timing_paulrnn*rep2', 'entropy_timing_paulrnn*rep3',
    #     #               'surprise_mean_paulrnn*2.5Hz', 'surprise_mean_paulrnn*3.5Hz', 'surprise_mean_paulrnn*5.5Hz',
    #     #               'surprise_mean_paulrnn*6.5Hz',
    #     #               'entropy_pitch_paulrnn*2.5Hz', 'entropy_pitch_paulrnn*3.5Hz', 'entropy_pitch_paulrnn*5.5Hz',
    #     #               'entropy_pitch_paulrnn*6.5Hz',
    #     #               'entropy_timing_paulrnn*2.5Hz', 'entropy_timing_paulrnn*3.5Hz', 'entropy_timing_paulrnn*5.5Hz',
    #     #               'entropy_timing_paulrnn*6.5Hz',
    #     #               'theta_1s*high_gamma_1s', 'delta_1s*high_gamma_1s', 'theta_1s*beta_1s', 'delta_1s*beta_1s',
    #     #               'theta_7s*high_gamma_7s', 'delta_7s*high_gamma_7s', 'theta_7s*beta_7s', 'delta_7s*beta_7s',
    #     #               'surprise_mean_paulrnn*beta_7s', 'entropy_pitch_paulrnn*beta_7s', 'entropy_timing_paulrnn*beta_7s'
    #     #               ],
    #     interactions=[],
    #     dependent_var='hrv',
    #     group='subject',
    #     rep='rep2'
    #     )

    # independent_vars = [
    #     'surprise_mean_paulrnn',
    #     'entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'onset_type', 'block',
    #     'position_in_block', '2.5Hz', '3.5Hz', '5.5Hz', '6.5Hz']
    # lme(data_matrix_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis/behavior_data_matrix.csv',
    #     independent_vars=independent_vars,
    #     interactions=['surprise_mean_paulrnn*entropy_pitch_paulrnn', 'surprise_mean_paulrnn*entropy_timing_paulrnn',
    #                   'entropy_pitch_paulrnn*entropy_timing_paulrnn',
    #                   ],
    #     dependent_var='heart_pulses',
    #     group='subject',
    #     rep='rep2'
    #     )

    '''barplot_on_behavior_matrix'''
    # barplot_on_behavior_matrix(data_matrix_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis/behavior_data_matrix_all.csv')

    '''trial_analysis'''
    # trial_analysis(data_matrix_path='/home/mathieu/These/data/meg_maestro/behavior_analysis/behavior_data_matrix_all.csv', save_path='/home/mathieu/These/data/meg_maestro/behavior_analysis')

    '''trial_analysis_all_rep1'''
    # trial_analysis_all_rep1(data_matrix_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis/behavior_data_matrix_all.csv', save_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis')

    '''bias_evolution'''
    # bias_evolution(stim_behavior_path='/home/mathieu/These/data/meg_maestro/behavior_analysis/stim_behavior_matrix.csv')

    '''bias_evolution_rep1'''
    # bias_evolution_rep1(stim_behavior_rep1_path='/home/mathieu/These/data/meg_maestro/behavior_analysis/stim_behavior_matrix_rep1.csv', variable='resp1')

    '''bias_evolution_surprise'''
    stim_behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/stim_behavior_matrix.csv'
    # bias_evolution_surprise(stim_behavior_path)

    '''plot_glmm_coefficients'''
    csv_path_list = ['/home/mathieupvc/These/data/meg_maestro/behavior_analysis/glmm_response_coefficients.csv', '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/glmm_correct_coefficients.csv', '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/glmm_rs_coefficients.csv']
    # csv_path_list = ['/home/mathieupvc/These/data/meg_maestro/behavior_analysis/glmm_response_coefficients.csv', '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/glmm_correct_coefficients.csv']
    model_name_list = ['Response', 'Correct', 'Reaction speed']
    # plot_glmm_coefficients(csv_path_list, model_name_list)

    '''save_pred_feat_csv'''
    data_matrix_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/behavior_data_matrix_all.csv'
    save_path = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis'
    # save_pred_feat_csv(data_matrix_path, save_path)
