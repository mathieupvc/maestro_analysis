from pathlib import Path
import pickle
import os

from pandas.io.formats.style import subset
from tqdm import tqdm

import itertools

import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

import pingouin as pg
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

# from scikit_posthocs import posthoc_ttest
from statannotations.Annotator import Annotator

from utils import gmsi_array_extraction

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.formatter.limits'] = [-2, 4]

def compute_d_prime(csv, hit='old'):

    if hit == 'old':
        true_pos = len(csv.loc[(csv['rep'] > 1) & (csv['resp'] == 2)]) / len(csv.loc[csv['rep'] > 1])
        false_pos = len(csv.loc[(csv['rep'] == 1) & (csv['resp'] == 2)]) / len(csv.loc[csv['rep'] == 1])
    elif hit == 'new':
        true_pos = len(csv.loc[(csv['rep'] == 1) & (csv['resp'] == 1)]) / len(csv.loc[csv['rep'] == 1])
        false_pos = len(csv.loc[(csv['rep'] > 1) & (csv['resp'] == 1)]) / len(csv.loc[csv['rep'] > 1])
    else:
        raise ValueError('hit should be "old" or "new"')
    if true_pos==1:
        true_pos = (len(csv.loc[csv['rep'] > 1]) - 0.5) / len(csv.loc[csv['rep'] > 1])
    if false_pos==1:
        false_pos = (len(csv.loc[csv['rep'] == 1]) - 0.5) / len(csv.loc[csv['rep'] == 1])
    if true_pos==0:
        true_pos = 0.5 / len(csv.loc[csv['rep'] > 1])
    if false_pos==0:
        false_pos = 0.5 / len(csv.loc[csv['rep'] == 1])

    return scipy.stats.norm.ppf(true_pos) - scipy.stats.norm.ppf(false_pos)


def compute_d_prime_seeg(behavior, groupby):
    """take as input a long format behavior matrix with all subject, not yet grouped. Uses rep=2 for hits and rep=1 for false alarm. rep should be last in groupby"""
    data = behavior.groupby(groupby, as_index=False).mean()
    print(data)

    df_lists = [[] for i in range(len(groupby) - 1)]  # -1 because we remove rep
    d_prime_list = []
    condition_lists = [np.unique(data[groupby[i]]) for i in range(len(groupby) - 1)]
    conditions = list(itertools.product(*condition_lists))  # combine all conditions
    for conds in conditions:
        current = data.copy()
        for c in range(len(conds)):
            current = current[current[groupby[c]] == conds[c]]
            df_lists[c].append(conds[c])
        hit = current[current['rep']==2]['correct'].iloc[0]
        false_alarm = 1 - current[current['rep']==1]['correct'].iloc[0]

        if hit == 1:  # TODO improve with true proportions, not arbitrary values
            hit = 0.95
        if false_alarm == 1:
            false_alarm = 0.95
        if hit == 0:
            hit = 0.05
        if false_alarm == 0:
            false_alarm = 0.05

        d_prime_list.append(scipy.stats.norm.ppf(hit) - scipy.stats.norm.ppf(false_alarm))

    d_prime = pd.DataFrame({"d_prime": d_prime_list})
    for c in range(len(groupby) - 1):
        d_prime[groupby[c]] = df_lists[c]
    print(d_prime)

    return d_prime

def compute_percent_correct(csv):

    return (len(csv.loc[(csv['rep'] > 1) & (csv['resp'] == 2)]) + len(csv.loc[(csv['rep'] == 1) & (csv['resp'] == 1)])) / len(csv)



def behavioral_matrix(data_path, subtract_overall=False):
    '''subtract_overall: if True, subtract the overall values (d', rs) from all conditions at the subject level to remove cross-subject variability'''

    columns = ['d_overall', 'rs_overall', 'percent_correct_overall', 'resp_overall', 'd_2.5Hz', 'rs_2.5Hz', 'percent_correct_2.5Hz', 'resp_2.5Hz', 'd_3.5Hz', 'rs_3.5Hz', 'percent_correct_3.5Hz', 'resp_3.5Hz', 'd_4.5Hz', 'rs_4.5Hz', 'percent_correct_4.5Hz', 'resp_4.5Hz',
               'd_5.5Hz', 'rs_5.5Hz', 'percent_correct_5.5Hz', 'resp_5.5Hz', 'd_6.5Hz', 'rs_6.5Hz', 'percent_correct_6.5Hz', 'resp_6.5Hz', 'd_sharp', 'rs_sharp', 'percent_correct_sharp', 'resp_sharp', 'd_smooth', 'rs_smooth', 'percent_correct_smooth', 'resp_smooth',
               'd_1st_rep', 'rs_1st_rep', 'percent_correct_1st_rep', 'resp_1st_rep', 'd_2nd_rep', 'rs_2nd_rep', 'percent_correct_2nd_rep', 'resp_2nd_rep', 'd_3rd_rep', 'rs_3rd_rep', 'percent_correct_3rd_rep', 'resp_3rd_rep', 'eff_overall', 'eff_2.5Hz',
               'eff_3.5Hz', 'eff_4.5Hz', 'eff_5.5Hz', 'eff_6.5Hz', 'eff_sharp', 'eff_smooth', 'eff_2nd_rep', 'eff_3rd_rep']

    matrix = pd.DataFrame(columns=columns)

    path = Path(data_path)
    subject_folders = sorted(x for x in path.iterdir() if x.is_dir())
    del subject_folders[5]  # subject 5 does not have behavioral data

    # remove bad subjects (-1 in index to account removed subject 5)
    del subject_folders[21]
    del subject_folders[19]
    del subject_folders[13]

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        path_to_csv = subject_path / ('P_' + subject_path.name.split('_')[-1] + '/P_' + subject_path.name.split('_')[-1] + '.csv')
        csv = pd.read_csv(path_to_csv)

        # csv = csv[10:]  # remove first block (in which the subject does not have any choice)
        csv = csv[csv['block'] > 2]  # remove two first block (in which there is unbalance compared to the rest of the experiment)

        matrix.loc[subject_path.name, 'd_overall'] = compute_d_prime(csv)
        matrix.loc[subject_path.name, 'rs_overall'] = (1/csv['rt']).median()
        matrix.loc[subject_path.name, 'eff_overall'] = matrix.loc[subject_path.name, 'd_overall'] * matrix.loc[subject_path.name, 'rs_overall']
        matrix.loc[subject_path.name, 'percent_correct_overall'] = compute_percent_correct(csv)
        matrix.loc[subject_path.name, 'resp_overall'] = csv['resp'].mean() - 1

        matrix.loc[subject_path.name, 'd_2.5Hz'] = compute_d_prime(csv[csv['freq'] == 2.5])
        matrix.loc[subject_path.name, 'percent_correct_2.5Hz'] = compute_percent_correct(csv[csv['freq'] == 2.5])
        matrix.loc[subject_path.name, 'rs_2.5Hz'] = (1/csv[csv['freq'] == 2.5]['rt']).median()
        matrix.loc[subject_path.name, 'eff_2.5Hz'] = matrix.loc[subject_path.name, 'd_2.5Hz'] * matrix.loc[
            subject_path.name, 'rs_2.5Hz']
        matrix.loc[subject_path.name, 'resp_2.5Hz'] = csv[csv['freq'] == 2.5]['resp'].mean() - 1
        matrix.loc[subject_path.name, 'd_3.5Hz'] = compute_d_prime(csv[csv['freq'] == 3.5])
        matrix.loc[subject_path.name, 'percent_correct_3.5Hz'] = compute_percent_correct(csv[csv['freq'] == 3.5])
        matrix.loc[subject_path.name, 'rs_3.5Hz'] = (1/csv[csv['freq'] == 3.5]['rt']).median()
        matrix.loc[subject_path.name, 'eff_3.5Hz'] = matrix.loc[subject_path.name, 'd_3.5Hz'] * matrix.loc[
            subject_path.name, 'rs_3.5Hz']
        matrix.loc[subject_path.name, 'resp_3.5Hz'] = csv[csv['freq'] == 3.5]['resp'].mean() - 1
        matrix.loc[subject_path.name, 'd_4.5Hz'] = compute_d_prime(csv[csv['freq'] == 4.5])
        matrix.loc[subject_path.name, 'percent_correct_4.5Hz'] = compute_percent_correct(csv[csv['freq'] == 4.5])
        matrix.loc[subject_path.name, 'rs_4.5Hz'] = (1/csv[csv['freq'] == 4.5]['rt']).median()
        matrix.loc[subject_path.name, 'eff_4.5Hz'] = matrix.loc[subject_path.name, 'd_4.5Hz'] * matrix.loc[
            subject_path.name, 'rs_4.5Hz']
        matrix.loc[subject_path.name, 'resp_4.5Hz'] = csv[csv['freq'] == 4.5]['resp'].mean() - 1
        matrix.loc[subject_path.name, 'd_5.5Hz'] = compute_d_prime(csv[csv['freq'] == 5.5])
        matrix.loc[subject_path.name, 'percent_correct_5.5Hz'] = compute_percent_correct(csv[csv['freq'] == 5.5])
        matrix.loc[subject_path.name, 'rs_5.5Hz'] = (1/csv[csv['freq'] == 5.5]['rt']).median()
        matrix.loc[subject_path.name, 'eff_5.5Hz'] = matrix.loc[subject_path.name, 'd_5.5Hz'] * matrix.loc[
            subject_path.name, 'rs_5.5Hz']
        matrix.loc[subject_path.name, 'resp_5.5Hz'] = csv[csv['freq'] == 5.5]['resp'].mean() - 1
        matrix.loc[subject_path.name, 'd_6.5Hz'] = compute_d_prime(csv[csv['freq'] == 6.5])
        matrix.loc[subject_path.name, 'percent_correct_6.5Hz'] = compute_percent_correct(csv[csv['freq'] == 6.5])
        matrix.loc[subject_path.name, 'rs_6.5Hz'] = (1/csv[csv['freq'] == 6.5]['rt']).median()
        matrix.loc[subject_path.name, 'eff_6.5Hz'] = matrix.loc[subject_path.name, 'd_6.5Hz'] * matrix.loc[
            subject_path.name, 'rs_6.5Hz']
        matrix.loc[subject_path.name, 'resp_6.5Hz'] = csv[csv['freq'] == 6.5]['resp'].mean() - 1

        matrix.loc[subject_path.name, 'd_sharp'] = compute_d_prime(csv[csv['onset'] == 0])
        matrix.loc[subject_path.name, 'percent_correct_sharp'] = compute_percent_correct(csv[csv['onset'] == 0])
        matrix.loc[subject_path.name, 'rs_sharp'] = (1/csv[csv['onset'] == 0]['rt']).median()
        matrix.loc[subject_path.name, 'eff_sharp'] = matrix.loc[subject_path.name, 'd_sharp'] * matrix.loc[
            subject_path.name, 'rs_sharp']
        matrix.loc[subject_path.name, 'resp_sharp'] = csv[csv['onset'] == 0]['resp'].mean() - 1
        matrix.loc[subject_path.name, 'd_smooth'] = compute_d_prime(csv[csv['onset'] == 2])
        matrix.loc[subject_path.name, 'percent_correct_smooth'] = compute_percent_correct(csv[csv['onset'] == 2])
        matrix.loc[subject_path.name, 'rs_smooth'] = (1/csv[csv['onset'] == 2]['rt']).median()
        matrix.loc[subject_path.name, 'eff_smooth'] = matrix.loc[subject_path.name, 'd_smooth'] * matrix.loc[
            subject_path.name, 'rs_smooth']
        matrix.loc[subject_path.name, 'resp_smooth'] = csv[csv['onset'] == 2]['resp'].mean() - 1

        matrix.loc[subject_path.name, 'd_1st_rep'] = compute_d_prime(csv, hit='new')
        matrix.loc[subject_path.name, 'rs_1st_rep'] = (1/csv[csv['rep'] == 1]['rt']).median()
        matrix.loc[subject_path.name, 'percent_correct_1st_rep'] = compute_percent_correct(csv[csv['rep'] == 1])
        matrix.loc[subject_path.name, 'resp_1st_rep'] = csv[csv['rep'] == 1]['resp'].mean() - 1
        matrix.loc[subject_path.name, 'd_2nd_rep'] = compute_d_prime(csv.loc[(csv['rep'] == 1) | (csv['rep'] == 2)])
        matrix.loc[subject_path.name, 'rs_2nd_rep'] = (1/csv[csv['rep'] == 2]['rt']).median()
        matrix.loc[subject_path.name, 'eff_2nd_rep'] = matrix.loc[subject_path.name, 'd_2nd_rep'] * matrix.loc[subject_path.name, 'rs_2nd_rep']
        matrix.loc[subject_path.name, 'percent_correct_2nd_rep'] = compute_percent_correct(csv[csv['rep'] == 2])
        matrix.loc[subject_path.name, 'resp_2nd_rep'] = csv[csv['rep'] == 2]['resp'].mean() - 1
        matrix.loc[subject_path.name, 'd_3rd_rep'] = compute_d_prime(csv.loc[(csv['rep'] == 1) | (csv['rep'] == 3)])
        matrix.loc[subject_path.name, 'rs_3rd_rep'] = (1/csv[csv['rep'] == 3]['rt']).median()
        matrix.loc[subject_path.name, 'eff_3rd_rep'] = matrix.loc[subject_path.name, 'd_3rd_rep'] * matrix.loc[subject_path.name, 'rs_3rd_rep']
        matrix.loc[subject_path.name, 'percent_correct_3rd_rep'] = compute_percent_correct(csv[csv['rep'] == 3])
        matrix.loc[subject_path.name, 'resp_3rd_rep'] = csv[csv['rep'] == 3]['resp'].mean() - 1

        if subtract_overall:
            for column in columns:
                if (column[0] == "d") & (column != 'd_overall'):
                    matrix.loc[subject_path.name, column] -= matrix.loc[subject_path.name, 'd_overall']
                if (column[:2] == "rs") & (column != 'rs_overall'):
                    matrix.loc[subject_path.name, column] -= matrix.loc[subject_path.name, 'rs_overall']
                if (column[0] == "e") & (column != 'eff_overall'):
                    matrix.loc[subject_path.name, column] -= matrix.loc[subject_path.name, 'eff_overall']
                if (column[0] == "p") & (column != 'percent_correct_overall'):
                    matrix.loc[subject_path.name, column] -= matrix.loc[subject_path.name, 'percent_correct_overall']
                if (column[:2] == "re") & (column != 'resp_overall'):
                    matrix.loc[subject_path.name, column] -= matrix.loc[subject_path.name, 'resp_overall']
            matrix.drop(columns=['d_overall', 'rs_overall', 'eff_overall', 'percent_correct_overall', 'resp_overall'], inplace=True)

    return matrix


def seeg_behavior_matrix(data_path, stim_path):
    path = Path(data_path)
    subject_folders = sorted(x for x in path.iterdir() if x.is_file())

    stim_pathA = Path(os.path.join(stim_path, 'block1'))
    stim_pathB = Path(os.path.join(stim_path, 'block2'))
    stimA = sorted(x for x in stim_pathA.iterdir() if x.is_file())
    stimA = [stim.name for stim in stimA]
    stimB = sorted(x for x in stim_pathB.iterdir() if x.is_file())
    stimB = [stim.name for stim in stimB]

    subject_list = []
    stim_name_list = []
    response_list = []
    block = []
    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        csv = pd.read_csv(subject_path, header=2, sep='\s+')
        csv = csv[csv['Event'] == 'Response']
        if csv['Subject'].iloc[0] == 'BOU_AL':
            csv['Subject'] = 'Bou_Al'
        if csv['Subject'].iloc[0] == 'BouNi':
            csv['Subject'] = 'Bou_Ni'
        if csv['Subject'].iloc[0] == 'THE_MO':
            csv['Subject'] = 'The_Mo'
        if csv['Subject'].iloc[0] == 'TouEt':
            csv['Subject'] = 'Tou_Et'
        if csv['Subject'].iloc[0] == 'TOU_M3':
            csv['Subject'] = 'Tou_Ma'
        if csv['Subject'].iloc[0] == 2:
            csv['Subject'] = 'Dav_To'
        csv['Trial'] = csv['Trial'].astype(int)  # some subjects have str trial numbers
        csv['Type'] = csv['Type'].astype(int)
        csv['Type'] = csv['Type'].replace(91, 1)  # some weird response coding
        csv['Type'] = csv['Type'].replace(92, 2)
        for trial in range(1, 91):  # check for duplicates
            resp = csv[csv['Trial']==trial]['Type'].tolist()
            if len(resp) == 1:
                response_list.append(resp[0])
            elif all(j == resp[0] for j in resp):
                response_list.append(resp[0])
            else:
                # response_list.append('different_responses')
                response_list.append(np.nan)  # for several button press recorded and different responses for a given trial
        if 'blockA' in str(subject_path):
            stim_name_list = stim_name_list + stimA
            subject_list = subject_list + [csv['Subject'].iloc[0]] * len(stimA)
            block = block + [1] * len(stimA)
        else:
            stim_name_list = stim_name_list + stimB
            subject_list = subject_list + [csv['Subject'].iloc[0]] * len(stimB)
            block = block + [2] * len(stimB)

    behavior = pd.DataFrame({'subject': subject_list, 'stim_name': stim_name_list, 'resp': response_list, 'seeg_block': block})
    behavior['trial'] = 0
    behavior['rep'] = 0
    behavior['rate'] = 0
    behavior['onset_type'] = 0
    for i in range(len(behavior)):
        behavior['trial'].iloc[i] = int(behavior['stim_name'].iloc[i].split('_')[0][5:8])
        behavior['rep'].iloc[i] = int(behavior['stim_name'].iloc[i].split('_')[1][3])
        behavior['rate'].iloc[i] = float(behavior['stim_name'].iloc[i].split('_')[2][0:3])
        behavior['onset_type'].iloc[i] = int(behavior['stim_name'].iloc[i].split('_')[4][5])
    behavior['correct'] = (behavior['resp'] == behavior['rep']).astype(int)
    behavior.loc[behavior['resp'].isnull(), 'correct'] = np.nan

    return behavior


def rate_repeat_interaction_matrix(data_path, subtract_overall=False, plot=True):

    columns = ['d_2.5Hz_rep_2', 'd_3.5Hz_rep_2', 'd_4.5Hz_rep_2', 'd_5.5Hz_rep_2', 'd_6.5Hz_rep_2',
               'd_2.5Hz_rep_3', 'd_3.5Hz_rep_3', 'd_4.5Hz_rep_3', 'd_5.5Hz_rep_3', 'd_6.5Hz_rep_3',
               'rs_2.5Hz_rep_1', 'rs_3.5Hz_rep_1', 'rs_4.5Hz_rep_1', 'rs_5.5Hz_rep_1', 'rs_6.5Hz_rep_1',
               'rs_2.5Hz_rep_2', 'rs_3.5Hz_rep_2', 'rs_4.5Hz_rep_2', 'rs_5.5Hz_rep_2', 'rs_6.5Hz_rep_2',
               'rs_2.5Hz_rep_3', 'rs_3.5Hz_rep_3', 'rs_4.5Hz_rep_3', 'rs_5.5Hz_rep_3', 'rs_6.5Hz_rep_3'
               'percent_correct_2.5Hz_rep_1', 'percent_correct_3.5Hz_rep_1', 'percent_correct_4.5Hz_rep_1', 'percent_correct_5.5Hz_rep_1', 'percent_correct_6.5Hz_rep_1',
               'percent_correct_2.5Hz_rep_2', 'percent_correct_3.5Hz_rep_2', 'percent_correct_4.5Hz_rep_2', 'percent_correct_5.5Hz_rep_2', 'percent_correct_6.5Hz_rep_2',
               'percent_correct_2.5Hz_rep_3', 'percent_correct_3.5Hz_rep_3', 'percent_correct_4.5Hz_rep_3', 'percent_correct_5.5Hz_rep_3', 'percent_correct_6.5Hz_rep_3',
               'resp_2.5Hz_rep_1', 'resp_3.5Hz_rep_1', 'resp_4.5Hz_rep_1', 'resp_5.5Hz_rep_1', 'resp_6.5Hz_rep_1',
               'resp_2.5Hz_rep_2', 'resp_3.5Hz_rep_2', 'resp_4.5Hz_rep_2', 'resp_5.5Hz_rep_2', 'resp_6.5Hz_rep_2',
               'resp_2.5Hz_rep_3', 'resp_3.5Hz_rep_3', 'resp_4.5Hz_rep_3', 'resp_5.5Hz_rep_3', 'resp_6.5Hz_rep_3']
    rates = [2.5, 3.5, 4.5, 5.5, 6.5]

    matrix = pd.DataFrame(columns=columns)

    path = Path(data_path)
    subject_folders = sorted(x for x in path.iterdir() if x.is_dir())
    del subject_folders[5]   # subject 5 does not have behavioral data

    # remove bad subjects (-1 in index to account removed subject 5)
    del subject_folders[21]
    del subject_folders[19]
    del subject_folders[13]

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        path_to_csv = subject_path / ('P_' + subject_path.name.split('_')[-1] + '/P_' + subject_path.name.split('_')[-1] + '.csv')
        csv = pd.read_csv(path_to_csv)

        # csv = csv[10:]  # remove first block (in which the subject does not have any choice)
        csv = csv[csv['block'] > 2]  # remove two first block (in which there is unbalance compared to the rest of the experiment)

        d_overall = compute_d_prime(csv)
        percent_correct_overall = compute_percent_correct(csv)
        rs_overall = (1 / csv['rt']).median()
        resp_overall = csv['resp'].mean() - 1

        for rep in range(2, 4):
            for rate in rates:
                matrix.loc[subject_path.name, 'd_'+str(rate)+'Hz_rep_'+str(rep)] = compute_d_prime(
                    csv[(csv['freq'] == rate) & ((csv['rep'] == 1) | (csv['rep'] == rep))]
                )
                if subtract_overall:
                    matrix.loc[subject_path.name, 'd_' + str(rate) + 'Hz_rep_' + str(rep)] -= d_overall
        for rep in range(1, 4):
            for rate in rates:
                matrix.loc[subject_path.name, 'percent_correct_'+str(rate)+'Hz_rep_'+str(rep)] = compute_percent_correct(
                    csv[(csv['freq'] == rate) & (csv['rep'] == rep)]
                )
                if subtract_overall:
                    matrix.loc[subject_path.name, 'percent_correct_' + str(rate) + 'Hz_rep_' + str(rep)] -= percent_correct_overall
        for rep in range(1, 4):
            for rate in rates:
                matrix.loc[subject_path.name, 'resp_'+str(rate)+'Hz_rep_'+str(rep)] = csv[(csv['freq'] == rate) & (csv['rep'] == rep)]['resp'].mean() - 1
                if subtract_overall:
                    matrix.loc[subject_path.name, 'resp_' + str(rate) + 'Hz_rep_' + str(rep)] -= resp_overall
        for rep in range(1, 4):
            for rate in rates:
                matrix.loc[subject_path.name, 'rs_'+str(rate)+'Hz_rep_'+str(rep)] = (1/csv[(csv['freq'] == rate) & (csv['rep'] == rep)]['rt']).median()
                if subtract_overall:
                    matrix.loc[subject_path.name, 'rs_' + str(rate) + 'Hz_rep_' + str(rep)] -= rs_overall

    if plot:
        custom_palette = sns.color_palette('Set3')[:5]
        ax = sns.boxplot(data=matrix, showmeans=True, palette=custom_palette)
        ax = sns.swarmplot(data=matrix, color=".25", size=4)
        plt.ylabel("d' score")
        plt.show()

    return matrix


def onset_repeat_interaction_matrix(data_path, subtract_overall=False, plot=True):

    columns = ['d_onset_0_rep_2', 'd_onset_2_rep_2',
               'd_onset_0_rep_3', 'd_onset_2_rep_3',
               'rs_onset_0_rep_1', 'rs_onset_2_rep_1',
               'rs_onset_0_rep_2', 'rs_onset_2_rep_2',
               'rs_onset_0_rep_3', 'rs_onset_2_rep_3',
               'percent_correct_onset_0_rep_1', 'percent_correct_onset_2_rep_1',
               'percent_correct_onset_0_rep_2', 'percent_correct_onset_2_rep_2',
               'percent_correct_onset_0_rep_3', 'percent_correct_onset_2_rep_3',
               'resp_onset_0_rep_1', 'resp_onset_2_rep_1',
               'resp_onset_0_rep_2', 'resp_onset_2_rep_2',
               'resp_onset_0_rep_3', 'resp_onset_2_rep_3',
               ]
    rates = [0, 2]  # variable name is rates but those are onset types

    matrix = pd.DataFrame(columns=columns)

    path = Path(data_path)
    subject_folders = sorted(x for x in path.iterdir() if x.is_dir())
    del subject_folders[5]   # subject 5 does not have behavioral data

    # remove bad subjects (-1 in index to account removed subject 5)
    del subject_folders[21]
    del subject_folders[19]
    del subject_folders[13]

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        path_to_csv = subject_path / ('P_' + subject_path.name.split('_')[-1] + '/P_' + subject_path.name.split('_')[-1] + '.csv')
        csv = pd.read_csv(path_to_csv)

        # csv = csv[10:]  # remove first block (in which the subject does not have any choice)
        csv = csv[csv['block'] > 2]  # remove two first block (in which there is unbalance compared to the rest of the experiment)

        d_overall = compute_d_prime(csv)
        percent_correct_overall = compute_percent_correct(csv)
        rs_overall = (1 / csv['rt']).median()
        resp_overall = csv['resp'].mean() - 1

        for rep in range(2, 4):
            for rate in rates:
                matrix.loc[subject_path.name, 'd_onset_'+str(rate)+'_rep_'+str(rep)] = compute_d_prime(
                    csv[(csv['onset'] == rate) & ((csv['rep'] == 1) | (csv['rep'] == rep))]
                )
                if subtract_overall:
                    matrix.loc[subject_path.name, 'd_onset_' + str(rate) + '_rep_' + str(rep)] -= d_overall
        for rep in range(1, 4):
            for rate in rates:
                matrix.loc[subject_path.name, 'percent_correct_onset_'+str(rate)+'_rep_'+str(rep)] = compute_percent_correct(
                    csv[(csv['onset'] == rate) & (csv['rep'] == rep)]
                )
                if subtract_overall:
                    matrix.loc[subject_path.name, 'percent_correct_onset_' + str(rate) + '_rep_' + str(rep)] -= percent_correct_overall
        for rep in range(1, 4):
            for rate in rates:
                matrix.loc[subject_path.name, 'resp_onset_'+str(rate)+'_rep_'+str(rep)] = csv[(csv['onset'] == rate) & (csv['rep'] == rep)]['resp'].mean() - 1
                if subtract_overall:
                    matrix.loc[subject_path.name, 'resp_onset_' + str(rate) + '_rep_' + str(rep)] -= resp_overall
        for rep in range(1, 4):
            for rate in rates:
                matrix.loc[subject_path.name, 'rs_onset_'+str(rate)+'_rep_'+str(rep)] = (1/csv[(csv['onset'] == rate) & (csv['rep'] == rep)]['rt']).median()
                if subtract_overall:
                    matrix.loc[subject_path.name, 'rs_onset_' + str(rate) + '_rep_' + str(rep)] -= rs_overall

    if plot:
        custom_palette = sns.color_palette('Set3')[:5]
        ax = sns.boxplot(data=matrix, showmeans=True, palette=custom_palette)
        ax = sns.swarmplot(data=matrix, color=".25", size=4)
        plt.ylabel("d' score")
        plt.show()

    return matrix


def rate_onset_interaction_matrix(data_path, subtract_overall=False):

    columns = [
               'rs_2.5Hz_onset_0', 'rs_3.5Hz_onset_0', 'rs_4.5Hz_onset_0', 'rs_5.5Hz_onset_0', 'rs_6.5Hz_onset_0',
               'rs_2.5Hz_onset_2', 'rs_3.5Hz_onset_2', 'rs_4.5Hz_onset_2', 'rs_5.5Hz_onset_2', 'rs_6.5Hz_onset_2',
               'resp_2.5Hz_onset_0', 'resp_3.5Hz_onset_0', 'resp_4.5Hz_onset_0', 'resp_5.5Hz_onset_0', 'resp_6.5Hz_onset_0',
               'resp_2.5Hz_onset_2', 'resp_3.5Hz_onset_2', 'resp_4.5Hz_onset_2', 'resp_5.5Hz_onset_2', 'resp_6.5Hz_onset_2']
    rates = [2.5, 3.5, 4.5, 5.5, 6.5]
    onsets = [0, 2]

    matrix = pd.DataFrame(columns=columns)

    path = Path(data_path)
    subject_folders = sorted(x for x in path.iterdir() if x.is_dir())
    del subject_folders[5]   # subject 5 does not have behavioral data

    # remove bad subjects (-1 in index to account removed subject 5)
    del subject_folders[21]
    del subject_folders[19]
    del subject_folders[13]

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        path_to_csv = subject_path / ('P_' + subject_path.name.split('_')[-1] + '/P_' + subject_path.name.split('_')[-1] + '.csv')
        csv = pd.read_csv(path_to_csv)

        # csv = csv[10:]  # remove first block (in which the subject does not have any choice)
        csv = csv[csv['block'] > 2]  # remove two first block (in which there is unbalance compared to the rest of the experiment)

        rs_overall = (1 / csv['rt']).median()
        resp_overall = csv['resp'].mean() - 1

        for onset in onsets:
            for rate in rates:
                matrix.loc[subject_path.name, 'resp_'+str(rate)+'Hz_onset_'+str(onset)] = csv[(csv['freq'] == rate) & (csv['onset'] == onset) & (csv['rep'] == 3)]['resp'].mean() - 1
                if subtract_overall:
                    matrix.loc[subject_path.name, 'resp_' + str(rate) + 'Hz_onset_' + str(onset)] -= resp_overall
        for onset in onsets:
            for rate in rates:
                matrix.loc[subject_path.name, 'rs_'+str(rate)+'Hz_onset_'+str(onset)] = (1/csv[(csv['freq'] == rate) & (csv['onset'] == onset) & (csv['rep'] == 3)]['rt']).median()
                if subtract_overall:
                    matrix.loc[subject_path.name, 'rs_' + str(rate) + 'Hz_onset_' + str(onset)] -= rs_overall

    return matrix


def plot_behaviors(matrix, d_list=None, rs_list=None, eff_list=None, resp_list=None, percent_correct_list=None, gmsi_path='/home/mathieupvc/These/data/meg_maestro/gmsi_processed.csv', anova_variable='rate', d=True, percent=True, rs=True, eff=True, resp=True, posthocs=True):

    gmsi_array = gmsi_array_extraction(gmsi_path=gmsi_path)
    exclude = [5, 14, 20, 22]
    gmsi_array = gmsi_array[[i for i in range(27) if i not in exclude]]

    # dot colors by gmsi score
    cmap_dots = plt.cm.get_cmap('viridis')
    gmsi_array = (gmsi_array - np.min(gmsi_array)) / (np.max(gmsi_array) - np.min(gmsi_array))  # normalize gmsi
    dot_colors = [cmap_dots(x) for x in gmsi_array]

    print(matrix)

    if d_list is None:
        d_list = [col for col in matrix.columns if col[0]=='d']
    if rs_list is None:
        rs_list = [col for col in matrix.columns if col[:2]=='rs']
    if eff_list is None:
        eff_list = [col for col in matrix.columns if col[0]=='e']
    if resp_list is None:
        resp_list = [col for col in matrix.columns if col[:2]=='re']
    if percent_correct_list is None:
        percent_correct_list = [col for col in matrix.columns if col[0]=='p']

    # d prime
    if d:
        custom_palette = sns.color_palette('Set3')
        # del custom_palette[8]  # We remove this color because we don't have d_prime for 1st_rep
        del custom_palette[7]  # Version without overall condition
        ax = sns.boxplot(data=matrix[d_list], showmeans=True, palette=custom_palette)
        ax = sns.swarmplot(data=matrix[d_list], color='k', size=4)
        # ax = sns.boxplot(data=matrix[d_list], x=anova_variable, y='percent_correct', showmeans=True, palette=sns.color_palette('Set3'), showfliers=False)
        # ax = sns.swarmplot(data=df, x=anova_variable, y='percent_correct', color='k', size=4)
        ax.set_ylabel("d' score", fontsize=17)
        ax.yaxis.set_tick_params(labelsize=13)
        ax.xaxis.set_tick_params(labelsize=15)

        # repeated measures ANOVA
        df = matrix[d_list].copy()
        df['subject_id'] = range(len(df))
        df = pd.melt(df, id_vars=['subject_id'], var_name=anova_variable, value_name="d'")
        anova = AnovaRM(data=df, depvar="d'", subject='subject_id', within=[anova_variable])
        anova = anova.fit()
        print('anova d')
        print(anova)
        ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

        mat = matrix[d_list].rename(columns={col: col[2:] for col in matrix[d_list].columns})
        mat['subject_id'] = range(len(mat))
        print(mat)
        df = pd.melt(mat, id_vars=['subject_id'], var_name=anova_variable, value_name="d")
        d_list = [cond[2:] for cond in d_list]
        pairs = []
        for var1 in range(len(d_list)):
            for var2 in range(var1 + 1, len(d_list)):
                pairs.append((d_list[var1], d_list[var2]))
        print(df)
        annotator = Annotator(ax, pairs, data=df, x=anova_variable, y='d')
        annotator.configure(test='t-test_paired', text_format='star', loc='inside', hide_non_significant=True)
        if posthocs:
            annotator.apply_and_annotate()

    # reaction speed
    if rs:
        plt.figure()
        color_list = [plt.cm.get_cmap('Set3')(x) for x in range(5)]
        ax = sns.boxplot(data=matrix[rs_list], showmeans=True, palette=color_list, showfliers=False)
        ax = sns.swarmplot(data=matrix[rs_list], color='k', size=4)
        plt.ylabel("Median Reaction Speed (s-1)")

        # repeated measures ANOVA
        df = matrix[rs_list].copy()
        df['subject_id'] = range(len(df))
        df = pd.melt(df, id_vars=['subject_id'], var_name=anova_variable, value_name="rs")
        anova = AnovaRM(data=df, depvar="rs", subject='subject_id', within=[anova_variable])
        anova = anova.fit()
        print('anova reaction speed')
        print(anova)
        ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

        mat = matrix[rs_list].rename(columns={col: col.replace('rs_', '') for col in matrix[rs_list].columns})
        mat['subject_id'] = range(len(mat))
        print(mat)
        df = pd.melt(mat, id_vars=['subject_id'], var_name=anova_variable, value_name="rs")
        rs_list = [cond[3:] for cond in rs_list]
        pairs = []
        for var1 in range(len(rs_list)):
            for var2 in range(var1 + 1, len(rs_list)):
                pairs.append((rs_list[var1], rs_list[var2]))
        annotator = Annotator(ax, pairs, data=df, x=anova_variable, y='rs')
        annotator.configure(test='t-test_paired', text_format='star', loc='inside', hide_non_significant=True)
        if posthocs:
            annotator.apply_and_annotate()
    #
    # # efficiency
    # if eff:
    #     plt.figure()
    #     ax = sns.boxplot(data=matrix[eff_list], showmeans=True, palette=sns.color_palette('Set3'), showfliers=False)
    #     ax = sns.swarmplot(data=matrix[eff_list], color=dot_colors, size=4)
    #     plt.ylabel("Efficiency (d' * rs) (individual mean removed)")
    #
    #     # repeated measures ANOVA
    #     df = matrix[eff_list].copy()
    #     df['subject_id'] = range(len(df))
    #     df = pd.melt(df, id_vars=['subject_id'], var_name=anova_variable, value_name="eff")
    #     anova = AnovaRM(data=df, depvar="eff", subject='subject_id', within=[anova_variable])
    #     anova = anova.fit()
    #     print('anova efficiency')
    #     print(anova)
    #     ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

    # resp
    if resp:
        plt.figure()
        color_list = [plt.cm.get_cmap('Set3')(x) for x in range(5)]
        ax = sns.boxplot(data=matrix[resp_list], showmeans=True, palette=color_list, showfliers=False, width=0.8, linewidth=3.5)
        ax = sns.swarmplot(data=matrix[resp_list], color='k', size=9)
        plt.ylabel("Average resp")

        # repeated measures ANOVA
        df = matrix[resp_list].copy()
        df['subject_id'] = range(len(df))
        df = pd.melt(df, id_vars=['subject_id'], var_name=anova_variable, value_name="resp")
        anova = AnovaRM(data=df, depvar="resp", subject='subject_id', within=[anova_variable])
        anova = anova.fit()
        print('anova resp')
        print(anova)
        ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

        mat = matrix[resp_list].rename(columns={col: col.replace('resp_', '') for col in matrix[resp_list].columns})
        mat['subject_id'] = range(len(mat))
        print(mat)
        df = pd.melt(mat, id_vars=['subject_id'], var_name=anova_variable, value_name="resp")
        resp_list = [cond[5:] for cond in resp_list]
        pairs = []
        for var1 in range(len(resp_list)):
            for var2 in range(var1 + 1, len(resp_list)):
                pairs.append((resp_list[var1], resp_list[var2]))
        annotator = Annotator(ax, pairs, data=df, x=anova_variable, y='resp')
        annotator.configure(test='t-test_paired', text_format='star', loc='inside', hide_non_significant=True)
        if posthocs:
            annotator.apply_and_annotate()

    # percent correct
    if percent:
        plt.figure()
        mat = matrix[percent_correct_list].rename(columns={col: col.replace('percent_correct_', '') for col in matrix[percent_correct_list].columns})
        df = mat
        df['subject_id'] = range(len(df))
        df['gmsi'] = gmsi_array
        print(mat)
        df = pd.melt(mat, id_vars=['subject_id', 'gmsi'], var_name=anova_variable, value_name="percent_correct")
        color_list = [plt.cm.get_cmap('Set3')(x) for x in range(5)]
        ax = sns.boxplot(data=df, x=anova_variable, y='percent_correct', showmeans=True, palette=color_list, showfliers=False)
        ax = sns.swarmplot(data=df, x=anova_variable, y='percent_correct', color='k', size=4)
        # ax = sns.swarmplot(data=df, x=anova_variable, y='percent_correct', hue='gmsi', palette='viridis', size=6)
        plt.ylabel("Percent correct")

        percent_correct_list = [cond[16:] for cond in percent_correct_list]
        pairs = []
        for var1 in range(len(percent_correct_list)):
            for var2 in range(var1+1, len(percent_correct_list)):
                pairs.append((percent_correct_list[var1], percent_correct_list[var2]))
        annotator = Annotator(ax, pairs, data=df, x=anova_variable, y='percent_correct')
        annotator.configure(test='t-test_paired', text_format='star', loc='inside', hide_non_significant=True)
        if posthocs:
            annotator.apply_and_annotate()

        # repeated measures ANOVA
        anova = AnovaRM(data=df, depvar="percent_correct", subject='subject_id', within=[anova_variable])
        anova = anova.fit()
        print('anova percent correct')
        print(anova)
        ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

    plt.show()


    return


def plot_seeg_behavior(data_path, stim_path):

    behavior = seeg_behavior_matrix(data_path, stim_path)
    behavior = behavior[behavior['resp'] != 'different_responses']
    behavior = behavior.dropna(subset='resp')
    behavior['resp'] = behavior['resp'].astype(int)
    behavior['resp'] = behavior['resp'] - 1  # set resp to 0 and 1

    # remove Bou_Ni
    behavior = behavior[behavior['subject'] != 'Bou_Ni']

    print(behavior)
    # print(behavior.groupby(['subject', 'rep', 'rate', 'onset_type'], as_index=False).mean())

    # behavior = behavior[behavior['subject'] !='BouNi'] # remove subjects
    # behavior = behavior[behavior['trial'] > 15]  # remove 15 first trials

    data = behavior.groupby(['subject', 'rep'], as_index=False).mean()
    ax = sns.barplot(data=data, y='rep', x='resp', palette='Set3', orient='h')
    # ax = sns.swarmplot(data=data, x='rep', y='resp', hue='subject', size=6)
    ax = sns.swarmplot(data=data, y='rep', x='resp', color='black', size=4, orient='h')
    ax.set_xlim(0, 1)
    anova = AnovaRM(data=data, depvar="resp", subject='subject', within=['rep'])
    anova = anova.fit()
    print('anova rep')
    print(anova)
    # ax.scatter(data[data['rep']==1]['resp'].mean(), 0, color='w', edgecolor='k', lw=2, s=75, zorder=10)
    # ax.scatter(data[data['rep']==2]['resp'].mean(), 1, color='w', edgecolor='k', lw=2, s=75, zorder=10)
    ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

    print('T-tests')
    test = scipy.stats.ttest_1samp(data[data['rep']==1]['resp'].to_numpy(), 0.5, alternative='less')
    print('Novelty', test)
    test = scipy.stats.ttest_1samp(data[data['rep'] == 2]['resp'].to_numpy(), 0.5, alternative='greater')
    print('Recognition', test)
    test = scipy.stats.ttest_1samp(data[data['rep'] == 2]['resp'].to_numpy() - data[data['rep'] == 1]['resp'].to_numpy(), 0, alternative='two-sided')
    print('Recognition - Novelty', test)


    plt.figure()
    data = behavior[behavior['rep']==1].groupby(['subject', 'rate'], as_index=False).mean()
    ax = sns.boxplot(data=data, x='rate', y='resp', showmeans=True, palette='Set3')
    ax = sns.swarmplot(data=data, x='rate', y='resp', hue='subject', size=6)
    ax.set_title('rep 1')
    anova = AnovaRM(data=data, depvar="resp", subject='subject', within=['rate'])
    anova = anova.fit()
    print('anova rate')
    print(anova)
    ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

    plt.figure()
    data = behavior[behavior['rep']==1].groupby(['subject', 'onset_type'], as_index=False).mean()
    ax = sns.boxplot(data=data, x='onset_type', y='resp', showmeans=True, palette='Set3')
    ax = sns.swarmplot(data=data, x='onset_type', y='resp', hue='subject', size=6)
    ax.set_title('rep 1')
    anova = AnovaRM(data=data, depvar="resp", subject='subject', within=['onset_type'])
    anova = anova.fit()
    print('anova onset_type')
    print(anova)
    ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

    plt.figure()
    data = behavior[behavior['rep'] == 2].groupby(['subject', 'rate'], as_index=False).mean()
    ax = sns.boxplot(data=data, x='rate', y='resp', showmeans=True, palette='Set3')
    ax = sns.swarmplot(data=data, x='rate', y='resp', hue='subject', size=6)
    ax.set_title('rep 2')
    anova = AnovaRM(data=data, depvar="resp", subject='subject', within=['rate'])
    anova = anova.fit()
    print('anova rate')
    print(anova)
    ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

    plt.figure()
    data = behavior[behavior['rep'] == 2].groupby(['subject', 'onset_type'], as_index=False).mean()
    ax = sns.boxplot(data=data, x='onset_type', y='resp', showmeans=True, palette='Set3')
    ax = sns.swarmplot(data=data, x='onset_type', y='resp', hue='subject', size=6)
    ax.set_title('rep 2')
    anova = AnovaRM(data=data, depvar="resp", subject='subject', within=['onset_type'])
    anova = anova.fit()
    print('anova onset_type')
    print(anova)
    ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

    plt.figure()
    d_prime = compute_d_prime_seeg(behavior, groupby=['subject', 'rep'])
    ax = sns.boxplot(data=d_prime, y='d_prime', showmeans=True, palette='Set3')
    ax = sns.swarmplot(data=d_prime, y='d_prime', hue='subject', size=6)

    plt.figure()
    d_prime = compute_d_prime_seeg(behavior, groupby=['subject', 'rate', 'rep'])
    ax = sns.boxplot(data=d_prime, x='rate', y='d_prime', showmeans=True, palette='Set3')
    ax = sns.swarmplot(data=d_prime, x='rate', y='d_prime', hue='subject', size=6)
    anova = AnovaRM(data=d_prime, depvar="d_prime", subject='subject', within=['rate'])
    anova = anova.fit()
    print('anova d_prime')
    print(anova)
    ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

    plt.figure()
    d_prime = compute_d_prime_seeg(behavior, groupby=['subject', 'onset_type', 'rep'])
    ax = sns.boxplot(data=d_prime, x='onset_type', y='d_prime', showmeans=True, palette='Set3')
    ax = sns.swarmplot(data=d_prime, x='onset_type', y='d_prime', hue='subject', size=6)
    anova = AnovaRM(data=d_prime, depvar="d_prime", subject='subject', within=['onset_type'])
    anova = anova.fit()
    print('anova onset_type')
    print(anova)
    ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

    plt.show()

def regr_nb_trials(data_path, bin_nb_trials=True):

    path = Path(data_path)
    subject_folders = sorted(x for x in path.iterdir() if x.is_dir())
    del subject_folders[5]

    fig, axs = plt.subplots(5, 6, sharex=True, sharey=True)

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        path_to_csv = subject_path / (
                'P_' + subject_path.name.split('_')[-1] + '/P_' + subject_path.name.split('_')[-1] + '.csv')
        csv = pd.read_csv(path_to_csv)

        df = pd.DataFrame(columns=['true', 'last_repeat'])
        for j in range(len(csv)):
            if csv.loc[j, 'rep'] == csv.loc[j, 'resp']:
                df.loc[j, 'true'] = 1
            elif (csv.loc[j, 'rep'] == 3) & (csv.loc[j, 'resp'] == 2):
                df.loc[j, 'true'] = 1
            else:
                df.loc[j, 'true'] = 0

            if csv.loc[j, 'rep'] == 1:
                df.loc[j, 'last_repeat'] = None
            else:
                current_stim = csv.loc[j, 'stim']
                block = csv.loc[j, 'block']
                rep = csv.loc[j, 'rep']
                freq = csv.loc[j, 'freq']
                onset = csv.loc[j, 'onset']
                last_stim = csv.loc[(csv['block'] == block - 1) & (csv['rep'] == rep - 1) & (csv['freq'] == freq) & (csv['onset'] == onset)]['stim']
                assert len(last_stim) == 1

                df.loc[j, 'last_repeat'] = current_stim - last_stim.iloc[0] - 1

        X = df['last_repeat'].to_numpy()
        keep = X!=None
        X = X[keep]
        if bin_nb_trials:
            bins = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
            values = np.array([0, 3, 9, 15, 21, 27, 33, 39, 45, 51, 57])
            ind_bin = np.digitize(X, bins)
            X = values[ind_bin]
        y = df['true'].to_numpy().astype('int')
        y = y[keep]
        X = X.reshape(-1, 1)
        model = LogisticRegression()
        model.fit(X, y)
        print(model.coef_)

        # Plot
        # plt.figure(figsize=(4, 3))
        # plt.clf()
        # plt.scatter(X.ravel(), y, color="black", zorder=20)
        X_test = np.linspace(0, 60, 300)
        axs.flat[i].plot(X_test, expit(X_test * model.coef_ + model.intercept_).ravel(), color="red", linewidth=3)
        # plt.legend(['Responses', 'Logistic fit'])
        axs.flat[i].set_ylim([0, 1])

    # fig.legend(['Logistic fit'])
    # plt.tight_layout()
    fig.supxlabel("Number of trials since last repeat")
    fig.supylabel("Accuracy")
    plt.show()

    return


def all_subj_regr_nb_trials(data_path):

    path = Path(data_path)
    subject_folders = sorted(x for x in path.iterdir() if x.is_dir())
    del subject_folders[5]

    df_all = pd.DataFrame(columns=['true', 'last_repeat'])
    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        path_to_csv = subject_path / (
                    'P_' + subject_path.name.split('_')[-1] + '/P_' + subject_path.name.split('_')[-1] + '.csv')
        csv = pd.read_csv(path_to_csv)

        df = pd.DataFrame(columns=['true', 'last_repeat'])
        for j in range(len(csv)):
            if csv.loc[j, 'rep'] == csv.loc[j, 'resp']:
                df.loc[j, 'true'] = 1
            elif (csv.loc[j, 'rep'] == 3) & (csv.loc[j, 'resp'] == 2):
                df.loc[j, 'true'] = 1
            else:
                df.loc[j, 'true'] = 0

            if csv.loc[j, 'rep'] == 1:
                df.loc[j, 'last_repeat'] = None
            else:
                current_stim = csv.loc[j, 'stim']
                block = csv.loc[j, 'block']
                rep = csv.loc[j, 'rep']
                freq = csv.loc[j, 'freq']
                onset = csv.loc[j, 'onset']
                last_stim = csv.loc[(csv['block'] == block - 1) & (csv['rep'] == rep - 1) & (csv['freq'] == freq) & (csv['onset'] == onset)]['stim']
                assert len(last_stim) == 1

                df.loc[j, 'last_repeat'] = current_stim - last_stim.iloc[0] - 1

        df_all = pd.concat([df_all, df])

    X = df_all['last_repeat'].to_numpy()
    keep = X!=None
    X = X[keep]
    y = df_all['true'].to_numpy().astype('int')
    y = y[keep]
    X = X.reshape(-1, 1)
    model = LogisticRegression()
    model.fit(X, y)
    print(model.coef_)

    # Plot
    plt.figure(figsize=(4, 3))
    plt.clf()
    plt.scatter(X.ravel(), y, color="black", zorder=20)
    X_test = np.linspace(0, 60, 300)
    plt.plot(X_test, expit(X_test * model.coef_ + model.intercept_).ravel(), color="red", linewidth=3)
    plt.xlabel("Number of trials since last repeat")
    plt.ylabel("Accuracy")
    plt.legend(['Responses', 'Logistic fit'])

    plt.show()

    return


def third_vs_second_rep(matrix):

    print(matrix)
    sns.scatterplot(data=matrix, x='d_2nd_rep', y='d_3rd_rep')
    X = np.linspace(matrix['d_2nd_rep'].min(), matrix['d_2nd_rep'].max(), 300)
    plt.plot(X, X, 'k', linewidth=3)

    plt.figure()
    sns.scatterplot(data=matrix, x='rs_2nd_rep', y='rs_3rd_rep')
    X = np.linspace(matrix['rs_2nd_rep'].min(), matrix['rs_2nd_rep'].max(), 300)
    plt.plot(X, X, 'k', linewidth=3)

    plt.figure()
    sns.scatterplot(data=matrix, x='rs_1st_rep', y='rs_2nd_rep')
    X = np.linspace(matrix['rs_1st_rep'].min(), matrix['rs_1st_rep'].max(), 300)
    plt.plot(X, X, 'k', linewidth=3)

    plt.show()
    return


def permanova_rate_repeat(matrix, n_permut=1000):

    print(matrix)
    print(matrix.columns)

    new_matrix = pd.DataFrame(columns=['subject', 'd', 'rate', 'rep'])
    ii = 0
    for index, row in matrix.iterrows():
        for col in matrix.columns:
            new_matrix.loc[ii, :] = [index, row[col], col.split('_')[1], col.split('_')[3]]
            ii += 1

    print(new_matrix)

    new_matrix['d'] = new_matrix['d'].astype(float)

    # print(pg.homoscedasticity(new_matrix, dv='d', group='rate'))
    # print(pg.homoscedasticity(new_matrix, dv='d', group='rep'))

    # aov = pg.anova(data=new_matrix, dv='d', between=['rate', 'rep'])  # get F-values of rate, rep and rate*rep
    rm_aov = pg.rm_anova(data=new_matrix, dv='d', within=['rate', 'rep'], subject='subject')
    print("rm_anova:")
    print(rm_aov.loc[:, ['Source', 'F', 'p-unc']])

    # Pairwise ttests
    rm_pair_ttest = pg.pairwise_ttests(data=new_matrix, dv='d', within=['rate', 'rep'], subject='subject', parametric=True, padjust='bonf')
    print(rm_pair_ttest.columns)
    # print(rm_pair_ttest[['Contrast', 'rate', 'A', 'B', 'Paired', 'Parametric', 'T', 'p-unc', 'BF10']])
    print(rm_pair_ttest)
    rm_pair_ttest_non_param = pg.pairwise_ttests(data=new_matrix, dv='d', within=['rate', 'rep'], subject='subject', parametric=False, padjust='bonf')
    print(rm_pair_ttest_non_param.columns)
    print(rm_pair_ttest_non_param)

    print("classic_anova:")
    aov = pg.anova(data=new_matrix, dv='d', between=['rate', 'rep'])
    F = aov['F'].to_numpy()[:-1]   # get F-values of rate, rep and rate*rep
    print(aov)

    # Permutation
    rng = np.random.default_rng(seed=33)

    permuted_matrix = new_matrix.copy()
    F_permut = np.zeros((n_permut, 3))
    for perm in range(n_permut):
        # print(perm)
        permuted_matrix['rate'] = rng.permutation(new_matrix['rate'])
        permuted_matrix['rep'] = rng.permutation(new_matrix['rep'])
        F_permut[perm, :] = pg.anova(data=permuted_matrix, dv='d', between=['rate', 'rep'])['F'].to_numpy()[:-1]

    p_value = (np.sum((F_permut >= F), axis=0) + 1) / (n_permut + 1)  # +1 to account for the F-value of th actual data
    print("p_values:", p_value)

    return


def rm_anova_rate_repeat(matrix, var="d"):

    if var == "d":
        matrix = matrix[[col for col in matrix.columns if col[0] == 'd']]
    elif var == 'percent_correct':
        matrix = matrix[[col for col in matrix.columns if col[0] == 'p']]
    print(matrix)
    new_matrix = pd.DataFrame(columns=['subject', var, 'rate', 'rep'])
    ii = 0
    for index, row in matrix.iterrows():
        for col in matrix.columns:
            if var == "d":
                new_matrix.loc[ii, :] = [index, row[col], col.split('_')[1], col.split('_')[3]]
            elif var == 'percent_correct':
                new_matrix.loc[ii, :] = [index, row[col], col.split('_')[2], col.split('_')[4]]  # accounts for the additional _ in percent_correct
            ii += 1

    new_matrix[var] = new_matrix[var].astype(float)
    print(new_matrix)

    # Check for homoscedasticity and normality
    print(pg.homoscedasticity(new_matrix, dv=var, group='rate'))
    print(pg.homoscedasticity(new_matrix, dv=var, group='rep'))
    print(pg.normality(new_matrix, dv=var, group='rate'))
    print(pg.normality(new_matrix, dv=var, group='rep'))

    # QQ-plots
    groups = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', 'rep2', 'rep3']
    fig, axes = plt.subplots(nrows=3, ncols=3)
    for i, group in enumerate(groups):
        if group[:3] == 'rep':
            data = new_matrix.loc[new_matrix['rep']==group[3], var]
        else:
            data = new_matrix.loc[new_matrix['rate']==group, var]
        sm.qqplot(data, line='45', ax=axes.flat[i], fit=True)
        axes.flat[i].set_title(group + var)

    for i in range(len(groups), 9):
        axes.flat[i].axis('off')

    plt.subplots_adjust(hspace=0.5)
    plt.show()

    # repeated measures ANOVA
    anova = AnovaRM(data=new_matrix, depvar=var, subject='subject', within=['rate', 'rep'])
    anova = anova.fit()
    print(f'anova {var}')
    print(anova)

    # # rm_anova
    # rm_aov = pg.rm_anova(data=new_matrix, dv='d', within=['rate', 'rep'], subject='subject')
    # print("rm_anova:")
    # print(rm_aov)
    #
    # # Pairwise ttests
    # rm_pair_ttest = pg.pairwise_ttests(data=new_matrix, dv='d', within=['rate', 'rep'], subject='subject',
    #                                    parametric=True, padjust='bonf')
    # print('Post-hoc pairwise ttest:')
    # print(rm_pair_ttest)

    return

def correct_incorrect_lists(data_path, save_folder):
    '''lists of correct and incorrect responses for 2nd and 3rd repetitions for each participant'''

    path = Path(data_path)
    subject_folders = sorted(x for x in path.iterdir() if x.is_dir())
    del subject_folders[5]  # subject 5 does not have behavioral data

    correct_incorrect_dict = {}

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        path_to_csv = subject_path / (
                    'P_' + subject_path.name.split('_')[-1] + '/P_' + subject_path.name.split('_')[-1] + '.csv')
        csv = pd.read_csv(path_to_csv)

        correct = csv.loc[(csv['rep'] > 1) & (csv['resp'] == 2)]
        incorrect = csv.loc[(csv['rep'] > 1) & (csv['resp'] == 1)]

        correct_list = correct.index.to_list()
        incorrect_list = incorrect.index.to_list()

        correct_incorrect_dict[subject_path.name] = [correct_list, incorrect_list]

    print(correct_incorrect_dict)

    with open(os.path.join(save_folder, 'correct_incorrect_lists.pkl'), 'wb') as f:
        pickle.dump(correct_incorrect_dict, f)

    return


def correct_lists(data_path, save_folder):
    '''lists of correct (only) responses for 1st, 2nd and 3rd repetitions for each participant'''

    path = Path(data_path)
    subject_folders = sorted(x for x in path.iterdir() if x.is_dir())
    del subject_folders[5]  # subject 5 does not have behavioral data

    correct_dict = {}

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        path_to_csv = subject_path / ('P_' + subject_path.name.split('_')[-1] + '/P_' + subject_path.name.split('_')[-1] + '.csv')
        csv = pd.read_csv(path_to_csv)

        csv = csv[10:]  # remove first block (in which the subject does not have any choice)

        rep1 = csv.loc[(csv['rep'] == 1) & (csv['resp'] == 1)]
        rep2 = csv.loc[(csv['rep'] == 2) & (csv['resp'] == 2)]
        rep3 = csv.loc[(csv['rep'] == 3) & (csv['resp'] == 2)]

        rep1_list = rep1.index.to_list()
        rep2_list = rep2.index.to_list()
        rep3_list = rep3.index.to_list()

        correct_dict[subject_path.name] = [rep1_list, rep2_list, rep3_list]

    print(correct_dict)

    with open(os.path.join(save_folder, 'correct_lists.pkl'), 'wb') as f:
        pickle.dump(correct_dict, f)

    return

def correct_incorrect_rep1_lists(data_path, save_folder):
    '''lists of correct and incorrect responses for 1st repetitions for each participant'''

    path = Path(data_path)
    subject_folders = sorted(x for x in path.iterdir() if x.is_dir())
    del subject_folders[5]  # subject 5 does not have behavioral data

    correct_incorrect_dict = {}

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        path_to_csv = subject_path / ('P_' + subject_path.name.split('_')[-1] + '/P_' + subject_path.name.split('_')[-1] + '.csv')
        csv = pd.read_csv(path_to_csv)

        csv = csv[10:]  # remove first block (in which the subject does not have any choice)

        correct = csv.loc[(csv['rep'] == 1) & (csv['resp'] == 1)]
        incorrect = csv.loc[(csv['rep'] == 1) & (csv['resp'] == 2)]

        correct_list = correct.index.to_list()
        incorrect_list = incorrect.index.to_list()

        correct_incorrect_dict[subject_path.name] = [correct_list, incorrect_list]

    print(correct_incorrect_dict)

    with open(os.path.join(save_folder, 'correct_incorrect_rep1_lists.pkl'), 'wb') as f:
        pickle.dump(correct_incorrect_dict, f)

    return


def correct_incorrect_all_reps_lists(data_path, save_folder):
    '''lists of correct and incorrect responses for 1st, 2nd and 3rd repetitions for each participant'''

    path = Path(data_path)
    subject_folders = sorted(x for x in path.iterdir() if x.is_dir())
    del subject_folders[5]  # subject 5 does not have behavioral data

    correct_incorrect_dict = {}

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        path_to_csv = subject_path / ('P_' + subject_path.name.split('_')[-1] + '/P_' + subject_path.name.split('_')[-1] + '.csv')
        csv = pd.read_csv(path_to_csv)

        csv = csv[10:]  # remove first block (in which the subject does not have any choice)

        correct = csv.loc[((csv['rep'] == 1) & (csv['resp'] == 1)) | ((csv['rep'] > 1) & (csv['resp'] == 2))]
        incorrect = csv.loc[((csv['rep'] == 1) & (csv['resp'] == 2)) | ((csv['rep'] > 1) & (csv['resp'] == 1))]

        correct_list = correct.index.to_list()
        incorrect_list = incorrect.index.to_list()

        correct_incorrect_dict[subject_path.name] = [correct_list, incorrect_list]

    print(correct_incorrect_dict)

    with open(os.path.join(save_folder, 'correct_incorrect_all_reps_lists.pkl'), 'wb') as f:
        pickle.dump(correct_incorrect_dict, f)

    return


def old_new_lists(data_path, save_path):
    '''lists of "old" and "new" responses (regardless of the actual truth) for each participant'''

    path = Path(data_path)
    subject_folders = sorted(x for x in path.iterdir() if x.is_dir())
    del subject_folders[5]  # subject 5 does not have behavioral data

    new_old_dict = {}

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)

        path_to_csv = subject_path / (
                    'P_' + subject_path.name.split('_')[-1] + '/P_' + subject_path.name.split('_')[-1] + '.csv')
        csv = pd.read_csv(path_to_csv)

        csv = csv[10:]  # remove first block (in which the subject does not have any choice)
        # csv = csv.loc[(csv['rep'] == 3)]  # keep 3rd rep only
        old = csv.loc[(csv['resp'] == 2)]
        new = csv.loc[(csv['resp'] == 1)]

        old_list = old.index.to_list()
        new_list = new.index.to_list()

        new_old_dict[subject_path.name] = [new_list, old_list]

    print(new_old_dict)

    with open(os.path.join(save_path, 'new_old_lists_rep.pkl'), 'wb') as f:
        pickle.dump(new_old_dict, f)
    # with open(os.path.join(save_path, 'new_old_lists_rep3.pkl'), 'wb') as f:
    #     pickle.dump(new_old_dict, f)

    return


def scatter_behavior_gmsi(matrix, column='d_overall', gmsi_path='/home/mathieupvc/These/data/meg_maestro/gmsi_processed.csv'):

    gmsi_array = gmsi_array_extraction(gmsi_path=gmsi_path)
    exclude = [5, 14, 20, 22]
    gmsi_array = gmsi_array[[i for i in range(27) if i not in exclude]]

    value_array = matrix[column].to_numpy()

    pearson_corr = scipy.stats.pearsonr(gmsi_array, value_array.astype(float))
    # correlation
    print('Pearson correlation = ', pearson_corr)
    print('Spearman correlation = ', scipy.stats.spearmanr(gmsi_array, value_array.astype(float)))

    plt.scatter(gmsi_array, value_array)
    plt.xlabel('gmsi')
    plt.ylabel(column)
    plt.text(plt.gca().get_xlim()[0], plt.gca().get_ylim()[1], f'r = {round(pearson_corr.statistic, 4)}, p = {round(pearson_corr.pvalue, 4)}')

    plt.show()

    return


def scatter_r2_gmsi(model_regression, control_regression, sign_channels_path, matrix, column='d_overall', tmin=50):

    significant_channels = np.load(sign_channels_path)

    control_r_squared = control_regression['r2'][:, :, tmin:]
    control_r_squared = control_r_squared[:, significant_channels, :]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    model_r_squared = model_r_squared[:, significant_channels, :]
    r2_diff = model_r_squared - control_r_squared
    times = model_regression['times'][tmin:]

    # average of max time of each channel
    mean_r2_control = np.mean(np.max(control_r_squared, axis=2), axis=1)
    mean_r2_model = np.mean(np.max(model_r_squared, axis=2), axis=1)
    mean_r2_diff = np.mean(np.max(r2_diff, axis=2), axis=1)
    # mean_r2_control = np.mean(control_r_squared, axis=(1, 2))
    # mean_r2_model = np.mean(model_r_squared, axis=(1, 2))
    # mean_r2_diff = np.mean(r2_diff, axis=(1, 2))

    value_array = matrix[column].to_numpy()

    # remove subject 5
    exclude = [5, 14, 20, 22]
    mean_r2_diff = mean_r2_diff[[i for i in range(27) if i not in exclude]]

    # correlation
    print('Pearson correlation = ', scipy.stats.pearsonr(mean_r2_diff, value_array.astype(float)))
    print('Spearman correlation = ', scipy.stats.spearmanr(mean_r2_diff, value_array.astype(float)))

    plt.scatter(mean_r2_diff, value_array)
    plt.xlabel('$\Delta$R²')
    plt.ylabel(column)

    plt.show()

    return


def ushape_testing(behavioral_matrix):

    print(behavioral_matrix)

    behavioral_matrix = behavioral_matrix[['d_2.5Hz', 'd_3.5Hz', 'd_4.5Hz', 'd_5.5Hz', 'd_6.5Hz']].to_numpy()
    print(behavioral_matrix)

    # parabolic fit
    Y = np.float64(behavioral_matrix.flatten())
    print(Y.shape)
    X = np.zeros((Y.shape[0], 2))
    print(X.shape)
    X[:, 0] = np.tile([2.5, 3.5, 4.5, 5.5, 6.5], behavioral_matrix.shape[0])
    X[:, 1] = X[:, 0] ** 2
    X = sm.add_constant(X)
    print(X.dtype)
    print(Y)
    ols = sm.OLS(Y, X).fit()
    print(ols)
    print(ols.summary())

    quadratic_param_list = []
    for i in range(behavioral_matrix.shape[0]):
        Y = np.float64(behavioral_matrix[i, :])
        X = np.zeros((Y.shape[0], 2))
        X[:, 0] = [2.5, 3.5, 4.5, 5.5, 6.5]
        X[:, 1] = X[:, 0] ** 2
        X = sm.add_constant(X)
        ols = sm.OLS(Y, X).fit()
        quadratic_param_list.append(ols.params[2])

    print(quadratic_param_list)
    print('mean =', np.mean(quadratic_param_list))
    print(scipy.stats.ttest_1samp(quadratic_param_list, 0, alternative='less'))

    return


def ushape_testing_between_conditions(behavioral_matrix):

    behavioral_matrix_cond1 = behavioral_matrix[['d_2.5Hz_rep_2', 'd_3.5Hz_rep_2', 'd_4.5Hz_rep_2', 'd_5.5Hz_rep_2',
                                                 'd_6.5Hz_rep_2']].to_numpy()
    behavioral_matrix_cond2 = behavioral_matrix[['d_2.5Hz_rep_3', 'd_3.5Hz_rep_3', 'd_4.5Hz_rep_3', 'd_5.5Hz_rep_3',
                                                 'd_6.5Hz_rep_3']].to_numpy()
    print(behavioral_matrix_cond1)

    # # parabolic fit
    # Y = np.float64(behavioral_matrix.flatten())
    # print(Y.shape)
    # X = np.zeros((Y.shape[0], 2))
    # print(X.shape)
    # X[:, 0] = np.tile([2.5, 3.5, 4.5, 5.5, 6.5], behavioral_matrix.shape[0])
    # X[:, 1] = X[:, 0] ** 2
    # X = sm.add_constant(X)
    # print(X.dtype)
    # print(Y)
    # ols = sm.OLS(Y, X).fit()
    # print(ols)
    # print(ols.summary())

    quadratic_param_list_cond1 = []
    for i in range(behavioral_matrix_cond1.shape[0]):
        Y = np.float64(behavioral_matrix_cond1[i, :])
        X = np.zeros((Y.shape[0], 2))
        X[:, 0] = [2.5, 3.5, 4.5, 5.5, 6.5]
        X[:, 1] = X[:, 0] ** 2
        X = sm.add_constant(X)
        ols = sm.OLS(Y, X).fit()
        quadratic_param_list_cond1.append(ols.params[2])

    print(quadratic_param_list_cond1)
    print('mean =', np.mean(quadratic_param_list_cond1))
    print(scipy.stats.ttest_1samp(quadratic_param_list_cond1, 0, alternative='less'))

    quadratic_param_list_cond2 = []
    for i in range(behavioral_matrix_cond2.shape[0]):
        Y = np.float64(behavioral_matrix_cond2[i, :])
        X = np.zeros((Y.shape[0], 2))
        X[:, 0] = [2.5, 3.5, 4.5, 5.5, 6.5]
        X[:, 1] = X[:, 0] ** 2
        X = sm.add_constant(X)
        ols = sm.OLS(Y, X).fit()
        quadratic_param_list_cond2.append(ols.params[2])

    print(quadratic_param_list_cond2)
    print('mean =', np.mean(quadratic_param_list_cond2))
    print(scipy.stats.ttest_1samp(quadratic_param_list_cond2, 0, alternative='less'))

    # paired difference ttest
    print('Difference testing')
    print(scipy.stats.ttest_1samp(np.array(quadratic_param_list_cond1) - np.array(quadratic_param_list_cond2), 0, alternative='less'))

    return


def compute_task_chance_level(behavior_path, permut=1000):

    path = Path(behavior_path)
    subject_folders = sorted(x for x in path.iterdir() if x.is_dir())
    del subject_folders[5]  # subject 5 does not have behavioral data

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        path_to_csv = subject_path / ('P_' + subject_path.name.split('_')[-1] + '/P_' + subject_path.name.split('_')[-1] + '.csv')
        csv = pd.read_csv(path_to_csv)

        csv = csv[10:]  # remove first block (in which the subject does not have any choice)

        rep = csv['rep'].to_numpy()
        rep[rep == 3] = 2

        response = csv['resp'].to_numpy()

        d_prime_array = np.zeros(permut + 1)
        percent_correct_array = np.zeros(permut + 1)
        rng = np.random.default_rng()
        for p in tqdm(range(permut)):
            random_response = rng.permutation(response)

            true_pos = np.sum((random_response == 2) & (rep == 2)) / np.sum(rep == 2)
            false_pos = np.sum((random_response == 2) & (rep == 1)) / np.sum(rep == 1)
            if true_pos == 1:
                true_pos = (len(csv.loc[csv['rep'] > 1]) - 0.5) / len(csv.loc[csv['rep'] > 1])
            if false_pos == 1:
                false_pos = (len(csv.loc[csv['rep'] == 1]) - 0.5) / len(csv.loc[csv['rep'] == 1])
            if true_pos == 0:
                true_pos = 0.5 / len(csv.loc[csv['rep'] > 1])
            if false_pos == 0:
                false_pos = 0.5 / len(csv.loc[csv['rep'] == 1])

            d_prime_array[p] = scipy.stats.norm.ppf(true_pos) - scipy.stats.norm.ppf(false_pos)

            percent_correct_array[p] = np.sum(random_response == rep) / rep.size

        percent_correct = np.sum(response == rep) / rep.size
        percent_correct_array[-1] = percent_correct
        print("Percent correct:", percent_correct)
        print("Percent correct p-value:", np.sum(percent_correct < percent_correct_array) / percent_correct_array.size)

        true_pos = np.sum((response == 2) & (rep == 2)) / np.sum(rep == 2)
        false_pos = np.sum((response == 2) & (rep == 1)) / np.sum(rep == 1)
        if true_pos == 1:
            true_pos = (len(csv.loc[csv['rep'] > 1]) - 0.5) / len(csv.loc[csv['rep'] > 1])
        if false_pos == 1:
            false_pos = (len(csv.loc[csv['rep'] == 1]) - 0.5) / len(csv.loc[csv['rep'] == 1])
        if true_pos == 0:
            true_pos = 0.5 / len(csv.loc[csv['rep'] > 1])
        if false_pos == 0:
            false_pos = 0.5 / len(csv.loc[csv['rep'] == 1])

        d_prime = scipy.stats.norm.ppf(true_pos) - scipy.stats.norm.ppf(false_pos)
        d_prime_array[-1] = d_prime
        print("d':", d_prime)
        print("d' p-value:", np.sum(d_prime < d_prime_array) / d_prime_array.size)

    return


def compute_task_chance_level_seeg(behavior_path, permut=1000):

    behavior_all = pd.read_csv(behavior_path)

    for i, subject in enumerate(np.unique(behavior_all['subject'])):
        print(subject)
        behavior = behavior_all[behavior_all['subject'] == subject]

        rep = behavior['rep1'].to_numpy()
        rep[rep == 0] = 2

        response = behavior['resp'].to_numpy()

        d_prime_array = np.zeros(permut + 1)
        percent_correct_array = np.zeros(permut + 1)
        rng = np.random.default_rng()
        for p in tqdm(range(permut)):
            random_response = rng.permutation(response)

            true_pos = np.sum((random_response == 2) & (rep == 2)) / np.sum(rep == 2)
            false_pos = np.sum((random_response == 2) & (rep == 1)) / np.sum(rep == 1)
            if true_pos == 1:
                true_pos = (np.sum(rep == 2) - 0.5) / np.sum(rep == 2)
            if false_pos == 1:
                false_pos = (np.sum(rep == 1) - 0.5) / np.sum(rep == 1)
            if true_pos == 0:
                true_pos = 0.5 / np.sum(rep == 2)
            if false_pos == 0:
                false_pos = 0.5 / np.sum(rep == 1)

            d_prime_array[p] = scipy.stats.norm.ppf(true_pos) - scipy.stats.norm.ppf(false_pos)

            percent_correct_array[p] = np.sum(random_response == rep) / rep.size

        percent_correct = np.sum(response == rep) / rep.size
        percent_correct_array[-1] = percent_correct
        print("Percent correct:", percent_correct)
        print("Percent correct p-value:", np.sum(percent_correct < percent_correct_array) / percent_correct_array.size)

        true_pos = np.sum((response == 2) & (rep == 2)) / np.sum(rep == 2)
        false_pos = np.sum((response == 2) & (rep == 1)) / np.sum(rep == 1)
        if true_pos == 1:
            true_pos = (np.sum(rep == 2) - 0.5) / np.sum(rep == 2)
        if false_pos == 1:
            false_pos = (np.sum(rep == 1) - 0.5) / np.sum(rep == 1)
        if true_pos == 0:
            true_pos = 0.5 / np.sum(rep == 2)
        if false_pos == 0:
            false_pos = 0.5 / np.sum(rep == 1)

        d_prime = scipy.stats.norm.ppf(true_pos) - scipy.stats.norm.ppf(false_pos)
        d_prime_array[-1] = d_prime
        print("d':", d_prime)
        print("d' p-value:", np.sum(d_prime < d_prime_array) / d_prime_array.size)

    return


def behavior_and_surprise_u_shape(behavior_path):
    behavior = pd.read_csv(behavior_path)
    print(behavior)
    behavior = behavior[behavior['subject'] != 'Bou_Ni']  # remove Bou_Ni
    behavior = behavior.dropna(subset=['correct'])  # remove Bou_Ni
    print(behavior)

    # # remove 50 first trials
    # behavior = behavior[~((behavior["seeg_block"] == 1.0) &
    #                    (behavior["stim_name"].str.extract(r"trial(\d+)").astype(int)[0] < 50))]
    # print(behavior)

    subjects = behavior['subject'].unique()
    # palette = sns.color_palette("Set3", len(subjects) + 1)
    # palette = [color for i, color in enumerate(palette) if i != 1][:len(subjects)]  # skip color 2
    # subject_colors = dict(zip(subjects, palette))
    subject_colors = ['lightgray' for s in range(len(subjects))]

    # Create plot
    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=(5, 5))

    jitters = np.linspace(-0.05, 0.05, len(subjects))

    # Plot scatter points and quadratic fit for each subject
    for s, subject in enumerate(subjects):
        sub_df = behavior[behavior['subject'] == subject]
        x = sub_df['surprise_mean_paulrnn']
        y = sub_df['correct']

        # Scatter plot
        # plt.scatter(x, y + jitters[s], color=subject_colors[subject], s=0.5, label=subject)
        plt.scatter(x, y + jitters[s], color=subject_colors[s], s=0.5, label=subject)

        # Quadratic fit
        coeffs = np.polyfit(x, y, 2)  # Fit y = ax^2 + bx + c
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
        # plt.plot(x_fit, y_fit, color=subject_colors[subject], alpha=1, lw=3)
        plt.plot(x_fit, y_fit, color=subject_colors[s], alpha=1, lw=1.5)

    # Average quadratic fit (on all data)
    x_all = behavior['surprise_mean_paulrnn']
    y_all = behavior['correct']
    coeffs_avg = np.polyfit(x_all, y_all, 2)
    x_fit_avg = np.linspace(x_all.min(), x_all.max(), 200)
    y_fit_avg = np.polyval(coeffs_avg, x_fit_avg)
    plt.plot(x_fit_avg, y_fit_avg, color='black', linewidth=6, label='Average Fit')

    # Labeling and legend
    plt.xlabel('Surprise', fontsize=17)
    plt.ylabel('Correct', fontsize=17)
    plt.gca().tick_params(axis='y', labelsize=17)
    plt.gca().tick_params(axis='x', labelsize=17)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    '''behavioral_matrix'''
    # behavioral_matrix("/home/mathieupvc/These/data/meg_maestro/data")

    '''plot_behaviors'''
    # plot_behaviors(behavioral_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=False), anova_variable='all_variables')
    # plot_behaviors(behavioral_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=False), d_list=['d_overall'], rs_list=['rs_overall'], percent_correct_list=['percent_correct_overall'], resp_list=['resp_overall'])
    # plot_behaviors(behavioral_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=False), d_list=['d_2nd_rep', 'd_3rd_rep'], rs_list=['rs_1st_rep', 'rs_2nd_rep', 'rs_3rd_rep'], eff_list=['eff_2nd_rep', 'eff_3rd_rep'], percent_correct_list=['percent_correct_1st_rep', 'percent_correct_2nd_rep', 'percent_correct_3rd_rep'], resp_list=['resp_1st_rep', 'resp_2nd_rep', 'resp_3rd_rep'], anova_variable='repetition_number')
    # plot_behaviors(behavioral_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=False), d_list=['d_2.5Hz', 'd_3.5Hz', 'd_4.5Hz', 'd_5.5Hz', 'd_6.5Hz'], rs_list=['rs_2.5Hz', 'rs_3.5Hz', 'rs_4.5Hz', 'rs_5.5Hz', 'rs_6.5Hz'], eff_list=['eff_2.5Hz', 'eff_3.5Hz', 'eff_4.5Hz', 'eff_5.5Hz', 'eff_6.5Hz'], percent_correct_list=['percent_correct_2.5Hz', 'percent_correct_3.5Hz', 'percent_correct_4.5Hz', 'percent_correct_5.5Hz', 'percent_correct_6.5Hz'], resp_list=['resp_2.5Hz', 'resp_3.5Hz', 'resp_4.5Hz', 'resp_5.5Hz', 'resp_6.5Hz'], anova_variable='rate')
    # plot_behaviors(behavioral_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=False), d_list=['d_sharp', 'd_smooth'], rs_list=['rs_sharp', 'rs_smooth'], eff_list=['eff_sharp', 'eff_smooth'], percent_correct_list=['percent_correct_sharp', 'percent_correct_smooth'], resp_list=['resp_sharp', 'resp_smooth'], anova_variable='onset_type')

    # plot_behaviors(rate_repeat_interaction_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=False, plot=False),
    #                d_list=[
    #                                      'd_2.5Hz_rep_2', 'd_3.5Hz_rep_2', 'd_4.5Hz_rep_2', 'd_5.5Hz_rep_2', 'd_6.5Hz_rep_2',
    #                                      'd_2.5Hz_rep_3', 'd_3.5Hz_rep_3', 'd_4.5Hz_rep_3', 'd_5.5Hz_rep_3', 'd_6.5Hz_rep_3'],
    #                percent_correct_list=['percent_correct_2.5Hz_rep_1', 'percent_correct_3.5Hz_rep_1', 'percent_correct_4.5Hz_rep_1', 'percent_correct_5.5Hz_rep_1', 'percent_correct_6.5Hz_rep_1',
    #                                      'percent_correct_2.5Hz_rep_2', 'percent_correct_3.5Hz_rep_2', 'percent_correct_4.5Hz_rep_2', 'percent_correct_5.5Hz_rep_2', 'percent_correct_6.5Hz_rep_2',
    #                                      'percent_correct_2.5Hz_rep_3', 'percent_correct_3.5Hz_rep_3', 'percent_correct_4.5Hz_rep_3', 'percent_correct_5.5Hz_rep_3', 'percent_correct_6.5Hz_rep_3'],
    #                resp_list=['resp_2.5Hz_rep_1', 'resp_3.5Hz_rep_1', 'resp_4.5Hz_rep_1', 'resp_5.5Hz_rep_1', 'resp_6.5Hz_rep_1'],
    #                           # 'resp_2.5Hz_rep_2', 'resp_3.5Hz_rep_2', 'resp_4.5Hz_rep_2', 'resp_5.5Hz_rep_2', 'resp_6.5Hz_rep_2',
    #                           # 'resp_2.5Hz_rep_3', 'resp_3.5Hz_rep_3', 'resp_4.5Hz_rep_3', 'resp_5.5Hz_rep_3', 'resp_6.5Hz_rep_3'],
    #                rs_list=['rs_2.5Hz_rep_1', 'rs_3.5Hz_rep_1', 'rs_4.5Hz_rep_1', 'rs_5.5Hz_rep_1', 'rs_6.5Hz_rep_1',
    #                                      'rs_2.5Hz_rep_2', 'rs_3.5Hz_rep_2', 'rs_4.5Hz_rep_2', 'rs_5.5Hz_rep_2', 'rs_6.5Hz_rep_2',
    #                                      'rs_2.5Hz_rep_3', 'rs_3.5Hz_rep_3', 'rs_4.5Hz_rep_3', 'rs_5.5Hz_rep_3', 'rs_6.5Hz_rep_3'],
    #                anova_variable='rate', rs=True, eff=False, resp=True, posthocs=False)
    #
    # plot_behaviors(
    #     onset_repeat_interaction_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=False,
    #                                    plot=False),
    #     d_list=[
    #         'd_onset_0_rep_2', 'd_onset_2_rep_2',
    # 'd_onset_0_rep_3', 'd_onset_2_rep_3',],
    #     percent_correct_list=['percent_correct_onset_0_rep_1', 'percent_correct_onset_2_rep_1',
    # 'percent_correct_onset_0_rep_2', 'percent_correct_onset_2_rep_2',
    # 'percent_correct_onset_0_rep_3', 'percent_correct_onset_2_rep_3'],
    #     resp_list=['resp_onset_0_rep_1', 'resp_onset_2_rep_1',
    #                'resp_onset_0_rep_2', 'resp_onset_2_rep_2',
    #                'resp_onset_0_rep_3', 'resp_onset_2_rep_3'],
    #     rs_list=['rs_onset_0_rep_1', 'rs_onset_2_rep_1',
    # 'rs_onset_0_rep_2', 'rs_onset_2_rep_2',
    # 'rs_onset_0_rep_3', 'rs_onset_2_rep_3',],
    #     anova_variable='rate', rs=True, eff=False, posthocs=False)

    # plot_behaviors(
    #     rate_onset_interaction_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=False),
    #     resp_list=['resp_2.5Hz_onset_0', 'resp_3.5Hz_onset_0', 'resp_4.5Hz_onset_0', 'resp_5.5Hz_onset_0', 'resp_6.5Hz_onset_0',
    #                'resp_2.5Hz_onset_2', 'resp_3.5Hz_onset_2', 'resp_4.5Hz_onset_2', 'resp_5.5Hz_onset_2', 'resp_6.5Hz_onset_2'],
    #     rs_list=['rs_2.5Hz_onset_0', 'rs_3.5Hz_onset_0', 'rs_4.5Hz_onset_0', 'rs_5.5Hz_onset_0', 'rs_6.5Hz_onset_0',
    #              'rs_2.5Hz_onset_2', 'rs_3.5Hz_onset_2', 'rs_4.5Hz_onset_2', 'rs_5.5Hz_onset_2', 'rs_6.5Hz_onset_2',],
    #     anova_variable='rate', d=False, percent=False, rs=True, eff=False, resp=True, posthocs=False)

    # rate with only rep 1 and 2
    # rate_rep_matrix = rate_repeat_interaction_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=False, plot=False)
    # # rate_rep_matrix['percent_correct_2.5Hz'] = (rate_rep_matrix['percent_correct_2.5Hz_rep_1'] + rate_rep_matrix['percent_correct_2.5Hz_rep_2']) / 2
    # # rate_rep_matrix['percent_correct_3.5Hz'] = (rate_rep_matrix['percent_correct_3.5Hz_rep_1'] + rate_rep_matrix['percent_correct_3.5Hz_rep_2']) / 2
    # # rate_rep_matrix['percent_correct_4.5Hz'] = (rate_rep_matrix['percent_correct_4.5Hz_rep_1'] + rate_rep_matrix['percent_correct_4.5Hz_rep_2']) / 2
    # # rate_rep_matrix['percent_correct_5.5Hz'] = (rate_rep_matrix['percent_correct_5.5Hz_rep_1'] + rate_rep_matrix['percent_correct_5.5Hz_rep_2']) / 2
    # # rate_rep_matrix['percent_correct_6.5Hz'] = (rate_rep_matrix['percent_correct_6.5Hz_rep_1'] + rate_rep_matrix['percent_correct_6.5Hz_rep_2']) / 2
    # rate_rep_matrix['percent_correct_2.5Hz'] = rate_rep_matrix['percent_correct_2.5Hz_rep_2']
    # rate_rep_matrix['percent_correct_3.5Hz'] = rate_rep_matrix['percent_correct_3.5Hz_rep_2']
    # rate_rep_matrix['percent_correct_4.5Hz'] = rate_rep_matrix['percent_correct_4.5Hz_rep_2']
    # rate_rep_matrix['percent_correct_5.5Hz'] = rate_rep_matrix['percent_correct_5.5Hz_rep_2']
    # rate_rep_matrix['percent_correct_6.5Hz'] = rate_rep_matrix['percent_correct_6.5Hz_rep_2']
    # print(rate_rep_matrix)
    # plot_behaviors(rate_rep_matrix,
    #     percent_correct_list=['percent_correct_2.5Hz', 'percent_correct_3.5Hz',
    #                           'percent_correct_4.5Hz', 'percent_correct_5.5Hz',
    #                           'percent_correct_6.5Hz'],
    #     anova_variable='rate', rs=False, eff=False, posthocs=True)

    '''scatter_behavior_gmsi'''
    # scatter_behavior_gmsi(behavioral_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=False), column='percent_correct_overall', gmsi_path='/home/mathieupvc/These/data/meg_maestro/gmsi_processed.csv')

    '''scatter_r2_gmsi'''
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/onset_07_12_23/regr_meg_paulrnn.pkl', 'rb') as file:
    #     model_regression = pickle.load(file)
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/onset_07_12_23/regr_meg_acoustic.pkl', 'rb') as file:
    #     control_regression = pickle.load(file)
    # gmsi_path = '/home/mathieupvc/These/data/meg_maestro/gmsi_processed.csv'
    # sign_channels_path = '/home/mathieupvc/These/data/meg_maestro/gmsi_analysis/significant_channels_200ms.npy'
    # scatter_r2_gmsi(model_regression, control_regression, sign_channels_path, behavioral_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=False), column='d_overall', tmin=50)

    '''regr_nb_trials'''
    # regr_nb_trials("/home/mathieupvc/These/data/meg_maestro/data")

    '''all_subj_regr_nb_trials'''
    # all_subj_regr_nb_trials("/home/mathieupvc/These/data/meg_maestro/data")

    '''third_vs_second_rep'''
    # third_vs_second_rep(behavioral_matrix("/home/mathieupvc/These/data/meg_maestro/data"))

    '''rate_repeat_interaction_matrix'''
    # rate_repeat_interaction_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=True)

    '''permanova_rate_repeat'''
    # permanova_rate_repeat(rate_repeat_interaction_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=False, plot=False), n_permut=1000)

    '''rm_anova_rate_repeat'''
    # rm_anova_rate_repeat(rate_repeat_interaction_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=False, plot=False), var='percent_correct')

    '''correct_incorrect_lists'''
    # correct_incorrect_lists(data_path="/home/mathieupvc/These/data/meg_maestro/data", save_folder='/home/mathieupvc/These/data/meg_maestro')
    # correct_incorrect_rep1_lists(data_path="/home/mathieupvc/These/data/meg_maestro/data", save_folder='/home/mathieupvc/These/data/meg_maestro')
    # correct_incorrect_all_reps_lists(data_path="/home/mathieupvc/These/data/meg_maestro/data", save_folder='/home/mathieupvc/These/data/meg_maestro')

    '''old_new_lists'''
    # old_new_lists(data_path="/home/mathieupvc/These/data/meg_maestro/data", save_path='/home/mathieupvc/These/data/meg_maestro')

    '''ushape_testing'''
    # ushape_testing(behavioral_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=True))

    '''ushape_testing_between_conditions'''
    # ushape_testing_between_conditions(rate_repeat_interaction_matrix("/home/mathieupvc/These/data/meg_maestro/data", subtract_overall=True, plot=False))

    '''compute_task_chance_level'''
    behavior_path = "/home/mathieupvc/These/data/meg_maestro/data"
    permut = 10000
    # compute_task_chance_level(behavior_path, permut=permut)

    '''compute_task_chance_level_seeg'''
    behavior_path = "/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv"
    permut = 10000
    # compute_task_chance_level_seeg(behavior_path, permut=permut)

    '''save_seeg_behavior_matrix'''
    # seeg_behavior_matrix('/home/mathieupvc/These/data/meg_maestro/seeg_behavior_data', stim_path='/home/mathieupvc/These/data/meg_maestro/exp_stim_seeg')

    '''plot_seeg_behavior'''
    # plot_seeg_behavior('/home/mathieu/These/data/meg_maestro/seeg_behavior_data', stim_path='/home/mathieu/These/data/meg_maestro/exp_stim_seeg')

    '''behavior_and_surprise_u_shape'''
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    behavior_and_surprise_u_shape(behavior_path)
