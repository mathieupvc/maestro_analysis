import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import scipy

import mne

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm

from maestro_analysis.utils import fixed_golden_aspect_ratio, gmsi_array_extraction, epoching_maestro
from maestro_analysis.stats import permut_test_spearman

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.formatter.limits'] = [-2, 4]

def compare_feature_distributions(epochs_path, feature_list, rate_list, tmin=2, tmax=7, min_time_interval=0.050):

    epochs = mne.read_epochs(epochs_path)

    # keep only notes with min time interval
    if min_time_interval is not None:
        print('Number of notes:', len(epochs.metadata))
        epochs = epochs[epochs.metadata['time_interval'] >= min_time_interval]
        print('Number of notes after min_time_interval:', len(epochs.metadata))

    metadata = epochs.metadata
    metadata['shifted_entropy_timing_paulrnn'] = np.nan  # add shifted features
    for i_note in range(metadata.shape[0]):
        if (i_note + 1 < len(metadata)) and (metadata.loc[metadata.index[i_note], 'stim_name'] == metadata.loc[metadata.index[i_note + 1], 'stim_name']):
            metadata.loc[metadata.index[i_note], 'shifted_entropy_timing_paulrnn'] = metadata.loc[metadata.index[i_note + 1], 'entropy_timing_paulrnn']

    epochs.metadata = metadata

    # remove onset response
    epochs = epochs[epochs.metadata['trial_time'].to_numpy() >= tmin]
    epochs = epochs[epochs.metadata['trial_time'].to_numpy() <= tmax]
    print('Number of notes after min/max trial time:', len(epochs.metadata))

    # add rate
    metadata = epochs.metadata
    metadata['rate (Hz)'] = 0  # add rate from stim name
    for i_note in range(metadata.shape[0]):
        metadata.loc[metadata.index[i_note], 'rate (Hz)'] = float(metadata.loc[metadata.index[i_note], 'stim_name'].split('_')[2][0:3])
    epochs.metadata = metadata
    epochs = epochs[epochs.metadata['rate (Hz)'].to_numpy() != 2.5]  # remove 2.5Hz

    metadata = epochs.metadata

    print(metadata.groupby('rate (Hz)')[feature_list].mean())
    print(metadata.groupby('rate (Hz)')[feature_list].std())

    for r, rate in enumerate(rate_list):

        # metadata[metadata['rate (Hz)'].to_numpy() == rate][feature_list] = (metadata[metadata['rate (Hz)'].to_numpy() == rate][feature_list] - metadata[metadata['rate (Hz)'].to_numpy() == rate][feature_list].mean()) / metadata[metadata['rate (Hz)'].to_numpy() == rate][feature_list].std()
        for feat in feature_list:
            metadata.loc[metadata['rate (Hz)'].to_numpy() == rate, feat] = (metadata.loc[metadata['rate (Hz)'].to_numpy() == rate, feat] - metadata.groupby('rate (Hz)')[feat].mean()[rate]) / metadata.groupby('rate (Hz)')[feat].std()[rate]
    epochs.metadata = metadata

    print(metadata.groupby('rate (Hz)')[feature_list].mean())
    print(metadata.groupby('rate (Hz)')[feature_list].std())

    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 0.5])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    color = ['#b6adff', '#BC7DB6', '#C24D6E', '#c81d25']
    sns.kdeplot(data=epochs.metadata, x=feature_list[0], hue='rate (Hz)', fill=False, common_norm=False, cumulative=False, palette=color, alpha=1, linewidth=3, ax=ax1)
    sns.kdeplot(data=epochs.metadata, x=feature_list[0], hue='rate (Hz)', fill=False, common_norm=False, cumulative=True, palette=color, linewidth=2, ax=ax2)
    sns.kdeplot(data=epochs.metadata, x=feature_list[1], hue='rate (Hz)', fill=False, common_norm=False, cumulative=False, palette=color, alpha=1, linewidth=3, ax=ax3)
    sns.kdeplot(data=epochs.metadata, x=feature_list[1], hue='rate (Hz)', fill=False, common_norm=False, cumulative=True, palette=color, linewidth=2, ax=ax4)
    ax2.set_ylabel('CDF')
    ax4.set_ylabel('CDF')

    plt.show()

    return


def r2_by_rate(model_regression, control_regression, data_type='meg', tmin=50, mode='diff', conditions='rate', save_path='/home/mathieupvc/These/figures_papier', best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_20channels_100ms.npy', seeg_pickle='/home/mathieupvc/These/figures_papier/seeg/f4.pkl'):

    if conditions == 'rate':
        cond_label = 'Hz'
    elif conditions == 'avg_surprise':
        cond_label = ''
    elif conditions == 'onset':
        cond_label = ''
    elif conditions == 'trial_time':
        cond_label = ''

    if data_type == 'meg':
        best_channels = np.load(best_channels_path)
        significant_channels = np.load(os.path.join(save_path, 'significant_channels_200ms.npy'))
        print(significant_channels.shape)
        print(significant_channels)

        control_r_squared = control_regression['r2'][:, :, tmin:, 0:]  # remove the 2.5Hz condition
        model_r_squared = model_regression['r2'][:, :, tmin:, 0:]
        betas = model_regression['betas'][:, model_regression['features'].index('surprise_mean_paulrnn'), :, tmin:, 0:]
        # betas = model_regression['betas'][:, model_regression['features'].index('shifted_surprise_mean_paulrnn'), :, tmin:, 1:]
        # betas = model_regression['betas'][:, model_regression['features'].index('shifted_entropy_timing_paulrnn'), :, tmin:, 1:]
        r2_diff = model_r_squared - control_r_squared
        times = model_regression['times'][tmin:]
        if conditions == 'rate':
            rates = model_regression['rates'][0:]
        elif conditions == 'avg_surprise':
            rates = model_regression['avg_surprise_quantiles'][0:]
        elif conditions == 'onset':
            rates = ['sharp', 'smooth']
        elif conditions == 'trial_time':
            rates = ['first_half', 'second_half']

        # best r2 channels
        # channels_r2_diff = np.mean(r2_diff[:, :, 65:80, :], axis=(0, 2, 3))
        # best_channels = np.argsort(channels_r2_diff)[::-1]
        # print(best_channels[:37])

    elif data_type == 'seeg':
        control_r_squared = np.load(control_regression)[:, tmin:, 0:]
        model_r_squared = np.load(model_regression)[:, tmin:, 0:]

        with open(seeg_pickle, 'rb') as f:
            seeg_data = pickle.load(f)

        # best_channels = np.array(seeg_data['mask_n100_10'])
        best_channels = np.array(seeg_data['ch_sign']).reshape((-1, 1))  # set the channels as subjects, and the channel dimension at size 1
        control_r_squared = control_r_squared[best_channels, :, :]
        model_r_squared = model_r_squared[best_channels, :, :]
        r2_diff = model_r_squared - control_r_squared
        times = seeg_data['times'][tmin:]
        if conditions == 'rate':
            rates = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        elif conditions == 'avg_surprise':
            rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        elif conditions == 'onset':
            rates = ['sharp', 'smooth']
        elif conditions == 'trial_time':
            rates = ['first_half', 'second_half']

    #'#B6ADFF'
    if len(rates) == 4:
        color = ['#b6adff', '#BC7DB6', '#C24D6E', '#c81d25']
    elif len(rates) == 5:
        color = ['#b6adff', '#bb89c9', '#bf6592', '#c4415c', '#c81d25']
    elif len(rates) == 6:
        color = ['#b6adff', '#bb89c9', '#bf6592', '#c4415c', '#c81d25', 'red']
    elif len(rates) == 2:
        color = ['#b6adff', '#c81d25']
    # cmap = plt.cm.get_cmap('magma')
    # color = [cmap(x) for x in np.linspace(0.3, 0.9, len(model_regression['rates']))]
    elif len(rates) == 10:
        cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#ff5a5f', '#6859D9', '#087e8b'])
        color = [cmap1(i) for i in np.linspace(0, 1, len(rates))]

    index_list = [[], []]
    value_list = [[], []]
    if data_type == 'meg':
        t1 = 60
        t2 = 80
        t3 = 80
        t4 = 100
    elif data_type == 'seeg':
        t1 = 60
        t2 = 75
        t3 = 75
        t4 = 90
    fig2, axes_temporal = plt.subplots(1, 1)
    for r, rate in enumerate(rates):
        if mode == 'diff':
            rate_r2_diff = r2_diff[:, :, :, r]  # select rate and average signif channels
        elif mode == 'control':
            rate_r2_diff = control_r_squared[:, :, :, r]  # select rate and average signif channels
        elif mode == 'model':
            rate_r2_diff = model_r_squared[:, :, :, r]  # select rate and average signif channels
        if data_type == 'meg':
            rate_r2_diff = np.mean(rate_r2_diff[:, significant_channels, :], axis=1)
        elif data_type == 'seeg':
            rate_r2_diff = np.mean(rate_r2_diff, axis=1)
        # rate_r2_diff = np.mean(rate_r2_diff[:, best_channels[:10], :], axis=1)
        # rate_r2_diff = np.mean(np.take_along_axis(rate_r2_diff, best_channels[:, :, np.newaxis], axis=1), axis=1)
        # rate_r2_diff = np.mean(r2_diff[:, :, :, r], axis=1)

        line, = axes_temporal.plot(times, np.mean(rate_r2_diff, axis=0), color=color[r], label=str(rate) + cond_label, lw=1)
        index = np.argmax(np.mean(rate_r2_diff, axis=0)[t1:t2]) + t1
        index_list[0].append(index)
        value_list[0].append(np.mean(rate_r2_diff, axis=0)[index])
        index = np.argmax(np.mean(rate_r2_diff, axis=0)[t3:t4]) + t3
        index_list[1].append(index)
        value_list[1].append(np.mean(rate_r2_diff, axis=0)[index])

    axes_temporal.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    axes_temporal.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    if mode == 'diff':
        axes_temporal.set_ylabel('$\Delta$R²')
    elif (mode == 'control') or (mode == 'model'):
        axes_temporal.set_ylabel('Control R²')
    axes_temporal.set_xlabel('Time (s)')
    leg = axes_temporal.legend()
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2)

    for r, rate in enumerate(rates):
        axes_temporal.scatter(times[index_list[0][r]], value_list[0][r], color=color[r], s=14)
        axes_temporal.axvline(times[index_list[0][r]], 0, (value_list[0][r] - axes_temporal.get_ylim()[0]) / (axes_temporal.get_ylim()[1] - axes_temporal.get_ylim()[0]), linestyle='--', linewidth=1, color=color[r], zorder=0)
        axes_temporal.scatter(times[index_list[1][r]], value_list[1][r], color=color[r], s=14)
        axes_temporal.axvline(times[index_list[1][r]], 0, (value_list[1][r] - axes_temporal.get_ylim()[0]) / (axes_temporal.get_ylim()[1] - axes_temporal.get_ylim()[0]), linestyle='--', linewidth=1, color=color[r], zorder=0)

    fixed_golden_aspect_ratio(axes_temporal)

    fig2.savefig(os.path.join(save_path, f'{conditions}_{mode}_r2_{data_type}.svg'))

    ####################################################################################################################
    # Betas
    ####################################################################################################################

    if data_type == 'meg':
        # manage channel polarities
        polarity_weights = (np.mean(betas[:, :, 65:80, :], axis=(0, 2, 3)) >= 0).astype(int)
        polarity_weights[polarity_weights==0] = -1
        betas = betas * polarity_weights[np.newaxis, :, np.newaxis, np.newaxis]

        index_list = [[], []]
        value_list = [[], []]
        fig2, axes_temporal = plt.subplots(1, 1)
        for r, rate in enumerate(rates):
            rate_betas = betas[:, :, :, r]  # select rate and average signif channels
            rate_betas = np.mean(rate_betas[:, significant_channels, :], axis=1)
            # rate_betas = np.mean(rate_betas[:, best_channels[:20], :], axis=1)
            # rate_betas = np.mean(np.take_along_axis(rate_betas, best_channels[:, :, np.newaxis], axis=1), axis=1)

            line, = axes_temporal.plot(times, np.mean(rate_betas, axis=0), color=color[r], label=str(rate) + cond_label)

            index = np.argmax(np.abs(np.mean(rate_betas, axis=0)[65:80])) + 65
            index_list[0].append(index)
            value_list[0].append(np.mean(rate_betas, axis=0)[index])
            index = np.argmax(np.abs(np.mean(rate_betas, axis=0)[85:100])) + 85
            index_list[1].append(index)
            value_list[1].append(np.mean(rate_betas, axis=0)[index])

        axes_temporal.axvline(0, linestyle='--', linewidth=2, color='k', zorder=0)
        axes_temporal.axhline(0, linestyle='--', linewidth=2, color='k', zorder=0)
        axes_temporal.set_ylabel(r'$\beta$ (a.u.)')
        axes_temporal.set_xlabel('Time (s)')
        leg = axes_temporal.legend()
        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(6)

        for r, rate in enumerate(rates):
            axes_temporal.scatter(times[index_list[0][r]], value_list[0][r], color=color[r], linewidth=5)
            axes_temporal.axvline(times[index_list[0][r]], 0, (value_list[0][r] - axes_temporal.get_ylim()[0]) / (axes_temporal.get_ylim()[1] - axes_temporal.get_ylim()[0]), linestyle='--', linewidth=2, color=color[r], zorder=0)
            axes_temporal.scatter(times[index_list[1][r]], value_list[1][r], color=color[r], linewidth=5)
            axes_temporal.axvline(times[index_list[1][r]], 0, (value_list[1][r] - axes_temporal.get_ylim()[0]) / (axes_temporal.get_ylim()[1] - axes_temporal.get_ylim()[0]), linestyle='--', linewidth=2, color=color[r], zorder=0)

        fixed_golden_aspect_ratio(axes_temporal)

        fig2.savefig(os.path.join(save_path, f'{conditions}_betas_surprise_meg.svg'))
        # fig2.savefig(os.path.join(save_path, 'rate_shifted_timing_meg.svg'))

    return


def boxplot_by_rate(model_regression, control_regression, data_type='meg', tmin=50, mode='diff', conditions='rate', save_path='/home/mathieupvc/These/figures_papier', best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_20channels_100ms.npy', seeg_pickle='/home/mathieupvc/These/figures_papier/seeg/f4.pkl', gmsi_path='/home/mathieupvc/These/data/meg_maestro/gmsi_processed.csv'):

    gmsi_array = gmsi_array_extraction(gmsi_path=gmsi_path)

    if conditions == 'rate':
        cond_label = 'Hz'
    elif conditions == 'avg_surprise':
        cond_label = ''
    elif conditions == 'onset':
        cond_label = ''
    elif conditions == 'trial_time':
        cond_label = ''

    if data_type == 'meg':
        best_channels = np.load(best_channels_path)
        significant_channels = np.load(os.path.join(save_path, 'significant_channels_200ms.npy'))
        print(significant_channels.shape)
        print(significant_channels)

        control_r_squared = control_regression['r2'][:, :, tmin:, 0:]  # remove the 2.5Hz condition
        model_r_squared = model_regression['r2'][:, :, tmin:, 0:]
        betas = model_regression['betas'][:, model_regression['features'].index('surprise_mean_paulrnn'), :, tmin:, 0:]
        r2_diff = model_r_squared - control_r_squared
        times = model_regression['times'][tmin:]
        if conditions == 'rate':
            rates = model_regression['rates'][0:]
        elif conditions == 'avg_surprise':
            rates = model_regression['avg_surprise_quantiles'][0:]
        elif conditions == 'onset':
            rates = ['sharp', 'smooth']
        elif conditions == 'trial_time':
            rates = ['first_half', 'second_half']

        # best r2 channels
        # channels_r2_diff = np.mean(r2_diff[:, :, 65:80, :], axis=(0, 2, 3))
        # best_channels = np.argsort(channels_r2_diff)[::-1]
        # print(best_channels[:37])

    elif data_type == 'seeg':
        control_r_squared = np.load(control_regression)[:, tmin:, 0:]
        model_r_squared = np.load(model_regression)[:, tmin:, 0:]

        with open(seeg_pickle, 'rb') as f:
            seeg_data = pickle.load(f)

        # best_channels = np.array(seeg_data['mask_n100_10'])
        best_channels = np.array(seeg_data['ch_sign']).reshape((-1, 1))  # set the channels as subjects, and the channel dimension at size 1
        control_r_squared = control_r_squared[best_channels, :, :]
        model_r_squared = model_r_squared[best_channels, :, :]
        # control_r_squared = control_r_squared[:, np.newaxis, :, :]
        # model_r_squared = model_r_squared[:, np.newaxis, :, :]
        r2_diff = model_r_squared - control_r_squared
        times = seeg_data['times'][tmin:]
        if conditions == 'rate':
            rates = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
            # rates = [3.5, 4.5, 5.5, 6.5, 7.5]
        elif conditions == 'avg_surprise':
            rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        elif conditions == 'onset':
            rates = ['sharp', 'smooth']
        elif conditions == 'trial_time':
            rates = ['first_half', 'second_half']

    peak_time = {}
    peak_value = {}
    if data_type == 'meg':
        t1 = 65
        t2 = 80
        # t1 = 80
        # t2 = 100
    elif data_type == 'seeg':
        t1 = 60
        t2 = 75
    for r, rate in enumerate(rates):
        if mode == 'diff':
            rate_r2_diff = r2_diff[:, :, :, r]  # select rate and average signif channels
        elif mode == 'control':
            rate_r2_diff = control_r_squared[:, :, :, r]  # select rate and average signif channels
        elif mode == 'model':
            rate_r2_diff = model_r_squared[:, :, :, r]  # select rate and average signif channels
        if data_type == 'meg':
            rate_r2_diff = rate_r2_diff[:, significant_channels, :]
        # rate_r2_diff = np.mean(rate_r2_diff[:, best_channels[:10], :], axis=1)
        # rate_r2_diff = np.take_along_axis(rate_r2_diff, best_channels[:, :, np.newaxis], axis=1)
        # rate_r2_diff = np.mean(r2_diff[:, :, :, r], axis=1)

        indices = np.argmax(rate_r2_diff[:, :, t1:t2], axis=2) + t1
        peak_time[str(rate) + cond_label] = np.mean(times[indices], axis=1)
        peak_value[str(rate) + cond_label] = np.mean(np.take_along_axis(rate_r2_diff, indices[:, :, np.newaxis], axis=2), axis=(1, 2))

        # peak_time[str(rate) + 'Hz'] = times[np.argmax(np.mean(rate_r2_diff[:, :, t1:t2], axis=1), axis=1) + t1]
        # peak_value[str(rate) + 'Hz'] = np.max(np.mean(rate_r2_diff[:, :, t1:t2], axis=1), axis=1)

    df_time = pd.DataFrame(peak_time)
    df_value = pd.DataFrame(peak_value)

    if len(rates) == 4:
        color = ['#b6adff', '#BC7DB6', '#C24D6E', '#c81d25']
    elif len(rates) == 5:
        color = ['#b6adff', '#bb89c9', '#bf6592', '#c4415c', '#c81d25']
    elif len(rates) == 6:
        color = ['#b6adff', '#bb89c9', '#bf6592', '#c4415c', '#c81d25', 'red']
    elif len(rates) == 2:
        color = ['#b6adff', '#c81d25']
    elif len(rates) > 6:
        cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#ff5a5f', '#6859D9', '#087e8b'])
        color = [cmap1(i) for i in np.linspace(0, 1, len(rates))]

    # dot colors by gmsi score
    cmap_dots = plt.cm.get_cmap('viridis')
    gmsi_array = (gmsi_array - np.min(gmsi_array)) / (np.max(gmsi_array) - np.min(gmsi_array))  # normalize gmsi
    if data_type == 'meg':
        dot_colors = [cmap_dots(x) for x in gmsi_array]
    else:
        dot_colors = 'k'
    # color = [cmap(x) for x in np.linspace(0.3, 0.9, len(model_regression['rates']))]

    fig1, axes_boxplot = plt.subplots(1, 1, figsize=(5, 5))
    sns.boxplot(data=df_time, ax=axes_boxplot, palette=color, fill=False, linecolor='black', width=0.6, linewidth=2, dodge=False, fliersize=0)
    sns.stripplot(data=df_time, color=dot_colors, size=3)
    if mode == 'diff':
        axes_boxplot.set_ylabel('$\Delta$R² peak time (s)')
    elif (mode == 'control') or (mode == 'model'):
        axes_boxplot.set_ylabel('Control R² peak time (s)')

    fig2, axes_boxplot2 = plt.subplots(1, 1, figsize=(5, 5))
    sns.boxplot(data=df_value, ax=axes_boxplot2, palette=color, fill=False, linecolor='black', width=0.6, linewidth=2, dodge=False, fliersize=0)
    sns.stripplot(data=df_value, color=dot_colors, size=3)
    if mode == 'diff':
        axes_boxplot2.set_ylabel('$\Delta$R² peak value')
    elif (mode == 'control') or (mode == 'model'):
        axes_boxplot2.set_ylabel('Control R² peak value')

    fig1.savefig(os.path.join(save_path, f'{conditions}_boxplot_{mode}_time_{data_type}.svg'))
    fig2.savefig(os.path.join(save_path, f'{conditions}_boxplot_{mode}_value_{data_type}.svg'))

    if mode == 'diff':
        test_data = r2_diff
    elif mode == 'control':
        test_data = control_r_squared
    elif mode == 'model':
        test_data = model_r_squared
    if data_type == 'meg':
        test_data = test_data[:, significant_channels, :, :]

    # permutation test value
    test_value = np.mean(np.max(test_data[:, :, t1:t2, :], axis=2), axis=1)  # shape (subjects, rates)
    # test_data = np.max(np.mean(test_data[:, :, t1:t2, :], axis=1), axis=1)  # shape (subjects, rates)
    stat, p = permut_test_spearman(test_value, rates, tail=-1, n_permut=100)
    print('Value stat (Spearman rank correlation) =', stat)
    print('p_value (H1: stat<0) =', p)
    spearman_value = [scipy.stats.spearmanr(np.column_stack((test_value[s, :], rates))).statistic for s in range(test_data.shape[0])]

    # permutation test timing
    test_time = np.mean(times[np.argmax(test_data[:, :, t1:t2, :], axis=2) + t1], axis=1)  # shape (subjects, rates)
    # test_data = times[np.argmax(np.mean(test_data[:, :, t1:t2, :], axis=1), axis=1) + t1]  # shape (subjects, rates)
    stat, p = permut_test_spearman(test_time, rates, tail=1, n_permut=100)
    print('Timing stat (Spearman rank correlation) =', stat)
    print('p_value (H1: stat>0) =', p)
    spearman_time = [scipy.stats.spearmanr(np.column_stack((test_time[s, :], rates))).statistic for s in range(test_data.shape[0])]

    # gmsi vs spearman
    if data_type == 'meg':
        fig3, axes_scatter = plt.subplots(1, 1, figsize=(4.5, 4.5))
        axes_scatter.scatter(gmsi_array, spearman_value)

        fig3.savefig(os.path.join(save_path, f'{conditions}_gmsi_spearman_{mode}_{data_type}.svg'))

        # print('corr gmsi-spearman:', np.corrcoef(gmsi_array, spearman)[0, 1])
        print('corr gmsi-spearman:', scipy.stats.spearmanr(gmsi_array, spearman_value, alternative='two-sided').statistic)

        # r(gmsi, deltaR²) vs deciles
        r_gmsi_delta_r2 = np.array([scipy.stats.spearmanr(gmsi_array, test_value[:, r]).statistic for r in range(test_value.shape[1])])
        fig5, axes_line = plt.subplots(2, 1)
        axes_line[0].scatter(rates, r_gmsi_delta_r2)
        axes_line[0].set_ylabel('r(gmsi, $\Delta$R² peak value)')
        r_gmsi_delta_r2 = np.array([scipy.stats.spearmanr(gmsi_array, test_time[:, r]).statistic for r in range(test_value.shape[1])])
        axes_line[1].scatter(rates, r_gmsi_delta_r2)
        axes_line[1].set_xlabel('rate or avg_surprise deciles')
        axes_line[1].set_ylabel('r(gmsi, $\Delta$R² peak time)')
        fig5.savefig(os.path.join(save_path, f'{conditions}_r_gmsi_deltar2_{mode}_{data_type}.svg'))

    # histogram of channel spearman
    if data_type == 'seeg':
        fig4, axes_hist = plt.subplots(1, 2, figsize=(4.5, 4.5))
        axes_hist[0].hist(spearman_value, bins=100)
        axes_hist[0].set_ylabel('n channels')
        axes_hist[0].set_xlabel('Spearman value')
        axes_hist[1].hist(spearman_time, bins=100)
        axes_hist[1].set_ylabel('n channels')
        axes_hist[1].set_xlabel('Spearman time')

        fig4.savefig(os.path.join(save_path, f'{conditions}_hist_spearman_{mode}_{data_type}.svg'))

    # repeated measures ANOVA
    df = pd.DataFrame(test_value, columns=rates)
    df['subject_id'] = range(len(df))
    df = pd.melt(df, id_vars=['subject_id'], var_name='quantile_or_rate', value_name='peak_value')
    anova = AnovaRM(data=df, depvar='peak_value', subject='subject_id', within=['quantile_or_rate'])
    anova = anova.fit()
    print('anova value')
    print(anova)

    df = pd.DataFrame(test_time, columns=rates)
    df['subject_id'] = range(len(df))
    df = pd.melt(df, id_vars=['subject_id'], var_name='quantile_or_rate', value_name='peak_time')
    anova = AnovaRM(data=df, depvar='peak_time', subject='subject_id', within=['quantile_or_rate'])
    anova = anova.fit()
    print('anova time')
    print(anova)

    # parabolic fit
    Y = np.mean(test_value, axis=0)
    X = np.zeros((len(rates), 2))
    X[:, 0] = rates
    X[:, 1] = X[:, 0]**2
    X = sm.add_constant(X)
    ols = sm.OLS(Y, X).fit()
    print(ols.summary())
    print('OLS with only quantile:')
    X = np.zeros((len(rates), 1))
    X[:, 0] = rates
    X = sm.add_constant(X)
    ols = sm.OLS(Y, X).fit()
    print(ols.summary())

    # post-hoc tests
    print('Running post-hoc tests...')
    p_matrix_value = np.ones((len(rates), len(rates)))
    p_matrix_time = np.ones((len(rates), len(rates)))
    n_tests = len(rates) * (len(rates) - 1) / 2
    for r1 in range(len(rates)):
        for r2 in range(r1+1, len(rates)):
            x = test_value[:, r1] - test_value[:, r2]
            t = np.mean(x) / (np.std(x) / np.sqrt(x.shape[0]))
            p_matrix_value[r1, r2] = 2 * (1 - scipy.stats.t.cdf(np.abs(t), x.shape[0] - 1)) * n_tests  # two-tailed, df is n-1, Bonferroni correction
            x = test_time[:, r1] - test_time[:, r2]
            t = np.mean(x) / (np.std(x) / np.sqrt(x.shape[0]))
            p_matrix_time[r1, r2] = 2 * (1 - scipy.stats.t.cdf(np.abs(t), x.shape[0] - 1)) * n_tests  # two-tailed, df is n-1, Bonferroni correction
    print('Post-hoc values:')
    print(p_matrix_value)
    print('Post-hoc timings:')
    print(p_matrix_time)

    ####################################################################################################################
    # Betas
    ####################################################################################################################

    # # manage channel polarities
    # polarity_weights = (np.mean(betas[:, :, 65:80, :], axis=(0, 2, 3)) >= 0).astype(int)
    # polarity_weights[polarity_weights==0] = -1
    # betas = betas * polarity_weights[np.newaxis, :, np.newaxis, np.newaxis]
    #
    # index_list = [[], []]
    # value_list = [[], []]
    # fig2, axes_temporal = plt.subplots(1, 1)
    # for r, rate in enumerate(rates):
    #     rate_betas = betas[:, :, :, r]  # select rate and average signif channels
    #     rate_betas = np.mean(rate_betas[:, significant_channels, :], axis=1)
    #     # rate_betas = np.mean(rate_betas[:, best_channels[:20], :], axis=1)
    #     # rate_betas = np.mean(np.take_along_axis(rate_betas, best_channels[:, :, np.newaxis], axis=1), axis=1)
    #
    #     line, = axes_temporal.plot(times, np.mean(rate_betas, axis=0), color=color[r], label=str(rate) + 'Hz')
    #
    #     index = np.argmax(np.abs(np.mean(rate_betas, axis=0)[65:80])) + 65
    #     index_list[0].append(index)
    #     value_list[0].append(np.mean(rate_betas, axis=0)[index])
    #     index = np.argmax(np.abs(np.mean(rate_betas, axis=0)[85:100])) + 85
    #     index_list[1].append(index)
    #     value_list[1].append(np.mean(rate_betas, axis=0)[index])
    #
    # axes_temporal.axvline(0, linestyle='--', linewidth=2, color='k', zorder=0)
    # axes_temporal.axhline(0, linestyle='--', linewidth=2, color='k', zorder=0)
    # axes_temporal.set_ylabel(r'$\beta$ (a.u.)')
    # axes_temporal.set_xlabel('Time (s)')
    # leg = axes_temporal.legend()
    # # set the linewidth of each legend object
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(6)
    #
    # for r, rate in enumerate(rates):
    #     axes_temporal.scatter(times[index_list[0][r]], value_list[0][r], color=color[r], linewidth=5)
    #     axes_temporal.axvline(times[index_list[0][r]], 0, (value_list[0][r] - axes_temporal.get_ylim()[0]) / (axes_temporal.get_ylim()[1] - axes_temporal.get_ylim()[0]), linestyle='--', linewidth=2, color=color[r], zorder=0)
    #     axes_temporal.scatter(times[index_list[1][r]], value_list[1][r], color=color[r], linewidth=5)
    #     axes_temporal.axvline(times[index_list[1][r]], 0, (value_list[1][r] - axes_temporal.get_ylim()[0]) / (axes_temporal.get_ylim()[1] - axes_temporal.get_ylim()[0]), linestyle='--', linewidth=2, color=color[r], zorder=0)
    #
    # fixed_golden_aspect_ratio(axes_temporal)
    #
    # fig2.savefig(os.path.join(save_path, 'rate_betas_surprise_meg.svg'))

    return


def save_ecg_epochs(data_path, wav_folder, save_path, t_min=-1):

    os.makedirs(save_path, exist_ok=True)

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        epochs = epoching_maestro(raw_path, wav_folder, allow_maxshield=False, tmin=t_min, baseline=False, bin_ecg=True)

        epochs.save(os.path.join(save_path, subject_path.name + '_ecg-epo.fif'), overwrite=True)

        del epochs

    return

def heartbeat_note_rate(data_matrix_path):

    data_matrix = pd.read_csv(data_matrix_path)

    rates = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']
    subjects = np.unique(data_matrix['subject'])
    print(subjects)

    subjects_list = []
    rate_list = []
    heart_pulses = []
    for subject in subjects:
        for rate in rates:
            rate_matrix = data_matrix[(data_matrix['subject'] == subject) & (data_matrix[rate] == 1)]
            subjects_list.append(subject)
            rate_list.append(rate)
            heart_pulses.append(rate_matrix['heart_pulses'].mean())

    df = pd.DataFrame({'subject': subjects_list, 'rate': rate_list, 'heart pulses': heart_pulses})
    print(df)

    # repeated measures ANOVA
    anova = AnovaRM(data=df, depvar="heart pulses", subject='subject', within=['rate'])
    anova = anova.fit()
    print('anova')
    print(anova)

    sns.boxplot(data=df, x='rate', y='heart pulses', showmeans=True, palette=sns.color_palette('Set3'), showfliers=False)
    sns.swarmplot(data=df, x='rate', y='heart pulses', color='k', size=4)

    plt.show()

    return


if __name__ == "__main__":

    '''compare_feature_distributions'''
    # compare_feature_distributions(epochs_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/maestro01_itdheb-epo.fif',
    #                               feature_list=['surprise_mean_paulrnn', 'shifted_entropy_timing_paulrnn'],
    #                               rate_list=[3.5, 4.5, 5.5, 6.5],
    #                               tmin=2,
    #                               tmax=7,
    #                               min_time_interval=0.050
    #                               )

    '''r2_by_rate'''
    # conditions = 'trial_time'
    # with open(f'/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/onset_07_12_23/regr_by_{conditions}_meg_paulrnn.pkl', 'rb') as file:
    #     model_regression = pickle.load(file)
    # with open(f'/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/onset_07_12_23/regr_by_{conditions}_meg_acoustic.pkl', 'rb') as file:
    #     control_regression = pickle.load(file)
    # model_regression = f'/home/mathieupvc/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta/regression/by_rate_05_01_24/regr_by_{conditions}_seeg_raw_mono_paulrnn_all.npy'
    # control_regression = f'/home/mathieupvc/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta/regression/by_rate_05_01_24/regr_by_{conditions}_seeg_raw_mono_acoustic_all.npy'
    # r2_by_rate(model_regression, control_regression, data_type='meg', tmin=50, mode='diff', conditions=conditions, save_path='/home/mathieupvc/These/data/meg_maestro/rate_analysis', best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_20channels_100ms.npy', seeg_pickle='/home/mathieupvc/These/figures_papier/seeg/f4.pkl')

    '''boxplot_by_rate'''
    # conditions = 'trial_time'
    # with open(f'/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/onset_07_12_23/regr_by_{conditions}_meg_paulrnn.pkl', 'rb') as file:
    #     model_regression = pickle.load(file)
    # with open(f'/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/onset_07_12_23/regr_by_{conditions}_meg_acoustic.pkl', 'rb') as file:
    #     control_regression = pickle.load(file)
    # model_regression = f'/home/mathieupvc/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta/regression/by_rate_05_01_24/regr_by_{conditions}_seeg_raw_mono_paulrnn_all.npy'
    # control_regression = f'/home/mathieupvc/These/data/meg_maestro/note_analysis_seeg/epochs_1s_all_meta/regression/by_rate_05_01_24/regr_by_{conditions}_seeg_raw_mono_acoustic_all.npy'
    # boxplot_by_rate(model_regression, control_regression, data_type='meg', tmin=50, mode='diff', conditions=conditions, save_path='/home/mathieupvc/These/data/meg_maestro/rate_analysis', best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_20channels_100ms.npy', seeg_pickle='/home/mathieupvc/These/figures_papier/seeg/f4.pkl', gmsi_path='/home/mathieupvc/These/data/meg_maestro/gmsi_processed.csv')

    '''heartbeat_note_rate'''
    heartbeat_note_rate(data_matrix_path='/home/mathieupvc/These/data/meg_maestro/behavior_analysis/behavior_data_matrix.csv')