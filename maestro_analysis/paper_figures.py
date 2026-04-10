import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import copy

import scipy.stats
from statannotations.Annotator import Annotator
from statsmodels.stats.anova import AnovaRM
from pingouin import pairwise_tests

import mne
from mne.stats import spatio_temporal_cluster_1samp_test, permutation_cluster_1samp_test
from mne.channels import find_ch_adjacency

from utils import fixed_golden_aspect_ratio, gmsi_array_extraction
from stats import permutation_cluster_test_adapted
from note_analysis import compute_note_density

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['figure.figsize'] = (7.20, 4.45)
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['axes.titlesize'] = 5
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['axes.labelsize'] = 5
matplotlib.rcParams['axes.labelpad'] = 2
matplotlib.rcParams['axes.linewidth'] = 0.4
matplotlib.rcParams['xtick.labelsize'] = 4
matplotlib.rcParams['xtick.major.size'] = 2
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['xtick.major.pad'] = 1
matplotlib.rcParams['ytick.labelsize'] = 4
matplotlib.rcParams['ytick.major.size'] = 2
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['ytick.major.pad'] = 1
matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams['legend.fontsize'] = 5
matplotlib.rcParams['axes.formatter.limits'] = [-2, 4]
# plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set2.colors)


def figure_2(model_regression, control_regression, permutation_folder, vp_path, vp_feature, feature_names, best_channels_path, tmin=50, save_path='/home/mathieupvc/These/figures_papier'):

    # fig = plt.figure(layout="constrained", figsize=(7.20, 4.45*2))
    # fig = plt.figure(layout="constrained", figsize=(3.60, 4.45))
    fig = plt.figure(layout="constrained", figsize=(3.60, 4.9))
    # fig = plt.figure(layout="constrained", figsize=(7.20, 4.45*1.4))
    # fig = plt.figure()
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 0.3]) #, hspace=0.1, wspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    # gs10 = gs[1, 0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.1)
    ax3 = fig.add_subplot(gs[0, 1])
    gs11 = gs[1, 1].subgridspec(2, 1, height_ratios=[1, 0.03])
    ax5 = fig.add_subplot(gs11[0, 0])
    ax6 = fig.add_subplot(gs11[1, 0])
    ax4 = fig.add_subplot(gs[2, :])

    ####################################################################################################################
    # Temporal traces and topoplot
    ####################################################################################################################

    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    r2_diff = model_r_squared - control_r_squared
    times = model_regression['times'][tmin:]

    # permutation r2
    permutation_folder = Path(permutation_folder)
    permutation_list = sorted(list(permutation_folder.glob('**/*perm*')))
    r2_permutations = np.empty((len(permutation_list), *model_r_squared.shape[1:]))  # already in t-values
    for p, perm in enumerate(permutation_list):
        with open(perm, 'rb') as file:
            r2_permutations[p, :, :] = pickle.load(file)['r2'][0, :, tmin:]

    r2_diff = np.swapaxes(r2_diff, 1, 2)  # reshape to (n_subjects, times, n_channels)
    r2_permutations = np.swapaxes(r2_permutations, 1, 2)  # reshape to (n_permut, times, n_channels)

    ch_adjacency, ch_names = find_ch_adjacency(model_regression['epochs_info'], 'mag')
    pval = 0.007  # arbitrary
    df = r2_diff.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    thresh = scipy.stats.t.ppf(1 - pval, df=df)  # T distribution, one-tailed
    print(thresh)
    t_obs, clusters, cluster_pv, H0, cluster_stats = permutation_cluster_test_adapted(r2_diff, r2_permutations,
                                                                                      threshold=thresh, tail=1,
                                                                                      adjacency=ch_adjacency, t_power=1)

    channels_indices = np.arange(r2_diff.shape[2])
    sorted_arg_p = np.argsort(cluster_pv)

    significant_clusters = np.array([clusters[arg] for arg in sorted_arg_p if cluster_pv[arg] < 0.05], dtype=object)
    n_significant_clusters = len(significant_clusters)
    print("Significant clusters:", n_significant_clusters)

    all_cluster_channels = []
    all_cluster_times = []
    for i, cluster in enumerate(significant_clusters):
        start_sample, end_sample = cluster[0][0], cluster[0][-1]
        print(start_sample, end_sample)

        cluster_channels = np.unique(cluster[1])
        all_cluster_channels += cluster_channels.tolist()
        cluster_times = np.unique(cluster[0])
        all_cluster_times += cluster_times.tolist()

    all_cluster_channels = np.unique(all_cluster_channels)
    all_cluster_times = np.unique(all_cluster_times)
    print('Number of significant channels:', len(all_cluster_channels))

    left_cluster_channels = []
    right_cluster_channels = []
    for chan in all_cluster_channels:
        x_pos = model_regression['epochs_info']['chs'][chan]['loc'][0]
        if x_pos < 0:
            left_cluster_channels.append(chan)
        else:
            right_cluster_channels.append(chan)
    print('Number of left channels:', len(left_cluster_channels))
    print('Number of right channels:', len(right_cluster_channels))

    # average channels that are part of at least one significant cluster
    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    r2_diff = model_r_squared - control_r_squared
    r2_diff_left = np.mean(r2_diff[:, left_cluster_channels, :], axis=1)
    r2_diff_right = np.mean(r2_diff[:, right_cluster_channels, :], axis=1)

    # # asymmetry test
    # left_minus_right = r2_diff_left - r2_diff_right
    # pval = 0.05  # arbitrary
    # df = left_minus_right.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    # thresh = scipy.stats.t.ppf(1 - pval/2, df=df)  # T distribution, two-tailed
    # t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(left_minus_right, threshold=thresh, tail=0,
    #                                                                      adjacency=None, t_power=1,
    #                                                                      n_permutations=10000, n_jobs=-1)
    #
    # print(cluster_pv)
    #
    # sorted_arg_p = np.argsort(cluster_pv)
    # asym_significant_clusters = np.array([clusters[arg] for arg in sorted_arg_p if cluster_pv[arg] < 0.05], dtype=object)
    #
    # asym_all_cluster_times = []
    # for i, cluster in enumerate(asym_significant_clusters):
    #     cluster_times = np.unique(cluster[0])
    #     asym_all_cluster_times += cluster_times.tolist()
    #
    # asym_all_cluster_times = np.unique(asym_all_cluster_times)

    # note density
    note_density = compute_note_density(
        epoch_path='/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/maestro01_itdheb-epo.fif')

    cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#878787', 'white', '#087e8b'])
    # cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#ff5a5f', 'white', '#087e8b'])
    im, cn = mne.viz.plot_topomap(np.mean(r2_diff, axis=(0, 2)),
                                  model_regression['epochs_info'], axes=ax5, show=False,
                                  mask=np.isin(channels_indices, all_cluster_channels),
                                  mask_params=dict(markersize=2.5, markerfacecolor='w', markeredgewidth=0.5,
                                                   markeredgecolor="k"),
                                  cmap=cmap1
                                  )
    clb = plt.colorbar(im, cax=ax6, orientation='horizontal', location='bottom')
    # ax5.set_title('$\Delta$R²')

    color = ['#878787', '#087e8b', '#ff5a5f', '#6859D9']
    # color = ['#878787', '#ff5a5f', '#e33b43', '#6859D9']

    control_r_squared = np.mean(control_r_squared[:, all_cluster_channels, :], axis=1)
    control_sem = np.std(control_r_squared, axis=0) / np.sqrt(control_r_squared.shape[0])
    control_r_squared = np.mean(control_r_squared, axis=0)
    line, = ax1.plot(times, control_r_squared, color=color[0], label='Control model')
    ax1.fill_between(times, control_r_squared - control_sem, control_r_squared + control_sem,
                                  color=color[0], alpha=0.5, linewidth=0)

    model_r_squared = np.mean(model_r_squared[:, all_cluster_channels, :], axis=1)
    model_sem = np.std(model_r_squared, axis=0) / np.sqrt(model_r_squared.shape[0])
    model_r_squared = np.mean(model_r_squared, axis=0)
    line, = ax1.plot(times, model_r_squared, color=color[1], label='Full model')
    ax1.fill_between(times, model_r_squared - model_sem, model_r_squared + model_sem, color=color[1],
                                  alpha=0.5, linewidth=0)

    cluster_sem = np.std(r2_diff_left, axis=0) / np.sqrt(r2_diff_left.shape[0])
    cluster_average = np.mean(r2_diff_left, axis=0)
    line, = ax2.plot(times, cluster_average, color=color[2],
                                  label=f'Left (N={len(left_cluster_channels)})')
    ax2.fill_between(times, cluster_average - cluster_sem, cluster_average + cluster_sem, color=color[2],
                                  alpha=0.5, linewidth=0)

    cluster_sem = np.std(r2_diff_right, axis=0) / np.sqrt(r2_diff_right.shape[0])
    cluster_average = np.mean(r2_diff_right, axis=0)
    line, = ax2.plot(times, cluster_average, color=color[3],
                                  label=f'Right (N={len(right_cluster_channels)})')
    ax2.fill_between(times, cluster_average - cluster_sem, cluster_average + cluster_sem, color=color[3],
                                  alpha=0.5, linewidth=0)
    print(all_cluster_times)
    masked_times = np.full(cluster_average.shape[0], True)  # non-significant times will be masked
    masked_times[all_cluster_times] = False  # store significant times
    mask = np.ma.masked_where(masked_times, ax1.get_ylim()[0] * np.ones(cluster_average.shape[0]))  # we mask what is not significant
    ax1.plot(times, mask, color='black', linewidth=2)

    # masked_times = np.full(cluster_average.shape[0], True)  # non-significant times will be masked
    # print(asym_all_cluster_times)
    # masked_times[asym_all_cluster_times] = False  # store significant times
    # mask = np.ma.masked_where(masked_times, ax2.get_ylim()[0] * np.ones(cluster_average.shape[0]))  # we mask what is not significant
    # ax2.plot(times, mask, color='black', linewidth=3)

    ax1.plot(times, note_density * ax1.get_ylim()[1], color=color[0], label='Note frequency', linestyle='--', lw=0.5, zorder=0)
    ax2.plot(times, note_density * ax2.get_ylim()[1], color=color[0], linestyle='--', lw=0.5, zorder=0)

    ax1.set_ylabel('R²')
    ax1.xaxis.set_ticklabels([])
    ax2.set_ylabel('$\Delta$R²')
    ax2.set_xlabel('Time (s)')
    for ax in [ax1, ax2]:
        ax.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
        ax.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
        ax.set_xlim([-0.5, 1])
        leg = ax.legend()
        # set the linewidth of each legend object
        for legobj in leg.legendHandles[:2]:
            legobj.set_linewidth(2)

        # fixed_golden_aspect_ratio(ax)

    ####################################################################################################################
    # Boxplot
    ####################################################################################################################

    best_channels = np.load(best_channels_path)

    # average channels that are part of at least one significant cluster
    # control_r_squared = control_regression['r2'][:, all_cluster_channels, tmin:]
    # model_r_squared = model_regression['r2'][:, all_cluster_channels, tmin:]
    control_r_squared = np.take_along_axis(control_regression['r2'][:, :, tmin:], best_channels[:, :, np.newaxis], axis=1)
    model_r_squared = np.take_along_axis(model_regression['r2'][:, :, tmin:], best_channels[:, :, np.newaxis], axis=1)
    # control_r_squared = control_regression['r2'][:, :, tmin:tmax]
    # model_r_squared = model_regression['r2'][:, :, tmin:tmax]
    # magenta_r_squared = magenta_regression['r2'][:, :, tmin:tmax]

    # max time on average of channels
    # mean_r2_control = np.max(np.mean(control_r_squared, axis=1), axis=1)
    # mean_r2_model = np.max(np.mean(model_r_squared, axis=1), axis=1)
    # mean_r2_magenta = np.max(np.mean(magenta_r_squared, axis=1), axis=1)

    # average of max time of each channel
    mean_r2_control = np.mean(np.max(control_r_squared, axis=2), axis=1)
    mean_r2_model = np.mean(np.max(model_r_squared, axis=2), axis=1)

    # average of time and channels
    # mean_r2_control = np.mean(control_r_squared, axis=(1, 2))
    # mean_r2_model = np.mean(model_r_squared, axis=(1, 2))
    # mean_r2_magenta = np.mean(magenta_r_squared, axis=(1, 2))

    # max time and channel
    # mean_r2_control = np.max(control_r_squared, axis=(1, 2))
    # mean_r2_model = np.max(model_r_squared, axis=(1, 2))
    # mean_r2_magenta = np.max(magenta_r_squared, axis=(1, 2))

    df = pd.DataFrame({'Control': mean_r2_control, 'Full model': mean_r2_model})

    color = ['#878787', '#087e8b', '#ff5a5f']
    # color = ['#878787', '#ff5a5f', '#087e8b']

    # sns.boxplot(data=df, ax=ax3, palette=color, fill=False, linecolor='black', width=0.6, linewidth=1, dodge=0, fliersize=0)
    sns.boxplot(data=df, ax=ax3, palette=color, width=0.6, linewidth=1, dodge=0, fliersize=0)

    # Connect points with lines
    for index, row in df.iterrows():
        jitter = np.random.normal(scale=0.07, size=1)
        x_vals = [0 + jitter, 1 + jitter]
        y_vals = [row['Control'], row['Full model']]
        ax3.scatter(x_vals[0], y_vals[0], s=1, marker='.', color='black', zorder=12)
        ax3.scatter(x_vals[1], y_vals[1], s=1, marker='.', color='black', zorder=12)
        if y_vals[1] >= y_vals[0]:
            ax3.plot(x_vals, y_vals, color=color[1], zorder=10, lw=0.5, alpha=0.6)
        else:
            ax3.plot(x_vals, y_vals, color=color[2], zorder=11, lw=0.5, alpha=0.9)

    ax3.set_ylabel('R²')
    ax3.set_ylim([ax3.get_ylim()[0], ax3.get_ylim()[1] * 1.15])

    # Create a custom colormap
    custom_lines = [Line2D([0], [0], color=color[1], lw=2),
                    Line2D([0], [0], color=color[2], lw=2)]
    ax3.legend(custom_lines, ['Improvement', 'No improvement'])
    # ax3.get_xaxis().set_visible(False)
    # ax3.spines[['bottom']].set_visible(False)
    # fixed_golden_aspect_ratio(ax3)

    # stat test
    print("Boxplot stats")
    print(scipy.stats.ttest_rel(mean_r2_model, mean_r2_control, alternative='two-sided'))

    ####################################################################################################################
    # Variance partitioning
    ####################################################################################################################

    # color_list = ['#087e8b', '#0b3954', '#B6ADFF', '#6859D9', '#ff5a5f', '#c81d25']
    color_list = ['#087e8b', '#0b3954', '#6859D9', '#c81d25']
    alpha = 0.8

    with open(os.path.join(vp_path, "regr_meg_paulrnn.pkl"), 'rb') as fp:
        reg_paulrnn = pickle.load(fp)
    with open(os.path.join(vp_path, "regr_meg_acoustic.pkl"), 'rb') as fp:
        reg_acoustic = pickle.load(fp)

    vp_data = {}
    for feat in vp_feature:
        with open(os.path.join(vp_path, "regr_meg_paulrnn_removed_" + feat + ".pkl"), 'rb') as fp:
            reg = pickle.load(fp)
            vp_data[feat] = reg

    times = reg_paulrnn['times'][tmin:]

    significant_channels = np.load(os.path.join(save_path, 'significant_channels.npy'))

    # retrieve results from full model
    data = {}
    y = reg_paulrnn['r2'][:, significant_channels, tmin:] - reg_acoustic['r2'][:, significant_channels, tmin:]
    # y = reg_paulrnn['r2'][:, :, tmin:] - reg_acoustic['r2'][:, :, tmin:]
    y = np.mean(y, axis=(0, 1))
    data['all'] = y

    # retrieve results for each unique feature
    for i, feat in enumerate(vp_feature):
        y = vp_data[feat]['r2'][:, significant_channels, tmin:] - reg_acoustic['r2'][:, significant_channels, tmin:]
        # y = vp_data[feat]['r2'][:, :, tmin:] - reg_acoustic['r2'][:, :, tmin:]
        y = np.mean(y, axis=(0, 1))
        data[feat] = data['all'] - y

    # retrieve shared variance
    data['shared'] = data['all'] - sum([data[feat] for feat in vp_feature])

    # merge features
    data[vp_feature[2]] = np.where(data[vp_feature[2]] > 0, data[vp_feature[2]], 0) + np.where(data[vp_feature[3]] > 0, data[vp_feature[3]], 0)
    data[vp_feature[4]] = np.where(data[vp_feature[4]] > 0, data[vp_feature[4]], 0) + np.where(data[vp_feature[5]] > 0, data[vp_feature[5]], 0)
    vp_feature = [vp_feature[i] for i in [0, 1, 2, 4]]
    feature_names = [feature_names[i] for i in [0, 1, 2, 4]]

    # fig
    suptitle = 'Variance Partitioning - all channels'
    y = np.where(data['shared'] > 0, data['shared'], 0)

    # axs.plot(times, y, color='black', label='shared')
    ax4.fill_between(times, 0, y, alpha=0.7, color='black', label='Shared', linewidth=0)
    # axs.fill_between(times, 0, y, alpha=alpha, color=color_list[0], label='Shared', linewidth=0)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Variance partitioning ($\Delta$R²)')
    ax4.set_xlim([-0.5, 1.0])
    ax4.set_ylim([0, 4.05e-4])

    for i, feat in enumerate(vp_feature):
        previous = copy.copy(y)
        y += np.where(data[feat] > 0, data[feat], 0)
        # axs.plot(times, y, label=feat)
        # axs.fill_between(times, previous, y, alpha=alpha, color = color_list[i+1], label=feature_names[i], linewidth=0)
        ax4.fill_between(times, previous, y, alpha=alpha, color=color_list[i], label=feature_names[i], linewidth=0)

    ax4.axvline(0, linestyle='--', linewidth=1, color='k', zorder=1)
    # ax4.axhline(0, linestyle='--', linewidth=1.5, color='k', zorder=1)

    # leg = ax4.legend()
    # # set the linewidth of each legend object
    # for legobj in leg.legendHandles[:2]:
    #     legobj.set_linewidth(2)
    alpha = [0.7] + [alpha] * len(color_list)
    color_list = ['black'] + color_list
    custom_lines = [Line2D([0], [0], color=color_list[i], alpha=alpha[i], lw=2) for i in range(len(feature_names) + 1)]
    ax4.legend(custom_lines, ['Shared'] + [feature_names[i] for i in range(len(feature_names))])

    # ax4.set_aspect(1, adjustable='box')
    # fixed_golden_aspect_ratio(ax4, golden_factor=1.5)


    fig.savefig(os.path.join(save_path, 'figure2.svg'))

    return


def figure_2_version_october_24(model_regression, control_regression, permutation_folder, vp_path, vp_feature, feature_names, best_channels_path, tmin=50, save_path='/home/mathieupvc/These/figures_papier'):

    # fig = plt.figure(layout="constrained", figsize=(3.60, 4.9))
    fig = plt.figure(figsize=(3.60, 4.9))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 0.4])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    ####################################################################################################################
    # Temporal traces and topoplot
    ####################################################################################################################

    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    r2_diff = model_r_squared - control_r_squared
    times = model_regression['times'][tmin:]

    # permutation r2
    permutation_folder = Path(permutation_folder)
    permutation_list = sorted(list(permutation_folder.glob('**/*perm*')))
    r2_permutations = np.empty((len(permutation_list), *model_r_squared.shape[1:]))  # already in t-values
    for p, perm in enumerate(permutation_list):
        with open(perm, 'rb') as file:
            r2_permutations[p, :, :] = pickle.load(file)['r2'][0, :, tmin:]

    r2_diff = np.swapaxes(r2_diff, 1, 2)  # reshape to (n_subjects, times, n_channels)
    r2_permutations = np.swapaxes(r2_permutations, 1, 2)  # reshape to (n_permut, times, n_channels)

    ch_adjacency, ch_names = find_ch_adjacency(model_regression['epochs_info'], 'mag')
    pval = 0.007  # arbitrary
    df = r2_diff.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    thresh = scipy.stats.t.ppf(1 - pval, df=df)  # T distribution, one-tailed
    print(thresh)
    t_obs, clusters, cluster_pv, H0, cluster_stats = permutation_cluster_test_adapted(r2_diff, r2_permutations,
                                                                                      threshold=thresh, tail=1,
                                                                                      adjacency=ch_adjacency, t_power=1)

    channels_indices = np.arange(r2_diff.shape[2])
    sorted_arg_p = np.argsort(cluster_pv)

    significant_clusters = np.array([clusters[arg] for arg in sorted_arg_p if cluster_pv[arg] < 0.05], dtype=object)
    n_significant_clusters = len(significant_clusters)
    print("Significant clusters:", n_significant_clusters)

    all_cluster_channels = []
    all_cluster_times = []
    for i, cluster in enumerate(significant_clusters):
        start_sample, end_sample = cluster[0][0], cluster[0][-1]
        print(start_sample, end_sample)

        cluster_channels = np.unique(cluster[1])
        all_cluster_channels += cluster_channels.tolist()
        cluster_times = np.unique(cluster[0])
        all_cluster_times += cluster_times.tolist()

    all_cluster_channels = np.unique(all_cluster_channels)
    all_cluster_times = np.unique(all_cluster_times)
    print('Number of significant channels:', len(all_cluster_channels))

    left_cluster_channels = []
    right_cluster_channels = []
    for chan in all_cluster_channels:
        x_pos = model_regression['epochs_info']['chs'][chan]['loc'][0]
        if x_pos < 0:
            left_cluster_channels.append(chan)
        else:
            right_cluster_channels.append(chan)
    print('Number of left channels:', len(left_cluster_channels))
    print('Number of right channels:', len(right_cluster_channels))

    # average channels that are part of at least one significant cluster
    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    r2_diff = model_r_squared - control_r_squared
    r2_diff_left = np.mean(r2_diff[:, left_cluster_channels, :], axis=1)
    r2_diff_right = np.mean(r2_diff[:, right_cluster_channels, :], axis=1)
    r2_diff = np.mean(r2_diff[:, :, :], axis=1)

    # # asymmetry test
    # left_minus_right = r2_diff_left - r2_diff_right
    # pval = 0.05  # arbitrary
    # df = left_minus_right.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    # thresh = scipy.stats.t.ppf(1 - pval/2, df=df)  # T distribution, two-tailed
    # t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(left_minus_right, threshold=thresh, tail=0,
    #                                                                      adjacency=None, t_power=1,
    #                                                                      n_permutations=10000, n_jobs=-1)
    #
    # print(cluster_pv)
    #
    # sorted_arg_p = np.argsort(cluster_pv)
    # asym_significant_clusters = np.array([clusters[arg] for arg in sorted_arg_p if cluster_pv[arg] < 0.05], dtype=object)
    #
    # asym_all_cluster_times = []
    # for i, cluster in enumerate(asym_significant_clusters):
    #     cluster_times = np.unique(cluster[0])
    #     asym_all_cluster_times += cluster_times.tolist()
    #
    # asym_all_cluster_times = np.unique(asym_all_cluster_times)

    # note density
    note_density = compute_note_density(
        epoch_path='/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/maestro01_itdheb-epo.fif')

    cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#878787', 'white', '#087e8b'])
    # cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#ff5a5f', 'white', '#087e8b'])

    color = ['#878787', '#087e8b', '#ff5a5f', '#6859D9']
    # color = ['#878787', '#ff5a5f', '#e33b43', '#6859D9']

    control_r_squared = np.mean(control_r_squared[:, all_cluster_channels, :], axis=1)
    control_sem = np.std(control_r_squared, axis=0) / np.sqrt(control_r_squared.shape[0])
    control_r_squared = np.mean(control_r_squared, axis=0)
    line, = ax2.plot(times, control_r_squared, color=color[0], label='Control model')
    ax2.fill_between(times, control_r_squared - control_sem, control_r_squared + control_sem,
                                  color=color[0], alpha=0.5, linewidth=0)

    model_r_squared = np.mean(model_r_squared[:, all_cluster_channels, :], axis=1)
    model_sem = np.std(model_r_squared, axis=0) / np.sqrt(model_r_squared.shape[0])
    model_r_squared = np.mean(model_r_squared, axis=0)
    line, = ax2.plot(times, model_r_squared, color=color[1], label='Full model')
    ax2.fill_between(times, model_r_squared - model_sem, model_r_squared + model_sem, color=color[1],
                                  alpha=0.5, linewidth=0)

    cluster_sem = np.std(r2_diff, axis=0) / np.sqrt(r2_diff.shape[0])
    cluster_average = np.mean(r2_diff, axis=0)
    line, = ax2.plot(times, cluster_average, color=color[2],
                                  label=f'Difference ($\Delta$R²)')
    ax2.fill_between(times, cluster_average - cluster_sem, cluster_average + cluster_sem, color=color[2],
                                  alpha=0.5, linewidth=0)

    print(all_cluster_times)
    masked_times = np.full(cluster_average.shape[0], True)  # non-significant times will be masked
    masked_times[all_cluster_times] = False  # store significant times
    mask = np.ma.masked_where(masked_times, ax2.get_ylim()[0] * np.ones(cluster_average.shape[0]))  # we mask what is not significant
    ax2.plot(times, mask, color='black', linewidth=2)

    # masked_times = np.full(cluster_average.shape[0], True)  # non-significant times will be masked
    # print(asym_all_cluster_times)
    # masked_times[asym_all_cluster_times] = False  # store significant times
    # mask = np.ma.masked_where(masked_times, ax2.get_ylim()[0] * np.ones(cluster_average.shape[0]))  # we mask what is not significant
    # ax2.plot(times, mask, color='black', linewidth=3)

    ax2.plot(times, note_density * ax2.get_ylim()[1], color=color[0], label='Note frequency', linestyle='--', lw=0.5, zorder=0)

    ax2.set_ylabel('R²')
    ax2.set_xlabel('Time (s)')
    for ax in [ax2]:
        ax.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
        ax.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
        ax.set_xlim([-0.5, 1])
        leg = ax.legend()
        # set the linewidth of each legend object
        for legobj in leg.legendHandles[:2]:
            legobj.set_linewidth(2)


    ##################### Remove ax 1
    ax1.set_axis_off()

    ####################################################################################################################
    # Variance partitioning
    ####################################################################################################################

    # color_list = ['#087e8b', '#0b3954', '#B6ADFF', '#6859D9', '#ff5a5f', '#c81d25']
    color_list = ['#087e8b', '#0b3954', '#6859D9', '#c81d25']
    alpha = 0.8

    with open(os.path.join(vp_path, "regr_meg_paulrnn.pkl"), 'rb') as fp:
        reg_paulrnn = pickle.load(fp)
    with open(os.path.join(vp_path, "regr_meg_acoustic.pkl"), 'rb') as fp:
        reg_acoustic = pickle.load(fp)

    vp_data = {}
    for feat in vp_feature:
        with open(os.path.join(vp_path, "regr_meg_paulrnn_removed_" + feat + ".pkl"), 'rb') as fp:
            reg = pickle.load(fp)
            vp_data[feat] = reg

    times = reg_paulrnn['times'][tmin:]

    significant_channels = np.load(os.path.join(save_path, 'significant_channels.npy'))

    # retrieve results from full model
    data = {}
    y = reg_paulrnn['r2'][:, significant_channels, tmin:] - reg_acoustic['r2'][:, significant_channels, tmin:]
    # y = reg_paulrnn['r2'][:, :, tmin:] - reg_acoustic['r2'][:, :, tmin:]
    y = np.mean(y, axis=(0, 1))
    data['all'] = y

    # retrieve results for each unique feature
    for i, feat in enumerate(vp_feature):
        y = vp_data[feat]['r2'][:, significant_channels, tmin:] - reg_acoustic['r2'][:, significant_channels, tmin:]
        # y = vp_data[feat]['r2'][:, :, tmin:] - reg_acoustic['r2'][:, :, tmin:]
        y = np.mean(y, axis=(0, 1))
        data[feat] = data['all'] - y

    # retrieve shared variance
    data['shared'] = data['all'] - sum([data[feat] for feat in vp_feature])

    # merge features
    data[vp_feature[2]] = np.where(data[vp_feature[2]] > 0, data[vp_feature[2]], 0) + np.where(data[vp_feature[3]] > 0, data[vp_feature[3]], 0)
    data[vp_feature[4]] = np.where(data[vp_feature[4]] > 0, data[vp_feature[4]], 0) + np.where(data[vp_feature[5]] > 0, data[vp_feature[5]], 0)
    vp_feature = [vp_feature[i] for i in [0, 1, 2, 4]]
    feature_names = [feature_names[i] for i in [0, 1, 2, 4]]

    # fig
    suptitle = 'Variance Partitioning - all channels'
    y = np.where(data['shared'] > 0, data['shared'], 0)

    # axs.plot(times, y, color='black', label='shared')
    ax3.fill_between(times, 0, y, alpha=0.7, color='black', label='Shared', linewidth=0)
    # axs.fill_between(times, 0, y, alpha=alpha, color=color_list[0], label='Shared', linewidth=0)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Variance partitioning ($\Delta$R²)')
    ax3.set_xlim([-0.5, 1.0])
    ax3.set_ylim([0, 4.05e-4])

    for i, feat in enumerate(vp_feature):
        previous = copy.copy(y)
        y += np.where(data[feat] > 0, data[feat], 0)
        # axs.plot(times, y, label=feat)
        # axs.fill_between(times, previous, y, alpha=alpha, color = color_list[i+1], label=feature_names[i], linewidth=0)
        ax3.fill_between(times, previous, y, alpha=alpha, color=color_list[i], label=feature_names[i], linewidth=0)

    ax3.axvline(0, linestyle='--', linewidth=1, color='k', zorder=1)
    # ax4.axhline(0, linestyle='--', linewidth=1.5, color='k', zorder=1)

    # leg = ax4.legend()
    # # set the linewidth of each legend object
    # for legobj in leg.legendHandles[:2]:
    #     legobj.set_linewidth(2)
    alpha = [0.7] + [alpha] * len(color_list)
    color_list = ['black'] + color_list
    custom_lines = [Line2D([0], [0], color=color_list[i], alpha=alpha[i], lw=2) for i in range(len(feature_names) + 1)]
    ax3.legend(custom_lines, ['Shared'] + [feature_names[i] for i in range(len(feature_names))])

    # ax4.set_aspect(1, adjustable='box')
    # fixed_golden_aspect_ratio(ax4, golden_factor=1.5)

    fig.savefig(os.path.join(save_path, 'figure2_october_24.svg'))

    ###############################################################
    # Plot effect size
    ###############################################################
    fig, ax = plt.subplots(1, 1)
    print(r2_diff.shape)
    # ax.plot(times, np.mean(r2_diff, axis=0), color=color[0])
    # ax.plot(times, np.std(r2_diff, axis=0), color=color[1])
    ax.plot(times, np.mean(r2_diff, axis=0) / np.std(r2_diff, axis=0), color=color[1])
    ax.axvline(times[np.argmax(np.mean(r2_diff, axis=0))], linewidth=0.5, color=color[2])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel("Cohen's d")
    ax.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax.set_xlim([-0.5, 1])
    ax.set_title('Effect size of delta R²')

    fig.savefig(os.path.join(save_path, 'delta_r_squared_effect_size.svg'))


    return


def figure_4(model_regression, control_regression, magenta_regression, best_channels_path, seeg_pickle, tmin=50, save_path='/home/mathieupvc/These/figures_papier'):

    # fig = plt.figure(layout="constrained", figsize=(7.20, 4.45*2))
    # fig = plt.figure(layout="constrained", figsize=(7.20, 4.45*1.4))
    # fig = plt.figure(layout="constrained", figsize=(7.20, 4.45*1))
    # fig = plt.figure(layout="constrained", figsize=(3.60, 4.45*1))
    fig = plt.figure(layout="constrained", figsize=(3.60, 4.530))
    # fig = plt.figure()
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 0.4]) #, hspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    # gs10 = gs[1, 0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.1)
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])
    # gs02 = gs[0, 2].subgridspec(2, 1, height_ratios=[1, 0.03])
    # ax5 = fig.add_subplot(gs02[0, 0])
    # ax6 = fig.add_subplot(gs02[1, 0])

    ####################################################################################################################
    # Temporal traces and topoplot
    ####################################################################################################################

    best_channels = np.load(best_channels_path)

    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    magenta_r_squared = magenta_regression['r2'][:, :, tmin:]
    r2_diff = model_r_squared - magenta_r_squared
    times = model_regression['times'][tmin:]

    r2_diff = np.swapaxes(r2_diff, 1, 2)  # reshape to (n_subjects, times, n_channels)

    ch_adjacency, ch_names = find_ch_adjacency(model_regression['epochs_info'], 'mag')
    pval = 0.009  # arbitrary
    df = r2_diff.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    thresh = scipy.stats.t.ppf(1 - pval, df=df)  # T distribution, one-tailed
    print(thresh)
    t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(r2_diff, threshold=thresh, tail=0,
                                                                         adjacency=ch_adjacency, t_power=1,
                                                                         n_permutations=10000, n_jobs=-1)

    print(cluster_pv)

    channels_indices = np.arange(r2_diff.shape[2])
    sorted_arg_p = np.argsort(cluster_pv)
    significant_clusters = np.array([clusters[arg] for arg in sorted_arg_p if cluster_pv[arg] < 0.05], dtype=object)

    n_significant_clusters = len(significant_clusters)
    print('Significant clusters:', n_significant_clusters)

    all_cluster_channels = []
    all_cluster_times = []
    for i, cluster in enumerate(significant_clusters):
        cluster_channels = np.unique(cluster[1])
        all_cluster_channels += cluster_channels.tolist()
        cluster_times = np.unique(cluster[0])
        all_cluster_times += cluster_times.tolist()

    all_cluster_channels = np.unique(all_cluster_channels)
    all_cluster_times = np.unique(all_cluster_times)
    print('Number of significant channels:', len(all_cluster_channels))

    # average channels that are part of at least one significant cluster
    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    magenta_r_squared = magenta_regression['r2'][:, :, tmin:]
    model_r_squared = model_r_squared - control_r_squared
    magenta_r_squared = magenta_r_squared - control_r_squared
    r2_diff = model_r_squared - magenta_r_squared

    # im, cn = mne.viz.plot_topomap(np.mean(r2_diff, axis=(0, 2)),
    #                               model_regression['epochs_info'], axes=ax5, show=False,
    #                               mask=np.isin(channels_indices, all_cluster_channels),
    #                               mask_params=dict(markersize=3, markerfacecolor='#087e8b', markeredgewidth=1,
    #                                                markeredgecolor="k")
    #                               )
    # clb = plt.colorbar(im, cax=ax6, orientation='horizontal', location='bottom')
    # axes_topo[0, 0].set_title('Poly RNN - Perf RNN')

    color = ['#087e8b', '#ff5a5f', '#e33b43', '#6859D9']

    # model_r_squared = np.mean(model_r_squared[:, all_cluster_channels, :], axis=1)
    model_r_squared = np.mean(np.take_along_axis(model_r_squared, best_channels[:, :, np.newaxis], axis=1), axis=1)
    model_sem = np.std(model_r_squared, axis=0) / np.sqrt(model_r_squared.shape[0])
    model_r_squared = np.mean(model_r_squared, axis=0)
    line, = ax1.plot(times, model_r_squared, color=color[0], label='Poly RNN')
    ax1.fill_between(times, model_r_squared - model_sem, model_r_squared + model_sem, color=color[0],
                               alpha=0.5, linewidth=0)

    # magenta_r_squared = np.mean(magenta_r_squared[:, all_cluster_channels, :], axis=1)
    magenta_r_squared = np.mean(np.take_along_axis(magenta_r_squared, best_channels[:, :, np.newaxis], axis=1), axis=1)
    magenta_sem = np.std(magenta_r_squared, axis=0) / np.sqrt(magenta_r_squared.shape[0])
    magenta_average = np.mean(magenta_r_squared, axis=0)
    line, = ax1.plot(times, magenta_average, color=color[1], label='Perf RNN')
    ax1.fill_between(times, magenta_average - magenta_sem, magenta_average + magenta_sem, color=color[1],
                               alpha=0.5, linewidth=0)

    r2_diff = np.mean(r2_diff[:, all_cluster_channels, :], axis=1)
    cluster_average = np.mean(r2_diff, axis=0)
    print(all_cluster_times)
    masked_times = np.full(cluster_average.shape[0], True)  # non-significant times will be masked
    masked_times[all_cluster_times] = False  # store significant times
    mask = np.ma.masked_where(masked_times, ax1.get_ylim()[0] * np.ones(cluster_average.shape[0]))  # we mask what is not significant
    ax1.plot(times, mask, color='black', linewidth=2)

    ax1.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax1.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax1.set_ylabel('$\Delta$R²')
    ax1.xaxis.set_ticklabels([])
    ax1.set_xlim([-0.5, 1])
    leg = ax1.legend()
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2)

    # fixed_golden_aspect_ratio(ax1)

    ####################################################################################################################
    # Boxplot meg
    ####################################################################################################################

    best_channels = np.load(best_channels_path)

    # average channels that are part of at least one significant cluster
    # control_r_squared = control_regression['r2'][:, significant_channels, tmin:tmax]
    # model_r_squared = model_regression['r2'][:, significant_channels, tmin:tmax]
    # magenta_r_squared = magenta_regression['r2'][:, significant_channels, tmin:tmax]
    control_r_squared = np.take_along_axis(control_regression['r2'][:, :, tmin:], best_channels[:, :, np.newaxis], axis=1)
    model_r_squared = np.take_along_axis(model_regression['r2'][:, :, tmin:], best_channels[:, :, np.newaxis], axis=1)
    magenta_r_squared = np.take_along_axis(magenta_regression['r2'][:, :, tmin:], best_channels[:, :, np.newaxis], axis=1)
    # control_r_squared = control_regression['r2'][:, :, tmin:tmax]
    # model_r_squared = model_regression['r2'][:, :, tmin:tmax]
    # magenta_r_squared = magenta_regression['r2'][:, :, tmin:tmax]

    # average of max time of each channel
    mean_r2_control = np.mean(np.max(control_r_squared, axis=2), axis=1)
    mean_r2_model = np.mean(np.max(model_r_squared, axis=2), axis=1)
    mean_r2_magenta = np.mean(np.max(magenta_r_squared, axis=2), axis=1)

    df = pd.DataFrame({'Perf RNN': mean_r2_magenta, 'Control': mean_r2_control, 'Poly RNN': mean_r2_model})

    color = ['#ff5a5f', '#878787', '#087e8b']

    sns.boxplot(data=df, ax=ax3, palette=color, fill=False, linecolor='black', width=0.6, linewidth=1, dodge=False, fliersize=0)

    # Connect points with lines
    for index, row in df.iterrows():
        jitter = np.random.normal(scale=0.07, size=1)
        x_vals = [0 + jitter, 1 + jitter, 2 + jitter]
        y_vals = [row['Perf RNN'], row['Control'], row['Poly RNN']]
        ax3.scatter(x_vals[0], y_vals[0], s=1, marker='.', color='black', zorder=12)
        ax3.scatter(x_vals[1], y_vals[1], s=1, marker='.', color='black', zorder=12)
        ax3.scatter(x_vals[2], y_vals[2], s=1, marker='.', color='black', zorder=12)
        if y_vals[1] >= y_vals[0]:
            ax3.plot(x_vals[:2], y_vals[:2], color=color[1], ls='--', linewidth=0.25, zorder=10)
        else:
            ax3.plot(x_vals[:2], y_vals[:2], color=color[0], linewidth=0.5, alpha=0.6, zorder=11)
        if y_vals[2] >= y_vals[1]:
            ax3.plot(x_vals[1:], y_vals[1:], color=color[2], linewidth=0.5, alpha=0.6, zorder=11)
        else:
            ax3.plot(x_vals[1:], y_vals[1:], color=color[1], ls='--', linewidth=0.25, zorder=10)

    ax3.set_ylabel('R²')
    ax3.set_ylim([ax3.get_ylim()[0], ax3.get_ylim()[1] * 1.1])

    # Create a custom colormap
    custom_lines = [Line2D([0], [0], color=color[2], lw=2),
                    Line2D([0], [0], color=color[1], ls='--', lw=0.5)]
    ax3.legend(custom_lines, ['Improvement', 'No improvement'])
    # ax3.get_xaxis().set_visible(False)

    # stats
    print('stats boxplot MEG')
    df['subject_id'] = range(len(df))
    stats_df = pd.melt(df, id_vars='subject_id', var_name='model', value_name="score")
    anova = AnovaRM(data=stats_df, depvar="score", subject='subject_id', within=['model'])
    anova = anova.fit()
    print(anova)
    print('Post-hocs')
    pd.set_option('display.max_columns', None)
    print(pairwise_tests(stats_df, 'score', within='model', subject='subject_id', padjust='bonf'))


    ####################################################################################################################
    # Boxplot seeg
    ####################################################################################################################

    with open(seeg_pickle, 'rb') as f:
        seeg_data = pickle.load(f)

    paulrnn = seeg_data['paulrnn'][:, tmin:]
    control = seeg_data['control'][:, tmin:]
    perfrnn = seeg_data['perfrnn'][:, tmin:]

    best_channels = np.array(seeg_data['mask_n100_20'])
    control_r_squared = control[best_channels, :]
    model_r_squared = paulrnn[best_channels, :]
    magenta_r_squared = perfrnn[best_channels, :]

    # average of max time of each channel
    mean_r2_control = np.mean(np.max(control_r_squared, axis=2), axis=1)
    mean_r2_model = np.mean(np.max(model_r_squared, axis=2), axis=1)
    mean_r2_magenta = np.mean(np.max(magenta_r_squared, axis=2), axis=1)

    # #remove outlier
    # mask = [True] * 10
    # mask[1] = False
    # mean_r2_control = mean_r2_control[mask]
    # mean_r2_model = mean_r2_model[mask]
    # mean_r2_magenta = mean_r2_magenta[mask]

    df = pd.DataFrame({'Perf RNN': mean_r2_magenta, 'Control': mean_r2_control, 'Poly RNN': mean_r2_model})

    color = ['#ff5a5f', '#878787', '#087e8b']

    sns.boxplot(data=df, ax=ax4, palette=color, fill=False, linecolor='black', width=0.6, linewidth=1, dodge=False,
                fliersize=0)

    # Connect points with lines
    for index, row in df.iterrows():
        jitter = np.random.normal(scale=0.07, size=1)
        x_vals = [0 + jitter, 1 + jitter, 2 + jitter]
        y_vals = [row['Perf RNN'], row['Control'], row['Poly RNN']]
        ax4.scatter(x_vals[0], y_vals[0], s=1, marker='.', color='black', zorder=12)
        ax4.scatter(x_vals[1], y_vals[1], s=1, marker='.', color='black', zorder=12)
        ax4.scatter(x_vals[2], y_vals[2], s=1, marker='.', color='black', zorder=12)
        if y_vals[1] >= y_vals[0]:
            ax4.plot(x_vals[:2], y_vals[:2], color=color[1], ls='--', linewidth=0.25, zorder=10)
        else:
            ax4.plot(x_vals[:2], y_vals[:2], color=color[0], linewidth=0.5, alpha=0.6, zorder=11)
        if y_vals[2] >= y_vals[1]:
            ax4.plot(x_vals[1:], y_vals[1:], color=color[2], linewidth=0.5, alpha=0.6, zorder=11)
        else:
            ax4.plot(x_vals[1:], y_vals[1:], color=color[1], ls='--', linewidth=0.25, zorder=10)

    ax4.set_ylabel('R²')
    ax4.set_ylim([ax4.get_ylim()[0], ax4.get_ylim()[1] * 1.3])
    import matplotlib.ticker as plticker
    loc = plticker.MultipleLocator(base=0.005)  # this locator puts ticks at regular intervals
    ax4.yaxis.set_major_locator(loc)

    # Create a custom colormap
    # custom_lines = [Line2D([0], [0], color=color[2], lw=3),
    #                 Line2D([0], [0], color=color[1], ls='--', lw=1)]
    # ax4.legend(custom_lines, ['Improvement', 'No improvement'])

    # stats
    print('stats boxplot sEEG')
    df['subject_id'] = range(len(df))
    stats_df = pd.melt(df, id_vars='subject_id', var_name='model', value_name="score")
    anova = AnovaRM(data=stats_df, depvar="score", subject='subject_id', within=['model'])
    anova = anova.fit()
    print(anova)
    print('Post-hocs')
    print(pairwise_tests(stats_df, 'score', within='model', subject='subject_id', padjust='bonf'))

    ####################################################################################################################
    # Temporal seeg
    ####################################################################################################################

    color = ['#087e8b', '#ff5a5f', '#e33b43', '#6859D9']

    model_r_squared = model_r_squared - control_r_squared
    magenta_r_squared = magenta_r_squared - control_r_squared

    print(model_r_squared.shape[0])

    model_r_squared = np.mean(model_r_squared, axis=1)  # average electrodes
    model_sem = np.std(model_r_squared, axis=0) / np.sqrt(model_r_squared.shape[0])
    model_r_squared = np.mean(model_r_squared, axis=0)
    line, = ax2.plot(times, model_r_squared, color=color[0], label='Poly RNN')
    ax2.fill_between(times, model_r_squared - model_sem, model_r_squared + model_sem, color=color[0], alpha=0.5, linewidth=0)

    magenta_r_squared = np.mean(magenta_r_squared, axis=1)
    magenta_sem = np.std(magenta_r_squared, axis=0) / np.sqrt(magenta_r_squared.shape[0])
    magenta_average = np.mean(magenta_r_squared, axis=0)
    line, = ax2.plot(times, magenta_average, color=color[1], label='Perf RNN')
    ax2.fill_between(times, magenta_average - magenta_sem, magenta_average + magenta_sem, color=color[1],
                     alpha=0.5, linewidth=0)

    ax2.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax2.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax2.set_ylabel('$\Delta$R²')
    ax2.set_xlabel('Time (s)')
    ax2.set_xlim([-0.5, 1])
    leg = ax2.legend()
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2)

    # fixed_golden_aspect_ratio(ax2)

    ####################################################################################################################
    # inset colorbar
    ####################################################################################################################

    ax5.set_axis_off()

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', ['#ff5a5f', 'white', '#087e8b'])
    cb_length = 0.3
    norm = plt.Normalize(vmin=-0.01, vmax=0.01)

    cax = ax5.inset_axes([(1 - cb_length) / 2, 0, cb_length, 0.05])  # left, bottom, width, height
    orientation = 'horizontal'
    # cmap = plt.get_cmap('hot')
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, orientation=orientation,
                                            # ticks=[0, 0.01, 0.02, 0.03, 0.04, 0.05],
                                            norm=norm)
    cbar.set_label(u'ΔR²')

    fig.savefig(os.path.join(save_path, 'figure_4.svg'))

    return


def figure_4_october_24(model_regression, control_regression, magenta_regression, best_channels_path, seeg_pickle, tmin=50, save_path='/home/mathieupvc/These/figures_papier'):

    fig = plt.figure(layout="constrained", figsize=(3.60, 4.9))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 0.4])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    ####################################################################################################################
    # Temporal traces and topoplot
    ####################################################################################################################

    best_channels = np.load(best_channels_path)

    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    magenta_r_squared = magenta_regression['r2'][:, :, tmin:]
    r2_diff = model_r_squared - magenta_r_squared
    times = model_regression['times'][tmin:]

    r2_diff = np.swapaxes(r2_diff, 1, 2)  # reshape to (n_subjects, times, n_channels)

    ch_adjacency, ch_names = find_ch_adjacency(model_regression['epochs_info'], 'mag')
    pval = 0.009  # arbitrary
    df = r2_diff.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    thresh = scipy.stats.t.ppf(1 - pval, df=df)  # T distribution, one-tailed
    print(thresh)
    t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(r2_diff, threshold=thresh, tail=0,
                                                                         adjacency=ch_adjacency, t_power=1,
                                                                         n_permutations=10000, n_jobs=-1)

    print(cluster_pv)

    channels_indices = np.arange(r2_diff.shape[2])
    sorted_arg_p = np.argsort(cluster_pv)
    significant_clusters = np.array([clusters[arg] for arg in sorted_arg_p if cluster_pv[arg] < 0.05], dtype=object)

    n_significant_clusters = len(significant_clusters)
    print('Significant clusters:', n_significant_clusters)

    all_cluster_channels = []
    all_cluster_times = []
    for i, cluster in enumerate(significant_clusters):
        cluster_channels = np.unique(cluster[1])
        all_cluster_channels += cluster_channels.tolist()
        cluster_times = np.unique(cluster[0])
        all_cluster_times += cluster_times.tolist()

    all_cluster_channels = np.unique(all_cluster_channels)
    all_cluster_times = np.unique(all_cluster_times)
    print('Number of significant channels:', len(all_cluster_channels))

    # average channels that are part of at least one significant cluster
    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    magenta_r_squared = magenta_regression['r2'][:, :, tmin:]
    model_r_squared = model_r_squared - control_r_squared
    magenta_r_squared = magenta_r_squared - control_r_squared
    r2_diff = model_r_squared - magenta_r_squared

    color = ['#087e8b', '#ff5a5f', '#e33b43', '#6859D9']

    # model_r_squared = np.mean(model_r_squared[:, all_cluster_channels, :], axis=1)
    model_r_squared = np.mean(np.take_along_axis(model_r_squared, best_channels[:, :, np.newaxis], axis=1), axis=1)
    model_sem = np.std(model_r_squared, axis=0) / np.sqrt(model_r_squared.shape[0])
    model_r_squared = np.mean(model_r_squared, axis=0)
    line, = ax3.plot(times, model_r_squared, color=color[0], label='Poly RNN')
    ax3.fill_between(times, model_r_squared - model_sem, model_r_squared + model_sem, color=color[0],
                               alpha=0.5, linewidth=0)

    # magenta_r_squared = np.mean(magenta_r_squared[:, all_cluster_channels, :], axis=1)
    magenta_r_squared = np.mean(np.take_along_axis(magenta_r_squared, best_channels[:, :, np.newaxis], axis=1), axis=1)
    magenta_sem = np.std(magenta_r_squared, axis=0) / np.sqrt(magenta_r_squared.shape[0])
    magenta_average = np.mean(magenta_r_squared, axis=0)
    line, = ax3.plot(times, magenta_average, color=color[1], label='Perf RNN')
    ax3.fill_between(times, magenta_average - magenta_sem, magenta_average + magenta_sem, color=color[1],
                               alpha=0.5, linewidth=0)

    r2_diff = np.mean(r2_diff[:, all_cluster_channels, :], axis=1)
    cluster_average = np.mean(r2_diff, axis=0)
    print(all_cluster_times)
    masked_times = np.full(cluster_average.shape[0], True)  # non-significant times will be masked
    masked_times[all_cluster_times] = False  # store significant times
    mask = np.ma.masked_where(masked_times, ax3.get_ylim()[0] * np.ones(cluster_average.shape[0]))  # we mask what is not significant
    ax3.plot(times, mask, color='black', linewidth=2)

    ax3.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax3.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax3.set_ylabel('$\Delta$R²')
    ax3.xaxis.set_ticklabels([])
    ax3.set_xlim([-0.5, 1])
    leg = ax3.legend()
    # set the linewidth of each legend object
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(2)

    ####################################################################################################################
    # Temporal seeg
    ####################################################################################################################

    with open(seeg_pickle, 'rb') as f:
        seeg_data = pickle.load(f)

    paulrnn = seeg_data['paulrnn'][:, tmin:]
    control = seeg_data['control'][:, tmin:]
    perfrnn = seeg_data['perfrnn'][:, tmin:]

    best_channels = np.array(seeg_data['mask_n100_20'])
    control_r_squared = control[best_channels, :]
    model_r_squared = paulrnn[best_channels, :]
    magenta_r_squared = perfrnn[best_channels, :]

    color = ['#087e8b', '#ff5a5f', '#e33b43', '#6859D9']

    model_r_squared = model_r_squared - control_r_squared
    magenta_r_squared = magenta_r_squared - control_r_squared

    print(model_r_squared.shape[0])

    model_r_squared = np.mean(model_r_squared, axis=1)  # average electrodes
    model_sem = np.std(model_r_squared, axis=0) / np.sqrt(model_r_squared.shape[0])
    model_r_squared = np.mean(model_r_squared, axis=0)
    line, = ax2.plot(times, model_r_squared, color=color[0], label='Poly RNN')
    ax2.fill_between(times, model_r_squared - model_sem, model_r_squared + model_sem, color=color[0], alpha=0.5, linewidth=0)

    magenta_r_squared = np.mean(magenta_r_squared, axis=1)
    magenta_sem = np.std(magenta_r_squared, axis=0) / np.sqrt(magenta_r_squared.shape[0])
    magenta_average = np.mean(magenta_r_squared, axis=0)
    line, = ax2.plot(times, magenta_average, color=color[1], label='Perf RNN')
    ax2.fill_between(times, magenta_average - magenta_sem, magenta_average + magenta_sem, color=color[1],
                     alpha=0.5, linewidth=0)

    ax2.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax2.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax2.set_ylabel('$\Delta$R²')
    ax2.set_xlabel('Time (s)')
    ax2.set_xlim([-0.5, 1])
    leg = ax2.legend()
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2)


    fig.savefig(os.path.join(save_path, 'figure_4_october_24.svg'))

    return


def figure_1(pickle_path, acoustic_feature_list, predictive_feature_list, n_notes_metadata, n_notes_data):
    """Save different figures for figure 1"""

    with open(pickle_path, 'rb') as f:
        seeg_example_chan = pickle.load(f)

    epochs = seeg_example_chan['epochs'][1]  # epochs of 'Bou_Ni_H9'
    epochs = epochs[epochs.metadata['stim_name'] == 'trial002_rep1_6.5Hz_clip10_piano0.wav']
    metadata = seeg_example_chan['epochs'][0].metadata  # the chosen clip is not in Bou_Ni
    metadata = metadata[metadata['stim_name'] == 'trial002_rep1_6.5Hz_clip9_piano0.wav']

    print(len(epochs))
    print(len(metadata))

    metadata = metadata[3:n_notes_metadata+3]  # drop 3 first notes
    print(metadata)
    acoustic = metadata[acoustic_feature_list]
    acoustic.loc[:, 'pitch'] = [pitch[0] for pitch in acoustic['pitch']]
    print(acoustic['pitch'])
    acoustic = ((acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())).apply(pd.to_numeric)
    print(acoustic)
    predictive = metadata[predictive_feature_list]
    predictive = ((predictive - predictive.min()) / (predictive.max() - predictive.min())).apply(pd.to_numeric)
    print(predictive)

    plt.switch_backend('TkAgg')

    fig, ax = plt.subplots(1, 1, dpi=300)
    cmap1 = LinearSegmentedColormap.from_list("mycmap", ['white', '#878787'])
    sns.heatmap(acoustic.to_numpy().T, ax=ax, cmap=cmap1, linewidths=1.5, linecolor='#878787', square=True)

    fig, ax = plt.subplots(1, 1, dpi=300)
    # cmap2 = LinearSegmentedColormap.from_list("mycmap2", ['white', '#087e8b'])
    cmap2 = LinearSegmentedColormap.from_list("mycmap2", ['white', '#e33b43'])
    sns.heatmap(predictive.to_numpy().T, ax=ax, cmap=cmap2, linewidths=1.5, linecolor='#e33b43', square=True)

    # R² plot
    times = seeg_example_chan['times'][50:]
    control_r2 = seeg_example_chan['ch_r2_control'][1][50:]
    paulrnn_r2 = seeg_example_chan['ch_r2_paulrnn'][1][50:]
    r2_diff = paulrnn_r2 - control_r2

    fig, ax = plt.subplots(1, 1, dpi=300)
    ax.plot(times, control_r2, color='#878787', label='A', linewidth=2)
    ax.plot(times, paulrnn_r2, color='#087e8b', label='AP', linewidth=2)
    ax.plot(times, r2_diff, color='#e33b43', label='AP - A', linewidth=2)  # '#6859d9ff'
    ax.set_ylabel('R²')
    ax.set_xlabel('Time (s)')
    ax.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax.set_xlim([-0.5, 1])
    leg = ax.legend()
    # # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3)

    # fixed_golden_aspect_ratio(ax)

    # betas
    betas = seeg_example_chan['ch_betas_paulrnn'][:, 1, 50:]
    features_list = (acoustic_feature_list + predictive_feature_list)[::-1]
    feature_indices = [seeg_example_chan['features_paulrnn'].index(feat) for feat in features_list]
    betas = betas[feature_indices]

    fig, ax = plt.subplots(1, 1, dpi=300)
    cmap3 = LinearSegmentedColormap.from_list('custom_cmap', ['#ff5a5f', 'white', '#087e8b'])
    # sns.heatmap(betas, ax=ax, cmap='bwr', linewidths=0, linecolor='#878787', square=False, yticklabels=features_list)
    # ax.imshow(betas, cmap='bwr')
    vmax = np.max(np.abs(betas))
    im = ax.pcolormesh(times, features_list, betas, shading='nearest', cmap=cmap3, vmin=-vmax, vmax=vmax)
    plt.colorbar(im)
    ax.set_xlim([-0.5, 1])
    ax.set_xlabel('Time (s)')
    ax.axvline(0, linestyle='--', linewidth=1, color='k')

    # betas temporal plot
    fig, ax = plt.subplots(1, 1, dpi=300)
    for f, feat in enumerate(features_list):
        if f < 3:
            color = '#087e8b'
        else:
            color = '#878787'
        im = ax.plot(times, betas[f] + 0.3*f, color=color, label=feat)
    ax.set_xlim([-0.5, 1])
    ax.set_xlabel('Time (s)')
    ax.axvline(0, linestyle='--', linewidth=1, color='k')

    # neural plots
    epochs = epochs[3+2:n_notes_data + 3+2]  # drop 3 first notes and 2 others (middle of notes picked for features)
    data = epochs.get_data()[:, 0, 50:]
    print(data.shape)
    data = (data - np.mean(data)) / np.std(data)

    matplotlib.rcParams['axes.linewidth'] = 2.5

    for i in range(data.shape[0]):
        fig, ax = plt.subplots(1, 1, dpi=300, figsize=(4, 4.45))
        ax.plot(times, data[i, :], color='#287FD0', linewidth=3.5)
        ax.axvline(0, linestyle='--', linewidth=2, color='k', zorder=0)
        ax.axhline(0, linestyle='--', linewidth=2, color='k', zorder=0)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        # fixed_golden_aspect_ratio(ax)

    plt.show()



def figure_3(seeg_pickle, vp_feature, feature_names, tmin=50, save_path='/home/mathieupvc/These/figures_papier'):

    # fig = plt.figure(layout="constrained", figsize=(3.60, 4.530))
    fig = plt.figure(layout="constrained", figsize=(3.60, 4.9))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 0.4]) #, hspace=0.1)
    ax3 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    # ax5 = fig.add_subplot(gs[2, :])

    fig2, ax2 = plt.subplots(1, 1, figsize=(1.4, 3))

    ####################################################################################################################
    # Boxplot seeg
    ####################################################################################################################

    with open(seeg_pickle, 'rb') as f:
        seeg_data = pickle.load(f)

    paulrnn = seeg_data['paulrnn'][:, tmin:]
    control = seeg_data['control'][:, tmin:]

    best_channels = np.array(seeg_data['mask_n100_20'])
    control_r_squared = control[best_channels, :]
    model_r_squared = paulrnn[best_channels, :]

    # average of max time of each channel
    mean_r2_control = np.mean(np.max(control_r_squared, axis=2), axis=1)
    mean_r2_model = np.mean(np.max(model_r_squared, axis=2), axis=1)

    # #remove outlier
    # mask = [True] * 10
    # mask[1] = False
    # mean_r2_control = mean_r2_control[mask]
    # mean_r2_model = mean_r2_model[mask]
    # mean_r2_magenta = mean_r2_magenta[mask]

    df = pd.DataFrame({'Control': mean_r2_control, 'Full model': mean_r2_model})

    color = ['#878787', '#087e8b', '#ff5a5f']

    # sns.boxplot(data=df, ax=ax2, palette=color, fill=False, linecolor='black', width=0.6, linewidth=1, dodge=0, fliersize=0)
    sns.boxplot(data=df, ax=ax2, palette=color, width=0.6, linewidth=1, dodge=0, fliersize=0)

    # Connect points with lines
    for index, row in df.iterrows():
        jitter = np.random.normal(scale=0.07, size=1)
        x_vals = [0 + jitter, 1 + jitter]
        y_vals = [row['Control'], row['Full model']]
        ax2.scatter(x_vals[0], y_vals[0], s=1, marker='.', color='black', zorder=12)
        ax2.scatter(x_vals[1], y_vals[1], s=1, marker='.', color='black', zorder=12)
        if y_vals[1] >= y_vals[0]:
            ax2.plot(x_vals, y_vals, color=color[1], zorder=10, lw=0.5, alpha=0.6)
        else:
            ax2.plot(x_vals, y_vals, color=color[2], zorder=11, lw=0.5, alpha=0.9)

    ax2.set_ylabel('R²')
    # import matplotlib.ticker as plticker
    # loc = plticker.MultipleLocator(base=0.005)  # this locator puts ticks at regular intervals
    # ax2.yaxis.set_major_locator(loc)

    # stat test
    print("Boxplot stats")
    print(scipy.stats.ttest_rel(mean_r2_model, mean_r2_control, alternative='two-sided'))

    ####################################################################################################################
    # Temporal seeg
    ####################################################################################################################

    paulrnn = seeg_data['paulrnn'][:, tmin:]
    control = seeg_data['control'][:, tmin:]
    print(control.shape)
    control_r_squared = control[seeg_data['ch_sign'], :]
    print(control_r_squared.shape)
    model_r_squared = paulrnn[seeg_data['ch_sign'], :]
    times = seeg_data['times'][tmin:]
    diff_r_squared = model_r_squared - control_r_squared

    color = ['#878787', '#087e8b', '#ff5a5f', '#6859D9']

    # control_r_squared = np.mean(control_r_squared, axis=1)
    control_sem = np.std(control_r_squared, axis=0) / np.sqrt(control_r_squared.shape[0])
    control_r_squared = np.mean(control_r_squared, axis=0)
    line, = ax1.plot(times, control_r_squared, color=color[0], label='Control model')
    ax1.fill_between(times, control_r_squared - control_sem, control_r_squared + control_sem,
                     color=color[0], alpha=0.5, linewidth=0)

    # model_r_squared = np.mean(model_r_squared, axis=1)
    model_sem = np.std(model_r_squared, axis=0) / np.sqrt(model_r_squared.shape[0])
    model_r_squared = np.mean(model_r_squared, axis=0)
    line, = ax1.plot(times, model_r_squared, color=color[1], label='Full model')
    ax1.fill_between(times, model_r_squared - model_sem, model_r_squared + model_sem, color=color[1],
                     alpha=0.5, linewidth=0)

    # diff_r_squared = np.mean(diff_r_squared, axis=1)
    diff_sem = np.std(diff_r_squared, axis=0) / np.sqrt(diff_r_squared.shape[0])
    diff_average = np.mean(diff_r_squared, axis=0)
    line, = ax1.plot(times, diff_average, color=color[2], label='Difference ($\Delta$R²)')
    ax1.fill_between(times, diff_average - diff_sem, diff_average + diff_sem, color=color[2],
                     alpha=0.5, linewidth=0)

    note_density = seeg_data['note_density'][tmin:]
    note_density[50] = np.nan
    note_density = note_density / np.nanmax(note_density)
    # print(note_density[50])
    ax1.plot(times, note_density * ax1.get_ylim()[1], color=color[0], label='Note frequency', linestyle='--', lw=0.5, zorder=0)
    # ax2.plot(times, note_density * ax2.get_ylim()[1], color=color[0], linestyle='--', lw=0.5, zorder=0)

    all_cluster_times = seeg_data['c_sign'][:-1, 0]  # remove last element which is one time sample
    all_cluster_times = np.concatenate([cluster for cluster in all_cluster_times])
    all_cluster_times = np.unique(all_cluster_times)
    print(all_cluster_times)
    all_cluster_times = all_cluster_times - tmin
    print(all_cluster_times)
    all_cluster_times = [time for time in all_cluster_times if time >= 0]
    print(all_cluster_times)
    masked_times = np.full(diff_average.shape[0], True)  # non-significant times will be masked
    masked_times[all_cluster_times] = False  # store significant times
    mask = np.ma.masked_where(masked_times, ax1.get_ylim()[0] * np.ones(diff_average.shape[0]))  # we mask what is not significant
    ax1.plot(times, mask, color='black', linewidth=2)

    ax1.set_ylabel('R²')
    # ax1.xaxis.set_ticklabels([])
    ax1.set_xlabel('Time (s)')
    # ax2.set_ylabel('$\Delta$R²')
    # ax2.set_xlabel('Time (s)')
    for ax in [ax1]:
        ax.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
        ax.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
        ax.set_xlim([-0.5, 1])
        leg = ax.legend()
        # set the linewidth of each legend object
        for legobj in leg.legendHandles[:2]:
            legobj.set_linewidth(2)

        # fixed_golden_aspect_ratio(ax)

    ####################################################################################################################
    # inset colorbar
    ####################################################################################################################

    ax3.set_axis_off()

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', ['white', '#087e8b'])
    cb_length = 0.3
    norm = plt.Normalize(vmin=0, vmax=0.05)

    cax = ax3.inset_axes([(1 - cb_length) / 2, 0, cb_length, 0.05])  # left, bottom, width, height
    orientation = 'horizontal'
    # cmap = plt.get_cmap('hot')
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, orientation=orientation,
                                            # ticks=[0, 0.01, 0.02, 0.03, 0.04, 0.05],
                                            norm=norm)
    cbar.set_label(u'ΔR²')

    ####################################################################################################################
    # Variance partitioning
    ####################################################################################################################

    # color_list = ['#087e8b', '#0b3954', '#B6ADFF', '#6859D9', '#ff5a5f', '#c81d25']
    color_list = ['#087e8b', '#0b3954', '#6859D9', '#ff5a5f']
    alpha = 0.8

    data = seeg_data['VP']
    times = seeg_data['times'][tmin:]

    # merge features
    data[vp_feature[2]] = np.where(data[vp_feature[2]] > 0, data[vp_feature[2]], 0) + np.where(data[vp_feature[3]] > 0,
                                                                                               data[vp_feature[3]], 0)
    data[vp_feature[4]] = np.where(data[vp_feature[4]] > 0, data[vp_feature[4]], 0) + np.where(data[vp_feature[5]] > 0,
                                                                                               data[vp_feature[5]], 0)
    vp_feature = [vp_feature[i] for i in [0, 1, 2, 4]]
    feature_names = [feature_names[i] for i in [0, 1, 2, 4]]

    # fig
    suptitle = 'Variance Partitioning - all channels'
    y = np.where(data['shared'] > 0, data['shared'], 0)
    y = y[tmin:]

    # axs.plot(times, y, color='black', label='shared')
    ax4.fill_between(times, 0, y, alpha=0.7, color='black', label='Shared', linewidth=0)
    # axs.fill_between(times, 0, y, alpha=alpha, color=color_list[0], label='Shared', linewidth=0)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Variance partitioning ($\Delta$R²)')
    ax4.set_xlim([-0.5, 1.0])
    ax4.set_ylim([0, 1.2e-2])

    for i, feat in enumerate(vp_feature):
        previous = copy.copy(y)
        y += np.where(data[feat] > 0, data[feat], 0)[tmin:]
        # axs.plot(times, y, label=feat)
        # axs.fill_between(times, previous, y, alpha=alpha, color = color_list[i+1], label=feature_names[i], linewidth=0)
        ax4.fill_between(times, previous, y, alpha=alpha, color=color_list[i], label=feature_names[i], linewidth=0)

    ax4.axvline(0, linestyle='--', linewidth=1, color='k', zorder=1)
    # ax4.axhline(0, linestyle='--', linewidth=1.5, color='k', zorder=1)

    # leg = ax4.legend()
    # # set the linewidth of each legend object
    # for legobj in leg.legendHandles[:2]:
    #     legobj.set_linewidth(2)
    alpha = [0.7] + [alpha] * len(color_list)
    color_list = ['black'] + color_list
    custom_lines = [Line2D([0], [0], color=color_list[i], alpha=alpha[i], lw=2) for i in range(len(feature_names) + 1)]
    ax4.legend(custom_lines, ['Shared'] + [feature_names[i] for i in range(len(feature_names))])


    fig.savefig(os.path.join(save_path, 'figure_3.svg'))
    fig2.savefig(os.path.join(save_path, 'boxplot_figure_3.svg'))

    return


def figure_liberto_data(eeg_pickle, threshold=0.0003, tmin=32, save_path='/home/mathieupvc/These/figures_papier'):

    # fig = plt.figure(layout="constrained", figsize=(3.60, 4.53))
    fig = plt.figure(layout="constrained", figsize=(3.60, 4.53))
    gs = GridSpec(2, 1, figure=fig) #, hspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    # fig2, ax3 = plt.subplots(1, 1, figsize=(1.4, 3))
    # fig2, ax3 = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 0.03]}, squeeze=False, figsize=(1.4, 1.6))
    fig2 = plt.figure(layout="constrained", figsize=(1.4, 1.6))
    gs2 = GridSpec(2, 3, height_ratios=[1, 0.03], figure=fig2)
    ax3 = fig2.add_subplot(gs2[0, :])
    ax4 = fig2.add_subplot(gs2[1, 0])

    ####################################################################################################################
    # Temporal delta R²
    ####################################################################################################################

    with open(eeg_pickle, 'rb') as f:
        eeg_data = pickle.load(f)

    times = eeg_data['times'][tmin:]
    model_r2 = np.array(eeg_data['A_env_idyom']['r2'])[:, :, tmin:]
    control_r2 = np.array(eeg_data['A_env']['r2'])[:, :, tmin:]

    model_r2 = np.mean(model_r2, axis=0)
    control_r2 = np.mean(control_r2, axis=0)
    r2_diff = model_r2 - control_r2

    color = ['#878787', '#087e8b', '#ff5a5f', '#6859D9']

    first_pos = True
    first_neg = True
    for chan in range(r2_diff.shape[0]):
        if np.max(r2_diff[chan, :]) > threshold:
            ax1.plot(times, r2_diff[chan, :], color=color[2], lw=0.3)
        else:
            ax1.plot(times, r2_diff[chan, :], color=color[0], lw=0.3)

    ax1.axhline(threshold, linestyle='--', linewidth=1, color=color[2], zorder=0)

    ax1.set_ylabel('$\Delta$R²')
    ax1.set_xlabel('Time (s)')
    ax1.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax1.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax1.set_xlim([-0.5, 1])
    leg = ax1.legend(['Selected channels', 'Other channels'])

    # set the linewidth of each legend object
    for legobj in leg.legendHandles[:2]:
        legobj.set_linewidth(2)

    # mask for boxplot
    mask = r2_diff >= threshold

    ####################################################################################################################
    # barplot
    ####################################################################################################################

    control_env = np.array(eeg_data['A_env']['r2'])[:, :, tmin:]
    control_all = np.array(eeg_data['A']['r2'])[:, :, tmin:]

    # compute deltas
    idyom_env = np.array(eeg_data['A_env_idyom']['r2'])[:, :, tmin:] - control_env
    idyom_all = np.array(eeg_data['A_idyom']['r2'])[:, :, tmin:] - control_all
    lstm_env = np.array(eeg_data['A_env_lstm']['r2'])[:, :, tmin:] - control_env
    lstm_all = np.array(eeg_data['A_lstm']['r2'])[:, :, tmin:] - control_all

    idyom_env = np.mean(idyom_env[:, mask], axis=1)
    idyom_all = np.mean(idyom_all[:, mask], axis=1)
    lstm_env = np.mean(lstm_env[:, mask], axis=1)
    lstm_all = np.mean(lstm_all[:, mask], axis=1)

    df = pd.DataFrame({'condition': np.concatenate(
        [['IDyOM + envelope'] * idyom_env.shape[0], ['PolyRNN + envelope'] * lstm_env.shape[0],
         ['IDyOM + all controls'] * idyom_all.shape[0], ['PolyRNN + all controls'] * lstm_all.shape[0]], axis=0),
                       'deltaR²': np.concatenate([idyom_env, lstm_env, idyom_all, lstm_all], axis=0)})

    # color = ['#878787', '#ff5a5f', '#878787', '#087e8b']
    color = ['#6859D9', '#087e8b', '#6859D9', '#087e8b']

    # sns.boxplot(data=df, x='condition', y='deltaR²', ax=ax2, palette=color, fill=False, linecolor='black', width=0.6, linewidth=2, dodge=0, fliersize=0)
    # sns.swarmplot(data=df, x='condition', y='deltaR²', ax=ax2, color='k', size=3, dodge=0)

    sns.barplot(data=df, x='condition', y='deltaR²', ax=ax2, errorbar='se', palette=color, fill=True, width=0.6, linewidth=2, dodge=0)
    # sns.barplot(data=df, x='condition', y='deltaR²', ax=ax2, palette=color, fill=True, linewidth=2, dodge=0)

    ax2.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)

    ax2.set(xlabel=None)
    ax2.set_ylabel('$\Delta$R²')
    leg = ax2.legend(['IDyOM', 'PolyRNN'])
    for legobj in leg.legendHandles[:2]:
        legobj.set_linewidth(2)

    pairs = []
    condition_list = ['IDyOM + envelope', 'PolyRNN + envelope', 'IDyOM + all controls', 'PolyRNN + all controls']
    for var1 in range(len(condition_list)):
        for var2 in range(var1 + 1, len(condition_list)):
            pairs.append((condition_list[var1], condition_list[var2]))
    annotator = Annotator(ax2, pairs, data=df, x='condition', y='deltaR²')
    annotator.configure(test='t-test_paired', comparisons_correction="Bonferroni", text_format='star', loc='inside', hide_non_significant=True)
    annotator.apply_and_annotate()

    print('1 sample ttest to show deltaR² > 0')
    for i, condition in enumerate(condition_list):
        print(condition)
        print(scipy.stats.ttest_1samp(df[df['condition']==condition]['deltaR²'], 0, alternative='two-sided'))

    ####################################################################################################################
    # topography
    ####################################################################################################################

    idyom_env = np.array(eeg_data['A_env_idyom']['r2'])[:, :, tmin:] - control_env
    selected_times = np.sum(mask, axis=0) > 0
    selected_chan = np.sum(mask, axis=1) > 0
    topo = np.mean(idyom_env[:, :, selected_times], axis=(0, 2))
    # topo = np.mean(idyom_env[:, :, :], axis=(0, 2))
    montage = eeg_data['montage']

    info = mne.create_info(montage.ch_names, 64, ch_types='eeg')
    print(info)
    info = info.set_montage(montage)
    print(info)

    # cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#ff5a5f', 'white', '#087e8b'])
    cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#878787', 'white', '#ff5a5f'])
    lim = np.max(np.abs(topo))
    im, cn = mne.viz.plot_topomap(topo, info, ch_type='eeg', axes=ax3, cmap=cmap1, vlim=(-lim, lim),
                                  mask=selected_chan,
                                  mask_params=dict(markersize=3, markerfacecolor='#ff5a5f', markeredgewidth=0.7, markeredgecolor="k"))


    # im, cn = mne.viz.plot_topomap(np.mean(r2_diff, axis=(0, 2)),
    #                               model_regression['epochs_info'], axes=axes_topo[1, 0], show=False,
    #                               mask=np.isin(channels_indices, all_cluster_channels),
    #                               mask_params=dict(markersize=1, markerfacecolor='white', markeredgewidth=0.2,
    #                                                markeredgecolor="k"),
    #                               cmap=cmap1,
    #                               vlim=(-lim, lim)
    #                               )
    clb = fig2.colorbar(im, cax=ax4, orientation='horizontal', location='bottom')
    # ax3.set_title('$\Delta$R²')


    fig.savefig(os.path.join(save_path, 'figure_liberto_data.svg'))
    fig2.savefig(os.path.join(save_path, 'topo_figure_liberto_data.svg'))

    return


def spatio_temporal_cluster_regression_cluster_averages(model_regression, control_regression, permutation_folder, features_to_plot, n_clusters_to_keep=5, tmin=50, save_path='/home/mathieupvc/These/figures_papier'):

    # color_list = ["#D73853","#0F9ED2","#db7f8e","#73D7DE"]
    # color_list = ["#D73853", "#007DAA", "#EC88B0", "#54C4AE"]
    color_list = ["#D73853", "#007DAA", "#EC88B0", "#54C4AE", "blue", "green", "gray", "black", "magenta", "yellow", "red", "cyan", "orange", "purple", "brown", "pink", "olive", "midnightblue", "lightsteelblue", "lime", "#D73853", "#007DAA", "#EC88B0", "#54C4AE"]

    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    r2_diff = model_r_squared - control_r_squared
    times = model_regression['times'][tmin:]
    betas = model_regression['betas'][:, :, :, tmin:]
    feature_indices = [model_regression['features'].index(feat) for feat in features_to_plot]

    # permutation r2
    permutation_folder = Path(permutation_folder)
    permutation_list = sorted(list(permutation_folder.glob('**/*perm*')))
    r2_permutations = np.empty((len(permutation_list), *model_r_squared.shape[1:]))  # already in t-values
    for p, perm in enumerate(permutation_list):
        with open(perm, 'rb') as file:
            r2_permutations[p, :, :] = pickle.load(file)['r2'][0, :, tmin:]

    r2_diff = np.swapaxes(r2_diff, 1, 2)  # reshape to (n_subjects, times, n_channels)
    r2_permutations = np.swapaxes(r2_permutations, 1, 2)  # reshape to (n_permut, times, n_channels)

    ch_adjacency, ch_names = find_ch_adjacency(model_regression['epochs_info'], 'mag')
    # pval = 0.01  # arbitrary
    # pval = 0.05  # arbitrary
    # pval = 0.03  # arbitrary
    pval = 0.007  # arbitrary
    df = r2_diff.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    thresh = scipy.stats.t.ppf(1 - pval, df=df)  # T distribution, one-tailed
    # t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(r2_diff,
    #                                                                      threshold=thresh,
    #                                                                      n_permutations=n_permut, tail=1,
    #                                                                      adjacency=ch_adjacency, n_jobs=-1,
    #                                                                      )
    t_obs, clusters, cluster_pv, H0, cluster_stats = permutation_cluster_test_adapted(r2_diff, r2_permutations, threshold=thresh, tail=1, adjacency=ch_adjacency, t_power=1)

    print(cluster_pv)
    print(cluster_stats.shape)

    channels_indices = np.arange(r2_diff.shape[2])
    sorted_arg_p = np.argsort(cluster_pv)
    if n_clusters_to_keep == 'all':
        significant_clusters = np.array([clusters[arg] for arg in sorted_arg_p])
    elif n_clusters_to_keep == 'p':
        # significant_clusters = np.array([cluster for (cluster, p_value) in zip(clusters, cluster_pv) if p_value < 0.05])
        significant_clusters = np.array([clusters[arg] for arg in sorted_arg_p if cluster_pv[arg] < 0.05], dtype=object)
    else:
        significant_clusters = np.array([clusters[arg] for arg in sorted_arg_p[:n_clusters_to_keep]])
    # significant_clusters = significant_clusters[np.argsort([cluster[0][0] for cluster in significant_clusters])]  # sort by time
    n_significant_clusters = len(significant_clusters)

    fig1, axes_topo = plt.subplots(3, n_significant_clusters, gridspec_kw={'height_ratios': [0.1, 1, 1]}, squeeze=False)
    fig2, axes_temporal = plt.subplots(2, 1, sharex=False)

    vlim = []
    for i, cluster in enumerate(significant_clusters):
        start_sample, end_sample = cluster[0][0], cluster[0][-1]
        vlim.append(np.max(np.abs(np.mean(r2_diff[:, start_sample:end_sample, :], axis=(0, 1)))))
    vlim = max(vlim)

    all_cluster_channels = []
    all_cluster_times = []
    for i, cluster in enumerate(significant_clusters):
        start_sample, end_sample = cluster[0][0], cluster[0][-1]
        print(start_sample, end_sample)
        masked_times = np.full(r2_diff.shape[1], True)  # non-significant times will be masked
        masked_times[start_sample:end_sample] = False  # store significant times
        im, cn = mne.viz.plot_topomap(np.mean(r2_diff[:, start_sample:end_sample, :], axis=(0, 1)),
                             model_regression['epochs_info'], axes=axes_topo[1, i], show=False,
                             mask=np.isin(channels_indices, cluster[1]), vlim=(-vlim, vlim)
                                    )
        if i == int(len(significant_clusters) / 2):
            gs = axes_topo[0, i].get_gridspec()
            # remove the underlying axes
            for ax in axes_topo[0, i-1:i+1]:
                ax.remove()
            axbig = fig1.add_subplot(gs[0, i-1:i+1])

            # clb = fig1.colorbar(im, cax=axes_topo[0, i-1:i+1], orientation='horizontal')
            clb = fig1.colorbar(im, cax=axbig, orientation='horizontal')
            clb.ax.set_title('Delta-r²')
        else:
            axes_topo[0, i].get_xaxis().set_visible(False)
            axes_topo[0, i].get_yaxis().set_visible(False)
            axes_topo[0, i].spines[['left', 'bottom']].set_visible(False)

        # axes_topo[1, i].set_xlabel(f'p = {round(cluster_pv[sorted_arg_p[i]], 6)}')
        axes_topo[1, i].set_xlabel(f'p = {round(cluster_pv[sorted_arg_p[i]], 6)} \n t = {round(cluster_stats[sorted_arg_p[i]], 2)}')

        cluster_channels = np.unique(cluster[1])
        all_cluster_channels += cluster_channels.tolist()
        cluster_times = np.unique(cluster[0])
        all_cluster_times += cluster_times.tolist()
        # cluster_average = np.mean(r2_diff[:, :, cluster_channels], axis=2)
        # cluster_average = np.mean(cluster_average, axis=0) / (np.std(cluster_average, axis=0) / np.sqrt(cluster_average.shape[0]))
        cluster_average = np.mean(r2_diff[:, :, cluster_channels.astype(int)], axis=(0, 2))
        # cluster_average[[i for i in range(cluster_average.shape[0]) if i not in cluster_times]] = np.nan
        # if i == 1 or i == 9:
        line, = axes_temporal[1].plot(times, cluster_average, color=color_list[i], linewidth=2, zorder=10-i)
        mask = np.ma.masked_where(masked_times, axes_temporal[1].get_ylim()[0] * np.ones(cluster_average.shape[0]))  # we mask what is not significant
        axes_temporal[1].plot(times, mask, color=line.get_color(), linewidth=3)

        axes_topo[2, i].axhline(0.99, color=line.get_color(), linewidth=5)
        axes_topo[2, i].get_xaxis().set_visible(False)
        axes_topo[2, i].get_yaxis().set_visible(False)
        axes_topo[2, i].spines[['left', 'bottom']].set_visible(False)

    axes_temporal[1].axvline(0, linestyle='--', color='k', zorder=0)
    axes_temporal[1].axhline(0, linestyle='--', color='k', zorder=0)
    axes_temporal[1].set_ylabel('Delta-r²')
    axes_temporal[1].set_xlabel('Time (s)')

    fixed_golden_aspect_ratio(axes_temporal[1])

    all_cluster_channels = np.unique(all_cluster_channels)
    all_cluster_times = np.unique(all_cluster_times)
    print('Number of significant channels:', len(all_cluster_channels))
    # average channels that are part of at least one significant cluster
    control_r_squared = np.mean(control_r_squared[:, all_cluster_channels, :], axis=1)
    model_r_squared = np.mean(model_r_squared[:, all_cluster_channels, :], axis=1)
    r2_diff = np.mean(r2_diff[:, :, all_cluster_channels], axis=2)
    lines = []

    control_sem = np.std(control_r_squared, axis=0) / np.sqrt(control_r_squared.shape[0])
    lines.append(axes_temporal[0].plot(times, np.mean(control_r_squared, axis=0), color='#1f77b4'))
    axes_temporal[0].fill_between(times, np.mean(control_r_squared, axis=0) - control_sem, np.mean(control_r_squared, axis=0) + control_sem, color='#1f77b4', alpha=0.5, linewidth=0)

    sem = np.std(model_r_squared, axis=0) / np.sqrt(model_r_squared.shape[0])
    lines.append(axes_temporal[0].plot(times, np.mean(model_r_squared, axis=0), color='#ff7f0e'))
    axes_temporal[0].fill_between(times, np.mean(model_r_squared, axis=0) - sem, np.mean(model_r_squared, axis=0) + sem, color='#ff7f0e', alpha=0.5, linewidth=0)

    diff_sem = np.std(r2_diff, axis=0) / np.sqrt(r2_diff.shape[0])
    lines.append(axes_temporal[0].plot(times, np.mean(r2_diff, axis=0), color='#2ca02c'))
    axes_temporal[0].fill_between(times, np.mean(r2_diff, axis=0) - diff_sem, np.mean(r2_diff, axis=0) + diff_sem, color='#2ca02c', alpha=0.5, linewidth=0)

    axes_temporal[0].legend([line[0] for line in lines], ['Control features only', 'All features', 'Difference'])
    # axes_temporal[0].legend([line[0] for line in lines], ['Performance RNN', 'Custom RNN', 'Difference'])

    axes_temporal[0].axvline(0, linestyle='--', color='k', zorder=0)
    axes_temporal[0].axhline(0, linestyle='--', color='k', zorder=0)
    axes_temporal[0].set_xlabel('Time (s)')
    # axes_temporal[0].get_xaxis().set_visible(False)
    # axes_temporal[0].spines[['bottom']].set_visible(False)
    axes_temporal[0].set_ylabel('r² score')

    fixed_golden_aspect_ratio(axes_temporal[0])

    ####################################################################################################################
    # Combine clusters
    ####################################################################################################################

    fig1, axes_topo = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.05]}, squeeze=False)
    fig2, axes_temporal = plt.subplots(2, 1, sharex=True)

    left_cluster_channels = []
    right_cluster_channels = []
    for chan in all_cluster_channels:
        x_pos = model_regression['epochs_info']['chs'][chan]['loc'][0]
        if x_pos < 0:
            left_cluster_channels.append(chan)
        else:
            right_cluster_channels.append(chan)
    print('Number of left channels:', len(left_cluster_channels))
    print('Number of right channels:', len(right_cluster_channels))

    # average channels that are part of at least one significant cluster
    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    r2_diff = model_r_squared - control_r_squared
    r2_diff_left = np.mean(r2_diff[:, left_cluster_channels, :], axis=1)
    r2_diff_right = np.mean(r2_diff[:, right_cluster_channels, :], axis=1)
    hemisphere_channels = [left_cluster_channels, right_cluster_channels]
    hemispheres = [r2_diff_left, r2_diff_right]

    # note density
    note_density = compute_note_density(epoch_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/maestro01_itdheb-epo.fif')

    im, cn = mne.viz.plot_topomap(np.mean(r2_diff, axis=(0, 2)),
                                  model_regression['epochs_info'], axes=axes_topo[0, 0], show=False,
                                  mask=np.isin(channels_indices, all_cluster_channels),
                                  mask_params=dict(markersize=12, markerfacecolor='w', markeredgewidth=3, markeredgecolor="k")
                                  )
    # clb = fig1.colorbar(im, cax=axes_topo[0, i-1:i+1], orientation='horizontal')
    clb = fig1.colorbar(im, cax=axes_topo[0, 1], orientation='vertical')
    clb.ax.set_title('$\Delta$R²')

    # color = ['#878787', '#F06E2D', '#713E5A']
    # color = ['#878787', '#FF9D00', '#772F1A']
    # color = ['#878787', '#F06E2D', '#772F1A']
    color = ['#878787', '#136290', '#e33b43', '#6859D9']
    color = ['#878787', '#087e8b', '#e33b43', '#6859D9']
    # color = ['#878787', '#e33b43', '#867ae1']
    # color = ['#878787', '#F06E2D', 'tab:green']
    # color = ['#878787', '#E33B43', 'tab:green', ]
    # color = ['#878787', 'tab:red', '#E33B43', ]

    control_r_squared = np.mean(control_r_squared[:, all_cluster_channels, :], axis=1)
    control_sem = np.std(control_r_squared, axis=0) / np.sqrt(control_r_squared.shape[0])
    control_r_squared = np.mean(control_r_squared, axis=0)
    line, = axes_temporal[0].plot(times, control_r_squared, color=color[0], label='Control model')
    axes_temporal[0].fill_between(times, control_r_squared - control_sem, control_r_squared + control_sem, color=color[0], alpha=0.5, linewidth=0)

    model_r_squared = np.mean(model_r_squared[:, all_cluster_channels, :], axis=1)
    model_sem = np.std(model_r_squared, axis=0) / np.sqrt(model_r_squared.shape[0])
    model_r_squared = np.mean(model_r_squared, axis=0)
    line, = axes_temporal[0].plot(times, model_r_squared, color=color[1], label='Full model')
    axes_temporal[0].fill_between(times, model_r_squared - model_sem, model_r_squared + model_sem, color=color[1], alpha=0.5, linewidth=0)

    cluster_sem = np.std(r2_diff_left, axis=0) / np.sqrt(r2_diff_left.shape[0])
    cluster_average = np.mean(r2_diff_left, axis=0)
    line, = axes_temporal[1].plot(times, cluster_average, color=color[2], label=f'Left (N={len(left_cluster_channels)})')
    axes_temporal[1].fill_between(times, cluster_average - cluster_sem, cluster_average + cluster_sem, color=color[2], alpha=0.5, linewidth=0)

    cluster_sem = np.std(r2_diff_right, axis=0) / np.sqrt(r2_diff_right.shape[0])
    cluster_average = np.mean(r2_diff_right, axis=0)
    line, = axes_temporal[1].plot(times, cluster_average, color=color[3], label=f'Right (N={len(right_cluster_channels)})')
    axes_temporal[1].fill_between(times, cluster_average - cluster_sem, cluster_average + cluster_sem, color=color[3], alpha=0.5, linewidth=0)
    print(all_cluster_times)
    masked_times = np.full(cluster_average.shape[0], True)  # non-significant times will be masked
    masked_times[all_cluster_times] = False  # store significant times
    mask = np.ma.masked_where(masked_times, axes_temporal[1].get_ylim()[0] * np.ones(cluster_average.shape[0]))  # we mask what is not significant
    axes_temporal[1].plot(times, mask, color='black', linewidth=6)

    axes_temporal[0].plot(times, note_density * axes_temporal[0].get_ylim()[1], color=color[0], label='Note density', linestyle='--', lw=1, zorder=0)
    axes_temporal[1].plot(times, note_density * axes_temporal[1].get_ylim()[1], color=color[0], linestyle='--', lw=1, zorder=0)

    # axes_topo[2, 0].axhline(0.99, color=line.get_color(), linewidth=5)
    # axes_topo[2, 0].get_xaxis().set_visible(False)
    # axes_topo[2, 0].get_yaxis().set_visible(False)
    # axes_topo[2, 0].spines[['left', 'bottom']].set_visible(False)
    fig1.tight_layout()

    axes_temporal[0].set_ylabel('R²')
    axes_temporal[1].set_ylabel('$\Delta$R²')
    axes_temporal[1].set_xlabel('Time (s)')
    for ax in axes_temporal:
        ax.axvline(0, linestyle='--', linewidth=2, color='k', zorder=0)
        ax.axhline(0, linestyle='--', linewidth=2, color='k', zorder=0)
        ax.set_xlim([-0.5, 1])
        leg = ax.legend()
        # set the linewidth of each legend object
        for legobj in leg.legendHandles[:2]:
            legobj.set_linewidth(6)

        fixed_golden_aspect_ratio(ax)

    fig1.savefig(os.path.join(save_path, 'significant_channels_r2_topomap_meg.svg'))
    fig2.savefig(os.path.join(save_path, 'significant_channels_r2_temporal_meg.svg'))

    ####################################################################################################################
    # Combine clusters left - right
    ####################################################################################################################

    fig1, axes_topo = plt.subplots(3, 2, gridspec_kw={'height_ratios': [0.1, 1, 1]}, squeeze=False)
    fig2, axes_temporal = plt.subplots(1, 1, sharex=False)

    left_cluster_channels = []
    right_cluster_channels = []
    for chan in all_cluster_channels:
        x_pos = model_regression['epochs_info']['chs'][chan]['loc'][0]
        if x_pos < 0:
            left_cluster_channels.append(chan)
        else:
            right_cluster_channels.append(chan)
    print('Number of left channels:', len(left_cluster_channels))
    print('Number of right channels:', len(right_cluster_channels))

    # average channels that are part of at least one significant cluster
    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    r2_diff = model_r_squared - control_r_squared
    r2_diff_left = np.mean(r2_diff[:, left_cluster_channels, :], axis=1)
    r2_diff_right = np.mean(r2_diff[:, right_cluster_channels, :], axis=1)
    hemisphere_channels = [left_cluster_channels, right_cluster_channels]
    hemispheres = [r2_diff_left, r2_diff_right]

    vlim = np.max(np.abs(np.mean(r2_diff, axis=(0, 2))))

    for i, hemisphere_r2 in enumerate(hemispheres):
        # masked_times = np.full(r2_diff.shape[1], True)  # non-significant times will be masked
        # masked_times[start_sample:end_sample] = False  # store significant times
        im, cn = mne.viz.plot_topomap(np.mean(r2_diff, axis=(0, 2)),
                                      model_regression['epochs_info'], axes=axes_topo[1, i], show=False,
                                      mask=np.isin(channels_indices, hemisphere_channels[i]), vlim=(-vlim, vlim)
                                      )
        if i == int(len(hemispheres) / 2):
            gs = axes_topo[0, i].get_gridspec()
            # remove the underlying axes
            for ax in axes_topo[0, i - 1:i + 1]:
                ax.remove()
            axbig = fig1.add_subplot(gs[0, i - 1:i + 1])

            # clb = fig1.colorbar(im, cax=axes_topo[0, i-1:i+1], orientation='horizontal')
            clb = fig1.colorbar(im, cax=axbig, orientation='horizontal')
            clb.ax.set_title('Delta-r²')
        else:
            axes_topo[0, i].get_xaxis().set_visible(False)
            axes_topo[0, i].get_yaxis().set_visible(False)
            axes_topo[0, i].spines[['left', 'bottom']].set_visible(False)

        cluster_average = np.mean(hemisphere_r2, axis=0)
        cluster_sem = np.std(hemisphere_r2, axis=0) / np.sqrt(hemisphere_r2.shape[0])
        line, = axes_temporal.plot(times, cluster_average, color=color_list[i], linewidth=2, zorder=10 - i)
        # axes_temporal.fill_between(times, cluster_average - cluster_sem, cluster_average + cluster_sem, color=color_list[i], alpha=0.5, linewidth=0)
        # mask = np.ma.masked_where(masked_times, axes_temporal[1].get_ylim()[0] * np.ones(
        #     cluster_average.shape[0]))  # we mask what is not significant
        # axes_temporal[1].plot(model_regression['times'], mask, color=line.get_color(), linewidth=3)

        axes_topo[2, i].axhline(0.99, color=line.get_color(), linewidth=5)
        axes_topo[2, i].get_xaxis().set_visible(False)
        axes_topo[2, i].get_yaxis().set_visible(False)
        axes_topo[2, i].spines[['left', 'bottom']].set_visible(False)

    axes_temporal.axvline(0, linestyle='--', color='k', zorder=0)
    axes_temporal.axhline(0, linestyle='--', color='k', zorder=0)
    axes_temporal.set_ylabel('Delta-r²')
    axes_temporal.set_xlabel('Time (s)')

    fixed_golden_aspect_ratio(axes_temporal)

    fig1.savefig(os.path.join(save_path, 'hemispheres_r2_topomap_meg.svg'))
    fig2.savefig(os.path.join(save_path, 'hemispheres_r2_temporal_meg.svg'))

    ####################################################################################################################
    # Betas
    ####################################################################################################################

    # reshape to (n_subjects, n_features, times, n_channels)
    betas = np.swapaxes(betas, 2, 3)
    for plot_i, f in enumerate(feature_indices):
        feature_betas = betas[:, f, :, :]

        fig3, axes_topo_betas = plt.subplots(3, n_significant_clusters, gridspec_kw={'height_ratios': [0.1, 1, 1]}, squeeze=False)
        fig4, axes_temporal_betas = plt.subplots(1, 1)

        vlim = []
        for i, cluster in enumerate(significant_clusters):
            start_sample, end_sample = cluster[0][0], cluster[0][-1]
            vlim.append(np.max(np.abs(np.mean(feature_betas[:, start_sample:end_sample, :], axis=(0, 1)))))
        vlim = max(vlim)

        for i, cluster in enumerate(significant_clusters):
            start_sample, end_sample = cluster[0][0], cluster[0][-1]
            masked_times = np.full(feature_betas.shape[1], True)  # non-significant times will be masked
            masked_times[start_sample:end_sample] = False  # store significant times
            im, cn = mne.viz.plot_topomap(np.mean(feature_betas[:, start_sample:end_sample, :], axis=(0, 1)),
                                 model_regression['epochs_info'], axes=axes_topo_betas[1, i], show=False,
                                 mask=np.isin(channels_indices, cluster[1]), vlim=(-vlim, vlim)
                                 )
            if i == int(len(significant_clusters) / 2):
                gs = axes_topo_betas[0, i].get_gridspec()
                # remove the underlying axes
                for ax in axes_topo_betas[0, i - 1:i + 1]:
                    ax.remove()
                axbig = fig3.add_subplot(gs[0, i - 1:i + 1])

                # clb = fig1.colorbar(im, cax=axes_topo[0, i-1:i+1], orientation='horizontal')
                clb = fig1.colorbar(im, cax=axbig, orientation='horizontal')
                clb.ax.set_title('Regression parameter (a.u.)')
            else:
                axes_topo_betas[0, i].get_xaxis().set_visible(False)
                axes_topo_betas[0, i].get_yaxis().set_visible(False)
                axes_topo_betas[0, i].spines[['left', 'bottom']].set_visible(False)

            cluster_channels = np.unique(cluster[1])
            cluster_average = np.mean(feature_betas[:, :, cluster_channels], axis=(0, 2))
            # if i == 1 or i == 9:
            line, = axes_temporal_betas.plot(times, cluster_average, color=color_list[i], linewidth=2, zorder=10 - i)
            mask = np.ma.masked_where(masked_times, axes_temporal_betas.get_ylim()[0] * np.ones(cluster_average.shape[0]))  # we mask what is not significant
            axes_temporal_betas.plot(times, mask, color=line.get_color(), linewidth=3)

            axes_topo_betas[2, i].axhline(0.99, color=line.get_color(), linewidth=5)
            axes_topo_betas[2, i].get_xaxis().set_visible(False)
            axes_topo_betas[2, i].get_yaxis().set_visible(False)
            axes_topo_betas[2, i].spines[['left', 'bottom']].set_visible(False)

        axes_temporal_betas.axvline(0, linestyle='--', color='k', zorder=0)
        axes_temporal_betas.axhline(0, linestyle='--', color='k', zorder=0)
        axes_temporal_betas.set_title(features_to_plot[plot_i])
        fig3.suptitle(features_to_plot[plot_i])
        axes_temporal_betas.set_ylabel('Regression parameter (a.u.)')
        axes_temporal_betas.set_xlabel('Time (s)')
        fixed_golden_aspect_ratio(axes_temporal_betas)

    # plt.show()

    np.save(os.path.join(save_path, 'significant_channels.npy'), all_cluster_channels)
    np.save(os.path.join(save_path, 'significant_clusters.npy'), significant_clusters)

    return


def model_comparison(model_regression, control_regression, magenta_regression, n_clusters_to_keep=5, tmin=50, save_path='/home/mathieupvc/These/figures_papier'):

    # color_list = ["#D73853","#0F9ED2","#db7f8e","#73D7DE"]
    # color_list = ["#D73853", "#007DAA", "#EC88B0", "#54C4AE"]
    color_list = ["#D73853", "#007DAA", "#EC88B0", "#54C4AE", "blue", "green", "gray", "black", "magenta", "yellow", "red", "cyan", "orange", "purple", "brown", "pink", "olive", "midnightblue", "lightsteelblue", "lime", "#D73853", "#007DAA", "#EC88B0", "#54C4AE"]

    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    magenta_r_squared = magenta_regression['r2'][:, :, tmin:]
    r2_diff = model_r_squared - magenta_r_squared
    times = model_regression['times'][tmin:]

    r2_diff = np.swapaxes(r2_diff, 1, 2)  # reshape to (n_subjects, times, n_channels)

    ch_adjacency, ch_names = find_ch_adjacency(model_regression['epochs_info'], 'mag')
    # pval = 0.05  # arbitrary
    # pval = 0.01  # arbitrary
    pval = 0.009  # arbitrary
    # pval = 0.006  # arbitrary
    df = r2_diff.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    thresh = scipy.stats.t.ppf(1 - pval, df=df)  # T distribution, one-tailed
    # t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(r2_diff,
    #                                                                      threshold=thresh,
    #                                                                      n_permutations=n_permut, tail=1,
    #                                                                      adjacency=ch_adjacency, n_jobs=-1,
    #                                                                      )
    t_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_1samp_test(r2_diff, threshold=thresh, tail=0, adjacency=ch_adjacency, t_power=1, n_permutations=10000, n_jobs=-1)

    print(cluster_pv)

    channels_indices = np.arange(r2_diff.shape[2])
    sorted_arg_p = np.argsort(cluster_pv)
    if n_clusters_to_keep == 'all':
        significant_clusters = np.array([clusters[arg] for arg in sorted_arg_p])
    elif n_clusters_to_keep == 'p':
        # significant_clusters = np.array([cluster for (cluster, p_value) in zip(clusters, cluster_pv) if p_value < 0.05])
        significant_clusters = np.array([clusters[arg] for arg in sorted_arg_p if cluster_pv[arg] < 0.05], dtype=object)
    else:
        significant_clusters = np.array([clusters[arg] for arg in sorted_arg_p[:n_clusters_to_keep]])
    # significant_clusters = significant_clusters[np.argsort([cluster[0][0] for cluster in significant_clusters])]  # sort by time
    n_significant_clusters = len(significant_clusters)

    all_cluster_channels = []
    all_cluster_times = []
    for i, cluster in enumerate(significant_clusters):
        cluster_channels = np.unique(cluster[1])
        all_cluster_channels += cluster_channels.tolist()
        cluster_times = np.unique(cluster[0])
        all_cluster_times += cluster_times.tolist()

    all_cluster_channels = np.unique(all_cluster_channels)
    all_cluster_times = np.unique(all_cluster_times)
    print('Number of significant channels:', len(all_cluster_channels))

    ####################################################################################################################
    # Combine clusters
    ####################################################################################################################

    fig1, axes_topo = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0.03, 1]}, squeeze=False, figsize=(0.9, 1.3))
    fig2, axes_temporal = plt.subplots(1, 1)

    # average channels that are part of at least one significant cluster
    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    magenta_r_squared = magenta_regression['r2'][:, :, tmin:]
    model_r_squared = model_r_squared - control_r_squared
    magenta_r_squared = magenta_r_squared - control_r_squared
    r2_diff = model_r_squared - magenta_r_squared

    # for t in range(0, 1000, 10):
    #     im, cn = mne.viz.plot_topomap(np.mean(r2_diff[:, :, t:t+10], axis=(0, 2)),
    #                                   model_regression['epochs_info'], show=False,
    #                                   mask=np.isin(channels_indices, all_cluster_channels),
    #                                   mask_params=dict(markersize=12, markerfacecolor='w', markeredgewidth=3,
    #                                                    markeredgecolor="k"), vlim=(-np.max(np.max(np.mean(r2_diff, axis=0))), np.max(np.max(np.mean(r2_diff, axis=0))))
    #                                   )
    #     plt.show()

    cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#ff5a5f', 'white', '#087e8b'])
    lim = np.max(np.abs(np.mean(r2_diff, axis=(0, 2))))
    im, cn = mne.viz.plot_topomap(np.mean(r2_diff, axis=(0, 2)),
                                  model_regression['epochs_info'], axes=axes_topo[1, 0], show=False,
                                  mask=np.isin(channels_indices, all_cluster_channels),
                                  mask_params=dict(markersize=1, markerfacecolor='white', markeredgewidth=0.2,
                                                   markeredgecolor="k"),
                                  cmap=cmap1,
                                  vlim=(-lim, lim)
                                  )
    clb = fig1.colorbar(im, cax=axes_topo[0, 0], orientation='horizontal', location='top')
    axes_topo[0, 0].set_title('Poly RNN - Perf RNN')

    color = ['#087e8b', '#ff5a5f', '#e33b43', '#6859D9']

    # control_r_squared = np.mean(control_r_squared[:, all_cluster_channels, :], axis=1)
    # control_sem = np.std(control_r_squared, axis=0) / np.sqrt(control_r_squared.shape[0])
    # control_r_squared = np.mean(control_r_squared, axis=0)
    # line, = axes_temporal.plot(times, control_r_squared, color=color[0], label='Control')
    # axes_temporal.fill_between(times, control_r_squared - control_sem, control_r_squared + control_sem, color=color[0],
    #                            alpha=0.5, linewidth=0)

    # best_channels = np.load('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_20channels_100ms.npy')

    model_r_squared = np.mean(model_r_squared[:, all_cluster_channels, :], axis=1)
    # model_r_squared = np.mean(np.take_along_axis(model_r_squared, best_channels[:, :, np.newaxis], axis=1), axis=1)
    model_sem = np.std(model_r_squared, axis=0) / np.sqrt(model_r_squared.shape[0])
    model_r_squared = np.mean(model_r_squared, axis=0)
    line, = axes_temporal.plot(times, model_r_squared, color=color[0], label='Poly RNN')
    axes_temporal.fill_between(times, model_r_squared - model_sem, model_r_squared + model_sem, color=color[0],
                               alpha=0.5, linewidth=0)

    magenta_r_squared = np.mean(magenta_r_squared[:, all_cluster_channels, :], axis=1)
    # magenta_r_squared = np.mean(np.take_along_axis(magenta_r_squared, best_channels[:, :, np.newaxis], axis=1), axis=1)
    magenta_sem = np.std(magenta_r_squared, axis=0) / np.sqrt(magenta_r_squared.shape[0])
    magenta_average = np.mean(magenta_r_squared, axis=0)
    line, = axes_temporal.plot(times, magenta_average, color=color[1], label='Perf RNN')
    axes_temporal.fill_between(times, magenta_average - magenta_sem, magenta_average + magenta_sem, color=color[1],
                               alpha=0.5, linewidth=0)

    r2_diff = np.mean(r2_diff[:, all_cluster_channels, :], axis=1)
    # cluster_sem = np.std(r2_diff, axis=0) / np.sqrt(r2_diff.shape[0])
    cluster_average = np.mean(r2_diff, axis=0)
    # line, = axes_temporal.plot(times, cluster_average, color=color[2], label='$\Delta$R²')
    # axes_temporal.fill_between(times, cluster_average - cluster_sem, cluster_average + cluster_sem, color=color[2],
    #                            alpha=0.5, linewidth=0)
    print(all_cluster_times)
    masked_times = np.full(cluster_average.shape[0], True)  # non-significant times will be masked
    masked_times[all_cluster_times] = False  # store significant times
    mask = np.ma.masked_where(masked_times, axes_temporal.get_ylim()[0] * np.ones(
        cluster_average.shape[0]))  # we mask what is not significant
    axes_temporal.plot(times, mask, color='black', linewidth=6)

    fig1.tight_layout()

    axes_temporal.axvline(0, linestyle='--', linewidth=2, color='k', zorder=0)
    axes_temporal.axhline(0, linestyle='--', linewidth=2, color='k', zorder=0)
    axes_temporal.set_ylabel('$\Delta$R²')
    axes_temporal.set_xlabel('Time (s)')
    leg = axes_temporal.legend()
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(6)

    fixed_golden_aspect_ratio(axes_temporal)


    ### plot the colorbar separately
    fig3, ax_cb = plt.subplots(1, 1)
    # ax_cb.set_axis_off()
    norm = plt.Normalize(vmin=-lim, vmax=lim)
    orientation = 'horizontal'
    # cmap = plt.get_cmap('hot')
    cbar = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap1, orientation=orientation,
                                            # ticks=[0, 0.01, 0.02, 0.03, 0.04, 0.05],
                                            norm=norm)
    # cbar.set_label(u'ΔR²')

    fig1.savefig(os.path.join(save_path, 'model_comparison_topomap_meg.svg'))
    fig2.savefig(os.path.join(save_path, 'model_comparison_temporal_meg.svg'))
    fig3.savefig(os.path.join(save_path, 'model_comparison_colorbar_meg.svg'))

    return


def variance_partitioning_plot(vp_path, vp_feature, feature_names, tmin=50, save_path='/home/mathieupvc/These/figures_papier'):

    # color_list = ['#087e8b', '#0b3954', '#B6ADFF', '#6859D9', '#ff5a5f', '#c81d25']
    color_list = ['#087e8b', '#0b3954', '#B6ADFF', '#c81d25']
    # color_list = ['#087e8b', '#0b3954', '#6859D9', '#c81d25']

    # cmap = plt.cm.get_cmap('magma')
    # cmap = sns.color_palette("Blues", as_cmap=True)
    # color_list = [cmap(x) for x in np.linspace(0.3, 0.9, len(vp_feature) + 1)]
    # color_list = [cmap(x) for x in np.linspace(0, 1, len(vp_feature) + 1)]
    # # color_list = sns.color_palette("Paired")
    # print(len(color_list))
    alpha = 0.8

    with open(os.path.join(vp_path, "regr_meg_paulrnn.pkl"), 'rb') as fp:
        reg_paulrnn = pickle.load(fp)
    with open(os.path.join(vp_path, "regr_meg_acoustic.pkl"), 'rb') as fp:
        reg_acoustic = pickle.load(fp)

    vp_data = {}
    for feat in vp_feature:
        with open(os.path.join(vp_path, "regr_meg_paulrnn_removed_" + feat + ".pkl"), 'rb') as fp:
            reg = pickle.load(fp)
            vp_data[feat] = reg

    times = reg_paulrnn['times'][tmin:]

    significant_channels = np.load(os.path.join(save_path, 'significant_channels.npy'))

    # retrieve results from full model
    data = {}
    y = reg_paulrnn['r2'][:, significant_channels, tmin:] - reg_acoustic['r2'][:, significant_channels, tmin:]
    # y = reg_paulrnn['r2'][:, :, tmin:] - reg_acoustic['r2'][:, :, tmin:]
    y = np.mean(y, axis=(0, 1))
    data['all'] = y

    # retrieve results for each unique feature
    for i, feat in enumerate(vp_feature):
        y = vp_data[feat]['r2'][:, significant_channels, tmin:] - reg_acoustic['r2'][:, significant_channels, tmin:]
        # y = vp_data[feat]['r2'][:, :, tmin:] - reg_acoustic['r2'][:, :, tmin:]
        y = np.mean(y, axis=(0, 1))
        data[feat] = data['all'] - y

    # retrieve shared variance
    data['shared'] = data['all'] - sum([data[feat] for feat in vp_feature])

    # merge features
    data[vp_feature[2]] = np.where(data[vp_feature[2]] > 0, data[vp_feature[2]], 0) + np.where(data[vp_feature[3]] > 0, data[vp_feature[3]], 0)
    data[vp_feature[4]] = np.where(data[vp_feature[4]] > 0, data[vp_feature[4]], 0) + np.where(data[vp_feature[5]] > 0, data[vp_feature[5]], 0)
    vp_feature = [vp_feature[i] for i in [0, 1, 2, 4]]
    feature_names = [feature_names[i] for i in [0, 1, 2, 4]]

    # fig
    suptitle = 'Variance Partitioning - all channels'
    # fig, axs = plt.subplots(figsize=(20, 12), dpi=300)
    fig, axs = plt.subplots()
    # fig.suptitle(suptitle)
    y = np.where(data['shared'] > 0, data['shared'], 0)

    # axs.plot(times, y, color='black', label='shared')
    axs.fill_between(times, 0, y, alpha=0.7, color='black', label='Shared', linewidth=0)
    # axs.fill_between(times, 0, y, alpha=alpha, color=color_list[0], label='Shared', linewidth=0)
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('R²')
    # axs.set_ylim([-0.002, 0.012])

    for i, feat in enumerate(vp_feature):
        previous = copy.copy(y)
        y += np.where(data[feat] > 0, data[feat], 0)
        # axs.plot(times, y, label=feat)
        # axs.fill_between(times, previous, y, alpha=alpha, color = color_list[i+1], label=feature_names[i], linewidth=0)
        axs.fill_between(times, previous, y, alpha=alpha, color=color_list[i], label=feature_names[i], linewidth=0)

    axs.axvline(0, linestyle='--', linewidth=2, color='k', zorder=1)
    axs.axhline(0, linestyle='--', linewidth=2, color='k', zorder=1)

    fixed_golden_aspect_ratio(axs)
    plt.legend()
    # plt.show()

    fig.savefig(os.path.join(save_path, 'variance_partitioning_meg.svg'))

    return


def histogram_r2_seeg(seeg_pickle, tmin=50, save_path='/home/mathieupvc/These/figures_papier'):

    with open(seeg_pickle, 'rb') as f:
        seeg_data = pickle.load(f)

    paulrnn = seeg_data['paulrnn'][:, tmin:]
    control = seeg_data['control'][:, tmin:]
    perfrnn = seeg_data['perfrnn'][:, tmin:]

    best_channels = np.array(seeg_data['mask_n100_20'])
    control_r_squared = control[best_channels, :]
    model_r_squared = paulrnn[best_channels, :]
    magenta_r_squared = perfrnn[best_channels, :]

    # average of max time of each channel
    control_r_squared = np.max(control_r_squared, axis=2).reshape(-1)
    model_r_squared = np.max(model_r_squared, axis=2).reshape(-1)
    magenta_r_squared = np.max(magenta_r_squared, axis=2).reshape(-1)

    diff_measure = 'max'  # mean or max
    scores = model_r_squared - magenta_r_squared

    ## fig
    # fig1, ax = plt.subplots(1, 1, figsize=(3, 2.5))
    fig1, ax = plt.subplots(1, 1, figsize=(1.9, 1.70))

    # Histogram
    n, _, _ = ax.hist(scores, bins=60, color='darkgrey')
    # Density Plot
    density = scipy.stats.gaussian_kde(scores)
    xlim = [-0.5e-2, 2.5e-2]
    xs = np.linspace(*xlim, 200)
    density.covariance_factor = lambda: .20
    density._compute_covariance()
    d = density(xs)
    ax.plot(xs, d * (np.max(n) / np.max(d)), color='black', lw=0.5)
    # ax.fill_between(xs, 0, d * (np.max(n) / np.max(d)), color='white', alpha=0.2, linewidth=0)

    if diff_measure == 'mean':
        ax.set_xlim([-0.0004, 0.0004])
    elif diff_measure == 'max':
        # ax.set_xlim([np.min(scores) - 0.003, 0.021])
        ax.set_xlim([-0.003, 0.021])

    ylim = [0, np.max(n) * 1.05]
    ax.vlines(0, ylim[0], ylim[1], linestyles='--', colors='black', lw=0.5)
    ax.vlines(np.mean(scores), ylim[0], ylim[1], linestyles='-', colors='#6859D9')
    ax.text(np.mean(scores), np.max(n) * 0.9, '   Mean Difference', fontsize='5', fontweight='bold', color='#6859D9', alpha=1)
    ax.set_ylim(ylim)

    if diff_measure == 'mean':
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_xlabel(u'ΔR²')  # r'$R²_{CustomRNN} - R²_{PerfRNN}$'
    # ax.xaxis.set_label_coords(1, -0.075)
    # arrows_y = -ylim[1] * 0.09
    arrows_y = 20
    arrow_props = dict(facecolor='black', edgecolor='#087e8b', arrowstyle='->', linewidth=1, mutation_aspect=1)
    ax.annotate('', xy=(0.0155, arrows_y), xytext=(0.0125, arrows_y), arrowprops=arrow_props, annotation_clip=False)
    ax.text(0.0155, arrows_y, 'Poly RNN', fontsize='5', color='#087e8b', horizontalalignment='left', verticalalignment='center', fontweight='bold')
    arrow_props = dict(facecolor='black', edgecolor='#ff5a5f', arrowstyle='->', linewidth=1, mutation_aspect=1)
    ax.annotate('', xy=(0.0095, arrows_y), xytext=(0.0125, arrows_y), arrowprops=arrow_props, annotation_clip=False)
    ax.text(0.0095, arrows_y, 'Perf RNN', fontsize='5', color='#ff5a5f', horizontalalignment='right', verticalalignment='center', fontweight='bold')

    ax.set_ylabel('Channel count')

    print('Mean:', np.mean(scores))
    print("Stat test")
    print(scipy.stats.ttest_1samp(scores, 0, alternative='two-sided'))

    plt.savefig(os.path.join(save_path, 'histogram_r2_seeg.svg'))

    return


def r2_boxplot(model_regression, control_regression, magenta_regression, tmin=100, tmax=150, save_path='/home/mathieupvc/These/figures_papier', best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_10channels_100ms.npy'):

    significant_channels = np.load(os.path.join(save_path, 'significant_channels.npy'))
    best_channels = np.load(best_channels_path)

    # fig1, axes_boxplot = plt.subplots(1, 1, squeeze=False, figsize=(8, 11))
    fig1, axes_boxplot = plt.subplots(1, 1, squeeze=False, figsize=(8, 11))

    # average channels that are part of at least one significant cluster
    # control_r_squared = control_regression['r2'][:, significant_channels, tmin:tmax]
    # model_r_squared = model_regression['r2'][:, significant_channels, tmin:tmax]
    # magenta_r_squared = magenta_regression['r2'][:, significant_channels, tmin:tmax]
    control_r_squared = np.take_along_axis(control_regression['r2'][:, :, tmin:tmax], best_channels[:, :, np.newaxis], axis=1)
    model_r_squared = np.take_along_axis(model_regression['r2'][:, :, tmin:tmax], best_channels[:, :, np.newaxis], axis=1)
    magenta_r_squared = np.take_along_axis(magenta_regression['r2'][:, :, tmin:tmax], best_channels[:, :, np.newaxis], axis=1)
    # control_r_squared = control_regression['r2'][:, :, tmin:tmax]
    # model_r_squared = model_regression['r2'][:, :, tmin:tmax]
    # magenta_r_squared = magenta_regression['r2'][:, :, tmin:tmax]

    # max time on average of channels
    # mean_r2_control = np.max(np.mean(control_r_squared, axis=1), axis=1)
    # mean_r2_model = np.max(np.mean(model_r_squared, axis=1), axis=1)
    # mean_r2_magenta = np.max(np.mean(magenta_r_squared, axis=1), axis=1)

    # average of max time of each channel
    mean_r2_control = np.mean(np.max(control_r_squared, axis=2), axis=1)
    mean_r2_model = np.mean(np.max(model_r_squared, axis=2), axis=1)
    mean_r2_magenta = np.mean(np.max(magenta_r_squared, axis=2), axis=1)

    # average of time and channels
    # mean_r2_control = np.mean(control_r_squared, axis=(1, 2))
    # mean_r2_model = np.mean(model_r_squared, axis=(1, 2))
    # mean_r2_magenta = np.mean(magenta_r_squared, axis=(1, 2))

    # max time and channel
    # mean_r2_control = np.max(control_r_squared, axis=(1, 2))
    # mean_r2_model = np.max(model_r_squared, axis=(1, 2))
    # mean_r2_magenta = np.max(magenta_r_squared, axis=(1, 2))

    df = pd.DataFrame({'Perf RNN': mean_r2_magenta, 'Control': mean_r2_control, 'Poly RNN': mean_r2_model})

    # color = ['#136290', '#878787', '#e33b43']
    # color = ['#136290', '#878787', '#087e8b']
    color = ['#ff5a5f', '#878787', '#087e8b']
    # ['#878787', '#087e8b', '#e33b43', '#6859D9'] '#ff5a5f'

    sns.boxplot(data=df, ax=axes_boxplot[0, 0], palette=color, fill=False, linecolor='black', width=0.6, linewidth=6, dodge=False, fliersize=0)

    # Connect points with lines
    for index, row in df.iterrows():
        jitter = np.random.normal(scale=0.03, size=1)
        x_vals = [0 + jitter, 1 + jitter, 2 + jitter]
        y_vals = [row['Perf RNN'], row['Control'], row['Poly RNN']]
        plt.scatter(x_vals[0], y_vals[0], linewidth=5, marker='.', color='black', zorder=12)
        plt.scatter(x_vals[1], y_vals[1], linewidth=5, marker='.', color='black', zorder=12)
        plt.scatter(x_vals[2], y_vals[2], linewidth=5, marker='.', color='black', zorder=12)
        if y_vals[1] >= y_vals[0]:
            plt.plot(x_vals[:2], y_vals[:2], color=color[1], ls='--', linewidth=2, zorder=10)
        else:
            plt.plot(x_vals[:2], y_vals[:2], color=color[0], linewidth=2, zorder=11)
        if y_vals[2] >= y_vals[1]:
            plt.plot(x_vals[1:], y_vals[1:], color=color[2], linewidth=2, zorder=11)
        else:
            plt.plot(x_vals[1:], y_vals[1:], color=color[1], ls='--', linewidth=2, zorder=10)

    axes_boxplot[0, 0].set_ylabel('R²')
    axes_boxplot[0, 0].set_ylim([axes_boxplot[0, 0].get_ylim()[0], axes_boxplot[0, 0].get_ylim()[1] * 1.1])

    # Create a custom colormap
    custom_lines = [Line2D([0], [0], color=color[2], lw=6),
                    Line2D([0], [0], color=color[1], ls='--', lw=2)]
    axes_boxplot[0, 0].legend(custom_lines, ['Predictive features improvement', 'No improvement'])

    # plt.show()
    fig1.savefig(os.path.join(save_path, 'boxplot.svg'))

    ####################################################################################################################
    # Poly RNN only
    ####################################################################################################################

    fig1, axes_boxplot = plt.subplots(1, 1, squeeze=False, figsize=(8, 11))

    # average channels that are part of at least one significant cluster
    control_r_squared = control_regression['r2'][:, significant_channels, tmin:tmax]
    model_r_squared = model_regression['r2'][:, significant_channels, tmin:tmax]
    magenta_r_squared = magenta_regression['r2'][:, significant_channels, tmin:tmax]
    # control_r_squared = np.take_along_axis(control_regression['r2'][:, :, tmin:tmax], best_channels[:, :, np.newaxis], axis=1)
    # model_r_squared = np.take_along_axis(model_regression['r2'][:, :, tmin:tmax], best_channels[:, :, np.newaxis], axis=1)
    # magenta_r_squared = np.take_along_axis(magenta_regression['r2'][:, :, tmin:tmax], best_channels[:, :, np.newaxis], axis=1)
    # control_r_squared = control_regression['r2'][:, :, tmin:tmax]
    # model_r_squared = model_regression['r2'][:, :, tmin:tmax]
    # magenta_r_squared = magenta_regression['r2'][:, :, tmin:tmax]

    # max time on average of channels
    # mean_r2_control = np.max(np.mean(control_r_squared, axis=1), axis=1)
    # mean_r2_model = np.max(np.mean(model_r_squared, axis=1), axis=1)
    # mean_r2_magenta = np.max(np.mean(magenta_r_squared, axis=1), axis=1)

    # average of max time of each channel
    mean_r2_control = np.mean(np.max(control_r_squared, axis=2), axis=1)
    mean_r2_model = np.mean(np.max(model_r_squared, axis=2), axis=1)

    # average of time and channels
    # mean_r2_control = np.mean(control_r_squared, axis=(1, 2))
    # mean_r2_model = np.mean(model_r_squared, axis=(1, 2))
    # mean_r2_magenta = np.mean(magenta_r_squared, axis=(1, 2))

    # max time and channel
    # mean_r2_control = np.max(control_r_squared, axis=(1, 2))
    # mean_r2_model = np.max(model_r_squared, axis=(1, 2))
    # mean_r2_magenta = np.max(magenta_r_squared, axis=(1, 2))

    df = pd.DataFrame({'Control': mean_r2_control, 'Full model': mean_r2_model})

    color = ['#878787', '#087e8b']

    sns.boxplot(data=df, ax=axes_boxplot[0, 0], palette=color, fill=False, linecolor='black', width=0.6, linewidth=6,
                dodge=False, fliersize=0)

    # Connect points with lines
    for index, row in df.iterrows():
        jitter = np.random.normal(scale=0.03, size=1)
        x_vals = [0 + jitter, 1 + jitter]
        y_vals = [row['Control'], row['Full model']]
        plt.scatter(x_vals[0], y_vals[0], linewidth=5, marker='.', color='black', zorder=12)
        plt.scatter(x_vals[1], y_vals[1], linewidth=5, marker='.', color='black', zorder=12)
        if y_vals[1] >= y_vals[0]:
            plt.plot(x_vals, y_vals, color=color[1], linewidth=2, zorder=11)
        else:
            plt.plot(x_vals, y_vals, color=color[0], ls='--', linewidth=2, zorder=10)

    axes_boxplot[0, 0].set_ylabel('R²')
    axes_boxplot[0, 0].set_ylim([axes_boxplot[0, 0].get_ylim()[0], axes_boxplot[0, 0].get_ylim()[1] * 1.1])

    # Create a custom colormap
    custom_lines = [Line2D([0], [0], color=color[1], lw=6),
                    Line2D([0], [0], color=color[0], ls='--', lw=2)]
    axes_boxplot[0, 0].legend(custom_lines, ['Predictive features improvement', 'No improvement'])

    # plt.show()
    fig1.savefig(os.path.join(save_path, 'boxplot_polyrnn.svg'))

    return


def betas_figure(model_regression, features_to_plot, save_path, tmin=-0.5, averaging_method='mean'):
    """
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times)
    :param epochs: used for times and info
    :param feature_list: must be of length n_features
    :param tmin:
    :param channel_subset: set it to True if the regression was not done on all channels. Topomaps cannot be plotted in that case.
    :return:
    """

    betas = model_regression['betas']

    if averaging_method == 'mean':
        def averaging_fct(params): return np.mean(params, axis=0)
    elif averaging_method == 't-values':
        def averaging_fct(params): return np.mean(params, axis=0) / (np.std(params, axis=0) / np.sqrt(params.shape[0]))

    feature_indices = [model_regression['features'].index(feat) for feat in features_to_plot]

    for plot_i, f in enumerate(feature_indices):
        feature_betas = betas[:, f, :, :]
        fig, axes = plt.subplots(1, 1)  # +1 row for the temporal plot

        evoked_betas = mne.EvokedArray(averaging_fct(feature_betas), info=model_regression['epochs_info'], tmin=model_regression['times'][0])
        evoked_betas.plot(show=False, axes=axes, unit=False, zorder='std', gfp=True, spatial_colors=True)
        axes.set_xlim([tmin, None])
        axes.axvline(0, linestyle='--', color='k', zorder=0)

        # remove nave added by mne
        for child in axes.get_children():
            if isinstance(child, matplotlib.text.Annotation):
                if child.xy == (0, 1):
                    child.remove()

        fixed_golden_aspect_ratio(axes)

        axes.set_title(features_to_plot[plot_i])
        if averaging_method == 'mean':
            axes.set_ylabel('Regression parameter (a.u.)')
        elif averaging_method == 't-values':
            axes.set_ylabel('t-values')

        fig.savefig(os.path.join(save_path, f'betas_{features_to_plot[plot_i]}_meg.svg'))

    return


def average_betas_figure(model_regression, features_to_plot, feature_names, save_path, tmin=50):
    """
    :param betas: regression betas of shape (n_subjects, n_features, n_channels, times)
    :param epochs: used for times and info
    :param feature_list: must be of length n_features
    :param tmin:
    :param channel_subset: set it to True if the regression was not done on all channels. Topomaps cannot be plotted in that case.
    :return:
    """

    color = ['#087e8b', '#0b3954', '#B6ADFF', '#6859D9', '#ff5a5f', '#c81d25']
    # color = ['#087e8b', '#0b3954', '#6859D9', '#ff5a5f', '#c81d25']

    betas = model_regression['betas'][:, :, :, tmin:]
    times = model_regression['times'][tmin:]

    # manage channel polarities based on surprise
    polarity_weights = (np.mean(betas[:, model_regression['features'].index('surprise_mean_paulrnn'), :, 65:80], axis=(0, 2)) >= 0).astype(int)
    polarity_weights[polarity_weights == 0] = -1
    betas = betas * polarity_weights[np.newaxis, np.newaxis, :, np.newaxis]

    feature_indices = [model_regression['features'].index(feat) for feat in features_to_plot]

    significant_channels = np.load(os.path.join(save_path, 'significant_channels.npy'))

    # note density
    note_density = compute_note_density(epoch_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/maestro01_itdheb-epo.fif')

    fig, axes = plt.subplots(1, 1)  # +1 row for the temporal plot
    for i, f in enumerate(feature_indices):
        feature_betas = betas[:, f, :, :]
        feature_betas = np.mean(feature_betas[:, significant_channels, :], axis=1)

        sem = np.std(feature_betas, axis=0) / np.sqrt(feature_betas.shape[0])
        axes.plot(times, np.mean(feature_betas, axis=0), color=color[i], label=feature_names[i])
        axes.fill_between(times, np.mean(feature_betas, axis=0) - sem, np.mean(feature_betas, axis=0) + sem, color=color[i], alpha=0.5, linewidth=0)

    axes.plot(times, note_density * axes.get_ylim()[1], color='#878787', label='Note density', linestyle='--', lw=0.5, zorder=0)

    # axes.set_xlim([tmin, None])
    axes.axvline(0, linestyle='--', color='k', zorder=0)
    axes.axhline(0, linestyle='--', color='k', zorder=0)
    axes.set_ylabel('Regression parameter (a.u.)')
    axes.set_xlabel('Times (s)')
    leg = axes.legend()

    # set the linewidth of each legend object
    for legobj in leg.legendHandles[:-1]:
        legobj.set_linewidth(4)

    fixed_golden_aspect_ratio(axes)

    fig.savefig(os.path.join(save_path, f'average_betas_meg.svg'))

    return


def erp_by_quantile(erp_path, quantile_list, tmin=50, best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_20channels_100ms.npy', save_path='/home/mathieupvc/These/figures_papier'):

    # color = ['#b6adff', '#bb89c9', '#bf6592', '#c4415c', '#c81d25']
    color = ['#087E8B', '#237A86', '#3F7681', '#5A727C', '#766E77', '#916A73', '#AD666E', '#C86269', '#E45E64', '#FF5A5F']

    best_channels = np.load(best_channels_path)

    times = np.linspace(-0.5, 1, 151)
    erp = np.load(erp_path)[:, :, tmin:, :]
    print(erp.shape)

    fig, axes = plt.subplots(1, 1)
    for q, quantile in enumerate(quantile_list):
        quantile_erp = erp[:, :, :, q]
        quantile_erp = np.take_along_axis(quantile_erp, best_channels[:, :, np.newaxis], axis=1)

        # quantile_erp = np.mean(np.abs(quantile_erp), axis=(0, 1))
        quantile_erp = np.mean(np.sqrt(np.mean(quantile_erp**2, axis=1)), axis=0)
        # quantile_erp = quantile_erp - np.mean(quantile_erp[:50])

        axes.plot(times, quantile_erp, color=color[q], label=f'Quantile {str(q+1)}')
    axes.legend()
    axes.axvline(0, linestyle='--', linewidth=1.5, color='k', zorder=0)
    # axes.axhline(0, linestyle='--', linewidth=1.5, color='k', zorder=0)
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('ERP RMS')

    fig.savefig(os.path.join(save_path, f'erp_by_quantile_meg.svg'))

    return


def note_density_scale(seeg_pickle, tmin=50):
    plt.switch_backend('TkAgg')

    with open(seeg_pickle, 'rb') as f:
        seeg_data = pickle.load(f)

    times = seeg_data['times'][tmin:]

    color = ['#878787', '#087e8b', '#e33b43', '#6859D9']

    fig, ax1 = plt.subplots(1, 1)

    note_density = seeg_data['note_density'][tmin:]
    note_density[50] = np.nan
    ax1.plot(times, note_density, color=color[0], label='Note frequency', linestyle='--', lw=0.5, zorder=0)

    ax1.set_ylabel('Note frequency')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('sEEG')
    import matplotlib.ticker as plticker
    loc = plticker.MultipleLocator(base=0.05)  # this locator puts ticks at regular intervals
    ax1.yaxis.set_major_locator(loc)
    ax1.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax1.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax1.set_xlim([-0.5, 1])
    leg = ax1.legend()

    # MEG
    note_density = compute_note_density(
        epoch_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/maestro01_itdheb-epo.fif',
        normalize=False)

    fig, ax1 = plt.subplots(1, 1)

    ax1.plot(times, note_density, color=color[0], label='Note frequency', linestyle='--', lw=0.5, zorder=0)

    ax1.set_ylabel('Note frequency')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('MEG')
    loc = plticker.MultipleLocator(base=0.05)  # this locator puts ticks at regular intervals
    ax1.yaxis.set_major_locator(loc)
    ax1.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax1.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax1.set_xlim([-0.5, 1])
    leg = ax1.legend()

    plt.show()


def poster_paul_31_05(model_regression, control_regression, permutation_folder, vp_path, vp_feature, feature_names, seeg_pickle, tmin=50, save_path='/home/mathieupvc/These/figures_papier'):

    fig = plt.figure(layout="constrained", figsize=(4.8, 3.26))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1]) #, hspace=0.1, wspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax4 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    ####################################################################################################################
    # Temporal traces and topoplot
    ####################################################################################################################

    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    r2_diff = model_r_squared - control_r_squared
    times = model_regression['times'][tmin:]

    # permutation r2
    permutation_folder = Path(permutation_folder)
    permutation_list = sorted(list(permutation_folder.glob('**/*perm*')))
    r2_permutations = np.empty((len(permutation_list), *model_r_squared.shape[1:]))  # already in t-values
    for p, perm in enumerate(permutation_list):
        with open(perm, 'rb') as file:
            r2_permutations[p, :, :] = pickle.load(file)['r2'][0, :, tmin:]

    r2_diff = np.swapaxes(r2_diff, 1, 2)  # reshape to (n_subjects, times, n_channels)
    r2_permutations = np.swapaxes(r2_permutations, 1, 2)  # reshape to (n_permut, times, n_channels)

    ch_adjacency, ch_names = find_ch_adjacency(model_regression['epochs_info'], 'mag')
    pval = 0.007  # arbitrary
    df = r2_diff.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    thresh = scipy.stats.t.ppf(1 - pval, df=df)  # T distribution, one-tailed
    print(thresh)
    t_obs, clusters, cluster_pv, H0, cluster_stats = permutation_cluster_test_adapted(r2_diff, r2_permutations,
                                                                                      threshold=thresh, tail=1,
                                                                                      adjacency=ch_adjacency, t_power=1)

    channels_indices = np.arange(r2_diff.shape[2])
    sorted_arg_p = np.argsort(cluster_pv)

    significant_clusters = np.array([clusters[arg] for arg in sorted_arg_p if cluster_pv[arg] < 0.05], dtype=object)
    n_significant_clusters = len(significant_clusters)
    print("Significant clusters:", n_significant_clusters)

    all_cluster_channels = []
    all_cluster_times = []
    for i, cluster in enumerate(significant_clusters):
        start_sample, end_sample = cluster[0][0], cluster[0][-1]
        print(start_sample, end_sample)

        cluster_channels = np.unique(cluster[1])
        all_cluster_channels += cluster_channels.tolist()
        cluster_times = np.unique(cluster[0])
        all_cluster_times += cluster_times.tolist()

    all_cluster_channels = np.unique(all_cluster_channels)
    all_cluster_times = np.unique(all_cluster_times)
    print('Number of significant channels:', len(all_cluster_channels))

    # average channels that are part of at least one significant cluster
    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    r2_diff = model_r_squared - control_r_squared

    # note density
    note_density = compute_note_density(
        epoch_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/maestro01_itdheb-epo.fif')

    color = ['#878787', '#087e8b', '#ff5a5f', '#6859D9']

    control_r_squared = np.mean(control_r_squared[:, all_cluster_channels, :], axis=1)
    control_sem = np.std(control_r_squared, axis=0) / np.sqrt(control_r_squared.shape[0])
    control_r_squared = np.mean(control_r_squared, axis=0)
    line, = ax1.plot(times, control_r_squared, color=color[0], label='Control model')
    ax1.fill_between(times, control_r_squared - control_sem, control_r_squared + control_sem,
                                  color=color[0], alpha=0.5, linewidth=0)

    model_r_squared = np.mean(model_r_squared[:, all_cluster_channels, :], axis=1)
    model_sem = np.std(model_r_squared, axis=0) / np.sqrt(model_r_squared.shape[0])
    model_r_squared = np.mean(model_r_squared, axis=0)
    line, = ax1.plot(times, model_r_squared, color=color[1], label='Full model')
    ax1.fill_between(times, model_r_squared - model_sem, model_r_squared + model_sem, color=color[1],
                                  alpha=0.5, linewidth=0)

    r2_diff = np.mean(r2_diff[:, all_cluster_channels, :], axis=1)
    cluster_sem = np.std(r2_diff, axis=0) / np.sqrt(r2_diff.shape[0])
    cluster_average = np.mean(r2_diff, axis=0)
    line, = ax1.plot(times, cluster_average, color=color[2],
                                  label='Difference ($\Delta$R²)')
    ax1.fill_between(times, cluster_average - cluster_sem, cluster_average + cluster_sem, color=color[2],
                                  alpha=0.5, linewidth=0)

    masked_times = np.full(cluster_average.shape[0], True)  # non-significant times will be masked
    masked_times[all_cluster_times] = False  # store significant times
    mask = np.ma.masked_where(masked_times, ax1.get_ylim()[0] * np.ones(cluster_average.shape[0]))  # we mask what is not significant
    ax1.plot(times, mask, color='black', linewidth=2)

    ax1.plot(times, note_density * ax1.get_ylim()[1], color=color[0], label='Note frequency', linestyle='--', lw=0.5, zorder=0)

    ax1.set_ylabel('R²')
    # ax1.xaxis.set_ticklabels([])
    ax1.set_xlabel('Time (s)')
    ax1.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax1.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
    ax1.set_xlim([-0.5, 1])
    leg = ax1.legend()
    # set the linewidth of each legend object
    for legobj in leg.legendHandles[:3]:
        legobj.set_linewidth(2)

    ####################################################################################################################
    # Variance partitioning
    ####################################################################################################################

    original_vp_feature = vp_feature.copy()

    # color_list = ['#087e8b', '#0b3954', '#B6ADFF', '#6859D9', '#ff5a5f', '#c81d25']
    color_list = ['#087e8b', '#0b3954', '#6859D9', '#c81d25']
    alpha = 0.8

    with open(os.path.join(vp_path, "regr_meg_paulrnn.pkl"), 'rb') as fp:
        reg_paulrnn = pickle.load(fp)
    with open(os.path.join(vp_path, "regr_meg_acoustic.pkl"), 'rb') as fp:
        reg_acoustic = pickle.load(fp)

    vp_data = {}
    for feat in vp_feature:
        with open(os.path.join(vp_path, "regr_meg_paulrnn_removed_" + feat + ".pkl"), 'rb') as fp:
            reg = pickle.load(fp)
            vp_data[feat] = reg

    times = reg_paulrnn['times'][tmin:]

    significant_channels = np.load(os.path.join(save_path, 'significant_channels.npy'))

    # retrieve results from full model
    data = {}
    y = reg_paulrnn['r2'][:, significant_channels, tmin:] - reg_acoustic['r2'][:, significant_channels, tmin:]
    # y = reg_paulrnn['r2'][:, :, tmin:] - reg_acoustic['r2'][:, :, tmin:]
    y = np.mean(y, axis=(0, 1))
    data['all'] = y

    # retrieve results for each unique feature
    for i, feat in enumerate(vp_feature):
        y = vp_data[feat]['r2'][:, significant_channels, tmin:] - reg_acoustic['r2'][:, significant_channels, tmin:]
        # y = vp_data[feat]['r2'][:, :, tmin:] - reg_acoustic['r2'][:, :, tmin:]
        y = np.mean(y, axis=(0, 1))
        data[feat] = data['all'] - y

    # retrieve shared variance
    data['shared'] = data['all'] - sum([data[feat] for feat in vp_feature])

    # merge features
    data[vp_feature[2]] = np.where(data[vp_feature[2]] > 0, data[vp_feature[2]], 0) + np.where(data[vp_feature[3]] > 0, data[vp_feature[3]], 0)
    data[vp_feature[4]] = np.where(data[vp_feature[4]] > 0, data[vp_feature[4]], 0) + np.where(data[vp_feature[5]] > 0, data[vp_feature[5]], 0)
    vp_feature = [vp_feature[i] for i in [0, 1, 2, 4]]
    feature_names = [feature_names[i] for i in [0, 1, 2, 4]]

    # fig
    suptitle = 'Variance Partitioning - all channels'
    y = np.where(data['shared'] > 0, data['shared'], 0)

    # axs.plot(times, y, color='black', label='shared')
    ax4.fill_between(times, 0, y, alpha=0.7, color='black', label='Shared', linewidth=0)
    # axs.fill_between(times, 0, y, alpha=alpha, color=color_list[0], label='Shared', linewidth=0)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Variance partitioning ($\Delta$R²)')
    ax4.set_xlim([-0.5, 1.0])
    ax4.set_ylim([0, 4.05e-4])

    for i, feat in enumerate(vp_feature):
        previous = copy.copy(y)
        y += np.where(data[feat] > 0, data[feat], 0)
        # axs.plot(times, y, label=feat)
        # axs.fill_between(times, previous, y, alpha=alpha, color = color_list[i+1], label=feature_names[i], linewidth=0)
        ax4.fill_between(times, previous, y, alpha=alpha, color=color_list[i], label=feature_names[i], linewidth=0)

    ax4.axvline(0, linestyle='--', linewidth=1, color='k', zorder=1)
    # ax4.axhline(0, linestyle='--', linewidth=1.5, color='k', zorder=1)

    # leg = ax4.legend()
    # # set the linewidth of each legend object
    # for legobj in leg.legendHandles[:2]:
    #     legobj.set_linewidth(2)
    alpha = [0.7] + [alpha] * len(color_list)
    color_list = ['black'] + color_list
    custom_lines = [Line2D([0], [0], color=color_list[i], alpha=alpha[i], lw=2) for i in range(len(feature_names) + 1)]
    ax4.legend(custom_lines, ['Shared'] + [feature_names[i] for i in range(len(feature_names))])

    ####################################################################################################################
    # Temporal seeg
    ####################################################################################################################

    with open(seeg_pickle, 'rb') as f:
        seeg_data = pickle.load(f)

    paulrnn = seeg_data['paulrnn'][:, tmin:]
    control = seeg_data['control'][:, tmin:]
    print(control.shape)
    control_r_squared = control[seeg_data['ch_sign'], :]
    print(control_r_squared.shape)
    model_r_squared = paulrnn[seeg_data['ch_sign'], :]
    times = seeg_data['times'][tmin:]
    diff_r_squared = model_r_squared - control_r_squared

    color = ['#878787', '#087e8b', '#ff5a5f', '#6859D9']

    # control_r_squared = np.mean(control_r_squared, axis=1)
    control_sem = np.std(control_r_squared, axis=0) / np.sqrt(control_r_squared.shape[0])
    control_r_squared = np.mean(control_r_squared, axis=0)
    line, = ax2.plot(times, control_r_squared, color=color[0], label='Control model')
    ax2.fill_between(times, control_r_squared - control_sem, control_r_squared + control_sem,
                     color=color[0], alpha=0.5, linewidth=0)

    # model_r_squared = np.mean(model_r_squared, axis=1)
    model_sem = np.std(model_r_squared, axis=0) / np.sqrt(model_r_squared.shape[0])
    model_r_squared = np.mean(model_r_squared, axis=0)
    line, = ax2.plot(times, model_r_squared, color=color[1], label='Full model')
    ax2.fill_between(times, model_r_squared - model_sem, model_r_squared + model_sem, color=color[1],
                     alpha=0.5, linewidth=0)

    # diff_r_squared = np.mean(diff_r_squared, axis=1)
    diff_sem = np.std(diff_r_squared, axis=0) / np.sqrt(diff_r_squared.shape[0])
    diff_average = np.mean(diff_r_squared, axis=0)
    line, = ax2.plot(times, diff_average, color=color[2], label='Difference ($\Delta$R²)')
    ax2.fill_between(times, diff_average - diff_sem, diff_average + diff_sem, color=color[2],
                     alpha=0.5, linewidth=0)

    note_density = seeg_data['note_density'][tmin:]
    note_density[50] = np.nan
    note_density = note_density / np.nanmax(note_density)
    # print(note_density[50])
    ax2.plot(times, note_density * ax2.get_ylim()[1], color=color[0], label='Note frequency', linestyle='--', lw=0.5,
             zorder=0)
    # ax2.plot(times, note_density * ax2.get_ylim()[1], color=color[0], linestyle='--', lw=0.5, zorder=0)

    all_cluster_times = seeg_data['c_sign'][:-1, 0]  # remove last element which is one time sample
    all_cluster_times = np.concatenate([cluster for cluster in all_cluster_times])
    all_cluster_times = np.unique(all_cluster_times)
    print(all_cluster_times)
    all_cluster_times = all_cluster_times - tmin
    print(all_cluster_times)
    all_cluster_times = [time for time in all_cluster_times if time >= 0]
    print(all_cluster_times)
    masked_times = np.full(diff_average.shape[0], True)  # non-significant times will be masked
    masked_times[all_cluster_times] = False  # store significant times
    mask = np.ma.masked_where(masked_times,
                              ax2.get_ylim()[0] * np.ones(diff_average.shape[0]))  # we mask what is not significant
    ax2.plot(times, mask, color='black', linewidth=2)

    ax2.set_ylabel('R²')
    # ax1.xaxis.set_ticklabels([])
    ax2.set_xlabel('Time (s)')
    # ax2.set_ylabel('$\Delta$R²')
    # ax2.set_xlabel('Time (s)')
    for ax in [ax2]:
        ax.axvline(0, linestyle='--', linewidth=1, color='k', zorder=0)
        ax.axhline(0, linestyle='--', linewidth=1, color='k', zorder=0)
        ax.set_xlim([-0.5, 1])
        leg = ax.legend()
        # set the linewidth of each legend object
        for legobj in leg.legendHandles[:3]:
            legobj.set_linewidth(2)

    ####################################################################################################################
    # Variance partitioning
    ####################################################################################################################

    vp_feature = original_vp_feature

    # color_list = ['#087e8b', '#0b3954', '#B6ADFF', '#6859D9', '#ff5a5f', '#c81d25']
    color_list = ['#087e8b', '#0b3954', '#6859D9', '#c81d25']
    alpha = 0.8

    data = seeg_data['VP']
    times = seeg_data['times'][tmin:]

    # merge features
    data[vp_feature[2]] = np.where(data[vp_feature[2]] > 0, data[vp_feature[2]], 0) + np.where(data[vp_feature[3]] > 0,
                                                                                               data[vp_feature[3]], 0)
    data[vp_feature[4]] = np.where(data[vp_feature[4]] > 0, data[vp_feature[4]], 0) + np.where(data[vp_feature[5]] > 0,
                                                                                               data[vp_feature[5]], 0)
    vp_feature = [vp_feature[i] for i in [0, 1, 2, 4]]
    # feature_names = [feature_names[i] for i in [0, 1, 2, 4]]

    # fig
    suptitle = 'Variance Partitioning - all channels'
    y = np.where(data['shared'] > 0, data['shared'], 0)
    y = y[tmin:]

    # axs.plot(times, y, color='black', label='shared')
    ax3.fill_between(times, 0, y, alpha=0.7, color='black', label='Shared', linewidth=0)
    # axs.fill_between(times, 0, y, alpha=alpha, color=color_list[0], label='Shared', linewidth=0)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Variance partitioning ($\Delta$R²)')
    ax3.set_xlim([-0.5, 1.0])
    ax3.set_ylim([0, 1.2e-2])

    for i, feat in enumerate(vp_feature):
        previous = copy.copy(y)
        y += np.where(data[feat] > 0, data[feat], 0)[tmin:]
        # axs.plot(times, y, label=feat)
        # axs.fill_between(times, previous, y, alpha=alpha, color = color_list[i+1], label=feature_names[i], linewidth=0)
        ax3.fill_between(times, previous, y, alpha=alpha, color=color_list[i], label=feature_names[i], linewidth=0)

    ax3.axvline(0, linestyle='--', linewidth=1, color='k', zorder=1)
    # ax4.axhline(0, linestyle='--', linewidth=1.5, color='k', zorder=1)

    # leg = ax4.legend()
    # # set the linewidth of each legend object
    # for legobj in leg.legendHandles[:2]:
    #     legobj.set_linewidth(2)
    alpha = [0.7] + [alpha] * len(color_list)
    color_list = ['black'] + color_list
    custom_lines = [Line2D([0], [0], color=color_list[i], alpha=alpha[i], lw=2) for i in range(len(feature_names) + 1)]
    ax3.legend(custom_lines, ['Shared'] + [feature_names[i] for i in range(len(feature_names))])

    fig.savefig(os.path.join(save_path, 'figure_poster_paul.svg'))

    return



def topo_meg_acoustic(model_regression, control_regression, permutation_folder, tmin=50, save_path='/home/mathieupvc/These/figures_papier'):

    # fig = plt.figure(layout="constrained", figsize=(7.20, 4.45*2))
    # fig = plt.figure(layout="constrained", figsize=(3.60, 4.45))
    fig = plt.figure(layout="constrained", figsize=(3.60, 4.9))
    # fig = plt.figure(layout="constrained", figsize=(7.20, 4.45*1.4))
    # fig = plt.figure()
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 0.03]) #, hspace=0.1, wspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    ####################################################################################################################
    # Temporal traces and topoplot
    ####################################################################################################################

    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    r2_diff = model_r_squared - control_r_squared
    times = model_regression['times'][tmin:]

    # permutation r2
    permutation_folder = Path(permutation_folder)
    permutation_list = sorted(list(permutation_folder.glob('**/*perm*')))
    r2_permutations = np.empty((len(permutation_list), *model_r_squared.shape[1:]))  # already in t-values
    for p, perm in enumerate(permutation_list):
        with open(perm, 'rb') as file:
            r2_permutations[p, :, :] = pickle.load(file)['r2'][0, :, tmin:]

    r2_diff = np.swapaxes(r2_diff, 1, 2)  # reshape to (n_subjects, times, n_channels)
    r2_permutations = np.swapaxes(r2_permutations, 1, 2)  # reshape to (n_permut, times, n_channels)

    ch_adjacency, ch_names = find_ch_adjacency(model_regression['epochs_info'], 'mag')
    pval = 0.007  # arbitrary
    df = r2_diff.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    thresh = scipy.stats.t.ppf(1 - pval, df=df)  # T distribution, one-tailed
    print(thresh)
    t_obs, clusters, cluster_pv, H0, cluster_stats = permutation_cluster_test_adapted(r2_diff, r2_permutations,
                                                                                      threshold=thresh, tail=1,
                                                                                      adjacency=ch_adjacency, t_power=1)

    channels_indices = np.arange(r2_diff.shape[2])
    sorted_arg_p = np.argsort(cluster_pv)

    significant_clusters = np.array([clusters[arg] for arg in sorted_arg_p if cluster_pv[arg] < 0.05], dtype=object)
    n_significant_clusters = len(significant_clusters)
    print("Significant clusters:", n_significant_clusters)

    all_cluster_channels = []
    all_cluster_times = []
    for i, cluster in enumerate(significant_clusters):
        start_sample, end_sample = cluster[0][0], cluster[0][-1]
        print(start_sample, end_sample)

        cluster_channels = np.unique(cluster[1])
        all_cluster_channels += cluster_channels.tolist()
        cluster_times = np.unique(cluster[0])
        all_cluster_times += cluster_times.tolist()

    all_cluster_channels = np.unique(all_cluster_channels)
    all_cluster_times = np.unique(all_cluster_times)
    print('Number of significant channels:', len(all_cluster_channels))

    # average channels that are part of at least one significant cluster
    control_r_squared = control_regression['r2'][:, :, tmin:]
    model_r_squared = model_regression['r2'][:, :, tmin:]
    r2_diff = model_r_squared - control_r_squared

    cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#878787', 'white', '#087e8b'])
    # cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#ff5a5f', 'white', '#087e8b'])
    im, cn = mne.viz.plot_topomap(np.mean(control_r_squared, axis=(0, 2)),
                                  model_regression['epochs_info'], axes=ax1, show=False,
                                  # cmap=cmap1
                                  )
    clb = plt.colorbar(im, cax=ax2, orientation='horizontal', location='bottom')
    clb.set_label('R²')

    fig.savefig(os.path.join(save_path, 'topo_meg_acoustic.svg'))

    return



if __name__ == '__main__':

    '''figure_2'''
    with open('/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_paulrnn.pkl', 'rb') as file:
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_magenta_perfrnn.pkl', 'rb') as file:
        model_regression = pickle.load(file)
    with open('/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_acoustic.pkl', 'rb') as file:
        control_regression = pickle.load(file)
    permutation_folder = '/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23'
    vp_path = '/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23'
    vp_feature = ['surprise_mean_paulrnn', 'shifted_surprise_mean_paulrnn',
                  'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn',
                  'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']
    feature_names = ['Surprise', 'Surprise (n+1)',
                     'Uncertainty', 'Uncertainty (n+1)',
                     'Predicted density', 'Predicted density (n+1)']
    best_channels_path = '/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_20channels_100ms.npy'
    # figure_2(model_regression, control_regression, permutation_folder, vp_path, vp_feature, feature_names, best_channels_path, tmin=50, save_path='/home/mathieu/These/figures_papier')

    # figure_2_version_october_24(model_regression, control_regression, permutation_folder, vp_path, vp_feature, feature_names, best_channels_path, tmin=50, save_path='/home/mathieu/These/figures_papier')

    '''figure_4'''
    with open('/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_paulrnn.pkl', 'rb') as file:
        model_regression = pickle.load(file)
    with open('/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_acoustic.pkl', 'rb') as file:
        control_regression = pickle.load(file)
    with open('/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_magenta_perfrnn.pkl', 'rb') as file:
        magenta_regression = pickle.load(file)
    vp_path = '/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23'
    vp_feature = ['surprise_mean_paulrnn', 'shifted_surprise_mean_paulrnn',
                  'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn',
                  'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']
    feature_names = ['Surprise', 'Surprise (n+1)',
                     'Uncertainty', 'Uncertainty (n+1)',
                     'Predicted density', 'Predicted density (n+1)']
    best_channels_path = '/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_20channels_100ms.npy'
    seeg_pickle = '/home/mathieu/These/figures_papier/seeg/f4.pkl'
    # figure_4(model_regression, control_regression, magenta_regression, best_channels_path, seeg_pickle, tmin=50, save_path='/home/mathieu/These/figures_papier')

    # figure_4_october_24(model_regression, control_regression, magenta_regression, best_channels_path, seeg_pickle, tmin=50, save_path='/home/mathieu/These/figures_papier')

    '''figure_1'''
    pickle_path = '/home/mathieupvc/These/figures_papier/seeg/seeg_exemplar_channels.pkl'
    acoustic_feature_list = ['loudness', 'pitch', 'time_interval', 'spectralflux']
    predictive_feature_list = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn']
    n_notes_metadata = 8
    n_notes_data = 4
    # figure_1(pickle_path, acoustic_feature_list, predictive_feature_list, n_notes_metadata, n_notes_data)

    '''figure_3'''
    vp_path = '/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23'
    vp_feature = ['surprise_mean_paulrnn', 'shifted_surprise_mean_paulrnn',
                  'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn',
                  'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']
    feature_names = ['Surprise', 'Surprise (n+1)',
                     'Uncertainty', 'Uncertainty (n+1)',
                     'Predicted density', 'Predicted density (n+1)']
    seeg_pickle = '/home/mathieu/These/figures_papier/seeg/f4.pkl'
    # figure_3(seeg_pickle, vp_feature, feature_names, tmin=50, save_path='/home/mathieu/These/figures_papier')

    '''figure_liberto_data'''
    eeg_pickle = '/home/mathieupvc/These/figures_papier/figure_data_liberto/eeg_note_level_results_and_mask.pkl'
    # figure_liberto_data(eeg_pickle, threshold=0.00035, tmin=32, save_path='/home/mathieupvc/These/figures_papier')

    '''spatio_temporal_cluster_regression_scores_cluster_averages'''
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_paulrnn.pkl', 'rb') as file:
    # # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_magenta_perfrnn.pkl', 'rb') as file:
    #     model_regression = pickle.load(file)
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_acoustic.pkl', 'rb') as file:
    #     control_regression = pickle.load(file)
    # features_list = ['surprise_mean_paulrnn', 'entropy_pitch_paulrnn']
    # # features_list = ['surprise_magenta_pitch_perfrnn']
    # # spatio_temporal_cluster_regression_cluster_averages(model_regression, control_regression, '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23', features_list, n_clusters_to_keep='p', tmin=50, )

    '''model_comparison'''
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_paulrnn.pkl', 'rb') as file:
    #     model_regression = pickle.load(file)
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_acoustic.pkl', 'rb') as file:
    #     control_regression = pickle.load(file)
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_magenta_perfrnn.pkl', 'rb') as file:
    #     magenta_regression = pickle.load(file)
    # # model_comparison(model_regression, control_regression, magenta_regression, n_clusters_to_keep='p', tmin=50, save_path='/home/mathieupvc/These/figures_papier')

    '''variance_partitioning_plot'''
    vp_path = '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23'
    vp_feature = ['surprise_mean_paulrnn', 'shifted_surprise_mean_paulrnn',
                  'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn',
                  'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']
    feature_names = ['Surprise', 'Surprise (n+1)',
                     'Uncertainty', 'Uncertainty (n+1)',
                     'Timing', 'Timing (n+1)']
    # variance_partitioning_plot(vp_path, vp_feature, feature_names, tmin=50, save_path='/home/mathieupvc/These/figures_papier')

    '''histogram_r2_seeg'''
    seeg_pickle = '/home/mathieupvc/These/figures_papier/seeg/f4.pkl'
    # histogram_r2_seeg(seeg_pickle, tmin=50, save_path='/home/mathieupvc/These/figures_papier')

    '''r2_boxplot'''
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_paulrnn.pkl', 'rb') as file:
    #     model_regression = pickle.load(file)
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_acoustic.pkl', 'rb') as file:
    #     control_regression = pickle.load(file)
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_magenta_perfrnn.pkl','rb') as file:
    #     magenta_regression = pickle.load(file)
    # # r2_boxplot(model_regression, control_regression, magenta_regression, tmin=50, tmax=201, save_path='/home/mathieupvc/These/figures_papier', best_channels_path='/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_20channels_100ms.npy')

    '''betas_figure'''
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_paulrnn.pkl', 'rb') as file:
    #     model_regression = pickle.load(file)
    # features_to_plot = ['surprise_mean_paulrnn', 'loudness_deriv']
    # # features_to_plot = ['rate', 'trial_time', 'time_interval', 'pitch', 'pitch_interval', 'loudness', 'brightness', 'periodicity', 'roughness', 'spectralflux', 'loudness_deriv', 'brightness_deriv', 'periodicity_deriv', 'roughness_deriv', 'spectralflux_deriv']
    # # betas_figure(model_regression, features_to_plot, save_path='/home/mathieupvc/These/figures_papier', averaging_method='mean')

    '''erp_by_quantile'''
    # erp_by_quantile('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/erp_surprise_mean_paulrnn_quantiles.npy', (np.log(np.linspace(1, 11, 11)) / np.log(11))[1:])

    '''average_betas_figure'''
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_paulrnn.pkl', 'rb') as file:
    #     model_regression = pickle.load(file)
    # features_to_plot = ['surprise_mean_paulrnn', 'shifted_surprise_mean_paulrnn', 'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn', 'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']
    # feature_names = ['Surprise', 'Surprise (n+1)', 'Pitch uncertainty', 'Pitch uncertainty (n+1)', 'Timing prediction', 'Timing prediction (n+1)']
    # # average_betas_figure(model_regression, features_to_plot, feature_names, save_path='/home/mathieupvc/These/figures_papier', tmin=50)

    '''note_density_scale'''
    # note_density_scale(seeg_pickle, tmin=50)

    '''poster_paul_31_05'''
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_paulrnn.pkl', 'rb') as file:
    # # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_magenta_perfrnn.pkl', 'rb') as file:
    #     model_regression = pickle.load(file)
    # with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_acoustic.pkl', 'rb') as file:
    #     control_regression = pickle.load(file)
    # permutation_folder = '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23'
    # vp_path = '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23'
    # vp_feature = ['surprise_mean_paulrnn', 'shifted_surprise_mean_paulrnn',
    #               'entropy_pitch_paulrnn', 'shifted_entropy_pitch_paulrnn',
    #               'entropy_timing_paulrnn', 'shifted_entropy_timing_paulrnn']
    # feature_names = ['Surprise', 'Surprise (n+1)',
    #                  'Pitch uncertainty', 'Pitch uncertainty (n+1)',
    #                  'Timing prediction', 'Timing prediction (n+1)']
    # best_channels_path = '/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/erp/best_20channels_100ms.npy'
    #
    # seeg_pickle = '/home/mathieupvc/These/figures_papier/seeg/f4.pkl'
    # # poster_paul_31_05(model_regression, control_regression, permutation_folder, vp_path, vp_feature, feature_names, seeg_pickle, tmin=50, save_path='/home/mathieupvc/These/figures_papier')

    '''topo_meg_acoustic'''
    with open('/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_paulrnn.pkl', 'rb') as file:
    # with open('/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_magenta_perfrnn.pkl', 'rb') as file:
        model_regression = pickle.load(file)
    with open('/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23/regr_meg_acoustic.pkl', 'rb') as file:
        control_regression = pickle.load(file)
    permutation_folder = '/home/mathieu/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/permut_03_11_23'
    # topo_meg_acoustic(model_regression, control_regression, permutation_folder, tmin=50, save_path='/home/mathieu/These/figures_papier')