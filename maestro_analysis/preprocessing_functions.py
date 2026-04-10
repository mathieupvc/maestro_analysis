import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import mne
from mne.preprocessing import find_bad_channels_maxwell


def concatenate_and_Maxfilter_blocks(sample_data_raw_file_list, fine_cal_file, crosstalk_file, bads=None, channel_types=None):
    '''Concatenates blocks of a single session and outputs a single MNE Raw instance. Applies Maxwell filtering with
    head repositioning so that all blocks share the same spatial reference (that of the first raw file in
    sample_data_raw_file_list). Applies an automatic detection of bad channels (noisy and flat) and enables
    visualization and manual removal of channels.
        sample_data_raw_file_list: list of str, contains paths to every block of the session
        fine_cal_file: str, path to calibration file (e.g. sss_cal_mgh.dat)
        crosstalk_file: str, path to crosstalk compensation file (e.g. ct_sparse_mgh.fif)
        bads: list, channels to set as bad directly (for instance if a given channel is known to be too noisy)
        channel_types: None or list, if list of channel types, enables visualization and manual removal of channels
    '''

    if bads is None:
        bads = []

    raw_list = [mne.io.read_raw_fif(sample, allow_maxshield=True) for sample in sample_data_raw_file_list]

    sss_list = []
    visu_done = False
    for raw in raw_list:
        # First we look for bad channels
        raw_check = raw.copy()
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check, cross_talk=crosstalk_file, calibration=fine_cal_file,
            return_scores=True, verbose=True)
        raw.info['bads'] = raw.info['bads'] + auto_noisy_chs + auto_flat_chs

        # Inspection of noisy and flat channels
        if channel_types is not None:
            if not visu_done:
                raw.plot(duration=5, n_channels=30)
                bad_channels_visu(channel_types, auto_scores)
                visu_done = True  # visualization of the first block only, bad channels that were selected manually will be set as bad in every block
                bads.extend(raw.info['bads'])
           # else:
           #     raw.plot(duration=25, n_channels=30)
           #     plt.show()

        # Add the known bad channels
        raw.info['bads'] += bads

        sss_raw = mne.preprocessing.maxwell_filter(raw, cross_talk=crosstalk_file, calibration=fine_cal_file,
                                                   verbose=True,
                                                   destination=sample_data_raw_file_list[0])

        sss_list.append(sss_raw)

    raw = mne.concatenate_raws(sss_list)

    return raw



def bad_channels_visu(channel_types, auto_scores):
    '''plot noisy and flat channels.
        channel_types: list of str
        auto_scores: output of mne.preprocessing.find_bad_channels_maxwell
    '''

    for ch_type in channel_types:
        '''Inspection of noisy channels'''
        # # Only select the data of channel type.
        ch_subset = auto_scores['ch_types'] == ch_type
        ch_names = auto_scores['ch_names'][ch_subset]
        scores = auto_scores['scores_noisy'][ch_subset]
        limits = auto_scores['limits_noisy'][ch_subset]
        bins = auto_scores['bins']  # The the windows that were evaluated.
        # We will label each segment by its start and stop time, with up to 3
        # digits before and 3 digits after the decimal place (1 ms precision).
        bin_labels = [f'{start:3.3f} – {stop:3.3f}'
                      for start, stop in bins]

        # We store the data in a Pandas DataFrame. The seaborn heatmap function
        # we will call below will then be able to automatically assign the correct
        # labels to all axes.
        data_to_plot = pd.DataFrame(data=scores,
                                    columns=pd.Index(bin_labels, name='Time (s)'),
                                    index=pd.Index(ch_names, name='Channel'))

        # First, plot the "raw" scores.
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f'Automated noisy channel detection: {ch_type}',
                     fontsize=16, fontweight='bold')
        sns.heatmap(data=data_to_plot, cmap='Reds', cbar_kws=dict(label='Score'),
                    ax=ax[0])
        [ax[0].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
            for x in range(1, len(bins))]
        ax[0].set_title('All Scores', fontweight='bold')

        # Now, adjust the color range to highlight segments that exceeded the limit.
        sns.heatmap(data=data_to_plot,
                    vmin=np.nanmin(limits),  # bads in input data have NaN limits
                    cmap='Reds', cbar_kws=dict(label='Score'), ax=ax[1])
        [ax[1].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
            for x in range(1, len(bins))]
        ax[1].set_title('Scores > Limit', fontweight='bold')

        # The figure title should not overlap with the subplots.
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        '''Inspection of flat channels'''
        # Only select the data of channel type.
        ch_subset = auto_scores['ch_types'] == ch_type
        ch_names = auto_scores['ch_names'][ch_subset]
        scores = auto_scores['scores_flat'][ch_subset]
        limits = auto_scores['limits_flat'][ch_subset]
        bins = auto_scores['bins']  # The the windows that were evaluated.
        # We will label each segment by its start and stop time, with up to 3
        # digits before and 3 digits after the decimal place (1 ms precision).
        bin_labels = [f'{start:3.3f} – {stop:3.3f}'
                      for start, stop in bins]

        # We store the data in a Pandas DataFrame. The seaborn heatmap function
        # we will call below will then be able to automatically assign the correct
        # labels to all axes.
        data_to_plot = pd.DataFrame(data=scores,
                                    columns=pd.Index(bin_labels, name='Time (s)'),
                                    index=pd.Index(ch_names, name='Channel'))

        # First, plot the "raw" scores.
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f'Automated flat channel detection: {ch_type}',
                     fontsize=16, fontweight='bold')
        sns.heatmap(data=data_to_plot, cmap='Reds', cbar_kws=dict(label='Score'),
                    ax=ax[0])
        [ax[0].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
            for x in range(1, len(bins))]
        ax[0].set_title('All Scores', fontweight='bold')

        # Now, adjust the color range to highlight segments that exceeded the limit.
        sns.heatmap(data=data_to_plot,
                    vmax=np.nanmax(limits),  # bads in input data have NaN limits
                    cmap='Reds', cbar_kws=dict(label='Score'), ax=ax[1])
        [ax[1].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
            for x in range(1, len(bins))]
        ax[1].set_title('Scores > Limit', fontweight='bold')

        # The figure title should not overlap with the subplots.
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()
