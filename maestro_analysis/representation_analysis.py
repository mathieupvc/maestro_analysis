import os
import random
from pathlib import Path
import pickle

import re

from statsmodels.genmod.families.links import loglog
from tqdm import tqdm
from collections import Counter

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

import mne
from mne.datasets import sample
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_1samp_test

from maestro_analysis.utils import epoching_maestro, select_seeg_channels_from_parcellation, select_parcel
from maestro_analysis.stats import permut_1sample_ttest_across_channels, already_permuted_1sample_ttest_across_channels, logistic_log_likelihood, ols_log_likelihood

import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.anova import AnovaRM
from statsmodels.regression.mixed_linear_model import MixedLM
import scipy
from scipy.stats import ttest_ind, ttest_1samp, pearsonr, spearmanr, chi2_contingency, binom_test, binomtest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

from collections import defaultdict

from dtaidistance import dtw

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def save_similarity_matrix(data_path, wav_folder, save_path, channel_type='mag', n_jobs=-1):
    """Load meg data and saves a pickle containing a similarity matrix for every subject (correlation between every stimulus pair)"""

    os.makedirs(save_path, exist_ok=True)

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    subject_list = []
    info_list = []
    stim_name_list = []
    sim_matrix_list = []
    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        epochs = epoching_maestro(raw_path, wav_folder, allow_maxshield=False, tmin=-0.5, baseline=True, bin_ecg=False, channel_type='mag', remove_responses=True)

        # select channels
        epochs.pick(channel_type)

        # preprocessing
        epochs.resample(200, n_jobs=n_jobs)
        epochs.filter(0.5, None, n_jobs=n_jobs)  # highpass filter
        epochs.crop(tmin=1, tmax=7)  # remove onset and offset
        data = epochs.get_data()

        subject_list.append(subject_path.name)
        stim_name_list.append(epochs.metadata['stim_name'].to_list())
        info_list.append(epochs.info)
        del epochs

        # compute corr matrix for each channel
        similarity_matrix = np.zeros((data.shape[0], data.shape[0], data.shape[1]))
        for c in range(data.shape[1]):
            similarity_matrix[:, :, c] = np.corrcoef(data[:, c, :])
        sim_matrix_list.append(similarity_matrix)

    similarity_dict = {'subjects': subject_list, 'info': info_list, 'stim_names': stim_name_list, 'similarity_matrices': sim_matrix_list}

    with open(os.path.join(save_path, "similarity_matrix.pkl"), 'wb') as f:
        pickle.dump(similarity_dict, f)

    print(f'Saved in {os.path.join(save_path, "similarity_matrix.pkl")}')

    return


def save_similarity_matrix_seeg(data_path, save_path, regions=None, n_jobs=-1):
    """Load seeg epochs and saves a pickle containing a similarity matrix for every subject (correlation between every stimulus pair)"""

    os.makedirs(save_path, exist_ok=True)

    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*.fif')))

    subject_list = []
    info_list = []
    stim_name_list = []
    sim_matrix_list = []
    for i, subject_path in enumerate(subject_paths):
        print(subject_path)
        epochs = mne.read_epochs(subject_path)

        # processing
        if regions is not None:
            epochs, _ = select_seeg_channels_from_parcellation(epochs, regions, first_parcel_label='advanced')

        if epochs is not None:
            epochs.resample(400, n_jobs=n_jobs)
            epochs.filter(0.5, None, n_jobs=n_jobs)  # highpass filter
            epochs.crop(tmin=0, tmax=8)  # remove onset and offset
            data = epochs.get_data()

            subject_list.append(subject_path.name[:6])
            stim_name_list.append(epochs.metadata['stim_name'].to_list())
            info_list.append(epochs.info)
            del epochs

            # compute corr matrix for each channel
            similarity_matrix = np.zeros((data.shape[0], data.shape[0], data.shape[1]))
            for c in range(data.shape[1]):
                similarity_matrix[:, :, c] = np.corrcoef(data[:, c, :])
            sim_matrix_list.append(similarity_matrix)

    similarity_dict = {'subjects': subject_list, 'info': info_list, 'stim_names': stim_name_list, 'similarity_matrices': sim_matrix_list}

    # if len(regions) == 1:
    #     with open(os.path.join(save_path, f"similarity_matrix_channels_{regions[0]}.pkl"), 'wb') as f:
    #         pickle.dump(similarity_dict, f)
    #     print(f'Saved in {os.path.join(save_path, f"similarity_matrix_channels_{regions[0]}.pkl")}')
    # else:
    with open(os.path.join(save_path, "similarity_matrix.pkl"), 'wb') as f:
        pickle.dump(similarity_dict, f)
    print(f'Saved in {os.path.join(save_path, "similarity_matrix.pkl")}')

    return


def save_similarity_matrix_time_windows_seeg(data_path, save_path, regions=None, n_jobs=-1):
    """Load seeg epochs and saves a pickle containing a similarity matrix for every subject (correlation between every stimulus pair)"""

    os.makedirs(save_path, exist_ok=True)

    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*.fif')))

    subject_list = []
    info_list = []
    stim_name_list = []
    sim_matrix_list = []
    for i, subject_path in enumerate(subject_paths):
        print(subject_path)
        epochs = mne.read_epochs(subject_path)

        # processing
        if regions is not None:
            epochs, _ = select_seeg_channels_from_parcellation(epochs, regions, first_parcel_label='advanced')

        if epochs is not None:
            epochs.resample(400, n_jobs=n_jobs)
            epochs.filter(0.5, None, n_jobs=n_jobs)  # highpass filter
            epochs.crop(tmin=0, tmax=8)  # remove onset and offset
            data = epochs.get_data()
            subject_list.append(subject_path.name[:6])
            stim_name_list.append(epochs.metadata['stim_name'].to_list())
            info_list.append(epochs.info)

            starts = [0, 1, 2, 3, 4, 5, 6, 7]
            similarity_matrix = np.zeros((len(starts), data.shape[0], data.shape[0], data.shape[1]))
            for s in range(len(starts)):
                start = epochs.time_as_index(starts[s])[0]
                end = epochs.time_as_index(starts[s] + 1)[0]  # 1 sec window
                # compute corr matrix for each channel
                for c in range(data.shape[1]):
                    similarity_matrix[s, :, :, c] = np.corrcoef(data[:, c, start:end])
            sim_matrix_list.append(similarity_matrix)

    similarity_dict = {'subjects': subject_list, 'info': info_list, 'stim_names': stim_name_list, 'similarity_matrices': sim_matrix_list}

    with open(os.path.join(save_path, "similarity_matrix_windows.pkl"), 'wb') as f:
        pickle.dump(similarity_dict, f)
    print(f'Saved in {os.path.join(save_path, "similarity_matrix_windows.pkl")}')

    return


def save_trajectory_distance_matrix_seeg(data_path, save_path, regions=None, n_jobs=-1):
    """Load seeg epochs and saves a pickle containing a similarity matrix for every subject (correlation between every stimulus pair)"""

    os.makedirs(save_path, exist_ok=True)

    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*.fif')))

    subject_list = []
    info_list = []
    stim_name_list = []
    distance_matrix_list = []
    for i, subject_path in enumerate(subject_paths):
        print(subject_path)
        epochs = mne.read_epochs(subject_path)

        # processing
        if regions is not None:
            epochs, _ = select_seeg_channels_from_parcellation(epochs, regions, first_parcel_label='advanced')

        if epochs is not None:
            epochs.resample(400, n_jobs=n_jobs)
            epochs.filter(0.5, None, n_jobs=n_jobs)  # highpass filter
            epochs.crop(tmin=0, tmax=8)  # remove onset and offset
            data = epochs.get_data()

            subject_list.append(subject_path.name[:6])
            stim_name_list.append(epochs.metadata['stim_name'].to_list())
            info_list.append(epochs.info)
            del epochs

            n_epochs, n_electrodes, n_time = data.shape
            data_transposed = data.transpose(0, 2, 1)
            reshaped_data = data_transposed.reshape(n_epochs * n_time, n_electrodes)

            n_components = 3
            pca = PCA(n_components=n_components)
            pca_data = pca.fit_transform(reshaped_data)
            pca_reshaped = pca_data.reshape(n_epochs, n_time, n_components)

            # pca_reshaped = pca_reshaped[:, :, 0]  # keep first component

            distance_matrix = np.zeros((n_epochs, n_epochs))
            from dtaidistance import dtw_ndim
            for i in range(n_epochs):
                for j in range(i, n_epochs):
                    # Compute pointwise Euclidean distance between the trajectories of epochs i and j
                    # pointwise_distance = np.linalg.norm(pca_reshaped[i] - pca_reshaped[j], axis=1)
                    # average_distance = np.corrcoef(pca_reshaped[i], pca_reshaped[j])[0,1]
                    average_distance = dtw_ndim.distance(pca_reshaped[i], pca_reshaped[j], max_dist=100)
                    print(average_distance)

                    # Average the pointwise distances across time
                    # average_distance = np.mean(pointwise_distance)

                    # Assign the distance to both (i, j) and (j, i) to ensure symmetry
                    distance_matrix[i, j] = average_distance
                    distance_matrix[j, i] = average_distance

            distance_matrix_list.append(distance_matrix)

    similarity_dict = {'subjects': subject_list, 'info': info_list, 'stim_names': stim_name_list, 'distance_matrices': distance_matrix_list}

    with open(os.path.join(save_path, "distance_matrix.pkl"), 'wb') as f:
        pickle.dump(similarity_dict, f)
    print(f'Saved in {os.path.join(save_path, "distance_matrix.pkl")}')

    return


def save_distance_matrix_seeg(data_path, save_path, regions=None, n_jobs=-1):
    """Load seeg epochs and saves a pickle containing a distance matrix for every subject"""

    os.makedirs(save_path, exist_ok=True)

    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*.fif')))

    subject_list = []
    info_list = []
    stim_name_list = []
    sim_matrix_list = []
    for i, subject_path in enumerate(subject_paths):
        print(subject_path)
        epochs = mne.read_epochs(subject_path)

        # processing
        if regions is not None:
            epochs, _ = select_seeg_channels_from_parcellation(epochs, regions, first_parcel_label='advanced')

        if epochs is not None:
            epochs.resample(400, n_jobs=n_jobs)
            epochs.filter(0.5, None, n_jobs=n_jobs)  # highpass filter
            epochs.crop(tmin=0, tmax=8)  # remove onset and offset
            data = epochs.get_data()

            subject_list.append(subject_path.name[:6])
            stim_name_list.append(epochs.metadata['stim_name'].to_list())
            info_list.append(epochs.info)
            del epochs

            # compute dtw matrix for each channel
            similarity_matrix = np.zeros((data.shape[0], data.shape[0], data.shape[1]))
            for c in tqdm(range(data.shape[1])):
                similarity_matrix[:, :, c] = dtw.distance_matrix_fast(data[:, c, :], parallel=True)
            sim_matrix_list.append(similarity_matrix)

    similarity_dict = {'subjects': subject_list, 'info': info_list, 'stim_names': stim_name_list, 'distance_matrices': sim_matrix_list}

    with open(os.path.join(save_path, "dtw_distance_matrix.pkl"), 'wb') as f:
        pickle.dump(similarity_dict, f)
    print(f'Saved in {os.path.join(save_path, "dtw_distance_matrix.pkl")}')

    return


def save_distance_matrix_windows_seeg(data_path, save_path, regions=None, n_jobs=-1):
    """Load seeg epochs and saves a pickle containing a distance matrix for every subject"""

    os.makedirs(save_path, exist_ok=True)

    path = Path(data_path)
    subject_paths = sorted(list(path.glob('*.fif')))

    subject_list = []
    info_list = []
    stim_name_list = []
    sim_matrix_list = []
    for i, subject_path in enumerate(subject_paths):
        print(subject_path)
        epochs = mne.read_epochs(subject_path)

        # processing
        if regions is not None:
            epochs, _ = select_seeg_channels_from_parcellation(epochs, regions, first_parcel_label='advanced')

        if epochs is not None:
            epochs.resample(400, n_jobs=n_jobs)
            epochs.filter(0.5, None, n_jobs=n_jobs)  # highpass filter
            epochs.crop(tmin=0, tmax=8)  # remove onset and offset
            data = epochs.get_data()

            subject_list.append(subject_path.name[:6])
            stim_name_list.append(epochs.metadata['stim_name'].to_list())
            info_list.append(epochs.info)

            starts = [0, 1, 2, 3, 4, 5, 6, 7]
            similarity_matrix = np.zeros((len(starts), data.shape[0], data.shape[0], data.shape[1]))
            for s in range(len(starts)):
                start = epochs.time_as_index(starts[s])[0]
                end = epochs.time_as_index(starts[s] + 1)[0]  # 1 sec window
                # compute dtw matrix for each channel
                for c in tqdm(range(data.shape[1])):
                    similarity_matrix[s, :, :, c] = dtw.distance_matrix_fast(data[:, c, start:end], parallel=True)
            sim_matrix_list.append(similarity_matrix)

    similarity_dict = {'subjects': subject_list, 'info': info_list, 'stim_names': stim_name_list, 'distance_matrices': sim_matrix_list}

    with open(os.path.join(save_path, "dtw_distance_matrix_windows.pkl"), 'wb') as f:
        pickle.dump(similarity_dict, f)
    print(f'Saved in {os.path.join(save_path, "dtw_distance_matrix_windows.pkl")}')

    return


def save_stim_amplitude(data_path, wav_folder, save_path, channel_type='mag', n_jobs=-1):
    """Load meg data and saves a pickle containing a stim amplitude for every subject (average amplitude of every stimulus)"""

    os.makedirs(save_path, exist_ok=True)

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    subject_list = []
    info_list = []
    stim_name_list = []
    stim_amplitude_list = []
    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        epochs = epoching_maestro(raw_path, wav_folder, allow_maxshield=False, tmin=-0.5, baseline=True, bin_ecg=False, channel_type='mag', remove_responses=True)

        # select channels
        epochs.pick(channel_type)

        # preprocessing
        epochs.resample(200, n_jobs=n_jobs)
        epochs.filter(0.5, None, n_jobs=n_jobs)  # highpass filter
        epochs.crop(tmin=1, tmax=7)  # remove onset and offset
        data = epochs.get_data()

        subject_list.append(subject_path.name)
        stim_name_list.append(epochs.metadata['stim_name'].to_list())
        info_list.append(epochs.info)
        del epochs

        # compute corr matrix for each channel
        stim_amplitude = np.sqrt(np.mean(data**2, axis=2))  # rms across time
        stim_amplitude_list.append(stim_amplitude)

    similarity_dict = {'subjects': subject_list, 'info': info_list, 'stim_names': stim_name_list, 'stim_amplitude': stim_amplitude_list}

    with open(os.path.join(save_path, "stim_amplitude.pkl"), 'wb') as f:
        pickle.dump(similarity_dict, f)

    print(f'Saved in {os.path.join(save_path, "stim_amplitude.pkl")}')

    return


def save_source_localized_similarity_matrix(data_path, wav_folder, save_path, channel_type='mag', n_jobs=-1):
    """Load meg data and saves a pickle containing a similarity matrix for every subject (correlation between every stimulus pair) in the source space. Source localization is done from a template."""

    os.makedirs(save_path, exist_ok=True)

    path = Path(data_path)
    subject_folders = sorted([x for x in path.iterdir() if x.is_dir()])

    # Set-up templates
    data_path = sample.data_path()
    subjects_dir = Path(data_path) / 'subjects'  # Use Path to construct file paths
    template_subject = 'sample'
    template_bem_path = subjects_dir / template_subject / 'bem' / f'{template_subject}-5120-bem-sol.fif'
    # Load or define the template source space
    template_src = mne.setup_volume_source_space(template_subject, pos=10.0, mri='T1.mgz', bem=None, subjects_dir=subjects_dir)

    # Function to create a transformation matrix
    def create_trans(epochs, subjects_dir):
        dig_montage = mne.channels.make_standard_montage('standard_1020')
        epochs.set_montage(dig_montage, match_case=False)
        trans = mne.coreg.estimate_head_mri_t('fsaverage', subjects_dir)
        return trans

    # Process the subject and compute the volume source estimate
    def process_subject_volume(epochs, subjects_dir, template_bem_path, template_src):
        bem = mne.read_bem_solution(template_bem_path)
        src = template_src
        trans = create_trans(epochs, subjects_dir)
        fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, meg=True, eeg=False)
        noise_cov = mne.compute_covariance(epochs, tmax=0., method='empirical')
        inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov, depth=0.8)
        stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2=1.0 / 9.0, method='dSPM',
                                                     pick_ori=None, return_generator=False, nave=1)
        return stcs

    for i, subject_path in enumerate(subject_folders):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        epochs = epoching_maestro(raw_path, wav_folder, allow_maxshield=False, tmin=-0.5, baseline=True, bin_ecg=False, channel_type='mag', remove_responses=True)

        # select channels
        epochs.pick(channel_type)

        # preprocessing
        epochs.resample(200, n_jobs=n_jobs)
        epochs.filter(0.5, None, n_jobs=n_jobs)  # highpass filter

        stcs = process_subject_volume(epochs, subjects_dir, template_bem_path, template_src)
        stcs = [stc.crop(tmin=1, tmax=7) for stc in stcs]  # remove onset and offset

        vertices_idx = np.where(np.any(stcs[0].data != 0, axis=0))[0]
        filtered_template_src = template_src.copy()
        for hemi in filtered_template_src:
            hemi['inuse'] = np.zeros_like(hemi['inuse'])
            hemi['vertno'] = hemi['vertno'][vertices_idx]
            hemi['inuse'][vertices_idx] = 1
        filtered_template_src.save(os.path.join(save_path, "template_sources.stc"), overwrite=True)

        stim_names = epochs.metadata['stim_name'].to_list()
        info = epochs.info
        del epochs

        n_sources = stcs[0].data.shape[0]
        n_epochs = len(stcs)

        # Initialize an array to store the correlation results
        corr_matrix = np.zeros((n_sources, n_epochs, n_epochs))

        # Compute the correlation between each pair of epochs for each source
        def individual_source_correlation(source_ind):
            """multiprocessing"""
            source_corr_matrix = np.zeros((n_epochs, n_epochs))
            for epoch1 in range(n_epochs):
                for epoch2 in range(n_epochs):
                    data_epoch1 = stcs[epoch1].data[source_ind, :]
                    data_epoch2 = stcs[epoch2].data[source_ind, :]
                    source_corr_matrix[epoch1, epoch2] = np.corrcoef(data_epoch1, data_epoch2)[0, 1]
            return source_corr_matrix

        # multiprocessing sources
        source_correlations = Parallel(n_jobs=-1)(delayed(individual_source_correlation)(s) for s in tqdm(range(n_sources)))
        for s in range(n_sources):
            corr_matrix[s, :, :] = source_correlations[s]

        similarity_dict = {'subject': subject_path.name, 'info': info, 'stim_names': stim_names, 'source_correlation': corr_matrix, 'vertices': stcs[0].vertices}

        with open(os.path.join(save_path, f"{subject_path.name}_source_similarity_matrix.pkl"), 'wb') as f:
            pickle.dump(similarity_dict, f)

    print('Done')

    return


def average_similarity(data_path, conditions=None, channels=None):

    if conditions is None:
        conditions = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']

    data = np.load(data_path, allow_pickle=True)

    subject_list = []
    condition_list = []
    condition_average = []
    for s, subject in enumerate(data["subjects"]):
        print(subject)
        for c, condition in enumerate(conditions):
            condition_indices = np.array([i for i in range(len(data['stim_names'][s])) if condition in data['stim_names'][s][i]])
            if channels is not None:
                chan_indices = mne.pick_channels(data['info'][s]['ch_names'], include=channels)
                condition_matrix = np.mean(data['similarity_matrices'][s][:, :, chan_indices], axis=2)  # average channels
            else:
                condition_matrix = np.mean(data['similarity_matrices'][s], axis=2)  # average channels

            # correlation in condition version
            # condition_matrix = condition_matrix[:, condition_indices]  # select condition
            # condition_matrix = condition_matrix[condition_indices, :]  # select condition
            # condition_average.append(np.mean(condition_matrix[np.triu_indices_from(condition_matrix, k=1)]))

            # correlation of condition trials with all trials version
            condition_matrix = condition_matrix[:, condition_indices]  # select condition
            condition_average.append(np.mean(condition_matrix))

            subject_list.append(subject)
            condition_list.append(condition)

    df = pd.DataFrame({'subject': subject_list, 'condition': condition_list, 'correlation': condition_average})

    ax = sns.boxplot(data=df, x='condition', y='correlation', showmeans=True, palette='Set3')
    ax = sns.swarmplot(data=df, x='condition', y='correlation', color='k', size=4)
    ax.set_ylabel('Intertrial correlation')

    # repeated measures ANOVA
    anova = AnovaRM(data=df, depvar="correlation", subject='subject', within=['condition'])
    anova = anova.fit()
    print('anova')
    print(anova)
    ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

    # mat = matrix[d_list].rename(columns={col: col[2:] for col in matrix[d_list].columns})
    # mat['subject_id'] = range(len(mat))
    # print(mat)
    # df = pd.melt(mat, id_vars=['subject_id'], var_name=anova_variable, value_name="d")
    # d_list = [cond[2:] for cond in d_list]
    # pairs = []
    # for var1 in range(len(d_list)):
    #     for var2 in range(var1 + 1, len(d_list)):
    #         pairs.append((d_list[var1], d_list[var2]))
    # print(df)
    # annotator = Annotator(ax, pairs, data=df, x=anova_variable, y='d')
    # annotator.configure(test='t-test_paired', text_format='star', loc='inside', hide_non_significant=True)
    # if posthocs:
    #     annotator.apply_and_annotate()

    plt.show()


def average_similarity_features(data_path, stim_features_csv, condition, n_bins=5, channels=None):

    stim_features = pd.read_csv(stim_features_csv)
    stim_features = stim_features[['stim_names', condition]]

    # Sort the DataFrame by values
    stim_features = stim_features.sort_values(by=condition).reset_index(drop=True)
    # Calculate the percentile bins
    percentile_bins = pd.qcut(stim_features[condition], q=n_bins, labels=False)
    # Create a dictionary to store lists of stimulus names for each percentile
    percentile_groups = {i: [] for i in range(n_bins)}
    # Assign stimulus names to their respective percentile groups
    for idx, percentile in enumerate(percentile_bins):
        percentile_groups[percentile].append(stim_features.loc[idx, 'stim_names'])

    # Convert dictionary to lists of stimulus names for each percentile
    percentile_lists = [percentile_groups[i] for i in range(n_bins)]

    data = np.load(data_path, allow_pickle=True)

    subject_list = []
    percentile_number_list = []
    percentile_average = []
    for s, subject in enumerate(data["subjects"]):
        print(subject)
        for percentile in range(len(percentile_lists)):
            percentile_indices = np.array([i for i in range(len(data['stim_names'][s])) if data['stim_names'][s][i] in percentile_lists[percentile]])
            if channels is not None:
                chan_indices = mne.pick_channels(data['info'][s]['ch_names'], include=channels)
                percentile_matrix = np.mean(data['similarity_matrices'][s][:, :, chan_indices], axis=2)  # average channels
            else:
                percentile_matrix = np.mean(data['similarity_matrices'][s], axis=2)  # average channels

            # percentile_matrix = percentile_matrix[:, percentile_indices]  # select condition
            # percentile_matrix = percentile_matrix[percentile_indices, :]  # select condition
            # percentile_average.append(np.mean(percentile_matrix[np.triu_indices_from(percentile_matrix, k=1)]))

            # correlation of condition trials with all trials version
            percentile_matrix = percentile_matrix[:, percentile_indices]  # select condition
            percentile_average.append(np.mean(percentile_matrix))

            subject_list.append(subject)
            percentile_number_list.append(percentile)

    df = pd.DataFrame({'subject': subject_list, 'percentile_'+condition: percentile_number_list, 'correlation': percentile_average})

    ax = sns.boxplot(data=df, x='percentile_'+condition, y='correlation', showmeans=True, palette='Set3')
    ax = sns.swarmplot(data=df, x='percentile_'+condition, y='correlation', color='k', size=4)
    ax.set_ylabel('Intertrial correlation')

    # repeated measures ANOVA
    anova = AnovaRM(data=df, depvar="correlation", subject='subject', within=['percentile_'+condition])
    anova = anova.fit()
    print('anova')
    print(anova)
    ax.text(0, ax.get_ylim()[1], f'ANOVA p = {round(anova.anova_table["Pr > F"].to_numpy()[0], 5)}')

    # mat = matrix[d_list].rename(columns={col: col[2:] for col in matrix[d_list].columns})
    # mat['subject_id'] = range(len(mat))
    # print(mat)
    # df = pd.melt(mat, id_vars=['subject_id'], var_name=anova_variable, value_name="d")
    # d_list = [cond[2:] for cond in d_list]
    # pairs = []
    # for var1 in range(len(d_list)):
    #     for var2 in range(var1 + 1, len(d_list)):
    #         pairs.append((d_list[var1], d_list[var2]))
    # print(df)
    # annotator = Annotator(ax, pairs, data=df, x=anova_variable, y='d')
    # annotator.configure(test='t-test_paired', text_format='star', loc='inside', hide_non_significant=True)
    # if posthocs:
    #     annotator.apply_and_annotate()

    plt.show()


def topo_similarity(data_path, conditions=None):

    if conditions is None:
        conditions = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']

    data = np.load(data_path, allow_pickle=True)

    subject_list = []
    correlation_array = [[] for c in range(len(conditions))]
    for s, subject in enumerate(data["subjects"]):
        print(subject)
        for c, condition in enumerate(conditions):
            condition_indices = np.array([i for i in range(len(data['stim_names'][s])) if condition in data['stim_names'][s][i]])
            condition_matrix = data['similarity_matrices'][s]  # keep all channels

            # condition_matrix = condition_matrix[:, condition_indices, :]  # select condition
            # condition_matrix = condition_matrix[condition_indices, :, :]  # select condition
            # triang = np.triu_indices_from(condition_matrix[:, :, 0], k=1)
            # correlations = np.mean(condition_matrix[triang[0], triang[1], :], axis=0)  # average across stim pairs

            # correlation of condition trials with all trials version
            condition_matrix = condition_matrix[:, condition_indices, :]  # select condition
            correlations = np.mean(condition_matrix, axis=(0, 1))  # average across stim pairs

            correlation_array[c].append(correlations)

    correlation_array = np.array(correlation_array)
    print(correlation_array.shape)

    ch_adjacency, ch_names = find_ch_adjacency(data['info'][1], 'mag')
    df = correlation_array.shape[1] - 1  # degrees of freedom numerator: n_observations - 1
    pval = 0.05  # arbitrary
    thresh = scipy.stats.t.ppf(1 - pval, df=df)  # T distribution, one-tailed
    print(thresh)
    channels_indices = np.arange(correlation_array.shape[2])
    for c, condition in enumerate(conditions):
        t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(correlation_array[c, :, :],
                                                                         threshold=thresh, n_permutations=1000, tail=0,
                                                                         adjacency=ch_adjacency, t_power=1)

        clusters = [clusters[arg][0] for arg in range(len(clusters)) if cluster_pv[arg] < 0.05]
        significant_channels = np.unique([channel for cluster in clusters for channel in cluster])

        im, _ = mne.viz.plot_topomap(np.mean(correlation_array[c, :, :], axis=0), data['info'][1], show=False,
                                     mask=np.isin(channels_indices, significant_channels),
                                     mask_params=dict(markersize=8, markeredgewidth=1))
        plt.colorbar(im)
        plt.title("Correlation " + condition)

    # average corr topo
    pval = 0.0001  # arbitrary
    thresh = scipy.stats.t.ppf(1 - pval, df=df)  # T distribution, one-tailed
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(np.mean(correlation_array, axis=0),
                                                                     threshold=thresh, n_permutations=1000, tail=1,
                                                                     adjacency=ch_adjacency, t_power=1)

    clusters = [clusters[arg][0] for arg in range(len(clusters)) if cluster_pv[arg] < 0.05]
    significant_channels = np.unique([channel for cluster in clusters for channel in cluster])

    im, _ = mne.viz.plot_topomap(np.mean(correlation_array, axis=(0, 1)), data['info'][1], show=False,
                                 mask=np.isin(channels_indices, significant_channels),
                                 mask_params=dict(markersize=8, markeredgewidth=1))
    plt.colorbar(im)
    plt.title("Correlation Average")


    # pairs = [[2, 0], [2, 1], [2, 3], [2, 4]]
    # pairs = [[1, 0], [1, 2], [1, 3], [1, 4]]
    pairs = [[0, 1]]
    # pairs = [[0, 1], [0, 2], [1, 2]]
    for pair in pairs:
        t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(correlation_array[pair[0], :, :] - correlation_array[pair[1], :, :],
                                                                         threshold=thresh, n_permutations=1000, tail=0,
                                                                         adjacency=ch_adjacency, t_power=1)

        clusters = [clusters[arg][0] for arg in range(len(clusters)) if cluster_pv[arg] < 0.05]
        significant_channels = np.unique([channel for cluster in clusters for channel in cluster])
        im, _ = mne.viz.plot_topomap(np.mean(correlation_array[pair[0], :, :] - correlation_array[pair[1], :, :], axis=0), data['info'][1], show=False,
                                     mask=np.isin(channels_indices, significant_channels),
                                     mask_params=dict(markersize=8, markeredgewidth=1))
        plt.colorbar(im)
        plt.title(f"Correlation {conditions[pair[0]]} - {conditions[pair[1]]}")

    plt.show()

def topo_ushape_similarity(data_path):

    conditions = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz']

    data = np.load(data_path, allow_pickle=True)

    quadratic_param_array = []
    for s, subject in tqdm(enumerate(data["subjects"])):
        # print(subject)
        correlations_array = []
        for c, condition in enumerate(conditions):
            condition_indices = np.array([i for i in range(len(data['stim_names'][s])) if condition in data['stim_names'][s][i]])
            condition_matrix = data['similarity_matrices'][s]  # keep all channels

            condition_matrix = condition_matrix[:, condition_indices, :]  # select condition
            condition_matrix = condition_matrix[condition_indices, :, :]  # select condition
            triang = np.triu_indices_from(condition_matrix[:, :, 0], k=1)
            correlations = np.mean(condition_matrix[triang[0], triang[1], :], axis=0)  # average across stim pairs

            # correlation of condition trials with all trials version
            # condition_matrix = condition_matrix[:, condition_indices, :]  # select condition
            # correlations = np.mean(condition_matrix, axis=(0, 1))  # average across stim pairs

            correlations_array.append(correlations)
        correlations_array = np.array(correlations_array)
        quadratic_param = np.zeros(correlations_array.shape[1])
        for channel in range(correlations_array.shape[1]):
            Y = correlations_array[:, channel]
            X = np.zeros((Y.shape[0], 1))
            # X[:, 0] = [2.25, 0.25, 0.25, 2.25, 6.25]
            X[:, 0] = [1.5, 0.5, 0.5, 1.5, 2.5]
            # X[:, 1] = [2.5, 3.5, 4.5, 5.5, 6.5]
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            # print(ols.params)
            quadratic_param[channel] = ols.params[1]
        quadratic_param_array.append(quadratic_param)

    quadratic_param_array = np.array(quadratic_param_array)
    print(quadratic_param_array.shape)

    ch_adjacency, ch_names = find_ch_adjacency(data['info'][1], 'mag')
    df = quadratic_param_array.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    pval = 0.05  # arbitrary
    thresh = scipy.stats.t.ppf(1 - pval, df=df)  # T distribution, one-tailed
    print(thresh)
    channels_indices = np.arange(quadratic_param_array.shape[1])
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(quadratic_param_array,
                                                                     threshold=thresh, n_permutations=1000, tail=1,
                                                                     adjacency=ch_adjacency, t_power=1)
    print(clusters)
    print(cluster_pv)
    clusters = [clusters[arg][0] for arg in range(len(clusters)) if cluster_pv[arg] < 0.05]
    significant_channels = np.unique([channel for cluster in clusters for channel in cluster])

    im, _ = mne.viz.plot_topomap(np.mean(quadratic_param_array, axis=0), data['info'][1], show=False,
                                 mask=np.isin(channels_indices, significant_channels),
                                 mask_params=dict(markersize=8, markeredgewidth=1))
    plt.colorbar(im)
    plt.title("U-shape of correlation with note rate")

    plt.show()


def topo_ushape_similarity_seeg(data_path):

    conditions = ['2.5Hz', '3.5Hz', '4.5Hz', '5.5Hz', '6.5Hz', '7.5Hz']

    data = np.load(data_path, allow_pickle=True)

    quadratic_param = []
    for s, subject in tqdm(enumerate(data["subjects"])):
        # print(subject)
        correlations_array = []
        for c, condition in enumerate(conditions):
            condition_indices = np.array([i for i in range(len(data['stim_names'][s])) if condition in data['stim_names'][s][i]])
            condition_matrix = data['similarity_matrices'][s]  # keep all channels

            condition_matrix = condition_matrix[:, condition_indices, :]  # select condition
            condition_matrix = condition_matrix[condition_indices, :, :]  # select condition
            triang = np.triu_indices_from(condition_matrix[:, :, 0], k=1)
            correlations = np.mean(condition_matrix[triang[0], triang[1], :], axis=0)  # average across stim pairs

            # correlation of condition trials with all trials version
            # condition_matrix = condition_matrix[:, condition_indices, :]  # select condition
            # correlations = np.mean(condition_matrix, axis=(0, 1))  # average across stim pairs

            correlations_array.append(correlations)
        correlations_array = np.array(correlations_array)
        for channel in range(correlations_array.shape[1]):
            Y = correlations_array[:, channel]
            X = np.zeros((Y.shape[0], 1))
            # X[:, 0] = [2.25, 0.25, 0.25, 2.25, 6.25]
            X[:, 0] = [1.5, 0.5, 0.5, 1.5, 2.5, 3.5]
            # X[:, 1] = [2.5, 3.5, 4.5, 5.5, 6.5]
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            # print(ols.params)
            quadratic_param.append(ols.params[1])

    quadratic_param_array = np.array(quadratic_param)
    print(quadratic_param_array.shape)

    print(np.mean(quadratic_param_array))
    print(scipy.stats.ttest_1samp(quadratic_param_array, 0, alternative='greater'))
    plt.hist(quadratic_param_array, bins=100)
    plt.xlabel('Regression parameter (interstimulus correlation ~ distance to 4Hz')
    plt.ylabel('Count (electrode)')

    plt.show()


def topo_feature_similarity(data_path, stim_features_csv, condition):

    stim_features = pd.read_csv(stim_features_csv)
    stim_features = stim_features[['stim_names', condition]]
    # stim_features[condition] = (stim_features[condition] - stim_features[condition].mean()) / stim_features[condition].std()
    print(stim_features)

    # Sort the DataFrame by values
    # stim_features = stim_features.sort_values(by=condition).reset_index(drop=True)

    data = np.load(data_path, allow_pickle=True)

    regression_param_array = []
    for s, subject in tqdm(enumerate(data["subjects"])):
        indices_to_keep = []
        feature_list = []
        for i, stim in enumerate(data['stim_names'][s]):
            if stim in stim_features['stim_names'].tolist():
                indices_to_keep.append(i)
                feature_list.append(stim_features[stim_features['stim_names'] == stim][condition].iloc[0])

        condition_matrix = data['similarity_matrices'][s]  # keep all channels

        condition_matrix = condition_matrix[indices_to_keep, :, :]  # select condition
        correlations_array = np.mean(condition_matrix, axis=1)

        regression_param = np.zeros(correlations_array.shape[1])
        for channel in range(correlations_array.shape[1]):
            Y = correlations_array[:, channel]
            X = np.zeros((Y.shape[0], 1))
            X[:, 0] = feature_list
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            # print(ols.params)
            regression_param[channel] = ols.params[1]
        regression_param_array.append(regression_param)

    regression_param_array = np.array(regression_param_array)
    print(regression_param_array.shape)

    ch_adjacency, ch_names = find_ch_adjacency(data['info'][1], 'mag')
    df = regression_param_array.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    pval = 0.5  # arbitrary
    thresh = scipy.stats.t.ppf(1 - pval, df=df)  # T distribution, two-tailed
    print(thresh)
    channels_indices = np.arange(regression_param_array.shape[1])
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(regression_param_array,
                                                                     threshold=thresh, n_permutations=1000, tail=0,
                                                                     adjacency=ch_adjacency, t_power=1)
    print(np.sum(np.abs(t_obs)>thresh))
    print(clusters)
    print(cluster_pv)
    clusters = [clusters[arg][0] for arg in range(len(clusters)) if cluster_pv[arg] < 0.05]
    significant_channels = np.unique([channel for cluster in clusters for channel in cluster])

    im, _ = mne.viz.plot_topomap(np.mean(regression_param_array, axis=0), data['info'][1], show=False,
                                 mask=np.isin(channels_indices, significant_channels),
                                 mask_params=dict(markersize=8, markeredgewidth=1))
    plt.colorbar(im)
    plt.title(f"Beta param: intertrial correlation ~ {condition}")

    plt.show()


def topo_feature_repetition_similarity(data_path, stim_features_csv, condition):

    stim_features = pd.read_csv(stim_features_csv)
    stim_features = stim_features[['stim_names', condition]]
    # stim_features[condition] = (stim_features[condition] - stim_features[condition].mean()) / stim_features[condition].std()
    print(stim_features)

    # Sort the DataFrame by values
    # stim_features = stim_features.sort_values(by=condition).reset_index(drop=True)

    data = np.load(data_path, allow_pickle=True)

    regression_param_array = []
    for s, subject in tqdm(enumerate(data["subjects"])):
        indices_to_keep = []
        feature_list = []
        correlations_array = []
        stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
        similarity_matrix = data['similarity_matrices'][s]  # keep all channels
        # similarity_matrix = data['stim_amplitude'][s]  # keep all channels
        for i, stim in enumerate(data['stim_names'][s]):
            if (stim in stim_features['stim_names'].tolist()) & ('rep1' in stim):
                indices_to_keep.append(i)
                if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                    if stim[9:].replace('rep1', 'rep3') in stim_names_without_trial_nb:
                        rep2_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                        rep3_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep3'))
                        correlations_array.append((similarity_matrix[i, rep2_index, :] + similarity_matrix[i, rep3_index, :] + similarity_matrix[rep2_index, rep3_index, :]) / 3)  # average of correlation of 3 reps
                        # correlations_array.append((similarity_matrix[i, :] + similarity_matrix[rep2_index, :] + similarity_matrix[rep3_index, :]) / 3)  # average amplitude 3 reps
                        feature_list.append(stim_features[stim_features['stim_names'] == stim][condition].iloc[0])

        correlations_array = np.array(correlations_array)

        regression_param = np.zeros(correlations_array.shape[1])
        for channel in range(correlations_array.shape[1]):
            Y = correlations_array[:, channel]
            X = np.zeros((Y.shape[0], 1))
            X[:, 0] = feature_list
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            regression_param[channel] = ols.params[1]
        regression_param_array.append(regression_param)

    regression_param_array = np.array(regression_param_array)
    print(regression_param_array.shape)

    ch_adjacency, ch_names = find_ch_adjacency(data['info'][1], 'mag')
    df = regression_param_array.shape[0] - 1  # degrees of freedom numerator: n_observations - 1
    pval = 0.05  # arbitrary
    thresh = scipy.stats.t.ppf(1 - pval/2, df=df)  # T distribution, two-tailed
    print(thresh)
    channels_indices = np.arange(regression_param_array.shape[1])
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(regression_param_array,
                                                                     threshold=thresh, n_permutations=1000, tail=0,
                                                                     adjacency=ch_adjacency, t_power=1)
    print(np.sum(np.abs(t_obs)>thresh))
    print(clusters)
    print(cluster_pv)
    clusters = [clusters[arg][0] for arg in range(len(clusters)) if cluster_pv[arg] < 0.05]
    significant_channels = np.unique([channel for cluster in clusters for channel in cluster])

    im, _ = mne.viz.plot_topomap(np.mean(regression_param_array, axis=0), data['info'][1], show=False,
                                 mask=np.isin(channels_indices, significant_channels),
                                 mask_params=dict(markersize=8, markeredgewidth=1))
    plt.colorbar(im)
    plt.title(f"Beta param: intrastim correlation ~ {condition}")
    # plt.title(f"Beta param: average RMS ~ {condition}")

    plt.show()


def feature_dissimilarity_seeg(data_path, stim_features_csv, condition):

    stim_features = pd.read_csv(stim_features_csv)
    stim_features = stim_features[['stim_name', 'subject', condition, 'correct']]
    # stim_features[condition] = (stim_features[condition] - stim_features[condition].mean()) / stim_features[condition].std()
    print(stim_features)

    # Sort the DataFrame by values
    # stim_features = stim_features.sort_values(by=condition).reset_index(drop=True)

    data = np.load(data_path, allow_pickle=True)

    channel_positions_list = []
    subject_id = []
    regression_param_array = []
    pvalues_array = []
    for s, subject in tqdm(enumerate(data["subjects"])):
        subject_stim_features = stim_features[stim_features['subject'] == subject]
        subject_info = data['info'][s]
        indices_to_keep = []
        feature_list = []
        for i, stim in enumerate(data['stim_names'][s]):
            if stim in subject_stim_features['stim_name'].tolist():
                if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                    indices_to_keep.append(i)
                    feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])

        condition_matrix = data['similarity_matrices'][s]  # keep all channels

        condition_matrix = condition_matrix[indices_to_keep, :, :]  # select condition
        condition_matrix = condition_matrix[:, indices_to_keep, :]  # select condition
        # Loop over the third dimension to apply the mask slice by slice
        correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
        for i in range(condition_matrix.shape[0]):
            current_corr = condition_matrix[i, :, :]  # ignore intra stim correlation
            mask = np.ones(condition_matrix.shape[1], dtype=bool)
            mask[i] = False
            correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)

        regression_param = []
        pvalues = np.zeros(correlations_array.shape[1])
        for channel in range(correlations_array.shape[1]):
            Y = correlations_array[:, channel]
            X = np.zeros((Y.shape[0], 1))
            # X = np.zeros((Y.shape[0], 2))
            X[:, 0] = feature_list
            # X[:, 1] = np.array(feature_list)**2
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            # print(ols.params)
            regression_param.append(ols.params[1])
            # regression_param.append(ols.params[2])
            pvalues[channel] = ols.pvalues[1]
            channel_positions_list.append(subject_info['chs'][channel]['loc'][:3])
            subject_id.append(s)
        regression_param_array += regression_param
        pvalues_array.append(pvalues)

    regression_param_array = np.array(regression_param_array)
    print(regression_param_array.shape)
    pvalues_array = np.concatenate(pvalues_array)
    print(np.sum(pvalues_array < 0.05), 'significant electrodes,', round(np.sum(pvalues_array < 0.05) / pvalues_array.shape[0] * 100, 3), "%")
    print(regression_param_array[pvalues_array < 0.05])

    # regression_param_array = np.concatenate(regression_param_array)
    # print(regression_param_array.shape)

    print(np.mean(regression_param_array))
    ttest = scipy.stats.ttest_1samp(regression_param_array, 0, alternative='two-sided')
    plt.hist(regression_param_array, bins=100)
    plt.gca().axvline(0, ls='--', color='k')
    plt.xlabel('Regression parameter (interstimulus correlation ~ surprise)')
    plt.ylabel('Count (electrode)')
    plt.text(plt.gca().get_xlim()[0], plt.gca().get_ylim()[1],f'regression beta = {round(np.mean(regression_param_array), 5)}, p = {round(ttest.pvalue, 5)}')

    fig, axs = plt.subplots(1, 3, sharey=True)
    dimensions = ['left-right (abs: distance from center)', 'posterior-anterior', 'ventral-dorsal']
    subject_colors = plt.get_cmap('Set1').colors
    colors = [subject_colors[s] for s in subject_id]
    for d, dimension_name in enumerate(dimensions):
        dim = [positions[d] for positions in channel_positions_list]
        if d == 0:
            dim = np.abs(dim)  # for left-right, we transform position to distance from center
        axs[d].scatter(dim, regression_param_array, color=colors)
        axs[d].set_xlabel(dimension_name)

        # Create a DataFrame for the mixed model
        data_dict = {
            'Value': regression_param_array,
            'Position': dim,
            'SubjectID': subject_id
        }
        model_data = pd.DataFrame(data_dict)
        model = MixedLM.from_formula('Value ~ Position', groups='SubjectID', data=model_data)
        result = model.fit()
        p_value = result.pvalues['Position']

        # fitted line
        x_vals = np.linspace(min(dim), max(dim), 100)
        y_vals = result.params['Intercept'] + result.params['Position'] * x_vals
        axs[d].plot(x_vals, y_vals, color='k', linestyle='--', linewidth=1.5)

        box_color = 'r' if p_value < 0.05 else 'white'
        axs[d].text(
            0.05, 0.95, f'p={p_value:.3g}', transform=axs[d].transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8)
        )
    axs[0].set_ylabel('Regression parameter (interstimulus correlation ~ surprise)')

    plt.show()

def feature_repetition_similarity_seeg(data_path, stim_features_csv, condition):

    stim_features = pd.read_csv(stim_features_csv)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    # stim_features[condition] = (stim_features[condition] - stim_features[condition].mean()) / stim_features[condition].std()
    print(stim_features)

    # Sort the DataFrame by values
    # stim_features = stim_features.sort_values(by=condition).reset_index(drop=True)

    data = np.load(data_path, allow_pickle=True)

    channel_positions_list = []
    subject_id = []
    regression_param_array = []
    aic_array = []
    pvalues_array = []
    for s, subject in tqdm(enumerate(data["subjects"])):
        subject_stim_features = stim_features[stim_features['subject'] == subject]
        subject_info = data['info'][s]
        indices_to_keep = []
        feature_list = []
        correlations_array = []
        stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
        similarity_matrix = data['similarity_matrices'][s]  # keep all channels
        # similarity_matrix = data['stim_amplitude'][s]  # keep all channels
        for i, stim in enumerate(data['stim_names'][s]):
            if 'rep2' in stim:
                # if subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0] == 1:
                    indices_to_keep.append(i)
                    if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                        rep1_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                        correlations_array.append(similarity_matrix[i, rep1_index, :])  # correlation of 2 reps
                        feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])

        correlations_array = np.array(correlations_array)

        regression_param = np.zeros(correlations_array.shape[1])
        pvalues = np.zeros(correlations_array.shape[1])
        for channel in range(correlations_array.shape[1]):
            Y = correlations_array[:, channel]
            X = np.zeros((Y.shape[0], 1))
            # X = np.zeros((Y.shape[0], 2))
            X[:, 0] = feature_list
            # X[:, 1] = np.array(feature_list)**2
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            regression_param[channel] = ols.params[1]
            # regression_param[channel] = ols.params[2]
            pvalues[channel] = ols.pvalues[1]
            aic_array.append(ols.aic)
            channel_positions_list.append(subject_info['chs'][channel]['loc'][:3])
            subject_id.append(s)
        regression_param_array.append(regression_param)
        pvalues_array.append(pvalues)

    regression_param_array = np.concatenate(regression_param_array)
    print(regression_param_array.shape)
    aic_array = np.array(aic_array)
    pvalues_array = np.concatenate(pvalues_array)
    print(np.sum(pvalues_array < 0.05), 'significant electrodes,', round(np.sum(pvalues_array < 0.05) / pvalues_array.shape[0] * 100, 3), "%")
    print(regression_param_array[pvalues_array < 0.05])

    print(np.mean(regression_param_array))
    ttest = scipy.stats.ttest_1samp(regression_param_array, 0, alternative='two-sided')
    print(scipy.stats.ttest_1samp(regression_param_array, 0, alternative='two-sided'))
    plt.hist(regression_param_array, bins=100)
    plt.gca().axvline(0, ls='--', color='k')
    plt.xlabel('Regression parameter (intrastimulus correlation ~ surprise)')
    # plt.xlabel('Regression parameter (intrastimulus correlation ~ surprise²)')
    plt.ylabel('Count (electrode)')
    plt.text(plt.gca().get_xlim()[0], plt.gca().get_ylim()[1], f'regression beta = {round(np.mean(regression_param_array), 5)}, p = {round(ttest.pvalue, 5)}, mean AIC = {round(np.mean(aic_array), 5)}')

    fig, axs = plt.subplots(1, 3, sharey=True)
    dimensions = ['left-right (abs: distance from center)', 'posterior-anterior', 'ventral-dorsal']
    subject_colors = plt.get_cmap('Set1').colors
    colors = [subject_colors[s] for s in subject_id]
    for d, dimension_name in enumerate(dimensions):
        dim = [positions[d] for positions in channel_positions_list]
        if d == 0:
            dim = np.abs(dim)  # for left-right, we transform position to distance from center
        axs[d].scatter(dim, regression_param_array, color=colors)
        axs[d].set_xlabel(dimension_name)

        # Create a DataFrame for the mixed model
        data_dict = {
            'Value': regression_param_array,
            'Position': dim,
            'SubjectID': subject_id
        }
        model_data = pd.DataFrame(data_dict)
        model = MixedLM.from_formula('Value ~ Position', groups='SubjectID', data=model_data)
        result = model.fit()
        p_value = result.pvalues['Position']

        # fitted line
        x_vals = np.linspace(min(dim), max(dim), 100)
        y_vals = result.params['Intercept'] + result.params['Position'] * x_vals
        axs[d].plot(x_vals, y_vals, color='k', linestyle='--', linewidth=1.5)

        box_color = 'r' if p_value<0.05 else 'white'
        axs[d].text(
            0.05, 0.95, f'p={p_value:.3g}', transform=axs[d].transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8)
        )
    axs[0].set_ylabel('Regression parameter (intrastimulus correlation ~ surprise)')

    plt.show()


def repetition_similarity_and_correct_seeg(data_path, stim_features_csv):

    stim_features = pd.read_csv(stim_features_csv)
    stim_features = stim_features[['stim_name', 'subject', "correct"]]
    # stim_features[condition] = (stim_features[condition] - stim_features[condition].mean()) / stim_features[condition].std()
    print(stim_features)

    # Sort the DataFrame by values
    # stim_features = stim_features.sort_values(by=condition).reset_index(drop=True)

    data = np.load(data_path, allow_pickle=True)

    channel_positions_list = []
    subject_id = []
    regression_param_array = []
    pvalues_array = []  # p value for the regression
    p_ttest_array = []  # p value of the contrast correct vs incorrect
    for s, subject in tqdm(enumerate(data["subjects"])):
        subject_stim_features = stim_features[stim_features['subject'] == subject]
        subject_info = data['info'][s]
        indices_to_keep = []
        feature_list = []
        correlations_array = []
        stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
        similarity_matrix = data['similarity_matrices'][s]  # keep all channels
        # similarity_matrix = data['stim_amplitude'][s]  # keep all channels
        for i, stim in enumerate(data['stim_names'][s]):
            if 'rep2' in stim:
                if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                    indices_to_keep.append(i)
                    if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                        rep1_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                        correlations_array.append(similarity_matrix[i, rep1_index, :])  # correlation of 2 reps
                        feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])

        correlations_array = np.array(correlations_array)

        regression_param = np.zeros(correlations_array.shape[1])
        pvalues = np.zeros(correlations_array.shape[1])
        p_ttest = np.zeros(correlations_array.shape[1])
        for channel in range(correlations_array.shape[1]):
            Y = correlations_array[:, channel]
            X = np.zeros((Y.shape[0], 1))
            # X = np.zeros((Y.shape[0], 2))
            X[:, 0] = feature_list
            # X[:, 1] = np.array(feature_list)**2
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            regression_param[channel] = ols.params[1]
            pvalues[channel] = ols.pvalues[1]
            # regression_param[channel] = ols.params[2]
            channel_positions_list.append(subject_info['chs'][channel]['loc'][:3])
            subject_id.append(s)
            # independent t-test
            correct_corr = Y[X[:, 1] == 1]  # axis 0 is the intercept
            incorrect_corr = Y[X[:, 1] == 0]
            p_ttest[channel] = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=1000, random_state=None, alternative='greater').pvalue  # testing of individual electrodes
        regression_param_array.append(regression_param)
        pvalues_array.append(pvalues)
        p_ttest_array.append(p_ttest)

    regression_param_array = np.concatenate(regression_param_array)
    print(regression_param_array.shape)
    pvalues_array = np.concatenate(pvalues_array)
    print(np.sum(pvalues_array<0.05), 'significant electrodes,', round(np.sum(pvalues_array<0.05)/pvalues_array.shape[0]*100, 3), "%")
    print(regression_param_array[pvalues_array<0.05])
    p_ttest_array = np.concatenate(p_ttest_array)
    print(np.sum(p_ttest_array < 0.05), 'significant electrodes with independent ttest,', round(np.sum(p_ttest_array < 0.05) / p_ttest_array.shape[0] * 100, 3), "%")
    print(regression_param_array[p_ttest_array < 0.05])

    print(np.mean(regression_param_array))
    ttest = scipy.stats.ttest_1samp(regression_param_array, 0, alternative='two-sided')
    print(scipy.stats.ttest_1samp(regression_param_array, 0, alternative='two-sided'))
    plt.hist(regression_param_array, bins=100)
    plt.gca().axvline(0, ls='--', color='k')
    # plt.xlabel('Regression parameter (intrastimulus correlation ~ surprise)')
    plt.xlabel('Regression parameter (intrastimulus correlation ~ correct)')
    plt.ylabel('Count (electrode)')
    plt.text(plt.gca().get_xlim()[0], plt.gca().get_ylim()[1], f'regression beta = {round(np.mean(regression_param_array), 5)}, p = {round(ttest.pvalue, 5)}')

    fig, axs = plt.subplots(1, 3, sharey=True)
    dimensions = ['left-right (abs: distance from center)', 'posterior-anterior', 'ventral-dorsal']
    subject_colors = plt.get_cmap('Set1').colors
    colors = [subject_colors[s] for s in subject_id]
    for d, dimension_name in enumerate(dimensions):
        dim = [positions[d] for positions in channel_positions_list]
        if d == 0:
            dim = np.abs(dim)  # for left-right, we transform position to distance from center
        axs[d].scatter(dim, regression_param_array, color=colors)
        axs[d].set_xlabel(dimension_name)

        # Create a DataFrame for the mixed model
        data_dict = {
            'Value': regression_param_array,
            'Position': dim,
            'SubjectID': subject_id
        }
        model_data = pd.DataFrame(data_dict)
        model = MixedLM.from_formula('Value ~ Position', groups='SubjectID', data=model_data)
        result = model.fit()
        p_value = result.pvalues['Position']

        # fitted line
        x_vals = np.linspace(min(dim), max(dim), 100)
        y_vals = result.params['Intercept'] + result.params['Position'] * x_vals
        axs[d].plot(x_vals, y_vals, color='k', linestyle='--', linewidth=1.5)

        box_color = 'r' if p_value < 0.05 else 'white'
        axs[d].text(
            0.05, 0.95, f'p={p_value:.3g}', transform=axs[d].transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8)
        )
    axs[0].set_ylabel('Regression parameter (intrastimulus correlation ~ correct)')

    plt.show()

def repetition_dissimilarity_and_correct_seeg(data_path, stim_features_csv):

    stim_features = pd.read_csv(stim_features_csv)
    stim_features = stim_features[['stim_name', 'subject', "correct"]]
    # stim_features[condition] = (stim_features[condition] - stim_features[condition].mean()) / stim_features[condition].std()
    print(stim_features)

    # Sort the DataFrame by values
    # stim_features = stim_features.sort_values(by=condition).reset_index(drop=True)

    data = np.load(data_path, allow_pickle=True)

    channel_positions_list = []
    subject_id = []
    regression_param_array = []
    pvalues_array = []  # p value for the regression
    p_ttest_array = []  # p value of the contrast correct vs incorrect
    for s, subject in tqdm(enumerate(data["subjects"])):
        subject_stim_features = stim_features[stim_features['subject'] == subject]
        subject_info = data['info'][s]
        indices_to_keep = []
        feature_list = []
        similarity_matrix = data['similarity_matrices'][s]  # keep all channels
        # similarity_matrix = data['stim_amplitude'][s]  # keep all channels
        for i, stim in enumerate(data['stim_names'][s]):
            if stim in subject_stim_features['stim_name'].tolist():
                if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                    indices_to_keep.append(i)
                    feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])

        condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
        condition_matrix = condition_matrix[:, indices_to_keep, :]  # select condition
        correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
        for i in range(condition_matrix.shape[0]):
            current_corr = condition_matrix[i, :, :]  # ignore intra stim correlation
            mask = np.ones(condition_matrix.shape[1], dtype=bool)
            mask[i] = False
            correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)

        regression_param = np.zeros(correlations_array.shape[1])
        pvalues = np.zeros(correlations_array.shape[1])
        p_ttest = np.zeros(correlations_array.shape[1])
        for channel in range(correlations_array.shape[1]):
            Y = correlations_array[:, channel]
            X = np.zeros((Y.shape[0], 1))
            # X = np.zeros((Y.shape[0], 2))
            X[:, 0] = feature_list
            # X[:, 1] = np.array(feature_list)**2
            X = sm.add_constant(X)
            ols = sm.OLS(Y, X).fit()
            regression_param[channel] = ols.params[1]
            # regression_param[channel] = ols.params[2]
            pvalues[channel] = ols.pvalues[1]
            channel_positions_list.append(subject_info['chs'][channel]['loc'][:3])
            subject_id.append(s)
            # independent t-test
            correct_corr = Y[X[:, 1] == 1]  # axis 0 is the intercept
            incorrect_corr = Y[X[:, 1] == 0]
            p_ttest[channel] = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=1000, random_state=None, alternative='less').pvalue  # testing of individual electrodes
        regression_param_array.append(regression_param)
        pvalues_array.append(pvalues)
        p_ttest_array.append(p_ttest)

    regression_param_array = np.concatenate(regression_param_array)
    print(regression_param_array.shape)
    pvalues_array = np.concatenate(pvalues_array)
    print(np.sum(pvalues_array < 0.05), 'significant electrodes,', round(np.sum(pvalues_array < 0.05) / pvalues_array.shape[0] * 100, 3), "%")
    print(regression_param_array[pvalues_array < 0.05])
    p_ttest_array = np.concatenate(p_ttest_array)
    print(np.sum(p_ttest_array < 0.05), 'significant electrodes with independent ttest,', round(np.sum(p_ttest_array < 0.05) / p_ttest_array.shape[0] * 100, 3), "%")
    print(regression_param_array[p_ttest_array < 0.05])

    print(np.mean(regression_param_array))
    ttest = scipy.stats.ttest_1samp(regression_param_array, 0, alternative='two-sided')
    print(scipy.stats.ttest_1samp(regression_param_array, 0, alternative='two-sided'))
    plt.hist(regression_param_array, bins=100)
    plt.gca().axvline(0, ls='--', color='k')
    # plt.xlabel('Regression parameter (intrastimulus correlation ~ surprise)')
    plt.xlabel('Regression parameter (interstim correlation ~ correct)')
    plt.ylabel('Count (electrode)')
    plt.text(plt.gca().get_xlim()[0], plt.gca().get_ylim()[1], f'regression beta = {round(np.mean(regression_param_array), 5)}, p = {round(ttest.pvalue, 5)}')

    fig, axs = plt.subplots(1, 3, sharey=True)
    dimensions = ['left-right (abs: distance from center)', 'posterior-anterior', 'ventral-dorsal']
    subject_colors = plt.get_cmap('Set1').colors
    colors = [subject_colors[s] for s in subject_id]
    for d, dimension_name in enumerate(dimensions):
        dim = [positions[d] for positions in channel_positions_list]
        if d == 0:
            dim = np.abs(dim)  # for left-right, we transform position to distance from center
        axs[d].scatter(dim, regression_param_array, color=colors)
        axs[d].set_xlabel(dimension_name)

        # Create a DataFrame for the mixed model
        data_dict = {
            'Value': regression_param_array,
            'Position': dim,
            'SubjectID': subject_id
        }
        model_data = pd.DataFrame(data_dict)
        model = MixedLM.from_formula('Value ~ Position', groups='SubjectID', data=model_data)
        result = model.fit()
        p_value = result.pvalues['Position']

        # fitted line
        x_vals = np.linspace(min(dim), max(dim), 100)
        y_vals = result.params['Intercept'] + result.params['Position'] * x_vals
        axs[d].plot(x_vals, y_vals, color='k', linestyle='--', linewidth=1.5)

        box_color = 'r' if p_value < 0.05 else 'white'
        axs[d].text(
            0.05, 0.95, f'p={p_value:.3g}', transform=axs[d].transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8)
        )
    axs[0].set_ylabel('Regression parameter (interstim correlation ~ correct)')

    plt.show()


def plot_similarity_matrices(rms_path, behavior_path, similarity_path, distance_path, regions, first_parcel_label=True):
    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    subject_names = []
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate(
                [[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            # pval = ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False,
                            #                                permutations=10000, random_state=None,
                            #                                alternative='greater').pvalue  # testing of individual electrodes
                            # if pval < 0.05:
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(
            significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    print(significant_electrodes)
    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])
    print(chan_count)

    ############## Intrastim correlation -> correct ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(similarity_path, allow_pickle=True)

    channel_positions_list = []
    subject_id = []
    regression_param_correct = []
    regression_param_feature = []
    all_correlations = []
    # pvalues_array = []  # p value for the regression
    p_ttest_array = []  # p value of the contrast correct vs incorrect
    for s, subject in tqdm(enumerate(data["subjects"])):
        subject_stim_features = stim_features[stim_features['subject'] == subject]
        subject_info = data['info'][s]
        ch_names = subject_info['ch_names']
        indices_to_keep = []
        correct_list = []
        feature_list = []
        correlations_array = []
        stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
        similarity_matrix = data['similarity_matrices'][s]  # keep all channels
        # similarity_matrix = data['distance_matrices'][s]  # keep all channels
        # similarity_matrix = 1/similarity_matrix

        first_trials_to_remove = 50
        for i, stim in enumerate(data['stim_names'][s]):
            if i > first_trials_to_remove:  # remove first trials
                if stim in subject_stim_features['stim_name'].tolist():
                    if 'rep2' in stim:
                        if not np.isnan(
                                subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                            # if subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0] < 0.15:  # check that it works without extreme trials
                            indices_to_keep.append(i)
                            if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                rep1_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                correlations_array.append(similarity_matrix[i, rep1_index, :])  # correlation of 2 reps
                                correct_list.append(
                                    subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[
                                        0])
                                feature_list.append(
                                    subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[
                                        0])

        correlations_array = np.array(correlations_array)

        # pvalues = np.zeros(correlations_array.shape[1])
        p_ttest = np.zeros(correlations_array.shape[1])
        for channel in range(correlations_array.shape[1]):
            if ch_names[channel] in significant_electrodes[subject]:
                # Y = correlations_array[:, channel]
                # all_correlations += Y.tolist()
                # Y = (Y - Y.mean()) / Y.std()
                # X = np.zeros((Y.shape[0], 1))
                # # X = np.zeros((Y.shape[0], 2))
                # X[:, 0] = correct_list
                # # X[:, 1] = np.array(feature_list)**2
                # X = sm.add_constant(X)

                Y = np.array(correct_list)
                X = np.zeros((Y.shape[0], 1))
                X[:, 0] = correlations_array[:, channel]
                all_correlations += correlations_array[:, channel].tolist()
                X = (X - X.mean()) / X.std()
                X = sm.add_constant(X)

                # # independent t-test
                # correct_corr = Y[X[:, 1] == 1]  # axis 0 is the intercept
                # incorrect_corr = Y[X[:, 1] == 0]
                # p_ttest[channel] = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=1000,
                #                              random_state=None,
                #                              # alternative='greater').pvalue  # testing of individual electrodes
                #                              alternative='less').pvalue  # testing of individual electrodes
                # if p_ttest[channel] < 0.05:

                # ols = sm.OLS(Y, X).fit()
                ols = sm.Logit(Y, X).fit()
                regression_param_correct.append(ols.params[1])
                # pvalues[channel] = ols.pvalues[1]
                # regression_param[channel] = ols.params[2]

                Y = X[:, 1]
                X = np.zeros((Y.shape[0], 1))
                # X = np.zeros((Y.shape[0], 2))
                X[:, 0] = feature_list
                # X[:, 1] = np.array(feature_list)**2
                X = (X - X.mean()) / X.std()
                X = sm.add_constant(X)
                ols = sm.OLS(Y, X).fit()
                regression_param_feature.append(ols.params[1])

                channel_positions_list.append(subject_info['chs'][channel]['loc'][:3])
                subject_id.append(s)
        # pvalues_array.append(pvalues)
        # p_ttest_array.append(p_ttest)

    regression_param_correct = np.array(regression_param_correct)
    print(regression_param_correct.shape)
    regression_param_feature = np.array(regression_param_feature)
    print(regression_param_feature.shape)
    # pvalues_array = np.concatenate(pvalues_array)
    # print(np.sum(pvalues_array < 0.05), 'significant electrodes,',
    #       round(np.sum(pvalues_array < 0.05) / pvalues_array.shape[0] * 100, 3), "%")
    # print(regression_param_array[pvalues_array < 0.05])
    # p_ttest_array = np.concatenate(p_ttest_array)
    # print(np.sum(p_ttest_array < 0.05), 'significant electrodes with independent ttest,',
    #       round(np.sum(p_ttest_array < 0.05) / p_ttest_array.shape[0] * 100, 3), "%")
    # print(regression_param_array[p_ttest_array < 0.05])

    list_of_pvalues = []

    # histogram of intrastim correlation
    fig, axs = plt.subplots(1, 1)
    all_correlations = np.array(all_correlations)
    print(all_correlations.shape)
    axs.hist(all_correlations, bins=100)
    ttest = scipy.stats.ttest_1samp(all_correlations, 0, alternative='two-sided')
    axs.axvline(0, ls='--', color='k')
    axs.axvline(np.mean(all_correlations), color='r', lw=2)
    axs.set_xlabel('Intrastimulus correlation')
    axs.set_ylabel('Count (epochs)')
    axs.text(plt.gca().get_xlim()[0], plt.gca().get_ylim()[1],
             f'Average correlation = {round(np.mean(all_correlations), 5)}, p = {round(ttest.pvalue, 5)}')

    fig, axs = plt.subplots(1, 2, sharey=True)
    x_regression = ['correct ~ intrastimulus correlation', f'intrastimulus correlation ~ {condition}']
    alternative = ['greater', 'less']
    for i, regression_param_array in enumerate([regression_param_correct, regression_param_feature]):
        print(np.mean(regression_param_array))
        ttest = scipy.stats.ttest_1samp(regression_param_array, 0, alternative=alternative[i])
        print(scipy.stats.ttest_1samp(regression_param_array, 0, alternative=alternative[i]))
        list_of_pvalues.append(ttest.pvalue)
        axs[i].hist(regression_param_array, bins=100)
        axs[i].axvline(0, ls='--', color='k')
        axs[i].axvline(np.mean(regression_param_array), color='r', lw=2)
        # plt.xlabel('Regression parameter (intrastimulus correlation ~ surprise)')
        # axs[i].set_xlabel(f'Regression parameter (intrastimulus correlation ~ {x_regression[i]})')
        axs[i].set_xlabel(f'Regression parameter ({x_regression[i]})')
        axs[i].set_ylabel('Count (electrode)')
        axs[i].text(plt.gca().get_xlim()[0], plt.gca().get_ylim()[1],
                    f'regression beta = {round(np.mean(regression_param_array), 5)}, p = {round(ttest.pvalue, 5)}')

    fig, axs = plt.subplots(2, 3, sharey='row')
    for i, regression_param_array in enumerate([regression_param_correct, regression_param_feature]):
        dimensions = ['left-right (abs: distance from center)', 'posterior-anterior', 'ventral-dorsal']
        subject_colors = plt.get_cmap('Set3').colors
        colors = [subject_colors[s] for s in subject_id]
        for d, dimension_name in enumerate(dimensions):
            dim = [positions[d] for positions in channel_positions_list]
            if d == 0:
                dim = np.abs(dim)  # for left-right, we transform position to distance from center
            axs[i, d].scatter(dim, regression_param_array, color=colors)
            axs[i, d].set_xlabel(dimension_name)

            # Create a DataFrame for the mixed model
            data_dict = {
                'Value': regression_param_array,
                'Position': dim,
                'SubjectID': subject_id
            }
            model_data = pd.DataFrame(data_dict)
            model = MixedLM.from_formula('Value ~ Position', groups='SubjectID', data=model_data)
            result = model.fit()
            p_value = result.pvalues['Position']

            # fitted line
            x_vals = np.linspace(min(dim), max(dim), 100)
            y_vals = result.params['Intercept'] + result.params['Position'] * x_vals
            axs[i, d].plot(x_vals, y_vals, color='k', linestyle='--', linewidth=1.5)

            box_color = 'r' if p_value < 0.05 else 'white'
            axs[i, d].text(
                0.05, 0.95, f'p={p_value:.3g}', transform=axs[i, d].transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8)
            )
        axs[i, 0].set_ylabel(f'Regression parameter (intrastimulus correlation ~ {x_regression[i]})')

    # plt.show()

    intrastim_params = regression_param_correct.copy()
    intrastim_feature_params = regression_param_feature.copy()

    ############## Interstim correlation -> incorrect ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(distance_path, allow_pickle=True)

    channel_positions_list = []
    subject_id = []
    regression_param_correct = []
    regression_param_feature = []
    # pvalues_array = []  # p value for the regression
    # p_ttest_array = []  # p value of the contrast correct vs incorrect
    for s, subject in tqdm(enumerate(data["subjects"])):

        # if not subject == 'Bou_Ni':

        subject_stim_features = stim_features[stim_features['subject'] == subject]
        subject_info = data['info'][s]
        ch_names = subject_info['ch_names']
        indices_to_keep = []
        correct_list = []
        feature_list = []
        stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
        # similarity_matrix = data['similarity_matrices'][s]  # keep all channels
        # similarity_matrix = -similarity_matrix  # transform into interstim distance
        # similarity_matrix = similarity_matrix**2  # transform into interstim distance
        # similarity_matrix = 1/similarity_matrix**2  # transform into interstim distance
        similarity_matrix = data['distance_matrices'][s]  # keep all channels
        paired_stim = []
        for i, stim in enumerate(data['stim_names'][s]):
            if i > first_trials_to_remove:  # remove first trials
                if ('rep1' in stim) | ('rep2' in stim):
                    # if 'rep1' in stim:
                    if stim in subject_stim_features['stim_name'].tolist():
                        if not np.isnan(
                                subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                            # if subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0] < 0.15:  # check that it works without extreme trials
                            indices_to_keep.append(i)
                            correct_list.append(
                                subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                            feature_list.append(
                                subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                            if 'rep2' in stim:
                                # indices_to_keep.append(i)
                                # correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                # feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                    paired_stim.append(
                                        stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))
                                else:
                                    paired_stim.append('no_pair')
                            if 'rep1' in stim:
                                # indices_to_keep.append(i)
                                # correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                # feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                                    paired_stim.append(
                                        stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2')))
                                else:
                                    paired_stim.append('no_pair')
                            # else:
                            #     raise ValueError('Neither rep1 nor rep2???')
        # plt.imshow(np.mean(similarity_matrix, axis=-1), cmap='gray')
        # plt.show()
        condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
        # condition_matrix = condition_matrix[:, indices_to_keep, :]  # select condition
        correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
        for i in range(condition_matrix.shape[0]):
            current_corr = condition_matrix[i, :, :]
            mask = np.ones(condition_matrix.shape[1], dtype=bool)
            mask[indices_to_keep[i]] = False  # remove current stim
            # mask[i] = False  # remove current stim
            # if not paired_stim[i] == 'no_pair':
            #     mask[paired_stim[i]] = False  # remove pair
            # for j in range(indices_to_keep[i], mask.shape[0]):
            #     mask[j] = False  # remove following trials
            mask[indices_to_keep[i]:] = False  # remove following trials
            # plt.hist(current_corr[mask, 10], bins=100)
            # plt.show()
            # print(np.mean(current_corr[mask, :], axis=0))
            # print(current_corr.shape)
            # print(mask)
            # print(current_corr[mask, :].shape)
            correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)
        # print(correlations_array.shape)
        # plt.figure()
        # plt.plot(correlations_array)
        # plt.show()

        # pvalues = np.zeros(correlations_array.shape[1])
        # p_ttest = np.zeros(correlations_array.shape[1])
        for channel in range(correlations_array.shape[1]):
            if ch_names[channel] in significant_electrodes[subject]:
                # Y = correlations_array[:, channel]
                # Y = (Y - Y.mean()) / Y.std()
                # X = np.zeros((Y.shape[0], 1))
                # # X = np.zeros((Y.shape[0], 2))
                # X[:, 0] = correct_list
                # # X[:, 1] = np.array(feature_list)**2
                # X = sm.add_constant(X)

                Y = np.array(correct_list)
                X = np.zeros((Y.shape[0], 1))
                X[:, 0] = correlations_array[:, channel]
                X = (X - X.mean()) / X.std()
                X = sm.add_constant(X)

                # # independent t-test
                # correct_corr = Y[X[:, 1] == 1]  # axis 0 is the intercept
                # incorrect_corr = Y[X[:, 1] == 0]
                # p_ttest[channel] = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=1000,
                #                              random_state=None,
                #                              # alternative='less').pvalue  # testing of individual electrodes
                #                              alternative='greater').pvalue  # testing of individual electrodes
                # if p_ttest[channel] < 0.05:

                # ols = sm.OLS(Y, X).fit()
                ols = sm.Logit(Y, X).fit()
                regression_param_correct.append(ols.params[1])
                # pvalues[channel] = ols.pvalues[1]
                # regression_param[channel] = ols.params[2]

                Y = X[:, 1]
                X = np.zeros((Y.shape[0], 1))
                # X = np.zeros((Y.shape[0], 2))
                X[:, 0] = feature_list
                # X[:, 1] = np.array(feature_list)**2
                X = (X - X.mean()) / X.std()
                X = sm.add_constant(X)
                ols = sm.OLS(Y, X).fit()
                regression_param_feature.append(ols.params[1])

                channel_positions_list.append(subject_info['chs'][channel]['loc'][:3])
                subject_id.append(s)
        # pvalues_array.append(pvalues)
        # p_ttest_array.append(p_ttest)

    regression_param_correct = np.array(regression_param_correct)
    print(regression_param_correct.shape)
    regression_param_feature = np.array(regression_param_feature)
    print(regression_param_feature.shape)
    # pvalues_array = np.concatenate(pvalues_array)
    # print(np.sum(pvalues_array < 0.05), 'significant electrodes,',
    #       round(np.sum(pvalues_array < 0.05) / pvalues_array.shape[0] * 100, 3), "%")
    # print(regression_param_array[pvalues_array < 0.05])
    # p_ttest_array = np.concatenate(p_ttest_array)
    # print(np.sum(p_ttest_array < 0.05), 'significant electrodes with independent ttest,',
    #       round(np.sum(p_ttest_array < 0.05) / p_ttest_array.shape[0] * 100, 3), "%")
    # print(regression_param_array[p_ttest_array < 0.05])

    fig, axs = plt.subplots(1, 2, sharey=True)
    x_regression = ['correct ~ interstimulus distance', f'interstimulus distance ~ {condition}']
    alternative = ['greater', 'greater']
    for i, regression_param_array in enumerate([regression_param_correct, regression_param_feature]):
        print(np.mean(regression_param_array))
        ttest = scipy.stats.ttest_1samp(regression_param_array, 0, alternative=alternative[i])
        print(scipy.stats.ttest_1samp(regression_param_array, 0, alternative=alternative[i]))
        list_of_pvalues.append(ttest.pvalue)
        axs[i].hist(regression_param_array, bins=100)
        axs[i].axvline(0, ls='--', color='k')
        axs[i].axvline(np.mean(regression_param_array), color='r', lw=2)
        # plt.xlabel('Regression parameter (intrastimulus correlation ~ surprise)')
        axs[i].set_xlabel(f'Regression parameter ({x_regression[i]})')
        axs[i].set_ylabel('Count (electrode)')
        axs[i].text(plt.gca().get_xlim()[0], plt.gca().get_ylim()[1],
                    f'regression beta = {round(np.mean(regression_param_array), 5)}, p = {round(ttest.pvalue, 5)}')

    fig, axs = plt.subplots(2, 3, sharey='row')
    for i, regression_param_array in enumerate([regression_param_correct, regression_param_feature]):
        dimensions = ['left-right (abs: distance from center)', 'posterior-anterior', 'ventral-dorsal']
        subject_colors = plt.get_cmap('Set3').colors
        colors = [subject_colors[s] for s in subject_id]
        for d, dimension_name in enumerate(dimensions):
            dim = [positions[d] for positions in channel_positions_list]
            if d == 0:
                dim = np.abs(dim)  # for left-right, we transform position to distance from center
            axs[i, d].scatter(dim, regression_param_array, color=colors)
            axs[i, d].set_xlabel(dimension_name)

            # Create a DataFrame for the mixed model
            data_dict = {
                'Value': regression_param_array,
                'Position': dim,
                'SubjectID': subject_id
            }
            model_data = pd.DataFrame(data_dict)
            model = MixedLM.from_formula('Value ~ Position', groups='SubjectID', data=model_data)
            result = model.fit()
            p_value = result.pvalues['Position']

            # fitted line
            x_vals = np.linspace(min(dim), max(dim), 100)
            y_vals = result.params['Intercept'] + result.params['Position'] * x_vals
            axs[i, d].plot(x_vals, y_vals, color='k', linestyle='--', linewidth=1.5)

            box_color = 'r' if p_value < 0.05 else 'white'
            axs[i, d].text(
                0.05, 0.95, f'p={p_value:.3g}', transform=axs[i, d].transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8)
            )
        axs[i, 0].set_ylabel(f'Regression parameter (interstimulus distance ~ {x_regression[i]})')

    interstim_params = regression_param_correct

    person_corr = pearsonr(intrastim_params, interstim_params, alternative='two-sided')
    print('Correlation test')
    print(person_corr)
    spearman_corr = spearmanr(intrastim_params, interstim_params, alternative='two-sided')
    print('Spearman')
    print(spearman_corr)

    subject_colors = plt.get_cmap('Set3').colors
    colors = [subject_colors[s] for s in subject_id]
    fig, axs = plt.subplots(1, 1)
    axs.scatter(intrastim_params, interstim_params, color=colors)
    axs.set_xlabel('Regression parameter (intrastimulus correlation ~ correct)')
    axs.set_ylabel('Regression parameter (interstimulus distance ~ correct)')
    axs.axvline(0, ls='--', color='k')
    axs.axhline(0, ls='--', color='k')

    # Create a DataFrame for the mixed model
    data_dict = {
        'intrastim_params': intrastim_params,
        'interstim_params': interstim_params,
        'SubjectID': subject_id
    }
    model_data = pd.DataFrame(data_dict)
    model = MixedLM.from_formula('interstim_params ~ intrastim_params', groups='SubjectID', data=model_data)
    result = model.fit()
    p_value = result.pvalues['intrastim_params']

    x_vals = np.linspace(min(intrastim_params), max(intrastim_params), 100)
    y_vals = result.params['Intercept'] + result.params['intrastim_params'] * x_vals
    axs.plot(x_vals, y_vals, color='k', linewidth=2)
    # slope, intercept = np.polyfit(intrastim_params, interstim_params, 1)
    # x_range = np.linspace(min(intrastim_params), max(intrastim_params), 100)
    # axs.plot(x_range, slope * x_range + intercept, color='k', lw=2)

    unique_subjects = np.unique(subject_id)
    for subj in unique_subjects:
        mask = subject_id == subj
        x_subj = intrastim_params[mask]
        y_subj = interstim_params[mask]
        slope, intercept = np.polyfit(x_subj, y_subj, 1)
        x_range = np.linspace(min(x_subj), max(x_subj), 100)
        axs.plot(x_range, slope * x_range + intercept, color=subject_colors[subj], alpha=0.7)

    # axs.text(0, axs.get_ylim()[1], f'rho = {round(spearman_corr.statistic, 2)}, p = {round(spearman_corr.pvalue, 5)}')
    axs.text(0, axs.get_ylim()[1], f"Beta = {round(result.params['intrastim_params'], 2)}, p = {round(p_value, 5)}")

    fig, axs = plt.subplots(1, 1)
    axs.scatter(intrastim_params, interstim_params, color='k', s=8)
    axs.set_xlabel('Regression parameter (intrastimulus correlation ~ correct)')
    axs.set_ylabel('Regression parameter (interstimulus distance ~ correct)')
    axs.axvline(0, ls='--', color='k')
    axs.axhline(0, ls='--', color='k')

    slope, intercept = np.polyfit(intrastim_params, interstim_params, 1)
    x_range = np.linspace(min(intrastim_params), max(intrastim_params), 100)
    axs.plot(x_range, slope * x_range + intercept, color='k', lw=2)

    axs.text(0, axs.get_ylim()[1], f'rho = {round(spearman_corr.statistic, 2)}, p = {round(spearman_corr.pvalue, 5)}')

    # boxplots
    df = pd.DataFrame({'Correct ~ intrastim correlation': intrastim_params,
                       'Intrastim correlation ~ surprise': intrastim_feature_params,
                       'Correct ~ interstim distance': interstim_params,
                       'Interstim distance ~ surprise': regression_param_feature})
    plt.figure()
    sns.boxplot(data=df, showmeans=True, showfliers=False, palette='Set3')
    sns.swarmplot(data=df, color=".25", size=2.5)
    plt.gca().axhline(0, ls='--', color='k')
    plt.ylabel('Regression parameters (a.u.)')

    for i, p in enumerate(list_of_pvalues):
        if p < 0.05:
            # Get the position to place the star
            x = i  # The x position corresponds to the box index
            y = df.iloc[:, i].max() + 0.05  # Slightly above the max value of the box
            plt.text(x, y, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')

    plt.show()

def full_representation_analysis_seeg(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=True, permutations=0):
    """Takes all electrodes from a big region (ex: all temporal lobe) and selected the subset of electrodes that show a
    significant RMS contrast (in either novelty detection or retrieval), then run intra and inter stimulus correlation
    analysis, linking them to correct response and surprise"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    subject_names = []
    del subject_paths[1]
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
        # for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty]]):
        # for cond, correct_incorrect in enumerate([[correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            # pval = ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False,
                            #                                permutations=10000, random_state=None,
                            #                                alternative='greater').pvalue  # testing of individual electrodes
                            # if pval < 0.05:
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    print(significant_electrodes)
    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])
    print(chan_count)


    ############## Representation analysis ##############

    # initialize random seed for permutations
    rng = np.random.default_rng(seed=32)

    # define the function that will be used for regressions (as well as permuted regressions)
    def run_regressions(representation_array):
        """representation_array contains intrastim correlation or interstim distance"""
        param_correct_list = []
        param_feature_list = []
        channel_positions = []
        subject_id_list = []
        for channel in range(representation_array.shape[1]):
            if ch_names[channel] in significant_electrodes[subject]:
                # Y = correlations_array[:, channel]
                # all_correlations += Y.tolist()
                # Y = (Y - Y.mean()) / Y.std()
                # X = np.zeros((Y.shape[0], 1))
                # # X = np.zeros((Y.shape[0], 2))
                # X[:, 0] = correct_list
                # # X[:, 1] = np.array(feature_list)**2
                # X = sm.add_constant(X)

                Y = np.array(correct_list)
                X = np.zeros((Y.shape[0], 1))
                X[:, 0] = representation_array[:, channel]
                X = (X - X.mean()) / X.std()
                X = sm.add_constant(X)

                # # independent t-test
                # correct_corr = Y[X[:, 1] == 1]  # axis 0 is the intercept
                # incorrect_corr = Y[X[:, 1] == 0]
                # p_ttest[channel] = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=1000,
                #                              random_state=None,
                #                              # alternative='greater').pvalue  # testing of individual electrodes
                #                              alternative='less').pvalue  # testing of individual electrodes
                # if p_ttest[channel] < 0.05:

                # ols = sm.OLS(Y, X).fit()
                ols = sm.Logit(Y, X).fit(disp=False)
                param_correct_list.append(ols.params[1])
                # pvalues[channel] = ols.pvalues[1]
                # regression_param[channel] = ols.params[2]

                Y = X[:, 1]
                X = np.zeros((Y.shape[0], 1))
                # X = np.zeros((Y.shape[0], 2))
                X[:, 0] = feature_list
                # X[:, 1] = np.array(feature_list)**2
                X = (X - X.mean(axis=0)) / X.std(axis=0)
                X = sm.add_constant(X)
                ols = sm.OLS(Y, X).fit()
                param_feature_list.append(ols.params[1])
                # regression_param_feature.append(ols.params[2])

                channel_positions.append(subject_info['chs'][channel]['loc'][:3])
                subject_id_list.append(s)
        return param_correct_list, param_feature_list, channel_positions, subject_id_list


    ############## Intrastim correlation -> correct ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(similarity_path, allow_pickle=True)

    channel_positions_list = []
    subject_id = []
    regression_param_correct = []
    regression_param_feature = []
    permuted_regression_param_correct = [[] for _ in range(permutations)]
    permuted_regression_param_feature = [[] for _ in range(permutations)]
    regression_param_correct_subject_mean = []
    regression_param_feature_subject_mean = []
    # pvalues_array = []  # p value for the regression
    p_ttest_array = []  # p value of the contrast correct vs incorrect

    for s, subject in tqdm(enumerate(data["subjects"])):
        if subject != 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            correlations_array = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['similarity_matrices'][s]  # keep all channels
            # similarity_matrix = data['distance_matrices'][s]  # keep all channels
            # similarity_matrix = 1/similarity_matrix

            # plt.figure()
            # plt.hist(subject_stim_features[condition], bins=100)
            # plt.show()

            first_trials_to_remove = 50
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:  # remove first trials
                    if stim in subject_stim_features['stim_name'].tolist():
                        # if 'rep2' in stim:
                        if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                            if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                                indices_to_keep.append(i)
                                correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                if 'rep1' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                                elif 'rep2' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                else:
                                    raise ValueError
                                correlations_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

            correlations_array = np.array(correlations_array)

            # pvalues = np.zeros(correlations_array.shape[1])
            p_ttest = np.zeros(correlations_array.shape[1])

            param_correct_list, param_feature_list, channel_positions, subject_id_list = run_regressions(correlations_array)
            regression_param_correct += param_correct_list
            regression_param_feature += param_feature_list
            channel_positions_list += channel_positions
            subject_id += subject_id_list

            # pvalues_array.append(pvalues)
            # p_ttest_array.append(p_ttest)

            # store subject average
            regression_param_correct_subject_mean.append(np.mean(param_correct_list))
            regression_param_feature_subject_mean.append(np.mean(param_feature_list))

            # run permutations
            if permutations > 0:
                print(f'Subject {s}, running permutations...')
                for perm in tqdm(range(permutations)):
                    permuted_correlations_array = rng.permutation(correlations_array, axis=0)  # shuffle trials in the same way for each channel, which keeps the data structure (correlation between channels within subjects)
                    param_correct_list, param_feature_list, _, _ = run_regressions(permuted_correlations_array)
                    permuted_regression_param_correct[perm] += param_correct_list
                    permuted_regression_param_feature[perm] += param_feature_list

    regression_param_correct = np.array(regression_param_correct)
    print(regression_param_correct.shape)
    regression_param_feature = np.array(regression_param_feature)
    print(regression_param_feature.shape)
    permuted_regression_param_correct = np.array(permuted_regression_param_correct)
    print(permuted_regression_param_correct.shape)
    permuted_regression_param_feature = np.array(permuted_regression_param_feature)
    print(permuted_regression_param_feature.shape)
    regression_param_correct_subject_mean = np.array(regression_param_correct_subject_mean)
    regression_param_feature_subject_mean = np.array(regression_param_feature_subject_mean)

    # pvalues_array = np.concatenate(pvalues_array)
    # print(np.sum(pvalues_array < 0.05), 'significant electrodes,',
    #       round(np.sum(pvalues_array < 0.05) / pvalues_array.shape[0] * 100, 3), "%")
    # print(regression_param_array[pvalues_array < 0.05])
    # p_ttest_array = np.concatenate(p_ttest_array)
    # print(np.sum(p_ttest_array < 0.05), 'significant electrodes with independent ttest,',
    #       round(np.sum(p_ttest_array < 0.05) / p_ttest_array.shape[0] * 100, 3), "%")
    # print(regression_param_array[p_ttest_array < 0.05])

    list_of_pvalues = []

    fig, axs = plt.subplots(1, 2, sharey=True)
    x_regression = ['correct ~ intrastimulus correlation', f'intrastimulus correlation ~ {condition}']
    alternative = ['greater', 'less']
    # alternative = ['greater', 'two-sided']
    permuted_regression_param_array = [permuted_regression_param_correct, permuted_regression_param_feature]
    regression_param_array_subject_mean = [regression_param_correct_subject_mean, regression_param_feature_subject_mean]
    for i, regression_param_array in enumerate([regression_param_correct, regression_param_feature]):
        print(np.mean(regression_param_array))

        # classical t-test (does not account for channel covariance within subjects)
        ttest = scipy.stats.ttest_1samp(regression_param_array, 0, alternative=alternative[i])
        print('ttest channel level:', ttest)
        list_of_pvalues.append(ttest.pvalue)

        # permuted t-test based on sign flips (can be too conservative)
        pval_permut = permut_1sample_ttest_across_channels(regression_param_array, subject_id, popmean=0, alternative=alternative[i], n_permutations=1000)
        print('pval_permut:', pval_permut)
        # list_of_pvalues.append(pval_permut)

        # permuted t-test bases on shuffling of neural representation values across trials (should be more powerful)
        pval_permut_representation = already_permuted_1sample_ttest_across_channels(regression_param_array, permuted_regression_param_array[i], popmean=0, alternative=alternative[i])
        print('pval_permut_representation:', pval_permut_representation)
        # list_of_pvalues.append(pval_permut_representation)

        # t-test at the subject level
        ttest = scipy.stats.ttest_1samp(regression_param_array_subject_mean[i], 0, alternative=alternative[i])
        print('ttest subject level:', ttest)
        # list_of_pvalues.append(ttest.pvalue)

        axs[i].hist(regression_param_array, bins=100)
        axs[i].axvline(0, ls='--', color='k')
        axs[i].axvline(np.mean(regression_param_array), color='r', lw=2)
        # plt.xlabel('Regression parameter (intrastimulus correlation ~ surprise)')
        # axs[i].set_xlabel(f'Regression parameter (intrastimulus correlation ~ {x_regression[i]})')
        axs[i].set_xlabel(f'Regression parameter ({x_regression[i]})')
        axs[i].set_ylabel('Count (electrode)')
        axs[i].text(plt.gca().get_xlim()[0], plt.gca().get_ylim()[1],
                 f'regression beta = {round(np.mean(regression_param_array), 5)}, p = {round(ttest.pvalue, 5)}')

    fig, axs = plt.subplots(2, 3, sharey='row')
    for i, regression_param_array in enumerate([regression_param_correct, regression_param_feature]):
        dimensions = ['left-right (abs: distance from center)', 'posterior-anterior', 'ventral-dorsal']
        subject_colors = plt.get_cmap('Set3').colors
        colors = [subject_colors[s] for s in subject_id]
        for d, dimension_name in enumerate(dimensions):
            dim = [positions[d] for positions in channel_positions_list]
            if d == 0:
                dim = np.abs(dim)  # for left-right, we transform position to distance from center
            axs[i, d].scatter(dim, regression_param_array, color=colors)
            axs[i, d].set_xlabel(dimension_name)

            # Create a DataFrame for the mixed model
            data_dict = {
                'Value': regression_param_array,
                'Position': dim,
                'SubjectID': subject_id
            }
            model_data = pd.DataFrame(data_dict)
            model = MixedLM.from_formula('Value ~ Position', groups='SubjectID', data=model_data)
            result = model.fit()
            p_value = result.pvalues['Position']

            # fitted line
            x_vals = np.linspace(min(dim), max(dim), 100)
            y_vals = result.params['Intercept'] + result.params['Position'] * x_vals
            axs[i, d].plot(x_vals, y_vals, color='k', linestyle='--', linewidth=1.5)

            box_color = 'r' if p_value < 0.05 else 'white'
            axs[i, d].text(
                0.05, 0.95, f'p={p_value:.3g}', transform=axs[i, d].transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8)
            )
        axs[i, 0].set_ylabel(f'Regression parameter (intrastimulus correlation ~ {x_regression[i]})')

    # plt.show()

    intrastim_params = regression_param_correct.copy()
    intrastim_feature_params = regression_param_feature.copy()


    ############## Interstim correlation -> incorrect ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(distance_path, allow_pickle=True)

    channel_positions_list = []
    subject_id = []
    regression_param_correct = []
    regression_param_feature = []
    permuted_regression_param_correct = [[] for _ in range(permutations)]
    permuted_regression_param_feature = [[] for _ in range(permutations)]
    regression_param_correct_subject_mean = []
    regression_param_feature_subject_mean = []
    # pvalues_array = []  # p value for the regression
    # p_ttest_array = []  # p value of the contrast correct vs incorrect
    for s, subject in tqdm(enumerate(data["subjects"])):
        if not subject == 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            # similarity_matrix = data['similarity_matrices'][s]  # keep all channels
            # similarity_matrix = 1-similarity_matrix  # transform into interstim distance
            # similarity_matrix = similarity_matrix**2  # transform into interstim distance
            # similarity_matrix = 1/similarity_matrix**2  # transform into interstim distance
            similarity_matrix = data['distance_matrices'][s]  # keep all channels
            paired_stim = []
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:  # remove first trials
                    if ('rep1' in stim) | ('rep2' in stim):
                    # if 'rep2' in stim:
                        if stim in subject_stim_features['stim_name'].tolist():
                            if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                # if subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0] < 0.15:  # check that it works without extreme trials
                                indices_to_keep.append(i)
                                correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                if 'rep2' in stim:
                                    # indices_to_keep.append(i)
                                    # correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    # feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                    if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                        paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))
                                    else:
                                        paired_stim.append('no_pair')
                                if 'rep1' in stim:
                                    # indices_to_keep.append(i)
                                    # correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    # feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                    if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                                        paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2')))
                                    else:
                                        paired_stim.append('no_pair')
                                # else:
                                #     raise ValueError('Neither rep1 nor rep2???')
            # plt.imshow(np.mean(similarity_matrix, axis=-1), cmap='gray')
            # plt.show()
            condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
            # condition_matrix = condition_matrix[:, indices_to_keep, :]  # select condition
            correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
            for i in range(condition_matrix.shape[0]):
                current_corr = condition_matrix[i, :, :]
                mask = np.ones(condition_matrix.shape[1], dtype=bool)
                mask[indices_to_keep[i]] = False  # remove current stim
                # mask[i] = False  # remove current stim
                # if not paired_stim[i] == 'no_pair':
                #     mask[paired_stim[i]] = False  # remove pair
                # for j in range(indices_to_keep[i], mask.shape[0]):
                #     mask[j] = False  # remove following trials
                mask[indices_to_keep[i]:] = False  # remove following trials
                # plt.hist(current_corr[mask, 10], bins=100)
                # plt.show()
                # print(np.mean(current_corr[mask, :], axis=0))
                # print(current_corr.shape)
                # print(mask)
                # print(current_corr[mask, :].shape)
                correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)
            # print(correlations_array.shape)
            # plt.figure()
            # plt.plot(correlations_array)
            # plt.show()

            # pvalues = np.zeros(correlations_array.shape[1])
            # p_ttest = np.zeros(correlations_array.shape[1])

            param_correct_list, param_feature_list, channel_positions, subject_id_list = run_regressions(correlations_array)
            regression_param_correct += param_correct_list
            regression_param_feature += param_feature_list
            channel_positions_list += channel_positions
            subject_id += subject_id_list

            # pvalues_array.append(pvalues)
            # p_ttest_array.append(p_ttest)

            # store subject average
            regression_param_correct_subject_mean.append(np.mean(param_correct_list))
            regression_param_feature_subject_mean.append(np.mean(param_feature_list))

            # run permutations
            if permutations > 0:
                print(f'Subject {s}, running permutations...')
                for perm in tqdm(range(permutations)):
                    permuted_correlations_array = rng.permutation(correlations_array, axis=0)  # shuffle trials in the same way for each channel, which keeps the data structure (correlation between channels within subjects)
                    param_correct_list, param_feature_list, _, _ = run_regressions(permuted_correlations_array)
                    permuted_regression_param_correct[perm] += param_correct_list
                    permuted_regression_param_feature[perm] += param_feature_list

    regression_param_correct = np.array(regression_param_correct)
    print(regression_param_correct.shape)
    regression_param_feature = np.array(regression_param_feature)
    print(regression_param_feature.shape)
    permuted_regression_param_correct = np.array(permuted_regression_param_correct)
    print(permuted_regression_param_correct.shape)
    permuted_regression_param_feature = np.array(permuted_regression_param_feature)
    print(permuted_regression_param_feature.shape)
    regression_param_correct_subject_mean = np.array(regression_param_correct_subject_mean)
    regression_param_feature_subject_mean = np.array(regression_param_feature_subject_mean)

    # pvalues_array = np.concatenate(pvalues_array)
    # print(np.sum(pvalues_array < 0.05), 'significant electrodes,',
    #       round(np.sum(pvalues_array < 0.05) / pvalues_array.shape[0] * 100, 3), "%")
    # print(regression_param_array[pvalues_array < 0.05])
    # p_ttest_array = np.concatenate(p_ttest_array)
    # print(np.sum(p_ttest_array < 0.05), 'significant electrodes with independent ttest,',
    #       round(np.sum(p_ttest_array < 0.05) / p_ttest_array.shape[0] * 100, 3), "%")
    # print(regression_param_array[p_ttest_array < 0.05])

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    x_regression = ['correct ~ interstimulus distance', f'interstimulus distance ~ {condition}']
    alternative = ['greater', 'greater']
    # alternative = ['greater', 'two-sided']
    permuted_regression_param_array = [permuted_regression_param_correct, permuted_regression_param_feature]
    regression_param_array_subject_mean = [regression_param_correct_subject_mean, regression_param_feature_subject_mean]
    for i, regression_param_array in enumerate([regression_param_correct, regression_param_feature]):
        print(np.mean(regression_param_array))

        # classical t-test (does not account for channel covariance within subjects)
        ttest = scipy.stats.ttest_1samp(regression_param_array, 0, alternative=alternative[i])
        print('ttest channel level:', ttest)
        list_of_pvalues.append(ttest.pvalue)

        # permuted t-test based on sign flips (can be too conservative)
        pval_permut = permut_1sample_ttest_across_channels(regression_param_array, subject_id, popmean=0, alternative=alternative[i], n_permutations=1000)
        print('pval_permut:', pval_permut)
        # list_of_pvalues.append(pval_permut)

        # permuted t-test bases on shuffling of neural representation values across trials (should be more powerful)
        pval_permut_representation = already_permuted_1sample_ttest_across_channels(regression_param_array, permuted_regression_param_array[i], popmean=0, alternative=alternative[i])
        print('pval_permut_representation:', pval_permut_representation)
        # list_of_pvalues.append(pval_permut_representation)

        # t-test at the subject level
        ttest = scipy.stats.ttest_1samp(regression_param_array_subject_mean[i], 0, alternative=alternative[i])
        print('ttest subject level:', ttest)
        # list_of_pvalues.append(ttest.pvalue)

        axs[i].hist(regression_param_array, bins=100)
        axs[i].axvline(0, ls='--', color='k')
        axs[i].axvline(np.mean(regression_param_array), color='r', lw=2)
        # plt.xlabel('Regression parameter (intrastimulus correlation ~ surprise)')
        axs[i].set_xlabel(f'Regression parameter ({x_regression[i]})')
        axs[i].set_ylabel('Count (electrode)')
        axs[i].text(plt.gca().get_xlim()[0], plt.gca().get_ylim()[1],
                    f'regression beta = {round(np.mean(regression_param_array), 5)}, p = {round(ttest.pvalue, 5)}')

    fig, axs = plt.subplots(2, 3, sharey='row')
    for i, regression_param_array in enumerate([regression_param_correct, regression_param_feature]):
        dimensions = ['left-right (abs: distance from center)', 'posterior-anterior', 'ventral-dorsal']
        subject_colors = plt.get_cmap('Set3').colors
        colors = [subject_colors[s] for s in subject_id]
        for d, dimension_name in enumerate(dimensions):
            dim = [positions[d] for positions in channel_positions_list]
            if d == 0:
                dim = np.abs(dim)  # for left-right, we transform position to distance from center
            axs[i, d].scatter(dim, regression_param_array, color=colors)
            axs[i, d].set_xlabel(dimension_name)

            # Create a DataFrame for the mixed model
            data_dict = {
                'Value': regression_param_array,
                'Position': dim,
                'SubjectID': subject_id
            }
            model_data = pd.DataFrame(data_dict)
            model = MixedLM.from_formula('Value ~ Position', groups='SubjectID', data=model_data)
            result = model.fit()
            p_value = result.pvalues['Position']

            # fitted line
            x_vals = np.linspace(min(dim), max(dim), 100)
            y_vals = result.params['Intercept'] + result.params['Position'] * x_vals
            axs[i, d].plot(x_vals, y_vals, color='k', linestyle='--', linewidth=1.5)

            box_color = 'r' if p_value < 0.05 else 'white'
            axs[i, d].text(
                0.05, 0.95, f'p={p_value:.3g}', transform=axs[i, d].transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor=box_color, alpha=0.8)
            )
        axs[i, 0].set_ylabel(f'Regression parameter (interstimulus distance ~ {x_regression[i]})')

    interstim_params = regression_param_correct

    person_corr = pearsonr(intrastim_params, interstim_params, alternative='two-sided')
    print('Correlation test')
    print(person_corr)
    spearman_corr = spearmanr(intrastim_params, interstim_params, alternative='two-sided')
    print('Spearman')
    print(spearman_corr)

    # intrastim vs interstim
    # subject_colors = plt.get_cmap('Set3').colors
    subject_colors = ['lightgray' for s in range(len(subject_id))]
    colors = [subject_colors[s] for s in subject_id]
    fig, axs = plt.subplots(1, 1, figsize=(5.2, 5))
    axs.scatter(intrastim_params, interstim_params, color=colors, s=12, alpha=0.3)
    axs.set_xlabel('\u03B2 (correct ~ intrastimulus correlation)', fontsize=13)
    axs.set_ylabel('\u03B2 (correct ~ interstimulus distance)', fontsize=13)
    axs.tick_params(axis='y', labelsize=13)
    axs.tick_params(axis='x', labelsize=13)
    axs.axvline(0, ls='--', color='k')
    axs.axhline(0, ls='--', color='k')

    # Create a DataFrame for the mixed model
    data_dict = {
        'intrastim_params': intrastim_params,
        'interstim_params': interstim_params,
        'SubjectID': subject_id
    }
    model_data = pd.DataFrame(data_dict)
    model = MixedLM.from_formula('interstim_params ~ intrastim_params', groups='SubjectID', data=model_data)
    result = model.fit()
    print(result.summary())
    p_value = result.pvalues['intrastim_params']

    x_vals = np.linspace(min(intrastim_params), max(intrastim_params), 100)
    y_vals = result.params['Intercept'] + result.params['intrastim_params'] * x_vals
    axs.plot(x_vals, y_vals, color='k', linewidth=5, zorder=10)
    # slope, intercept = np.polyfit(intrastim_params, interstim_params, 1)
    # x_range = np.linspace(min(intrastim_params), max(intrastim_params), 100)
    # axs.plot(x_range, slope * x_range + intercept, color='k', lw=2)

    # # To plot a confidence interval
    # cov_matrix = result.cov_params()
    # var_intercept = cov_matrix.loc['Intercept', 'Intercept']
    # var_slope = cov_matrix.loc['intrastim_params', 'intrastim_params']
    # cov_intercept_slope = cov_matrix.loc['Intercept', 'intrastim_params']
    # SE_y_vals = np.sqrt(var_intercept + (x_vals ** 2) * var_slope + 2 * x_vals * cov_intercept_slope)
    # from scipy.stats import t
    # t_crit = t.ppf(0.975, df=result.df_resid)  # t-value for 95% CI
    # CI_upper = y_vals + t_crit * SE_y_vals
    # CI_lower = y_vals - t_crit * SE_y_vals
    # axs.fill_between(x_vals, CI_lower, CI_upper, color="red", alpha=0.2, label="95% CI")

    unique_subjects = np.unique(subject_id)
    for subj in unique_subjects:
        mask = subject_id == subj
        x_subj = intrastim_params[mask]
        y_subj = interstim_params[mask]
        slope, intercept = np.polyfit(x_subj, y_subj, 1)
        x_range = np.linspace(min(x_subj), max(x_subj), 100)
        axs.plot(x_range, slope * x_range + intercept, color=subject_colors[subj], lw=3, alpha=1)

    # axs.text(0, axs.get_ylim()[1], f'rho = {round(spearman_corr.statistic, 2)}, p = {round(spearman_corr.pvalue, 5)}')
    # axs.text(0, axs.get_ylim()[1], f"Beta = {round(result.params['intrastim_params'], 2)}, p = {round(p_value, 5)}")
    plt.tight_layout()

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.scatter(intrastim_params, interstim_params, color='k', s=8)
    axs.set_xlabel('Regression parameter (correct ~ intrastimulus correlation)')
    axs.set_ylabel('Regression parameter (correct ~ interstimulus distance)')
    axs.axvline(0, ls='--', color='k')
    axs.axhline(0, ls='--', color='k')

    slope, intercept = np.polyfit(intrastim_params, interstim_params, 1)
    x_range = np.linspace(min(intrastim_params), max(intrastim_params), 100)
    axs.plot(x_range, slope * x_range + intercept, color='k', lw=2)

    axs.text(0, axs.get_ylim()[1], f'rho = {round(spearman_corr.statistic, 2)}, p = {round(spearman_corr.pvalue, 5)}')

    # intrastim vs interstim surprise
    fig, axs = plt.subplots(1, 1, figsize=(5.2, 5))
    axs.scatter(intrastim_feature_params, regression_param_feature, color=colors, s=12, alpha=0.3)
    axs.set_xlabel('\u03B2 (intrastimulus correlation ~ surprise)', fontsize=13)
    axs.set_ylabel('\u03B2 (interstimulus distance ~ surprise)', fontsize=13)
    axs.tick_params(axis='y', labelsize=13)
    axs.tick_params(axis='x', labelsize=13)
    axs.axvline(0, ls='--', color='k')
    axs.axhline(0, ls='--', color='k')

    # Create a DataFrame for the mixed model
    data_dict = {
        'interstim_feature_params': regression_param_feature,
        'intrastim_feature_params': intrastim_feature_params,
        'SubjectID': subject_id
    }
    model_data = pd.DataFrame(data_dict)
    model = MixedLM.from_formula('interstim_feature_params ~ intrastim_feature_params', groups='SubjectID', data=model_data)
    result = model.fit()
    print(result.summary())
    p_value = result.pvalues['intrastim_feature_params']

    x_vals = np.linspace(min(intrastim_feature_params), max(intrastim_feature_params), 100)
    y_vals = result.params['Intercept'] + result.params['intrastim_feature_params'] * x_vals
    axs.plot(x_vals, y_vals, color='k', linewidth=5, zorder=10)
    # slope, intercept = np.polyfit(intrastim_params, interstim_params, 1)
    # x_range = np.linspace(min(intrastim_params), max(intrastim_params), 100)
    # axs.plot(x_range, slope * x_range + intercept, color='k', lw=2)

    unique_subjects = np.unique(subject_id)
    for subj in unique_subjects:
        mask = subject_id == subj
        x_subj = intrastim_feature_params[mask]
        y_subj = regression_param_feature[mask]
        slope, intercept = np.polyfit(x_subj, y_subj, 1)
        x_range = np.linspace(min(x_subj), max(x_subj), 100)
        axs.plot(x_range, slope * x_range + intercept, color=subject_colors[subj], lw=3)

    # axs.text(0, axs.get_ylim()[1], f'rho = {round(spearman_corr.statistic, 2)}, p = {round(spearman_corr.pvalue, 5)}')
    # axs.text(0, axs.get_ylim()[1],
    #          f"Beta = {round(result.params['intrastim_feature_params'], 2)}, p = {round(p_value, 5)}")
    plt.tight_layout()

    # intrastim correct and surprise
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.scatter(intrastim_feature_params, intrastim_params, color=colors, s=2, alpha=0.7)
    axs.set_xlabel('Regression parameter (intrastimulus correlation ~ surprise)')
    axs.set_ylabel('Regression parameter (correct ~ intrastimulus correlation)')
    axs.axvline(0, ls='--', color='k')
    axs.axhline(0, ls='--', color='k')

    # Create a DataFrame for the mixed model
    data_dict = {
        'intrastim_params': intrastim_params,
        'intrastim_feature_params': intrastim_feature_params,
        'SubjectID': subject_id
    }
    model_data = pd.DataFrame(data_dict)
    model = MixedLM.from_formula('intrastim_params ~ intrastim_feature_params', groups='SubjectID', data=model_data)
    result = model.fit()
    p_value = result.pvalues['intrastim_feature_params']

    x_vals = np.linspace(min(intrastim_feature_params), max(intrastim_feature_params), 100)
    y_vals = result.params['Intercept'] + result.params['intrastim_feature_params'] * x_vals
    axs.plot(x_vals, y_vals, color='k', linewidth=4)
    # slope, intercept = np.polyfit(intrastim_params, interstim_params, 1)
    # x_range = np.linspace(min(intrastim_params), max(intrastim_params), 100)
    # axs.plot(x_range, slope * x_range + intercept, color='k', lw=2)

    unique_subjects = np.unique(subject_id)
    for subj in unique_subjects:
        mask = subject_id == subj
        x_subj = intrastim_feature_params[mask]
        y_subj = intrastim_params[mask]
        slope, intercept = np.polyfit(x_subj, y_subj, 1)
        x_range = np.linspace(min(x_subj), max(x_subj), 100)
        axs.plot(x_range, slope * x_range + intercept, color=subject_colors[subj], lw=2)

    # axs.text(0, axs.get_ylim()[1], f'rho = {round(spearman_corr.statistic, 2)}, p = {round(spearman_corr.pvalue, 5)}')
    axs.text(0, axs.get_ylim()[1], f"Beta = {round(result.params['intrastim_feature_params'], 2)}, p = {round(p_value, 5)}")

    # interstim correct and surprise
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.scatter(regression_param_feature, interstim_params, color=colors, s=2, alpha=0.7)
    axs.set_xlabel('Regression parameter (interstimulus distance ~ surprise)')
    axs.set_ylabel('Regression parameter (correct ~ interstimulus distance)')
    axs.axvline(0, ls='--', color='k')
    axs.axhline(0, ls='--', color='k')

    # Create a DataFrame for the mixed model
    data_dict = {
        'interstim_params': interstim_params,
        'interstim_feature_params': regression_param_feature,
        'SubjectID': subject_id
    }
    model_data = pd.DataFrame(data_dict)
    model = MixedLM.from_formula('interstim_params ~ interstim_feature_params', groups='SubjectID', data=model_data)
    result = model.fit()
    p_value = result.pvalues['interstim_feature_params']

    x_vals = np.linspace(min(regression_param_feature), max(regression_param_feature), 100)
    y_vals = result.params['Intercept'] + result.params['interstim_feature_params'] * x_vals
    axs.plot(x_vals, y_vals, color='k', linewidth=4)
    # slope, intercept = np.polyfit(intrastim_params, interstim_params, 1)
    # x_range = np.linspace(min(intrastim_params), max(intrastim_params), 100)
    # axs.plot(x_range, slope * x_range + intercept, color='k', lw=2)

    unique_subjects = np.unique(subject_id)
    for subj in unique_subjects:
        mask = subject_id == subj
        x_subj = regression_param_feature[mask]
        y_subj = interstim_params[mask]
        slope, intercept = np.polyfit(x_subj, y_subj, 1)
        x_range = np.linspace(min(x_subj), max(x_subj), 100)
        axs.plot(x_range, slope * x_range + intercept, color=subject_colors[subj], lw=2)

    # axs.text(0, axs.get_ylim()[1], f'rho = {round(spearman_corr.statistic, 2)}, p = {round(spearman_corr.pvalue, 5)}')
    axs.text(0, axs.get_ylim()[1],
             f"Beta = {round(result.params['interstim_feature_params'], 2)}, p = {round(p_value, 5)}")


    # boxplots correct
    df = pd.DataFrame({'Similarity': intrastim_params, 'Separability': interstim_params})
    plt.figure(figsize=(5, 5))
    sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5'], orient='v', inner=None)
    sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
    plt.gca().axhline(0, ls='--', color='k')
    plt.ylabel('\u03B2 (correct ~ condition)', fontsize=17)
    # plt.xticks(plt.xticks()[0], plt.xticks()[1], rotation=90, va='center')
    plt.tick_params(axis='y', labelsize=17)
    plt.tick_params(axis='x', labelsize=17)

    plt.scatter(0, np.mean(intrastim_params), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(1, np.mean(interstim_params), color='w', edgecolor='k', lw=2, s=75, zorder=10)

    for i, p in enumerate([list_of_pvalues[0], list_of_pvalues[2]]):
        if p < 0.05:
            # Get the position to place the star
            x = i  # The x position corresponds to the box index
            y = df.iloc[:, i].max() + 0.15  # Slightly above the max value of the box
            plt.text(x, y, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
    plt.tight_layout()

    # boxplots
    df = pd.DataFrame({'Similarity': intrastim_feature_params, 'Separability': regression_param_feature})
    plt.figure(figsize=(5, 5))
    sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5'], orient='v', inner=None)
    sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
    plt.gca().axhline(0, ls='--', color='k')
    plt.ylabel('\u03B2 (condition ~ surprise)', fontsize=17)
    # plt.yticks(plt.yticks()[0], plt.yticks()[1], rotation=90, va='center')
    plt.tick_params(axis='y', labelsize=17)
    plt.tick_params(axis='x', labelsize=17)

    plt.scatter(0, np.mean(intrastim_feature_params), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(1, np.mean(regression_param_feature), color='w', edgecolor='k', lw=2, s=75, zorder=10)

    for i, p in enumerate([list_of_pvalues[1], list_of_pvalues[3]]):
        if p < 0.05:
            # Get the position to place the star
            x = i  # The x position corresponds to the box index
            y = df.iloc[:, i].max() + 0.15  # Slightly above the max value of the box
            plt.text(x, y, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
    plt.tight_layout()


    plt.show()



def full_representation_analysis_by_region(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=True, remove_outliers=False):
    """Takes all electrodes from a big region (ex: all temporal lobe) and selected the subset of electrodes that show a
    significant RMS contrast (in either novelty detection or retrieval), then run intra and inter stimulus correlation
    analysis, linking them to correct response and surprise"""

    # conditions = ['correct ~ intrastimulus correlation', 'correct ~ interstimulus distance', 'intrastimulus correlation ~ surprise', 'interstimulus distance ~ surprise']
    conditions = ['correct ~ similarity', 'correct ~ separability', 'similarity ~ surprise', 'separability ~ surprise']
    ############## Electrode selection based on RMS contrast ##############

    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    subject_names = []
    del subject_paths[1]
    print(subject_paths)

    unique_parcellations = []
    for i, subject_path in enumerate(tqdm(subject_paths)):
        rms_obj = np.load(subject_path, allow_pickle=True)
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        unique_parcellations += parcellations
    unique_parcellations = np.unique(unique_parcellations)
    # regions_to_remove = ['Unknown', 'Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter', 'ctx-rh-unknown', 'ctx-lh-unknown', 'CSF']
    # regions_to_remove = []
    regions_to_remove = ['Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter', 'CSF', 'Unknown']
    unique_parcellations = unique_parcellations[np.isin(unique_parcellations, regions_to_remove, invert=True)]
    unique_parcellations = sorted(unique_parcellations, key=lambda x: x[::-1], reverse=True)
    print(unique_parcellations)
    print('Number of regions:', len(unique_parcellations))

    if regions is not None:
        unique_parcellations = [region for region in unique_parcellations if region in regions]
    regression_parameters_dicts = [{region: [] for region in unique_parcellations} for c in range(len(conditions))]

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        significant_electrodes[subject_name] = {}
        for c in range(rms.shape[1]):
            for region in regions:
                if any(ch_parcel in region for ch_parcel in parcellations[c]):
                    significant_electrodes[subject_name][rms_obj['ch_names'][c]] = region


    ############## Intrastim correlation -> correct ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(similarity_path, allow_pickle=True)

    for s, subject in tqdm(enumerate(data["subjects"])):
        if subject != 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            correlations_array = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['similarity_matrices'][s]  # keep all channels

            first_trials_to_remove = 50
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:  # remove first trials
                    if stim in subject_stim_features['stim_name'].tolist():
                        # if 'rep2' in stim:
                        if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                            if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                                indices_to_keep.append(i)
                                correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                if 'rep1' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                                elif 'rep2' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                else:
                                    raise ValueError
                                correlations_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

            correlations_array = np.array(correlations_array)

            for channel in range(correlations_array.shape[1]):
                if ch_names[channel] in significant_electrodes[subject].keys():
                    Y = np.array(correct_list)
                    X = np.zeros((Y.shape[0], 1))
                    X[:, 0] = correlations_array[:, channel]
                    X = (X - X.mean()) / X.std()
                    X = sm.add_constant(X)

                    ols = sm.Logit(Y, X).fit()
                    regression_parameters_dicts[0][significant_electrodes[subject][ch_names[channel]]].append(ols.params[1])

                    Y = X[:, 1]
                    X = np.zeros((Y.shape[0], 1))
                    X[:, 0] = feature_list
                    X = (X - X.mean(axis=0)) / X.std(axis=0)
                    X = sm.add_constant(X)
                    ols = sm.OLS(Y, X).fit()
                    regression_parameters_dicts[2][significant_electrodes[subject][ch_names[channel]]].append(ols.params[1])


    ############## Interstim correlation -> incorrect ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(distance_path, allow_pickle=True)

    for s, subject in tqdm(enumerate(data["subjects"])):

        if not subject == 'Bou_Ni':

            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['distance_matrices'][s]  # keep all channels
            paired_stim = []
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:  # remove first trials
                    if ('rep1' in stim) | ('rep2' in stim):
                        if stim in subject_stim_features['stim_name'].tolist():
                            if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                indices_to_keep.append(i)
                                correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                if 'rep2' in stim:
                                    if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                        paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))
                                    else:
                                        paired_stim.append('no_pair')
                                if 'rep1' in stim:
                                    if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                                        paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2')))
                                    else:
                                        paired_stim.append('no_pair')

            condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
            correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
            for i in range(condition_matrix.shape[0]):
                current_corr = condition_matrix[i, :, :]
                mask = np.ones(condition_matrix.shape[1], dtype=bool)
                mask[indices_to_keep[i]] = False  # remove current stim
                mask[indices_to_keep[i]:] = False  # remove following trials
                correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)

            for channel in range(correlations_array.shape[1]):
                if ch_names[channel] in significant_electrodes[subject].keys():
                    Y = np.array(correct_list)
                    X = np.zeros((Y.shape[0], 1))
                    X[:, 0] = correlations_array[:, channel]
                    X = (X - X.mean()) / X.std()
                    X = sm.add_constant(X)

                    ols = sm.Logit(Y, X).fit()
                    regression_parameters_dicts[1][significant_electrodes[subject][ch_names[channel]]].append(ols.params[1])

                    Y = X[:, 1]
                    X = np.zeros((Y.shape[0], 1))
                    X[:, 0] = feature_list
                    X = (X - X.mean(axis=0)) / X.std(axis=0)
                    X = sm.add_constant(X)
                    ols = sm.OLS(Y, X).fit()
                    regression_parameters_dicts[3][significant_electrodes[subject][ch_names[channel]]].append(ols.params[1])

    alternatives = ['greater', 'greater', 'less', 'greater']
    # Calculate and save average rms for each condition and region
    region_rms_per_condition = []
    error_bars_per_condition = []
    regions_per_condition = []
    pvalue_per_condition = []
    tvalue_per_condition = []
    for cond in range(len(conditions)):
        condition_rms = []
        condition_error_bars = []
        condition_regions = []
        condition_pvalue = []
        condition_tvalue = []
        for region in unique_parcellations:
            region_rms = np.array(regression_parameters_dicts[cond][region])
            condition_regions.append(region)
            mean_rms = np.mean(region_rms, axis=0)
            condition_rms.append(mean_rms)
            error_rms = np.std(region_rms, axis=0) / np.sqrt(region_rms.shape[0])
            condition_error_bars.append(error_rms)
            ttest = ttest_1samp(region_rms, 0, alternative=alternatives[cond])
            condition_pvalue.append(ttest.pvalue)
            condition_tvalue.append(ttest.statistic)
        region_rms_per_condition.append(condition_rms)
        error_bars_per_condition.append(condition_error_bars)
        regions_per_condition.append(condition_regions)
        pvalue_per_condition.append(condition_pvalue)
        tvalue_per_condition.append(condition_tvalue)

    # FDR correction
    for c in range(len(pvalue_per_condition)):
        _, pval = mne.stats.fdr_correction(pvalue_per_condition[c], alpha=0.05, method='indep')
        pvalue_per_condition[c] = pval

    new_names = {'ctx-lh-transversetemporal': 'L-TT', 'ctx-lh-superiortemporal': 'L-STG',
                 'ctx-lh-middletemporal': 'L-MTG', 'ctx-lh-inferiortemporal': 'L-ITG',
                 'ctx-lh-temporalpole': 'L-TP','ctx-lh-entorhinal': 'L-EC',
                 'ctx-lh-parahippocampal': 'L-PHC',
                 'Left-Hippocampus': 'L-HPC', 'Left-Amygdala': 'L-AMY',
                 'ctx-rh-transversetemporal': 'R-TT', 'ctx-rh-superiortemporal': 'R-STG',
                 'ctx-rh-middletemporal': 'R-MTG', 'ctx-rh-inferiortemporal': 'R-ITG',
                 'ctx-rh-temporalpole': 'R-TP', 'ctx-rh-entorhinal': 'R-EC',
                 'ctx-rh-parahippocampal': 'R-PHC',
                 'Right-Hippocampus': 'R-HPC', 'Right-Amygdala': 'R-AMY'
                 }

    cmap = list(plt.get_cmap('Set3').colors)
    del cmap[8]
    region_colors = {region: cmap[i % int(len(new_names.keys()) / 2)] for i, region in enumerate(new_names.keys())}

    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
    # Plot each condition's data in a separate subplot
    for i, (condition, regions, region_rms, error_bars, pvalues, tvalues) in enumerate(zip(conditions[0:2], regions_per_condition[0:2], region_rms_per_condition[0:2], error_bars_per_condition[0:2], pvalue_per_condition[0:2], tvalue_per_condition[0:2])):
        for r in range(len(regions)):
            print(condition)
            print(regions[r])
            print(tvalues[r])
            print(pvalues[r])
            print(len(regression_parameters_dicts[i][regions[r]]))

        sorted_id = np.argsort(region_rms)
        region_rms = np.array(region_rms)[sorted_id]
        regions = np.array(regions)[sorted_id]
        new_names_regions = [new_names[reg] for reg in regions]
        pvalues = np.array(pvalues)[sorted_id]
        colors = [region_colors[regions[r]] if p < 0.05 else 'w' for r, p in enumerate(pvalues)]
        axs[i].barh(new_names_regions, region_rms, xerr=None, capsize=5, color=colors, edgecolor='k', lw=1)
        axs[i].axvline(0, color='k', linestyle='--', linewidth=1.5)
        axs[i].tick_params(axis='y', labelsize=13)
        axs[i].tick_params(axis='x', labelsize=17)

        # Add individual sample points as scatter dots
        for j, region in enumerate(regions):
            samples = np.array(regression_parameters_dicts[i][region])
            if remove_outliers:
                samples = samples[(np.abs(samples - samples.mean())) <= 2 * samples.std()]
            # Add scatter plot with jitter on x-axis
            y_values = np.full(samples.shape, j) + np.random.uniform(-0.1, 0.1, samples.shape)  # Jitter for clarity
            axs[i].scatter(samples, y_values, color='black', alpha=0.7, s=2, label='Samples')
        axs[i].set_xlabel(f'\u03B2 ({condition})', fontsize=17)
        # axs[i].set_title(condition)
    plt.tight_layout()

    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
    # Plot each condition's data in a separate subplot
    for i, (condition, regions, region_rms, error_bars, pvalues, tvalues) in enumerate(
            zip(conditions[2:4], regions_per_condition[2:4], region_rms_per_condition[2:4],
                error_bars_per_condition[2:4], pvalue_per_condition[2:4], tvalue_per_condition[2:4])):

        for r in range(len(regions)):
            print(condition)
            print(regions[r])
            print(tvalues[r])
            print(pvalues[r])
            print(len(regression_parameters_dicts[i][regions[r]]))

        if i==0:
            sorted_id = np.argsort(region_rms)[::-1]
        else:
            sorted_id = np.argsort(region_rms)
        region_rms = np.array(region_rms)[sorted_id]
        regions = np.array(regions)[sorted_id]
        new_names_regions = [new_names[reg] for reg in regions]
        pvalues = np.array(pvalues)[sorted_id]
        colors = [region_colors[regions[r]] if p < 0.05 else 'w' for r, p in enumerate(pvalues)]
        axs[i].barh(new_names_regions, region_rms, xerr=None, capsize=5, color=colors, edgecolor='k', lw=1)
        axs[i].axvline(0, color='k', linestyle='--', linewidth=1.5)
        axs[i].tick_params(axis='y', labelsize=13)
        axs[i].tick_params(axis='x', labelsize=17)

        # Add individual sample points as scatter dots
        for j, region in enumerate(regions):
            samples = np.array(regression_parameters_dicts[i+2][region])
            if remove_outliers:
                samples = samples[(np.abs(samples - samples.mean())) <= 2 * samples.std()]
            # Add scatter plot with jitter on x-axis
            y_values = np.full(samples.shape, j) + np.random.uniform(-0.1, 0.1, samples.shape)  # Jitter for clarity
            axs[i].scatter(samples, y_values, color='black', alpha=0.7, s=2, label='Samples')
        axs[i].set_xlabel(f'\u03B2 ({condition})', fontsize=17)
        # axs[i].set_title(condition)

    plt.tight_layout()

    plt.show()


def save_full_representation_stat_individual_electrode(save_path, behavior_path, similarity_path, distance_path, condition):
    """save for 2d projections plotting"""
    ############## Intrastim correlation -> correct ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(similarity_path, allow_pickle=True)

    intrastim_correct_params = []
    intrastim_correct_pvalues = []
    intrastim_surprise_params = []
    intrastim_surprise_pvalues = []

    for s, subject in tqdm(enumerate(data["subjects"])):
        if subject != 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            indices_to_keep = []
            correct_list = []
            feature_list = []
            correlations_array = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['similarity_matrices'][s]  # keep all channels

            first_trials_to_remove = 50
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:  # remove first trials
                    if stim in subject_stim_features['stim_name'].tolist():
                        # if 'rep2' in stim:
                        if not np.isnan(
                                subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                            if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (
                                    stim[9:].replace('rep1',
                                                     'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                                indices_to_keep.append(i)
                                correct_list.append(
                                    subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[
                                        0])
                                feature_list.append(
                                    subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[
                                        0])
                                if 'rep1' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                                elif 'rep2' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                else:
                                    raise ValueError
                                correlations_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

            correlations_array = np.array(correlations_array)

            intrastim_correct_params.append([])
            intrastim_correct_pvalues.append([])
            intrastim_surprise_params.append([])
            intrastim_surprise_pvalues.append([])
            for channel in range(correlations_array.shape[1]):
                Y = np.array(correct_list)
                X = np.zeros((Y.shape[0], 1))
                X[:, 0] = correlations_array[:, channel]
                X = (X - X.mean()) / X.std()
                X = sm.add_constant(X)
                ols = sm.Logit(Y, X).fit()
                intrastim_correct_params[-1].append(ols.params[1])
                intrastim_correct_pvalues[-1].append(ols.pvalues[1])

                Y = X[:, 1]
                X = np.zeros((Y.shape[0], 1))
                X[:, 0] = feature_list
                X = (X - X.mean(axis=0)) / X.std(axis=0)
                X = sm.add_constant(X)
                ols = sm.OLS(Y, X).fit()
                intrastim_surprise_params[-1].append(ols.params[1])
                intrastim_surprise_pvalues[-1].append(ols.pvalues[1])


    ############## Interstim correlation -> incorrect ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(distance_path, allow_pickle=True)

    interstim_correct_params = []
    interstim_correct_pvalues = []
    interstim_surprise_params = []
    interstim_surprise_pvalues = []

    for s, subject in tqdm(enumerate(data["subjects"])):
        if subject != 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['distance_matrices'][s]  # keep all channels
            paired_stim = []
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:  # remove first trials
                    if ('rep1' in stim) | ('rep2' in stim):
                    # if 'rep1' in stim:
                        if stim in subject_stim_features['stim_name'].tolist():
                            if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                # if subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0] < 0.15:  # check that it works without extreme trials
                                indices_to_keep.append(i)
                                correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                if 'rep2' in stim:
                                    # indices_to_keep.append(i)
                                    # correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    # feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                    if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                        paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))
                                    else:
                                        paired_stim.append('no_pair')
                                if 'rep1' in stim:
                                    # indices_to_keep.append(i)
                                    # correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    # feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                    if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                                        paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2')))
                                    else:
                                        paired_stim.append('no_pair')
                                # else:
                                #     raise ValueError('Neither rep1 nor rep2???')
            condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
            # condition_matrix = condition_matrix[:, indices_to_keep, :]  # select condition
            correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
            for i in range(condition_matrix.shape[0]):
                current_corr = condition_matrix[i, :, :]
                mask = np.ones(condition_matrix.shape[1], dtype=bool)
                mask[indices_to_keep[i]] = False  # remove current stim
                mask[indices_to_keep[i]:] = False  # remove following trials
                correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)

            interstim_correct_params.append([])
            interstim_correct_pvalues.append([])
            interstim_surprise_params.append([])
            interstim_surprise_pvalues.append([])
            for channel in range(correlations_array.shape[1]):
                Y = np.array(correct_list)
                X = np.zeros((Y.shape[0], 1))
                X[:, 0] = correlations_array[:, channel]
                X = (X - X.mean()) / X.std()
                X = sm.add_constant(X)
                ols = sm.Logit(Y, X).fit()
                interstim_correct_params[-1].append(ols.params[1])
                interstim_correct_pvalues[-1].append(ols.pvalues[1])

                Y = X[:, 1]
                X = np.zeros((Y.shape[0], 1))
                X[:, 0] = feature_list
                X = (X - X.mean(axis=0)) / X.std(axis=0)
                X = sm.add_constant(X)
                ols = sm.OLS(Y, X).fit()
                interstim_surprise_params[-1].append(ols.params[1])
                interstim_surprise_pvalues[-1].append(ols.pvalues[1])

    representation_stats = {'intrastim_correct_params': intrastim_correct_params,
                            'intrastim_correct_pvalues': intrastim_correct_pvalues,
                            'intrastim_surprise_params': intrastim_surprise_params,
                            'intrastim_surprise_pvalues': intrastim_surprise_pvalues,
                            'interstim_correct_params': interstim_correct_params,
                            'interstim_correct_pvalues': interstim_correct_pvalues,
                            'interstim_surprise_params': interstim_surprise_params,
                            'interstim_surprise_pvalues': interstim_surprise_pvalues,
                            }
    np.save(os.path.join(save_path, 'representation_stats'), representation_stats, allow_pickle=True)



def representation_distance_and_correct_scatter(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=True):
    """Takes all electrodes from a big region (ex: all temporal lobe) and selected the subset of electrodes that show a
    significant RMS contrast (in either novelty detection or retrieval), then run intra and inter stimulus correlation
    analysis, linking them to correct response and surprise"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    subject_names = []
    del subject_paths[1]
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
        # for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty]]):
        # for cond, correct_incorrect in enumerate([[correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            # pval = ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False,
                            #                                permutations=10000, random_state=None,
                            #                                alternative='greater').pvalue  # testing of individual electrodes
                            # if pval < 0.05:
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    print(significant_electrodes)
    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])
    print(chan_count)


    ############## Intrastim correlation -> correct ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    # data = np.load(data_path, allow_pickle=True)
    data = np.load(similarity_path, allow_pickle=True)

    channel_positions_list = []
    subject_id = []
    correct_full_list = []
    feature_full_list = []
    intrastim_distance_list = []
    # pvalues_array = []  # p value for the regression
    p_ttest_array = []  # p value of the contrast correct vs incorrect
    signif_chan_list = []
    for s, subject in tqdm(enumerate(data["subjects"])):
        if subject != 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            correlations_array = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['similarity_matrices'][s]  # keep all channels
            # similarity_matrix = data['distance_matrices'][s]  # keep all channels
            # similarity_matrix = -similarity_matrix
            # similarity_matrix = 1 / similarity_matrix
            first_trials_to_remove = 50  # very sensitive to the first trials, then stabilizes (mostly for interstim distance, which makes sense)
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:  # remove first trials
                    if stim in subject_stim_features['stim_name'].tolist():
                        # if 'rep2' in stim:
                        if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                            if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                                indices_to_keep.append(i)
                                correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                if 'rep1' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                                elif 'rep2' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                else:
                                    raise ValueError
                                correlations_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

            correlations_array = np.array(correlations_array)

            # pvalues = np.zeros(correlations_array.shape[1])
            p_ttest_list = []
            for channel in range(correlations_array.shape[1]):
                if ch_names[channel] in significant_electrodes[subject]:
                    Y = correlations_array[:, channel]
                    X = np.zeros((Y.shape[0], 1))
                    # X = np.zeros((Y.shape[0], 2))
                    X[:, 0] = correct_list
                    # X[:, 1] = np.array(feature_list)**2
                    X = sm.add_constant(X)

                    # independent t-test
                    correct_corr = Y[X[:, 1] == 1]  # axis 0 is the intercept
                    incorrect_corr = Y[X[:, 1] == 0]
                    # p_ttest = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=10000,
                    #                              random_state=None,
                    #                              alternative='greater').pvalue  # testing of individual electrodes
                    #                              # alternative='less').pvalue  # testing of individual electrodes
                    #
                    # p_ttest_list.append(p_ttest)

                    # ols = sm.OLS(Y, X).fit()
                    # param = ols.params[1]
                    # p_ttest = ols.pvalues[1]

                    X = np.zeros((Y.shape[0], 1))
                    X[:, 0] = feature_list
                    X = sm.add_constant(X)
                    ols = sm.OLS(Y, X).fit()
                    param = ols.params[1]
                    p_ttest = ols.pvalues[1]

                    if (param < 0) & (p_ttest < 0.1):  # one-tailed test
                    # if param < 0:
                    # if (param > 0) & (p_ttest[channel] < 0.1):
                    # if p_ttest[channel] < 0.05:
                    # if p_ttest[channel] < 0.1:
                        signif_chan_list.append(subject+'_'+ch_names[channel])
                        correct_full_list += correct_list
                        feature_full_list += feature_list
                        # intrastim_distance_list += Y.tolist()
                        intrastim_distance_list += ((Y - np.mean(Y)) / np.std(Y)).tolist()

                        subject_id += [s for _ in range(len(correct_list))]
                        # channel_positions_list.append(subject_info['chs'][channel]['loc'][:3])
                    # if (param > 0) & (p_ttest[channel] < 0.1):

    intrastim_distance_list = np.array(intrastim_distance_list)
    correct_full_list = np.array(correct_full_list)
    feature_full_list = np.array(feature_full_list)

    print(signif_chan_list)

    fig, axs = plt.subplots(1, 1, figsize=(2, 2))
    df = pd.DataFrame({'x': intrastim_distance_list, 'y': correct_full_list})
    df = df.sort_values('x')
    df['y_smooth'] = df['y'].rolling(window=30, min_periods=1, center=True).mean()
    # axs.scatter(intrastim_distance_list, correct_full_list, s=0.5)
    # axs.scatter(df['x'], df['y_smooth'], alpha=1, s=0.5, label='rolling mean', color=plt.get_cmap("tab10")(1))
    axs.scatter(df['x'], df['y_smooth'], alpha=1, s=0.5, label='rolling mean', color='lightgray')
    # axs.axvline(0, ls='--', color='k')
    axs.set_xlabel(f'Similarity', fontsize=12)
    axs.set_ylabel('Correct', fontsize=12)
    axs.tick_params(axis='y', labelsize=12)
    axs.tick_params(axis='x', labelsize=12)
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    print('n_data_point:', intrastim_distance_list.shape)
    print("correlation intrastim_distance_list and correct_full_list", pearsonr(intrastim_distance_list, correct_full_list, alternative='greater'))

    # num_bins = 5
    num_bins = 10
    # bins = np.linspace(intrastim_distance_list.min(), intrastim_distance_list.max(), num_bins + 1)
    # bin_indices = np.digitize(intrastim_distance_list, bins)  # Assign each point to a bin
    # bin_means = [correct_full_list[bin_indices == i].mean() for i in range(1, len(bins))]
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # plt.plot(bin_centers, bin_means, marker='o', color='red', label="Bin correlation")
    #
    # # num_bins = 10  # Number of bins
    # df = pd.DataFrame({'distance': intrastim_distance_list, 'correct': correct_full_list})
    # df['bin'], bin_edges = pd.qcut(df['distance'], q=num_bins, retbins=True, labels=False)
    # bin_means = df.groupby('bin')['correct'].mean()
    # bin_centers = df.groupby('bin')['distance'].median()
    # plt.plot(bin_centers, bin_means, marker='o', color='green', label="Quantile correlation")

    slope, intercept = np.polyfit(intrastim_distance_list, correct_full_list, 1)
    x_range = np.linspace(min(intrastim_distance_list), max(intrastim_distance_list), 100)
    plt.plot(x_range, slope * x_range + intercept, color='k', label='linear fit', lw=2)

    # quadratic, slope, intercept = np.polyfit(intrastim_distance_list, correct_full_list, 2)
    # x_range = np.linspace(min(intrastim_distance_list), max(intrastim_distance_list), 100)
    # plt.plot(x_range, quadratic * x_range ** 2 + slope * x_range + intercept, color='b', label='quadratic fit')
    plt.tight_layout()
    # plt.legend()

    # plt.show()

    # correlation vs feature
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    df = pd.DataFrame({'x': feature_full_list, 'y': intrastim_distance_list})
    df_avg = df.groupby('x', as_index=False)['y'].mean()
    # axs.scatter(feature_full_list, intrastim_distance_list, s=2, alpha=0.2)
    axs.scatter(df_avg['x'], df_avg['y'], alpha=1, s=20, label='average', color=plt.get_cmap("tab10")(1))
    # axs.axvline(0, ls='--', color='k')
    axs.set_ylabel(f'Similarity', fontsize=17)
    axs.set_xlabel('Surprise', fontsize=17)
    axs.tick_params(axis='y', labelsize=17)
    axs.tick_params(axis='x', labelsize=17)
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # bins = np.linspace(feature_full_list.min(), feature_full_list.max(), num_bins + 1)
    # bin_indices = np.digitize(feature_full_list, bins)  # Assign each point to a bin
    # bin_means = [intrastim_distance_list[bin_indices == i].mean() for i in range(1, len(bins))]
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # plt.plot(bin_centers, bin_means, marker='o', color='red', label="Bin surprise")
    #
    # # num_bins = 10  # Number of bins
    # df = pd.DataFrame({'distance': feature_full_list, 'correct': intrastim_distance_list})
    # df['bin'], bin_edges = pd.qcut(df['distance'], q=num_bins, retbins=True, labels=False)
    # bin_means = df.groupby('bin')['correct'].mean()
    # bin_centers = df.groupby('bin')['distance'].median()
    # plt.plot(bin_centers, bin_means, marker='o', color='green', label="Quantile surprise")

    x_range = np.linspace(min(feature_full_list), max(feature_full_list), 100)
    slope, intercept = np.polyfit(feature_full_list, intrastim_distance_list, 1)
    plt.plot(x_range, slope * x_range + intercept, color='k', label='linear fit', lw=3)

    quadratic, slope, intercept = np.polyfit(feature_full_list, intrastim_distance_list, 2)
    # plt.plot(x_range, quadratic * x_range**2 + slope * x_range + intercept, color='b', label='quadratic fit')

    # exponential fit
    def func(x, a, b, c):
        return a * np.exp(b * x) + c
    popt, pcov = curve_fit(func, feature_full_list, intrastim_distance_list, maxfev = 10000)
    # plt.plot(x_range, func(x_range, *popt), color='m', label='exp fit')
    # compute AIC
    resid = intrastim_distance_list - func(feature_full_list, *popt)
    N = len(intrastim_distance_list)
    k = len(popt)
    sigma2 = np.sum(resid ** 2) / N
    logL = -0.5 * N * (np.log(2 * np.pi * sigma2) + 1)
    aic_exponential = 2 * k - 2 * logL

    plt.legend()

    # compare linear and quadratic models' AIC
    Y = intrastim_distance_list
    X = np.zeros((Y.shape[0], 1))
    X[:, 0] = feature_full_list
    X = sm.add_constant(X)
    ols = sm.OLS(Y, X).fit()
    aic_linear = ols.aic
    X = np.zeros((Y.shape[0], 2))
    X[:, 0] = feature_full_list
    X[:, 1] = feature_full_list**2
    X = sm.add_constant(X)
    ols = sm.OLS(Y, X).fit()
    aic_quadratic = ols.aic
    # # plt.text(plt.xlim()[0], plt.ylim()[1], f"Linear fit AIC = {round(aic_linear, 2)}, quadratic fit AIC = {round(aic_quadratic, 2)}")
    # plt.text(plt.xlim()[0], plt.ylim()[1], f"Lin AIC = {round(aic_linear, 2)}, Exp AIC = {round(aic_exponential, 2)}")
    plt.tight_layout()



    ############## Interstim correlation -> incorrect ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    # data = np.load(data_path, allow_pickle=True)
    data = np.load(distance_path, allow_pickle=True)

    channel_positions_list = []
    subject_id = []
    correct_full_list = []
    feature_full_list = []
    interstim_distance_list = []
    signif_chan_list2 = []
    # pvalues_array = []  # p value for the regression
    # p_ttest_array = []  # p value of the contrast correct vs incorrect
    for s, subject in tqdm(enumerate(data["subjects"])):

        if not subject == 'Bou_Ni':

            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            # similarity_matrix = data['similarity_matrices'][s]  # keep all channels
            # similarity_matrix = -similarity_matrix
            similarity_matrix = data['distance_matrices'][s]  # keep all channels
            paired_stim = []
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:  # remove first trials
                    if ('rep1' in stim) | ('rep2' in stim):
                    # if 'rep2' in stim:
                        if stim in subject_stim_features['stim_name'].tolist():
                            if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                indices_to_keep.append(i)
                                correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                if 'rep2' in stim:
                                    # indices_to_keep.append(i)
                                    # correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    # feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                    if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                        paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))
                                    else:
                                        paired_stim.append('no_pair')
                                if 'rep1' in stim:
                                    # indices_to_keep.append(i)
                                    # correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    # feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                    if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                                        paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2')))
                                    else:
                                        paired_stim.append('no_pair')
                                # else:
                                #     raise ValueError('Neither rep1 nor rep2???')
            # plt.imshow(np.mean(similarity_matrix, axis=-1), cmap='gray')
            # plt.show()
            condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
            # condition_matrix = condition_matrix[:, indices_to_keep, :]  # select condition
            correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
            for i in range(condition_matrix.shape[0]):
                current_corr = condition_matrix[i, :, :]
                mask = np.ones(condition_matrix.shape[1], dtype=bool)
                mask[indices_to_keep[i]:] = False  # remove current stim and following trials
                correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)
            # print(correlations_array.shape)
            # plt.figure()
            # plt.plot(correlations_array)
            # plt.show()

            # pvalues = np.zeros(correlations_array.shape[1])
            p_ttest = np.zeros(correlations_array.shape[1])
            for channel in range(correlations_array.shape[1]):
                if ch_names[channel] in significant_electrodes[subject]:
                    Y = correlations_array[:, channel]
                    X = np.zeros((Y.shape[0], 1))
                    # X = np.zeros((Y.shape[0], 2))
                    X[:, 0] = correct_list
                    # X[:, 1] = np.array(feature_list)**2
                    X = sm.add_constant(X)

                    # independent t-test
                    correct_corr = Y[X[:, 1] == 1]  # axis 0 is the intercept
                    incorrect_corr = Y[X[:, 1] == 0]
                    # p_ttest[channel] = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=10000,
                    #                              random_state=None,
                    #                              # alternative='less').pvalue  # testing of individual electrodes
                    #                              alternative='greater').pvalue  # testing of individual electrodes

                    # ols = sm.OLS(Y, X).fit()
                    # param = ols.params[1]
                    # p_ttest[channel] = ols.pvalues[1]

                    X = np.zeros((Y.shape[0], 1))
                    X[:, 0] = feature_list
                    X = sm.add_constant(X)
                    ols = sm.OLS(Y, X).fit()
                    param = ols.params[1]
                    p_ttest[channel] = ols.pvalues[1]

                    if (param > 0) & (p_ttest[channel] < 0.1):  # one-tailed test
                    # if (param < 0) & (p_ttest[channel] < 0.1):
                    # if param > 0:
                    # if p_ttest[channel] < 0.05:
                    # if p_ttest[channel] < 0.1:
                        signif_chan_list2.append(subject + '_' + ch_names[channel])
                        correct_full_list += correct_list
                        feature_full_list += feature_list
                        # interstim_distance_list += Y.tolist()
                        interstim_distance_list += ((Y - np.mean(Y)) / np.std(Y)).tolist()

                        subject_id += [s for _ in range(len(correct_list))]

                        # channel_positions_list.append(subject_info['chs'][channel]['loc'][:3])
                    # if (param > 0) & (p_ttest[channel] < 0.05):

    interstim_distance_list = np.array(interstim_distance_list)
    correct_full_list = np.array(correct_full_list)
    feature_full_list = np.array(feature_full_list)

    print(signif_chan_list2)

    fig, axs = plt.subplots(1, 1, figsize=(2, 2))
    df = pd.DataFrame({'x': interstim_distance_list, 'y': correct_full_list})
    df = df.sort_values('x')
    df['y_smooth'] = df['y'].rolling(window=30, min_periods=1, center=True).mean()
    # axs.scatter(interstim_distance_list, correct_full_list, s=0.5)
    # axs.scatter(df['x'], df['y_smooth'], alpha=1, s=0.5, label='rolling mean', color=plt.get_cmap("tab10")(1))
    axs.scatter(df['x'], df['y_smooth'], alpha=1, s=0.5, label='rolling mean', color='lightgray')
    # axs.axvline(0, ls='--', color='k')
    axs.set_xlabel(f'Separability', fontsize=12)
    axs.set_ylabel('Correct', fontsize=12)
    axs.tick_params(axis='y', labelsize=12)
    axs.tick_params(axis='x', labelsize=12)
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    print('n_data_point:', interstim_distance_list.shape)
    print("correlation interstim_distance_list and correct_full_list", pearsonr(interstim_distance_list, correct_full_list, alternative='greater'))

    # bins = np.linspace(interstim_distance_list.min(), interstim_distance_list.max(), num_bins + 1)
    # bin_indices = np.digitize(interstim_distance_list, bins)  # Assign each point to a bin
    # bin_means = [correct_full_list[bin_indices == i].mean() for i in range(1, len(bins))]
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # plt.plot(bin_centers, bin_means, marker='o', color='red', label="Bin distance")
    #
    # # num_bins = 10  # Number of bins
    # df = pd.DataFrame({'distance': interstim_distance_list, 'correct': correct_full_list})
    # df['bin'], bin_edges = pd.qcut(df['distance'], q=num_bins, retbins=True, labels=False)
    # bin_means = df.groupby('bin')['correct'].mean()
    # bin_centers = df.groupby('bin')['distance'].median()
    # plt.plot(bin_centers, bin_means, marker='o', color='green', label="Quantile distance")

    slope, intercept = np.polyfit(interstim_distance_list, correct_full_list, 1)
    x_range = np.linspace(min(interstim_distance_list), max(interstim_distance_list), 100)
    plt.plot(x_range, slope * x_range + intercept, color='k', label='linear fit', lw=2)

    # quadratic, slope, intercept = np.polyfit(interstim_distance_list, correct_full_list, 2)
    # x_range = np.linspace(min(interstim_distance_list), max(interstim_distance_list), 100)
    # plt.plot(x_range, quadratic * x_range ** 2 + slope * x_range + intercept, color='b', label='quadratic fit')
    plt.tight_layout()
    # plt.legend()

    # interstim distance vs surprise
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    df = pd.DataFrame({'x': feature_full_list, 'y': interstim_distance_list})
    df_avg = df.groupby('x', as_index=False)['y'].mean()
    # axs.scatter(feature_full_list, interstim_distance_list, s=2, alpha=0.2)
    axs.scatter(df_avg['x'], df_avg['y'], alpha=1, s=20, label='average', color=plt.get_cmap("tab10")(1))
    # axs.axvline(0, ls='--', color='k')
    axs.set_ylabel(f'Separability', fontsize=17)
    axs.set_xlabel('Surprise', fontsize=17)
    axs.tick_params(axis='y', labelsize=17)
    axs.tick_params(axis='x', labelsize=17)
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # bins = np.linspace(feature_full_list.min(), feature_full_list.max(), num_bins + 1)
    # bin_indices = np.digitize(feature_full_list, bins)  # Assign each point to a bin
    # bin_means = [interstim_distance_list[bin_indices == i].mean() for i in range(1, len(bins))]
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # plt.plot(bin_centers, bin_means, marker='o', color='red', label="Bin surprise")
    #
    # # num_bins = 10  # Number of bins
    # df = pd.DataFrame({'distance': feature_full_list, 'correct': interstim_distance_list})
    # df['bin'], bin_edges = pd.qcut(df['distance'], q=num_bins, retbins=True, labels=False)
    # bin_means = df.groupby('bin')['correct'].mean()
    # bin_centers = df.groupby('bin')['distance'].median()
    # plt.plot(bin_centers, bin_means, marker='o', color='green', label="Quantile surprise")

    slope, intercept = np.polyfit(feature_full_list, interstim_distance_list, 1)
    x_range = np.linspace(min(feature_full_list), max(feature_full_list), 100)
    plt.plot(x_range, slope * x_range + intercept, color='k', label='linear fit', lw=3)

    quadratic, slope, intercept = np.polyfit(feature_full_list, interstim_distance_list, 2)
    x_range = np.linspace(min(feature_full_list), max(feature_full_list), 100)
    # plt.plot(x_range, quadratic * x_range ** 2 + slope * x_range + intercept, color='b', label='quadratic fit')

    # exponential fit
    popt, pcov = curve_fit(func, feature_full_list, interstim_distance_list)
    # plt.plot(x_range, func(x_range, *popt), color='m', label='exp fit')
    # compute AIC
    resid = interstim_distance_list - func(feature_full_list, *popt)
    N = len(interstim_distance_list)
    k = len(popt)
    sigma2 = np.sum(resid ** 2) / N
    logL = -0.5 * N * (np.log(2 * np.pi * sigma2) + 1)
    aic_exponential = 2 * k - 2 * logL

    plt.legend()

    # compare linear and quadratic models' AIC
    Y = interstim_distance_list
    X = np.zeros((Y.shape[0], 1))
    X[:, 0] = feature_full_list
    X = sm.add_constant(X)
    ols = sm.OLS(Y, X).fit()
    aic_linear = ols.aic
    X = np.zeros((Y.shape[0], 2))
    X[:, 0] = feature_full_list
    X[:, 1] = feature_full_list ** 2
    X = sm.add_constant(X)
    ols = sm.OLS(Y, X).fit()
    aic_quadratic = ols.aic
    # plt.text(plt.xlim()[0], plt.ylim()[1],
    #          # f"Lin AIC = {round(aic_linear, 2)}, quad AIC = {round(aic_quadratic, 2)}, exp AIC = {round(aic_exponential, 2)}")
    #          f"Lin AIC = {round(aic_linear, 2)}, Exp AIC = {round(aic_exponential, 2)}")
    plt.tight_layout()


    plt.show()


def representation_distance_distributions(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=True):
    """Takes all electrodes from a big region (ex: all temporal lobe) and selected the subset of electrodes that show a
    significant RMS contrast (in either novelty detection or retrieval), then run intra and inter stimulus correlation
    analysis, linking them to correct response and surprise"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    subject_names = []
    del subject_paths[1]
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    print(significant_electrodes)
    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])
    print(chan_count)


    ############## Intrastim correlation -> correct ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(similarity_path, allow_pickle=True)

    intrastim_distance_list = []

    for s, subject in tqdm(enumerate(data["subjects"])):
        if subject != 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            correlations_array = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['similarity_matrices'][s]  # keep all channels
            first_trials_to_remove = 50  # very sensitive to the first trials, then stabilizes (mostly for interstim distance, which makes sense)
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:  # remove first trials
                    if stim in subject_stim_features['stim_name'].tolist():
                        if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                            if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                                indices_to_keep.append(i)
                                correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                if 'rep1' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                                elif 'rep2' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                else:
                                    raise ValueError
                                correlations_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

            correlations_array = np.array(correlations_array)

            for channel in range(correlations_array.shape[1]):
                if ch_names[channel] in significant_electrodes[subject]:
                    intrastim_distance_list.append(np.mean(correlations_array[:, channel]))

    intrastim_distance_list = np.array(intrastim_distance_list)

    ############## Interstim correlation -> incorrect ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(distance_path, allow_pickle=True)

    interstim_distance_list = []
    for s, subject in tqdm(enumerate(data["subjects"])):

        if not subject == 'Bou_Ni':

            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['distance_matrices'][s]  # keep all channels
            paired_stim = []
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:  # remove first trials
                    if ('rep1' in stim) | ('rep2' in stim):
                            if stim in subject_stim_features['stim_name'].tolist():
                                if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                    indices_to_keep.append(i)
                                    correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                    if 'rep2' in stim:
                                        if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                            paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))
                                        else:
                                            paired_stim.append('no_pair')
                                    if 'rep1' in stim:
                                        if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                                            paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2')))
                                        else:
                                            paired_stim.append('no_pair')

            condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
            correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
            for i in range(condition_matrix.shape[0]):
                current_corr = condition_matrix[i, :, :]
                mask = np.ones(condition_matrix.shape[1], dtype=bool)
                mask[indices_to_keep[i]] = False  # remove current stim
                if not paired_stim[i] == 'no_pair':
                    mask[paired_stim[i]] = False  # remove pair
                correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)

            for channel in range(correlations_array.shape[1]):
                if ch_names[channel] in significant_electrodes[subject]:
                    interstim_distance_list.append(np.mean(correlations_array[:, channel]))

    interstim_distance_list = np.array(interstim_distance_list)

    # plot distributions
    distance_names = ['Similarity', 'Separability']
    palettes = ['#FF928A', '#978EF5']
    for i, distance in enumerate([intrastim_distance_list, interstim_distance_list]):
        df = pd.DataFrame({distance_names[i]: distance})
        plt.figure(figsize=(5, 5))
        sns.violinplot(data=df, showmeans=True, showfliers=False, palette=[palettes[i]], orient='v', inner=None)
        sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
        plt.gca().axhline(0, ls='--', color='k')
        plt.ylabel(distance_names[i], fontsize=17)
        plt.tick_params(axis='y', labelsize=17)
        plt.tick_params(axis='x', labelsize=17)
        # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.scatter(0, np.mean(distance), color='w', edgecolor='k', lw=2, s=75, zorder=10)
        plt.tight_layout()
    plt.show()




def intra_and_interstim_and_correct(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=True):
    """"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    del subject_paths[1]
    subject_names = []
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
        # for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty]]):
        # for cond, correct_incorrect in enumerate([[correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            # pval = ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False,
                            #                                permutations=10000, random_state=None,
                            #                                alternative='greater').pvalue  # testing of individual electrodes
                            # if pval < 0.05:
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    print(significant_electrodes)
    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])
    print(chan_count)


    ############## Representation analysis ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(similarity_path, allow_pickle=True)
    distance_data = np.load(distance_path, allow_pickle=True)

    channel_positions_list = []
    subject_id = []
    correct_full_list = []
    intrastim_distance_full_array = []
    interstim_distance_full_array = []
    # pvalues_array = []  # p value for the regression
    p_ttest_array = []  # p value of the contrast correct vs incorrect
    for s, subject in tqdm(enumerate(data["subjects"])):
        if subject != 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            intrastim_distance_array = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['similarity_matrices'][s]  # keep all channels
            distance_matrix = distance_data['distance_matrices'][s]  # keep all channels
            first_trials_to_remove = 50
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:
                    if stim in subject_stim_features['stim_name'].tolist():
                        # if 'rep2' in stim:
                        if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                            if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                                indices_to_keep.append(i)
                                correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                if 'rep1' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                                elif 'rep2' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                else:
                                    raise ValueError
                                intrastim_distance_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

            intrastim_distance_array = np.array(intrastim_distance_array)
            # condition_matrix = -similarity_matrix[indices_to_keep, :, :]  # select condition
            condition_matrix = distance_matrix[indices_to_keep, :, :]
            interstim_distance_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
            for i in range(condition_matrix.shape[0]):
                current_corr = condition_matrix[i, :, :]
                mask = np.ones(condition_matrix.shape[1], dtype=bool)
                mask[indices_to_keep[i]:] = False  # remove current stim and following trials
                interstim_distance_array[i, :] = np.mean(current_corr[mask, :], axis=0)

            channels_to_keep = []
            for channel in range(intrastim_distance_array.shape[1]):
                if ch_names[channel] in significant_electrodes[subject]:
                    X = np.array(correct_list)
                    Y = intrastim_distance_array[:, channel]
                    correct_corr = Y[X == 1]  # axis 0 is the intercept
                    incorrect_corr = Y[X == 0]
                    p_ttest_intra = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=10000,
                                                 random_state=None,
                                                 # alternative='less').pvalue  # testing of individual electrodes
                                                 alternative='greater').pvalue  # testing of individual electrodes
                    Y = interstim_distance_array[:, channel]
                    correct_corr = Y[X == 1]  # axis 0 is the intercept
                    incorrect_corr = Y[X == 0]
                    p_ttest_inter = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=10000,
                                              random_state=None,
                                              alternative='greater').pvalue  # testing of individual electrodes
                                              # alternative='less').pvalue  # testing of individual electrodes
                    # if (p_ttest_intra < 0.05) | (p_ttest_inter < 0.05):
                    if (p_ttest_intra < 0.1) | (p_ttest_inter < 0.1):
                    # if (p_ttest_intra < 0.05) & (p_ttest_inter < 0.05):
                        channels_to_keep.append(channel)

            print(intrastim_distance_array.shape)
            print(interstim_distance_array.shape)
            print(len(correct_list))
            if len(channels_to_keep) > 0:
                correct_full_list += correct_list
                intra = np.mean(intrastim_distance_array[:, channels_to_keep], axis=1)
                inter = np.mean(interstim_distance_array[:, channels_to_keep], axis=1)
                intrastim_distance_full_array += ((intra - np.mean(intra)) / np.std(intra)).tolist()
                interstim_distance_full_array += ((inter - np.mean(inter)) / np.std(inter)).tolist()
                subject_id += [s for _ in range(intra.shape[0])]

                # from scipy.stats import gaussian_kde
                # # Compute KDE weighted by correctness
                # xy = np.vstack([np.mean(intrastim_distance_array[:, channels_to_keep], axis=1), np.mean(interstim_distance_array[:, channels_to_keep], axis=1)])
                # kde = gaussian_kde(xy, weights=np.array(correct_list))
                #
                # # Create a grid for density estimation
                # # x_grid, y_grid = np.meshgrid(
                # #     np.linspace(np.mean(intrastim_distance_array, axis=1).min(), np.mean(intrastim_distance_array, axis=1).max(), 100),
                # #     np.linspace(np.mean(interstim_distance_array, axis=1).min(), np.mean(interstim_distance_array, axis=1).max(), 100))
                # # z = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)
                #
                # # Plot KDE
                # plt.figure(figsize=(8, 6))
                # # plt.contourf(x_grid, y_grid, z, levels=20, cmap="Reds")  # Density of correct responses
                # # plt.colorbar(label="Density of Correct Responses")
                # plt.scatter(np.mean(intrastim_distance_array[:, channels_to_keep], axis=1)[np.array(correct_list) == 1],
                #             np.mean(interstim_distance_array[:, channels_to_keep], axis=1)[np.array(correct_list) == 1], c="k", s=5, alpha=1)
                # plt.scatter(np.mean(intrastim_distance_array[:, channels_to_keep], axis=1)[np.array(correct_list) == 0],
                #             np.mean(interstim_distance_array[:, channels_to_keep], axis=1)[np.array(correct_list) == 0], c="r", s=5, alpha=1)
                # plt.xlabel("intrastim distance")
                # plt.ylabel("interstim distance")
                # plt.show()

    correct_full_list = np.array(correct_full_list)
    intrastim_distance_full_array = np.array(intrastim_distance_full_array)
    interstim_distance_full_array = np.array(interstim_distance_full_array)
    print(correct_full_list.shape)
    print(intrastim_distance_full_array.shape)
    print(interstim_distance_full_array.shape)

    # # Define grid
    # bins = 10
    # H_correct, xedges, yedges = np.histogram2d(intrastim_distance_full_array, interstim_distance_full_array, bins=bins, weights=correct_full_list)
    # H_total, _, _ = np.histogram2d(intrastim_distance_full_array, interstim_distance_full_array, bins=bins)
    # # Avoid division by zero
    # density_correct = np.divide(H_correct, H_total, out=np.zeros_like(H_correct), where=H_total > 0)
    # # Plot 2D histogram
    # plt.figure(figsize=(8, 6))
    # plt.imshow(density_correct.T, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect="auto", cmap="Reds")
    # plt.colorbar(label="Proportion of Correct Responses")
    # plt.xlabel("intrastim distance")
    # plt.ylabel("interstim distance")
    # plt.show()

    print(intrastim_distance_full_array.min())
    print(intrastim_distance_full_array.max())
    print(interstim_distance_full_array.min())
    print(interstim_distance_full_array.max())

    from scipy.stats import gaussian_kde
    # Compute KDE weighted by correctness
    xy = np.vstack([intrastim_distance_full_array, interstim_distance_full_array])
    kde = gaussian_kde(xy, weights=correct_full_list)

    # Create a grid for density estimation
    x_grid, y_grid = np.meshgrid(np.linspace(intrastim_distance_full_array.min(), intrastim_distance_full_array.max(), 100),
                                 np.linspace(interstim_distance_full_array.min(), interstim_distance_full_array.max(), 100))
    z = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)

    print(intrastim_distance_full_array.shape)
    person_corr = pearsonr(intrastim_distance_full_array, interstim_distance_full_array, alternative='two-sided')
    print('Correlation test')
    print(person_corr)
    spearman_corr = spearmanr(intrastim_distance_full_array, interstim_distance_full_array, alternative='two-sided')
    print('Spearman')
    print(spearman_corr)

    # test with mixed model
    data_dict = {
        'inter': interstim_distance_full_array,
        'intra': intrastim_distance_full_array,
        'correct': correct_full_list,
        'SubjectID': subject_id
    }
    model_data = pd.DataFrame(data_dict)
    model = MixedLM.from_formula('inter ~ intra', groups='SubjectID', data=model_data)
    result = model.fit()
    print(result.summary())

    print('Predict correct from simi and sepa')
    model = MixedLM.from_formula('correct ~ intra + inter + intra*inter', groups='SubjectID', data=model_data)
    result = model.fit()
    print(result.summary())

    # Plot KDE
    # plt.figure(figsize=(8, 6))
    # plt.contourf(x_grid, y_grid, z, levels=20, cmap="Reds")  # Density of correct responses
    # plt.colorbar(label="Density of Correct Responses")

    # slope, intercept = np.polyfit(intrastim_distance_full_array, interstim_distance_full_array, 1)
    # x_range = np.linspace(min(intrastim_distance_full_array), max(intrastim_distance_full_array), 100)
    # plt.plot(x_range, slope * x_range + intercept, color='k')
    # slope, intercept = np.polyfit(intrastim_distance_full_array[correct_full_list == 1],
    #                               interstim_distance_full_array[correct_full_list == 1], 1)
    # x_range = np.linspace(min(intrastim_distance_full_array[correct_full_list == 1]),
    #                       max(intrastim_distance_full_array[correct_full_list == 1]), 100)
    # plt.plot(x_range, slope * x_range + intercept, color='g')
    # slope, intercept = np.polyfit(intrastim_distance_full_array[correct_full_list == 0],
    #                               interstim_distance_full_array[correct_full_list == 0], 1)
    # x_range = np.linspace(min(intrastim_distance_full_array[correct_full_list == 0]),
    #                       max(intrastim_distance_full_array[correct_full_list == 0]), 100)
    # plt.plot(x_range, slope * x_range + intercept, color='r')

    num_bins = 10
    # bins = np.linspace(intrastim_distance_full_array.min(), intrastim_distance_full_array.max(), num_bins + 1)
    # bin_indices = np.digitize(intrastim_distance_full_array, bins)  # Assign each point to a bin
    # bin_means = [interstim_distance_full_array[bin_indices == i].mean() for i in range(1, len(bins))]
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # plt.plot(bin_centers, bin_means, marker='o', color='k', label="All data")
    #
    # # bins = np.linspace(intrastim_distance_full_array[correct_full_list == 1].min(), intrastim_distance_full_array[correct_full_list == 1].max(), num_bins + 1)
    # bin_indices = np.digitize(intrastim_distance_full_array[correct_full_list == 1], bins)  # Assign each point to a bin
    # bin_means = [interstim_distance_full_array[correct_full_list == 1][bin_indices == i].mean() for i in range(1, len(bins))]
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # plt.plot(bin_centers, bin_means, marker='o', color='g', label="Correct")
    #
    # # bins = np.linspace(intrastim_distance_full_array[correct_full_list == 0].min(), intrastim_distance_full_array[correct_full_list == 0].max(), num_bins + 1)
    # bin_indices = np.digitize(intrastim_distance_full_array[correct_full_list == 0], bins)  # Assign each point to a bin
    # bin_means = [interstim_distance_full_array[correct_full_list == 0][bin_indices == i].mean() for i in range(1, len(bins))]
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # plt.plot(bin_centers, bin_means, marker='o', color='r', label="Incorrect")

    fig, axs = plt.subplots(1, 2, figsize=(5, 5))
    intra = intrastim_distance_full_array[correct_full_list == 1]
    inter = interstim_distance_full_array[correct_full_list == 1]
    to_keep = ((np.abs(intra - intra.mean())) <= 2 * intra.std()) & ((np.abs(inter - inter.mean())) <= 2 * inter.std())
    intra = intra[to_keep]
    inter = inter[to_keep]
    axs[0].scatter(intra, inter, c='#ff5a5f', s=5, alpha=0.2)
    df = pd.DataFrame({'x': intrastim_distance_full_array[correct_full_list == 1], 'y': interstim_distance_full_array[correct_full_list == 1]})
    df = df.sort_values('x')
    df['y_smooth'] = df['y'].rolling(window=30, min_periods=1, center=True).mean()
    axs[0].plot(df['x'], df['y_smooth'], alpha=1, color="#ff5a5f", lw=3, label='Correct')

    intra = intrastim_distance_full_array[correct_full_list == 0]
    inter = interstim_distance_full_array[correct_full_list == 0]
    to_keep = ((np.abs(intra - intra.mean())) <= 2 * intra.std()) & ((np.abs(inter - inter.mean())) <= 2 * inter.std())
    intra = intra[to_keep]
    inter = inter[to_keep]
    axs[0].scatter(intra, inter, c="#087e8b", s=5, alpha=0.2)
    df = pd.DataFrame({'x': intrastim_distance_full_array[correct_full_list == 0], 'y': interstim_distance_full_array[correct_full_list == 0]})
    df = df.sort_values('x')
    df['y_smooth'] = df['y'].rolling(window=30, min_periods=1, center=True).mean()
    axs[0].plot(df['x'], df['y_smooth'], alpha=1, color="#087e8b", lw=3, label='Incorrect')

    axs[0].legend()
    axs[0].set_xlabel("Similarity", fontsize=17)
    axs[0].set_ylabel("Separability", fontsize=17)
    axs[0].tick_params(axis='y', labelsize=17)
    axs[0].tick_params(axis='x', labelsize=17)
    # plt.text(0, plt.ylim()[1], f"Overall correlation: rho = {round(person_corr.statistic, 2)}, p = {round(person_corr.pvalue, 5)}")


    # plt.figure(figsize=(8, 6))
    # bins = np.linspace(interstim_distance_full_array.min(), interstim_distance_full_array.max(), num_bins + 1)
    # bin_indices = np.digitize(interstim_distance_full_array, bins)  # Assign each point to a bin
    # bin_means = [intrastim_distance_full_array[bin_indices == i].mean() for i in range(1, len(bins))]
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # plt.plot(bin_centers, bin_means, marker='o', color='k', label="All data")
    #
    # # bins = np.linspace(intrastim_distance_full_array[correct_full_list == 1].min(), intrastim_distance_full_array[correct_full_list == 1].max(), num_bins + 1)
    # bin_indices = np.digitize(interstim_distance_full_array[correct_full_list == 1], bins)  # Assign each point to a bin
    # bin_means = [intrastim_distance_full_array[correct_full_list == 1][bin_indices == i].mean() for i in
    #              range(1, len(bins))]
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # plt.plot(bin_centers, bin_means, marker='o', color='g', label="Correct")
    #
    # # bins = np.linspace(intrastim_distance_full_array[correct_full_list == 0].min(), intrastim_distance_full_array[correct_full_list == 0].max(), num_bins + 1)
    # bin_indices = np.digitize(interstim_distance_full_array[correct_full_list == 0], bins)  # Assign each point to a bin
    # bin_means = [intrastim_distance_full_array[correct_full_list == 0][bin_indices == i].mean() for i in
    #              range(1, len(bins))]
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # plt.plot(bin_centers, bin_means, marker='o', color='r', label="Incorrect")

    intra = intrastim_distance_full_array[correct_full_list == 1]
    inter = interstim_distance_full_array[correct_full_list == 1]
    to_keep = ((np.abs(intra - intra.mean())) <= 2 * intra.std()) & ((np.abs(inter - inter.mean())) <= 2 * inter.std())
    intra = intra[to_keep]
    inter = inter[to_keep]
    axs[1].scatter(inter,
                intra, c="#ff5a5f", s=5, alpha=0.2)
    df = pd.DataFrame({'x': inter,
                       'y': intra})
    df = df.sort_values('x')
    df['y_smooth'] = df['y'].rolling(window=30, min_periods=1, center=True).mean()
    axs[1].plot(df['x'], df['y_smooth'], alpha=1, color="#ff5a5f", lw=3, label='Correct')


    print(interstim_distance_full_array.shape)
    intra = intrastim_distance_full_array[correct_full_list == 0]
    inter = interstim_distance_full_array[correct_full_list == 0]
    to_keep = ((np.abs(intra - intra.mean())) <= 2 * intra.std()) & ((np.abs(inter - inter.mean())) <= 2 * inter.std())
    intra = intra[to_keep]
    inter = inter[to_keep]
    axs[1].scatter(inter,
                intra, c="#087e8b", s=5, alpha=0.2)
    df = pd.DataFrame({'x': inter,
                       'y': intra})
    df = df.sort_values('x')
    df['y_smooth'] = df['y'].rolling(window=30, min_periods=1, center=True).mean()
    axs[1].plot(df['x'], df['y_smooth'], alpha=1, color="#087e8b", lw=3, label='Incorrect')

    axs[1].set_ylabel("Similarity", fontsize=17)
    axs[1].set_xlabel("Separability", fontsize=17)
    axs[1].tick_params(axis='y', labelsize=17)
    axs[1].tick_params(axis='x', labelsize=17)

    plt.tight_layout()


    # heatmap
    df = pd.DataFrame({'correct': correct_full_list, 'intrastim_correlation': intrastim_distance_full_array, 'interstim_distance': interstim_distance_full_array})
    df["interstim_distance_bin"], x_bins = pd.cut(df["interstim_distance"], bins=20, retbins=True)
    df["intrastim_correlation_bin"], y_bins = pd.cut(df["intrastim_correlation"], bins=20, retbins=True)
    # df["interstim_distance_bin"], x_bins = pd.qcut(df["interstim_distance"], q=20, retbins=True)
    # df["intrastim_correlation_bin"], y_bins = pd.qcut(df["intrastim_correlation"], q=20, retbins=True)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    df["interstim_distance_bin"] = df["interstim_distance_bin"].cat.rename_categories(np.round(x_centers, 2))
    df["intrastim_correlation_bin"] = df["intrastim_correlation_bin"].cat.rename_categories(np.round(y_centers, 2))

    heatmap_data = df.groupby(["interstim_distance_bin", "intrastim_correlation_bin"])["correct"].mean().unstack()
    print(heatmap_data)

    # Plot heatmap
    fig, axes = plt.subplots(1, 3, figsize=(5, 5))
    sns.heatmap(heatmap_data, cmap="viridis", annot=False, ax=axes[0])
    axes[0].invert_yaxis()
    axes[0].set_title("Proportion of Correct Responses")

    # smooth
    heatmap_filled = heatmap_data.fillna(0)  # Replace NaNs with 0
    weight_matrix = ~heatmap_data.isna()  # Matrix indicating which bins had data
    smoothed_data = gaussian_filter(heatmap_filled, sigma=1.5, mode="nearest")
    smoothed_weights = gaussian_filter(weight_matrix.astype(float), sigma=1.5, mode="nearest")
    heatmap_smoothed = smoothed_data / smoothed_weights
    heatmap_smoothed[weight_matrix == 0] = np.nan  # Re-mask missing values
    print(heatmap_smoothed)
    sns.heatmap(heatmap_smoothed, cmap="viridis", annot=False, ax=axes[1])
    # sns.heatmap(np.log(heatmap_smoothed), cmap="viridis", annot=False)
    axes[1].invert_yaxis()
    axes[1].set_title("Smoothed (gaussian filter)")

    # interpolate
    x_idx, y_idx = np.meshgrid(np.arange(heatmap_smoothed.shape[1]), np.arange(heatmap_smoothed.shape[0]))
    valid_mask = ~np.isnan(heatmap_smoothed)  # Mask of non-NaN values
    valid_x = x_idx[valid_mask]
    valid_y = y_idx[valid_mask]
    valid_values = heatmap_smoothed[valid_mask]
    heatmap_interpolated = griddata((valid_x, valid_y), valid_values, (x_idx, y_idx), method='cubic')
    sns.heatmap(heatmap_interpolated, cmap="viridis", annot=False, ax=axes[2])
    # sns.heatmap(np.log(heatmap_smoothed), cmap="viridis", annot=False)
    axes[2].invert_yaxis()
    axes[2].set_title("Cubic interpolation")

    #  replot interpolated heatmap
    plt.figure(figsize=(5, 5))
    cmap1 = LinearSegmentedColormap.from_list("mycmap", ['#087e8b', '#ff5a5f'])
    ax = sns.heatmap(heatmap_interpolated, cmap=cmap1, annot=False)
    plt.gca().invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel('Similarity bins', fontsize=17)
    plt.ylabel('Separability bins', fontsize=17)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=17)
    cbar.set_label('Proportion correct', fontsize=17, labelpad=20)
    cbar.ax.yaxis.label.set_rotation(270)
    plt.tight_layout()

    plt.show()


def impact_of_first_trials_to_remove(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=True):
    """Plot the evolution of results with first_trials_to_remove"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    subject_names = []
    del subject_paths[1]

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])


    ############## Representation analysis ##############

    # define the function that will be used for regressions (as well as permuted regressions)
    def run_regressions(representation_array):
        """representation_array contains intrastim correlation or interstim distance"""
        param_correct_list = []
        param_feature_list = []
        for channel in range(representation_array.shape[1]):
            if ch_names[channel] in significant_electrodes[subject]:
                Y = np.array(correct_list)
                X = np.zeros((Y.shape[0], 1))
                X[:, 0] = representation_array[:, channel]
                X = (X - X.mean()) / X.std()
                X = sm.add_constant(X)

                ols = sm.Logit(Y, X).fit(disp=False)
                param_correct_list.append(ols.params[1])

                Y = X[:, 1]
                X = np.zeros((Y.shape[0], 1))
                X[:, 0] = feature_list
                X = (X - X.mean(axis=0)) / X.std(axis=0)
                X = sm.add_constant(X)
                ols = sm.OLS(Y, X).fit()
                param_feature_list.append(ols.params[1])

        return param_correct_list, param_feature_list


    ############## Intrastim correlation -> correct ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]

    data = np.load(similarity_path, allow_pickle=True)

    # range of first_trials_to_remove
    first_trials_to_remove_list = [i for i in range(105)]

    regression_param_correct = [[] for _ in range(len(first_trials_to_remove_list))]
    regression_param_feature = [[] for _ in range(len(first_trials_to_remove_list))]

    for c, first_trials_to_remove in enumerate(tqdm(first_trials_to_remove_list)):
        for s, subject in enumerate(data["subjects"]):
            if subject != 'Bou_Ni':
                subject_stim_features = stim_features[stim_features['subject'] == subject]
                subject_info = data['info'][s]
                ch_names = subject_info['ch_names']
                indices_to_keep = []
                correct_list = []
                feature_list = []
                correlations_array = []
                stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
                similarity_matrix = data['similarity_matrices'][s]  # keep all channels

                for i, stim in enumerate(data['stim_names'][s]):
                    if i > first_trials_to_remove:  # remove first trials
                        if stim in subject_stim_features['stim_name'].tolist():
                            if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                                    indices_to_keep.append(i)
                                    correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                    if 'rep1' in stim:
                                        rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                                    elif 'rep2' in stim:
                                        rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                    else:
                                        raise ValueError
                                    correlations_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

                correlations_array = np.array(correlations_array)

                param_correct_list, param_feature_list = run_regressions(correlations_array)
                regression_param_correct[c] += param_correct_list
                regression_param_feature[c] += param_feature_list


    regression_param_correct = np.array(regression_param_correct)
    regression_param_feature = np.array(regression_param_feature)

    intrastim_params = regression_param_correct.copy()
    intrastim_feature_params = regression_param_feature.copy()


    ############## Interstim correlation -> incorrect ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]

    data = np.load(distance_path, allow_pickle=True)

    regression_param_correct = [[] for _ in range(len(first_trials_to_remove_list))]
    regression_param_feature = [[] for _ in range(len(first_trials_to_remove_list))]

    for c, first_trials_to_remove in enumerate(tqdm(first_trials_to_remove_list)):
        for s, subject in enumerate(data["subjects"]):
            if not subject == 'Bou_Ni':
                subject_stim_features = stim_features[stim_features['subject'] == subject]
                subject_info = data['info'][s]
                ch_names = subject_info['ch_names']
                indices_to_keep = []
                correct_list = []
                feature_list = []
                stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
                similarity_matrix = data['distance_matrices'][s]  # keep all channels
                paired_stim = []
                for i, stim in enumerate(data['stim_names'][s]):
                    if i > first_trials_to_remove:  # remove first trials
                        if ('rep1' in stim) | ('rep2' in stim):
                            if stim in subject_stim_features['stim_name'].tolist():
                                if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                    indices_to_keep.append(i)
                                    correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                    if 'rep2' in stim:
                                        if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                            paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))
                                        else:
                                            paired_stim.append('no_pair')
                                    if 'rep1' in stim:
                                        if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                                            paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2')))
                                        else:
                                            paired_stim.append('no_pair')
                condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
                correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
                for i in range(condition_matrix.shape[0]):
                    current_corr = condition_matrix[i, :, :]
                    mask = np.ones(condition_matrix.shape[1], dtype=bool)
                    mask[indices_to_keep[i]] = False  # remove current stim
                    mask[indices_to_keep[i]:] = False  # remove following trials
                    correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)

                param_correct_list, param_feature_list = run_regressions(correlations_array)
                regression_param_correct[c] += param_correct_list
                regression_param_feature[c] += param_feature_list


    regression_param_correct = np.array(regression_param_correct)
    regression_param_feature = np.array(regression_param_feature)
    interstim_params = regression_param_correct.copy()
    interstim_feature_params = regression_param_feature.copy()

    plt.figure(figsize=(5, 5))
    labels = ['Correct ~ Similarity', 'Similarity ~ Surprise', 'Correct ~ Separability', 'Separability ~ Surprise']
    colors = ['#FF928A', '#978EF5', '#fdb462', '#bc80bd']
    for i, params in enumerate([intrastim_params, intrastim_feature_params, interstim_params, interstim_feature_params]):
        mean = np.mean(params, axis=1)
        sem = np.std(params, ddof=1, axis=1) / np.sqrt(params.shape[1])
        plt.plot(first_trials_to_remove_list, mean, c=colors[i], label=labels[i], lw=3)
        plt.fill_between(first_trials_to_remove_list, mean - sem, mean + sem, color=colors[i], alpha=0.3, lw=0)
    plt.gca().axhline(0, ls='--', color='k', zorder=0)
    plt.legend()
    plt.xlabel("First trials to remove", fontsize=17)
    plt.ylabel("\u03B2 parameters", fontsize=17)
    plt.tick_params(axis='y', labelsize=17)
    plt.tick_params(axis='x', labelsize=17)
    plt.tight_layout()

    plt.savefig('/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/corr_and_distance/first_trial_choice_impact.svg')
    plt.show()


def full_dual_mechanism_model(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=True):
    """Select significant channels in intracorr and interdist, fit their exponential relationships with surprise (f_i),
    and run a logistic regression from surprise to behavior (correct response) with f_i as independent variables"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    del subject_paths[1]
    subject_names = []
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            # for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty]]):
            # for cond, correct_incorrect in enumerate([[correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    print(significant_electrodes)
    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])
    print(chan_count)


    ############## Representation analysis ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(similarity_path, allow_pickle=True)
    distance_data = np.load(distance_path, allow_pickle=True)

    # exp function for fitting
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c
    # def exp_func(x, b, c):
    #     return np.exp(b * x) + c  # without param a (creates degeneracy with b)
    # quadratic fit
    def quadr_func(x, a, b, c):
        return a * x**2 + b * x + c
    # linear fit
    def linear_func(x, a, b):
        return a * x + b

    func = linear_func

    subject_id = []
    correct_full_list = []
    intrastim_distance_full_array = []
    interstim_distance_full_array = []
    coefs_intra = []
    coefs_inter = []
    saved_model = []
    behavioral_models = []  # one logistic regression per subject, using surprise and surprise²
    curve_param_intra = []  # parameters of the exponental fit
    curve_param_inter = []
    correlation_intra_inter = []  # check correlation between the two predictors
    r2_intra = []  # r2 of exp fit
    r2_inter = []
    r2_intra_selected = []  # r2 of selected channels only
    r2_inter_selected = []
    logL_full_model = []
    logL_intra = []
    logL_inter = []
    logL_null = []
    logL_behavioral_model = []
    # pvalues_array = []  # p value for the regression
    p_ttest_array = []  # p value of the contrast correct vs incorrect
    for s, subject in tqdm(enumerate(data["subjects"])):
        if subject != 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            intrastim_distance_array = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['similarity_matrices'][s]  # keep all channels
            distance_matrix = distance_data['distance_matrices'][s]  # keep all channels
            first_trials_to_remove = 50
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:
                    if stim in subject_stim_features['stim_name'].tolist():
                        # if 'rep2' in stim:
                        if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                            if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                                indices_to_keep.append(i)
                                correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                if 'rep1' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                                elif 'rep2' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                else:
                                    raise ValueError
                                intrastim_distance_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

            feature_list = np.array(feature_list)
            feature_z = (feature_list - feature_list.mean()) / feature_list.std()
            correct_list = np.array(correct_list)
            intrastim_distance_array = np.array(intrastim_distance_array)
            condition_matrix = distance_matrix[indices_to_keep, :, :]
            interstim_distance_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
            for i in range(condition_matrix.shape[0]):
                current_corr = condition_matrix[i, :, :]
                mask = np.ones(condition_matrix.shape[1], dtype=bool)
                mask[indices_to_keep[i]:] = False  # remove current stim and following trials
                interstim_distance_array[i, :] = np.mean(current_corr[mask, :], axis=0)

            for channel in range(intrastim_distance_array.shape[1]):
                if ch_names[channel] in significant_electrodes[subject]:
                    Y = intrastim_distance_array[:, channel]
                    X = np.zeros((Y.shape[0], 1))
                    X[:, 0] = feature_list
                    X = sm.add_constant(X)
                    ols = sm.OLS(Y, X).fit()
                    param_intra = ols.params[1]
                    p_intra = ols.pvalues[1]

                    Y = interstim_distance_array[:, channel]
                    X = np.zeros((Y.shape[0], 1))
                    X[:, 0] = feature_list
                    X = sm.add_constant(X)
                    ols = sm.OLS(Y, X).fit()
                    param_inter = ols.params[1]
                    p_inter = ols.pvalues[1]

                    # if ((param_intra < 0) & (p_intra < 0.1)) | ((param_inter > 0) & (p_inter < 0.1)):  # either is significant, one-tailed
                    # if ((param_intra < 0) & (p_intra < 0.1)) & ((param_inter > 0) & (p_inter < 0.1)):  # both are significant, one-tailed
                    if 1==1:  # all channels
                        # fit exponentials
                        intra_z = (intrastim_distance_array[:, channel] - np.mean(intrastim_distance_array[:, channel])) / np.std(intrastim_distance_array[:, channel])
                        # intra_z = intrastim_distance_array[:, channel]
                        inter_z = (interstim_distance_array[:, channel] - np.mean(interstim_distance_array[:, channel])) / np.std(interstim_distance_array[:, channel])
                        # inter_z = interstim_distance_array[:, channel]
                        maxfev = 100000
                        if func == exp_func:
                            popt_intra, pcov_intra = curve_fit(exp_func, feature_z, intra_z, p0 = [1, 0, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)
                            popt_inter, pcov_inter = curve_fit(exp_func, feature_z, inter_z, p0 = [1, 0, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)
                        elif func == linear_func:
                            popt_intra, pcov_intra = curve_fit(linear_func, feature_z, intra_z, maxfev=maxfev)
                            popt_inter, pcov_inter = curve_fit(linear_func, feature_z, inter_z, maxfev=maxfev)
                        predicted_intra = func(feature_z, *popt_intra)
                        predicted_inter = func(feature_z, *popt_inter)
                        r2_intra.append(1 - np.sum((predicted_intra - intra_z)**2) / np.sum((intra_z - intra_z.mean())**2))
                        r2_inter.append(1 - np.sum((predicted_inter - inter_z)**2) / np.sum((inter_z - inter_z.mean())**2))

                        r2_threshold = 0.05  # if either fit reaches the threshold, we keep the channel
                        # if (r2_intra[-1] > r2_threshold) | (r2_inter[-1] > r2_threshold):
                        if ((r2_intra[-1] > r2_threshold) & (popt_intra[1] < 0)) | ((r2_inter[-1] > r2_threshold) & (popt_inter[1] > 0)):
                            # if (popt_intra[1] < 0) | (popt_inter[1] > 0):

                            r2_intra_selected.append(r2_intra[-1])
                            r2_inter_selected.append(r2_inter[-1])

                            # b_threshold = -0.1
                            # if (popt_intra[1] < 0) & (popt_inter[1] > 0):
                            # if (popt_intra[0] < 0) & (popt_inter[0] > 0):

                            # Fit logistic regression
                            X = np.zeros((predicted_intra.shape[0], 2))
                            X[:, 0] = predicted_intra
                            X[:, 1] = predicted_inter
                            # X[:, 0] = intra_z
                            # X[:, 1] = inter_z
                            model = LogisticRegression(penalty=None)
                            model.fit(X, correct_list)

                            coefs_intra.append(model.coef_[0, 0])
                            coefs_inter.append(model.coef_[0, 1])
                            saved_model.append(model)
                            curve_param_intra.append(popt_intra)
                            curve_param_inter.append(popt_inter)
                            subject_id.append(s)
                            correlation_intra_inter.append(np.corrcoef(X[:, 0], X[:, 1])[0, 1])

                            # store model logL
                            proba = model.predict_proba(X)
                            logL_full_model.append(logistic_log_likelihood(correct_list, proba[:, 1]))

                            # control models with only one predictor or one intercept
                            model = LogisticRegression(penalty=None)
                            model.fit(X[:, 0][:, np.newaxis], correct_list)
                            logL_intra.append(logistic_log_likelihood(correct_list, model.predict_proba(X[:, 0][:, np.newaxis])[:, 1]))
                            model = LogisticRegression(penalty=None)
                            model.fit(X[:, 1][:, np.newaxis], correct_list)
                            logL_inter.append(logistic_log_likelihood(correct_list, model.predict_proba(X[:, 1][:, np.newaxis])[:, 1]))
                            logL_null.append(logistic_log_likelihood(correct_list, np.sum(correct_list) / correct_list.shape[0]))

            # subject's behavioral model
            if (len(subject_id) > 0) & (subject_id[-1] == s):  # only if at least 1 channel of that subject was added
                X = np.zeros((feature_z.shape[0], 2))
                X[:, 0] = feature_z
                X[:, 1] = feature_z**2
                model = LogisticRegression(penalty=None)
                model.fit(X, correct_list)
                behavioral_models.append(model)
                logL_behavioral_model.append(logistic_log_likelihood(correct_list, model.predict_proba(X)[:, 1]))

    print('n_channels:', len(coefs_intra))

    # stat test
    test_intra = ttest_1samp(coefs_intra, 0, alternative='greater')
    test_inter = ttest_1samp(coefs_inter, 0, alternative='greater')
    pval_intra = test_intra.pvalue
    pval_inter = test_inter.pvalue
    print("test_intra: ", test_intra)
    print("test_inter: ", test_inter)

    pval_intra_permut = permut_1sample_ttest_across_channels(coefs_intra, subject_id, popmean=0, alternative='greater', n_permutations=10000)
    pval_inter_permut = permut_1sample_ttest_across_channels(coefs_inter, subject_id, popmean=0, alternative='greater', n_permutations=10000)
    print("pval_intra_permut: ", pval_intra_permut)
    print("pval_inter_permut: ", pval_inter_permut)

    # boxplots
    df = pd.DataFrame({'Similarity': coefs_intra, 'Separability': coefs_inter})
    plt.figure(figsize=(2.5, 2.5))
    sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5'], orient='v', inner=None, linewidth=1)
    # sns.swarmplot(data=df, color=".25", size=1.5, orient='v')
    plt.gca().axhline(0, ls='--', color='k', lw=0.5)
    plt.ylabel('Logistic \u03B2', fontsize=13)
    # plt.yticks(plt.yticks()[0], plt.yticks()[1], rotation=90, va='center')
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)

    plt.scatter(0, np.mean(coefs_intra), color='w', edgecolor='k', lw=1.5, s=30, zorder=10)
    plt.scatter(1, np.mean(coefs_inter), color='w', edgecolor='k', lw=1.5, s=30, zorder=10)

    # add stars
    for i, p in enumerate([pval_intra, pval_inter]):
        if p < 0.05:
            # Get the position to place the star
            x = i  # The x position corresponds to the box index
            y = df.iloc[:, i].max() + 0.15  # Slightly above the max value of the box
            plt.text(x, y, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
    plt.tight_layout()

    plt.savefig('/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/corr_and_distance/dual_mechanism_model.svg')

    # curve parameters
    curve_param_intra = np.array(curve_param_intra)
    curve_param_inter = np.array(curve_param_inter)
    print(curve_param_intra.shape)

    cond_names = ['Similarity fit', 'Separability fit']
    alternative_b = ['less', 'greater']
    for i, params in enumerate([curve_param_intra, curve_param_inter]):
        df = pd.DataFrame({'param a': params[:, 0] / params[:, 0].std(), 'param b': params[:, 1] / params[:, 1].std()})
        plt.figure(figsize=(5.2, 5))
        sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5'], orient='v', inner=None)
        sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
        plt.gca().axhline(0, ls='--', color='k')
        plt.ylabel(cond_names[i], fontsize=13)
        # plt.yticks(plt.yticks()[0], plt.yticks()[1], rotation=90, va='center')
        plt.tick_params(axis='y', labelsize=13)
        plt.tick_params(axis='x', labelsize=13)

        plt.scatter(0, np.mean(params[:, 0] / params[:, 0].std()), color='w', edgecolor='k', lw=2, s=75, zorder=10)
        plt.scatter(1, np.mean(params[:, 1] / params[:, 1].std()), color='w', edgecolor='k', lw=2, s=75, zorder=10)

        # add stars
        pval_a = ttest_1samp(params[:, 0], 0, alternative='greater').pvalue
        pval_b = ttest_1samp(params[:, 1], 0, alternative=alternative_b[i]).pvalue
        # print(params[:, 0].mean())
        # print(params[:, 1].mean())
        for j, p in enumerate([pval_a, pval_b]):
            # print(p)
            if p < 0.05:
                # Get the position to place the star
                x = j  # The x position corresponds to the box index
                y = df.iloc[:, j].max() + 0.25  # Slightly above the max value of the box
                plt.text(x, y, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')

        plt.tight_layout()

    # param b only
    df = pd.DataFrame({'similarity': curve_param_intra[:, 0], 'separability': curve_param_inter[:, 0]})
    plt.figure(figsize=(5.2, 5))
    sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5'], orient='v',  inner=None)
    sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
    plt.gca().axhline(0, ls='--', color='k')
    plt.ylabel('Parameter b', fontsize=13)
    # plt.yticks(plt.yticks()[0], plt.yticks()[1], rotation=90, va='center')
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)

    plt.scatter(0, np.mean(curve_param_intra[:, 0]), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(1, np.mean(curve_param_inter[:, 0]), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.tight_layout()

    # correlation between predictors
    print(correlation_intra_inter)
    plt.figure()
    df = pd.DataFrame({'correlation_intra_inter': np.array(correlation_intra_inter)})
    sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A'], orient='v', inner=None)
    sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
    plt.gca().axhline(0, ls='--', color='k')
    plt.ylabel('Correlation(predicted_similarity - predicted_separability)', fontsize=13)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)
    plt.scatter(0, np.mean(correlation_intra_inter), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.tight_layout()

    # r2 of exp fits
    df = pd.DataFrame({'r2_intra': r2_intra, 'r2_inter': r2_inter})
    plt.figure(figsize=(5.2, 5))
    sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5'], orient='v', inner=None)
    sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
    plt.gca().axhline(0, ls='--', color='k')
    plt.ylabel('r2 exp fits', fontsize=13)
    # plt.yticks(plt.yticks()[0], plt.yticks()[1], rotation=90, va='center')
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)

    plt.scatter(0, np.mean(r2_intra), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(1, np.mean(r2_inter), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.tight_layout()

    plt.figure()
    plt.scatter(r2_intra, r2_inter, s=0.1)
    plt.xlabel('r2_intra')
    plt.ylabel('r2_inter')
    print(np.sum(np.array(r2_intra) >= 0.05))
    print(np.sum(np.array(r2_inter) >= 0.05))

    # compare model logL
    logL_full_model = np.array(logL_full_model)
    logL_intra = np.array(logL_intra)
    logL_inter = np.array(logL_inter)
    logL_null = np.array(logL_null)

    test_intra = ttest_1samp(logL_full_model - logL_intra, 0, alternative='greater')
    print("ttest logL full model - similarity: ", test_intra)
    test_inter = ttest_1samp(logL_full_model - logL_inter, 0, alternative='greater')
    print("ttest logL full model - separability: ", test_inter)
    test_null = ttest_1samp(logL_full_model - logL_null, 0, alternative='greater')
    print("ttest logL full model - null: ", test_null)

    df = pd.DataFrame({'Full model': logL_full_model, 'Similarity': logL_intra, 'Separability': logL_inter, 'Null': logL_null})
    plt.figure(figsize=(5.2, 5))
    sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5', '#fdb462', '#bc80bd'], orient='v', inner=None)
    sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
    plt.ylabel('Logistic regression log likelihood', fontsize=13)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)

    plt.scatter(0, np.mean(logL_full_model), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(1, np.mean(logL_intra), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(2, np.mean(logL_inter), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(3, np.mean(logL_null), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.tight_layout()


    # plot surprise - neural patterns relationships
    simulated_surprise = np.linspace(feature_z.min(), feature_z.max(), 100)
    curve_param_intra = [curve_param_intra[c, :] for c in range(curve_param_intra.shape[0])]
    simulated_similarity = np.array([func(simulated_surprise, *popt) for popt in curve_param_intra])
    curve_param_inter = [curve_param_inter[c, :] for c in range(curve_param_inter.shape[0])]
    simulated_separability = np.array([func(simulated_surprise, *popt) for popt in curve_param_inter])
    colors = ['#FF928A', '#978EF5']
    fig, axes = plt.subplots(2, 1, figsize=(2.5, 5), sharex=True)
    for ax, simulated in enumerate([simulated_similarity, simulated_separability]):
        axes[ax].plot(simulated_surprise, simulated.T, c=colors[ax], alpha=0.3, lw=0.5)
        mean = np.mean(simulated, axis=0)
        axes[ax].plot(simulated_surprise, mean, c=colors[ax], lw=4)
        axes[ax].set_ylim(mean.min(), mean.max())
        axes[ax].tick_params(axis='y', labelsize=17)
        axes[ax].tick_params(axis='x', labelsize=17)
        axes[ax].set_xlim(simulated_surprise.min(), simulated_surprise.max())
    axes[1].set_xlabel('Surprise', fontsize=17)
    axes[0].set_ylabel('Similarity', fontsize=17)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[1].set_ylabel('Separability', fontsize=17)
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.tight_layout()

    fig.savefig("/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/corr_and_distance/predicted_neural_patterns.svg")

    # plot surprise -> P(correct)
    print(len(saved_model))
    print(simulated_similarity.shape)
    print(simulated_separability.shape)
    proba_correct = np.zeros_like(simulated_similarity)
    for c in range(proba_correct.shape[0]):  # channel models
        X = np.zeros((proba_correct.shape[1], 2))
        X[:, 0] = simulated_similarity[c, :]
        X[:, 1] = simulated_separability[c, :]
        proba_correct[c, :] = saved_model[c].predict_proba(X)[:, 1]
    proba_correct_behavioral = np.zeros((len(behavioral_models), simulated_surprise.shape[0]))
    for s in range(len(behavioral_models)):  # subject models (correct ~ surprise + surprise²)
        X = np.zeros((simulated_surprise.shape[0], 2))
        X[:, 0] = simulated_surprise
        X[:, 1] = simulated_surprise**2
        proba_correct_behavioral[s, :] = behavioral_models[s].predict_proba(X)[:, 1]

    plt.figure(figsize=(5, 5))
    plt.plot(simulated_surprise, proba_correct.T, c='#fdb462', alpha=0.3, lw=0.5, zorder=0)
    weights = logL_full_model - logL_full_model.min() - logL_full_model.max()  # translate to get positive values
    weights = weights / weights.sum()
    sorted_perf = np.argsort(logL_full_model)[::-1]
    # mean = np.sum(weights[:, np.newaxis] * proba_correct, axis=0)  # weighted average with log likelihood
    # mean = np.mean(proba_correct[sorted_perf[:20], :], axis=0)  # take n best channels on logistic regression
    mean = np.mean(proba_correct, axis=0)  # average of all channels
    behavioral_mean = np.mean(proba_correct_behavioral, axis=0)
    plt.plot(simulated_surprise, mean, c='#fdb462', lw=4, label='Model average')
    plt.plot(simulated_surprise, behavioral_mean, c='k', lw=4, label='Quadratic surprise model', zorder=1)
    plt.xlabel('Surprise', fontsize=17)
    plt.ylabel('Correct', fontsize=17)
    plt.legend(fontsize=17)
    plt.tick_params(axis='y', labelsize=17)
    plt.tick_params(axis='x', labelsize=17)
    plt.xlim(simulated_surprise.min(), simulated_surprise.max())
    # plt.ylim(mean.min(), mean.max())
    # plt.ylim(0, 1)
    plt.tight_layout()


    plt.savefig("/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/corr_and_distance/predicted_correct_from_surprise.svg")

    # Comparison with surprise² model
    logL_behavioral_model = np.array(logL_behavioral_model)
    print(np.unique(subject_id))
    print(logL_behavioral_model.shape)
    logL_full_model_by_subject = [np.mean(logL_full_model[subject_id == s]) for s in np.unique(subject_id)]
    print(len(logL_full_model_by_subject))
    test_comparison = ttest_1samp(logL_full_model_by_subject - logL_behavioral_model, 0, alternative='two-sided')
    print("ttest logL full model - surprise model: ", test_comparison)

    df = pd.DataFrame(
        {'Full model': logL_full_model_by_subject, 'Surprise model': logL_behavioral_model})
    plt.figure(figsize=(5, 5))
    sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5'], orient='v', inner=None)
    sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
    plt.ylabel('Logistic regression log likelihood', fontsize=13)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)

    plt.scatter(0, np.mean(logL_full_model_by_subject), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(1, np.mean(logL_behavioral_model), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.tight_layout()


    plt.show()



def full_dual_mechanism_model_R2_threshold_impact(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=True):
    """Select significant channels in intracorr and interdist, fit their exponential relationships with surprise (f_i),
    and run a logistic regression from surprise to behavior (correct response) with f_i as independent variables"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    del subject_paths[1]
    subject_names = []
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])

    ############## Representation analysis ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]

    data = np.load(similarity_path, allow_pickle=True)
    distance_data = np.load(distance_path, allow_pickle=True)

    # exp function for fitting
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    r2_threshold_list = np.linspace(0, 0.2, 100)  # if either fit reaches the threshold, we keep the channel

    coefs_intra = [[] for _ in range(len(r2_threshold_list))]
    coefs_inter = [[] for _ in range(len(r2_threshold_list))]

    for c, r2_threshold in enumerate(tqdm(r2_threshold_list)):
        for s, subject in enumerate(data["subjects"]):
            if subject != 'Bou_Ni':
                subject_stim_features = stim_features[stim_features['subject'] == subject]
                subject_info = data['info'][s]
                ch_names = subject_info['ch_names']
                indices_to_keep = []
                correct_list = []
                feature_list = []
                intrastim_distance_array = []
                stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
                similarity_matrix = data['similarity_matrices'][s]  # keep all channels
                distance_matrix = distance_data['distance_matrices'][s]  # keep all channels
                first_trials_to_remove = 50
                for i, stim in enumerate(data['stim_names'][s]):
                    if i > first_trials_to_remove:
                        if stim in subject_stim_features['stim_name'].tolist():
                            if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                                    indices_to_keep.append(i)
                                    correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                    if 'rep1' in stim:
                                        rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                                    elif 'rep2' in stim:
                                        rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                    else:
                                        raise ValueError
                                    intrastim_distance_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

                feature_list = np.array(feature_list)
                correct_list = np.array(correct_list)
                intrastim_distance_array = np.array(intrastim_distance_array)
                condition_matrix = distance_matrix[indices_to_keep, :, :]
                interstim_distance_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
                for i in range(condition_matrix.shape[0]):
                    current_corr = condition_matrix[i, :, :]
                    mask = np.ones(condition_matrix.shape[1], dtype=bool)
                    mask[indices_to_keep[i]:] = False  # remove current stim and following trials
                    interstim_distance_array[i, :] = np.mean(current_corr[mask, :], axis=0)

                for channel in range(intrastim_distance_array.shape[1]):
                    if ch_names[channel] in significant_electrodes[subject]:
                        # fit exponentials
                        intra_z = (intrastim_distance_array[:, channel] - np.mean(intrastim_distance_array[:, channel])) / np.std(intrastim_distance_array[:, channel])
                        inter_z = (interstim_distance_array[:, channel] - np.mean(interstim_distance_array[:, channel])) / np.std(interstim_distance_array[:, channel])
                        feature_z = (feature_list - feature_list.mean()) / feature_list.std()
                        maxfev = 100000
                        popt_intra, pcov_intra = curve_fit(exp_func, feature_z, intra_z, p0 = [1, 0, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)
                        popt_inter, pcov_inter = curve_fit(exp_func, feature_z, inter_z, p0 = [1, 0, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)
                        predicted_intra = exp_func(feature_z, *popt_intra)
                        predicted_inter = exp_func(feature_z, *popt_inter)
                        r2_intra = 1 - np.sum((predicted_intra - intra_z)**2) / np.sum((intra_z - intra_z.mean())**2)
                        r2_inter = 1 - np.sum((predicted_inter - inter_z)**2) / np.sum((inter_z - inter_z.mean())**2)

                        if (r2_intra > r2_threshold) | (r2_inter > r2_threshold):
                            # Fit logistic regression
                            X = np.zeros((predicted_intra.shape[0], 2))
                            X[:, 0] = predicted_intra
                            X[:, 1] = predicted_inter
                            model = LogisticRegression(penalty=None)
                            model.fit(X, correct_list)

                            coefs_intra[c].append(model.coef_[0, 0])
                            coefs_inter[c].append(model.coef_[0, 1])

    print([len(coefs) for coefs in coefs_intra])
    print([len(coefs) for coefs in coefs_inter])

    fig, axes = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
    labels = ['Similarity', 'Separability']
    colors = ['#FF928A', '#978EF5']
    for i, params in enumerate([coefs_intra, coefs_inter]):
        mean = np.array([np.mean(p) for p in params])  # different number of channels
        sem = np.array([np.std(p, ddof=1) / np.sqrt(len(p)) for p in params])
        # axes[0].plot(r2_threshold_list, mean, c=colors[i], label=labels[i], lw=3)
        # axes[0].fill_between(r2_threshold_list, mean - sem, mean + sem, color=colors[i], alpha=0.3, lw=0)
        axes.plot(r2_threshold_list, mean / sem, c=colors[i], label=labels[i], lw=3)
    # for ax in axes:
    axes.axhline(0, ls='--', color='k', zorder=0)
    # axes[0].set_ylabel("\u03B2 parameters")
    axes.legend(fontsize=17)
    axes.set_ylabel("\u03B2 parameters (t-values)", fontsize=17)
    axes.set_xlabel("R² threshold", fontsize=17)
    plt.tick_params(axis='y', labelsize=17)
    plt.tick_params(axis='x', labelsize=17)

    plt.tight_layout()

    plt.savefig('/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/corr_and_distance/r2_threshold_choice_impact.svg')
    plt.show()


def full_dual_mechanism_model_linear_exp_comparison(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=True):
    """Run one model with exponential fits and one with linear fits and compares them in terms of loglikelihood"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    del subject_paths[1]
    subject_names = []
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    print(significant_electrodes)
    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])
    print(chan_count)


    ############## Representation analysis ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(similarity_path, allow_pickle=True)
    distance_data = np.load(distance_path, allow_pickle=True)

    # exp function for fitting
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c
    # linear fit
    def linear_func(x, a, b):
        return a * x + b

    logL_models = [[], []]
    AIC_intra_fits = [[], []]
    AIC_inter_fits = [[], []]
    n_params = [3, 2]  # exp, linear
    for f, func in enumerate([exp_func, linear_func]):
        for s, subject in tqdm(enumerate(data["subjects"])):
            if subject != 'Bou_Ni':
                subject_stim_features = stim_features[stim_features['subject'] == subject]
                subject_info = data['info'][s]
                ch_names = subject_info['ch_names']
                indices_to_keep = []
                correct_list = []
                feature_list = []
                intrastim_distance_array = []
                stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
                similarity_matrix = data['similarity_matrices'][s]  # keep all channels
                distance_matrix = distance_data['distance_matrices'][s]  # keep all channels
                first_trials_to_remove = 50
                for i, stim in enumerate(data['stim_names'][s]):
                    if i > first_trials_to_remove:
                        if stim in subject_stim_features['stim_name'].tolist():
                            if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                                    indices_to_keep.append(i)
                                    correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                    if 'rep1' in stim:
                                        rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                                    elif 'rep2' in stim:
                                        rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                    else:
                                        raise ValueError
                                    intrastim_distance_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

                feature_list = np.array(feature_list)
                feature_z = (feature_list - feature_list.mean()) / feature_list.std()
                correct_list = np.array(correct_list)
                intrastim_distance_array = np.array(intrastim_distance_array)
                condition_matrix = distance_matrix[indices_to_keep, :, :]
                interstim_distance_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
                for i in range(condition_matrix.shape[0]):
                    current_corr = condition_matrix[i, :, :]
                    mask = np.ones(condition_matrix.shape[1], dtype=bool)
                    mask[indices_to_keep[i]:] = False  # remove current stim and following trials
                    interstim_distance_array[i, :] = np.mean(current_corr[mask, :], axis=0)

                for channel in range(intrastim_distance_array.shape[1]):
                    if ch_names[channel] in significant_electrodes[subject]:
                        # fit func
                        intra_z = (intrastim_distance_array[:, channel] - np.mean(intrastim_distance_array[:, channel])) / np.std(intrastim_distance_array[:, channel])
                        inter_z = (interstim_distance_array[:, channel] - np.mean(interstim_distance_array[:, channel])) / np.std(interstim_distance_array[:, channel])
                        maxfev = 100000

                        # popt_intra, pcov_intra = curve_fit(exp_func, feature_z, intra_z, p0=[1, 0, 0],
                        #                                    bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
                        #                                    maxfev=maxfev)
                        # popt_inter, pcov_inter = curve_fit(exp_func, feature_z, inter_z, p0 = [1, 0, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)

                        # popt_intra, pcov_intra = curve_fit(linear_func, feature_z, intra_z, maxfev=maxfev)
                        # popt_inter, pcov_inter = curve_fit(linear_func, feature_z, inter_z, maxfev=maxfev)

                        if func == exp_func:
                            popt_intra, pcov_intra = curve_fit(func, feature_z, intra_z, p0 = [1, 0, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)
                            popt_inter, pcov_inter = curve_fit(func, feature_z, inter_z, p0 = [1, 0, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)
                        elif func == linear_func:
                            popt_intra, pcov_intra = curve_fit(func, feature_z, intra_z, maxfev=maxfev)
                            popt_inter, pcov_inter = curve_fit(func, feature_z, inter_z, maxfev=maxfev)
                        else:
                            raise ValueError
                        predicted_intra = func(feature_z, *popt_intra)
                        predicted_inter = func(feature_z, *popt_inter)
                        r2_intra = 1 - np.sum((predicted_intra - intra_z)**2) / np.sum((intra_z - intra_z.mean())**2)
                        r2_inter = 1 - np.sum((predicted_inter - inter_z)**2) / np.sum((inter_z - inter_z.mean())**2)

                        r2_threshold = 0.05  # if either fit reaches the threshold, we keep the channel
                        if (r2_intra > r2_threshold) | (r2_inter > r2_threshold):
                            # Fit logistic regression
                            X = np.zeros((predicted_intra.shape[0], 2))
                            X[:, 0] = predicted_intra
                            X[:, 1] = predicted_inter
                            model = LogisticRegression(penalty=None)
                            model.fit(X, correct_list)

                            # store model logL
                            proba = model.predict_proba(X)
                            logL_models[f].append(logistic_log_likelihood(correct_list, proba[:, 1]))

                            # compute AIC of fits
                            AIC_intra_fits[f].append(2 * n_params[f] - 2 * ols_log_likelihood(intra_z, predicted_intra))
                            AIC_inter_fits[f].append(2 * n_params[f] - 2 * ols_log_likelihood(inter_z, predicted_inter))

    # compare model logL
    print(logL_models)
    logL_exp_model = np.array(logL_models[0])
    logL_linear_model = np.array(logL_models[1])
    print('exp channels:', logL_exp_model.shape[0])
    print('linear channels:', logL_linear_model.shape[0])

    print(np.mean(logL_exp_model))
    print(np.mean(logL_linear_model))

    test_comparison = ttest_ind(logL_exp_model, logL_linear_model, axis=0, equal_var=True, permutations=10000, alternative='greater')
    print("Independent ttest logL exp - linear: ", test_comparison)

    df = pd.DataFrame({  # handles different sizes
        'Value': np.concatenate([logL_exp_model, logL_linear_model]),
        'Model': (['Exp fits'] * len(logL_exp_model)) +
                 (['Linear fits'] * len(logL_linear_model))
    })
    plt.figure(figsize=(5.2, 5))
    sns.violinplot(data=df, x='Model', y='Value', showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5'], inner=None)
    sns.swarmplot(data=df, x='Model', y='Value', color=".25", size=2.5)
    plt.ylabel('Logistic regression log likelihood', fontsize=13)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)

    plt.scatter(0, np.mean(logL_exp_model), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(1, np.mean(logL_linear_model), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.tight_layout()


    # compare curve fit logL
    AIC_exp_intra = np.array(AIC_intra_fits[0])
    AIC_linear_intra = np.array(AIC_intra_fits[1])
    AIC_exp_inter = np.array(AIC_inter_fits[0])
    AIC_linear_inter = np.array(AIC_inter_fits[1])

    test_comparison = ttest_ind(AIC_exp_intra, AIC_linear_intra, axis=0, equal_var=True, permutations=10000, alternative='less')
    print("Independent AIC_exp_intra - AIC_linear_intra: ", test_comparison)

    df = pd.DataFrame({  # handles different sizes
        'Value': np.concatenate([AIC_exp_intra, AIC_linear_intra]),
        'Model': (['Exp fits'] * len(AIC_exp_intra)) +
                 (['Linear fits'] * len(AIC_linear_intra))
    })
    plt.figure(figsize=(5.2, 5))
    sns.violinplot(data=df, x='Model', y='Value', showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5'],
                   inner=None)
    sns.swarmplot(data=df, x='Model', y='Value', color=".25", size=2.5)
    plt.ylabel('Curve fit AIC', fontsize=13)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)

    plt.scatter(0, np.mean(AIC_exp_intra), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(1, np.mean(AIC_linear_intra), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.tight_layout()

    test_comparison = ttest_ind(AIC_exp_inter, AIC_linear_inter, axis=0, equal_var=True, permutations=10000, alternative='less')
    print("Independent AIC_exp_inter - AIC_linear_inter: ", test_comparison)

    df = pd.DataFrame({  # handles different sizes
        'Value': np.concatenate([AIC_exp_inter, AIC_linear_inter]),
        'Model': (['Exp fits'] * len(AIC_exp_inter)) +
                 (['Linear fits'] * len(AIC_linear_inter))
    })
    plt.figure(figsize=(5.2, 5))
    sns.violinplot(data=df, x='Model', y='Value', showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5'],
                   inner=None)
    sns.swarmplot(data=df, x='Model', y='Value', color=".25", size=2.5)
    plt.ylabel('Curve fit AIC', fontsize=13)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)

    plt.scatter(0, np.mean(AIC_exp_inter), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(1, np.mean(AIC_linear_inter), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.tight_layout()



    plt.show()



def full_dual_mechanism_model_separated_channels(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=True, channel_selection='weight'):
    """For each channel's similarity and separability, fit the exponential relationships with surprise (f_i). Then,
    select similarity channels and separability channels and aggregate (or weighted average by R2 values). Finally,
    run a logistic regression from surprise to behavior (correct response) with f_i as independent variables"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    del subject_paths[1]
    subject_names = []
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    print(significant_electrodes)
    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])
    print(chan_count)


    ############## Representation analysis ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(similarity_path, allow_pickle=True)
    distance_data = np.load(distance_path, allow_pickle=True)

    # exp function for fitting
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c
    # def exp_func(x, b, c):
    #     return np.exp(b * x) + c  # without param a (creates degeneracy with b)
    # quadratic fit
    def quadr_func(x, a, b, c):
        return a * x**2 + b * x + c
    # linear fit
    def linear_func(x, a, b):
        return a * x + b

    func = exp_func

    subject_id = []
    coefs_intra = []
    coefs_inter = []
    curve_param_intra = []  # parameters of the exponental fit
    curve_param_inter = []
    correlation_intra_inter = []  # check correlation between the two predictors
    predicted_intra_all_subjects = []
    predicted_inter_all_subjects = []
    correct_all_subjects = []
    feature_all_subjects = []
    for s, subject in tqdm(enumerate(data["subjects"])):
        if subject != 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            intrastim_distance_array = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['similarity_matrices'][s]  # keep all channels
            distance_matrix = distance_data['distance_matrices'][s]  # keep all channels
            first_trials_to_remove = 50
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:
                    if stim in subject_stim_features['stim_name'].tolist():
                        # if 'rep2' in stim:
                        if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                            if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                                indices_to_keep.append(i)
                                correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                if 'rep1' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                                elif 'rep2' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                else:
                                    raise ValueError
                                intrastim_distance_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

            feature_list = np.array(feature_list)
            feature_z = (feature_list - feature_list.mean()) / feature_list.std()
            intrastim_distance_array = np.array(intrastim_distance_array)
            condition_matrix = distance_matrix[indices_to_keep, :, :]
            interstim_distance_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
            for i in range(condition_matrix.shape[0]):
                current_corr = condition_matrix[i, :, :]
                mask = np.ones(condition_matrix.shape[1], dtype=bool)
                mask[indices_to_keep[i]:] = False  # remove current stim and following trials
                interstim_distance_array[i, :] = np.mean(current_corr[mask, :], axis=0)

            # reduce huge outlier in subject 3
            if s == 3:
                interstim_distance_array[3, :] = np.mean(interstim_distance_array)

            predicted_intra_subject = []
            predicted_inter_subject = []
            r2_intra = []
            r2_inter = []
            for channel in range(intrastim_distance_array.shape[1]):
                if ch_names[channel] in significant_electrodes[subject]:
                    # fit exponentials
                    intra_z = (intrastim_distance_array[:, channel] - np.mean(intrastim_distance_array[:, channel])) / np.std(intrastim_distance_array[:, channel])
                    inter_z = (interstim_distance_array[:, channel] - np.mean(interstim_distance_array[:, channel])) / np.std(interstim_distance_array[:, channel])
                    maxfev = 100000
                    if func == exp_func:
                        popt_intra, pcov_intra = curve_fit(func, feature_z, intra_z, p0 = [1, -1, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, 0, np.inf]), maxfev=maxfev)
                        popt_inter, pcov_inter = curve_fit(func, feature_z, inter_z, p0 = [1, 1, 0], bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)
                        # popt_intra, pcov_intra = curve_fit(func, feature_z, intra_z, p0=[-1, 0], bounds=([-np.inf, -np.inf], [0, np.inf]), maxfev=maxfev)
                        # popt_inter, pcov_inter = curve_fit(func, feature_z, inter_z, p0=[1, 0], bounds=([0, -np.inf], [np.inf, np.inf]), maxfev=maxfev)
                    else:
                        deg = 1 if func == linear_func else 2
                        popt_intra = np.polyfit(feature_z, intra_z, deg)
                        popt_inter = np.polyfit(feature_z, inter_z, deg)
                    predicted_intra = func(feature_z, *popt_intra)
                    predicted_inter = func(feature_z, *popt_inter)

                    # remove channels that show an inverted relationship
                    param_to_check = 0
                    if func == exp_func:
                        param_to_check = 1
                    if popt_intra[param_to_check] < 0:
                        r2_intra.append(1 - np.sum((predicted_intra - intra_z)**2) / np.sum((intra_z - intra_z.mean())**2))
                        predicted_intra_subject.append(predicted_intra)
                        # predicted_intra_subject.append(intra_z)
                    if popt_inter[param_to_check] > 0:
                        r2_inter.append(1 - np.sum((predicted_inter - inter_z)**2) / np.sum((inter_z - inter_z.mean())**2))
                        predicted_inter_subject.append(predicted_inter)
                        # predicted_inter_subject.append(inter_z)

            r2_intra = np.array(r2_intra)
            r2_inter = np.array(r2_inter)
            predicted_intra_subject = np.array(predicted_intra_subject)
            predicted_inter_subject = np.array(predicted_inter_subject)

            if channel_selection == 'weight':
                # weighted average across channels by R2
                r2_intra = np.maximum(0, r2_intra)  # ensure r2 is positive
                r2_inter = np.maximum(0, r2_inter)
                weight_intra = r2_intra / r2_intra.sum()
                weight_inter = r2_inter / r2_inter.sum()
                predicted_intra_subject = np.sum(weight_intra[:, np.newaxis] * predicted_intra_subject, axis=0)
                predicted_inter_subject = np.sum(weight_inter[:, np.newaxis] * predicted_inter_subject, axis=0)

            elif channel_selection == 'best':
                n_best = 10  # keep n_best channels
                predicted_intra_subject = np.mean(predicted_intra_subject[np.argsort(r2_intra)[::-1], :][:n_best, :], axis=0)
                predicted_inter_subject = np.mean(predicted_inter_subject[np.argsort(r2_inter)[::-1], :][:n_best, :], axis=0)

            elif channel_selection == 'r2_threshold':
                r2_threshold = 0.05  # keep channels that exceed threshold
                n_best = 10
                print(np.sum(r2_intra >= r2_threshold), np.sum(r2_inter >= r2_threshold))
                if np.sum(r2_intra >= r2_threshold) == 0:  # use r2 weighting of all channels is none exceeds the threshold
                    predicted_intra_subject = np.mean(predicted_intra_subject[np.argsort(r2_intra)[::-1], :][:n_best, :], axis=0)
                else:
                    predicted_intra_subject = np.mean(predicted_intra_subject[r2_intra >= r2_threshold, :], axis=0)
                if np.sum(r2_inter >= r2_threshold) == 0:
                    predicted_inter_subject = np.mean(predicted_inter_subject[np.argsort(r2_inter)[::-1], :][:n_best, :], axis=0)
                else:
                    predicted_inter_subject = np.mean(predicted_inter_subject[r2_inter >= r2_threshold, :], axis=0)

            else:
                predicted_intra_subject = np.mean(predicted_intra_subject, axis=0)
                predicted_inter_subject = np.mean(predicted_inter_subject, axis=0)

            predicted_intra_all_subjects += predicted_intra_subject.tolist()
            predicted_inter_all_subjects += predicted_inter_subject.tolist()
            subject_id += [s for _ in range(predicted_intra_subject.shape[0])]
            correct_all_subjects += correct_list
            feature_all_subjects += feature_z.tolist()

            # Fit logistic regression per subject
            X = np.zeros((predicted_intra_subject.shape[0], 2))
            X[:, 0] = predicted_intra_subject
            X[:, 1] = predicted_inter_subject
            model = LogisticRegression(penalty=None)
            model.fit(X, correct_list)

            coefs_intra.append(model.coef_[0, 0])
            coefs_inter.append(model.coef_[0, 1])

            # store model likelihood
            # TODO

    # stat test
    pval_intra = ttest_1samp(coefs_intra, 0, alternative='greater').pvalue
    pval_inter = ttest_1samp(coefs_inter, 0, alternative='greater').pvalue
    print("pval_intra: ", pval_intra)
    print("pval_inter: ", pval_inter)

    # boxplots
    df = pd.DataFrame({'intra \u03C1': coefs_intra, 'inter d': coefs_inter})
    plt.figure(figsize=(5.2, 5))
    sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5'], orient='v', inner=None)
    sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
    plt.gca().axhline(0, ls='--', color='k')
    plt.ylabel('\u03B2 parameters', fontsize=13)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)

    plt.scatter(0, np.mean(coefs_intra), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(1, np.mean(coefs_inter), color='w', edgecolor='k', lw=2, s=75, zorder=10)

    # add stars
    for i, p in enumerate([pval_intra, pval_inter]):
        if p < 0.05:
            # Get the position to place the star
            x = i  # The x position corresponds to the box index
            y = df.iloc[:, i].max() + 0.15  # Slightly above the max value of the box
            plt.text(x, y, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
    plt.tight_layout()

    # save data for mixed model
    predicted_intra_all_subjects = np.array(predicted_intra_all_subjects)
    predicted_inter_all_subjects = np.array(predicted_inter_all_subjects)
    subject_id = np.array(subject_id)
    correct_all_subjects = np.array(correct_all_subjects)
    feature_all_subjects = np.array(feature_all_subjects)

    df = pd.DataFrame({
        "predicted_similarity": predicted_intra_all_subjects,
        "predicted_separability": predicted_inter_all_subjects,
        "correct": correct_all_subjects,
        "subject": subject_id,
    })

    # save for R analysis
    # df.to_csv("/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/corr_and_distance/dual_mechanism_data.csv", index=False)

    # correlation
    plt.figure()
    plt.scatter(predicted_intra_all_subjects, predicted_inter_all_subjects)
    print(np.corrcoef(predicted_intra_all_subjects, predicted_inter_all_subjects)[0, 1])

    # neural patterns ~ surprise
    fig, axs = plt.subplots(1, 2)
    axs[0].scatter(feature_all_subjects, predicted_intra_all_subjects)
    axs[0].set_xlabel('surprise')
    axs[0].set_ylabel('similarity')
    axs[1].scatter(feature_all_subjects, predicted_inter_all_subjects)
    axs[1].set_xlabel('surprise')
    axs[1].set_ylabel('separability')


    plt.show()


def only_separability_dual_mechanism(rms_path, behavior_path, distance_path, condition, regions, first_parcel_label=True, channel_selection='weight'):
    """Test of separability in rep 1"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    del subject_paths[1]
    subject_names = []
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    print(significant_electrodes)
    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])
    print(chan_count)


    ############## Representation analysis ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(similarity_path, allow_pickle=True)
    distance_data = np.load(distance_path, allow_pickle=True)

    # exp function for fitting
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c
    # def exp_func(x, b, c):
    #     return np.exp(b * x) + c  # without param a (creates degeneracy with b)
    # quadratic fit
    def quadr_func(x, a, b, c):
        return a * x**2 + b * x + c
    # linear fit
    def linear_func(x, a, b):
        return a * x + b

    func = exp_func

    subject_id = []
    coefs_inter = []
    coefs_inter_channels = []
    feature_params = []
    predicted_inter_all_subjects = []
    correct_all_subjects = []
    feature_all_subjects = []
    for s, subject in tqdm(enumerate(data["subjects"])):
        if subject != 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            intrastim_distance_array = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            distance_matrix = distance_data['distance_matrices'][s]  # keep all channels
            first_trials_to_remove = 50
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:
                    if stim in subject_stim_features['stim_name'].tolist():
                        # if ('rep1' in stim) | ('rep2' in stim):
                        if ('rep2' in stim):
                        # if ('rep1' in stim):
                            if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                indices_to_keep.append(i)
                                correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])

            feature_list = np.array(feature_list)
            feature_z = (feature_list - feature_list.mean()) / feature_list.std()
            condition_matrix = distance_matrix[indices_to_keep, :, :]
            interstim_distance_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
            for i in range(condition_matrix.shape[0]):
                current_corr = condition_matrix[i, :, :]
                mask = np.ones(condition_matrix.shape[1], dtype=bool)
                mask[indices_to_keep[i]:] = False  # remove current stim and following trials
                interstim_distance_array[i, :] = np.mean(current_corr[mask, :], axis=0)

            # reduce huge outlier in subject 3
            if s == 3:
                print(np.argmax(np.mean(interstim_distance_array, axis=1)))
                # interstim_distance_array[3, :] = np.mean(interstim_distance_array, axis=0)
                # interstim_distance_array[6, :] = np.mean(interstim_distance_array, axis=0)

            predicted_inter_subject = []
            r2_inter = []
            for channel in range(interstim_distance_array.shape[1]):
                if ch_names[channel] in significant_electrodes[subject]:
                    # fit exponentials
                    inter_z = (interstim_distance_array[:, channel] - np.mean(interstim_distance_array[:, channel])) / np.std(interstim_distance_array[:, channel])
                    maxfev = 100000
                    if func == exp_func:
                        popt_inter, pcov_inter = curve_fit(func, feature_z, inter_z, p0 = [1, 0, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)
                        # popt_inter, pcov_inter = curve_fit(func, feature_z, inter_z, p0=[1, 0], bounds=([0, -np.inf], [np.inf, np.inf]), maxfev=maxfev)
                    else:
                        deg = 1 if func == linear_func else 2
                        popt_inter = np.polyfit(feature_z, inter_z, deg)
                    predicted_inter = func(feature_z, *popt_inter)

                    # remove channels that show an inverted relationship
                    param_to_check = 0
                    if func == exp_func:
                        param_to_check = 1

                    # if popt_inter[param_to_check] > 0:
                    r2_inter.append(1 - np.sum((predicted_inter - inter_z)**2) / np.sum((inter_z - inter_z.mean())**2))
                    predicted_inter_subject.append(predicted_inter)
                    # predicted_inter_subject.append(inter_z)

                    # Fit logistic regression per channel
                    X = np.zeros((inter_z.shape[0], 1))
                    X[:, 0] = predicted_inter
                    # X[:, 0] = inter_z
                    model = LogisticRegression(penalty=None)
                    model.fit(X, correct_list)
                    coefs_inter_channels.append(model.coef_[0, 0])

                    feature_params.append(popt_inter[param_to_check])

            r2_inter = np.array(r2_inter)
            predicted_inter_subject = np.array(predicted_inter_subject)

            if channel_selection == 'weight':
                # weighted average across channels by R2
                r2_inter = np.maximum(0, r2_inter)
                weight_inter = r2_inter / r2_inter.sum()
                predicted_inter_subject = np.sum(weight_inter[:, np.newaxis] * predicted_inter_subject, axis=0)

            elif channel_selection == 'best':
                n_best = 10  # keep n_best channels
                predicted_inter_subject = np.mean(predicted_inter_subject[np.argsort(r2_inter)[::-1], :][:n_best, :], axis=0)

            elif channel_selection == 'r2_threshold':
                r2_threshold = 0.05  # keep channels that exceed threshold
                n_best = 10
                if np.sum(r2_inter >= r2_threshold) == 0:
                    predicted_inter_subject = np.mean(predicted_inter_subject[np.argsort(r2_inter)[::-1], :][:n_best, :], axis=0)
                else:
                    predicted_inter_subject = np.mean(predicted_inter_subject[r2_inter >= r2_threshold, :], axis=0)

            else:
                predicted_inter_subject = np.mean(predicted_inter_subject, axis=0)

            if not np.isnan(predicted_inter_subject).all():  # check that at least one channel was selected
                predicted_inter_all_subjects += predicted_inter_subject.tolist()
                subject_id += [s for _ in range(predicted_inter_subject.shape[0])]
                correct_all_subjects += correct_list
                feature_all_subjects += feature_z.tolist()

                # Fit logistic regression per subject
                X = np.zeros((predicted_inter_subject.shape[0], 1))
                X[:, 0] = predicted_inter_subject
                model = LogisticRegression(penalty=None)
                model.fit(X, correct_list)

                coefs_inter.append(model.coef_[0, 0])

                # store model likelihood
                # TODO

    # stat test
    test_inter = ttest_1samp(coefs_inter, 0, alternative='greater')
    pval_inter = test_inter.pvalue
    print("test_inter: ", test_inter)

    # boxplots
    df = pd.DataFrame({'inter d': coefs_inter})
    plt.figure(figsize=(5.2, 5))
    sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A'], orient='v', inner=None)
    sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
    plt.gca().axhline(0, ls='--', color='k')
    plt.ylabel('\u03B2 parameters', fontsize=13)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)

    plt.scatter(0, np.mean(coefs_inter), color='w', edgecolor='k', lw=2, s=75, zorder=10)

    # add stars
    for i, p in enumerate([pval_inter]):
        if p < 0.05:
            # Get the position to place the star
            x = i  # The x position corresponds to the box index
            y = df.iloc[:, i].max() + 0.15  # Slightly above the max value of the box
            plt.text(x, y, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
    plt.tight_layout()

    # save data for mixed model
    predicted_inter_all_subjects = np.array(predicted_inter_all_subjects)
    subject_id = np.array(subject_id)
    correct_all_subjects = np.array(correct_all_subjects)
    feature_all_subjects = np.array(feature_all_subjects)

    # neural patterns ~ surprise
    plt.figure()
    plt.scatter(feature_all_subjects, predicted_inter_all_subjects, c=subject_id, cmap='tab10')
    plt.xlabel('surprise')
    plt.ylabel('separability')

    # Channel-level
    coefs_inter_channels = np.array(coefs_inter_channels)
    feature_params = np.array(feature_params)
    test_inter_channels = ttest_1samp(coefs_inter_channels, 0, alternative='greater')
    pval_inter_channels = test_inter_channels.pvalue
    print("test_inter_channels: ", test_inter_channels)
    test_feature_params = ttest_1samp(feature_params, 0, alternative='greater')
    pval_feature_params = test_feature_params.pvalue
    print("test_feature_params: ", test_feature_params)

    df = pd.DataFrame({'inter d': coefs_inter_channels, 'inter d ~ surprise': feature_params})
    plt.figure(figsize=(5.2, 5))
    sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5'], orient='v', inner=None)
    sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
    plt.gca().axhline(0, ls='--', color='k')
    plt.ylabel('\u03B2 parameters', fontsize=13)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)

    plt.scatter(0, np.mean(coefs_inter_channels), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(1, np.mean(feature_params), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    for i, p in enumerate([pval_inter_channels, pval_feature_params]):
        if p < 0.05:
            x = i  # The x position corresponds to the box index
            y = df.iloc[:, i].max() + 0.15  # Slightly above the max value of the box
            plt.text(x, y, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
    plt.tight_layout()


    plt.show()


def regress_correct_on_suprise_and_neural_patterns(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=True):
    """Test the direct effect of surprise on behavior, accounting for similarity and separability"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    del subject_paths[1]
    subject_names = []
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    print(significant_electrodes)
    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])
    print(chan_count)

    ############## Representation analysis ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(similarity_path, allow_pickle=True)
    distance_data = np.load(distance_path, allow_pickle=True)

    # exp function for fitting
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    coefs_surprise_squared = []
    coefs_surprise = []
    coefs_surprise_squared_control = []
    for s, subject in tqdm(enumerate(data["subjects"])):
        if subject != 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            intrastim_distance_array = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['similarity_matrices'][s]  # keep all channels
            distance_matrix = distance_data['distance_matrices'][s]  # keep all channels
            first_trials_to_remove = 50
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:
                    if stim in subject_stim_features['stim_name'].tolist():
                        if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                            if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                                indices_to_keep.append(i)
                                correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                if 'rep1' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                                elif 'rep2' in stim:
                                    rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                else:
                                    raise ValueError
                                intrastim_distance_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

            feature_list = np.array(feature_list)
            feature_z = (feature_list - feature_list.mean()) / feature_list.std()
            intrastim_distance_array = np.array(intrastim_distance_array)
            condition_matrix = distance_matrix[indices_to_keep, :, :]
            interstim_distance_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
            for i in range(condition_matrix.shape[0]):
                current_corr = condition_matrix[i, :, :]
                mask = np.ones(condition_matrix.shape[1], dtype=bool)
                mask[indices_to_keep[i]:] = False  # remove current stim and following trials
                interstim_distance_array[i, :] = np.mean(current_corr[mask, :], axis=0)

            for channel in range(intrastim_distance_array.shape[1]):
                if ch_names[channel] in significant_electrodes[subject]:
                    # fit exponentials
                    intra_z = (intrastim_distance_array[:, channel] - np.mean(intrastim_distance_array[:, channel])) / np.std(intrastim_distance_array[:, channel])
                    inter_z = (interstim_distance_array[:, channel] - np.mean(interstim_distance_array[:, channel])) / np.std(interstim_distance_array[:, channel])
                    maxfev = 100000
                    popt_intra, pcov_intra = curve_fit(exp_func, feature_z, intra_z, p0 = [1, 0, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)
                    popt_inter, pcov_inter = curve_fit(exp_func, feature_z, inter_z, p0 = [1, 0, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)
                    predicted_intra = exp_func(feature_z, *popt_intra)
                    predicted_inter = exp_func(feature_z, *popt_inter)
                    r2_intra = 1 - np.sum((predicted_intra - intra_z)**2) / np.sum((intra_z - intra_z.mean())**2)
                    r2_inter = 1 - np.sum((predicted_inter - inter_z)**2) / np.sum((inter_z - inter_z.mean())**2)

                    r2_threshold = 0.05  # if either fit reaches the threshold, we keep the channel
                    # if (r2_intra > r2_threshold) | (r2_inter > r2_threshold):
                    if ((r2_intra > r2_threshold) & (popt_intra[1] < 0)) | ((r2_inter > r2_threshold) & (popt_inter[1] > 0)):
                        # Fit logistic regression quadratic
                        X = np.zeros((predicted_intra.shape[0], 4))
                        X[:, 0] = feature_z
                        X[:, 1] = feature_z**2
                        # X[:, 2] = predicted_intra
                        # X[:, 3] = predicted_inter
                        X[:, 2] = intra_z
                        X[:, 3] = inter_z
                        model = LogisticRegression(penalty=None)
                        model.fit(X, correct_list)

                        coefs_surprise_squared.append(model.coef_[0, 1])

                        # Fit logistic regression linear
                        X = np.zeros((predicted_intra.shape[0], 3))
                        X[:, 0] = feature_z
                        # X[:, 1] = predicted_intra
                        # X[:, 2] = predicted_inter
                        X[:, 1] = intra_z
                        X[:, 2] = inter_z
                        model = LogisticRegression(penalty=None)
                        model.fit(X, correct_list)

                        coefs_surprise.append(model.coef_[0, 0])

                        # Fit logistic regression control model (only surprise and squared surprise)
                        X = np.zeros((predicted_intra.shape[0], 2))
                        X[:, 0] = feature_z
                        X[:, 1] = feature_z ** 2
                        model = LogisticRegression(penalty=None)
                        model.fit(X, correct_list)

                        coefs_surprise_squared_control.append(model.coef_[0, 1])

    print('n_channels:', len(coefs_surprise_squared))

    # stat test
    test_surprise_squared = ttest_1samp(coefs_surprise_squared, 0, alternative='greater')
    test_surprise = ttest_1samp(coefs_surprise, 0, alternative='greater')
    test_surprise_squared_control = ttest_1samp(coefs_surprise_squared_control, 0, alternative='greater')
    pval_surprise_squared = test_surprise_squared.pvalue
    pval_surprise = test_surprise.pvalue
    pval_surprise_squared_control = test_surprise_squared_control.pvalue
    print("test_surprise_squared: ", test_surprise_squared)
    print("test_surprise: ", test_surprise)
    print("test_surprise_squared_control: ", test_surprise_squared_control)

    # boxplots
    df = pd.DataFrame({'surprise²': coefs_surprise_squared, 'surprise': coefs_surprise, 'surprise² control': coefs_surprise_squared_control})
    plt.figure(figsize=(5.2, 5))
    sns.violinplot(data=df, showmeans=True, showfliers=False, palette=['#FF928A', '#978EF5', '#fdb462'], orient='v', inner=None)
    sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
    plt.gca().axhline(0, ls='--', color='k')
    plt.ylabel('\u03B2 parameters', fontsize=13)
    plt.tick_params(axis='y', labelsize=13)
    plt.tick_params(axis='x', labelsize=13)

    plt.scatter(0, np.mean(coefs_surprise_squared), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(1, np.mean(coefs_surprise), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(2, np.mean(coefs_surprise_squared_control), color='w', edgecolor='k', lw=2, s=75, zorder=10)

    # add stars
    for i, p in enumerate([pval_surprise_squared, pval_surprise, pval_surprise_squared_control]):
        if p < 0.05:
            x = i  # The x position corresponds to the box index
            y = df.iloc[:, i].max() + 0.15  # Slightly above the max value of the box
            plt.text(x, y, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
    plt.tight_layout()

    # plt.savefig('/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/corr_and_distance/dual_mechanism_model.svg')

    plt.show()


def linear_vs_exp_by_region(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=True):
    """Fit similarity ~ surprise and separability ~ surprise with linear and exponential functions per channel.
    Not threshold on R², every channel is tested.
    Then compute the proportion of best function per region (in terms of AIC or logL) and save as csv"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    del subject_paths[1]
    subject_names = []
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject
    significant_electrodes_region_names = []

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        significant_electrodes[subject_name] = []
        for c in range(rms.shape[1]):
            for region in regions:
                if any(ch_parcel in region for ch_parcel in parcellations[c]):
                    significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
                    significant_electrodes_region_names.append(parcellations[c][0])

    print(significant_electrodes)
    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])
    print(chan_count)


    ############## Representation analysis ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(similarity_path, allow_pickle=True)
    distance_data = np.load(distance_path, allow_pickle=True)

    # exp function for fitting
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c
    # linear fit
    def linear_func(x, a, b):
        return a * x + b

    logL_models = [[], []]
    AIC_intra_fits = [[], []]
    AIC_inter_fits = [[], []]
    logL_intra_fits = [[], []]
    logL_inter_fits = [[], []]
    n_params = [3, 2]  # exp, linear
    for f, func in enumerate([exp_func, linear_func]):
        for s, subject in tqdm(enumerate(data["subjects"])):
            if subject != 'Bou_Ni':
                subject_stim_features = stim_features[stim_features['subject'] == subject]
                subject_info = data['info'][s]
                ch_names = subject_info['ch_names']
                indices_to_keep = []
                correct_list = []
                feature_list = []
                intrastim_distance_array = []
                stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
                similarity_matrix = data['similarity_matrices'][s]  # keep all channels
                distance_matrix = distance_data['distance_matrices'][s]  # keep all channels
                first_trials_to_remove = 50
                for i, stim in enumerate(data['stim_names'][s]):
                    if i > first_trials_to_remove:
                        if stim in subject_stim_features['stim_name'].tolist():
                            if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                if (stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb) & (stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb):  # check that both reps are in the data
                                    indices_to_keep.append(i)
                                    correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                    if 'rep1' in stim:
                                        rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2'))
                                    elif 'rep2' in stim:
                                        rep_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                    else:
                                        raise ValueError
                                    intrastim_distance_array.append(similarity_matrix[i, rep_index, :])  # correlation of 2 reps

                feature_list = np.array(feature_list)
                feature_z = (feature_list - feature_list.mean()) / feature_list.std()
                correct_list = np.array(correct_list)
                # feature_z = (correct_list - correct_list.mean()) / correct_list.std()  # TODO: remove (test of correct)
                intrastim_distance_array = np.array(intrastim_distance_array)
                condition_matrix = distance_matrix[indices_to_keep, :, :]
                interstim_distance_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
                for i in range(condition_matrix.shape[0]):
                    current_corr = condition_matrix[i, :, :]
                    mask = np.ones(condition_matrix.shape[1], dtype=bool)
                    mask[indices_to_keep[i]:] = False  # remove current stim and following trials
                    interstim_distance_array[i, :] = np.mean(current_corr[mask, :], axis=0)

                for channel in range(intrastim_distance_array.shape[1]):
                    if ch_names[channel] in significant_electrodes[subject]:
                        # fit func
                        intra_z = (intrastim_distance_array[:, channel] - np.mean(intrastim_distance_array[:, channel])) / np.std(intrastim_distance_array[:, channel])
                        inter_z = (interstim_distance_array[:, channel] - np.mean(interstim_distance_array[:, channel])) / np.std(interstim_distance_array[:, channel])
                        maxfev = 100000

                        if func == exp_func:
                            popt_intra, pcov_intra = curve_fit(func, feature_z, intra_z, p0 = [1, 0, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)
                            popt_inter, pcov_inter = curve_fit(func, feature_z, inter_z, p0 = [1, 0, 0], bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]), maxfev=maxfev)
                        elif func == linear_func:
                            popt_intra, pcov_intra = curve_fit(func, feature_z, intra_z, maxfev=maxfev)
                            popt_inter, pcov_inter = curve_fit(func, feature_z, inter_z, maxfev=maxfev)
                        else:
                            raise ValueError
                        predicted_intra = func(feature_z, *popt_intra)
                        predicted_inter = func(feature_z, *popt_inter)
                        r2_intra = 1 - np.sum((predicted_intra - intra_z)**2) / np.sum((intra_z - intra_z.mean())**2)
                        r2_inter = 1 - np.sum((predicted_inter - inter_z)**2) / np.sum((inter_z - inter_z.mean())**2)

                        # compute AIC of fits
                        AIC_intra_fits[f].append(2 * n_params[f] - 2 * ols_log_likelihood(intra_z, predicted_intra))
                        AIC_inter_fits[f].append(2 * n_params[f] - 2 * ols_log_likelihood(inter_z, predicted_inter))
                        logL_intra_fits[f].append(ols_log_likelihood(intra_z, predicted_intra))
                        logL_inter_fits[f].append(ols_log_likelihood(inter_z, predicted_inter))

    def normalize_region(r):
        # remove hemisphere prefix for cortical labels
        if r.startswith("ctx-lh-") or r.startswith("ctx-rh-"):
            return r.split("-", 2)[-1]

        # remove Left-/Right- for subcortical labels
        if r.startswith("Left-") or r.startswith("Right-"):
            return r.split("-", 1)[-1]

        return r

    print(significant_electrodes_region_names)
    AIC_intra_fits = np.array([AIC_intra_fits[0][i] - AIC_intra_fits[1][i] for i in range(len(AIC_intra_fits[0]))])  # negative when exp is better
    AIC_inter_fits = np.array([AIC_inter_fits[0][i] - AIC_inter_fits[1][i] for i in range(len(AIC_inter_fits[0]))])
    logL_inter_fits = np.array([logL_inter_fits[1][i] - logL_inter_fits[0][i] for i in range(len(logL_inter_fits[0]))])  # opposite, negative when exp is better
    logL_intra_fits = np.array([logL_intra_fits[1][i] - logL_intra_fits[0][i] for i in range(len(logL_intra_fits[0]))])

    df = pd.DataFrame({
        "region": significant_electrodes_region_names,
    })
    # df["region_norm"] = df["region"].apply(normalize_region)  # merge both hemispheres
    df["region_norm"] = df["region"]  # individual hemispheres

    value_names = ['aic_similarity', 'aic_separability', 'logL_similarity', 'logL_separability']
    # --- add value columns
    for name, vals in zip(value_names, [AIC_intra_fits, AIC_inter_fits, logL_intra_fits, logL_inter_fits]):
        df[name] = vals

    # --- build final region-level dataframe
    proportion_df = df[["region_norm"]].drop_duplicates().set_index("region_norm")

    for name in value_names:
        proportion_df[name] = (
            df.assign(neg=df[name] < 0)
            .groupby("region_norm")["neg"]
            .mean()
        )

    proportion_df = proportion_df.reset_index()
    pd.set_option("display.max_columns", None)
    print(proportion_df)

    proportion_df.to_csv("/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/corr_and_distance/linear_vs_exp.csv")


def population_overlaps_analysis(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=True):
    """Takes all electrodes from a big region (ex: all temporal lobe) and selected the subset of electrodes that show a
    significant RMS contrast (in either novelty detection or retrieval), then run intra and inter stimulus correlation
    analysis, linking them to correct response and surprise"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    del subject_paths[1]
    subject_names = []
    print(subject_paths)

    significant_electrodes = {'all_electrodes': [], 'novelty': [], 'retrieval': [], 'intra_correct': [], 'intra_surprise': [], 'inter_correct': [], 'inter_surprise': []}  # dict of signif electrodes by subject
    electrodes_regions_dict = {}
    region_electrodes = []  # labels electrodes of the chosen region from rms data, to be used with distance data

    all_regions = False
    if regions is None:
        all_regions = True

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        # if regions is None, we keep all regions
        if all_regions:
            regions = np.unique(parcellations)
            to_remove = ['Unknown', 'Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter']
            regions = [reg for reg in regions if reg not in to_remove]

        conditions = ['novelty', 'retrieval']
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
        # for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty]]):
        # for cond, correct_incorrect in enumerate([[correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            region_electrodes.append(subject_name + '_' + rms_obj['ch_names'][c])
                            electrodes_regions_dict[subject_name + '_' + rms_obj['ch_names'][c]] = region
                            if subject_name + '_' + rms_obj['ch_names'][c] not in significant_electrodes['all_electrodes']:  # avoid replicate in all electrodes
                                significant_electrodes['all_electrodes'].append(subject_name + '_' + rms_obj['ch_names'][c])

                            pval = ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False,
                                                           permutations=10000, random_state=None,
                                                           alternative='greater').pvalue  # testing of individual electrodes
                            if pval < 0.05:
                                significant_electrodes[conditions[cond]].append(subject_name + '_' + rms_obj['ch_names'][c])

    print(significant_electrodes)
    region_electrodes = np.unique(region_electrodes)

    print(electrodes_regions_dict)


    ############## Intrastim correlation -> correct ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    # data = np.load(data_path, allow_pickle=True)
    data = np.load(similarity_path, allow_pickle=True)

    for s, subject in tqdm(enumerate(data["subjects"])):
        if subject != 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            correlations_array = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['similarity_matrices'][s]  # keep all channels
            # similarity_matrix = data['distance_matrices'][s]  # keep all channels
            # similarity_matrix = 1 / similarity_matrix
            first_trials_to_remove = 50  # very sensitive to the first trials, then stabilizes (mostly for interstim distance, which makes sense)
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:  # remove first trials
                    if stim in subject_stim_features['stim_name'].tolist():
                        if 'rep2' in stim:
                            if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                indices_to_keep.append(i)
                                if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                    rep1_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                    correlations_array.append(similarity_matrix[i, rep1_index, :])  # correlation of 2 reps
                                    correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])

            correlations_array = np.array(correlations_array)

            # pvalues = np.zeros(correlations_array.shape[1])
            # p_ttest = np.zeros(correlations_array.shape[1])
            for channel in range(correlations_array.shape[1]):
                if subject + '_' + ch_names[channel] in region_electrodes:
                    Y = correlations_array[:, channel]
                    X = np.zeros((Y.shape[0], 1))
                    # X = np.zeros((Y.shape[0], 2))
                    X[:, 0] = correct_list
                    # X[:, 1] = np.array(feature_list)**2
                    X = sm.add_constant(X)

                    # independent t-test
                    correct_corr = Y[X[:, 1] == 1]  # axis 0 is the intercept
                    incorrect_corr = Y[X[:, 1] == 0]
                    p_ttest = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=10000,
                                                 random_state=None,
                                                 alternative='greater').pvalue  # testing of individual electrodes
                                                 # alternative='less').pvalue  # testing of individual electrodes
                    if p_ttest < 0.05:
                        significant_electrodes['intra_correct'].append(subject + '_' + ch_names[channel])

                    # ols = sm.OLS(Y, X).fit()
                    # param = ols.params[1]
                    # p_ttest[channel] = ols.pvalues[1]

                    X = np.zeros((Y.shape[0], 1))
                    X[:, 0] = feature_list
                    X = sm.add_constant(X)
                    ols = sm.OLS(Y, X).fit()
                    param = ols.params[1]
                    p_ttest = ols.pvalues[1]

                    if (p_ttest < 0.1) & (param < 0):  # 0.1 because it is a one-tailed test
                        significant_electrodes['intra_surprise'].append(subject + '_' + ch_names[channel])

    print(significant_electrodes)


    ############## Interstim correlation -> incorrect ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    # data = np.load(data_path, allow_pickle=True)
    data = np.load(distance_path, allow_pickle=True)

    for s, subject in tqdm(enumerate(data["subjects"])):

        if not subject == 'Bou_Ni':

            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            # similarity_matrix = data['similarity_matrices'][s]  # keep all channels
            # similarity_matrix = -similarity_matrix
            similarity_matrix = data['distance_matrices'][s]  # keep all channels
            paired_stim = []
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:  # remove first trials
                    if 'rep2' in stim:
                            if stim in subject_stim_features['stim_name'].tolist():
                                if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                    indices_to_keep.append(i)
                                    correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                    if 'rep2' in stim:
                                        # indices_to_keep.append(i)
                                        # correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                        # feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                        if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                            paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))
                                        else:
                                            paired_stim.append('no_pair')
                                    if 'rep1' in stim:
                                        # indices_to_keep.append(i)
                                        # correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                        # feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                        if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                                            paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2')))
                                        else:
                                            paired_stim.append('no_pair')
                                    # else:
                                    #     raise ValueError('Neither rep1 nor rep2???')
            # plt.imshow(np.mean(similarity_matrix, axis=-1), cmap='gray')
            # plt.show()
            condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
            # condition_matrix = condition_matrix[:, indices_to_keep, :]  # select condition
            correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
            for i in range(condition_matrix.shape[0]):
                current_corr = condition_matrix[i, :, :]
                mask = np.ones(condition_matrix.shape[1], dtype=bool)
                mask[indices_to_keep[i]] = False  # remove current stim
                # mask[i] = False  # remove current stim
                if not paired_stim[i] == 'no_pair':
                    mask[paired_stim[i]] = False  # remove pair
                # for j in range(indices_to_keep[i], mask.shape[0]):
                #     mask[j] = False  # remove following trials

                # plt.hist(current_corr[mask, 10], bins=100)
                # plt.show()
                # print(np.mean(current_corr[mask, :], axis=0))
                correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)
            # print(correlations_array.shape)
            # plt.figure()
            # plt.plot(correlations_array)
            # plt.show()

            # pvalues = np.zeros(correlations_array.shape[1])
            # p_ttest = np.zeros(correlations_array.shape[1])
            for channel in range(correlations_array.shape[1]):
                if subject + '_' + ch_names[channel] in region_electrodes:
                    Y = correlations_array[:, channel]
                    X = np.zeros((Y.shape[0], 1))
                    # X = np.zeros((Y.shape[0], 2))
                    X[:, 0] = correct_list
                    # X[:, 1] = np.array(feature_list)**2
                    X = sm.add_constant(X)

                    # independent t-test
                    correct_corr = Y[X[:, 1] == 1]  # axis 0 is the intercept
                    incorrect_corr = Y[X[:, 1] == 0]
                    p_ttest = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=10000,
                                                 random_state=None,
                                                 # alternative='less').pvalue  # testing of individual electrodes
                                                 alternative='greater').pvalue  # testing of individual electrodes
                    if p_ttest < 0.05:
                        significant_electrodes['inter_correct'].append(subject + '_' + ch_names[channel])

                    # ols = sm.OLS(Y, X).fit()
                    # param = ols.params[1]
                    # p_ttest[channel] = ols.pvalues[1]

                    X = np.zeros((Y.shape[0], 1))
                    X[:, 0] = feature_list
                    X = sm.add_constant(X)
                    ols = sm.OLS(Y, X).fit()
                    param = ols.params[1]
                    p_ttest = ols.pvalues[1]

                    if (p_ttest < 0.1) & (param > 0):  # 0.1 because it is a one-tailed test
                        significant_electrodes['inter_surprise'].append(subject + '_' + ch_names[channel])

    print(significant_electrodes)

    new_names = {'ctx-lh-transversetemporal': 'lTTG', 'ctx-lh-superiortemporal': 'lSTG', 'ctx-rh-transversetemporal': 'rTTG', 'ctx-rh-superiortemporal': 'rSTG',
               'ctx-lh-middletemporal': 'lMTG', 'ctx-rh-middletemporal': 'rMTG',
               'Left-Hippocampus': 'Left hippocampus', 'Left-Amygdala': 'Left amygdala', 'ctx-lh-parahippocampal': 'lPHG', 'ctx-lh-temporalpole': 'lTmP',
               'ctx-lh-inferiortemporal': 'lITG', 'ctx-lh-entorhinal': 'lEnt', 'Right-Hippocampus': 'Right hippocampus', 'Right-Amygdala': 'Right amygdala',
               'ctx-rh-parahippocampal': 'rPHG', 'ctx-rh-temporalpole': 'rTmP', 'ctx-rh-inferiortemporal': 'rITG', 'ctx-rh-entorhinal': 'rEnt'}

    # region pie charts
    from collections import Counter
    baseline_counts = Counter(electrodes_regions_dict[e] for e in significant_electrodes['all_electrodes'] if e in electrodes_regions_dict)
    print(baseline_counts)
    total_baseline = sum(baseline_counts.values())
    unique_regions = sorted(set(electrodes_regions_dict[e] for e in significant_electrodes['all_electrodes'] if e in electrodes_regions_dict))

    cmap = plt.get_cmap('Set3').colors
    # cmap = [cmap[s] for s in range(int(len(regions) / 2))]
    # region_colors = cmap + cmap
    # cmap = plt.get_cmap("Set3")  # Use a colormap with enough distinct colors
    # cmap = plt.get_cmap("tab20")  # Tab20 has 20 distinct colors
    # region_colors = {region: cmap(i % len(cmap.colors)) for i, region in enumerate(unique_regions)}
    region_colors = {region: cmap[i % int(len(regions)/2)] for i, region in enumerate(regions)}
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))  # Arrange in a 2x3 grid
    axes = axes.flatten()  # Flatten to iterate easily
    for idx, (condition, electrodes) in enumerate(significant_electrodes.items()):
        # Count the occurrences of each region in this condition
        electrodes = sorted(electrodes)
        region_counts = Counter(electrodes_regions_dict[e] for e in electrodes if e in electrodes_regions_dict)

        # Compute binomial tests
        significant_regions = []
        for region, count in region_counts.items():
            p_expected = baseline_counts.get(region, 0) / total_baseline  # Expected proportion
            print(len(electrodes))
            p_value = binom_test(count, n=len(electrodes), p=p_expected, alternative='greater')  # One-tailed test
            if p_value < 0.05:
                significant_regions.append(region)

        significant_regions = [new_names[reg] for reg in significant_regions]  # convert to new names

        # Plot pie chart
        # region_counts = {key: value for key, value in sorted(region_counts.items())}
        region_counts = {key: region_counts[key] for key in regions if key in region_counts}
        print(region_counts)
        colors = [region_colors[region] for region in region_counts.keys()]
        # labels = [f"{region} ({count})" for region, count in region_counts.items()]
        labels = [f"{new_names[region]} ({count})" for region, count in region_counts.items()]
        wedges, texts = axes[idx].pie(region_counts.values(), labels=labels, colors=colors, startangle=140)
        for t, text in enumerate(texts):
            region_name = labels[t].split(" (")[0]  # remove count
            # region_name = labels[t][:-4]  # remove count
            print(region_name)
            if region_name in significant_regions:
                text.set_color('red')  # Highlight significant regions in red
        axes[idx].set_title(condition)

        # Perform chi-square test (skip first condition which is "all electrodes")
        if condition != "all_electrodes":
            observed = [region_counts.get(region, 0) for region in unique_regions]  # Fill missing with 0
            print(observed)
            expected = [baseline_counts.get(region, 0) * sum(observed) / sum(baseline_counts.values()) for region in unique_regions]
            # Run chi-square test
            chi2_stat, p_value, _, _ = chi2_contingency([observed, expected])
            # Print results
            print(f"Chi-square test for condition: {condition}")
            print(f"Chi2 statistic = {chi2_stat:.2f}, p-value = {p_value:.4f}")
            if p_value < 0.05:
                print("→ Significant difference in proportions compared to baseline!\n")
            else:
                print("→ No significant difference in proportions.\n")

    plt.tight_layout()


    # barplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))  # Arrange in a 2x3 grid
    axes = axes.flatten()  # Flatten to iterate easily
    for idx, (condition, electrodes) in enumerate(significant_electrodes.items()):
        # Count the occurrences of each region in this condition
        electrodes = sorted(electrodes)
        region_counts = Counter(electrodes_regions_dict[e] for e in electrodes if e in electrodes_regions_dict)
        baseline_counts = Counter(electrodes_regions_dict[e] for e in significant_electrodes['all_electrodes'] if e in electrodes_regions_dict)

        # Compute binomial tests
        significant_regions = []
        print(condition)
        for region, count in region_counts.items():
            p_expected = baseline_counts.get(region, 0) / total_baseline  # Expected proportion
            # p_value = binom_test(count, n=len(electrodes), p=p_expected, alternative='greater')  # One-tailed test
            btest = binomtest(count, n=len(electrodes), p=p_expected, alternative='greater')  # One-tailed test
            print(region)
            print(btest)
            print('CI:', btest.proportion_ci())
            if btest.pvalue < 0.05:
                significant_regions.append(region)

        significant_regions = [new_names[reg] for reg in significant_regions]  # convert to new names

        # Plot pie chart
        region_counts = {key: region_counts[key] for key in regions if key in region_counts}
        region_counts_baseline = {key: baseline_counts[key] for key in regions if key in baseline_counts}
        print(region_counts)
        print(region_counts_baseline)
        # labels = [new_names[region] for region in region_counts.keys()]
        labels = [new_names[region] for region in regions]
        # counts = list(region_counts.values())
        counts = [region_counts.get(region, 0) for region in regions]
        baseline_counts = [region_counts_baseline[region] for region in region_counts_baseline.keys()]
        # colors = [region_colors[region] for region in region_counts.keys()]
        colors = [region_colors[region] for region in regions]
        print(len(labels))
        print(len(baseline_counts))
        axes[idx].barh(labels, counts, color=colors)
        axes[idx].barh(labels, baseline_counts, color='none', edgecolor=colors, linewidth=2.8)
        # axes[idx].invert_yaxis()
        axes[idx].set_xlabel('Count')

        for t, region_name in enumerate(labels):
            if region_name in significant_regions:
                axes[idx].text(baseline_counts[t] + 0.7, t - 0.08, '*', va='center_baseline', fontsize=14, color='black', fontweight='bold')

        axes[idx].set_title(condition)

    plt.tight_layout()

    # barplots of proportion
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))  # Arrange in a 2x3 grid
    axes = axes.flatten()  # Flatten to iterate easily
    for idx, (condition, electrodes) in enumerate(significant_electrodes.items()):
        # Count the occurrences of each region in this condition
        electrodes = sorted(electrodes)
        region_counts = Counter(electrodes_regions_dict[e] for e in electrodes if e in electrodes_regions_dict)
        baseline_counts = Counter(electrodes_regions_dict[e] for e in significant_electrodes['all_electrodes'] if
                                  e in electrodes_regions_dict)
        print(region_counts)
        print(baseline_counts)

        # Plot pie chart
        region_counts = {key: region_counts[key] for key in regions if key in region_counts}
        region_counts_baseline = {key: baseline_counts[key] for key in regions if key in baseline_counts}
        print(region_counts)
        print(region_counts_baseline)
        # labels = [new_names[region] for region in region_counts.keys()]
        labels = [new_names[region] for region in regions]
        # counts = list(region_counts.values())
        counts = [region_counts.get(region, 0) / region_counts_baseline[region] for region in regions]
        baseline_counts = [region_counts_baseline[region] for region in region_counts_baseline.keys()]
        # colors = [region_colors[region] for region in region_counts.keys()]
        colors = [region_colors[region] for region in regions]
        print(len(labels))
        print(len(baseline_counts))
        axes[idx].barh(labels, counts, color=colors)
        # wedges, texts = axes[idx].pie(region_counts.values(), labels=labels, colors=colors, startangle=140)
        # axes[idx].invert_yaxis()
        axes[idx].set_xlabel('Significant electrodes proportion')
        axes[idx].set_title(condition)

    plt.tight_layout()


    # Venn diagrams for functionnal overlaps
    from venn import venn
    # Convert all values to sets
    significant_electrodes = {key: set(value) for key, value in significant_electrodes.items()}

    fig, axes = plt.subplots(3, 3)

    filtered_data = {k: v for k, v in significant_electrodes.items() if k in {'all_electrodes', 'novelty', 'retrieval'}}
    venn(filtered_data, ax = axes.flat[0])
    filtered_data = {k: v for k, v in significant_electrodes.items() if k in {'all_electrodes', 'intra_correct', 'inter_correct'}}
    venn(filtered_data, ax=axes.flat[1])
    filtered_data = {k: v for k, v in significant_electrodes.items() if k in {'all_electrodes', 'intra_surprise', 'inter_surprise'}}
    venn(filtered_data, ax=axes.flat[2])
    filtered_data = {k: v for k, v in significant_electrodes.items() if k in {'all_electrodes', 'intra_correct', 'intra_surprise'}}
    venn(filtered_data, ax=axes.flat[3])
    filtered_data = {k: v for k, v in significant_electrodes.items() if k in {'all_electrodes', 'inter_correct', 'inter_surprise'}}
    venn(filtered_data, ax=axes.flat[4])
    filtered_data = {k: v for k, v in significant_electrodes.items() if k in {'novelty', 'intra_correct', 'intra_surprise'}}
    venn(filtered_data, ax=axes.flat[5])
    filtered_data = {k: v for k, v in significant_electrodes.items() if k in {'novelty', 'inter_correct', 'inter_surprise'}}
    venn(filtered_data, ax=axes.flat[6])
    filtered_data = {k: v for k, v in significant_electrodes.items() if k in {'retrieval', 'intra_correct', 'intra_surprise'}}
    venn(filtered_data, ax=axes.flat[7])
    filtered_data = {k: v for k, v in significant_electrodes.items() if k in {'retrieval', 'inter_correct', 'inter_surprise'}}
    venn(filtered_data, ax=axes.flat[8])

    # # euler diagrams, does not work with python < 3.9.... :'(
    # matplotlib_set_diagrams
    # from matplotlib_set_diagrams import EulerDiagram
    # fig, axes = plt.subplots(3, 3)
    # filtered_data = {k: v for k, v in significant_electrodes.items() if k in {'all_electrodes', 'novelty', 'retrieval'}}
    # EulerDiagram.from_sets(filtered_data, ax=axes.flat[0])


    plt.show()



def representation_analysis_windows_seeg(rms_path, behavior_path, similarity_path, similarity_window_path, distance_path, distance_windows_path, feature, regions, first_parcel_label=True):
    """Takes all electrodes from a big region (ex: all temporal lobe) and selected the subset of electrodes that show a
    significant RMS contrast (in either novelty detection or retrieval), then run intra and inter stimulus correlation
    analysis, linking them to correct response and surprise"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    del subject_paths[1]
    subject_names = []
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate(
                [[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
            # for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty]]):
            # for cond, correct_incorrect in enumerate([[correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            # pval = ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False,
                            #                                permutations=10000, random_state=None,
                            #                                alternative='greater').pvalue  # testing of individual electrodes
                            # if pval < 0.05:
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    print(significant_electrodes)
    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])
    print(chan_count)


    ############## Intrastim correlation, no window (to select significant channels ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', feature, "correct"]]
    print(stim_features)

    # data = np.load(data_path, allow_pickle=True)
    data = np.load(similarity_path, allow_pickle=True)

    correct_channels = {}
    feature_channels = {}
    for s, subject in tqdm(enumerate(data["subjects"])):
        if subject != 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            correlations_array = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['similarity_matrices'][s]  # keep all channels
            # similarity_matrix = data['distance_matrices'][s]  # keep all channels
            # similarity_matrix = -similarity_matrix
            # similarity_matrix = 1 / similarity_matrix
            first_trials_to_remove = 50  # very sensitive to the first trials, then stabilizes (mostly for interstim distance, which makes sense)
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:  # remove first trials
                    if stim in subject_stim_features['stim_name'].tolist():
                        if 'rep2' in stim:
                            if not np.isnan(
                                    subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                indices_to_keep.append(i)
                                if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                    rep1_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                    correlations_array.append(similarity_matrix[i, rep1_index, :])  # correlation of 2 reps
                                    correct_list.append(
                                        subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                    feature_list.append(
                                        subject_stim_features[subject_stim_features['stim_name'] == stim][feature].iloc[0])

            correlations_array = np.array(correlations_array)

            correct_channels[subject] = []
            feature_channels[subject] = []

            for channel in range(correlations_array.shape[1]):
                if ch_names[channel] in significant_electrodes[subject]:
                    Y = np.array(correct_list)
                    X = np.zeros((Y.shape[0], 1))
                    X[:, 0] = correlations_array[:, channel]
                    X = (X - X.mean()) / X.std()
                    X = sm.add_constant(X)

                    ols = sm.Logit(Y, X).fit()
                    param = ols.params[1]
                    p_ttest = ols.pvalues[1]
                    if (p_ttest < 0.1) & (param > 0):
                        correct_channels[subject].append(channel)

                    Y = X[:, 1]
                    X = np.zeros((Y.shape[0], 1))
                    X[:, 0] = feature_list
                    X = (X - X.mean()) / X.std()
                    X = sm.add_constant(X)
                    ols = sm.OLS(Y, X).fit()
                    param = ols.params[1]
                    p_ttest = ols.pvalues[1]

                    if (param < 0) & (p_ttest < 0.1):  # one-tailed test
                        feature_channels[subject].append(channel)


    ############## Interstim distance, no window (to select significant channels) ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', feature, "correct"]]
    print(stim_features)

    data = np.load(distance_path, allow_pickle=True)

    channel_positions_list = []
    subject_id = []
    regression_param_correct = []
    regression_param_feature = []
    correct_channels_inter = {}
    feature_channels_inter = {}
    for s, subject in tqdm(enumerate(data["subjects"])):
        if subject != 'Bou_Ni':
            subject_stim_features = stim_features[stim_features['subject'] == subject]
            subject_info = data['info'][s]
            ch_names = subject_info['ch_names']
            indices_to_keep = []
            correct_list = []
            feature_list = []
            stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
            similarity_matrix = data['distance_matrices'][s]  # keep all channels
            paired_stim = []
            for i, stim in enumerate(data['stim_names'][s]):
                if i > first_trials_to_remove:  # remove first trials
                    if stim in subject_stim_features['stim_name'].tolist():
                        if not np.isnan(
                                subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                            indices_to_keep.append(i)
                            correct_list.append(
                                subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                            feature_list.append(
                                subject_stim_features[subject_stim_features['stim_name'] == stim][feature].iloc[0])
                            if 'rep2' in stim:
                                if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                    paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))
                                else:
                                    paired_stim.append('no_pair')
                            if 'rep1' in stim:
                                if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                                    paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2')))
                                else:
                                    paired_stim.append('no_pair')

            condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
            # condition_matrix = condition_matrix[:, indices_to_keep, :]  # select condition
            correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
            for i in range(condition_matrix.shape[0]):
                current_corr = condition_matrix[i, :, :]
                mask = np.ones(condition_matrix.shape[1], dtype=bool)
                mask[indices_to_keep[i]] = False  # remove current stim
                # mask[i] = False  # remove current stim
                if not paired_stim[i] == 'no_pair':
                    mask[paired_stim[i]] = False  # remove pair
                # for j in range(indices_to_keep[i], mask.shape[0]):
                #     mask[j] = False  # remove following trials
                correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)

            correct_channels_inter[subject] = []
            feature_channels_inter[subject] = []

            for channel in range(correlations_array.shape[1]):
                if ch_names[channel] in significant_electrodes[subject]:
                    Y = np.array(correct_list)
                    X = np.zeros((Y.shape[0], 1))
                    X[:, 0] = correlations_array[:, channel]
                    X = (X - X.mean()) / X.std()
                    X = sm.add_constant(X)

                    ols = sm.Logit(Y, X).fit()
                    param = ols.params[1]
                    p_ttest = ols.pvalues[1]
                    if (p_ttest < 0.1) & (param > 0):
                        correct_channels_inter[subject].append(channel)

                    Y = X[:, 1]
                    X = np.zeros((Y.shape[0], 1))
                    X[:, 0] = feature_list
                    X = (X - X.mean()) / X.std()
                    X = sm.add_constant(X)

                    ols = sm.OLS(Y, X).fit()
                    param = ols.params[1]
                    p_ttest = ols.pvalues[1]
                    if (param > 0) & (p_ttest < 0.1):  # one-tailed test
                        feature_channels_inter[subject].append(channel)


    ############## Windows loop ##############

    n_windows = 8
    conditions = ['Correct ~ intrastim correlation', 'Intrastim correlation ~ surprise', 'Correct ~ interstim distance', 'Interstim distance ~ surprise']
    fig, axs = plt.subplots(1, len(conditions), sharey=True)

    for window in range(n_windows):

        ############## Intrastim correlation ##############
        stim_features = pd.read_csv(behavior_path)
        stim_features = stim_features[['stim_name', 'subject', feature, "correct"]]
        print(stim_features)

        data = np.load(similarity_window_path, allow_pickle=True)

        channel_positions_list = []
        subject_id = []
        regression_param_correct = []
        regression_param_feature = []
        all_correlations = []
        # pvalues_array = []  # p value for the regression
        p_ttest_array = []  # p value of the contrast correct vs incorrect
        for s, subject in tqdm(enumerate(data["subjects"])):
            if subject != 'Bou_Ni':
                subject_stim_features = stim_features[stim_features['subject'] == subject]
                subject_info = data['info'][s]
                ch_names = subject_info['ch_names']
                indices_to_keep = []
                correct_list = []
                feature_list = []
                correlations_array = []
                stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
                similarity_matrix = data['similarity_matrices'][s][window]
                print(similarity_matrix.shape)
                # similarity_matrix = data['distance_matrices'][s]
                # similarity_matrix = 1/similarity_matrix
                # first_trials_to_remove = 50  # very sensitive to the first trials, then stabilizes (mostly for interstim distance, which makes sense)
                for i, stim in enumerate(data['stim_names'][s]):
                    if i > first_trials_to_remove:
                        if stim in subject_stim_features['stim_name'].tolist():
                            if 'rep2' in stim:
                                if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                    indices_to_keep.append(i)
                                    if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                        rep1_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                        correlations_array.append(similarity_matrix[i, rep1_index, :])  # correlation of 2 reps
                                        correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                        feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][feature].iloc[0])

                correlations_array = np.array(correlations_array)

                print(correlations_array.shape)
                print(len(ch_names))

                for channel in range(correlations_array.shape[1]):
                    if ch_names[channel] in significant_electrodes[subject]:
                        Y = np.array(correct_list)
                        X = np.zeros((Y.shape[0], 1))
                        X[:, 0] = correlations_array[:, channel]
                        all_correlations += correlations_array[:, channel].tolist()
                        X = (X - X.mean()) / X.std()
                        X = sm.add_constant(X)

                        if channel in correct_channels[subject]:
                            ols = sm.Logit(Y, X).fit()
                            # if (ols.pvalues[1] < 0.1) & (ols.params[1] > 0):
                            regression_param_correct.append(ols.params[1])

                        Y = X[:, 1]
                        X = np.zeros((Y.shape[0], 1))
                        # X = np.zeros((Y.shape[0], 2))
                        X[:, 0] = feature_list
                        # X[:, 1] = np.array(feature_list)**2
                        X = (X - X.mean()) / X.std()
                        X = sm.add_constant(X)

                        if channel in feature_channels[subject]:
                            ols = sm.OLS(Y, X).fit()
                            # if (ols.pvalues[1] < 0.1) & (ols.params[1] < 0):
                            regression_param_feature.append(ols.params[1])

                        channel_positions_list.append(subject_info['chs'][channel]['loc'][:3])
                        subject_id.append(s)

        regression_param_correct = np.array(regression_param_correct) / np.std(regression_param_correct)
        print(regression_param_correct.shape)
        regression_param_feature = np.array(regression_param_feature) / np.std(regression_param_feature)
        print(regression_param_feature.shape)


        ############## Interstim correlation ##############
        stim_features = pd.read_csv(behavior_path)
        stim_features = stim_features[['stim_name', 'subject', feature, "correct"]]

        data = np.load(distance_windows_path, allow_pickle=True)

        regression_param_correct_inter = []
        regression_param_feature_inter = []
        for s, subject in tqdm(enumerate(data["subjects"])):
            if subject != 'Bou_Ni':
                subject_stim_features = stim_features[stim_features['subject'] == subject]
                subject_info = data['info'][s]
                ch_names = subject_info['ch_names']
                indices_to_keep = []
                correct_list = []
                feature_list = []
                stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
                similarity_matrix = data['distance_matrices'][s][window]  # keep all channels
                paired_stim = []
                for i, stim in enumerate(data['stim_names'][s]):
                    if i > first_trials_to_remove:  # remove first trials
                        if stim in subject_stim_features['stim_name'].tolist():
                            if not np.isnan(
                                    subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                                indices_to_keep.append(i)
                                correct_list.append(
                                    subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(
                                    subject_stim_features[subject_stim_features['stim_name'] == stim][feature].iloc[0])
                                if 'rep2' in stim:
                                    if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                        paired_stim.append(
                                            stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))
                                    else:
                                        paired_stim.append('no_pair')
                                if 'rep1' in stim:
                                    if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                                        paired_stim.append(
                                            stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2')))
                                    else:
                                        paired_stim.append('no_pair')
                condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
                # condition_matrix = condition_matrix[:, indices_to_keep, :]  # select condition
                correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
                for i in range(condition_matrix.shape[0]):
                    current_corr = condition_matrix[i, :, :]
                    mask = np.ones(condition_matrix.shape[1], dtype=bool)
                    mask[indices_to_keep[i]] = False  # remove current stim
                    # mask[i] = False  # remove current stim
                    if not paired_stim[i] == 'no_pair':
                        mask[paired_stim[i]] = False  # remove pair
                    # for j in range(indices_to_keep[i], mask.shape[0]):
                    #     mask[j] = False  # remove following trials
                    correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)

                for channel in range(correlations_array.shape[1]):
                    if ch_names[channel] in significant_electrodes[subject]:
                        Y = np.array(correct_list)
                        X = np.zeros((Y.shape[0], 1))
                        X[:, 0] = correlations_array[:, channel]
                        X = (X - X.mean()) / X.std()
                        X = sm.add_constant(X)

                        if channel in correct_channels_inter[subject]:
                            ols = sm.Logit(Y, X).fit()
                            regression_param_correct_inter.append(ols.params[1])

                        Y = X[:, 1]
                        X = np.zeros((Y.shape[0], 1))
                        X[:, 0] = feature_list
                        X = (X - X.mean()) / X.std()
                        X = sm.add_constant(X)

                        if channel in feature_channels_inter[subject]:
                            ols = sm.OLS(Y, X).fit()
                            regression_param_feature_inter.append(ols.params[1])

        regression_param_correct_inter = np.array(regression_param_correct_inter) / np.std(regression_param_correct_inter)
        print(regression_param_correct_inter.shape)
        regression_param_feature_inter = np.array(regression_param_feature_inter) / np.std(regression_param_feature_inter)
        print(regression_param_feature_inter.shape)


        ############## Plotting ##############

        params_conditions = [regression_param_correct, regression_param_feature, regression_param_correct_inter, regression_param_feature_inter]

        for i, (condition, params) in enumerate(zip(conditions, params_conditions)):
            population_p = ttest_1samp(params, 0).pvalue
            axs[i].bar(window, np.mean(params), yerr=np.std(params) / np.sqrt(len(params)), capsize=5,
                       color=('orange' if population_p < 0.05 else 'skyblue'))
            axs[i].axhline(0, color='gray', linestyle='--', linewidth=1.5)

            # Add individual sample points as scatter dots
            colors = 'black'
            x_values = np.full(len(params), window) + np.random.uniform(-0.1, 0.1, len(params))  # Jitter for clarity
            axs[i].scatter(x_values, params, color=colors, alpha=0.7, s=5, label='Samples')
            axs[i].set_title(condition)
            axs[i].set_xticks([x for x in range(n_windows)], [x for x in range(n_windows)])
            axs[i].set_xlabel(f'From x to x+1 seconds')
    axs[0].set_ylabel(f'Regression parameters')

    plt.show()


def intra_and_interstim_save_for_decoding(save_path, rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=True):
    """"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    subject_names = []
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
        # for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty]]):
        # for cond, correct_incorrect in enumerate([[correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            # pval = ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False,
                            #                                permutations=10000, random_state=None,
                            #                                alternative='greater').pvalue  # testing of individual electrodes
                            # if pval < 0.05:
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    print(significant_electrodes)
    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])
    print(chan_count)


    ############## Representation analysis ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(similarity_path, allow_pickle=True)
    distance_data = np.load(distance_path, allow_pickle=True)

    subject_list = []
    channel_list = []
    correct_full_list = []
    surprise_full_list = []
    intrastim_distance_full_array = []
    interstim_distance_full_array = []
    for s, subject in tqdm(enumerate(data["subjects"])):
        subject_stim_features = stim_features[stim_features['subject'] == subject]
        subject_info = data['info'][s]
        ch_names = subject_info['ch_names']
        indices_to_keep = []
        correct_list = []
        feature_list = []
        intrastim_distance_array = []
        stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
        similarity_matrix = data['similarity_matrices'][s]  # keep all channels
        distance_matrix = distance_data['distance_matrices'][s]  # keep all channels
        paired_stim = []
        first_trials_to_remove = 50
        for i, stim in enumerate(data['stim_names'][s]):
            if i > first_trials_to_remove:
                if stim in subject_stim_features['stim_name'].tolist():
                    if 'rep2' in stim:
                        if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                            if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                indices_to_keep.append(i)
                                rep1_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                                intrastim_distance_array.append(similarity_matrix[i, rep1_index, :])  # correlation of 2 reps
                                correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                                feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                                paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))

        intrastim_distance_array = np.array(intrastim_distance_array)
        # condition_matrix = -similarity_matrix[indices_to_keep, :, :]  # select condition
        condition_matrix = distance_matrix[indices_to_keep, :, :]
        interstim_distance_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
        for i in range(condition_matrix.shape[0]):
            current_corr = condition_matrix[i, :, :]
            mask = np.ones(condition_matrix.shape[1], dtype=bool)
            mask[indices_to_keep[i]] = False  # remove current stim
            if not paired_stim[i] == 'no_pair':
                mask[paired_stim[i]] = False  # remove pair
            interstim_distance_array[i, :] = np.mean(current_corr[mask, :], axis=0)

        channels_to_keep = []
        for channel in range(intrastim_distance_array.shape[1]):
            if ch_names[channel] in significant_electrodes[subject]:
                subject_list += [subject for _ in range(len(correct_list))]
                channel_list += [ch_names[channel] for _ in range(len(correct_list))]
                correct_full_list += correct_list
                surprise_full_list += feature_list
                intrastim_distance_full_array += intrastim_distance_array[:, channel].tolist()
                # correct_corr = Y[X == 1]  # axis 0 is the intercept
                # incorrect_corr = Y[X == 0]
                # p_ttest_intra = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=10000,
                #                              random_state=None,
                #                              # alternative='less').pvalue  # testing of individual electrodes
                #                              alternative='greater').pvalue  # testing of individual electrodes
                interstim_distance_full_array += interstim_distance_array[:, channel].tolist()
                # correct_corr = Y[X == 1]  # axis 0 is the intercept
                # incorrect_corr = Y[X == 0]
                # p_ttest_inter = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=10000,
                #                           random_state=None,
                #                           alternative='greater').pvalue  # testing of individual electrodes
                #                           # alternative='less').pvalue  # testing of individual electrodes
                # # if (p_ttest_intra < 0.05) | (p_ttest_inter < 0.05):
                # if (p_ttest_intra < 0.1) | (p_ttest_inter < 0.1):
                # # if (p_ttest_intra < 0.05) & (p_ttest_inter < 0.05):
                #     channels_to_keep.append(channel)



        # if len(channels_to_keep) > 0:
        #     correct_full_list += correct_list
        #     intra = np.mean(intrastim_distance_array[:, channels_to_keep], axis=1)
        #     inter = np.mean(interstim_distance_array[:, channels_to_keep], axis=1)
        #     intrastim_distance_full_array += ((intra - np.mean(intra)) / np.std(intra)).tolist()
        #     interstim_distance_full_array += ((inter - np.mean(inter)) / np.std(inter)).tolist()

    df = pd.DataFrame({'subject': subject_list, 'electrode': channel_list, 'correct': correct_full_list, 'surprise': surprise_full_list, 'intrastim_corr': intrastim_distance_full_array, 'interstim_dist': interstim_distance_full_array})
    df.to_csv(os.path.join(save_path, 'representation_decoding.csv'))


def representation_analysis_seeg_on_rms(rms_path, behavior_path, condition, regions, first_parcel_label=True):
    """Takes all electrodes from a big region (ex: all temporal lobe) and selected the subset of electrodes that show a
    significant RMS contrast (in either novelty detection or retrieval), then run intra and inter stimulus correlation
    analysis, linking them to correct response and surprise"""

    ############## Electrode selection based on RMS contrast ##############

    behavior_all = pd.read_csv(behavior_path)
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    del subject_paths[1]
    subject_names = []
    print(subject_paths)

    significant_electrodes = {}  # dict of signif electrodes by subject

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
        correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
        incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)
        # print(np.array(rms_obj['ch_names']))

        significant_electrodes[subject_name] = []
        for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval]]):
        # for cond, correct_incorrect in enumerate([[correct_novelty, incorrect_novelty]]):
        # for cond, correct_incorrect in enumerate([[correct_retrieval, incorrect_retrieval]]):
            correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
            incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
            if (len(correct) > 0) & (len(incorrect) > 0):
                for c in range(rms.shape[1]):
                    for region in regions:
                        if any(ch_parcel in region for ch_parcel in parcellations[c]):
                            # pval = ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False,
                            #                                permutations=10000, random_state=None,
                            #                                alternative='greater').pvalue  # testing of individual electrodes
                            # if pval < 0.05:
                            significant_electrodes[subject_name].append(rms_obj['ch_names'][c])
        significant_electrodes[subject_name] = np.unique(significant_electrodes[subject_name]).tolist()  # keep each electrode only once

    print(significant_electrodes)
    chan_count = 0
    for subject in significant_electrodes:
        chan_count += len(significant_electrodes[subject])
    print(chan_count)


    ############## rms ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    subject_id = []
    regression_param_correct = []
    regression_param_feature = []
    all_correlations = []
    # pvalues_array = []  # p value for the regression
    p_ttest_array = []  # p value of the contrast correct vs incorrect
    for s, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        subject_names.append(subject_name)
        subject_stim_features = stim_features[stim_features['subject'] == subject_name]
        correct_list = []
        feature_list = []
        rms_array = []

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj['rms']
        ch_names = rms_obj['ch_names']
        for i, stim in enumerate(rms_obj['stim_name']):
            if stim in subject_stim_features['stim_name'].tolist():
                if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                    rms_array.append(rms[i, :])
                    correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                    feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])

        rms_array = np.array(rms_array)
        print(rms_array.shape)

        # pvalues = np.zeros(correlations_array.shape[1])
        p_ttest = np.zeros(rms_array.shape[1])
        for channel in range(rms_array.shape[1]):
            if ch_names[channel] in significant_electrodes[subject_name]:
                # Y = correlations_array[:, channel]
                # all_correlations += Y.tolist()
                # Y = (Y - Y.mean()) / Y.std()
                # X = np.zeros((Y.shape[0], 1))
                # # X = np.zeros((Y.shape[0], 2))
                # X[:, 0] = correct_list
                # # X[:, 1] = np.array(feature_list)**2
                # X = sm.add_constant(X)

                Y = np.array(correct_list)
                X = np.zeros((Y.shape[0], 1))
                X[:, 0] = rms_array[:, channel]
                all_correlations += rms_array[:, channel].tolist()
                X = (X - X.mean()) / X.std()
                X = sm.add_constant(X)

                ols = sm.Logit(Y, X).fit()
                regression_param_correct.append(ols.params[1])
                # pvalues[channel] = ols.pvalues[1]
                # regression_param[channel] = ols.params[2]

                Y = X[:, 1]
                # X = np.zeros((Y.shape[0], 1))
                X = np.zeros((Y.shape[0], 2))
                X[:, 0] = feature_list
                X[:, 1] = np.array(feature_list)**2
                X = (X - X.mean(axis=0)) / X.std(axis=0)
                X = sm.add_constant(X)
                ols = sm.OLS(Y, X).fit()
                # regression_param_feature.append(ols.params[1])
                regression_param_feature.append(ols.params[2])

                subject_id.append(s)
        # pvalues_array.append(pvalues)
        # p_ttest_array.append(p_ttest)

    regression_param_correct = np.array(regression_param_correct)
    print(regression_param_correct.shape)
    regression_param_feature = np.array(regression_param_feature)
    print(regression_param_feature.shape)
    # pvalues_array = np.concatenate(pvalues_array)
    # print(np.sum(pvalues_array < 0.05), 'significant electrodes,',
    #       round(np.sum(pvalues_array < 0.05) / pvalues_array.shape[0] * 100, 3), "%")
    # print(regression_param_array[pvalues_array < 0.05])
    # p_ttest_array = np.concatenate(p_ttest_array)
    # print(np.sum(p_ttest_array < 0.05), 'significant electrodes with independent ttest,',
    #       round(np.sum(p_ttest_array < 0.05) / p_ttest_array.shape[0] * 100, 3), "%")
    # print(regression_param_array[p_ttest_array < 0.05])

    list_of_pvalues = []

    # histogram of rms
    fig, axs = plt.subplots(1, 1)
    all_correlations = np.array(all_correlations)
    print(all_correlations.shape)
    axs.hist(all_correlations, bins=100)
    ttest = scipy.stats.ttest_1samp(all_correlations, 0, alternative='two-sided')
    axs.axvline(0, ls='--', color='k')
    axs.axvline(np.mean(all_correlations), color='r', lw=2)
    axs.set_xlabel('RMS')
    axs.set_ylabel('Count (epochs)')
    axs.text(plt.gca().get_xlim()[0], plt.gca().get_ylim()[1],
                f'Average RMS = {round(np.mean(all_correlations), 5)}, p = {round(ttest.pvalue, 5)}')

    fig, axs = plt.subplots(1, 2, sharey=True)
    x_regression = ['correct ~ rms', f'rms ~ {condition}²']
    alternative = ['two-sided', 'two-sided']
    for i, regression_param_array in enumerate([regression_param_correct, regression_param_feature]):
        print(np.mean(regression_param_array))
        ttest = scipy.stats.ttest_1samp(regression_param_array, 0, alternative=alternative[i])
        print(scipy.stats.ttest_1samp(regression_param_array, 0, alternative=alternative[i]))
        list_of_pvalues.append(ttest.pvalue)
        axs[i].hist(regression_param_array, bins=100)
        axs[i].axvline(0, ls='--', color='k')
        axs[i].axvline(np.mean(regression_param_array), color='r', lw=2)
        # plt.xlabel('Regression parameter (intrastimulus correlation ~ surprise)')
        # axs[i].set_xlabel(f'Regression parameter (intrastimulus correlation ~ {x_regression[i]})')
        axs[i].set_xlabel(f'Regression parameter ({x_regression[i]})')
        axs[i].set_ylabel('Count (electrode)')
        axs[i].text(plt.gca().get_xlim()[0], plt.gca().get_ylim()[1],
                 f'regression beta = {round(np.mean(regression_param_array), 5)}, p = {round(ttest.pvalue, 5)}')

    params = regression_param_correct.copy()
    feature_params = regression_param_feature.copy()

    # boxplots
    df = pd.DataFrame({'Correct ~ RMS': params, 'RMS ~ surprise²': feature_params})
    plt.figure(figsize=(5, 5))
    sns.violinplot(data=df, showfliers=False, palette='Set3', orient='v', inner=None)
    sns.swarmplot(data=df, color=".25", size=2.5, orient='v')
    plt.gca().axhline(0, ls='--', color='k')
    plt.ylabel('\u03B2 parameters', fontsize=17)
    # plt.yticks(plt.yticks()[0], plt.yticks()[1], rotation=90, va='center')
    plt.tick_params(axis='y', labelsize=17)
    plt.tick_params(axis='x', labelsize=17)

    plt.scatter(0, np.mean(params), color='w', edgecolor='k', lw=2, s=75, zorder=10)
    plt.scatter(1, np.mean(feature_params), color='w', edgecolor='k', lw=2, s=75, zorder=10)

    for i, p in enumerate(list_of_pvalues):
        if p < 0.05:
            # Get the position to place the star
            x = i  # The x position corresponds to the box index
            y = df.iloc[:, i].max() + 0.05  # Slightly above the max value of the box
            plt.text(x, y, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
    plt.tight_layout()

    plt.show()


def distance_analysis_seeg(behavior_path, data_path, condition, regions, first_parcel_label=True):
    """Takes all electrodes from a big region (ex: all temporal lobe) and selected the subset of electrodes that show a
    significant RMS contrast (in either novelty detection or retrieval), then run intra and inter stimulus correlation
    analysis, linking them to correct response and surprise"""

    ############## Intrastim correlation -> correct ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(data_path, allow_pickle=True)

    subject_id = []
    regression_param_correct = []
    regression_param_feature = []
    pvalues_array = []  # p value for the regression
    p_ttest_array = []  # p value of the contrast correct vs incorrect
    for s, subject in tqdm(enumerate(data["subjects"])):
        subject_stim_features = stim_features[stim_features['subject'] == subject]
        subject_info = data['info'][s]
        indices_to_keep = []
        correct_list = []
        feature_list = []
        distance_array = []
        stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
        distance_matrix = data['distance_matrices'][s]
        for i, stim in enumerate(data['stim_names'][s]):
            if stim in subject_stim_features['stim_name'].tolist():
                if 'rep2' in stim:
                    print(subject)
                    if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                        indices_to_keep.append(i)
                        if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                            rep1_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                            distance_array.append(distance_matrix[i, rep1_index])  # correlation of 2 reps
                            correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                            feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])

        distance_array = np.array(distance_array)

        # p_ttest = np.zeros(distance_array.shape[1])

        Y = distance_array
        X = np.zeros((Y.shape[0], 1))
        # X = np.zeros((Y.shape[0], 2))
        X[:, 0] = correct_list
        # X[:, 1] = np.array(feature_list)**2
        X = sm.add_constant(X)

        # # independent t-test
        # correct_corr = Y[X[:, 1] == 1]  # axis 0 is the intercept
        # incorrect_corr = Y[X[:, 1] == 0]
        # p_ttest[channel] = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=1000,
        #                              random_state=None,
        #                              alternative='greater').pvalue  # testing of individual electrodes
        # if p_ttest[channel] < 0.05:

        ols = sm.OLS(Y, X).fit()
        regression_param_correct.append(ols.params[1])
        pvalues_array.append(ols.pvalues[1])
        # regression_param[channel] = ols.params[2]

        X = np.zeros((Y.shape[0], 1))
        # X = np.zeros((Y.shape[0], 2))
        X[:, 0] = feature_list
        # X[:, 1] = np.array(feature_list)**2
        X = sm.add_constant(X)
        ols = sm.OLS(Y, X).fit()
        regression_param_feature.append(ols.params[1])

        subject_id.append(s)

        # p_ttest_array.append(p_ttest)

    regression_param_correct = np.array(regression_param_correct)
    print(regression_param_correct.shape)
    regression_param_feature = np.array(regression_param_feature)
    print(regression_param_feature.shape)

    print(pvalues_array)

    fig, axs = plt.subplots(1, 2, sharey=True)
    x_regression = ['correct', condition]
    for i, regression_param_array in enumerate([regression_param_correct, regression_param_feature]):
        print(np.mean(regression_param_array))
        ttest = scipy.stats.ttest_1samp(regression_param_array, 0, alternative='two-sided')
        print(scipy.stats.ttest_1samp(regression_param_array, 0, alternative='two-sided'))
        axs[i].hist(regression_param_array, bins=100)
        axs[i].axvline(0, ls='--', color='k')
        # plt.xlabel('Regression parameter (intrastimulus correlation ~ surprise)')
        axs[i].set_xlabel(f'Regression parameter (intrastimulus distance ~ {x_regression[i]})')
        axs[i].set_ylabel('Count (subject)')
        axs[i].text(plt.gca().get_xlim()[0], plt.gca().get_ylim()[1],
                 f'regression beta = {round(np.mean(regression_param_array), 5)}, p = {round(ttest.pvalue, 5)}')


    ############## Interstim correlation -> incorrect ##############

    stim_features = pd.read_csv(behavior_path)
    stim_features = stim_features[['stim_name', 'subject', condition, "correct"]]
    print(stim_features)

    data = np.load(data_path, allow_pickle=True)

    subject_id = []
    regression_param_correct = []
    regression_param_feature = []
    # pvalues_array = []  # p value for the regression
    # p_ttest_array = []  # p value of the contrast correct vs incorrect
    for s, subject in tqdm(enumerate(data["subjects"])):

        # if not subject == 'Bou_Ni':

        subject_stim_features = stim_features[stim_features['subject'] == subject]
        subject_info = data['info'][s]
        ch_names = subject_info['ch_names']
        indices_to_keep = []
        correct_list = []
        feature_list = []
        stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
        distance_matrix = data['distance_matrices'][s]  # keep all channels
        paired_stim = []
        for i, stim in enumerate(data['stim_names'][s]):
            if i > -1:  # remove first trial
                if stim in subject_stim_features['stim_name'].tolist():
                    if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                        indices_to_keep.append(i)
                        correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                        feature_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim][condition].iloc[0])
                        if 'rep2' in stim:
                            if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                                paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1')))
                            else:
                                paired_stim.append('no_pair')
                        if 'rep1' in stim:
                            if stim[9:].replace('rep1', 'rep2') in stim_names_without_trial_nb:
                                paired_stim.append(stim_names_without_trial_nb.index(stim[9:].replace('rep1', 'rep2')))
                            else:
                                paired_stim.append('no_pair')
                        # else:
                        #     raise ValueError('Neither rep1 nor rep2???')

        condition_matrix = distance_matrix[indices_to_keep, :]  # select condition
        condition_matrix = condition_matrix[:, indices_to_keep]  # select condition
        distance_array = np.zeros(condition_matrix.shape[0])
        for i in range(condition_matrix.shape[0]):
            current_corr = condition_matrix[i, :]
            mask = np.ones(condition_matrix.shape[1], dtype=bool)
            # mask[indices_to_keep[i]] = False  # remove current stim
            mask[i] = False  # remove current stim
            # if not paired_stim[i] == 'no_pair':
            #     mask[paired_stim[i]] = False  # remove pair
            # for j in range(indices_to_keep[i], mask.shape[0]):
            #     mask[j] = False  # remove following trials

            # plt.hist(current_corr[mask, 10], bins=100)
            # plt.show()
            # print(np.mean(current_corr[mask, :], axis=0))
            distance_array[i] = np.mean(current_corr[mask], axis=0)
        # plt.plot(distance_array)
        # plt.show()

        # pvalues = np.zeros(correlations_array.shape[1])
        # p_ttest = np.zeros(correlations_array.shape[1])
        Y = distance_array
        X = np.zeros((Y.shape[0], 1))
        # X = np.zeros((Y.shape[0], 2))
        X[:, 0] = correct_list
        # X[:, 1] = np.array(feature_list)**2
        X = sm.add_constant(X)

        # # independent t-test
        # correct_corr = Y[X[:, 1] == 1]  # axis 0 is the intercept
        # incorrect_corr = Y[X[:, 1] == 0]
        # p_ttest[channel] = ttest_ind(correct_corr, incorrect_corr, axis=0, equal_var=False, permutations=1000,
        #                              random_state=None,
        #                              alternative='less').pvalue  # testing of individual electrodes
        # if p_ttest[channel] < 0.05:

        ols = sm.OLS(Y, X).fit()
        regression_param_correct.append(ols.params[1])
        # pvalues[channel] = ols.pvalues[1]
        # regression_param[channel] = ols.params[2]

        X = np.zeros((Y.shape[0], 1))
        # X = np.zeros((Y.shape[0], 2))
        X[:, 0] = feature_list
        # X[:, 1] = np.array(feature_list)**2
        X = sm.add_constant(X)
        ols = sm.OLS(Y, X).fit()
        regression_param_feature.append(ols.params[1])

        subject_id.append(s)
        # pvalues_array.append(pvalues)
        # p_ttest_array.append(p_ttest)

    regression_param_correct = np.array(regression_param_correct)
    print(regression_param_correct.shape)
    regression_param_feature = np.array(regression_param_feature)
    print(regression_param_feature.shape)
    # pvalues_array = np.concatenate(pvalues_array)
    # print(np.sum(pvalues_array < 0.05), 'significant electrodes,',
    #       round(np.sum(pvalues_array < 0.05) / pvalues_array.shape[0] * 100, 3), "%")
    # print(regression_param_array[pvalues_array < 0.05])
    # p_ttest_array = np.concatenate(p_ttest_array)
    # print(np.sum(p_ttest_array < 0.05), 'significant electrodes with independent ttest,',
    #       round(np.sum(p_ttest_array < 0.05) / p_ttest_array.shape[0] * 100, 3), "%")
    # print(regression_param_array[p_ttest_array < 0.05])

    fig, axs = plt.subplots(1, 2, sharey=True)
    x_regression = ['correct', condition]
    for i, regression_param_array in enumerate([regression_param_correct, regression_param_feature]):
        print(np.mean(regression_param_array))
        ttest = scipy.stats.ttest_1samp(regression_param_array, 0, alternative='two-sided')
        print(scipy.stats.ttest_1samp(regression_param_array, 0, alternative='two-sided'))
        axs[i].hist(regression_param_array, bins=100)
        axs[i].axvline(0, ls='--', color='k')
        # plt.xlabel('Regression parameter (intrastimulus correlation ~ surprise)')
        axs[i].set_xlabel(f'Regression parameter (interstimulus distance ~ {x_regression[i]})')
        axs[i].set_ylabel('Count (subject)')
        axs[i].text(plt.gca().get_xlim()[0], plt.gca().get_ylim()[1],
                    f'regression beta = {round(np.mean(regression_param_array), 5)}, p = {round(ttest.pvalue, 5)}')

    plt.show()


def save_csv_similarity_seeg(data_path, stim_features_csv, save_path):

    stim_features = pd.read_csv(stim_features_csv)
    stim_features = stim_features[['stim_name', 'subject', 'surprise_mean_paulrnn', 'surprise_possum_paulrnn', "correct"]]
    # stim_features[condition] = (stim_features[condition] - stim_features[condition].mean()) / stim_features[condition].std()
    print(stim_features)

    data = np.load(data_path, allow_pickle=True)

    channel_positions_list = []
    subject_id = []
    channel_id = []
    correct_list = []
    surprise_list = []
    positive_surprise_list = []
    correlations_list = []
    onset_list = []
    rate_list = []
    for s, subject in tqdm(enumerate(data["subjects"])):
        subject_stim_features = stim_features[stim_features['subject'] == subject]
        indices_to_keep = []
        stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
        similarity_matrix = data['similarity_matrices'][s]  # keep all channels
        for c in range(similarity_matrix.shape[-1]):
            for i, stim in enumerate(data['stim_names'][s]):
                if 'rep2' in stim:
                    if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                        indices_to_keep.append(i)
                        if stim[9:].replace('rep2', 'rep1') in stim_names_without_trial_nb:
                            rep1_index = stim_names_without_trial_nb.index(stim[9:].replace('rep2', 'rep1'))
                            correlations_list.append(similarity_matrix[i, rep1_index, c])  # correlation of 2 reps
                            correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                            surprise_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["surprise_mean_paulrnn"].iloc[0])
                            positive_surprise_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["surprise_possum_paulrnn"].iloc[0])
                            subject_id.append(subject)
                            channel_id.append(c)
                            onset_list.append(stim.split('_')[4][:6])
                            rate_list.append(float(stim[14:17]))

    # save csv for R analysis
    csv = pd.DataFrame({'subject': subject_id, 'channel': channel_id, 'correct': correct_list, 'onset': onset_list, 'rate': rate_list, 'surprise': surprise_list, 'positive_surprise': positive_surprise_list, 'correlation': correlations_list})
    csv.to_csv(os.path.join(save_path, Path(data_path).stem + '_similarity.csv'))


def save_csv_dissimilarity_seeg(data_path, stim_features_csv, save_path):

    stim_features = pd.read_csv(stim_features_csv)
    stim_features = stim_features[['stim_name', 'subject', 'surprise_mean_paulrnn', 'surprise_possum_paulrnn', "correct"]]
    # stim_features[condition] = (stim_features[condition] - stim_features[condition].mean()) / stim_features[condition].std()
    print(stim_features)

    data = np.load(data_path, allow_pickle=True)

    channel_positions_list = []
    subject_id = []
    channel_id = []
    correct_list = []
    rep_list = []
    surprise_list = []
    positive_surprise_list = []
    correlations_list = []
    onset_list = []
    rate_list = []
    for s, subject in tqdm(enumerate(data["subjects"])):
        subject_stim_features = stim_features[stim_features['subject'] == subject]
        indices_to_keep = []
        stim_names_without_trial_nb = [data['stim_names'][s][i][9:] for i in range(len(data['stim_names'][s]))]
        similarity_matrix = data['similarity_matrices'][s]  # keep all channels
        subject_correct_list = []
        subject_rep_list = []
        subject_surprise_list = []
        subject_positive_surprise_list = []
        subject_onset_list = []
        subject_rate_list = []
        for i, stim in enumerate(data['stim_names'][s]):
            if stim in subject_stim_features['stim_name'].tolist():
                if not np.isnan(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0]):
                    indices_to_keep.append(i)
                    subject_rep_list.append(int(stim[12]))
                    subject_correct_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["correct"].iloc[0])
                    subject_surprise_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["surprise_mean_paulrnn"].iloc[0])
                    subject_positive_surprise_list.append(subject_stim_features[subject_stim_features['stim_name'] == stim]["surprise_possum_paulrnn"].iloc[0])
                    subject_onset_list.append(stim.split('_')[4][:6])
                    subject_rate_list.append(float(stim[14:17]))

        condition_matrix = similarity_matrix[indices_to_keep, :, :]  # select condition
        condition_matrix = condition_matrix[:, indices_to_keep, :]  # select condition
        correlations_array = np.zeros((condition_matrix.shape[0], condition_matrix.shape[2]))
        for i in range(condition_matrix.shape[0]):
            current_corr = condition_matrix[i, :, :]  # ignore intra stim correlation
            mask = np.ones(condition_matrix.shape[1], dtype=bool)
            mask[i] = False
            correlations_array[i, :] = np.mean(current_corr[mask, :], axis=0)
        for c in range(correlations_array.shape[-1]):
            correlations_list += correlations_array[:, c].tolist()
            correct_list += subject_correct_list
            surprise_list += subject_surprise_list
            positive_surprise_list += subject_positive_surprise_list
            rep_list += subject_rep_list
            onset_list += subject_onset_list
            rate_list += subject_rate_list
            subject_id += [subject for _ in range(len(indices_to_keep))]
            channel_id += [c for _ in range(len(indices_to_keep))]

    # save csv for R analysis
    csv = pd.DataFrame({'subject': subject_id, 'channel': channel_id, 'correct': correct_list, 'rep': rep_list, 'onset': onset_list, 'rate': rate_list, 'surprise': surprise_list, 'positive_surprise': positive_surprise_list, 'correlation': correlations_list})
    csv.to_csv(os.path.join(save_path, Path(data_path).stem + '_dissimilarity.csv'))


def channel_parcellation_to_csv(rms_path, regions_to_remove, output_folder, first_parcel_label=True):
    """Save a csv file containing the number of channels per region in the parcellation."""
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    del subject_paths[1]
    print(subject_paths)

    all_subjects_parcellations = []

    for i, subject_path in enumerate(tqdm(subject_paths)):
        rms_obj = np.load(subject_path, allow_pickle=True)
        parcellations = rms_obj['ch_parcellations']
        # print(len(parcellations))
        if first_parcel_label == True:
            parcellations = [[parcel[0]] for parcel in parcellations]
        elif first_parcel_label == "advanced":
            go_next = ['Unknown', 'ctx-rh-unknown', 'ctx-lh-unknown']
            parcellations = select_parcel(parcellations, go_next)

        parcellations = [parcel[0] for parcel in parcellations]
        print(parcellations)
        all_subjects_parcellations += parcellations

    print(all_subjects_parcellations)
    print(len(all_subjects_parcellations))

    # Filter out unwanted regions
    filtered_all_subjects_parcellations = [parcel for parcel in all_subjects_parcellations if parcel not in regions_to_remove]
    print(len(filtered_all_subjects_parcellations))

    # count in regions of interest
    regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal',
               'ctx-rh-superiortemporal', 'ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus',
               'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal',
               'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal',
               'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal']  # all temporal lobe
    # regions = ['Left-Hippocampus', 'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole',
    #            'ctx-lh-inferiortemporal', 'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala',
    #            'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal',
    #            'ctx-rh-entorhinal']  # medial temporal
    all_subjects_parcellations_regions = [parcel for parcel in filtered_all_subjects_parcellations if parcel in regions]
    print('Number of channels in the temporal lobes:', len(all_subjects_parcellations_regions))

    # Count occurrences of each region
    region_counts = Counter(all_subjects_parcellations)
    # region_counts = Counter(filtered_all_subjects_parcellations)

    # Prepare data for a CSV file
    df = pd.DataFrame(list(region_counts.items()), columns=['Region', 'Channel Count'])

    # Output the CSV file (avoiding external dependency on `openpyxl`)
    csv_output_path = os.path.join(output_folder, 'channel_parcellations.csv')
    df.to_csv(csv_output_path, index=False)

    print(f"CSV file with channel parcellation table saved at: {csv_output_path}")


def number_of_shafts_per_subject(rms_path):
    path = Path(rms_path)
    subject_paths = sorted(list(path.glob(f'*epochs_rms.pkl')))
    subject_names = []
    del subject_paths[1]
    print(subject_paths)

    n_shafts = []
    for i, subject_path in enumerate(tqdm(subject_paths)):
        rms_obj = np.load(subject_path, allow_pickle=True)
        info = rms_obj['info']

        seeg_chs = mne.pick_types(info, seeg=True)
        ch_names = [info['ch_names'][i] for i in seeg_chs]

        shafts = defaultdict(list)

        for ch in ch_names:
            # remove trailing numbers (e.g. A12 → A)
            shaft = re.sub(r'\d+$', '', ch)
            shafts[shaft].append(ch)

        print(f"Number of shafts: {len(shafts)}")
        n_shafts.append(len(shafts))
        # for shaft, contacts in shafts.items():
        #     print(shaft, len(contacts))
    print('Mean n shafts:', np.median(n_shafts))


if __name__ == '__main__':

    '''average_similarity'''
    # conditions = ['rep1', 'rep2', 'rep3']
    conditions = ['piano0', 'piano2']
    # conditions = None
    channels = None
    # channels = ['MEG0341', 'MEG0321', 'MEG0331', 'MEG0211', 'MEG0221', 'MEG0231', 'MEG0241', 'MEG1611', 'MEG1621', 'MEG0411', 'MEG0441', 'MEG1811']
    # channels = ['MEG1221', 'MEG1231', 'MEG1241', 'MEG1121', 'MEG1311', 'MEG1321']
    # average_similarity('/home/mathieupvc/These/data/meg_maestro/representation_analysis/similarity_matrix.pkl', conditions=conditions, channels=channels)

    '''average_similarity_features'''
    data_path = '/home/mathieupvc/These/data/meg_maestro/representation_analysis/similarity_matrix.pkl'
    stim_features_csv = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/stim_predictive_features.csv'
    condition = 'surprise'
    channels = None
    # channels = ['MEG0341', 'MEG0321', 'MEG0331', 'MEG0211', 'MEG0221', 'MEG0231', 'MEG0241', 'MEG1611', 'MEG1621', 'MEG0411', 'MEG0441', 'MEG1811']
    n_bins = 5
    # average_similarity_features(data_path, stim_features_csv, condition, n_bins=n_bins, channels=channels)

    '''topo_similarity'''
    # conditions = ['rep1', 'rep2', 'rep3']
    conditions = ['piano0', 'piano2']
    # conditions = None
    # topo_similarity('/home/mathieupvc/These/data/meg_maestro/representation_analysis/similarity_matrix.pkl', conditions=conditions)

    '''topo_ushape_similarity'''
    # topo_ushape_similarity('/home/mathieupvc/These/data/meg_maestro/representation_analysis/similarity_matrix.pkl')

    '''topo_feature_similarity'''
    data_path = '/home/mathieupvc/These/data/meg_maestro/representation_analysis/similarity_matrix.pkl'
    stim_features_csv = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/stim_predictive_features.csv'
    condition = 'timing_predictability'
    # topo_feature_similarity(data_path, stim_features_csv, condition)

    '''topo_feature_repetition_similarity'''
    data_path = '/home/mathieupvc/These/data/meg_maestro/representation_analysis/similarity_matrix.pkl'
    # data_path = '/home/mathieupvc/These/data/meg_maestro/representation_analysis/stim_amplitude.pkl'
    stim_features_csv = '/home/mathieupvc/These/data/meg_maestro/behavior_analysis/stim_predictive_features.csv'
    condition = 'surprise'
    # condition = 'pitch_entropy'
    # condition = 'timing_predictability'
    # topo_feature_repetition_similarity(data_path, stim_features_csv, condition)

    '''save_similarity_matrix_seeg'''
    data_path = '/home/mathieu/These/data/meg_maestro/seeg_trials'
    save_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg'
    # regions = ['Left-Hippocampus', 'Right-Hippocampus']  # hippocampus
    # regions = ['Left-Hippocampus']  # left hippocampus
    # regions = ['Left-Amygdala', 'Right-Amygdala']  # amygdala
    # regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal', 'ctx-rh-superiortemporal']  # superior temporal
    # regions = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal']  # superior temporal no heschl
    # regions = ['ctx-lh-transversetemporal', 'ctx-rh-transversetemporal']  # heschl
    regions = ['Left-Hippocampus', 'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole',
               'ctx-lh-inferiortemporal', 'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala',
               'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal']  # medial temporal
    # regions = ['Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole',
    #                     'ctx-lh-inferiortemporal', 'ctx-lh-entorhinal', 'Right-Amygdala',
    #                     'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal']  # medial temporal no amygdala
    # regions = ['ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-entorhinal',
    #            'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-entorhinal']  # medial temporal reduced (no hippocampus no amygdala, no inferior temporal)
    # regions = ['ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal']  # inferior temporal
    # regions = ['ctx-lh-middletemporal', 'ctx-rh-middletemporal']  # middle temporal
    # regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal',
    #            'ctx-rh-superiortemporal', 'ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus',
    #            'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal',
    #            'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal',
    #            'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal']  # temporal lobe
    # regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis', 'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal',
    #            'ctx-rh-rostralmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-rh-parstriangularis', 'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-lateralorbitofrontal']  # prefrontal
    regions = None
    # save_similarity_matrix_seeg(data_path, save_path, regions=regions, n_jobs=-1)

    '''save_trajectory_distance_matrix_seeg'''
    data_path = '/home/mathieu/These/data/meg_maestro/seeg_trials'
    save_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg'
    # save_trajectory_distance_matrix_seeg(data_path, save_path, regions=regions, n_jobs=-1)

    '''feature_repetition_similarity_seeg'''
    # data_path = '/home/mathieupvc/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix.pkl'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_left_hippocampus.pkl'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_channels_Left-Hippocampus_1-7.pkl'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_hippo.pkl'
    stim_features_csv = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    condition = 'surprise_mean_paulrnn'
    # condition = 'surprise_possum_paulrnn'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_heschl.pkl'
    # feature_repetition_similarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_superior_temporal.pkl'
    # feature_repetition_similarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_superior_temporal_no_heschl.pkl'
    # feature_repetition_similarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_middle_temporal.pkl'
    # feature_repetition_similarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_inferior_temporal.pkl'
    # feature_repetition_similarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_amygdala.pkl'
    # feature_repetition_similarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal_reduced.pkl'
    # feature_repetition_similarity_seeg(data_path, stim_features_csv, condition)
    data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal.pkl'
    # feature_repetition_similarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_hippocampus.pkl'
    # feature_repetition_similarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_channels_Left-Hippocampus.pkl'
    # feature_repetition_similarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_temporal_lobe.pkl'
    # feature_repetition_similarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_prefrontal.pkl'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal_no_hippocampus.pkl'
    # feature_repetition_similarity_seeg(data_path, stim_features_csv, condition)

    '''repetition_similarity_and_correct_seeg'''
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_heschl.pkl'
    # repetition_similarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_superior_temporal.pkl'
    # repetition_similarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_superior_temporal_no_heschl.pkl'
    # repetition_similarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_middle_temporal.pkl'
    # repetition_similarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_inferior_temporal.pkl'
    # repetition_similarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_amygdala.pkl'
    # repetition_similarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal_reduced.pkl'
    # repetition_similarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal.pkl'
    # repetition_similarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_hippocampus.pkl'
    # repetition_similarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_channels_Left-Hippocampus.pkl'
    # repetition_similarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_temporal_lobe.pkl'
    # repetition_similarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_prefrontal.pkl'
    # repetition_similarity_and_correct_seeg(data_path, stim_features_csv)

    '''feature_dissimilarity_seeg'''
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_left_hippocampus.pkl'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_channels_Left-Hippocampus_1-7.pkl'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_hippocampus.pkl'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_superior_temporal.pkl'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal.pkl'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_temporal_lobe.pkl'
    # stim_features_csv = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    condition = 'surprise_mean_paulrnn'
    # condition = 'surprise_possum_paulrnn'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_heschl.pkl'
    # feature_dissimilarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_superior_temporal.pkl'
    # feature_dissimilarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_superior_temporal_no_heschl.pkl'
    # feature_dissimilarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_middle_temporal.pkl'
    # feature_dissimilarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_inferior_temporal.pkl'
    # feature_dissimilarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_amygdala.pkl'
    # feature_dissimilarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal_reduced.pkl'
    # feature_dissimilarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal.pkl'
    # feature_dissimilarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_hippocampus.pkl'
    # feature_dissimilarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_channels_Left-Hippocampus.pkl'
    # feature_dissimilarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_temporal_lobe.pkl'
    # feature_dissimilarity_seeg(data_path, stim_features_csv, condition)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_prefrontal.pkl'
    # feature_dissimilarity_seeg(data_path, stim_features_csv, condition)

    '''topo_ushape_similarity for sEEG'''
    # topo_ushape_similarity_seeg('/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_left_hippocampus.pkl')

    '''repetition_dissimilarity_and_correct_seeg'''
    stim_features_csv = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_heschl.pkl'
    # repetition_dissimilarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_superior_temporal.pkl'
    # repetition_dissimilarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_superior_temporal_no_heschl.pkl'
    # repetition_dissimilarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_middle_temporal.pkl'
    # repetition_dissimilarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_inferior_temporal.pkl'
    # repetition_dissimilarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_amygdala.pkl'
    # repetition_dissimilarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal_reduced.pkl'
    # repetition_dissimilarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal.pkl'
    # repetition_dissimilarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_hippocampus.pkl'
    # repetition_dissimilarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_channels_Left-Hippocampus.pkl'
    # repetition_dissimilarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_temporal_lobe.pkl'
    # repetition_dissimilarity_and_correct_seeg(data_path, stim_features_csv)
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_prefrontal.pkl'
    # repetition_dissimilarity_and_correct_seeg(data_path, stim_features_csv)

    '''full_representation_analysis_seeg'''
    rms_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels'
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_temporal_lobe.pkl'
    # similarity_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal.pkl'
    # distance_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal.pkl'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal_reduced.pkl'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_hippocampus.pkl'
    condition = 'surprise_mean_paulrnn'
    # condition = 'surprise_possum_paulrnn'
    first_parcel_label = "advanced"
    regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal',
               'ctx-rh-superiortemporal', 'ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus',
               'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal',
               'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal',
               'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal']  # all temporal lobe
    # regions = ['ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-entorhinal',
    #            'ctx-rh-parahippocampal', 'ctx-rh-temporalpole',
    #            'ctx-rh-entorhinal']  # medial temporal reduced (no hippocampus no amygdala, no inferior temporal)
    # full_representation_analysis_seeg(rms_path, behavior_path, data_path, condition, regions, first_parcel_label=first_parcel_label)

    regions = ['Left-Hippocampus', 'Right-Hippocampus']  # hippocampus
    # regions = ['Left-Hippocampus']  # left hippocampus
    # regions = ['Right-Hippocampus']  # left hippocampus
    # regions = ['Left-Amygdala', 'Right-Amygdala']  # amygdala
    # regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal', 'ctx-rh-superiortemporal']  # superior temporal
    # regions = ['ctx-lh-superiortemporal', 'ctx-rh-superiortemporal']  # superior temporal no heschl
    # regions = ['ctx-lh-transversetemporal', 'ctx-rh-transversetemporal']  # heschl
    # regions = ['Left-Hippocampus', 'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole',
    #            'ctx-lh-inferiortemporal', 'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala',
    #            'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal',
    #            'ctx-rh-entorhinal']  # medial temporal
    regions = ['ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus',
               'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal',
               'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal',
               'ctx-rh-inferiortemporal']  # regions of temporal lobe that show a power contrast
    # regions = ['Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole',
    #                     'ctx-lh-inferiortemporal', 'ctx-lh-entorhinal', 'Right-Amygdala',
    #                     'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal']  # medial temporal no amygdala
    # regions = ['ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-entorhinal',
    #            'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-entorhinal']  # medial temporal reduced (no hippocampus no amygdala, no inferior temporal)
    # regions = ['ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal']  # inferior temporal
    # regions = ['ctx-lh-middletemporal', 'ctx-rh-middletemporal']  # middle temporal
    # regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal',
    #            'ctx-rh-superiortemporal', 'ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus',
    #            'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal',
    #            'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal',
    #            'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal']  # temporal lobe
    # regions = ['ctx-lh-rostralmiddlefrontal', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parstriangularis', 'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-lateralorbitofrontal',
    #            'ctx-rh-rostralmiddlefrontal', 'ctx-rh-caudalmiddlefrontal', 'ctx-rh-parstriangularis', 'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-lateralorbitofrontal']  # prefrontal
    similarity_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_all_electrodes.pkl'
    distance_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/dtw_distances/dtw_distance_matrix.pkl'
    # full_representation_analysis_seeg(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=first_parcel_label, permutations=0)

    '''save_full_representation_stat_individual_electrode'''
    save_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/corr_and_distance'
    # save_full_representation_stat_individual_electrode(save_path, behavior_path, similarity_path, distance_path, condition)

    '''representation_distance_and_correct_scatter'''
    similarity_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_all_electrodes.pkl'
    distance_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/dtw_distances/dtw_distance_matrix.pkl'
    # representation_distance_and_correct_scatter(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=first_parcel_label)

    # representation_distance_distributions(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=first_parcel_label)

    '''intra_and_interstim_and_correct'''
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal.pkl'
    intra_and_interstim_and_correct(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=first_parcel_label)

    '''impact_of_first_trials_to_remove'''
    # impact_of_first_trials_to_remove(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=first_parcel_label)

    '''full_dual_mechanism_model'''
    # full_dual_mechanism_model(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=first_parcel_label)
    # full_dual_mechanism_model_R2_threshold_impact(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=first_parcel_label)
    # full_dual_mechanism_model_linear_exp_comparison(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=first_parcel_label)
    # full_dual_mechanism_model_separated_channels(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=first_parcel_label, channel_selection='weight')
    # only_separability_dual_mechanism(rms_path, behavior_path, distance_path, condition, regions, first_parcel_label=first_parcel_label, channel_selection='none')

    '''regress_correct_on_suprise_and_neural_patterns'''
    # regress_correct_on_suprise_and_neural_patterns(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=first_parcel_label)

    '''linear_vs_exp_by_region'''
    regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal',
               'ctx-rh-superiortemporal', 'ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus',
               'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal',
               'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal',
               'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal', 'ctx-rh-entorhinal']  # all temporal lobe
    # linear_vs_exp_by_region(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=first_parcel_label)

    '''intra_and_interstim_save_for_decoding'''
    save_path = "/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/corr_and_distance"
    # intra_and_interstim_save_for_decoding(save_path, rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=first_parcel_label)

    '''population_overlaps_analysis'''
    regions = None
    regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-lh-middletemporal', 'ctx-lh-inferiortemporal', 'ctx-lh-temporalpole', 'ctx-lh-entorhinal', 'ctx-lh-parahippocampal', 'Left-Hippocampus', 'Left-Amygdala',
               'ctx-rh-transversetemporal', 'ctx-rh-superiortemporal', 'ctx-rh-middletemporal', 'ctx-rh-inferiortemporal', 'ctx-rh-temporalpole', 'ctx-rh-entorhinal', 'ctx-rh-parahippocampal', 'Right-Hippocampus', 'Right-Amygdala']
    # population_overlaps_analysis(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label=first_parcel_label)

    '''save_similarity_matrix_time_windows_seeg'''
    data_path = '/home/mathieu/These/data/meg_maestro/seeg_trials'
    save_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg'
    regions = None
    # save_similarity_matrix_time_windows_seeg(data_path, save_path, regions=regions, n_jobs=-1)

    '''full_representation_analysis_by_region'''
    # regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-lh-middletemporal',
    #            'ctx-lh-inferiortemporal', 'ctx-lh-temporalpole', 'ctx-lh-entorhinal', 'ctx-lh-parahippocampal',
    #            'Left-Hippocampus', 'Left-Amygdala',
    #            'ctx-rh-transversetemporal', 'ctx-rh-superiortemporal', 'ctx-rh-middletemporal',
    #            'ctx-rh-inferiortemporal', 'ctx-rh-temporalpole', 'ctx-rh-entorhinal', 'ctx-rh-parahippocampal',
    #            'Right-Hippocampus', 'Right-Amygdala']  # temporal lobe
    regions = ['ctx-lh-middletemporal', 'ctx-rh-middletemporal', 'Left-Hippocampus',
               'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole', 'ctx-lh-inferiortemporal',
               'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala', 'ctx-rh-parahippocampal',
               'ctx-rh-inferiortemporal']  # regions of temporal lobe that show a power contrast
    # full_representation_analysis_by_region(rms_path, behavior_path, similarity_path, distance_path, condition, regions, first_parcel_label='advanced', remove_outliers=True)

    '''representation_analysis_windows_seeg'''
    rms_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels'
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    similarity_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_all_electrodes.pkl'
    similarity_windows_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_windows_all_electrodes.pkl'
    distance_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/dtw_distances/dtw_distance_matrix.pkl'
    distance_windows_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/dtw_distances/dtw_distance_matrix_windows.pkl'
    # regions = ['Left-Hippocampus', 'Left-Amygdala', 'ctx-lh-parahippocampal', 'ctx-lh-temporalpole',
    #            'ctx-lh-inferiortemporal', 'ctx-lh-entorhinal', 'Right-Hippocampus', 'Right-Amygdala',
    #            'ctx-rh-parahippocampal', 'ctx-rh-temporalpole', 'ctx-rh-inferiortemporal',
    #            'ctx-rh-entorhinal']  # medial temporal
    # regions = ['Left-Hippocampus', 'Right-Hippocampus']
    # regions = ['ctx-lh-transversetemporal', 'ctx-lh-superiortemporal', 'ctx-rh-transversetemporal',
    #            'ctx-rh-superiortemporal',]
    # regions = ['ctx-lh-middletemporal', 'ctx-rh-middletemporal']
    # regions = ['ctx-lh-inferiortemporal', 'ctx-rh-inferiortemporal']
    # regions = ['ctx-lh-temporalpole', 'ctx-rh-temporalpole']
    condition = 'surprise_mean_paulrnn'
    first_parcel_label = "advanced"
    # representation_analysis_windows_seeg(rms_path, behavior_path, similarity_path, similarity_windows_path, distance_path, distance_windows_path, condition, regions, first_parcel_label=first_parcel_label)

    '''representation_analysis_seeg_on_rms'''
    # representation_analysis_seeg_on_rms(rms_path, behavior_path, condition, regions, first_parcel_label=first_parcel_label)

    '''distance_analysis_seeg'''
    data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/distance_matrix.pkl'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/distance_matrix_medial_temporal.pkl'
    # distance_analysis_seeg(behavior_path, data_path, condition, regions, first_parcel_label=True)

    '''save_csv_similarity_seeg'''
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_medial_temporal.pkl'
    # data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_hippocampus.pkl'
    data_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/similarity_matrix_channels_Left-Hippocampus.pkl'
    stim_features_csv = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    save_path = '/home/mathieu/These/data/meg_maestro/representation_analysis/seeg/csv_correlations'
    # save_csv_similarity_seeg(data_path, stim_features_csv, save_path)

    '''save_csv_dissimilarity_seeg'''
    # save_csv_dissimilarity_seeg(data_path, stim_features_csv, save_path)

    '''channel_parcellation_to_csv'''
    rms_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels'
    regions_to_remove = ['Right-Cerebral-White-Matter', 'Left-Cerebral-White-Matter', 'CSF', 'Unknown']
    output_excel_folder = '/home/mathieu/These/data/meg_maestro/seeg_visualizations/seeg_channel_parcellations'
    first_parcel_label = "advanced"
    # channel_parcellation_to_csv(rms_path, regions_to_remove, output_excel_folder, first_parcel_label=first_parcel_label)

    '''number_of_shafts_per_subject'''
    # number_of_shafts_per_subject('/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels')
