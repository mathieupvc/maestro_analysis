from pathlib import Path
import os

import numpy as np
import scipy.stats
from itertools import product

from tqdm import tqdm

import mne
from mne.stats import permutation_t_test, permutation_cluster_test, permutation_cluster_1samp_test
from mne.channels import find_ch_adjacency

import matplotlib.pyplot as plt
import seaborn as sns

# from pingouin import ttest


def trf_nb_signif_channels_hist(data_path, scores_path, trf_path, channel_type='grad', stat_test='permut', save_format='png'):
    '''scores_path: epochs scores'''

    scores = np.load(scores_path, allow_pickle=True)

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    nb_signif_channels = [[], [], []]
    for i, subject_path in enumerate(sorted(subject_folders)):
        print(subject_path)
        raw_path = list(subject_path.glob('**/*raw_sss.fif'))[0]
        raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
        raw.pick(['meg'])

        channels_to_keep = mne.channel_indices_by_type(raw.info)[channel_type]

        # Find channels with significant score
        p_am = test_channels_significance(scores[i, 0][:, channels_to_keep], info=raw.info, test=stat_test,
                                          channel_type=channel_type)  # Permutation across epochs
        p_a = test_channels_significance(scores[i, 1][:, channels_to_keep], info=raw.info, test=stat_test,
                                         channel_type=channel_type)
        p_diff = test_channels_significance(scores[i, 0][:, channels_to_keep] - scores[i, 1][:, channels_to_keep],
                                            info=raw.info, test=stat_test, channel_type=channel_type)

        p_new = []
        if channel_type == 'grad':  # combine p_values of the 2 gradiometers of a same channel
            for p in [p_am, p_a, p_diff]:
                p = p.reshape((len(p) // 2, 2))
                p = np.minimum(p[:, 0], p[:, 1])  # take the lowest p_value
                p_new.append(p)
        else:
            p_new = [p_am, p_a, p_diff]
        for j, p in enumerate(p_new):
            nb_signif_channels[j].append(np.sum(p <= 0.025))

    # plot histograms
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 12), dpi=300, sharex=True, sharey=True)
    for i in range(len(axes)):
        sns.histplot(nb_signif_channels[i], binwidth=1, ax=axes[i])

    plt.savefig(os.path.join(trf_path, 'nb_signif_channels_hist_' + channel_type + '.' + save_format))

    return


def test_channels_significance(data, info=None, test='permut', channel_type='grad', reduce_grad=True, tail=1, return_t_obs=False):
    '''
    data: shape (observations, channels)
    info: mne Info object (contains channels locations), for test='permut_cluster'
    test: {'permut', 'permut_cluster'}
    '''

    # TODO: check tail and p-values (in trf_topomap)

    if test == 'permut':
        t_obs, p, H0 = permutation_t_test(data, tail=tail, n_jobs=-1)  # Permutation across subjects
    elif test == 'permut_cluster':
        assert info is not None
        adjacency, ch_names = find_ch_adjacency(info, ch_type=channel_type)
        # TODO check threshold
        df = data.shape[0] - 1  # degrees of freedom for the test
        if tail==0:
            pval = 0.05
            thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
        elif tail==1:
            pval = 0.025  # one-tail
            thresh = scipy.stats.t.ppf(1 - pval, df)  # one-tailed, t distribution
        else:
            raise ValueError("tail should be in {1, 0}")
        # F_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(data, n_permutations=2000, tail=0, adjacency=adjacency, n_jobs=-1, out_type='indices', max_step=1, threshold=dict(start=0, step=0.2))
        t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(data, n_permutations=2000, tail=tail, adjacency=adjacency, n_jobs=-1, out_type='indices', max_step=1, threshold=thresh)
        p = np.ones(data.shape[1])  # Initialize p_values to one
        for cluster, p_value in zip(clusters, cluster_pv):
            # p[cluster] = np.minimum(p[cluster], cluster_pv[i] * np.ones_like(p[cluster])) # TODO: check
            p[cluster] = p_value

    else:
        raise ValueError("test should be in {'permut', 'permut_cluster'}")

    if (channel_type == 'grad') and reduce_grad:  # combine p_values of the 2 gradiometers of a same channel
        p = p.reshape((len(p) // 2, 2))
        p = np.minimum(p[:, 0], p[:, 1])  # take the lowest p_value

    if return_t_obs:
        return p, t_obs
    else:
        return p


def get_significant_channels_names(data_path, scores_path, channel_type='grad', stat_test='permut'):

    scores = np.load(scores_path, allow_pickle=True)

    path = Path(data_path)
    subject_folders = [x for x in path.iterdir() if x.is_dir()]

    # Plot average trf topomap
    raw_path = list(subject_folders[0].glob('**/*raw_sss.fif'))[0]
    raw = mne.io.read_raw_fif(raw_path, preload=False)  # load the raw file to get the Info object
    raw.pick(['meg'])
    channels_to_keep = mne.channel_indices_by_type(raw.info)[channel_type]
    averaged_epochs_scores = np.swapaxes(np.array([[np.mean(scores[i,j], axis=0) for j in range(scores.shape[1])] for i in range(scores.shape[0])]), 1, 2)  # Average over epochs for each participant
    raw.pick([channel_type])

    # Find channels with significant score
    p_diff = test_channels_significance(
        averaged_epochs_scores[:, channels_to_keep, 0] - averaged_epochs_scores[:, channels_to_keep, 1],
        info=raw.info, test=stat_test, channel_type=channel_type, reduce_grad=False
    )

    # define the statistical threshold (alpha)
    alpha = 0.05

    return p_diff <= alpha


def t_test_trf_kernels(coef_list, stat_test='ttest'):
    '''
    t_test between the first condition and each other ones
    :param coef_list: length equal to the number of conditions, each element is a numpy array of shape (nb_subjects, nb_trf_features, time_points)
    :return:
    '''

    significant_range = [np.zeros((coef_list[0].shape[1], coef_list[0].shape[2])) for _ in range(len(coef_list) - 1)]

    if stat_test == 'ttest':
        for condition in range(1, len(coef_list)):
            for feature in range(coef_list[condition].shape[1]):
                for time in range(coef_list[condition].shape[2]):
                    stats = ttest(coef_list[condition][:, feature, time], coef_list[0][:, feature, time], paired=True, alternative='two-sided')
                    significant_range[condition-1][feature, time] = stats['p-val']

    if stat_test == 'ttest_bonf':
        for condition in range(1, len(coef_list)):
            for feature in range(coef_list[condition].shape[1]):
                for time in range(coef_list[condition].shape[2]):
                    stats = ttest(coef_list[condition][:, feature, time], coef_list[0][:, feature, time], paired=True, alternative='two-sided')
                    significant_range[condition-1][feature, time] = stats['p-val'] * coef_list[condition].shape[2]  # Bonferroni correction

    elif stat_test=='permut_cluster':
        for condition in range(1, len(coef_list)):
            for feature in range(coef_list[condition].shape[1]):
                pval = 0.05  # arbitrary
                dfn = 1  # degrees of freedom numerator: n_conditions - 1
                dfd = coef_list[condition][:, feature, :].shape[0] + coef_list[0][:, feature, :].shape[0] - 2  # degrees of freedom denominator: n_observations - n_conditions
                thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution
                F_obs, clusters, cluster_pv, H0 = permutation_cluster_test(
                    [coef_list[condition][:, feature, :], coef_list[0][:, feature, :]], n_permutations=2000,
                    tail=0, adjacency=None, n_jobs=-1, out_type='indices', max_step=1, threshold=thresh
                )
                p = np.ones(coef_list[condition][:, feature, :].shape[1])  # Initialize p_values to one
                for cluster, p_value in zip(clusters, cluster_pv):
                    p[cluster] = p_value
                significant_range[condition - 1][feature, :] = p

    return significant_range


def one_sample_t_test_trf_kernels(coef_list, stat_test='permut_cluster'):
    '''
    one sample t_test of a temporal trace.
    :param coef_list: numpy array of shape (nb_subjects, time_points)
    :return:
    '''

    if stat_test=='permut_cluster':
        pval = 0.05  # arbitrary
        df = coef_list.shape[0] - 1 # degrees of freedom numerator: n_observations - 1
        thresh = scipy.stats.t.ppf(1 - pval/2, df=df)  # T distribution, two-tailed
        t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(coef_list, n_permutations=2000,
            tail=0, adjacency=None, n_jobs=-1, out_type='indices', max_step=1, threshold=thresh)

        p = np.ones(coef_list.shape[1])  # Initialize p_values to one
        for cluster, p_value in zip(clusters, cluster_pv):
            p[cluster] = p_value

    else:
        raise ValueError("stat_test should be in {'permut_cluster'}")

    return p


def permutation_cluster_test_adapted(X, t_permut, threshold=None, tail=0, adjacency=None, max_step=1, t_power=1, out_type='indices'):
    """adapted from MNE for our custom permutations. In this version, t-values of the permutations must be computed beforehand.
    X should be the difference between full and control models, with shape (n_subjects, n_times, n_channels)
    """

    from mne.stats.cluster_level import _check_option, _setup_adjacency, _validate_type, _find_clusters, _cluster_indices_to_mask, _cluster_mask_to_indices, _pval_from_histogram, _reshape_clusters
    from mne.utils import logger, warn, _pl

    _check_option("out_type", out_type, ["mask", "indices"])
    _check_option("tail", tail, [-1, 0, 1])
    if not isinstance(threshold, dict):
        threshold = float(threshold)
        if (
                tail < 0 < threshold
                or tail > 0 > threshold
                or tail == 0
                and threshold < 0
        ):
            raise ValueError(
                "incompatible tail and threshold signs, got "
                "%s and %s" % (tail, threshold)
            )

    # check dimensions in X.
    X = X[:, np.newaxis] if X.ndim == 1 else X
    n_samples = X.shape[0]
    n_times = X.shape[1]

    sample_shape = X.shape[1:]
    if X.shape[1:] != sample_shape:
        raise ValueError("All samples must have the same size")

    # flatten the last dimensions in case the data is high dimensional
    X = np.reshape(X, (X.shape[0], -1))
    n_tests = X.shape[1]

    if adjacency is not None and adjacency is not False:
        adjacency = _setup_adjacency(adjacency, n_tests, n_times)

    # compute t-stat for original data
    t_obs = np.mean(X, axis=0) / (np.std(X, axis=0) / np.sqrt(X.shape[0]))
    _validate_type(t_obs, np.ndarray, "return value of stat_fun")
    logger.info("stat_fun(H1): min=%f max=%f" % (np.min(t_obs), np.max(t_obs)))

    logger.info("Running initial clustering …")
    out = _find_clusters(
        t_obs,
        threshold,
        tail,
        adjacency,
        max_step=max_step,
        t_power=t_power,
        show_info=True,
    )
    clusters, cluster_stats = out

    # The stat should have the same shape as the samples
    t_obs.shape = sample_shape

    logger.info(f"Found {len(clusters)} cluster{_pl(clusters)}")

    # convert clusters to old format
    if adjacency is not None and adjacency is not False:
        # our algorithms output lists of indices by default
        if out_type == "mask":
            clusters = _cluster_indices_to_mask(clusters, n_tests)
    else:
        # ndimage outputs slices or boolean masks by default
        if out_type == "indices":
            clusters = _cluster_mask_to_indices(clusters, t_obs.shape)

    if len(clusters) == 0:
        warn("No clusters found, returning empty H0, clusters, and cluster_pv")
        return t_obs, np.array([]), np.array([]), np.array([])

    # if there are some clusters, find clusters in permutations

    # flatten the last dimensions in case the data is high dimensional
    t_permut = np.reshape(t_permut, (t_permut.shape[0], -1))

    max_cluster_sums = np.empty(t_permut.shape[0], dtype=np.double)
    for permut in range(t_permut.shape[0]):
        # Find cluster on randomized stats
        # logger.info("stat_fun(H1): min=%f max=%f" % (np.min(t_permut[permut, :]), np.max(t_permut[permut, :])))
        out = _find_clusters(
            t_permut[permut, :],
            threshold=threshold,
            tail=tail,
            max_step=max_step,
            adjacency=adjacency,
            t_power=t_power,
            show_info=False,
        )
        perm_clusters_sums = out[1]
        # print(perm_clusters_sums)
        if len(perm_clusters_sums) > 0:
            # get max with sign info
            idx_max = np.argmax(np.abs(perm_clusters_sums))
            max_cluster_sums[permut] = perm_clusters_sums[idx_max]
        else:
            max_cluster_sums[permut] = 0

    H0 = max_cluster_sums.tolist()

    # include original (true) ordering
    if tail == -1:  # up tail
        orig = cluster_stats.min()
    elif tail == 1:
        orig = cluster_stats.max()
    else:
        orig = abs(cluster_stats).max()
    H0.insert(0, orig)
    print(H0)
    H0 = np.array(H0)
    logger.debug("Computing cluster p-values")
    cluster_pv = _pval_from_histogram(cluster_stats, H0, tail)

    # The clusters should have the same shape as the samples
    clusters = _reshape_clusters(clusters, sample_shape)

    return t_obs, clusters, cluster_pv, H0, cluster_stats

def permutation_cluster_test_adapted_seeg(X, t_permut, threshold=None, tail=1, adjacency=None, max_step=1, t_power=1, out_type='indices'):
    """adapted from MNE for our custom permutations. In this version, R² values are used as input, from both the real observation and the permutations
    X should be the difference between full and control models, with shape (n_times, n_channels)
    """

    from mne.stats.cluster_level import _check_option, _setup_adjacency, _validate_type, _find_clusters, _cluster_indices_to_mask, _cluster_mask_to_indices, _pval_from_histogram, _reshape_clusters
    from mne.utils import logger, warn, _pl

    _check_option("out_type", out_type, ["mask", "indices"])
    _check_option("tail", tail, [-1, 0, 1])
    if not isinstance(threshold, dict):
        threshold = float(threshold)
        if (
                tail < 0 < threshold
                or tail > 0 > threshold
                or tail == 0
                and threshold < 0
        ):
            raise ValueError(
                "incompatible tail and threshold signs, got "
                "%s and %s" % (tail, threshold)
            )

    # check dimensions in X
    n_times = X.shape[0]
    n_channels = X.shape[1]

    sample_shape = X.shape
    if X.shape != sample_shape:
        raise ValueError("All samples must have the same size")

    # flatten the dimensions
    X = np.squeeze(np.reshape(X, (-1, 1)))
    n_tests = X.shape[0]

    if adjacency is not None and adjacency is not False:
        adjacency = _setup_adjacency(adjacency, n_tests, n_times)

    # !! The t-stat is replaced by the R² values we already have
    t_obs = X
    _validate_type(t_obs, np.ndarray, "return value of stat_fun")
    logger.info("stat_fun(H1): min=%f max=%f" % (np.min(t_obs), np.max(t_obs)))

    logger.info("Running initial clustering …")
    out = _find_clusters(
        t_obs,
        threshold,
        tail,
        adjacency,
        max_step=max_step,
        t_power=t_power,
        show_info=True,
    )
    clusters, cluster_stats = out

    # The stat should have the same shape as the samples
    t_obs.shape = sample_shape

    logger.info(f"Found {len(clusters)} cluster{_pl(clusters)}")

    # convert clusters to old format
    if adjacency is not None and adjacency is not False:
        # our algorithms output lists of indices by default
        if out_type == "mask":
            clusters = _cluster_indices_to_mask(clusters, n_tests)
    else:
        # ndimage outputs slices or boolean masks by default
        if out_type == "indices":
            clusters = _cluster_mask_to_indices(clusters, t_obs.shape)

    if len(clusters) == 0:
        warn("No clusters found, returning empty H0, clusters, and cluster_pv")
        return t_obs, np.array([]), np.array([]), np.array([])

    # if there are some clusters, find clusters in permutations

    # flatten the last dimensions in case the data is high dimensional
    t_permut = np.reshape(t_permut, (t_permut.shape[0], -1))

    max_cluster_sums = np.empty(t_permut.shape[0], dtype=np.double)
    for permut in range(t_permut.shape[0]):
        # Find cluster on randomized stats
        # logger.info("stat_fun(H1): min=%f max=%f" % (np.min(t_permut[permut, :]), np.max(t_permut[permut, :])))
        out = _find_clusters(
            t_permut[permut, :],
            threshold=threshold,
            tail=tail,
            max_step=max_step,
            adjacency=adjacency,
            t_power=t_power,
            show_info=False,
        )
        perm_clusters_sums = out[1]
        # print(perm_clusters_sums)
        if len(perm_clusters_sums) > 0:
            # get max with sign info
            idx_max = np.argmax(np.abs(perm_clusters_sums))
            max_cluster_sums[permut] = perm_clusters_sums[idx_max]
        else:
            max_cluster_sums[permut] = 0

    H0 = max_cluster_sums.tolist()

    # include original (true) ordering
    if tail == -1:  # up tail
        orig = cluster_stats.min()
    elif tail == 1:
        orig = cluster_stats.max()
    else:
        orig = abs(cluster_stats).max()
    H0.insert(0, orig)
    print(H0)
    H0 = np.array(H0)
    logger.debug("Computing cluster p-values")
    cluster_pv = _pval_from_histogram(cluster_stats, H0, tail)

    # The clusters should have the same shape as the samples
    clusters = _reshape_clusters(clusters, sample_shape)

    return t_obs, clusters, cluster_pv, H0


def select_channels_from_clusters(significant_clusters, sample_min, sample_max):

    channels = []
    for cluster in significant_clusters:
        for i, t in enumerate(cluster[0]):
            if (t >= sample_min) & (t <= sample_max):
                channels.append(cluster[1][i])
    channels = np.unique(channels)

    return channels


def permut_test_spearman(test_data, conditions, tail=-1, n_permut=1000):
    """permutation test of spearman rank correlation on test_data of shape (n_subjects, n_conditions), with conditions
    being for instance note rates. Permute the conditions"""

    # non-permuted stat
    stat = np.mean([scipy.stats.spearmanr(np.column_stack((test_data[s, :], conditions))).statistic for s in range(test_data.shape[0])])
    # stat = np.mean([np.corrcoef(test_data[s, :], conditions)[0, 1] for s in range(test_data.shape[0])])

    null_distrib = [stat]  # the non-permuted version is added to the distribution
    for p in tqdm(range(n_permut)):
        permuted_data = np.take_along_axis(test_data, np.random.randn(*test_data.shape).argsort(axis=1), axis=1)
        null_distrib.append(np.mean([scipy.stats.spearmanr(np.column_stack((permuted_data[s, :], conditions))).statistic for s in range(permuted_data.shape[0])]))
        # null_distrib.append(np.mean([np.corrcoef(permuted_data[s, :], conditions)[0, 1] for s in range(permuted_data.shape[0])]))
    null_distrib = np.array(null_distrib)
    if tail == -1:
        p = np.sum(stat >= null_distrib) / len(null_distrib)
    elif tail == 1:
        p = np.sum(stat <= null_distrib) / len(null_distrib)
    if tail == 0:
        p = np.sum(stat <= np.abs(null_distrib)) / len(null_distrib)

    return stat, p


def unpaired_t_test(*args):
    """Perform an unpaired two-sample t test with unequal sample sizes."""
    n_classes = len(args)
    assert n_classes == 2  # only made for 2 samples

    n1 = len(args[0])
    n2 = len(args[1])
    df = n1 + n2 - 2
    std1 = np.std(args[0], axis=0, ddof=1)
    std2 = np.std(args[1], axis=0, ddof=1)
    m1 = np.mean(args[0], axis=0)
    m2 = np.mean(args[1], axis=0)
    svar = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / df
    denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    t = (m1 - m2) / denom

    return t


def permut_1sample_ttest_across_channels(a, subject_id, popmean=0, alternative='two-sided', n_permutations=1000):
    """One sample t-test across channels with permutations (random sign flips) of channel values. Channels from the same
     subject are permuted together to maintain data structure.

     Parameters:
        a : array-like
            Array of data points (e.g., measurements) for all subjects and channels.
        subject_id : array-like
            Array of subject identifiers, indicating which subject each channel corresponds to.
        popmean : float, optional
            Population mean to test against, default is 0.
        alternative : {'two-sided', 'greater', 'less'}, optional
            The alternative hypothesis to test, default is 'two-sided'.
        n_permutations : int, optional
            Number of permutations to perform, default is 1000.

    Returns:
        p_value : float
            p-value from the permutation test.
     """

    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    subject_id = np.asarray(subject_id)

    # Compute unique subjects
    unique_subjects = np.unique(subject_id)
    n_subjects = len(unique_subjects)

    # Precompute the original t-statistic
    def t_stat(data):
        return (np.mean(data) - popmean) / (np.std(data, ddof=1) / np.sqrt(len(data)))

    t_obs = t_stat(a)

    # Generate all combinations of sign flips if possible
    max_combinations = 2 ** n_subjects
    if n_permutations > max_combinations:  # do all combinations
        all_signs = list(product([-1, 1], repeat=n_subjects))
    else:
        all_signs = [np.random.choice([-1, 1], size=n_subjects) for _ in range(n_permutations)]

    # Initialize array to store permuted t-values
    t_permuted = []

    for signs in all_signs:
        flipped_a = a.copy()
        for i, subj in enumerate(unique_subjects):
            # Apply the same sign flip to all channels of each subject
            flipped_a[subject_id == subj] *= signs[i]
        t_permuted.append(t_stat(flipped_a))

    if not n_permutations > max_combinations:
        t_permuted.append(t_obs)  # add the observed statistic to the distribution (if doing all combinations, it is already added in the previous step)
    t_permuted = np.array(t_permuted)

    # Compute p-value based on the alternative hypothesis
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(t_permuted) >= np.abs(t_obs))
    elif alternative == 'greater':
        p_value = np.mean(t_permuted >= t_obs)
    elif alternative == 'less':
        p_value = np.mean(t_permuted <= t_obs)
    else:
        raise ValueError("Invalid alternative hypothesis. Choose from 'two-sided', 'greater', 'less'.")

    return p_value


def already_permuted_1sample_ttest_across_channels(a, a_permuted, popmean=0, alternative='two-sided'):
    """If permutations are already computed (for example across trials before regression fitting) in
    a_permuted (n_permutations, n_channels), compute the t-values to construct a null distribution
    and then compute the p-value of the observed data. Data structure (for example, channel dependence within subjects)
    must be maintained by the permutation design."""

    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    a_permuted = np.asarray(a_permuted)

    # Precompute the original t-statistic
    def t_stat(data):
        return (np.mean(data) - popmean) / (np.std(data, ddof=1) / np.sqrt(len(data)))

    t_obs = t_stat(a)

    # Initialize array to store permuted t-values
    t_permuted = [t_stat(a_permuted[perm, :]) for perm in range(a_permuted.shape[0])]
    t_permuted.append(t_obs)  # add the observed statistic to the distribution
    t_permuted = np.array(t_permuted)

    # Compute p-value based on the alternative hypothesis
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(t_permuted) >= np.abs(t_obs))
    elif alternative == 'greater':
        p_value = np.mean(t_permuted >= t_obs)
    elif alternative == 'less':
        p_value = np.mean(t_permuted <= t_obs)
    else:
        raise ValueError("Invalid alternative hypothesis. Choose from 'two-sided', 'greater', 'less'.")

    return p_value


def logistic_log_likelihood(y_true, y_pred):
    """Compute log likelihood for logistic regression (bernoulli distribution)"""
    return np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def ols_log_likelihood(y_true, y_pred):
    """Compute log likelihood for gaussian noise assumption (typically after OLS or other curve fitting)"""
    n = len(y_true)
    residuals = y_true - y_pred
    sse = np.sum(residuals ** 2)
    sigma2 = sse / n
    ll = -0.5 * n * (np.log(2 * np.pi) + 1 + np.log(sigma2))
    return ll