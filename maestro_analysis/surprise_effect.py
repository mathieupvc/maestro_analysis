from pathlib import Path

from scipy.odr import quadratic
from tqdm import tqdm

import pandas as pd
import numpy as np

from scipy.stats import ttest_1samp
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

def rms_and_surprise_seeg(data_path, behavior_path, regions, first_parcel_label=True, analysis_name='rms'):
    """electrode-level test (FDR corrected) used to plot rms, gamma power or other scalar saved with the same format."""
    behavior_all = pd.read_csv(behavior_path)
    path = Path(data_path)
    subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
    subject_names = []
    print(subject_paths)

    rms_lists = []
    surprise_lists = []
    quadratic_lists = []
    subjects_quadratic = []
    subject_id = []  # used for color

    for i, subject_path in enumerate(tqdm(subject_paths)):
        subject_name = Path(subject_path).name[:6]
        print(subject_name)
        subject_names.append(subject_name)
        behavior = behavior_all[behavior_all['subject'] == subject_name]
        behavior = behavior[behavior['correct']==1]  # keep only correct
        behavior = behavior[behavior['rep2']==1]
        stim_names = behavior['stim_name'].tolist()
        surprise = behavior['surprise_mean_paulrnn'].tolist()
        print(stim_names)
        print(surprise)

        rms_obj = np.load(subject_path, allow_pickle=True)
        rms = rms_obj[analysis_name]
        parcellations = rms_obj['ch_parcellations']
        if first_parcel_label:
            parcellations = [[parcel[0]] for parcel in parcellations]
        # print(np.array(rms_obj['ch_names']))
        print(rms_obj['stim_name'])
        print(len(surprise))
        print(len(rms_obj['stim_name']))
        epochs_ordered = []
        surprise_to_keep = []
        for e, name in enumerate(stim_names):
            if name in rms_obj['stim_name']:
                epochs_ordered.append(rms_obj['stim_name'].index(name))
                surprise_to_keep.append(e)
        surprise = np.array(surprise)[surprise_to_keep]
        print(epochs_ordered)
        print(len(epochs_ordered))
        print(len(surprise))
        s_quadra = []
        for c in range(rms.shape[1]):
            for region in regions:
                if any(ch_parcel in region for ch_parcel in parcellations[c]):
                    rms_lists.append(rms[epochs_ordered, c])
                    surprise_lists.append(surprise)
                    subject_id.append(i)
                    # plt.figure()
                    # plt.scatter(surprise, rms[epochs_ordered, c])
                    # plt.show()
                    X = np.zeros((surprise.shape[0], 2))
                    # X = np.zeros((surprise.shape[0], 1))
                    X[:, 0] = surprise
                    X[:, 1] = surprise**2
                    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
                    y = rms[epochs_ordered, c]
                    y = (y - np.mean(y)) / np.std(y)
                    model = LinearRegression(fit_intercept=True)
                    model.fit(X, y)
                    quadratic_lists.append(model.coef_[1])
                    s_quadra.append(model.coef_[1])
                    print(model.coef_)

        subjects_quadratic.append(np.mean(s_quadra))

    fig, axs = plt.subplots(1, 1, sharey=True)
    subject_colors = plt.get_cmap('Set1').colors
    # Plot each condition's data in a separate subplot
    for i, (surprise_chan, rms_chan, s_id) in enumerate(zip(surprise_lists, rms_lists, subject_id)):
        axs.scatter(surprise_chan, rms_chan, color=subject_colors[s_id])
        # axs.scatter(x_values, rms, color=colors, alpha=1, s=20, label='Samples')
    axs.set_ylabel(f'{analysis_name} relative to baseline')
    axs.set_xlabel(f'surprise')


    # Plot each condition's data in a separate subplot
    surprise = np.concatenate(surprise_lists, axis=0)
    rms = np.concatenate(rms_lists, axis=0)
    print(surprise.shape)
    print(rms.shape)

    plt.figure()
    plt.hist(rms, bins=100)
    plt.show()

    fig, axs = plt.subplots(1, 1, sharey=True)

    bin_edges = np.linspace(surprise.min(), surprise.max(), 10, endpoint=True)
    bins = np.digitize(surprise, bins=bin_edges)
    # bins = pd.qcut(surprise, q=10, labels=False, duplicates='drop')

    df = pd.DataFrame({'surprise': surprise, 'rms': rms, 'bins': bins})
    print(df)
    grouped = df.groupby('bins').agg({'surprise': 'mean', 'rms': 'mean'}).reset_index()
    print(grouped)
    bins = grouped['bins'].values
    binned_surprise = grouped['surprise'].values
    mean_rms = grouped['rms'].values
    axs.bar(bins, mean_rms)
        # axs.scatter(x_values, rms, color=colors, alpha=1, s=20, label='Samples')
    axs.set_ylabel(f'{analysis_name} relative to baseline')
    axs.set_xlabel(f'surprise')

    plt.show()

    # quadratic test
    print(quadratic_lists)
    print(ttest_1samp(quadratic_lists, 0, alternative='two-sided'))

    # subject quadra test
    print(subjects_quadratic)
    print(ttest_1samp(subjects_quadratic, 0, alternative='two-sided'))

    return


if __name__ == "__main__":
    """rms_and_surprise_seeg"""
    data_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels'
    behavior_path = '/home/mathieu/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv'
    first_parcel_label = True
    analysis_name = 'rms'
    # regions = ['Left-Hippocampus', 'Right-Hippocampus']
    regions = ['Left-Hippocampus']
    rms_and_surprise_seeg(data_path, behavior_path, regions, first_parcel_label=True, analysis_name='rms')
