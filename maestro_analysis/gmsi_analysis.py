import pickle

import numpy as np
import scipy

from utils import gmsi_array_extraction

def r2_vs_gmsi(model_regression, control_regression, gmsi_path, sign_channels_path, tmin=50):

    gmsi_array = gmsi_array_extraction(gmsi_path=gmsi_path)
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

    print('Correlations wit gMSI:')
    print('Control:', np.corrcoef(gmsi_array, mean_r2_control, rowvar=True)[0, 1])
    print('Model:', np.corrcoef(gmsi_array, mean_r2_model)[0, 1])
    print('Difference:', np.corrcoef(gmsi_array, mean_r2_diff)[0, 1])

    print('Pearson correlation = ', scipy.stats.pearsonr(gmsi_array, mean_r2_diff))
    print('Spearman correlation = ', scipy.stats.spearmanr(gmsi_array, mean_r2_diff))

    return


if __name__ == '__main__':
    '''r2_vs_gmsi'''
    with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/onset_07_12_23/regr_meg_paulrnn.pkl', 'rb') as file:
        model_regression = pickle.load(file)
    with open('/home/mathieupvc/These/data/meg_maestro/note_analysis/epochs_1s_all_meta/regression/onset_07_12_23/regr_meg_acoustic.pkl', 'rb') as file:
        control_regression = pickle.load(file)
    gmsi_path = '/home/mathieupvc/These/data/meg_maestro/gmsi_processed.csv'
    sign_channels_path = '/home/mathieupvc/These/data/meg_maestro/gmsi_analysis/significant_channels_200ms.npy'
    r2_vs_gmsi(model_regression, control_regression, gmsi_path, sign_channels_path=sign_channels_path, tmin=50)
