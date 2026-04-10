import argparse
import pickle

import numpy as np
from tqdm import tqdm

from maestro_analysis.note_analysis import regression_predictive_features

# ========================================================================
'''Argument parser'''
# ========================================================================

parser = argparse.ArgumentParser()

parser.add_argument("-ep", "--epochs_path", type=str, help="folder containing .fif epochs")
parser.add_argument("-cp", "--control_path", type=str, help=".pkl file containing control regression")
parser.add_argument("-s", "--save_path", type=str, help="destination path of the pickle files (1 file per permutation)")
parser.add_argument("-dt", "--data_type", type=str, help="'meg' or 'seeg'")
parser.add_argument("-c", "--run_control", default=False, action='store_true', help="run the control regression or not. If True, control_path should be set accordingly to load the new control")
parser.add_argument("-ns", "--run_non_shuffled", default=False, action='store_true', help="run the non-shuffled regression or not")
parser.add_argument("-p", "--run_permutations", default=False, action='store_true', help="run the permutations or not")
parser.add_argument("-sp", "--starting_permutation", type=int, default=1, help="permutations go from 1 to 1000, this parameter enables starting from a given index")

args = parser.parse_args()

epochs_path = args.epochs_path
control_path = args.control_path
save_path = args.save_path
data_type = args.data_type
run_control = args.run_control
run_non_shuffled = args.run_non_shuffled
run_permutations = args.run_permutations
starting_permutation = args.starting_permutation
if starting_permutation > 1000:
    raise ValueError('starting_permutation should be <= 1000')

print('epochs_path:', epochs_path)
print('control_path:', control_path)
print('save_path:', save_path)
print('data_type:', data_type)
print('run_control:', run_control)
print('run_non_shuffled:', run_non_shuffled)
print('run_permutations:', run_permutations)
print('starting_permutation:', starting_permutation)


# Run control regression
if run_control:
    regression_predictive_features(epochs_path,
                                   save_path,
                                   confounds='all',
                                   model=None,
                                   data_type=data_type,
                                   signal_type='raw',
                                   montage='mono',
                                   specific_channels=None,
                                   add_shifted=True,
                                   tmin=2,
                                   tmax=7,
                                   n_folds=5,
                                   alphas=np.logspace(2, 7, 6),
                                   permutation=False,
                                   r2_control=None,
                                   min_time_interval=0.050
                                   )

# Run non-shuffled regression
if run_non_shuffled:
    regression_predictive_features(epochs_path,
                                   save_path,
                                   confounds='all',
                                   model='paulrnn',
                                   data_type=data_type,
                                   signal_type='raw',
                                   montage='mono',
                                   specific_channels=None,
                                   add_shifted=True,
                                   tmin=2,
                                   tmax=7,
                                   n_folds=5,
                                   alphas=np.logspace(2, 7, 6),
                                   permutation=False,
                                   r2_control=None,
                                   min_time_interval=0.050
                                   )

# load control in case of MEG
if data_type == 'meg':
    with open(control_path, 'rb') as file:
        control = pickle.load(file)
    control = control['r2']
else:
    control = None

if run_permutations:
    for permut in tqdm(range(starting_permutation, 1001)):

        regression_predictive_features(epochs_path,
                                       save_path,
                                       confounds='all',
                                       model='paulrnn',
                                       data_type=data_type,
                                       signal_type='raw',
                                       montage='mono',
                                       specific_channels=None,
                                       add_shifted=True,
                                       tmin=2,
                                       tmax=7,
                                       n_folds=5,
                                       alphas=np.logspace(2, 7, 6),
                                       permutation=permut,
                                       r2_control=control,
                                       min_time_interval=0.050
                                       )

print('Done')
