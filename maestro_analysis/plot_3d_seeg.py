# Need the mne-python installer environment to run, and run this code outside of pycharm
# source ~/mne-python/1.8.0_0/bin/activate

# import numpy as np
#
# import mne
# from mne.viz import Brain
#
# import matplotlib.pyplot as plt
#
# sample_path = mne.datasets.sample.data_path()
# subjects_dir = sample_path / "subjects"
# # Fetch the fsaverage template (this will download it if not already present)
# fsaverage_dir = mne.datasets.fetch_fsaverage(verbose=True)
# # Set the subjects_dir to the directory where fsaverage is downloaded
# # Define the subject as 'fsaverage'
# subject = 'fsaverage'
#
# # Create a Brain object for visualization, using both hemispheres ('hemi' = 'both')
# brain = Brain(subject, subjects_dir=subjects_dir, cortex='low_contrast', alpha=0.2)
#
# # List of regions of interest from the aparc+aseg parcellation
#
# # # encoding fdr corrected
# # regions_of_interest = ['Left-VentralDC', 'ctx-lh-parstriangularis', 'ctx-lh-superiorfrontal', 'ctx-rh-parahippocampal']
#
# # encoding
# # regions_of_interest = ['Left-Hippocampus', 'ctx-lh-parsopercularis', 'ctx-rh-inferiorparietal']
#
# # novelty fdr corrected
# # regions_of_interest = ['ctx-rh-rostralmiddlefrontal']
#
# # # novelty
# # regions_of_interest = ['Left-Amygdala', 'Right-Hippocampus']
# #
# # # retrieval
# regions_of_interest = ['Left-Thalamus-Proper', 'Right-Thalamus-Proper']
#
# # Highlight specific regions from aparc+aseg
# # for region in regions_of_interest:
# brain.add_volume_labels(aseg='aparc+aseg', labels=regions_of_interest, alpha=0.7, legend=True)  # Adjust color and transparency (alpha)
# # Show the plot with highlighted regions
# brain.show()




# plot 3d rms results
# import numpy as np
# import mne
# from mne.viz import Brain
# import matplotlib.pyplot as plt
# import pandas as pd
# from pathlib import Path
# from tqdm import tqdm
# from scipy.stats import ttest_ind
# import pickle
#
# """electrode-level test (FDR corrected) used to plot rms, gamma power or other scalar saved with the same format."""
# analysis_name = 'rms'
# behavior_all = pd.read_csv('/home/mathieupvc/These/data/meg_maestro/behavior_analysis/seeg_behavior/behavior_seeg_data_matrix.csv')
# path = Path('/home/mathieupvc/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels')
# subject_paths = sorted(list(path.glob(f'*epochs_{analysis_name}.pkl')))
# subject_names = []
# print(subject_paths)
#
# conditions = ['correct_incorrect', 'encoding', 'novelty', 'retrieval', 'memory']
#
# tvalues = [[] for c in range(len(conditions))]
# pvalues = [[] for c in range(len(conditions))]
# infos = [[] for c in range(len(conditions))]  # used for color
#
# for i, subject_path in enumerate(tqdm(subject_paths)):
#     subject_name = Path(subject_path).name[:6]
#     subject_names.append(subject_name)
#     behavior = behavior_all[behavior_all['subject'] == subject_name]
#     correct_all = behavior[(behavior['correct'] == 1)]['stim_name'].tolist()
#     incorrect_all = behavior[(behavior['correct'] == 0)]['stim_name'].tolist()
#     correct_novelty = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
#     incorrect_novelty = behavior[(behavior['correct'] == 0) & (behavior['rep1'] == 1)]['stim_name'].tolist()
#     correct_retrieval = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
#     incorrect_retrieval = behavior[(behavior['correct'] == 0) & (behavior['rep2'] == 1)]['stim_name'].tolist()
#     correct_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 1)]['stim_identity'].tolist()
#     incorrect_rep2 = behavior[(behavior['rep2'] == 1) & (behavior['correct'] == 0)]['stim_identity'].tolist()
#     correct_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(correct_rep2))]['stim_name'].tolist()
#     incorrect_encoding = behavior[(behavior['rep1'] == 1) & (behavior['stim_identity'].isin(incorrect_rep2))]['stim_name'].tolist()
#     memorized = behavior[(behavior['correct'] == 1) & (behavior['rep2'] == 1)]['stim_name'].tolist()
#     novel = behavior[(behavior['correct'] == 1) & (behavior['rep1'] == 1)]['stim_name'].tolist()
#
#     with open(subject_path, 'rb') as f:
#         rms_obj = pickle.load(f)
#     # rms_obj = np.load(subject_path, allow_pickle=True)
#     rms = rms_obj[analysis_name]
#
#     for cond, correct_incorrect in enumerate([[correct_all, incorrect_all], [correct_encoding, incorrect_encoding], [correct_novelty, incorrect_novelty], [correct_retrieval, incorrect_retrieval], [memorized, novel]]):
#         correct = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[0]]
#         incorrect = [e for e in range(len(rms_obj['stim_name'])) if rms_obj['stim_name'][e] in correct_incorrect[1]]
#         if (len(correct) > 0) & (len(incorrect) > 0):
#             infos[cond].append(rms_obj['info'])
#             for c in tqdm(range(rms.shape[1])):
#                 test_result = ttest_ind(rms[correct, c], rms[incorrect, c], axis=0, equal_var=False, permutations=1000, random_state=None, alternative='two-sided')  # testing of individual electrodes
#                 tvalues[cond].append(test_result.statistic)
#                 pvalues[cond].append(test_result.pvalue)
#
# # FDR correction
# for c in range(len(pvalues)):
#     _, pval = mne.stats.fdr_correction(pvalues[c], alpha=0.05, method='indep')
#     pvalues[c] = pval
#
# subject_colors = plt.get_cmap('Set1').colors
# # Plot each condition's data in a separate subplot
# for i, (condition, tval, pval, info_list) in enumerate(zip(conditions, tvalues, pvalues, infos)):
#     # concatenate infos
#     merged_info = info_list[0].copy()
#     for info in info_list[1:]:
#         for ch in info['chs']:
#             merged_info['chs'].append(ch)
#         merged_info['ch_names'].extend(info['ch_names'])
#         # merged_info['nchan'] = len(merged_info['ch_names']
#     print(merged_info)
#     print(merged_info['ch_names'])
#     print(len(merged_info['ch_names']))
#     # print(merged_info['chs'])
#     tval = np.array(tval)
#     print(tval.shape)
#
#     sample_path = mne.datasets.sample.data_path()
#     subjects_dir = sample_path / "subjects"
#     # Fetch the fsaverage template (this will download it if not already present)
#     fsaverage_dir = mne.datasets.fetch_fsaverage(verbose=True)
#     # Set the subjects_dir to the directory where fsaverage is downloaded
#     # Define the subject as 'fsaverage'
#     subject = 'fsaverage'
#
#     # Create a Brain object for visualization, using both hemispheres ('hemi' = 'both')
#     brain = Brain(subject, subjects_dir=subjects_dir, cortex='low_contrast', alpha=0.2)
#     brain.add_sensors(merged_info)
#     brain.show()


################################## plot 3d correct incorrect ###############################
import numpy as np
import mne
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colorbar as mcolorbar

sample_path = mne.datasets.sample.data_path()
subjects_dir = sample_path / "subjects"
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)
subject = 'fsaverage'

# contrast_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/rms_contrast_results_correct_incorrect.npy'
contrast_path = '/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/rms_contrast_results_novelty_retrieval.npy'
contrast_obj = np.load(contrast_path, allow_pickle=True).item()
# global_montage = mne.channels.read_dig_fif('/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/global_montage.fif')
global_montage = mne.channels.read_dig_fif('/home/mathieu/These/data/meg_maestro/pac_analysis/seeg/epochs_rms_all_channels/global_montage_novelty_retrieval.fif')
global_montage.ch_names = [str(i) for i in range(len(global_montage.ch_names))]
info = mne.create_info(global_montage.ch_names, 400., ch_types='seeg')
info.set_montage(global_montage)

# tvalues = contrast_obj['tvalues']
tvalues = contrast_obj['tvalues_diff']
# pvalues = contrast_obj['pvalues']
vmax = np.max(np.abs(tvalues))
norm = plt.Normalize(vmin=-vmax, vmax=vmax)
cmap = cm.seismic
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', ['#ff5a5f', 'white', '#087e8b'])
sensor_colors = [cmap(norm(t)) for t in tvalues]
# for c in range(len(sensor_colors)):
#     if pvalues[c]>=0.05:
#         sensor_colors[c] = cmap(0.5)

head_fs = mne.transforms.Transform('head', 'mri',trans=None)
fig = mne.viz.plot_alignment(info, trans=head_fs, subject=subject, subjects_dir=subjects_dir, surfaces='pial',
                             coord_frame='mri', show_axes=False, sensor_colors=sensor_colors)
# Set the background color after the figure is created
fig.plotter.set_background('white')  # Use this for PyVista plots

# Optionally render the plot if needed
# fig.show()  # Make sure to call show() to visualize the figure

fig, ax = plt.subplots(figsize=(6, 1))
cb = mcolorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
cb.set_label('Contrast t-value')
plt.show()
