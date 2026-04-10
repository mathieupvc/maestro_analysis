import os
import pathlib

import numpy as np
import pandas as pd
import mne

from maestro_analysis.utils import epoching_maestro_note_level

#%%
# path = 'C:/Users/paull/Science/PhD/2022_deep_music/meg/'
path = 'D:/2022_deep_music/meg/'
raw_path = path + 'maestro01_itdheb/210629raw_sss.fif'
wav_folder = path + 'wav/'
midi_folder = path + 'midi/'
acoustic_folder = path + 'acoustic/'
perfrnn_dir = path + 'models/perfRNN/'
paulrnn_dir = path + 'models/paulRNN/'

epochs = epoching_maestro_note_level(raw_path,wav_folder, midi_folder, acoustic_folder, perfrnn_dir, paulrnn_dir,
                                      t_min = -1, t_max = 1,
                                      channel_type = 'mag', downsample = True, detrend = True,
                                      multiple_notes = 'max',n_jobs=1)

listening_epochs = epochs['note_on']
listening_epochs.apply_baseline((None, -0.1))
listening_evoked = listening_epochs.average()
# listening_evoked.plot()
# listening_evoked.plot_topomap()

metadata = epochs.metadata
epochs.save('C:/Users/paull/Science/PhD/2022_deep_music/meg/preprocessed/maestro01_itdheb_epochs.fif', overwrite = True)


#%%
import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg') #QtAgg
import matplotlib.pyplot as plt
from maestro_analysis.note_analysis import plot_regression_betas, regression_predictive_features

epochs_path = 'C:/Users/paull/Science/PhD/2022_deep_music/meg/preprocessed/'
save_path = 'C:/Users/paull/Science/PhD/2022_deep_music/meg/regression_results/'

regression_predictive_features(epochs_path, save_path, confounds='all', model = 'paulrnn', 
                               specific_channels=None, rms=False, shifted=True, tmin=1)
features_names = np.load(save_path + 'regression_betas_features_names.npy')

plot_regression_betas(save_path + 'regression_betas_predictive_features.npy', 
                      save_path + 'regression_betas_features_names.npy', 
                      tmin=-1, tmax=1, multiple_channels='abs', 
                      features_plot=['surprise_paulrnn', 'entropy_pitch_paulrnn', 'entropy_timing_paulrnn'], 
                      backend = 'QtAgg')

#%%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import mne

def plot_topo_regression_betas(betas_path, features_path, epochs_path, time=None, features_plot='all', backend = None):
    '''
    Topographical plot of beta values from regression analysis
    betas_path: path to betas files
    features_names: list of name of features used in the regression analysis
    time: time range (smaller than [-1:1] sec) within which betas values are averaged
    features_plot: 'all', or list of features to plot
    backend: matplotlib backend to switch to before plotting. If None, the current backend is used. Otherwise, choices are 'TkAgg', 'QtAgg'
    
    '''
    
    if time is None:
        time = [0.08, 0.13]
        
    if backend is not None:
        if backend == 'TkAgg':
            plt.switch_backend('TkAgg')
        if backend == 'QtAgg':
            plt.switch_backend('QtAgg')
            
    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epochs.fif')))
    
    features_names = np.load(features_path)
    if features_plot == 'all':
        features_plot_idx = [_ for _ in range(len(features_names))]
    else:
        features_plot_idx = [features_names.tolist().index(_) for _ in features_plot]
        
    #load epochs
    epochs = mne.read_epochs(epochs_list[0])
    time_sample = [np.where(epochs.times == time[0])[0][0], np.where(epochs.times == time[1])[0][0]]
    
    #load betas
    betas = np.load(betas_path) #dim : n_subject, n_features, n_channels, n_time_points
    betas = betas[:,:,:,time_sample[0]:time_sample[1]] #select betas within the desired time range
    betas = np.mean(betas, axis=3) #average beta values within that time-range
    
    beta_t_values = np.mean(betas, axis=0) / (np.std(betas, axis=0) / np.sqrt(betas.shape[0]))
    # beta_t_values = np.squeeze(betas[0,:,:]) #just to check one subject

    # Topo plot
    fig, axes = plt.subplots(len(features_plot_idx), 1)
    vmax = np.max(np.abs(beta_t_values))
    for i, f in enumerate(features_plot_idx):
        im, cn = mne.viz.plot_topomap(beta_t_values[f, :], epochs.info, show=False, axes=axes[i], ch_type='mag', vmin=-vmax, vmax=vmax)  # plot t-scores
        clb = fig.colorbar(im, ax=axes[i])
        if f == 0:
            clb.ax.set_title('Beta t-value', fontsize=15)  # title on top of colorbar
        axes[i].set_title(features_names[f] + '['+str(time[0])+', '+str(time[1])+'] s')

    plt.show()

    return


epochs_path = 'C:/Users/paull/Science/PhD/2022_deep_music/meg/preprocessed/'
save_path = 'C:/Users/paull/Science/PhD/2022_deep_music/meg/regression_results/'
plot_topo_regression_betas(save_path + 'regression_betas_predictive_features.npy', 
                      save_path + 'regression_betas_features_names.npy', 
                      epochs_path,
                      time=[0.08, 0.13],
                      features_plot='all', 
                      backend = 'QtAgg')


#%%

import numpy as np
import matplotlib.pyplot as plt
from maestro_analysis.note_analysis import regression_predictive_features

path = 'D:/2022_deep_music/seeg/'
fs_path = path + 'freesurfer/'
epochs_path = path + 'epochs_note_level_test/'
save_path = path + 'regression_note_level_test/'
model = 'paulrnn'

reg_paulrnn = regression_predictive_features(epochs_path, save_path, 
                                    confounds='all', model = model, 
                                    data_type = 'seeg', signal_type = 'raw', montage = 'mono',
                                    specific_channels=None, shifted=True, tmin=1,
                                    n_folds = 2, alphas = (1e-1, 1e0, 1e1))

#retrieve times and feature list
times = reg_paulrnn[list(reg_paulrnn.keys())[0]]['times']
features = reg_paulrnn[list(reg_paulrnn.keys())[0]]['features']
print(features)
show_feature = 'surprise_mean_paulrnn'
f_idx = features.index(show_feature)
y = []
ch_names = []
for sub in reg_paulrnn.keys():
  y.append(reg_paulrnn[sub]['subject_r2'][:,:])
  ch_names_sub = [ch + '_' + sub for ch in reg_paulrnn[sub]['ch_names']]
  ch_names += ch_names_sub
ch_names = np.array(ch_names)
y = np.concatenate(y,axis=0)  
#select time window
tw = [-1, 1]
lim = [np.abs(times - tw[0]).argmin(), np.abs(times - tw[1]).argmin()]
y = y[:,lim[0]:lim[1]]
times = times[lim[0]:lim[1]]

ch_max = np.max(np.abs(y),axis=1)
ch_mask = ch_max > np.percentile(ch_max, 90)
print(sum(ch_mask))
y_show = y[ch_mask,:]

#plot
plt.plot(times,y_show.T)
plt.legend(ch_names[ch_mask])
plt.show()

#%% noise ceiling MEG

import pickle
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def regression_noise_ceiling(epochs_path, save_path,
                                   tmin=1, n_folds = 1, alphas = (1e-1, 1e0, 1e1)):
    """
    epochs_path: path to folder containing all epochs.fif files
    save_path: path to folder where results will be saved
    tmin: remove notes before a given time (to avoid onset response)
    n_folds: If 1, just fit a linear regression. If > 1, performs n-folds cv with ridge regression.
    
    Compute noise ceiling for MEG dataset.
    ! The regression has to be done exactly the same as in the main regression analysis.
    ! UPDATE THIS FUNCTION IF NECESSARY !
    """
    
    alphas = np.array(alphas)
    
    #create save dir
    Path(save_path).mkdir(parents=True, exist_ok=True)

    #get the correct epochs path depending on the desired dataset
    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))

    #prepare dict to save output
    output = {'betas':[],
             'r2':[],
             'n_folds':n_folds,
             'alphas':alphas}
    
    #find shared epochs among all subjects
    metadatas = []
    for i, subject_epochs in enumerate(epochs_list):
        print(subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs, preload=False,verbose=0)
        metadata = epochs.metadata
        metadata['pitch'] = metadata['pitch'].apply(lambda x: x[0])   #pitch info was in list format
        metadata['rate'] = np.nan  #add rate from stim name
        metadata['pitch_interval'] = np.nan #add pitch interval
        metadata['epoch_ID'] = None
        for i_note in range(metadata.shape[0]):
            metadata.loc[i_note,'rate'] = float(metadata['stim_name'][i_note].split('_')[2][0:3])
            if metadata['trial_time'][i_note] > 0:
                metadata.loc[i_note,'pitch_interval'] = np.abs(metadata.loc[i_note-1,'pitch'] - metadata.loc[i_note,'pitch'])
            #create epoch ID based on stim_name and trial_time, used to merge metadatas later
            metadata.loc[i_note,'epoch_ID'] = int(metadata['stim_name'][i_note][5:8])*1e3 + np.round(metadata['trial_time'][i_note],6)
        metadatas.append(metadata)
    m_common = metadatas[0].copy()
    idx_drop = []
    for i in range(len(metadatas)):
        m_ = metadatas[i]
        for j in range(m_common.shape[0]):
            if m_common['epoch_ID'][j] not in m_['epoch_ID'].tolist():
                idx_drop.append(j)
    idx_drop = np.unique(idx_drop)            
    m_common = m_common.drop(index=idx_drop)

    #retrieve brain data for all subjects
    X = []
    for i, subject_epochs in enumerate(epochs_list):
        epochs = mne.read_epochs(subject_epochs, preload=True,verbose=0)
        output['times'] = epochs.times
        
        #first retrieve list of epochs common to all for this particular subject
        metadata = metadatas[i]
        idx_drop = []
        for j in range(metadata.shape[0]):
            if metadata['epoch_ID'][j] not in m_common['epoch_ID'].tolist():
                idx_drop.append(j)
        metadata = metadata.drop(index=idx_drop)
        epochs.drop(idx_drop)
        
        # remove onset response
        # epochs = epochs[metadata['trial_time'].to_numpy() >= tmin]
        X.append(epochs.get_data())
        del epochs
    X = np.stack(X,axis=0)

    #main loop for regression
    #for i, subject_epochs in enumerate(epochs_list):
    for i, subject_epochs in enumerate([epochs_list[0]]):
        #load data
        y = X[i,:,:,:]  # (trials, channels, time)
        x_idx = [_ for _ in range(len(epochs_list)) if _ != i]
        x = np.mean(X[x_idx,:,:,:],axis=0)
        

        # standardization of features and meg
        x = (x - np.mean(x, axis=0, keepdims=True)) / np.std(x, axis=0, keepdims=True)
        y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)  # removes the relative importance of subjects, channels, and time points
        
        # independent regression for every time point and channel
        # subject_betas = np.zeros((x.shape[1], y.shape[1], y.shape[2]))
        subject_r2 = np.zeros((y.shape[1], y.shape[2]))
        if n_folds > 1:
          cv = KFold(n_splits=n_folds)
          cv.get_n_splits(x)
        for c in range(y.shape[1]):
          for t in range(y.shape[2]):
              if n_folds == 1:
                regr = LinearRegression().fit(x[:,c,t], y[:, c, t])
                # subject_betas[:, c, t] = regr.coef_.T
                subject_r2[c, t] = r2_score(y[:,c,t], regr.predict(x[:,c,t]),multioutput='raw_values')
              elif n_folds > 1:
                cv_scores = np.full((alphas.shape[0], n_folds), np.nan)
                for i, alpha in enumerate(alphas):
                  for j, (train, test) in enumerate(cv.split(x)):
                    regr = Ridge(alpha=alpha)
                    regr.fit(x[train,c,t].reshape(-1,1),y[train, c, t])
                    pred = regr.predict(x[test,c,t].reshape(-1,1))
                    cv_scores[i,j] = r2_score(y[test,c,t], pred, multioutput='raw_values')
                #store cv scores and best alphas
                cv_scores = np.mean(cv_scores, axis=1) #mean cross-validated r² across folds
                best_alphas = alphas[np.argmax(cv_scores,axis = 0)] #store best alpha for each timepoint
                cv_scores = np.max(cv_scores, axis = 0) #for each timepoint, keep the highest cross-val r² across alphas
                subject_r2[c, t] = cv_scores
                # #fit again with best alpha and store betas
                # regr = Ridge(best_alphas).fit(x[:,c,:],y[:,c,:])
                # subject_betas[:, c, :] = regr.coef_
            
        #store regression results
        # output['betas'].append(subject_betas)
        output['r2'].append(subject_r2)
    # output['betas'] = np.array(output['betas'])
    output['r2'] = np.array(output['r2'])

    with open(save_path + 'NC_meg.pkl', 'wb') as fp:
        pickle.dump(output, fp)

    return output

path = 'D:/2022_deep_music/meg/'
epochs_path = path + 'epochs_note_level/'
save_path = path + 'regression_note_level_test/'

NC = regression_noise_ceiling(epochs_path, save_path, tmin=1, n_folds = 4, alphas = (1e0, 1e5))
times = np.linspace(-1,1,201)
y = NC['r2'][1,:,:]

ch_max = np.max(y,axis=1)
ch_mask = ch_max > np.percentile(ch_max, 95)
print(sum(ch_mask))
# y_show = y[f_idx, ch_mask,:]
y_show = y[ch_mask,:]

#plot
plt.plot(times,y_show.T)
plt.hlines(0,times[0], times[-1], linestyles='--', color='gray')
plt.legend(ch_names[ch_mask])
# plt.title(features[f_idx])
plt.title('delta R²')
plt.show()


#%% noise ceiling MEG _ v2

import pickle
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def regression_noise_ceiling(epochs_path, save_path,
                                   tmin=1, n_folds = 1, alphas = (1e-1, 1e0, 1e1)):
    """
    epochs_path: path to folder containing all epochs.fif files
    save_path: path to folder where results will be saved
    tmin: remove notes before a given time (to avoid onset response)
    n_folds: If 1, just fit a linear regression. If > 1, performs n-folds cv with ridge regression.
    
    Compute noise ceiling for MEG dataset.
    ! The regression has to be done exactly the same as in the main regression analysis.
    ! UPDATE THIS FUNCTION IF NECESSARY !
    """
    
    alphas = np.array(alphas)
    
    #create save dir
    Path(save_path).mkdir(parents=True, exist_ok=True)

    #get the correct epochs path depending on the desired dataset
    path = Path(epochs_path)
    epochs_list = sorted(list(path.glob('**/*epo.fif')))

    #prepare dict to save output
    output = {'betas':[],
             'r2':[],
             'n_folds':n_folds,
             'alphas':alphas}
    
    #find shared epochs among all subjects
    metadatas = []
    for i, subject_epochs in enumerate(epochs_list):
        print("Retrieving metadata from ", subject_epochs.name)
        epochs = mne.read_epochs(subject_epochs, preload=False,verbose=0)
        metadata = epochs.metadata
        metadata['pitch'] = metadata['pitch'].apply(lambda x: x[0])   #pitch info was in list format
        metadata['rate'] = np.nan  #add rate from stim name
        metadata['pitch_interval'] = np.nan #add pitch interval
        metadata['epoch_ID'] = None
        for i_note in range(metadata.shape[0]):
            metadata.loc[i_note,'rate'] = float(metadata['stim_name'][i_note].split('_')[2][0:3])
            if metadata['trial_time'][i_note] > 0:
                metadata.loc[i_note,'pitch_interval'] = np.abs(metadata.loc[i_note-1,'pitch'] - metadata.loc[i_note,'pitch'])
            #create epoch ID based on stim_name and trial_time, used to merge metadatas later
            metadata.loc[i_note,'epoch_ID'] = int(metadata['stim_name'][i_note][5:8])*1e3 + np.round(metadata['trial_time'][i_note],6)
        metadatas.append(metadata)
    m_common = metadatas[0].copy()
    idx_drop = []
    for i in range(len(metadatas)):
        m_ = metadatas[i]
        for j in range(m_common.shape[0]):
            if m_common['epoch_ID'][j] not in m_['epoch_ID'].tolist():
                idx_drop.append(j)
    idx_drop = np.unique(idx_drop)            
    m_common = m_common.drop(index=idx_drop)

    #main loop for regression
    #for i, subject_epochs in enumerate(epochs_list):
    for i, subject_epochs in enumerate([epochs_list[0]]):
        #retrieve brain data for the subject of interest
        epochs = mne.read_epochs(subject_epochs, preload=True,verbose=0)
        output['times'] = epochs.times
        output['ch_names'] = epochs.ch_names
        metadata = metadatas[i]
        #compare metadatas and remove epochs that are not common to all subjects
        idx_drop = []
        for j in range(metadata.shape[0]):
            if metadata['epoch_ID'][j] not in m_common['epoch_ID'].tolist():
                idx_drop.append(j)
        metadata = metadata.drop(index=idx_drop)
        epochs.drop(idx_drop)
        epochs = epochs[metadata['trial_time'].to_numpy() >= tmin] # remove onset response
        y = epochs.get_data()
        
        #retrieve brain data for all other subjects
        x = np.full(y.shape, 0.0)
        for j, others_epochs in enumerate(epochs_list):
            if j != i:
                epochs = mne.read_epochs(others_epochs, preload=True,verbose=0)
                metadata = metadatas[j]
                #compare metadatas and remove epochs that are not common to all subjects
                idx_drop = []
                for h in range(metadata.shape[0]):
                    if metadata['epoch_ID'][h] not in m_common['epoch_ID'].tolist():
                        idx_drop.append(h)
                metadata = metadata.drop(index=idx_drop)
                epochs.drop(idx_drop)
                epochs = epochs[metadata['trial_time'].to_numpy() >= tmin] # remove onset response
                x += epochs.get_data()
                del epochs

        # standardization
        x = (x - np.mean(x, axis=0, keepdims=True)) / np.std(x, axis=0, keepdims=True)
        y = (y - np.mean(y, axis=0, keepdims=True)) / np.std(y, axis=0, keepdims=True)  # removes the relative importance of subjects, channels, and time points
        
        # independent regression for every time point and channel
        print('Running regression of subject n°', str(i))
        # subject_betas = np.zeros((x.shape[1], y.shape[1], y.shape[2]))
        subject_r2 = np.zeros((y.shape[1], y.shape[2]))
        if n_folds > 1:
          cv = KFold(n_splits=n_folds)
          cv.get_n_splits(x)
        for c in range(y.shape[1]):
          for t in range(y.shape[2]):
              if n_folds == 1:
                regr = LinearRegression().fit(x[:,c,t], y[:, c, t])
                # subject_betas[:, c, t] = regr.coef_.T
                subject_r2[c, t] = r2_score(y[:,c,t], regr.predict(x[:,c,t]),multioutput='raw_values')
              elif n_folds > 1:
                cv_scores = np.full((alphas.shape[0], n_folds), np.nan)
                for i, alpha in enumerate(alphas):
                  for j, (train, test) in enumerate(cv.split(x)):
                    regr = Ridge(alpha=alpha)
                    regr.fit(x[train,c,t].reshape(-1,1),y[train, c, t])
                    pred = regr.predict(x[test,c,t].reshape(-1,1))
                    cv_scores[i,j] = r2_score(y[test,c,t], pred, multioutput='raw_values')
                #store cv scores and best alphas
                cv_scores = np.mean(cv_scores, axis=1) #mean cross-validated r² across folds
                best_alphas = alphas[np.argmax(cv_scores,axis = 0)] #store best alpha for each timepoint
                cv_scores = np.max(cv_scores, axis = 0) #for each timepoint, keep the highest cross-val r² across alphas
                subject_r2[c, t] = cv_scores
                # #fit again with best alpha and store betas
                # regr = Ridge(best_alphas).fit(x[:,c,:],y[:,c,:])
                # subject_betas[:, c, :] = regr.coef_
          print("Best r² in", epochs.ch_names[c], " :", str(np.max(subject_r2[c,:])))
            
        #store regression results
        # output['betas'].append(subject_betas)
        output['r2'].append(subject_r2)
    # output['betas'] = np.array(output['betas'])
    output['r2'] = np.array(output['r2'])

    with open(save_path + 'NC_meg.pkl', 'wb') as fp:
        pickle.dump(output, fp)

    return output

path = 'D:/2022_deep_music/meg/'
epochs_path = path + 'epochs_note_level/'
save_path = path + 'regression_note_level_test/'

NC = regression_noise_ceiling(epochs_path, save_path, tmin=1, n_folds = 4, alphas = (1e0, 1e5))
times = np.linspace(-1,1,201)
y = NC['r2'][0,:,:]

tw = [-0.4, 1]
lim = [np.abs(times - tw[0]).argmin(), np.abs(times - tw[1]).argmin()]
# y = y[:,:,lim[0]:lim[1]]
y = y[:,lim[0]:lim[1]]
times = times[lim[0]:lim[1]]


ch_max = np.max(y,axis=1)
ch_mask = ch_max > np.percentile(ch_max, 90)
print(sum(ch_mask))
# y_show = y[f_idx, ch_mask,:]
y_show = y[ch_mask,:]

#plot
plt.plot(times,y_show.T)
plt.hlines(0,times[0], times[-1], linestyles='--', color='gray')
plt.legend(ch_names[ch_mask])
# plt.title(features[f_idx])
plt.title('Noise ceiling - R²')
plt.show()