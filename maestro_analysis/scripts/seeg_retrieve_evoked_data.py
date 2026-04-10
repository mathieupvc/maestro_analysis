import mne
mne.set_log_level(verbose=False)
mne.utils.set_config('MNE_USE_CUDA', 'true')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pickle

plt.switch_backend('QtAgg')
path = 'F:/2022_deep_music/seeg/'
sub_list = ['Bou_Al', 'Bou_Ni', 'Dav_To', 'Dri_Am', 'Gra_Lu', 'Pen_Va', 'Sou_Fa', 'The_Mo', 'Tou_Et', 'Tou_Ma']
save_path = path + 'evoked/'
epochs_tw = (-1,10)
baseline_tw = (-1, 0)
#previously : epochs tw = -0.5,1
#           : baseline = None

#parameters for automatic epochs/channel rejection
#list order corresponds to subject order
#These rejection thresholds were set by hand after visual inspection. They will be applied as : mean(max(abs(Y))) + thresh * std(max(abs(Y)))
reject_thresh_ch = [10, 10, 10, 4, 10, 10, 10, 10, 10, 10] #Bou_Ni : 10, Tou_Et : 10, Sou_Fa : 10, The_Mo = 10, Dri_Am : 4 , Gra_Lu : 10, Dav_To : 10, Bou_Al : 10, Pen_Va : 10, Tou_Ma :10
reject_thresh_e = [10, 10, 10, 6, 10, 10, 10, 10, 10, 10] #Bou_Ni : 10, Tou_Et : 10, Sou_Fa : 10, The_Mo = 10, Dri_Am : 6 , Gra_Lu : 10, Dav_To : 10, Bou_Al : 10, Pen_Va : 10, Tou_Ma : 10

out = {} #store evoked response
for i_sub, sub in enumerate(sub_list): #sub_list
    print('---',sub, '---' )
    #load data
    preprocessed_path = path + 'data_mne_fif/' + sub + '/preprocessed_IDA/'
    raw = mne.io.read_raw_fif(preprocessed_path + sub + '_raw_mono_preprocessed_IDA.fif',preload=True)
    fs = raw.info['sfreq']
    print(raw.info)

    #trial lists
    if sub == 'Bou_Ni':
        trial_list = [_ for _ in range(90,180)] #only second block for Bou_Ni
        trial_list = [_ for _ in range(146)] #trial missing for Pen_Va
    else:
        trial_list = [_ for _ in range(180)]

    #retrieve events
    events, _ = mne.events_from_annotations(raw)
    if sub == 'Tou_Et':
        events = events[np.where(np.isin(events[:,2], [3,4,5,7]))[0],:] #keep triggers at stim start only
        events[np.where(np.isin(events[:,2], [3,4,5]))[0],2] = 1 #redefine stim start
        events[np.where(np.isin(events[:,2], [7]))[0],2] = 2 #redefine stim end
    if sub == 'Bou_Ni':
        events = events[np.where(np.isin(events[:,2], [6,7,8,11]))[0],:]
        events[np.where(np.isin(events[:,2], [6,7,8]))[0],2] = 1
        events[np.where(np.isin(events[:,2], [11]))[0],2] = 2 
    if sub == 'Sou_Fa':
        events = events[np.where(np.isin(events[:,2], [6,7,8,10]))[0],:] 
        events[np.where(np.isin(events[:,2], [6,7,8]))[0],2] = 1 
        events[np.where(np.isin(events[:,2], [10]))[0],2] = 2
    if sub == 'The_Mo':
        events = events[np.where(np.isin(events[:,2], [6,7,8,5]))[0],:] 
        events[np.where(np.isin(events[:,2], [6,7,8]))[0],2] = 1
        events[np.where(np.isin(events[:,2], [5]))[0],2] = 2
    if sub == 'Dri_Am':
        events = events[np.where(np.isin(events[:,2], [3,4,5,8]))[0],:] 
        events[np.where(np.isin(events[:,2], [3,4,5]))[0],2] = 1 
        events[np.where(np.isin(events[:,2], [8]))[0],2] = 2 
    if sub == 'Gra_Lu':
        events = events[np.where(np.isin(events[:,2], [6,7,8,13]))[0],:] 
        events[np.where(np.isin(events[:,2], [6,7,8]))[0],2] = 1 
        events[np.where(np.isin(events[:,2], [13]))[0],2] = 2 
    if sub == 'Dav_To':
        events = events[np.where(np.isin(events[:,2], [6,7,8,5]))[0],:]
        events[np.where(np.isin(events[:,2], [6,7,8]))[0],2] = 1 
        events[np.where(np.isin(events[:,2], [5]))[0],2] = 2 
    if sub == 'Bou_Al':
        events = events[np.where(np.isin(events[:,2], [4,5,6,9]))[0],:] 
        events[np.where(np.isin(events[:,2], [4,5,6]))[0],2] = 1 
        events[np.where(np.isin(events[:,2], [9]))[0],2] = 2 
    if sub == 'Pen_Va':
        events = events[np.where(np.isin(events[:,2], [6,7,8,5]))[0],:] 
        events[np.where(np.isin(events[:,2], [6,7,8]))[0],2] = 1 
        events[np.where(np.isin(events[:,2], [5]))[0],2] = 2 
    if sub == 'Tou_Ma':
        events = events[np.where(np.isin(events[:,2], [4,5,6,8]))[0],:] 
        events[np.where(np.isin(events[:,2], [4,5,6]))[0],2] = 1 
        events[np.where(np.isin(events[:,2], [8]))[0],2] = 2
    events_dict = {'Stim': 1, 'Stim_end':2}

    #make sure than trial durations (time between trigger start and trigger end) are correct
    bad_e_idx = []
    if sub != 'Bou_Ni' and sub != 'Gra_Lu' and sub != 'Pen_Va': #trial duration is not consistent for Bou_Ni,
        i_trial = 0                                                                             # 2 trigg stim_end missing for Pen_Va
        for i_event in range(events.shape[0]):
            if i_event%2 == 1:
                trial_duration = (events[i_event,0] - events[i_event-1,0])/raw.info['sfreq']
                if trial_duration < 8.5 or trial_duration > 8.52:
                    bad_e_idx.append(trial_list[i_trial])
                    print('Trial number ' + str() + ' was rejected because of wrong duration : ' + str(trial_duration))
                i_trial += 1

    #define epochs
    epochs_trial = mne.Epochs(raw, events, event_id=events_dict, tmin=epochs_tw[0], tmax=epochs_tw[1], baseline=baseline_tw, preload=True,event_repeated='error')
    epochs_trial = epochs_trial["Stim"]

    # ---- Channel/epochs rejection -----
    #retrieve data
    Y = epochs_trial.get_data()
    hm = np.max(np.abs(Y),axis=2).T #get extrem value of each epoch

    #channel rejection
    CH = np.mean(hm,axis=1)
    CH_thresh = np.mean(CH)+reject_thresh_ch[i_sub]*np.std(CH)
    bad_ch_idx = []
    for i, ch in enumerate(raw.ch_names):
        if CH[i] > CH_thresh:
            bad_ch_idx.append(i)
    bad_ch = [epochs_trial.ch_names[idx] for idx in bad_ch_idx]

    if len(bad_ch)>0:
        print("Dropping bad channels" + str(bad_ch))
        epochs_trial.drop_channels(bad_ch)
    else:
        print("No bad channels \n")

    #epoch rejection
    E = np.mean(hm,axis=0)
    E_thresh = np.mean(E)+reject_thresh_e[i_sub]*np.std(E)
    for i in range(len(E)):
        if E[i] > E_thresh:
            bad_e_idx.append(i)

    
    if len(bad_e_idx)>0:
        print("Dropping bad epochs : " + str(np.array(trial_list)[bad_e_idx]))
        epochs_trial.drop(bad_e_idx)
        kept_epochs = [_ for _ in trial_list if _ not in bad_e_idx] #keep track of which epoch/stim was kept
    else:
        print("No bad epochs")
        kept_epochs = trial_list

    # #check output
    # picks = [ch for ch in epochs_trial.ch_names if 'H' in ch and 'GPH' not in ch]
    # epochs_trial.plot_image(picks=picks, show=True)

    #shape : epochs, channels, timepoints
    Y = epochs_trial.get_data()
    # Y[:,bad_ch,:] = np.nan #another way to do channel rejection, without dropping anything
    # Y[bad_e_idx,:,:] = np.nan
    
    #prepare output
    out_sub = {}
    out_sub['epochs'] = epochs_trial
    out_sub['epochs_tw'] = epochs_tw
    out_sub['baseline_tw'] = baseline_tw
    out_sub['trial_list'] = trial_list
    out_sub['kept_epochs'] = kept_epochs
    out_sub['bad_ch'] = bad_ch
    out_sub['bad_ch_idx'] = bad_ch_idx
    out_sub['bad_e_idx'] = bad_e_idx

    out[sub] = out_sub

with open(save_path + 'evoked_data.pkl', 'wb') as handle:
    pickle.dump(out, handle)