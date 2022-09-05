from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import scipy
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, find_peaks, savgol_filter
import numpy as np
import os

import mne
import wfdb


units = ['mV']
sig_name = ['I']
sub = ['r01','r04','r07','r08','r10']

for i in range (0,5): #i 0to 5 subject
    file = sub[i]+'.edf'
    data = mne.io.read_raw_edf(file)
    raw_data = data.get_data()
    fecg = scipy.signal.decimate(raw_data[0,:],4)
    ch1 = scipy.signal.decimate(raw_data[1,:],4)
    ch2 = scipy.signal.decimate(raw_data[2,:],4)
    ch3 = scipy.signal.decimate(raw_data[3,:],4)
    ch4 = scipy.signal.decimate(raw_data[4,:],4)
    
    for kh in range(0,74):
    
        wfdb.wrsamp(sub[i]+'_1_'+str(kh), fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(fecg[1000*kh:1000*(kh+1)], axis=1), write_dir = 'fecg_ground')
        wfdb.wrsamp(sub[i]+'_1_'+str(kh)+'_5', fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(fecg[(1000*kh) +500:(1000*(kh+1)) +500], axis=1), write_dir = 'fecg_ground')
        wfdb.wrsamp(sub[i]+'_1_'+str(kh), fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(ch1[1000*kh:1000*(kh+1)], axis=1), write_dir = 'mixture')
        wfdb.wrsamp(sub[i]+'_1_'+str(kh)+'_5', fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(ch1[(1000*kh) +500:(1000*(kh+1)) +500], axis=1), write_dir = 'mixture')

    for kh in range(0,74):
        wfdb.wrsamp(sub[i]+'_2_'+str(kh), fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(fecg[1000*kh:1000*(kh+1)], axis=1), write_dir = 'fecg_ground')
        wfdb.wrsamp(sub[i]+'_2_'+str(kh)+'_5', fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(fecg[(1000*kh) +500:(1000*(kh+1)) +500], axis=1), write_dir = 'fecg_ground')
        wfdb.wrsamp(sub[i]+'_2_'+str(kh), fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(ch2[1000*kh:1000*(kh+1)], axis=1), write_dir = 'mixture')
        wfdb.wrsamp(sub[i]+'_2_'+str(kh)+'_5', fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(ch2[(1000*kh) +500:(1000*(kh+1)) +500], axis=1), write_dir = 'mixture')

    for kh in range(0,74):
        wfdb.wrsamp(sub[i]+'_3_'+str(kh), fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(fecg[1000*kh:1000*(kh+1)], axis=1), write_dir = 'fecg_ground')
        wfdb.wrsamp(sub[i]+'_3_'+str(kh)+'_5', fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(fecg[(1000*kh) +500:(1000*(kh+1)) +500], axis=1), write_dir = 'fecg_ground')
        wfdb.wrsamp(sub[i]+'_3_'+str(kh), fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(ch3[1000*kh:1000*(kh+1)], axis=1), write_dir = 'mixture')
        wfdb.wrsamp(sub[i]+'_3_'+str(kh)+'_5', fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(ch3[(1000*kh) +500:(1000*(kh+1)) +500], axis=1), write_dir = 'mixture')

    for kh in range(0,74):
        wfdb.wrsamp(sub[i]+'_4_'+str(kh), fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(fecg[1000*kh:1000*(kh+1)], axis=1), write_dir = 'fecg_ground')
        wfdb.wrsamp(sub[i]+'_4_'+str(kh)+'_5', fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(fecg[(1000*kh) +500:(1000*(kh+1)) +500], axis=1), write_dir = 'fecg_ground')
        wfdb.wrsamp(sub[i]+'_4_'+str(kh), fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(ch4[1000*kh:1000*(kh+1)], axis=1), write_dir = 'mixture')
        wfdb.wrsamp(sub[i]+'_4_'+str(kh)+'_5', fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(ch4[(1000*kh) +500:(1000*(kh+1)) +500], axis=1), write_dir = 'mixture')
    
