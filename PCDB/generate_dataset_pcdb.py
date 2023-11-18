from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import scipy
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, find_peaks, savgol_filter
import numpy as np
import os

import wfdb
import scipy.io


sub = ['a03']
channel = ['I', 'II', 'III', 'IV']
fs = 250
units = ['mV']
sig_name =  ['I']

maECG, fields = wfdb.rdsamp(sub[0])
maECG_I = scipy.signal.decimate(maECG[:,0],4)
maECG_II = scipy.signal.decimate(maECG[:,1],4)
maECG_III = scipy.signal.decimate(maECG[:,2],4)
maECG_IV = scipy.signal.decimate(maECG[:,3],4)
print(maECG.shape)
for kh in range(0,15):
    wfdb.wrsamp(sub[0]+'_'+'I'+'_'+str(kh), fs = fs, units=units, sig_name=sig_name, p_signal=np.expand_dims(maECG_I[1000*kh:1000*(kh+1)], axis=1), write_dir = 'maECG')
    wfdb.wrsamp(sub[0]+'_'+'II'+'_'+str(kh), fs = fs, units=units, sig_name=sig_name, p_signal=np.expand_dims(maECG_II[1000*kh:1000*(kh+1)], axis=1), write_dir = 'maECG')
    wfdb.wrsamp(sub[0]+'_'+'III'+'_'+str(kh), fs = fs, units=units, sig_name=sig_name, p_signal=np.expand_dims(maECG_III[1000*kh:1000*(kh+1)], axis=1), write_dir = 'maECG')
    wfdb.wrsamp(sub[0]+'_'+'IV'+'_'+str(kh), fs = fs, units=units, sig_name=sig_name, p_signal=np.expand_dims(maECG_IV[1000*kh:1000*(kh+1)], axis=1), write_dir = 'maECG')
    