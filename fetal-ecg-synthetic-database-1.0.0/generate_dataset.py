from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, find_peaks, savgol_filter
import numpy as np
import os

import wfdb




sub = ['01','02','03','04','05','06','07','08','09','10']
SNR = ['00', '03', '06', '09', '12']
channel = [1,8,11,14,19,22,25,32]
# channel = [25]
fs = 250

out_len= 75000
#sample number 1200 by subject, 240 sample by SNR

samp_num= 360000  #total 360000




for i in range (0,10): #i 0to 10 subject
    for j in range(0,5): #j 0 to 5
        for k in range(1,6): #k k 1to 6
            for c in range(0,6): # c 0to 6
                fecg_path = 'sub'+ sub[i]+'/snr'+ SNR[j] + 'dB/sub'+ sub[i]+'_snr'+ SNR[j]+'dB_l'+ str(k) +'_c' + str(c)+'_fecg1'
                mecg_path = 'sub'+ sub[i]+'/snr'+ SNR[j] + 'dB/sub'+ sub[i]+'_snr'+ SNR[j]+'dB_l'+ str(k) +'_c' + str(c)+'_mecg'
                # noise1_path = 'sub'+ sub[i]+'/snr'+ SNR[j] + 'dB/sub'+ sub[i]+'_snr'+ SNR[j]+'dB_l'+ str(k) +'_c' + str(c)+'_noise1'
                noise2_path = 'sub'+ sub[i]+'/snr'+ SNR[j] + 'dB/sub'+ sub[i]+'_snr'+ SNR[j]+'dB_l'+ str(k) +'_c' + str(c)+'_noise2'
                # print(i)
                
                fecg, fields = wfdb.rdsamp(fecg_path, sampfrom=0, sampto=75000)
                mecg, fields = wfdb.rdsamp(mecg_path, sampfrom=0, sampto=75000)
                # noise1, fields = wfdb.rdsamp(noise1_path, sampfrom=0, sampto=75000)
                noise2, fields = wfdb.rdsamp(noise2_path, sampfrom=0, sampto=75000)
                mixture = fecg+mecg+noise2
                # print("noise1 size is ----->>>>>>>>>",noise1.shape)
                
                units = ['mV']
                for kh in range(0,75):
                    for ch in range(0, len(channel)):
                        sig_name =  [str(channel[ch])]
                        wfdb.wrsamp('sub'+ sub[i]+'_snr'+ SNR[j]+'dB_l'+ str(k) +'_c' + str(c)+'_fecg1_'+str(kh)+'_'+str(channel[ch]), fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(fecg[1000*kh:1000*(kh+1),channel[ch]], axis=1), write_dir = 'fecg_ground')
                        # print("ranges is ----------->>>>>>>>>>>",1000*kh,'-----------------',1000*(kh+1))
                        wfdb.wrsamp('sub'+ sub[i]+'_snr'+ SNR[j]+'dB_l'+ str(k) +'_c' + str(c)+'_mixture_'+str(kh)+'_'+str(channel[ch]), fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(mixture[1000*kh:1000*(kh+1),channel[ch]], axis=1), write_dir = 'mixture')
                        wfdb.wrsamp('sub'+ sub[i]+'_snr'+ SNR[j]+'dB_l'+ str(k) +'_c' + str(c)+'_mecg_'+str(kh)+'_'+str(channel[ch]), fs = 250, units=units, sig_name=sig_name, p_signal=np.expand_dims(mecg[1000*kh:1000*(kh+1),channel[ch]], axis=1), write_dir = 'mecg_ground')
                

