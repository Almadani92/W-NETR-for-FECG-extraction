from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, find_peaks, savgol_filter
import numpy as np
import os

# import wfdb

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y



sub = ['01','02','03','04','05','06','07','08','09','10']
SNR = ['00', '03', '06', '09', '12']
channel = [1,8,11,14,19,22,25,32]
# channel = [25]
fs = 250

out_len= 75000
#sample number 1200 by subject, 240 sample by SNR

samp_num= 360000  #total 360000
# x_train = np.empty((samp_num,out_len))
# fecg_train = np.empty((samp_num,out_len))
# mecg_train = np.empty((samp_num,out_len))
# noise1_train = np.empty((samp_num,out_len))
# noise2_train = np.empty((samp_num,out_len))


# print(x_train.shape)

# sam_num=0
#channel 1, 8, 11, 14, 19, 22, 25 and 32

fecg_paths_train = []
mecg_paths_train = []
mixture_paths_train = []


fecg_paths_val = []
mecg_paths_val = []
mixture_paths_val = []


fecg_paths_test = []
mecg_paths_test = []
mixture_paths_test = []





for i in range (0,10): #i 0to 10 subject
    for j in range(0,5): #j 0 to 5
        for k in range(1,6): #k k 1to 6
            for c in range(0,6): # c 0to 6
                for kh in range(0,75):
                    for ch in range(0, len(channel)):
                
                        fecg_path = 'fetal-ecg-synthetic-database-1.0.0'+'/fecg_ground'+'/sub'+ sub[i]+'_snr'+ SNR[j]+'dB_l'+ str(k) +'_c' + str(c)+'_fecg1_'+str(kh)+'_'+str(channel[ch])
                        mecg_path = 'fetal-ecg-synthetic-database-1.0.0'+'/mecg_ground'+'/sub'+ sub[i]+'_snr'+ SNR[j]+'dB_l'+ str(k) +'_c' + str(c)+'_mecg_'+str(kh)+'_'+str(channel[ch])
                        mixture_path = 'fetal-ecg-synthetic-database-1.0.0'+'/mixture'+'/sub'+ sub[i]+'_snr'+ SNR[j]+'dB_l'+ str(k) +'_c' + str(c)+'_mixture_'+str(kh)+'_'+str(channel[ch])
                        # print(i)
                        if i==9: 
                            fecg_paths_test.append(fecg_path)
                            mecg_paths_test.append(mecg_path)
                            mixture_paths_test.append(mixture_path)
                            
                        elif i==8: 
                            fecg_paths_val.append(fecg_path)
                            mecg_paths_val.append(mecg_path)
                            mixture_paths_val.append(mixture_path)       
                         

                        else:
                        
                            fecg_paths_train.append(fecg_path)
                            mecg_paths_train.append(mecg_path)
                            mixture_paths_train.append(mixture_path)
                    

                    
fecg_paths_train = np.array(fecg_paths_train)
mecg_paths_train = np.array(mecg_paths_train)
mixture_paths_train = np.array(mixture_paths_train)

fecg_paths_val = np.array(fecg_paths_val)
mecg_paths_val = np.array(mecg_paths_val)
mixture_paths_val = np.array(mixture_paths_val)

fecg_paths_test = np.array(fecg_paths_test)
mecg_paths_test = np.array(mecg_paths_test)
mixture_paths_test = np.array(mixture_paths_test)


np.save('fecg_paths_train.npy',fecg_paths_train)
np.save('mecg_paths_train.npy',mecg_paths_train)
np.save('mixture_paths_train.npy',mixture_paths_train)


np.save('fecg_paths_val.npy',fecg_paths_val)
np.save('mecg_paths_val.npy',mecg_paths_val)
np.save('mixture_paths_val.npy',mixture_paths_val)


np.save('fecg_paths_test.npy',fecg_paths_test)
np.save('mecg_paths_test.npy',mecg_paths_test)
np.save('mixture_paths_test.npy',mixture_paths_test)









