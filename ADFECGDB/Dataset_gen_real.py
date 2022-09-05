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

fecg_paths_train = []
mixture_paths_train = []


fecg_paths_val = []
mixture_paths_val = []



sub = ['r01','r04','r07','r08','r10']

for i in range (0,5): #i 0to 5 subject
                
    for kh in range(0,74):       
        fecg_path = 'ADFECGDB/fecg_ground/'+sub[i]+'_1_'+str(kh) 
        mixture_path = 'ADFECGDB/mixture/'+sub[i]+'_1_'+str(kh) 
        if sub[i]=='r01':
            fecg_paths_val.append(fecg_path)
            mixture_paths_val.append(mixture_path)    
        else:
            fecg_paths_train.append(fecg_path)
            mixture_paths_train.append(mixture_path)

        fecg_path = 'ADFECGDB/fecg_ground/'+sub[i]+'_1_'+str(kh)+'_5'
        mixture_path = 'ADFECGDB/mixture/'+sub[i]+'_1_'+str(kh)+'_5'
        if sub[i]=='r01':
            fecg_paths_val.append(fecg_path)
            mixture_paths_val.append(mixture_path)    
        else:
            fecg_paths_train.append(fecg_path)
            mixture_paths_train.append(mixture_path)
        
    for kh in range(0,74):       
        fecg_path = 'ADFECGDB/fecg_ground/'+sub[i]+'_2_'+str(kh) 
        mixture_path = 'ADFECGDB/mixture/'+sub[i]+'_2_'+str(kh) 
        if sub[i]=='r04':
            fecg_paths_val.append(fecg_path)
            mixture_paths_val.append(mixture_path)    
        else:
            fecg_paths_train.append(fecg_path)
            mixture_paths_train.append(mixture_path)

        fecg_path = 'ADFECGDB/fecg_ground/'+sub[i]+'_2_'+str(kh)+'_5' 
        mixture_path = 'ADFECGDB/mixture/'+sub[i]+'_2_'+str(kh)+'_5'
        if sub[i]=='r04':
            fecg_paths_val.append(fecg_path)
            mixture_paths_val.append(mixture_path)    
        else:
            fecg_paths_train.append(fecg_path)
            mixture_paths_train.append(mixture_path)

    for kh in range(0,74):       
        fecg_path = 'ADFECGDB/fecg_ground/'+sub[i]+'_3_'+str(kh)
        mixture_path = 'ADFECGDB/mixture/'+sub[i]+'_3_'+str(kh)
        if sub[i]=='r08':
            fecg_paths_val.append(fecg_path)
            mixture_paths_val.append(mixture_path)    
        else:
            fecg_paths_train.append(fecg_path)
            mixture_paths_train.append(mixture_path)

        fecg_path = 'ADFECGDB/fecg_ground/'+sub[i]+'_3_'+str(kh)+'_5'
        mixture_path = 'ADFECGDB/mixture/'+sub[i]+'_3_'+str(kh)+'_5'
        if sub[i]=='r08':
            fecg_paths_val.append(fecg_path)
            mixture_paths_val.append(mixture_path)    
        else:
            fecg_paths_train.append(fecg_path)
            mixture_paths_train.append(mixture_path)  
        
    for kh in range(0,74):            
        fecg_path = 'ADFECGDB/fecg_ground/'+sub[i]+'_4_'+str(kh) 
        mixture_path = 'ADFECGDB/mixture/'+sub[i]+'_4_'+str(kh) 
        if sub[i]=='r10':
            fecg_paths_val.append(fecg_path)
            mixture_paths_val.append(mixture_path)    
        else:
            fecg_paths_train.append(fecg_path)
            mixture_paths_train.append(mixture_path)

        fecg_path = 'ADFECGDB/fecg_ground/'+sub[i]+'_4_'+str(kh)+'_5'
        mixture_path = 'ADFECGDB/mixture/'+sub[i]+'_4_'+str(kh)+'_5'
        if sub[i]=='r10':
            fecg_paths_val.append(fecg_path)
            mixture_paths_val.append(mixture_path)    
        else:
            fecg_paths_train.append(fecg_path)
            mixture_paths_train.append(mixture_path)
        


                    

                    
fecg_paths_train = np.array(fecg_paths_train)
mixture_paths_train = np.array(mixture_paths_train)

fecg_paths_val = np.array(fecg_paths_val)
mixture_paths_val = np.array(mixture_paths_val)


np.save('fecg_paths_train.npy',fecg_paths_train)
np.save('mixture_paths_train.npy',mixture_paths_train)


np.save('fecg_paths_val.npy',fecg_paths_val)
np.save('mixture_paths_val.npy',mixture_paths_val)











