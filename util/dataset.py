# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import os
import torch.utils.data as data
from PIL import Image

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.signal import butter, lfilter, filtfilt, find_peaks, savgol_filter

import wfdb


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


def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

"""# Load Dataset"""

class Dataset(data.Dataset):

    def __init__(self, root='./', load_set='train', transform=None):
        self.root = root#os.path.expanduser(root)
        self.transform = transform
        self.load_set = load_set  # 'train','val','test'
        


        self.fecg_paths = np.load(os.path.join(root, 'fecg_paths_%s.npy'%self.load_set))
        self.mecg_paths = np.load(os.path.join(root, 'mecg_paths_%s.npy'%self.load_set))
        self.mixture_paths = np.load(os.path.join(root, 'mixture_paths_%s.npy'%self.load_set))
        
        # self.points2d = np.load(os.path.join(root, 'points2d-%s.npy'%self.load_set))
        # self.points3d = np.load(os.path.join(root, 'points3d-%s.npy'%self.load_set))
        
        #if shuffle:
        #    random.shuffle(data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """
        fecg, fields = wfdb.rdsamp(self.fecg_paths[index], sampfrom=0, sampto=1000)
        mecg, fields = wfdb.rdsamp(self.mecg_paths[index], sampfrom=0, sampto=1000)
        mixture, fields = wfdb.rdsamp(self.mixture_paths[index], sampfrom=0, sampto=1000)
    
        mixture = mixture[0:992,0]
        mecg = mecg[0:992,0]
        fecg = fecg[0:992,0]
        # print("mixture shape is -->>>>>>>>>>>>>>>>>",mixture.shape)
        mixture =butter_bandpass_filter(mixture, 3,90, 250, 3)
        fecg =butter_bandpass_filter(fecg, 3,90, 250, 3)
        mecg =butter_bandpass_filter(mecg, 3,90, 250, 3)
        
        mixture = (mixture-np.mean(mixture))/np.var(mixture) 
        fecg = (fecg-np.mean(mixture))/np.var(mixture) 
        mecg = (mecg-np.mean(mixture))/np.var(mixture)
        
        
        
        # b = np.min(mixture)
        # mixture = (mixture - b) 
        # q = np.max(mixture)
        # mixture = mixture / q
        # fecg = (fecg - b) / q
        # mecg = (mecg - b) / q
        
        # print("max and min of mixuter is ->>>", np.min(mixture),"--------------", np.max(mixture))
        
        mixture = np.expand_dims(mixture, axis=1)
        fecg = np.expand_dims(fecg, axis=1)
        mecg = np.expand_dims(mecg, axis=1)



        # if self.transform is not None:
            # image = self.transform(image)

        return mixture, mecg, fecg

    def __len__(self):
        return len(self.fecg_paths)
