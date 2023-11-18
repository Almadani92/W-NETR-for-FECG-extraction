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

    def __init__(self, root='./', load_set='val', transform=None):
        self.root = root#os.path.expanduser(root)
        self.transform = transform
        self.load_set = load_set  # 'train','val','test'
        
        self.maECG_paths = np.load(os.path.join(root, 'maECG_paths_%s.npy'%self.load_set))
        
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
        maECG, fields = wfdb.rdsamp(self.maECG_paths[index], sampfrom=0, sampto=992)
        # print(self.mixture_paths[index])
    
        maECG = maECG[0:992,0]
        # print("mixture shape is -->>>>>>>>>>>>>>>>>",mixture.shape)
        maECG =butter_bandpass_filter(maECG, 3,90, 250, 3)
        
        maECG = ((maECG-np.mean(maECG))/np.var(maECG)) *20        
        # maECG = ((maECG-np.mean(maECG))/np.var(maECG)) *6 
        
        
        # b = np.min(mixture)
        # mixture = (mixture - b) 
        # q = np.max(mixture)
        # mixture = mixture / q
        # fecg = (fecg - b) / q
        # mecg = (mecg - b) / q
        
        # print("max and min of mixuter is ->>>", np.min(fecg),"--------------", np.max(fecg))
        
        maECG = np.expand_dims(maECG, axis=1)



        # if self.transform is not None:
            # image = self.transform(image)

        return maECG

    def __len__(self):
        return len(self.maECG_paths)
