
from init import Options
from networks import build_net, update_learning_rate, build_UNETR, My_build_UNETR
from util.dataset import Dataset
import logging
import os
import sys
import tempfile
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (EnsureType, Compose, LoadImaged, AddChanneld, Transpose,Activations,AsDiscrete, RandGaussianSmoothd, CropForegroundd, SpatialPadd,
                              ScaleIntensityd, ToTensord, RandSpatialCropd, Rand3DElasticd, RandAffined, RandZoomd,
    Spacingd, Orientationd, Resized, ThresholdIntensityd, RandShiftIntensityd, BorderPadd, RandGaussianNoised, RandAdjustContrastd,NormalizeIntensityd,RandFlipd)

from monai.visualize import plot_2d_or_3d_image

def main():
    opt = Options().parse()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # check gpus
    if opt.gpu_ids != '-1':
        num_gpus = len(opt.gpu_ids.split(','))
    else:
        num_gpus = 0
    print('number of GPU:', num_gpus)

    
    root = '/home/murad/summer semester/my code/fetal-ecg-synthetic-database-1.0.0'

    valset = Dataset(root=root, load_set='val', transform=None)
    val_loader = DataLoader(valset, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate, pin_memory=False)



       # build the network

    net = build_UNETR() # UneTR
    # print(net)
    net.cuda()

    #loss_function = monai.losses.DiceCELoss(sigmoid=True)
    loss_function = torch.nn.HuberLoss()
    # loss_function = torch.nn.L1Loss()
    # loss_function = torch.nn.MSELoss()
    torch.backends.cudnn.benchmark = opt.benchmark
    lambda1 = 4
    lambda2 = 4500


    if opt.test is True:
    
        print('Begin testing the network...')
        
        net.load_state_dict(torch.load(opt.output_file+'10'+'simulation.pkl'))
        test_fecg_pred=[]
        test_mecg_pred=[]
        test_inputs=[]
        test_fecg_label=[]
        test_mecg_label=[]
        
        running_loss = 0.0
        for v, val_data in enumerate(val_loader):
            # get the inputs
            inputs, mecg_label, fecg_label = val_data
            dataset_len = 90000
            
            inputs = np.einsum('ijk->ikj', inputs)
            mecg_label = np.einsum('ijk->ikj', mecg_label)
            fecg_label = np.einsum('ijk->ikj', fecg_label)
            
            inputs = torch.from_numpy(inputs)
            mecg_label = torch.from_numpy(mecg_label)
            fecg_label = torch.from_numpy(fecg_label)
            
            
            inputs = Variable(inputs)
            mecg_label = Variable(mecg_label)
            fecg_label = Variable(fecg_label)
            
            inputs = inputs.float().cuda()
            mecg_label = mecg_label.float().cuda()
            fecg_label = fecg_label.float().cuda()
    
            mecg_pred, fecg_pred = net(inputs)
            
            test_fecg_pred.append(fecg_pred.cpu().detach().numpy())
            test_mecg_pred.append(mecg_pred.cpu().detach().numpy())
            test_inputs.append(inputs.cpu().detach().numpy())
            test_fecg_label.append(fecg_label.cpu().detach().numpy())
            test_mecg_label.append(mecg_label.cpu().detach().numpy())

            loss = (lambda2)* loss_function(fecg_pred, fecg_label)
            running_loss += loss.data
            
            
        print('val error: %.5f' % (running_loss / (v+1)))
        
        test_fecg_pred = np.array(test_fecg_pred,dtype=object)
        test_mecg_pred = np.array(test_mecg_pred,dtype=object)
        test_inputs = np.array(test_inputs,dtype=object)
        test_fecg_label = np.array(test_fecg_label,dtype=object)
        test_mecg_label = np.array(test_mecg_label,dtype=object)
        
        np.save('test_inputs.npy',test_inputs)
        np.save('test_fecg_pred.npy',test_fecg_pred)
        np.save('test_fecg_label.npy',test_fecg_label)
        np.save('test_mecg_pred.npy',test_mecg_pred)
        np.save('test_mecg_label.npy',test_mecg_label)

if __name__ == "__main__":
    main()
