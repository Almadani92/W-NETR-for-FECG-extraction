import timeit
from init import Options
from networks_real import build_net, update_learning_rate, build_UNETR, My_build_UNETR
from util.dataset_pcdb import Dataset
# from networks import build_net
import logging
import os
import sys
import tempfile
from glob import glob

# import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
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

    
    root = '/home/murad/summer semester/my code/PCDB'

    
    valset = Dataset(root=root, load_set='val', transform=None)
    val_loader = DataLoader(valset, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate, pin_memory=False)

       # build the network

    net = build_UNETR() # UneTR
    # print(net)
    net.cuda()
    # if num_gpus > 1:
        # net = torch.nn.DataParallel(net)
        

    #loss_function = monai.losses.DiceCELoss(sigmoid=True)
    loss_function = torch.nn.HuberLoss()
    # loss_function = torch.nn.L1Loss()
    # loss_function = torch.nn.MSELoss()
    torch.backends.cudnn.benchmark = opt.benchmark
    # lambda1 = 1
    lambda2 = 4500



    if opt.test is True:
        print('Begin testing the network...')
        
        net.load_state_dict(torch.load(opt.output_file+'50'+'_real.pkl'))
        test_fecg_pred=[]
        # test_mecg_pred=[]
        test_inputs=[]
        # test_fecg_label=[]
        
        running_loss = 0.0
        for v, val_data in enumerate(val_loader):
            # get the inputs
            inputs = val_data
            dataset_len = 120
            
            inputs = np.einsum('ijk->ikj', inputs)
            
            inputs = torch.from_numpy(inputs)
            
            inputs = Variable(inputs
            
            inputs = inputs.float().cuda()
    
            start = timeit.default_timer()
            mecg_pred, fecg_pred = net(inputs)
            stop = timeit.default_timer()
            
            test_fecg_pred.append(fecg_pred.cpu().detach().numpy())
            test_inputs.append(inputs.cpu().detach().numpy())

            
            
        # print('val error: %.5f' % (running_loss / (v+1)))
        
        print('Time: ', stop - start) 
        
        test_fecg_pred = np.array(test_fecg_pred,dtype=object)
        test_inputs = np.array(test_inputs,dtype=object)
        
        np.save('test_inputs.npy',test_inputs)
        np.save('test_fecg_pred.npy',test_fecg_pred)

        


if __name__ == "__main__":
    main()
