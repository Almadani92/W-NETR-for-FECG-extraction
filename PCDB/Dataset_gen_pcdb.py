from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os

# import wfdb


# input_paths_train = []
# output_paths_train = []


maECG_paths_val = []

# for i in range(0,1370):

    # input_paths_train.append('dataset/inputs/train'+str(i)+'.npy')
    # output_paths_train.append('dataset/outputs/train'+str(i)+'.npy')

sub = ['a03']
	
for kh in range(0,15):
    maECG_paths_val.append('PCDB/maECG/'+sub[0]+'_'+'I'+'_'+str(kh))
    
for kh in range(0,15):
    maECG_paths_val.append('PCDB/maECG/'+sub[0]+'_'+'II'+'_'+str(kh))
    
for kh in range(0,15):
    maECG_paths_val.append('PCDB/maECG/'+sub[0]+'_'+'III'+'_'+str(kh))
    
for kh in range(0,15):
    maECG_paths_val.append('PCDB/maECG/'+sub[0]+'_'+'IV'+'_'+str(kh))

maECG_paths_val = np.array(maECG_paths_val)
np.save('maECG_paths_val.npy',maECG_paths_val)











