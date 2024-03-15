#example for running HiTSs


#import needed libraries
import os
import sys
import torch
import numpy as np
 
module_path = os.path.abspath(os.path.join('../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import ResNet as net

# adjustables
lr = 1e-3                     # learning rate
max_epoch = 100000            # the maximum training epoch 
batch_size = 320              # training batch size
n_forward = 4                 # number of steps forward to train
noise = 0.0
step_size = 1                 # step sizes to train on (1,4,16) for fluid
system = fluid
dt = 0.025


data_dir = os.path.join('../../data/', system)    
model_dir = os.path.join('../../models/', system)


#load data
train_data = np.load(os.path.join(data_dir, 'train_noise{}.npy'.format(noise)))
val_data = np.load(os.path.join(data_dir, 'val_noise{}.npy'.format(noise)))
test_data = np.load(os.path.join(data_dir, 'test_noise{}.npy'.format(noise)))
n_train, _, ndim = train_data.shape
n_val = val_data.shape[0]
n_test = test_data.shape[0]


# create dataset object
dataset = net.DataSet(train_data, val_data, test_data, dt, step_size, n_forward)

#create model
model = net.ResNet(arch=arch, dt=dt, step_size=step_size)

# train model
model.train_net(dataset, max_epoch=max_epoch, batch_size=batch_size, lr=lr,
                model_path=os.path.join(model_dir, model_name))


