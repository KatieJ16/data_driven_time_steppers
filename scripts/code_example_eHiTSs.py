#example for running ensemble HiTSs

#import needed libraries
import os
import sys
import torch
import numpy as np
 
module_path = os.path.abspath(os.path.join('../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import Resnet_multiscale_general as net

# adjustables
lr = 1e-3                     # learning rate
max_epoch = 100000            # the maximum training epoch 
batch_size = 320              # training batch size
n_forward = 3                 # number of steps forward to train
noise = 0.0
step_size = 1
system = fluid
dt = 0.025
step_sizes  = [1, 4, 16]
combos_file = "all_combos_fluid4.npy"


data_dir = os.path.join('../../data/', system)    
model_dir = os.path.join('../../models/', system)


#load data
train_data = np.load(os.path.join(data_dir, 'train_noise{}.npy'.format(noise)))
val_data = np.load(os.path.join(data_dir, 'val_noise{}.npy'.format(noise)))
test_data = np.load(os.path.join(data_dir, 'test_noise{}.npy'.format(noise)))
n_train, _, ndim = train_data.shape
n_val = val_data.shape[0]
n_test = test_data.shape[0]

arch = [ndim, 256, ndim]


#create model
model = net.ResNet(arch=arch, dt=dt, step_sizes=step_sizes, n_poss=n_poss, combos_file = combos_file)

# train large first, then all together (just training a little so it doesn't take ages
#only train indiviual when we are making new object
for i in  step_sizes:     
    print("step_size = ", i)
    dataset = net.DataSet(train_data, val_data, test_data, dt, i, n_forward)
    model.train_net_single(dataset, max_epoch=1000, batch_size=batch_size, lr=lr,
                    model_path=os.path.join(model_dir, model_name), print_every=100, type=str(i))
# training
n_forward = min(int(np.max(step_sizes)*4/np.min(step_sizes)) + 1, int(train_data.shape[1] / np.min(step_sizes))-1)
print("n_forward = ", n_forward)
dataset = net.DataSet(train_data, val_data, test_data, dt, smallest_step, n_forward)
model.train_net(dataset, max_epoch=max_epoch, batch_size=batch_size, lr=lr,
                model_path=os.path.join(model_dir, model_name), print_every=1)


