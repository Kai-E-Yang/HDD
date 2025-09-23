# import standard modules
import time
import math
import h5py
import os,glob
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.utils as utils
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import StepLR,ExponentialLR,CosineAnnealingLR,LambdaLR
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import json

def mod_pi_half(value):
    return (value + np.pi / 2) % np.pi - np.pi / 2

# Define the data loader and processer
def random_sample_from_dataset(dataset, num_samples):
    indices = torch.randint(0, len(dataset), (num_samples,))
    samples = [dataset[i] for i in indices]

    # Check if the dataset has labels
    if isinstance(samples[0], tuple) and len(samples[0]) == 2:
        data, labels = zip(*samples)
        return torch.stack(data), torch.stack(labels)
    else:
        return torch.stack(samples), None

class CustomDataLoader(DataLoader):
    def __iter__(self):
        for batch in super().__iter__():
            if isinstance(batch, tuple) and len(batch) == 2:
                batch_data, batch_labels = batch
            else:
                batch_data = batch
                batch_labels = None
                
            if batch_data.size(0) < self.batch_size:
                # num_samples_needed = self.batch_size - batch_data.size(0)
                # pad_data, _ = random_sample_from_dataset(self.dataset, num_samples_needed)
                # batch_data = torch.cat([batch_data, pad_data], dim=0)
                batch_data=batch_data
            
            if batch_labels is not None:
                yield batch_data, batch_labels
            else:
                yield batch_data

def load_data(filename='muram_tau_layers.h5',istart=5,hsize=16):
    jump=1
    with h5py.File(filename,'r') as f:
        bx = f['bx'][:][::jump,::jump,istart:istart+hsize]
        by = f['by'][:][::jump,::jump,istart:istart+hsize]
        bz = f['bz'][:][::jump,::jump,istart:istart+hsize]
        # x3d = f['x3d'][:][::jump,::jump,istart:istart+hsize]
        # y3d = f['y3d'][:][::jump,::jump,istart:istart+hsize]
        t3d = f['tz3d'][:][::jump,::jump,istart:istart+hsize]
        bscale=np.nanmax(np.sqrt(f['bx'][:]**2 + f['by'][:]**2 + f['bz'][:]**2))

    bx/=bscale
    by/=bscale
    bz/=bscale

    tmp=bx+1j*by
    phi=np.mod(np.angle(tmp),np.pi)
    phi_1=mod_pi_half(np.angle(tmp))
    bt=np.sqrt(bx**2+by**2)

    bx=bt*np.cos(phi)
    by=bt*np.sin(phi)

    # dx=x3d[1,0,0]-x3d[0,0,0]
    # dy=y3d[0,1,0]-y3d[0,0,0]

    bx1=bt*np.cos(phi_1)
    by1=bt*np.sin(phi_1)

    Zout=np.zeros_like(bx)
    for iz in range(Zout.shape[-1]):
        Zout[...,iz]=iz*0.014398932
    return bx,bx1,by,by1,bz,Zout,t3d,bscale

def process_data(filename='muram_tau_layers.h5',size=128,hsize=16,batch_size=4,pad=8,blockNum=2,device="cpu"):
    print(f'Loading data ... from {filename}')
    data_train=np.zeros((blockNum*blockNum,6,size+2*pad,size+2*pad,hsize))
    bx,bx1,by,by1,bz,Zini,_,_=load_data(filename=filename,hsize=hsize,istart=0)

    for ix in tqdm(range(blockNum)):
        for iy in range(blockNum):
            idx=ix*blockNum+iy
            xstart = ix + 2
            ystart = iy + 2
            data_train[idx,0]=bx[xstart*size-pad:(xstart+1)*size+pad,ystart*size-pad:(ystart+1)*size+pad,:]
            data_train[idx,1]=by[xstart*size-pad:(xstart+1)*size+pad,ystart*size-pad:(ystart+1)*size+pad,:]
            data_train[idx,2]=bz[xstart*size-pad:(xstart+1)*size+pad,ystart*size-pad:(ystart+1)*size+pad,:]
            data_train[idx,3]=Zini[xstart*size-pad:(xstart+1)*size+pad,ystart*size-pad:(ystart+1)*size+pad,:]
            data_train[idx,4]=bx1[xstart*size-pad:(xstart+1)*size+pad,ystart*size-pad:(ystart+1)*size+pad,:]
            data_train[idx,5]=by1[xstart*size-pad:(xstart+1)*size+pad,ystart*size-pad:(ystart+1)*size+pad,:]

    train_dataset = torch.tensor(data_train,dtype=torch.float32)
    train_error = train_dataset[:,:3].clone().detach()*0.0

    return train_dataset,train_error

def load_json_config(custom_path=None):
    
    # If a custom config exists, load and merge it.
    if custom_path and os.path.exists(custom_path):
        with open(custom_path, "r") as f:
            custom_config = json.load(f)
        # config = recursive_merge(config, custom_config)
    
    config_data = custom_config.get("config_data", {})
    config_model = custom_config.get("config_model", {})
    config_loss = custom_config.get("config_loss", {})
    config_train = custom_config.get("config_train", {})

    # Merge or update configurations as needed.
    config_all = {**config_data, **config_loss, **config_train, **config_model}

    return config_all