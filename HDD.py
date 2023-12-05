# This code is Hawaii Disambiguity Decoder (HDD) found by SPIN4D project.
# It is used to disambiguate the azimuthal angle of 
# the magnetic field from the Stokes polarimetric data inversion
# It can give disambiguate the horizontal magnetic field with 180 degree
# ambiguity and give the prediction of the vertical height based on the 
# divergence equaton only. 
# This is only a very week constraint on the data but works well.
# v0.0.1 2023/11/17 K. Y.
# yangkai@hawaii.edu

# import standard modules
import numpy as np
import os,glob
import time

# import torch.autograd as autograd         # computation graph
import torch.nn as nn                     # neural networks
# import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
# from collections import OrderedDict

import torch
# from torch import Tensor                  # tensor node in the computation graph
import torch.nn.init as init
import torch.nn.utils as utils
# from torch import autograd
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# from torchvision import models

# from ray import tune
# from ray.air import Checkpoint, session
# from ray.tune.schedulers import ASHAScheduler


import h5py
import math

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

def load_data(filename='muram_tau_layers.h5',size=128,istart=5,hsize=16,xstart=0,ystart=0,pad=8):
    with h5py.File(filename,'r') as f:
        bx = f['bx'][:][xstart*size-pad:(xstart+1)*size+pad,ystart*size-pad:(ystart+1)*size+pad,istart:istart+hsize]
        by = f['by'][:][xstart*size-pad:(xstart+1)*size+pad,ystart*size-pad:(ystart+1)*size+pad,istart:istart+hsize]
        bz = f['bz'][:][xstart*size-pad:(xstart+1)*size+pad,ystart*size-pad:(ystart+1)*size+pad,istart:istart+hsize]
        x3d = f['x3d'][:][xstart*size-pad:(xstart+1)*size+pad,ystart*size-pad:(ystart+1)*size+pad,istart:istart+hsize]
        y3d = f['y3d'][:][xstart*size-pad:(xstart+1)*size+pad,ystart*size-pad:(ystart+1)*size+pad,istart:istart+hsize]
        t3d = f['tz3d'][:][xstart*size-pad:(xstart+1)*size+pad,ystart*size-pad:(ystart+1)*size+pad,istart:istart+hsize]
        bscale=np.nanmax(np.sqrt(f['bx'][:][...,istart:istart+hsize]**2 + f['by'][:][...,istart:istart+hsize]**2 + f['bz'][:][...,istart:istart+hsize]**2))

    bx/=bscale
    by/=bscale
    bz/=bscale
    tmp=bx+1j*by
    phi=np.mod(np.angle(tmp),np.pi)
    bt=np.sqrt(bx**2+by**2)
    bx=bt*np.cos(phi)
    by=bt*np.sin(phi)
    random_array = np.random.rand(bx.shape[0],bx.shape[1],bx.shape[2])
    random_array = np.where(random_array > 0.5, 1., -1.)
    bx*=random_array
    by*=random_array

    dx=x3d[1,0,0]-x3d[0,0,0]
    dy=y3d[0,1,0]-y3d[0,0,0]

    Zout=np.zeros_like(bx)
    for iz in range(Zout.shape[-1]):
        Zout[...,iz]=iz*max(dx,dy)
    return bx,by,bz,dx,dy,Zout,t3d

def process_data(filename='muram_tau_layers.h5',size=128,hsize=16,batch_size=4,pad=8,blockNum=2,device="cpu"):
    # blockNum=2
    data_train=np.zeros((blockNum*blockNum,4,size+2*pad,size+2*pad,hsize))
    print('Loading data ...')
    for ix in range(blockNum):
        for iy in range(blockNum):
            idx=ix*blockNum+iy
            bx,by,bz,dx,dy,Zini,t3d=load_data(filename=filename,size=size,hsize=hsize,istart=3,xstart=ix+2,ystart=iy+1,pad=pad)
            data_train[idx,0]=bx
            data_train[idx,1]=by
            data_train[idx,2]=bz
            data_train[idx,3]=Zini

    train_dataset = torch.tensor(data_train,dtype=torch.float32)
    return train_dataset,dx,dy,t3d

class UNet3D_total(nn.Module):
    def __init__(self, in_channels, out_channels, dx=0.1, dy=0.1, deep=1, nchan=64):
        super(UNet3D_total, self).__init__()
        self.deep = deep
        self.dh = max(dx, dy)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder
        for ix in range(deep):
            if ix == 0:
                self.encoders.append(self.conv_block(in_channels, nchan))
                self.pools.append(nn.AvgPool3d(kernel_size=2))
            else:
                self.encoders.append(self.conv_block(2**(ix-1)*nchan, 2**ix*nchan))
                self.pools.append(nn.AvgPool3d(kernel_size=2))
            
        # Bottleneck
        self.bottleneck = self.conv_block(2**(deep-1)*nchan, 2**deep*nchan)

        # Decoder
        for ix in range(deep):
            self.upconvs.append(nn.ConvTranspose3d(2**(ix+1)*nchan, 2**ix*nchan, kernel_size=2, stride=2))
            self.decoders.append(self.conv_block(2**(ix+1)*nchan, 2**ix*nchan))

        # Final output
        self.outconv = nn.Conv3d(nchan, out_channels, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder
        connections = []
        for i in range(self.deep):
            x = self.encoders[i](x)
            connections.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(self.deep - 1, -1, -1):
            x = self.upconvs[i](x)
            x = torch.cat((connections[i], x), dim=1)
            x = self.decoders[i](x)

        # Output
        out = self.outconv(x)
        tmp,_ = torch.sort(out[:,-1:,:,:,:],dim=-1)
        out[:,-1:,:,:,:]=tmp
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class UNet3DZ(nn.Module):
    def __init__(self, in_channels, out_channels, dx=0.1, dy=0.1, deep=1, nchan=64):
        super(UNet3DZ, self).__init__()
        self.deep = deep
        self.dh = max(dx, dy)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder
        for ix in range(deep):
            if ix == 0:
                self.encoders.append(self.conv_block(in_channels, nchan))
                self.pools.append(nn.AvgPool3d(kernel_size=2))
            else:
                self.encoders.append(self.conv_block(2**(ix-1)*nchan, 2**ix*nchan))
                self.pools.append(nn.AvgPool3d(kernel_size=2))
            
        # Bottleneck
        self.bottleneck = self.conv_block(2**(deep-1)*nchan, 2**deep*nchan)

        # Decoder
        for ix in range(deep):
            self.upconvs.append(nn.ConvTranspose3d(2**(ix+1)*nchan, 2**ix*nchan, kernel_size=2, stride=2))
            self.decoders.append(self.conv_block(2**(ix+1)*nchan, 2**ix*nchan))

        # Final output
        self.outconv = nn.Conv3d(nchan, out_channels, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder
        connections = []
        for i in range(self.deep):
            x = self.encoders[i](x)
            connections.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(self.deep - 1, -1, -1):
            x = self.upconvs[i](x)
            x = torch.cat((connections[i], x), dim=1)
            x = self.decoders[i](x)

        # Output
        out = self.outconv(x)
        out,_ = torch.sort(out,dim=-1)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, dx=0.1, dy=0.1, deep=1, nchan=64):
        super(UNet3D, self).__init__()
        self.deep = deep
        self.dh = max(dx, dy)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder
        for ix in range(deep):
            if ix == 0:
                self.encoders.append(self.conv_block(in_channels, nchan))
                self.pools.append(nn.AvgPool3d(kernel_size=2))
            else:
                self.encoders.append(self.conv_block(2**(ix-1)*nchan, 2**ix*nchan))
                self.pools.append(nn.AvgPool3d(kernel_size=2))
            
        # Bottleneck
        self.bottleneck = self.conv_block(2**(deep-1)*nchan, 2**deep*nchan)

        # Decoder
        for ix in range(deep):
            self.upconvs.append(nn.ConvTranspose3d(2**(ix+1)*nchan, 2**ix*nchan, kernel_size=2, stride=2))
            self.decoders.append(self.conv_block(2**(ix+1)*nchan, 2**ix*nchan))

        # Final output
        self.outconv = nn.Conv3d(nchan, out_channels, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder
        connections = []
        for i in range(self.deep):
            x = self.encoders[i](x)
            connections.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(self.deep - 1, -1, -1):
            x = self.upconvs[i](x)
            x = torch.cat((connections[i], x), dim=1)
            x = self.decoders[i](x)

        # Output
        out = self.outconv(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class UNet3DZ_res(nn.Module):
    def __init__(self, in_channels, out_channels,dx=0.1,dy=0.1,deep=1,nchan=64):
        super(UNet3DZ_res, self).__init__()
        self.deep=deep
        self.dh=max(dx,dy)
        # Encoder
        self.enc1 = self.conv_block(in_channels, nchan)
        # self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.pool1 = nn.AvgPool3d(kernel_size=2)

        if self.deep==2:
            self.enc2 = self.conv_block(nchan, 2*nchan)
            # self.pool2 = nn.MaxPool3d(kernel_size=2)
            self.pool2 = nn.AvgPool3d(kernel_size=2)

        # Bottleneck
        if self.deep==2:
            self.bottleneck = self.conv_block(2*nchan, 4*nchan)
        elif self.deep==1:
            self.bottleneck = self.conv_block(nchan, 2*nchan)

        # Decoder
        if self.deep==2:
            self.upconv2 = nn.ConvTranspose3d(4*nchan, 2*nchan, kernel_size=2, stride=2)
            self.dec2 = self.conv_block(4*nchan, 2*nchan)

        self.upconv1 = nn.ConvTranspose3d(2*nchan, nchan, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(2*nchan, nchan)

        # Final output
        self.outconv = nn.Conv3d(nchan, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),  # BatchNorm added
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),  # BatchNorm added
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        if self.deep==2:
            e2 = self.enc2(p1)
            p2 = self.pool2(e2)

        # Bottleneck
        if self.deep==1:
            b = self.bottleneck(p1)
        elif self.deep==2:
            b = self.bottleneck(p2)

        # Decoder
        if self.deep==2:
            d2 = self.upconv2(b)
            d2 = torch.cat((e2, d2), dim=1)
            d2 = self.dec2(d2)

        if self.deep==2:
            d1 = self.upconv1(d2)
        elif self.deep==1:
            d1 = self.upconv1(b)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        # Output
        out = self.outconv(d1) + x[:,3:4]
        return out
    
class UNet3DTotal_res(nn.Module):
    def __init__(self, in_channels, out_channels,dx=0.1,dy=0.1,deep=1,nchan=64):
        super(UNet3DTotal_res, self).__init__()
        self.deep=deep
        self.dh=max(dx,dy)
        # Encoder
        # nchan0=int(nchan/4)
        # nchan1=int(3*nchan/4)

        # self.enc1_dilation = self.conv_block_dilation(in_channels, nchan0)
        # self.enc1 = self.conv_block(in_channels, nchan1)

        self.enc1 = self.conv_block(in_channels, nchan)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        # self.pool1 = nn.AvgPool3d(kernel_size=2)

        if self.deep==2:
            self.enc2 = self.conv_block(nchan, 2*nchan)
            self.pool2 = nn.MaxPool3d(kernel_size=2)
            # self.pool2 = nn.AvgPool3d(kernel_size=2)

        # Bottleneck
        if self.deep==2:
            self.bottleneck = self.conv_block(2*nchan, 4*nchan)
        elif self.deep==1:
            self.bottleneck = self.conv_block(nchan, 2*nchan)

        # Decoder
        if self.deep==2:
            self.upconv2 = nn.ConvTranspose3d(4*nchan, 2*nchan, kernel_size=2, stride=2)
            self.dec2 = self.conv_block(4*nchan, 2*nchan)

        self.upconv1 = nn.ConvTranspose3d(2*nchan, nchan, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(2*nchan, nchan)

        # Final output
        self.outconv = nn.Conv3d(nchan, out_channels, kernel_size=1)
    def conv_block_dilation(self, in_channels, out_channels, dilation=1):
        # print(f"Debug: in_channels={in_channels}, out_channels={out_channels}, dilation={dilation}")
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block


    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),  # BatchNorm added
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),  # BatchNorm added
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder
        # e1_d = self.enc1_dilation(x)
        e1 = self.enc1(x)
        # e1 = torch.cat((e1, e1_d), dim=1)
        p1 = self.pool1(e1)

        if self.deep==2:
            e2 = self.enc2(p1)
            p2 = self.pool2(e2)

        # Bottleneck
        if self.deep==1:
            b = self.bottleneck(p1)
        elif self.deep==2:
            b = self.bottleneck(p2)

        # Decoder
        if self.deep==2:
            d2 = self.upconv2(b)
            d2 = torch.cat((e2, d2), dim=1)
            d2 = self.dec2(d2)

        if self.deep==2:
            d1 = self.upconv1(d2)
        elif self.deep==1:
            d1 = self.upconv1(b)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        out = self.outconv(d1) 
        # Output
        # for idz in range(out.shape[4]):
            # out[:,3:4,:,:,idz]+=idz*self.dh*2
        out[:,3:4] += x[:,3:4]
        return out

def create_model(inchannels=3,outchannels=6,dx=0.1,dy=0.1,nchan=64,deep=1,device="cpu"):
    return UNet3D(inchannels, outchannels,dx=dx,dy=dy,nchan=nchan,deep=deep).to(device)

def create_model_z(inchannels=3,outchannels=1,dx=0.1,dy=0.1,nchan=64,deep=1,device="cpu"):
    return UNet3DZ(inchannels, outchannels,dx=dx,dy=dy,nchan=nchan,deep=deep).to(device)

def create_model_total(inchannels=3,outchannels=4,dx=0.1,dy=0.1,nchan=64,deep=1,device="cpu"):
    return UNet3D_total(inchannels, outchannels,dx=dx,dy=dy,nchan=nchan,deep=deep).to(device)

# def create_model_z_res(inchannels=3,outchannels=1,dx=0.1,dy=0.1,nchan=64,deep=1,device="cpu"):
#     return UNet3DZ_res(inchannels, outchannels,dx=dx,dy=dy,nchan=nchan,deep=deep).to(device)

# def create_model_total_res(inchannels=4,outchannels=4,dx=0.1,dy=0.1,nchan=64,deep=1,device="cpu"):
#     return UNet3DTotal_res(inchannels, outchannels,dx=dx,dy=dy,nchan=nchan,deep=deep).to(device)

# define the custom loss function
class CustomLoss_Z(nn.Module):
    def __init__(self,dx=1,dy=1,nlayer=16,eps=1e-10,w_div=1e9,w_mon=1e8,w_smooth=1e1,w_std=1e2):
        super(CustomLoss_Z, self).__init__()
        self.dx=dx
        self.dy=dy
        self.dh=max(dx,dy)
        self.w_div=w_div
        self.w_mon=w_mon
        self.w_std=w_std
        self.w_smooth=w_smooth
        self.logthickness2=math.log((nlayer*self.dh)**2)
        self.eps = eps  # Small constant to avoid log(0)
    def forward(self, outputs, targets):
        # get the observed B field from the input.
        bx_t,by_t,bz_t=targets[:,0:1],targets[:,1:2],targets[:,2:3]
        # get the predicted height z from the NNs.
        z_p = outputs[:,0:1]


        # enforce the predicted height for each tau layer is smooth
        dz=z_p[...,1:] - z_p[...,0:-1]
        dz2=dz**2
        # laplacedz2 = 4*dz2[:,:,1:-1,1:-1,1:-1] - \
        #             dz2[:,:,:-2,1:-1,1:-1] - \
        #             dz2[:,:,2:,1:-1,1:-1] - \
        #             dz2[:,:,1:-1,:-2,1:-1] - \
        #             dz2[:,:,1:-1,2:,1:-1]

        laplacedz2 = 6*dz2[:,:,1:-1,1:-1,1:-1] - \
                    dz2[:,:,:-2,1:-1,1:-1] - \
                    dz2[:,:,2:,1:-1,1:-1] - \
                    dz2[:,:,1:-1,:-2,1:-1] - \
                    dz2[:,:,1:-1,2:,1:-1] - \
                    dz2[:,:,1:-1,1:-1,2:] - \
                    dz2[:,:,1:-1,1:-1,:-2]
        loss_smooth=torch.mean(laplacedz2**2)

        std_array = torch.zeros((dz2.shape[0],dz2.shape[1],dz2.shape[-1]))
        for i_batch in range(std_array.shape[0]):
            for i_chan in range(std_array.shape[1]):
                for i_z in range(std_array.shape[2]):
                    std_array[i_batch,i_chan,i_z]=torch.std(dz[i_batch,i_chan,:,:,i_z])

        loss_std=torch.mean(std_array)

        # define the divergence of the predicted field in the integrated form.
        bx_000,by_000,bz_000,z_000=bx_t[:,:,0:-1,0:-1,0:-1],by_t[:,:,0:-1,0:-1,0:-1],bz_t[:,:,0:-1,0:-1,0:-1],z_p[:,:,0:-1,0:-1,0:-1]
        bx_100,by_100,bz_100,z_100=bx_t[:,:,1:,0:-1,0:-1],by_t[:,:,1:,0:-1,0:-1],bz_t[:,:,1:,0:-1,0:-1],z_p[:,:,1:,0:-1,0:-1]
        bx_010,by_010,bz_010,z_010=bx_t[:,:,0:-1,1:,0:-1],by_t[:,:,0:-1,1:,0:-1],bz_t[:,:,0:-1,1:,0:-1],z_p[:,:,0:-1,1:,0:-1]
        bx_110,by_110,bz_110,z_110=bx_t[:,:,1:,1:,0:-1],by_t[:,:,1:,1:,0:-1],bz_t[:,:,1:,1:,0:-1],z_p[:,:,1:,1:,0:-1]

        bx_001,by_001,bz_001,z_001=bx_t[:,:,0:-1,0:-1,1:],by_t[:,:,0:-1,0:-1,1:],bz_t[:,:,0:-1,0:-1,1:],z_p[:,:,0:-1,0:-1,1:]
        bx_101,by_101,bz_101,z_101=bx_t[:,:,1:,0:-1,1:],by_t[:,:,1:,0:-1,1:],bz_t[:,:,1:,0:-1,1:],z_p[:,:,1:,0:-1,1:]
        bx_011,by_011,bz_011,z_011=bx_t[:,:,0:-1,1:,1:],by_t[:,:,0:-1,1:,1:],bz_t[:,:,0:-1,1:,1:],z_p[:,:,0:-1,1:,1:]
        bx_111,by_111,bz_111,z_111=bx_t[:,:,1:,1:,1:],by_t[:,:,1:,1:,1:],bz_t[:,:,1:,1:,1:],z_p[:,:,1:,1:,1:]

        loss_div=torch.mean((\
                0.25*(bx_100+bx_110+bx_101+bx_111)*self.dy*0.5*(torch.abs(z_101-z_100)+torch.abs(z_111-z_110))-\
                0.25*(bx_000+bx_010+bx_001+bx_011)*self.dy*0.5*(torch.abs(z_001-z_000)+torch.abs(z_011-z_010))+\
                0.25*(by_010+by_110+by_011+by_111)*self.dx*0.5*(torch.abs(z_011-z_010)+torch.abs(z_111-z_110))-\
                0.25*(by_000+by_100+by_001+by_101)*self.dx*0.5*(torch.abs(z_001-z_000)+torch.abs(z_101-z_100))+\
                0.25*(bz_001 + bz_011 + bz_101 + bz_111)*self.dx*self.dy-\
                0.25*(bz_000 + bz_010 + bz_100 + bz_110)*self.dx*self.dy+\
                (bx_001+bx_101+bx_111)*self.dy*(z_001-z_101)/6+\
                (bx_011+bx_111+bx_101)*self.dy*(z_011-z_111)/6+\
                (by_101+by_111+by_011)*self.dx*(z_101-z_111)/6+\
                (by_001+by_011+by_111)*self.dx*(z_001-z_011)/6-\
                (bx_000+bx_100+bx_110)*self.dy*(z_000-z_100)/6-\
                (bx_010+bx_110+bx_100)*self.dy*(z_010-z_110)/6-\
                (by_100+by_110+by_010)*self.dx*(z_100-z_110)/6-\
                (by_000+by_010+by_110)*self.dx*(z_000-z_010)/6
                )**2/(\
                ((bx_000+bx_001+bx_010+bx_011 + bx_100+bx_101+bx_110+bx_111)*0.125)**2+\
                ((by_000+by_001+by_010+by_011 + by_100+by_101+by_110+by_111)*0.125)**2+\
                ((bz_000+bz_001+bz_010+bz_011 + bz_100+bz_101+bz_110+bz_111)*0.125)**2+1e-10\
                ))

        loss_div*=self.w_div
        loss_std*=self.w_std
        loss_smooth*=self.w_smooth     
        return loss_div,loss_smooth+loss_std


class CustomLoss_Z_total(nn.Module):
    def __init__(self,dx=1,dy=1,nlayer=16,eps=1e-10,w_div=1e9,w_mon=1e8,w_smooth=1e1,w_b=1e9):
        super(CustomLoss_Z_total, self).__init__()
        self.dx=dx
        self.dy=dy
        self.dh=max(dx,dy)
        self.w_b=w_b
        self.w_div=w_div
        self.w_mon=w_mon
        self.w_smooth=w_smooth
        self.logthickness2=math.log((nlayer*self.dh)**2)
        self.eps = eps  # Small constant to avoid log(0)
    def forward(self, outputs, targets):
        # get the observed B field from the input.
        bx_t,by_t,bz_t=targets[:,0:1],targets[:,1:2],targets[:,2:3]
        bx_p,by_p,bz_p,z_p=outputs[:,0:1],outputs[:,1:2],outputs[:,2:3],outputs[:,3:4]

        dz2=(z_p[...,1:] - z_p[...,0:-1])**2
        # laplacedz2 = 4*dz2[:,:,1:-1,1:-1,:] - \
        #             dz2[:,:,:-2,1:-1,:] - \
        #             dz2[:,:,2:,1:-1,:] - \
        #             dz2[:,:,1:-1,:-2,:] - \
        #             dz2[:,:,1:-1,2:,:]

        laplacedz2 = 6*dz2[:,:,1:-1,1:-1,1:-1] - \
                    dz2[:,:,:-2,1:-1,1:-1] - \
                    dz2[:,:,2:,1:-1,1:-1] - \
                    dz2[:,:,1:-1,:-2,1:-1] - \
                    dz2[:,:,1:-1,2:,1:-1] - \
                    dz2[:,:,1:-1,1:-1,2:] - \
                    dz2[:,:,1:-1,1:-1,:-2]
        loss_smooth=torch.mean(laplacedz2**2)

        loss_b = torch.mean((bx_p-bx_t)**2+(by_p-by_t)**2+(bz_p-bz_t)**2)
        
        # loss_smooth=torch.mean(torch.abs(laplacedz2)**2/self.dh**2)

        # define the divergence of the predicted field in the integrated form.
        bx_000,by_000,bz_000,z_000=bx_p[:,:,0:-1,0:-1,0:-1],by_p[:,:,0:-1,0:-1,0:-1],bz_p[:,:,0:-1,0:-1,0:-1],z_p[:,:,0:-1,0:-1,0:-1]
        bx_100,by_100,bz_100,z_100=bx_p[:,:,1:,0:-1,0:-1],by_p[:,:,1:,0:-1,0:-1],bz_p[:,:,1:,0:-1,0:-1],z_p[:,:,1:,0:-1,0:-1]
        bx_010,by_010,bz_010,z_010=bx_p[:,:,0:-1,1:,0:-1],by_p[:,:,0:-1,1:,0:-1],bz_p[:,:,0:-1,1:,0:-1],z_p[:,:,0:-1,1:,0:-1]
        bx_110,by_110,bz_110,z_110=bx_p[:,:,1:,1:,0:-1],by_p[:,:,1:,1:,0:-1],bz_p[:,:,1:,1:,0:-1],z_p[:,:,1:,1:,0:-1]

        bx_001,by_001,bz_001,z_001=bx_p[:,:,0:-1,0:-1,1:],by_p[:,:,0:-1,0:-1,1:],bz_p[:,:,0:-1,0:-1,1:],z_p[:,:,0:-1,0:-1,1:]
        bx_101,by_101,bz_101,z_101=bx_p[:,:,1:,0:-1,1:],by_p[:,:,1:,0:-1,1:],bz_p[:,:,1:,0:-1,1:],z_p[:,:,1:,0:-1,1:]
        bx_011,by_011,bz_011,z_011=bx_p[:,:,0:-1,1:,1:],by_p[:,:,0:-1,1:,1:],bz_p[:,:,0:-1,1:,1:],z_p[:,:,0:-1,1:,1:]
        bx_111,by_111,bz_111,z_111=bx_p[:,:,1:,1:,1:],by_p[:,:,1:,1:,1:],bz_p[:,:,1:,1:,1:],z_p[:,:,1:,1:,1:]

        loss_div=torch.mean((\
                0.25*(bx_100+bx_110+bx_101+bx_111)*self.dy*0.5*(torch.abs(z_101-z_100)+torch.abs(z_111-z_110))-\
                0.25*(bx_000+bx_010+bx_001+bx_011)*self.dy*0.5*(torch.abs(z_001-z_000)+torch.abs(z_011-z_010))+\
                0.25*(by_010+by_110+by_011+by_111)*self.dx*0.5*(torch.abs(z_011-z_010)+torch.abs(z_111-z_110))-\
                0.25*(by_000+by_100+by_001+by_101)*self.dx*0.5*(torch.abs(z_001-z_000)+torch.abs(z_101-z_100))+\
                0.25*(bz_001 + bz_011 + bz_101 + bz_111)*self.dx*self.dy-\
                0.25*(bz_000 + bz_010 + bz_100 + bz_110)*self.dx*self.dy+\
                (bx_001+bx_101+bx_111)*self.dy*(z_001-z_101)/6+\
                (bx_011+bx_111+bx_101)*self.dy*(z_011-z_111)/6+\
                (by_101+by_111+by_011)*self.dx*(z_101-z_111)/6+\
                (by_001+by_011+by_111)*self.dx*(z_001-z_011)/6-\
                (bx_000+bx_100+bx_110)*self.dy*(z_000-z_100)/6-\
                (bx_010+bx_110+bx_100)*self.dy*(z_010-z_110)/6-\
                (by_100+by_110+by_010)*self.dx*(z_100-z_110)/6-\
                (by_000+by_010+by_110)*self.dx*(z_000-z_010)/6
                )**2/(\
                ((bx_000+bx_001+bx_010+bx_011 + bx_100+bx_101+bx_110+bx_111)*0.125)**2+\
                ((by_000+by_001+by_010+by_011 + by_100+by_101+by_110+by_111)*0.125)**2+\
                ((bz_000+bz_001+bz_010+bz_011 + bz_100+bz_101+bz_110+bz_111)*0.125)**2+1e-10\
                ))

        loss_div*=self.w_div
        loss_b*=self.w_b
        loss_smooth*=self.w_smooth     
        return loss_div,loss_smooth,loss_b

        # return loss_div,loss_mon,loss_smooth
class CustomLoss_div(nn.Module):
    def __init__(self,dx=0.1,dy=0.1,nlayer=16,w_b=1e3,w_parallel=1e3,w_div=1e2):
        super(CustomLoss_div, self).__init__()
        self.dx=dx
        self.dy=dy
        self.dh=max(dx,dy)
        self.w_b=w_b
        self.w_parallel=w_parallel
        self.w_div=w_div
        self.logthickness=math.log(self.dh)
    def forward(self, outputs, targets):
        # get the predicted magnetic field from the NNs.
        bx_p,by_p,bz_p=outputs[:,0:1],outputs[:,1:2],outputs[:,2:3]
        # enforce the predicted B field close to the observations.
        bx_t,by_t,bz_t=targets[:,0:1],targets[:,1:2],targets[:,2:3]

        loss_b=torch.mean((bx_p**2+by_p**2-bx_t**2-by_t**2)**2/(bx_t**2+by_t**2+1e-10))+\
                torch.mean((bz_p-bz_t)**4/(bz_t**2+1e-10))

        loss_parallel=torch.mean((bx_p*by_t - by_p*bx_t)**2/(bx_t**2+by_t**2+bz_t**2 + 1e-10))

        bx_000,by_000,bz_000,z_000=bx_p[:,:,0:-1,0:-1,0:-1],by_p[:,:,0:-1,0:-1,0:-1],bz_p[:,:,0:-1,0:-1,0:-1],targets[:,3:4,0:-1,0:-1,0:-1]
        bx_100,by_100,bz_100,z_100=bx_p[:,:,1:,0:-1,0:-1],by_p[:,:,1:,0:-1,0:-1],bz_p[:,:,1:,0:-1,0:-1],targets[:,3:4,1:,0:-1,0:-1]
        bx_010,by_010,bz_010,z_010=bx_p[:,:,0:-1,1:,0:-1],by_p[:,:,0:-1,1:,0:-1],bz_p[:,:,0:-1,1:,0:-1],targets[:,3:4,0:-1,1:,0:-1]
        bx_110,by_110,bz_110,z_110=bx_p[:,:,1:,1:,0:-1],by_p[:,:,1:,1:,0:-1],bz_p[:,:,1:,1:,0:-1],targets[:,3:4,1:,1:,0:-1]

        bx_001,by_001,bz_001,z_001=bx_p[:,:,0:-1,0:-1,1:],by_p[:,:,0:-1,0:-1,1:],bz_p[:,:,0:-1,0:-1,1:],targets[:,3:4,0:-1,0:-1,1:]
        bx_101,by_101,bz_101,z_101=bx_p[:,:,1:,0:-1,1:],by_p[:,:,1:,0:-1,1:],bz_p[:,:,1:,0:-1,1:],targets[:,3:4,1:,0:-1,1:]
        bx_011,by_011,bz_011,z_011=bx_p[:,:,0:-1,1:,1:],by_p[:,:,0:-1,1:,1:],bz_p[:,:,0:-1,1:,1:],targets[:,3:4,0:-1,1:,1:]
        bx_111,by_111,bz_111,z_111=bx_p[:,:,1:,1:,1:],by_p[:,:,1:,1:,1:],bz_p[:,:,1:,1:,1:],targets[:,3:4,1:,1:,1:]

        loss_div=torch.mean((\
                0.25*(bx_100+bx_110+bx_101+bx_111)*self.dy*0.5*(z_101-z_100+z_111-z_110)-\
                0.25*(bx_000+bx_010+bx_001+bx_011)*self.dy*0.5*(z_001-z_000+z_011-z_010)+\
                0.25*(by_010+by_110+by_011+by_111)*self.dx*0.5*(z_011-z_010+z_111-z_110)-\
                0.25*(by_000+by_100+by_001+by_101)*self.dx*0.5*(z_001-z_000+z_101-z_100)+\
                0.25*(bz_001 + bz_011 + bz_101 + bz_111)*self.dx*self.dy-\
                0.25*(bz_000 + bz_010 + bz_100 + bz_110)*self.dx*self.dy+\
                (bx_001+bx_101+bx_111)*self.dy*(z_001-z_101)/6+\
                (bx_011+bx_111+bx_101)*self.dy*(z_011-z_111)/6+\
                (by_101+by_111+by_011)*self.dx*(z_101-z_111)/6+\
                (by_001+by_011+by_111)*self.dx*(z_001-z_011)/6-\
                (bx_000+bx_100+bx_110)*self.dy*(z_000-z_100)/6-\
                (bx_010+bx_110+bx_100)*self.dy*(z_010-z_110)/6-\
                (by_100+by_110+by_010)*self.dx*(z_100-z_110)/6-\
                (by_000+by_010+by_110)*self.dx*(z_000-z_010)/6 \
                )**2/(\
                ((bx_000+bx_001+bx_010+bx_011 + bx_100+bx_101+bx_110+bx_111)*0.125)**2+\
                ((by_000+by_001+by_010+by_011 + by_100+by_101+by_110+by_111)*0.125)**2+\
                ((bz_000+bz_001+bz_010+bz_011 + bz_100+bz_101+bz_110+bz_111)*0.125)**2+1e-10\
                ))/self.dx**2/self.dy**2

        loss_b*=self.w_b
        loss_parallel*=self.w_parallel
        loss_div*=self.w_div
        return loss_b+loss_parallel,loss_div
class CustomLoss_sign(nn.Module):
    def __init__(self,dx=0.1,dy=0.1,nlayer=16,w_b=1e3,w_parallel=1e3,w_div=1e2):
        super(CustomLoss_sign, self).__init__()
        self.dx=dx
        self.dy=dy
        self.dh=max(dx,dy)
        self.w_b=w_b
        self.w_parallel=w_parallel
        self.w_div=w_div
        self.logthickness=math.log(self.dh)
    def forward(self, outputs, targets):
        # get the predicted magnetic field from the NNs.
        bx_p,by_p,bz_p=outputs[:,0:1],outputs[:,1:2],outputs[:,2:3]
        # enforce the predicted B field close to the observations.
        bx_t,by_t,bz_t=targets[:,0:1],targets[:,1:2],targets[:,2:3]

        # bt_p=torch.sqrt(bx_p**2+by_p**2)
        # bt_t=torch.sqrt(bx_t**2+by_t**2)
        # loss_b=torch.mean((bt_p-bt_t)**2/(bt_t+1e-8))+\
        #         torch.mean((bz_p-bz_t)**2/(torch.abs(bz_t)+1e-8))
        # loss_parallel=torch.mean(torch.abs(bx_p*by_t - by_p*bx_t)/(bt_t+1e-8))

        loss_b=torch.mean((bx_p**2+by_p**2-bx_t**2-by_t**2)**2/(bx_t**2+by_t**2+1e-10))+\
                torch.mean((bz_p-bz_t)**4/(bz_t**2+1e-8))

        loss_parallel=torch.mean((bx_p*by_t - by_p*bx_t)**2/(bx_t**2+by_t**2+bz_t**2 + 1e-8))


        bx_000,by_000,bz_000=bx_p[:,:,0:-1,0:-1,0:-1],by_p[:,:,0:-1,0:-1,0:-1],bz_p[:,:,0:-1,0:-1,0:-1]
        bx_100,by_100,bz_100=bx_p[:,:,1:,0:-1,0:-1],by_p[:,:,1:,0:-1,0:-1],bz_p[:,:,1:,0:-1,0:-1]
        bx_010,by_010,bz_010=bx_p[:,:,0:-1,1:,0:-1],by_p[:,:,0:-1,1:,0:-1],bz_p[:,:,0:-1,1:,0:-1]
        bx_110,by_110,bz_110=bx_p[:,:,1:,1:,0:-1],by_p[:,:,1:,1:,0:-1],bz_p[:,:,1:,1:,0:-1]

        bx_001,by_001,bz_001=bx_p[:,:,0:-1,0:-1,1:],by_p[:,:,0:-1,0:-1,1:],bz_p[:,:,0:-1,0:-1,1:]
        bx_101,by_101,bz_101=bx_p[:,:,1:,0:-1,1:],by_p[:,:,1:,0:-1,1:],bz_p[:,:,1:,0:-1,1:]
        bx_011,by_011,bz_011=bx_p[:,:,0:-1,1:,1:],by_p[:,:,0:-1,1:,1:],bz_p[:,:,0:-1,1:,1:]
        bx_111,by_111,bz_111=bx_p[:,:,1:,1:,1:],by_p[:,:,1:,1:,1:],bz_p[:,:,1:,1:,1:]

        loss_div=torch.mean((\
                (bx_100+bx_110+bx_101+bx_111)-\
                (bx_000+bx_010+bx_001+bx_011)+\
                (by_010+by_110+by_011+by_111)-\
                (by_000+by_100+by_001+by_101)+\
                (bz_001 + bz_011 + bz_101 + bz_111)-\
                (bz_000 + bz_010 + bz_100 + bz_110)
                )**2/(\
                ((bx_000+bx_001+bx_010+bx_011 + bx_100+bx_101+bx_110+bx_111)*0.125)**2+\
                ((by_000+by_001+by_010+by_011 + by_100+by_101+by_110+by_111)*0.125)**2+\
                ((bz_000+bz_001+bz_010+bz_011 + bz_100+bz_101+bz_110+bz_111)*0.125)**2+1e-10\
                ))

        loss_b*=self.w_b
        loss_parallel*=self.w_parallel
        loss_div*=self.w_div
        return loss_b+loss_parallel,loss_div


class CustomLoss_total(nn.Module):
    def __init__(self,dx=0.1,dy=0.1,w_b=1e5,w_parallel=1e5,w_div=1e4,w_mon=1e3,w_smooth=1e0):
        super(CustomLoss_total, self).__init__()
        self.dx=dx
        self.dy=dy
        self.dh=max(dx,dy)
        self.w_b=w_b
        self.w_parallel=w_parallel
        self.w_div=w_div
        self.w_mon=w_mon
        self.w_smooth=w_smooth
        # self.logthickness=math.log(self.dh)
    def forward(self, outputs, targets):
        # get the predicted magnetic field from the NNs.
        bx_p,by_p,bz_p,z_p=outputs[:,0:1],outputs[:,1:2],outputs[:,2:3],outputs[:,3:4]
        # enforce the predicted B field close to the observations.
        bx_t,by_t,bz_t=targets[:,0:1],targets[:,1:2],targets[:,2:3]

        loss_b=torch.mean((bx_p**2+by_p**2-bx_t**2-by_t**2)**2/(bx_t**2+by_t**2+1e-10))+\
                torch.mean((bz_p-bz_t)**4/(bz_t**2+1e-10))

        loss_parallel=torch.mean((bx_p*by_t - by_p*bx_t)**2/(bx_t**2+by_t**2+bz_t**2 + 1e-10))

        # ensure the predicted height is monotonous w.r.t. tau layers
        dz=z_p[...,1:] - z_p[...,0:-1]
        # max_values = torch.max(torch.clamp(-dz, min=0)**2)
        max_values = torch.max(torch.clamp(-dz, min=0))
        loss_mon=max_values.mean()

        # dz2=dz**2
        dz2=torch.abs(dz)
        laplacedz2 = 4*dz2[:,:,1:-1,1:-1,:] - \
                    dz2[:,:,:-2,1:-1,:] - \
                    dz2[:,:,2:,1:-1,:] - \
                    dz2[:,:,1:-1,:-2,:] - \
                    dz2[:,:,1:-1,2:,:]
        loss_smooth=torch.mean(torch.abs(laplacedz2))

        bx_000,by_000,bz_000,z_000=bx_p[:,:,0:-1,0:-1,0:-1],by_p[:,:,0:-1,0:-1,0:-1],bz_p[:,:,0:-1,0:-1,0:-1],z_p[:,:,0:-1,0:-1,0:-1]
        bx_100,by_100,bz_100,z_100=bx_p[:,:,1:,0:-1,0:-1],by_p[:,:,1:,0:-1,0:-1],bz_p[:,:,1:,0:-1,0:-1],z_p[:,:,1:,0:-1,0:-1]
        bx_010,by_010,bz_010,z_010=bx_p[:,:,0:-1,1:,0:-1],by_p[:,:,0:-1,1:,0:-1],bz_p[:,:,0:-1,1:,0:-1],z_p[:,:,0:-1,1:,0:-1]
        bx_110,by_110,bz_110,z_110=bx_p[:,:,1:,1:,0:-1],by_p[:,:,1:,1:,0:-1],bz_p[:,:,1:,1:,0:-1],z_p[:,:,1:,1:,0:-1]

        bx_001,by_001,bz_001,z_001=bx_p[:,:,0:-1,0:-1,1:],by_p[:,:,0:-1,0:-1,1:],bz_p[:,:,0:-1,0:-1,1:],z_p[:,:,0:-1,0:-1,1:]
        bx_101,by_101,bz_101,z_101=bx_p[:,:,1:,0:-1,1:],by_p[:,:,1:,0:-1,1:],bz_p[:,:,1:,0:-1,1:],z_p[:,:,1:,0:-1,1:]
        bx_011,by_011,bz_011,z_011=bx_p[:,:,0:-1,1:,1:],by_p[:,:,0:-1,1:,1:],bz_p[:,:,0:-1,1:,1:],z_p[:,:,0:-1,1:,1:]
        bx_111,by_111,bz_111,z_111=bx_p[:,:,1:,1:,1:],by_p[:,:,1:,1:,1:],bz_p[:,:,1:,1:,1:],z_p[:,:,1:,1:,1:]

        loss_div=torch.mean((\
                0.25*(bx_100+bx_110+bx_101+bx_111)*self.dy*0.5*(torch.abs(z_101-z_100)+torch.abs(z_111-z_110))-\
                0.25*(bx_000+bx_010+bx_001+bx_011)*self.dy*0.5*(torch.abs(z_001-z_000)+torch.abs(z_011-z_010))+\
                0.25*(by_010+by_110+by_011+by_111)*self.dx*0.5*(torch.abs(z_011-z_010)+torch.abs(z_111-z_110))-\
                0.25*(by_000+by_100+by_001+by_101)*self.dx*0.5*(torch.abs(z_001-z_000)+torch.abs(z_101-z_100))+\
                0.25*(bz_001 + bz_011 + bz_101 + bz_111)*self.dx*self.dy-\
                0.25*(bz_000 + bz_010 + bz_100 + bz_110)*self.dx*self.dy+\
                (bx_001+bx_101+bx_111)*self.dy*(z_001-z_101)/6+\
                (bx_011+bx_111+bx_101)*self.dy*(z_011-z_111)/6+\
                (by_101+by_111+by_011)*self.dx*(z_101-z_111)/6+\
                (by_001+by_011+by_111)*self.dx*(z_001-z_011)/6-\
                (bx_000+bx_100+bx_110)*self.dy*(z_000-z_100)/6-\
                (bx_010+bx_110+bx_100)*self.dy*(z_010-z_110)/6-\
                (by_100+by_110+by_010)*self.dx*(z_100-z_110)/6-\
                (by_000+by_010+by_110)*self.dx*(z_000-z_010)/6 \
                )**2/(\
                ((bx_000+bx_001+bx_010+bx_011 + bx_100+bx_101+bx_110+bx_111)*0.125)**2+\
                ((by_000+by_001+by_010+by_011 + by_100+by_101+by_110+by_111)*0.125)**2+\
                ((bz_000+bz_001+bz_010+bz_011 + bz_100+bz_101+bz_110+bz_111)*0.125)**2+1e-10\
                ))/self.dx**2/self.dy**2

        loss_b*=self.w_b
        loss_parallel*=self.w_parallel
        loss_div*=self.w_div
        loss_smooth*=self.w_smooth
        loss_mon*=self.w_mon
        return loss_b,loss_parallel,loss_div,loss_smooth,loss_mon


# define the training function
# def train_z(model,criterion,train_loader,loss_record=[],lr=0.001,num_epochs=1000,device="cpu"):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)  # Adjust the scheduler settings as needed
#     tic_total = time.time()
#     for epoch in range(num_epochs):
#         model.train()  # Set the model to training mode
#         loss_tmp=0
#         for batch in train_loader:  # We don't have true labels in the provided code, so using a placeholder
#             if isinstance(batch, tuple) and len(batch) == 2:
#                 batch_data, _ = batch
#             else:
#                 batch_data = batch
#             # batch_data=batch_data.to(device)
#             optimizer.zero_grad()        
#             # Forward pass
#             outputs = model(batch_data)
#             loss_div,loss_mon,loss_smooth=criterion(outputs, batch_data)
#             loss=loss_div+loss_mon+loss_smooth
#             # Backward pass and optimization
#             loss.backward()
#             # Gradient clipping
#             utils.clip_grad_norm_(model.parameters(), 10)
#             optimizer.step()
#             loss_tmp+=loss.item()
#         scheduler.step()
#         if (epoch+1)%100==0:
#             print(f'    Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, loss_div: {loss_div.item():.4f}, loss_mon: {loss_mon.item():.4f}, loss_smooth: {loss_smooth.item():.4f}')
#         loss_record.append(loss_tmp)
#     toc_total = time.time()
#     print(f"    Total training time: {(toc_total - tic_total)/60:.4f} min")
#     print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
#     return model,loss_record

# def train_sign(model,criterion,train_loader,loss_record=[],lr=0.001,num_epochs=1000,device="cpu"):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
#     scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)  # Adjust the scheduler settings as needed
#     tic_total = time.time()
#     for epoch in range(num_epochs):
#         model.train()  # Set the model to training mode
#         loss_tmp=0
#         for batch in train_loader:  # We don't have true labels in the provided code, so using a placeholder
#             if isinstance(batch, tuple) and len(batch) == 2:
#                 batch_data, _ = batch
#             else:
#                 batch_data = batch
#             # batch_data=batch_data.to(device)
#             optimizer.zero_grad()        
#             # Forward pass
#             outputs = model(batch_data)
#             loss_b,loss_div=criterion(outputs, batch_data)
#             loss=loss_b+loss_div
#             # Backward pass and optimization
#             loss.backward()
#             # Gradient clipping
#             utils.clip_grad_norm_(model.parameters(), 10)
#             optimizer.step()
#             loss_tmp+=loss.item()
#         scheduler.step()
#         if (epoch+1)%100==0:
#             print(f'    Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, loss_b: {loss_b.item():.4f}, loss_div: {loss_div.item():.4f}')
#         loss_record.append(loss_tmp)
#     toc_total = time.time()
#     print(f"    Total training time: {(toc_total - tic_total)/60:.4f} min")
#     print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
#     return model,loss_record


# def train_sign(model, criterion, train_loader, loss_record=[], lr=0.001, num_epochs=1000, device="cpu"):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
#     scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
#     scaler = GradScaler()
#     tic_total = time.time()

#     for epoch in range(num_epochs):
#         model.train()
#         loss_tmp = 0
#         for batch in train_loader:
#             if isinstance(batch, tuple) and len(batch) == 2:
#                 batch_data, _ = batch
#             else:
#                 batch_data = batch
#             optimizer.zero_grad()
#             with autocast():
#                 outputs = model(batch_data)
#                 loss_b, loss_div = criterion(outputs, batch_data)
#                 loss = loss_b + loss_div
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             utils.clip_grad_norm_(model.parameters(), 10)
#             scaler.step(optimizer)
#             scaler.update()
#             loss_tmp += loss.item()
#             scheduler.step()

#         if (epoch + 1) % 100 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, loss_b: {loss_b.item():.4f}, loss_div: {loss_div.item():.4f}')
#         loss_record.append(loss_tmp)
#     toc_total = time.time()
#     print(f"Total training time: {(toc_total - tic_total)/60:.4f} min")
#     print("-" * 70)
#     return model, loss_record

def train_sign(model, criterion, train_loader, loss_record=[], lr=0.001, num_epochs=1000, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    # scaler = GradScaler()
    tic_total = time.time()

    for epoch in range(num_epochs):
        model.train()
        loss_tmp = 0
        for batch in train_loader:
            if isinstance(batch, tuple) and len(batch) == 2:
                batch_data, _ = batch
            else:
                batch_data = batch
            optimizer.zero_grad()

            # with autocast():
            #     outputs = model(batch_data)
            #     loss_b, loss_div = criterion(outputs, batch_data)
            #     loss = loss_b+loss_div 
            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # utils.clip_grad_norm_(model.parameters(), 10)
            # scaler.step(optimizer)
            # scaler.update()

            outputs = model(batch_data)
            loss_b, loss_div = criterion(outputs, batch_data)
            loss = loss_b+loss_div
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            loss_tmp += loss.item()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, loss_div: {loss_div.item():.4f}, loss_b: {loss_b.item():.4f}')
        loss_record.append(loss_tmp)

    toc_total = time.time()
    print(f"Total training time: {(toc_total - tic_total)/60:.4f} min")
    print("-" * 70)
    return model, loss_record

def train_z(model, criterion, train_loader, loss_record=[], lr=0.001, num_epochs=1000, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    scaler = GradScaler()
    tic_total = time.time()

    for epoch in range(num_epochs):
        model.train()
        loss_tmp = 0
        for batch in train_loader:
            if isinstance(batch, tuple) and len(batch) == 2:
                batch_data, _ = batch
            else:
                batch_data = batch
            optimizer.zero_grad()

            # with autocast():
            #     outputs = model(batch_data)
            #     loss_div, loss_mon, loss_smooth = criterion(outputs, batch_data)
            #     loss = loss_div + loss_mon + loss_smooth
            # scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            # utils.clip_grad_norm_(model.parameters(), 10)
            # scaler.step(optimizer)
            # scaler.update()

            outputs = model(batch_data)
            loss_div, loss_smooth = criterion(outputs, batch_data)
            loss = loss_div + loss_smooth

            # loss_div, loss_mon, loss_smooth = criterion(outputs, batch_data)
            # loss = loss_div + loss_mon + loss_smooth
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            loss_tmp += loss.item()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, loss_div: {loss_div.item():.4f}, loss_smooth: {loss_smooth.item():.4f}')
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, loss_div: {loss_div.item():.4f}, loss_mon: {loss_mon.item():.4f}, loss_smooth: {loss_smooth.item():.4f}')
        loss_record.append(loss_tmp)

    toc_total = time.time()
    print(f"Total training time: {(toc_total - tic_total)/60:.4f} min")
    print("-" * 70)
    return model, loss_record

def train_z_total(model, criterion, train_loader, loss_record=[], lr=0.001, num_epochs=1000, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    scaler = GradScaler()
    tic_total = time.time()

    for epoch in range(num_epochs):
        model.train()
        loss_tmp = 0
        for batch in train_loader:
            if isinstance(batch, tuple) and len(batch) == 2:
                batch_data, _ = batch
            else:
                batch_data = batch
            optimizer.zero_grad()

            outputs = model(batch_data)
            loss_div, loss_smooth, loss_b = criterion(outputs, batch_data)
            loss = loss_div + loss_smooth + loss_b

            # loss_div, loss_mon, loss_smooth = criterion(outputs, batch_data)
            # loss = loss_div + loss_mon + loss_smooth
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            loss_tmp += loss.item()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, loss_div: {loss_div.item():.4f}, loss_smooth: {loss_smooth.item():.4f}, loss_b: {loss_b.item():.4f}')
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, loss_div: {loss_div.item():.4f}, loss_mon: {loss_mon.item():.4f}, loss_smooth: {loss_smooth.item():.4f}')
        loss_record.append(loss_tmp)

    toc_total = time.time()
    print(f"Total training time: {(toc_total - tic_total)/60:.4f} min")
    print("-" * 70)
    return model, loss_record

# def train_div(model,criterion,train_loader,loss_record=[],lr=0.001,num_epochs=1000,device="cpu"):
#     # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)  # Adjust the scheduler settings as needed
#     # torch.autograd.set_detect_anomaly(True)
#     tic_total = time.time()
#     for epoch in range(num_epochs):
#     # for epoch in range(100):
#         model.train()  # Set the model to training mode
#         loss_tmp=0
#         for batch in train_loader:  # We don't have true labels in the provided code, so using a placeholder
#             if isinstance(batch, tuple) and len(batch) == 2:
#                 batch_data, _ = batch
#             else:
#                 batch_data = batch
#             # batch_data=batch_data.to(device)
#             optimizer.zero_grad()        
#             # Forward pass
#             outputs = model(batch_data)
#             loss_b,loss_div=criterion(outputs, batch_data)
#             loss=loss_b+loss_div
#             # Backward pass and optimization
#             loss.backward()
#             # Gradient clipping
#             utils.clip_grad_norm_(model.parameters(), 10)
#             optimizer.step()
#             loss_tmp+=loss.item()
#         scheduler.step()
#         if (epoch+1)%100==0:
#             print(f'    Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, loss_b: {loss_b.item():.4f}, loss_div: {loss_div.item():.4f}')
#         loss_record.append(loss_tmp)
#     toc_total = time.time()
#     print(f"    Total training time: {(toc_total - tic_total)/60:.4f} min")
#     print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
#     return model,loss_record

# # define the training function
# def train_total(model,criterion,train_loader,loss_record=[],lr=0.01,num_epochs=1000,device="cpu"):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = StepLR(optimizer, step_size=500, gamma=0.5)  # Adjust the scheduler settings as needed
#     # torch.autograd.set_detect_anomaly(True)
#     tic_total = time.time()
#     model.train()  # Set the model to training mode
#     for epoch in range(num_epochs):
#         loss_tmp=0
#         for batch in train_loader:  # We don't have true labels in the provided code, so using a placeholder
#             if isinstance(batch, tuple) and len(batch) == 2:
#                 batch_data, _ = batch
#             else:
#                 batch_data = batch
#             batch_data=batch_data.to(device)
#             optimizer.zero_grad()        
#             # Forward pass
#             outputs = model(batch_data)
#             loss_b,loss_parallel,loss_div,loss_smooth,loss_mon=criterion(outputs, batch_data)
#             loss=loss_b+loss_parallel+loss_div+loss_smooth+loss_mon
#             # Backward pass and optimization
#             loss.backward()
#             # Gradient clipping
#             utils.clip_grad_norm_(model.parameters(), 10)
#             optimizer.step()
#             loss_tmp+=loss.item()
#         scheduler.step()
#         if (epoch+1)%100==0:
#             print(f'    Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, loss_b: {loss_b.item():.4f}, loss_parallel: {loss_parallel.item():.4f}, loss_div: {loss_div.item():.4f}, loss_smooth: {loss_smooth.item():.4f}, loss_mon: {loss_mon.item():.4f}')
#         loss_record.append(loss_tmp)
#     toc_total = time.time()
#     print(f"    Total training time: {(toc_total - tic_total)/60:.4f} min")
#     print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
#     return model,loss_record

# def train_lbfgs(model,criterion,train_loader,loss_record=[],lr=1,num_epochs=1000,device="cpu"):
#     optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
#     model.train()  # Set the model to training mode
#     for epoch in range(num_epochs):
#         for batch_data in train_loader:
#             batch_data = batch_data.to(device)
#             def closure():
#                 optimizer.zero_grad()
#                 outputs = model(batch_data)
#                 loss_b,loss_parallel,loss_div,loss_smooth,loss_mon=criterion(outputs, batch_data)
#                 loss=loss_b+loss_parallel+loss_div+loss_smooth+loss_mon
#                 loss.backward()
#                 utils.clip_grad_norm_(model.parameters(), 10)
#                 return loss
#             loss_tmp=optimizer.step(closure)
#             loss_record.append(loss_tmp.item())
#         if (epoch+1)%100==0:
#             print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss_tmp.item():.4f},')
#     return model,loss_record


def test_accuracy(bxyz_pred,z_pred, device="cpu"):
    data_filename='muram_tau_layers_200g.h5'
    size=64
    hsize=32
    train_dataset,_,_,t3d=process_data(filename=data_filename,size=size,hsize=hsize,batch_size=1,device=device)
    err_b=nn.MSELoss(train_dataset,bxyz_pred)
    err_z=nn.MSELoss(z_pred-z_pred.min(),t3d-t3d.min())
    return err_b,err_z

def save_predictions(check_data,predictionIn,loss_record,filename='pred_CNN_3D.h5'):
    with h5py.File(filename,'w') as f:
        f.create_dataset('prediction',data=predictionIn)
        f.create_dataset('loss',data=loss_record)
        f.create_dataset('check_data',data=check_data.cpu().detach().numpy())
    return

def save_predictions_model(model,check_data,loss_record,filename='pred_CNN_3D.h5',size=128,nchannel=31,model_filename='trained_model_3D.pth'):
    model.to("cpu")
    prediction=model(check_data).cpu().detach().numpy()
    torch.save(model, model_filename)
    with h5py.File(filename,'w') as f:
        f.create_dataset('prediction',data=prediction)
        f.create_dataset('loss',data=loss_record)
        f.create_dataset('check_data',data=check_data.cpu().detach().numpy())
    return

def update_bobs(model,obs_data,dx,dy,device="cpu"):
    model.to(device)
    obs_data=obs_data.to(device)
    bpred=model(obs_data).cpu().detach()
    mask = (bpred[:,0:1]*obs_data[:,0:1]+bpred[:,1:2]*obs_data[:,1:2]) < 0
    out=torch.zeros(obs_data.shape[0],obs_data.shape[1],obs_data.shape[2],obs_data.shape[3],obs_data.shape[4])
    out[:,0:4]=obs_data.clone().detach()
    out[:,0:1][mask]*=-1
    out[:,1:2][mask]*=-1
    return out

def update_bobs_sub(obs_data,bpred):
    mask = (bpred[:,0:1]*obs_data[:,0:1]+bpred[:,1:2]*obs_data[:,1:2]) < 0
    out=torch.zeros(obs_data.shape[0],obs_data.shape[1],obs_data.shape[2],obs_data.shape[3],obs_data.shape[4])
    out[:,0:4]=obs_data.clone().detach()
    out[:,0:1][mask]*=-1
    out[:,1:2][mask]*=-1
    return out

def update_Z(model,obs_data,device="cpu"):
    model.to(device)
    obs_data=obs_data.to(device)
    Z_pred=model(obs_data).cpu().detach()
    out=obs_data.clone().detach()
    out[:,3:4]=Z_pred
    return out

def subtrain(datasetIn,modelIn,trainIn,criterionIn,lr=0.001,num_epochs=1000,device="cpu"):
    loss_record=[]
    prediction=[]
    for ix in range(datasetIn.shape[0]):
        train_loader = CustomDataLoader(datasetIn[ix:ix+1].to(device), batch_size=1, shuffle=True)
        modelIn,loss_tmp=trainIn(modelIn,criterionIn,train_loader,lr=lr,num_epochs=num_epochs,device=device)
        loss_record=np.concatenate((loss_record,loss_tmp),axis=0)
        predict_tmp=modelIn(datasetIn[ix:ix+1].to(device)).cpu().detach().numpy()
        prediction=np.concatenate((prediction,predict_tmp),axis=0)
    return modelIn,loss_record

def main_step_sub_0():
    # Set seeds
    torch.manual_seed(123456)
    np.random.seed(123456)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device',device)
    data_filename='muram_tau_layers_200g.h5'
    # data_filename='muram_tau_layers.h5'

    size=64
    hsize=16
    blockNum=2
    # batch_size=int(blockNum*blockNum)
    batch_size=blockNum
    input_channels = 4
    output_channels = 3
    num_epochs=20000
    nchan=64
    deep=2
    pad=16
    # record time
    tic_start = time.time()
    w_div=1e2
    w_b=1e3
    w_parallel=1e3
    # first load and process data
    train_dataset,dx,dy,_=process_data(filename=data_filename,size=size,hsize=hsize,batch_size=batch_size,device=device,pad=pad,blockNum=blockNum)
    model_b=create_model(inchannels=input_channels,outchannels=output_channels,dx=dx,dy=dy,nchan=nchan,deep=deep,device=device)
    print(model_b)
    criterion = CustomLoss_sign(dx=dx,dy=dy,nlayer=hsize,w_div=w_div,w_b= w_b,w_parallel=w_parallel).to(device)
    print("Starting training for 3D B ...")
    prediction,loss_record_div=subtrain(train_dataset,model_b,train_sign,criterion,lr=0.001,num_epochs=num_epochs,device=device)
    print("Saving the prediction for 3D B ...")
    save_predictions(train_dataset,prediction,loss_record_div,filename='pred_CNN_3D_B.h5')
    tic_end = time.time()
    print(f"Total training time: {(tic_end - tic_start)/60:.4f} min")

def main_step_sub_1():
    # Set seeds
    torch.manual_seed(123456)
    np.random.seed(123456)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device',device)
    data_filename='muram_tau_layers_200g.h5'
    # data_filename='muram_tau_layers.h5'
    size=64
    hsize=16
    blockNum=2
    # batch_size=int(blockNum*blockNum)
    batch_size=blockNum
    input_channels = 4
    output_channels = 3
    nchan=64
    deep=2
    pad=16
    # record time
    tic_start = time.time()
    # first load and process data
    train_dataset,dx,dy,_=process_data(filename=data_filename,size=size,hsize=hsize,batch_size=batch_size,device=device,pad=pad,blockNum=blockNum)
# ----------------------------------------------------------------------------------------------
    # update the angle for the obs data and stitch Z variable
    with h5py.File('pred_CNN_3D_B.h5','r') as f:
        bpred = f['prediction'][:]
    train_dataset=update_bobs_sub(train_dataset,bpred)
    input_channels = 4
    output_channels = 1
    num_epochs=10000
    w_smooth=1e5
    print("Create 3D Z model and loss function ...")
    model_z=create_model_z(inchannels=input_channels,outchannels=output_channels,dx=dx,dy=dy,nchan=nchan,deep=deep,device=device)
    criterion = CustomLoss_Z(w_smooth=w_smooth,dx=dx,dy=dy).to(device)
    print("Starting training for 3D Z ...")
    prediction,loss_record_div=subtrain(train_dataset,model_z,train_z,criterion,lr=0.001,num_epochs=num_epochs,device=device)
    print("Saving the prediction for 3D B ...")
    save_predictions(train_dataset,prediction,loss_record_div,filename='pred_CNN_3D_B.h5')
    tic_end = time.time()
    print(f"Total training time: {(tic_end - tic_start)/60:.4f} min")


def main_step_0():
    # Set seeds
    torch.manual_seed(123456)
    np.random.seed(123456)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device',device)
    data_filename='muram_tau_layers_200g.h5'
    # data_filename='muram_tau_layers.h5'

    size=64
    hsize=16
    blockNum=6
    # batch_size=int(blockNum*blockNum)
    batch_size=blockNum
    input_channels = 4
    output_channels = 3
    num_epochs=15000
    nchan=128
    deep=1
    pad=16
    # record time
    tic_start = time.time()
    w_div=1e2
    w_b=1e3
    w_parallel=1e3
    # first load and process data
    train_dataset,dx,dy,_=process_data(filename=data_filename,size=size,hsize=hsize,batch_size=batch_size,device=device,pad=pad,blockNum=blockNum)
    train_loader = CustomDataLoader(train_dataset.to(device), batch_size=batch_size, shuffle=True)
    # train the model only on the sign of the divergence
    model_b=create_model(inchannels=input_channels,outchannels=output_channels,dx=dx,dy=dy,nchan=nchan,deep=deep,device=device)
    print(model_b)
    criterion = CustomLoss_sign(dx=dx,dy=dy,nlayer=hsize,w_div=w_div,w_b= w_b,w_parallel=w_parallel).to(device)
    print("Starting training for 3D B ...")
    model_b,loss_record_div=train_sign(model_b,criterion,train_loader,num_epochs=num_epochs,device=device)
    print("Saving the model for 3D B ...")
    save_predictions_model(model_b,train_dataset,loss_record_div,filename='pred_CNN_3D_B.h5',model_filename='trained_model_3D_B.pth')
    tic_end = time.time()
    print(f"Total training time: {(tic_end - tic_start)/60:.4f} min")

def main_step_1():
    # Set seeds
    torch.manual_seed(123456)
    np.random.seed(123456)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device',device)
    data_filename='muram_tau_layers_200g.h5'
    # data_filename='muram_tau_layers.h5'
    size=64
    hsize=16
    blockNum=6
    # batch_size=int(blockNum*blockNum)
    batch_size=blockNum
    input_channels = 4
    output_channels = 3
    nchan=64
    deep=1
    pad=16
    # record time
    tic_start = time.time()
    # first load and process data
    # train_dataset,dx,dy,_=process_data(filename=data_filename,size=size,hsize=hsize,batch_size=batch_size,device=device,pad=pad,blockNum=blockNum)
    # train_loader = CustomDataLoader(train_dataset.to(device), batch_size=batch_size, shuffle=True)

# # ----------------------------------------------------------------------------------------------
#     # update the angle for the obs data and stitch Z variable
    _,dx,dy,_=process_data(filename=data_filename,size=size,hsize=hsize,batch_size=batch_size,device=device,pad=pad,blockNum=blockNum)
    with h5py.File('pred_CNN_3D_B.h5','r') as f:
        train_dataset = f['check_data'][:]
    train_dataset=torch.tensor(train_dataset,dtype=torch.float32)
    model_b=torch.load('trained_model_3D_B.pth')
    train_dataset= update_bobs(model_b,train_dataset,dx,dy)
    train_loader = CustomDataLoader(train_dataset.to(device), batch_size=batch_size, shuffle=True)
    input_channels = 4
    output_channels = 1
    num_epochs=15000
    w_smooth=1e4
    # w_smooth=1e5
    w_std=1e1
    print("Create 3D Z model and loss function ...")
    # create_model_z
    model_z=create_model_z(inchannels=input_channels,outchannels=output_channels,dx=dx,dy=dy,nchan=nchan,deep=deep,device=device)
    criterion = CustomLoss_Z(w_smooth=w_smooth,w_std=w_std,dx=dx,dy=dy).to(device)
    print("Starting training for 3D Z ...")
    model_z,loss_record_z=train_z(model_z,criterion,train_loader,num_epochs=num_epochs,device=device)

    tic_end = time.time()
    print("Saving the model for 3D Z ...")
    save_predictions_model(model_z,train_dataset,loss_record_z,filename='pred_CNN_3D_Z.h5',model_filename='trained_model_3D_Z.pth')
    print(f"Total training time: {(tic_end - tic_start)/60:.4f} min")

if __name__ == "__main__":
    main_step_0()
    main_step_1()