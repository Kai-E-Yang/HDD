# This code is Hawaiian Disambiguity Decoder (HDD)
# It is used to disambiguate the azimuthal angle of 
# the magnetic field from the Stokes polarimetric data inversion
# It can solve the 180-degree ambiguity of the horizontal magnetic field
# and give the prediction of the vertical height based on the 
# divergence equaton. 
# v0.0 2023/11/17 K. Y.
# v0.29 2025/01/01 K. Y.
# yangkai@hawaii.edu; kyang@seti.org


# import standard modules
import time
import math
import numpy as np
from collections import OrderedDict

# import pytorch modules
import torch
import torch.nn as nn                     # neural networks
import torch.nn.utils as utils
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR,ExponentialLR,CosineAnnealingLR,LambdaLR


def median_filter3d(input, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be odd')
    if input.dim() != 5:
        raise ValueError('Input tensor must have 5 dimensions (N, C, D, H, W)')

    N, C, D, H, W = input.shape
    padding = kernel_size // 2

    input_padded = F.pad(input, pad=(padding, padding, padding, padding, padding, padding), mode='reflect')

    x_unfolded = input_padded.unfold(dimension=2, size=kernel_size, step=1)  # D dimension
    x_unfolded = x_unfolded.unfold(dimension=3, size=kernel_size, step=1)    # H dimension
    x_unfolded = x_unfolded.unfold(dimension=4, size=kernel_size, step=1)    # W dimension

    x_unfolded = x_unfolded.contiguous().view(N, C, D, H, W, -1)

    median = x_unfolded.median(dim=-1).values
    return median

def cal_j_3points(bxIn,byIn,bzIn,zIn,dx=0.1,dy=0.1,dzmin=0.1):
    bx_1 = bxIn[:,:,1:-1,1:-1,1:-1]

    bx_x0 = bxIn[:,:,0:-2,1:-1,1:-1]
    bx_x2 = bxIn[:,:,2:,1:-1,1:-1]

    bx_y0 = bxIn[:,:,1:-1,0:-2,1:-1]
    bx_y2 = bxIn[:,:,1:-1,2:,1:-1]

    bx_z0 = bxIn[:,:,1:-1,1:-1,0:-2]
    bx_z2 = bxIn[:,:,1:-1,1:-1,2:]

    # map by on the 0, 1, 2 point
    by_1 = byIn[:,:,1:-1,1:-1,1:-1]

    by_x0 = byIn[:,:,0:-2,1:-1,1:-1]
    by_x2 = byIn[:,:,2:,1:-1,1:-1]

    by_y0 = byIn[:,:,1:-1,0:-2,1:-1]
    by_y2 = byIn[:,:,1:-1,2:,1:-1]

    by_z0 = byIn[:,:,1:-1,1:-1,0:-2]
    by_z2 = byIn[:,:,1:-1,1:-1,2:]

    # map bz on the 0, 1, 2 point
    bz_1 = bzIn[:,:,1:-1,1:-1,1:-1]

    bz_x0 = bzIn[:,:,0:-2,1:-1,1:-1]
    bz_x2 = bzIn[:,:,2:,1:-1,1:-1]

    bz_y0 = bzIn[:,:,1:-1,0:-2,1:-1]
    bz_y2 = bzIn[:,:,1:-1,2:,1:-1]

    bz_z0 = bzIn[:,:,1:-1,1:-1,0:-2]
    bz_z2 = bzIn[:,:,1:-1,1:-1,2:]

    # map z on the 0, 1, 2 point along x
    z_x0 = zIn[:,:,0:-2,1:-1,1:-1]
    z_x2 = zIn[:,:,2:,1:-1,1:-1]

    z_y0 = zIn[:,:,1:-1,0:-2,1:-1]
    z_y2 = zIn[:,:,1:-1,2:,1:-1]

    z_z0 = zIn[:,:,1:-1,1:-1,0:-2]
    z_z2 = zIn[:,:,1:-1,1:-1,2:]


    dx_bz = (bz_x2 - bz_x0)*dy*(z_z2-z_z0).abs() + (bz_z2 - bz_z0)*(z_x2-z_x0)*dy
    dy_bz = (bz_y2 - bz_y0)*dx*(z_z2-z_z0).abs() + (bz_z2 - bz_z0)*(z_y2-z_y0)*dx

    dx_by = (by_x2 - by_x0)*dy*(z_z2-z_z0).abs() + (by_z2 - by_z0)*(z_x2-z_x0)*dy
    dz_by = (by_z2 - by_z0)*2*dx*dy

    dy_bx = (bx_y2 - bx_y0)*dx*(z_z2-z_z0).abs() + (bx_z2 - bx_z0)*(z_y2-z_y0)*dx
    dz_bx = (bx_z2 - bx_z0)*2*dx*dy

    jx = (dy_bz - dz_by)
    jy = (dz_bx - dx_bz)
    jz = (dx_by - dy_bx)


    return jx,jy,jz


def cal_j(BxIn,ByIn,BzIn,ZIn,dx=0.1,dy=0.1,dzmin=0.1):
    #    |        |
    # ---*---dy---*----
    #    |        |
    #    dx       dx
    #    |        |
    # ---*---dy---*----
    #    |        |
    # dz = (ZIn[...,1:]-ZIn[...,:-1])
    # dz = torch.where((dz==0), torch.tensor(float(dzmin), device=dz.device), dz)
    dz=dzmin


    dxBx = (BxIn[...,:-1,:,:] - BxIn[...,1:,:,:])/dx
    dxBy = (ByIn[...,:-1,:,:] - ByIn[...,1:,:,:])/dx
    dxBz = (BzIn[...,:-1,:,:] - BzIn[...,1:,:,:])/dx

    dyBx = (BxIn[...,:,:-1,:] - BxIn[...,:,1:,:])/dy
    dyBy = (ByIn[...,:,:-1,:] - ByIn[...,:,1:,:])/dy
    dyBz = (BzIn[...,:,:-1,:] - BzIn[...,:,1:,:])/dy

    dzBx = (BxIn[...,:,:,:-1] - BxIn[...,:,:,1:])/dz
    dzBy = (ByIn[...,:,:,:-1] - ByIn[...,:,:,1:])/dz
    dzBz = (BzIn[...,:,:,:-1] - BzIn[...,:,:,1:])/dz

    # avedxBx = 0.5*(dxBx[:,:-1,:] + dxBx[:,1:,:])
    avedxBy = 0.5*(dxBy[...,:,:-1,:] + dxBy[...,:,1:,:])
    avedyBx = 0.5*(dyBx[...,:-1,:,:] + dyBx[...,1:,:,:])
    # avedyBy = 0.5*(dyBy[:-1,:,:] + dyBy[1:,:,:])

    avedxBz = 0.5*(dxBz[...,:,:,:-1] + dxBz[...,:,:,1:])
    avedzBx = 0.5*(dzBx[...,:-1,:,:] + dzBx[...,1:,:,:])

    avedyBz = 0.5*(dyBz[...,:,:,:-1] + dyBz[...,:,:,1:])
    avedzBy = 0.5*(dzBy[...,:,:-1,:] + dzBy[...,:,1:,:])

    jx = avedyBz - avedzBy
    jy = avedzBx - avedxBz
    jz = avedxBy - avedyBx
    return jx,jy,jz
    

def safe_divide(numerator, denominator, eps=1e-8):
    denominator = torch.where(denominator.abs() > eps, denominator, torch.full_like(denominator, fill_value=eps))
    return numerator / denominator

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


def naninfmean(tensor):
    # Replace NaNs with negative infinity
    tensor = torch.where(torch.isnan(tensor), torch.tensor(float('inf'), device=tensor.device), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.tensor(float(0.0), device=tensor.device), tensor)

    # Compute the maximum
    return torch.mean(tensor)


def laplace2d5(datain):
    laplacedz = 4*datain[:,:,1:-1,1:-1,:] - \
                datain[:,:,:-2,1:-1,:] - \
                datain[:,:,2:,1:-1,:] - \
                datain[:,:,1:-1,:-2,:] - \
                datain[:,:,1:-1,2:,:]
    return laplacedz

def laplace2d9(datain):
    laplacedz = 20./6.0*datain[:,:,1:-1,1:-1,:] - \
                4.0/6.0*datain[:,:,:-2,1:-1,:] - \
                4.0/6.0*datain[:,:,2:,1:-1,:] - \
                4.0/6.0*datain[:,:,1:-1,:-2,:] - \
                4.0/6.0*datain[:,:,1:-1,2:,:] - \
                1.0/6.0*datain[:,:,:-2,:-2,:] - \
                1.0/6.0*datain[:,:,:-2,2:,:] - \
                1.0/6.0*datain[:,:,2:,:-2,:] - \
                1.0/6.0*datain[:,:,2:,2:,:]
    return laplacedz

def laplace3d7(datain):
    laplacedz = 6*datain[:,:,1:-1,1:-1,1:-1] - \
                datain[:,:,:-2,1:-1,1:-1] - \
                datain[:,:,2:,1:-1,1:-1] - \
                datain[:,:,1:-1,:-2,1:-1] - \
                datain[:,:,1:-1,2:,1:-1] - \
                datain[:,:,1:-1,1:-1,2:] - \
                datain[:,:,1:-1,1:-1,:-2]
    return laplacedz

def laplace3d27(datain):
    laplacedz = 88*datain[:,:,1:-1,1:-1,1:-1] - \
                6*datain[:,:,:-2,1:-1,1:-1] - \
                6*datain[:,:,2:,1:-1,1:-1] - \
                6*datain[:,:,1:-1,:-2,1:-1] - \
                6*datain[:,:,1:-1,2:,1:-1] - \
                6*datain[:,:,1:-1,1:-1,2:] - \
                6*datain[:,:,1:-1,1:-1,:-2] - \
                3*datain[:,:,2:,2:,1:-1] - \
                3*datain[:,:,:-2,2:,1:-1] - \
                3*datain[:,:,2:,:-2,1:-1] - \
                3*datain[:,:,:-2,:-2,1:-1] - \
                3*datain[:,:,2:,1:-1,2:] - \
                3*datain[:,:,:-2,1:-1,2:] - \
                3*datain[:,:,2:,1:-1,:-2] - \
                3*datain[:,:,:-2,1:-1,:-2] - \
                3*datain[:,:,1:-1,2:,2:] - \
                3*datain[:,:,1:-1,:-2,2:] - \
                3*datain[:,:,1:-1,2:,:-2] - \
                3*datain[:,:,1:-1,:-2,:-2] - \
                2*datain[:,:,:-2,:-2,:-2] - \
                2*datain[:,:,:-2,:-2,2:] - \
                2*datain[:,:,:-2,2:,:-2] - \
                2*datain[:,:,:-2,2:,2:] - \
                2*datain[:,:,2:,:-2,:-2] - \
                2*datain[:,:,2:,:-2,2:] - \
                2*datain[:,:,2:,2:,:-2] - \
                2*datain[:,:,2:,2:,2:]  
    return laplacedz


# define the custom loss function
class CustomLoss_B(nn.Module):
    def __init__(self,height_reg=1,laplace_reg=1,thickness_max=1,thickness_min=1,\
                dx=1,dy=1,dz=1,eps=1e-10,w_b=1e10,w_parallel=1e10,w_div=1e9,w_div_min=1e9,w_div_max=1e9,\
                w_height_max=1e6,w_height_min=1e1,w_thick_max=1e6,w_thick_min=1e2,w_laplace=1e2):
        super(CustomLoss_B, self).__init__()
        self.dx=1
        self.dy=dy/dx
        self.dz=dz/dx
        self.w_b=w_b
        self.w_parallel=w_parallel
        self.w_div=w_div
        self.w_div_min=w_div_min
        self.w_div_max=w_div_max
        self.w_thick_max=w_thick_max
        self.w_thick_min=w_thick_min
        self.w_height_max=w_height_max
        self.w_height_min=w_height_min
        self.w_laplace=w_laplace
        self.eps = eps  # Small constant to avoid log(0)
        self.height_reg=height_reg
        self.laplace_reg=laplace_reg
        self.thickness_max=thickness_max
        self.thickness_min=thickness_min
    def forward(self,pred_b,targets,errors,iepoch,epoch_max):
        bx_t,by_t,bz_t=targets[:,0:1],targets[:,1:2],targets[:,2:3]
        bx_p,by_p,bz_p=pred_b[:,0:1],pred_b[:,1:2],pred_b[:,2:3]


        bt_p = torch.sqrt(bx_p**2+by_p**2)
        bt_t = torch.sqrt(bx_t**2+by_t**2)
        bt = torch.sqrt(bx_t**2+by_t**2)



        loss_b=torch.nanmean((bt_p-bt_t)**4/(bx_t**2+by_t**2+1e-8))+\
            10*torch.nanmean((bz_p-bz_t)**4/(bz_t**2+1e-8))

        loss_parallel=torch.nanmean((bx_p*by_t - by_p*bx_t)**2/(bx_t**2+by_t**2+bz_t**2 + 1e-8))


        loss_div_t,std_t=cal_div_sign(bx_p,by_p,bz_t,self.dx,self.dy,self.dz)

        loss_div_p,std_p=cal_div_sign(bx_p,by_p,bz_p,self.dx,self.dy,self.dz)


        return loss_b,loss_parallel,loss_div_p,std_p,loss_div_t,std_t

class CustomLoss_Z(nn.Module):
    def __init__(self,height_reg=1,laplace_reg=1,thickness_max=1,thickness_min=1,\
                dx=1,dy=1,dz=1,dz_min=1,dz_std=0.023076871,eps=1e-10,w_b=1e10,w_parallel=1e10,w_div=1e9,w_div_min=1e9,w_div_max=1e9,\
                w_height_max=1e6,w_height_min=1e1,w_thick_max=1e6,w_thick_min=1e2,w_laplace=1e2):
        super(CustomLoss_Z, self).__init__()
        self.dx=dx
        self.dy=dy
        self.dz_min=dz_min
        self.dz_std=dz_std
        self.w_b=w_b
        self.w_parallel=w_parallel
        self.w_div=w_div
        self.w_thick_max=w_thick_max
        self.w_thick_min=w_thick_min
        self.w_height_max=w_height_max
        self.w_height_min=w_height_min
        self.w_laplace=w_laplace
        self.eps = eps  # Small constant to avoid log(0)
        self.height_reg=height_reg
        self.laplace_reg=laplace_reg
        self.thickness_max=thickness_max
        self.thickness_min=thickness_min

    def forward(self,pred_z,iepoch,epoch_max):
        dz=pred_z[...,1:] - pred_z[...,:-1]
        laplacedz=laplace3d7(dz/self.dx)
        loss_laplace=torch.nanmean(laplacedz.pow(2))

        smoothz=median_filter3d(dz/self.dx,kernel_size=3)
        loss_smooth=torch.nanmean((dz/self.dx-smoothz).pow(2))

        loss_mon = torch.max(torch.clamp(-(dz-self.dz_min)/self.dx, min=0))


        loss_average = torch.median(pred_z[...,0])**2

        return loss_smooth,loss_mon,loss_average


class CustomLoss_Z_B(nn.Module):    
    def __init__(self,height_reg=1,laplace_reg=1,thickness_max=1,thickness_min=1,\
                dx=1,dy=1,dz=1,eps=1e-10,w_b=1e10,w_parallel=1e10,w_div=1e9,w_div_min=1e9,w_div_max=1e9,\
                w_height_max=1e6,w_height_min=1e1,w_thick_max=1e6,w_thick_min=1e2,w_laplace=1e2):
        super(CustomLoss_Z_B, self).__init__()
        self.dx=dx
        self.dy=dy
        self.dz=dz
        self.w_b=w_b
        self.w_parallel=w_parallel
        self.w_div=w_div
        self.w_div_min=w_div_min
        self.w_div_max=w_div_max
        self.w_thick_max=w_thick_max
        self.w_thick_min=w_thick_min
        self.w_height_max=w_height_max
        self.w_height_min=w_height_min
        self.w_laplace=w_laplace
        self.eps = eps  # Small constant to avoid log(0)
        self.height_reg=height_reg
        self.laplace_reg=laplace_reg
        self.thickness_max=thickness_max
        self.thickness_min=thickness_min
    def forward(self,pred_b,pred_z,targets,iepoch,epoch_max):
        bx_t,by_t,bz_t=targets[:,0:1],targets[:,1:2],targets[:,2:3]
        bt_p=torch.sqrt(pred_b[:,0:1]**2+pred_b[:,1:2]**2)
        bt_t=torch.sqrt(targets[:,0:1]**2+targets[:,1:2]**2)

        bx_p,by_p=pred_b[:,0:1],pred_b[:,1:2]
        bz_p=pred_b[:,2:3]
        mask=2.0*((bx_p*bx_t+by_p*by_t)>0)-1.0



        loss_div_p,std_p=cal_div_c_old(bx_p,by_p,bz_t,pred_z/self.dx,self.dx/self.dx,self.dy/self.dx)

        loss_div_t,std_t=cal_div_c_old(bx_t*mask,by_t*mask,bz_t,pred_z/self.dx,self.dx/self.dx,self.dy/self.dx)


        jx_pred,jy_pred,jz_pred= cal_j(bx_t*mask,by_t*mask,bz_p,pred_z/self.dx,dx=self.dx/self.dx,dy=self.dy/self.dx,dzmin=self.dz/self.dx)
        jx_filter= median_filter3d(jx_pred,kernel_size=5)
        jy_filter= median_filter3d(jy_pred,kernel_size=5)
        jz_filter= median_filter3d(jz_pred,kernel_size=5)

        loss_j=torch.nanmean((jx_filter-jx_pred)**2)\
                +torch.nanmean((jy_filter-jy_pred)**2)\
                +torch.nanmean((jz_filter-jz_pred)**2)

        bx_filter_1= median_filter3d(bx_t*mask,kernel_size=5)
        by_filter_1= median_filter3d(by_t*mask,kernel_size=5)

        bx_filter_0= median_filter3d(bx_p,kernel_size=5)
        by_filter_0= median_filter3d(by_p,kernel_size=5)

        loss_b_smooth=torch.nanmean((bx_t*mask-bx_filter_1)**2)\
                +torch.nanmean((by_t*mask-by_filter_1)**2)\
                +torch.nanmean((bx_p-bx_filter_0)**2)\
                +torch.nanmean((by_p-by_filter_0)**2)

        return loss_div_p,std_p,loss_div_t,std_t,loss_j,loss_b_smooth

def cal_div_sign(bxIn,byIn,bzIn,dx,dy,dz):
    bx_000,by_000,bz_000=bxIn[:,:,0:-1,0:-1,0:-1],byIn[:,:,0:-1,0:-1,0:-1],bzIn[:,:,0:-1,0:-1,0:-1]
    bx_100,by_100,bz_100=bxIn[:,:,1:,0:-1,0:-1],byIn[:,:,1:,0:-1,0:-1],bzIn[:,:,1:,0:-1,0:-1]
    bx_010,by_010,bz_010=bxIn[:,:,0:-1,1:,0:-1],byIn[:,:,0:-1,1:,0:-1],bzIn[:,:,0:-1,1:,0:-1]
    bx_110,by_110,bz_110=bxIn[:,:,1:,1:,0:-1],byIn[:,:,1:,1:,0:-1],bzIn[:,:,1:,1:,0:-1]

    bx_001,by_001,bz_001=bxIn[:,:,0:-1,0:-1,1:],byIn[:,:,0:-1,0:-1,1:],bzIn[:,:,0:-1,0:-1,1:]
    bx_101,by_101,bz_101=bxIn[:,:,1:,0:-1,1:],byIn[:,:,1:,0:-1,1:],bzIn[:,:,1:,0:-1,1:]
    bx_011,by_011,bz_011=bxIn[:,:,0:-1,1:,1:],byIn[:,:,0:-1,1:,1:],bzIn[:,:,0:-1,1:,1:]
    bx_111,by_111,bz_111=bxIn[:,:,1:,1:,1:],byIn[:,:,1:,1:,1:],bzIn[:,:,1:,1:,1:]


    res_flx=(\
        (bx_100+bx_110+bx_101+bx_111)*dy*dz-\
        (bx_000+bx_010+bx_001+bx_011)*dy*dz+\
        (by_010+by_110+by_011+by_111)*dx*dz-\
        (by_000+by_100+by_001+by_101)*dx*dz+\
        (bz_001 + bz_011 + bz_101 + bz_111)*dx*dy-\
        (bz_000 + bz_010 + bz_100 + bz_110)*dx*dy
        )**2
    avg_b=(\
            ((bx_000+bx_001+bx_010+bx_011 + bx_100+bx_101+bx_110+bx_111)*0.125)**2+\
            ((by_000+by_001+by_010+by_011 + by_100+by_101+by_110+by_111)*0.125)**2+\
            ((bz_000+bz_001+bz_010+bz_011 + bz_100+bz_101+bz_110+bz_111)*0.125)**2+1e-8\
            )
    # flx = res_flx/torch.sqrt(avg_b)

    flx = res_flx**2/avg_b

    loss_div=torch.nanmean(flx)

    loss_std=torch.nanmean((flx - loss_div)**2)

    return loss_div,loss_std


def cal_div_c_old(bxIn,byIn,bzIn,zIn,dx,dy):
    # define the divergence of the predicted field in the integrated form.
    bx_000,by_000,bz_000,z_000=bxIn[:,:,0:-1,0:-1,0:-1],byIn[:,:,0:-1,0:-1,0:-1],bzIn[:,:,0:-1,0:-1,0:-1],zIn[:,:,0:-1,0:-1,0:-1]
    bx_100,by_100,bz_100,z_100=bxIn[:,:,1:,0:-1,0:-1],byIn[:,:,1:,0:-1,0:-1],bzIn[:,:,1:,0:-1,0:-1],zIn[:,:,1:,0:-1,0:-1]
    bx_010,by_010,bz_010,z_010=bxIn[:,:,0:-1,1:,0:-1],byIn[:,:,0:-1,1:,0:-1],bzIn[:,:,0:-1,1:,0:-1],zIn[:,:,0:-1,1:,0:-1]
    bx_110,by_110,bz_110,z_110=bxIn[:,:,1:,1:,0:-1],byIn[:,:,1:,1:,0:-1],bzIn[:,:,1:,1:,0:-1],zIn[:,:,1:,1:,0:-1]

    bx_001,by_001,bz_001,z_001=bxIn[:,:,0:-1,0:-1,1:],byIn[:,:,0:-1,0:-1,1:],bzIn[:,:,0:-1,0:-1,1:],zIn[:,:,0:-1,0:-1,1:]
    bx_101,by_101,bz_101,z_101=bxIn[:,:,1:,0:-1,1:],byIn[:,:,1:,0:-1,1:],bzIn[:,:,1:,0:-1,1:],zIn[:,:,1:,0:-1,1:]
    bx_011,by_011,bz_011,z_011=bxIn[:,:,0:-1,1:,1:],byIn[:,:,0:-1,1:,1:],bzIn[:,:,0:-1,1:,1:],zIn[:,:,0:-1,1:,1:]
    bx_111,by_111,bz_111,z_111=bxIn[:,:,1:,1:,1:],byIn[:,:,1:,1:,1:],bzIn[:,:,1:,1:,1:],zIn[:,:,1:,1:,1:]

    res_flx1=(\
                0.25*(bx_100+bx_110+bx_101+bx_111)*dy*0.5*((z_101-z_100).abs()+(z_111-z_110).abs())-\
                0.25*(bx_000+bx_010+bx_001+bx_011)*dy*0.5*((z_001-z_000).abs()+(z_011-z_010).abs())+\
                0.25*(by_010+by_110+by_011+by_111)*dx*0.5*((z_011-z_010).abs()+(z_111-z_110).abs())-\
                0.25*(by_000+by_100+by_001+by_101)*dx*0.5*((z_001-z_000).abs()+(z_101-z_100).abs())+\
                0.5*( (bz_001 + bz_101 + bz_111)/3 + (bz_001+bz_111+bz_011)/3 )*dx*dy-\
                0.5*( (bz_000 + bz_100 + bz_110)/3 + (bz_000+bz_110+bz_010)/3 )*dx*dy+\
                (bx_001+bx_101+bx_111)*dy*(z_001-z_101)/6+\
                (bx_001+bx_011+bx_111)*dy*(z_011-z_111)/6+\
                (by_001+by_101+by_111)*dx*(z_101-z_111)/6+\
                (by_001+by_011+by_111)*dx*(z_001-z_011)/6-\
                ((bx_000+bx_100+bx_110)*dy*(z_000-z_100)/6+\
                (bx_000+bx_010+bx_110)*dy*(z_010-z_110)/6+\
                (by_000+by_100+by_110)*dx*(z_100-z_110)/6+\
                (by_000+by_010+by_110)*dx*(z_000-z_010)/6)\
            )**2


    ave_b=(\
            ((bx_000+bx_001+bx_010+bx_011 + bx_100+bx_101+bx_110+bx_111)*0.125)**2+\
            ((by_000+by_001+by_010+by_011 + by_100+by_101+by_110+by_111)*0.125)**2+\
            ((bz_000+bz_001+bz_010+bz_011 + bz_100+bz_101+bz_110+bz_111)*0.125)**2+1e-8\
        )

    # flx1 = res_flx1/torch.sqrt(ave_b)

    flx1 = res_flx1**2/ave_b

    loss_div=torch.nanmean(flx1)

    loss_std=torch.nanmean((flx1 - loss_div)**2)

    return loss_div,loss_std
