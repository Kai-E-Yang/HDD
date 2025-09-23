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
from torch.optim.lr_scheduler import StepLR

from HDD_v29 import CustomLoss_B,CustomLoss_Z,CustomLoss_Z_B
from dataload import process_data

import sys
sys.path.append('./')
from pytorch3dunet.unet3d.model import UNet3D,ResidualUNet3D

def adaptive_grad_clip(parameters, clip_factor=1e-2, grad_clipper=None, eps=1e-3):
    for p in parameters:
        if p.grad is not None:
            with torch.no_grad():
                # Clipping by Norm
                param_norm = p.data.norm()
                grad_norm = p.grad.data.norm()
                max_norm = param_norm * clip_factor
                clip_coef = (max_norm / (grad_norm + eps)).clamp(max=1.0)
                p.grad.data.mul_(clip_coef)

                # Compute gradient value (e.g., max absolute value in the gradient tensor)
                grad_abs_max = p.grad.data.abs().max()

                # Update adaptive max_grad_value
                if grad_clipper is not None:
                    max_grad_value = grad_clipper.update(grad_abs_max.item())
                    # Clipping by Value using adaptive max_grad_value
                    p.grad.data.clamp_(min=-max_grad_value, max=max_grad_value)

class AdaptiveGradValueClipper:
    def __init__(self, beta=0.9, multiplier=2.0, eps=1e-3):
        self.beta = beta
        self.multiplier = multiplier
        self.eps = eps
        self.ema_grad_value = None  # Initialize EMA of gradient values

    def update(self, grad_value):
        if self.ema_grad_value is None:
            self.ema_grad_value = grad_value
        else:
            self.ema_grad_value = self.beta * self.ema_grad_value + (1 - self.beta) * grad_value

        # Compute adaptive max_grad_value
        max_grad_value = self.multiplier * (self.ema_grad_value + self.eps)
        return max_grad_value

def train_couple(model_b,optimizer_b,model_z,optimizer_z,criterion_b,criterion_z,criterion_z_b,train_loader,lr_start=1e-3,lr_end=1e-6,loss_record=[],step_size=2000, num_epochs=1000,start_lr=1e-7,warmup_epochs=50, device="cpu",config=None):

    scheduler_b = StepLR(optimizer_b, step_size=step_size, gamma=0.5)
    scheduler_z = StepLR(optimizer_z, step_size=step_size, gamma=0.5)

    grad_clipper_b = AdaptiveGradValueClipper(beta=0.9, multiplier=2.0, eps=1e-3)
    grad_clipper_z = AdaptiveGradValueClipper(beta=0.9, multiplier=2.0, eps=1e-3)

    tic_total = time.time()
    # for epoch in range(num_epochs):
    for epoch in range(config['extend_num_epochs']):
        model_b.train()
        model_z.train()

        loss_tmp_b = 0
        loss_tmp_parallel = 0
        loss_tmp_div_sign_p = 0
        loss_tmp_div_sign_t = 0
        loss_tmp_laplace = 0
        loss_tmp_div = 0
        loss_tmp_div_j = 0
        loss_tmp_mon = 0
        loss_tmp_average = 0
        loss_tmp_smoothB = 0

        std_tmp_div_sign=0
        std_tmp_div=0

        count=0
        for batch_data,batch_err in train_loader:

            batch_data=batch_data.to(device)
            batch_err=batch_err.to(device)

            optimizer_b.zero_grad()
            optimizer_z.zero_grad()


            # add a new tanh from the model output, to constrain the final data in the range from -1 to 1
            pred_b = torch.tanh(model_b(batch_data))
            pred_z = torch.tanh(model_z(batch_data))

            # several terms here are not used during the training, but computed;
            # they will be tested in the next version and application on the real observation.

            loss_b, loss_parallel,loss_div_sign_p,std_div_sign1, loss_div_sign_t,std_div_sign2 = criterion_b(pred_b,batch_data,batch_err,epoch,num_epochs)

            loss_laplace,loss_mon,loss_average = criterion_z(pred_z,epoch,num_epochs)
            loss_div_p,std_div_p,loss_div_t,std_div_t,loss_div_j,loss_b_smooth = criterion_z_b(pred_b,pred_z,batch_data,epoch,num_epochs)

            w_div_sign=(config['w_div_sign_end']-config['w_div_sign_start'])*0.5*(math.tanh(10*(epoch/num_epochs-0.45))+1)+config['w_div_sign_start']            
            w_div=config['w_div']

            loss = loss_b*config['w_b']\
                    +loss_parallel*config['w_parallel']\
                    +loss_div_p*w_div\
                    +loss_div_sign_p*w_div_sign\
                    +loss_laplace*config['w_laplace']\
                    +loss_mon*config['w_mon']

            loss.backward()


            adaptive_grad_clip(model_b.parameters(), clip_factor=1e-2, grad_clipper=grad_clipper_b)
            optimizer_b.step()

            adaptive_grad_clip(model_z.parameters(), clip_factor=1e-2, grad_clipper=grad_clipper_z)
            optimizer_z.step()


            loss_tmp_b += loss_b.item()
            loss_tmp_parallel += loss_parallel.item()
            loss_tmp_div_sign1 += loss_div_sign_p.item()
            loss_tmp_div_sign2 += loss_div_sign_t.item()

            loss_tmp_div += (loss_div_p+loss_div_t).item()
            loss_tmp_div_j += loss_div_j.item()
            loss_tmp_laplace += loss_laplace.item()
            loss_tmp_mon += loss_mon.item()
            loss_tmp_smoothB += loss_b_smooth.item()

            std_tmp_div_sign+=(std_div_sign1+std_div_sign2).item()
            std_tmp_div+=(std_div_p+std_div_t).item()
            
            count+=1

        scheduler_b.step()
        scheduler_z.step()

        loss_tmp=[
            loss_tmp_b/count,
            loss_tmp_parallel/count,
            loss_tmp_div_sign_p/count,
            loss_tmp_div_sign_t/count,
            loss_tmp_laplace/count,
            loss_tmp_mon/count,
            loss_tmp_div/count,
            loss_tmp_div_j/count,
            loss_tmp_smoothB/count,
            std_tmp_div_sign/count,
            std_tmp_div/count
        ]

        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch+1}/{num_epochs}], LossB: {loss_tmp[0]:.2e}, '
                f'LossPa: {loss_tmp[1]:.2e}, LossST: {loss_tmp[2]:.2e}, '
                f'LossSP: {loss_tmp[3]:.2e}, LossLap: {loss_tmp[4]:.2e}, '
                f'LossMon: {loss_tmp[5]:.2e}, LossDiv: {loss_tmp[6]:.2e}, '
                f'LossDivJ: {loss_tmp[7]:.2e}, LossSB: {loss_tmp[8]:.2e}, '
                f'StdSDiv: {loss_tmp[9]:.2e}, StdDiv: {loss_tmp[10]:.2e}')

        loss_record.append(loss_tmp)
        if ((epoch+1)%2000==0 or (epoch+1)==num_epochs):
            print(f"Saving check point {epoch} ...")
            save_check_point(
                model_b,
                model_z,
                optimizer_b,
                optimizer_z,
                loss_record,
                epoch,
                model_filename_b='checkpoint_model_B_muram_v28_PR_couple')

            print("- -"*10)

    toc_total = time.time()
    toc_total = time.time()
    print(f"Total training time: {(toc_total - tic_total)/60:.4f} min")
    print("-" * 70)
    return loss_record


def save_check_point(model_b,model_z,optimizer_b,optimizer_z,loss_record,epochin,\
                        model_filename_b='checkpoint_model'):
    model_filename_b=model_filename_b+'_'+str(epochin).zfill(5)+'.pth'

    model_b.eval()
    model_z.eval()

    if isinstance(model_b, torch.nn.DataParallel) or isinstance(model_b, torch.nn.parallel.DistributedDataParallel):
        b_state_dict = model_b.module.state_dict()  # Access the underlying model
    else:
        b_state_dict = model_b.state_dict()

    if isinstance(model_z, torch.nn.DataParallel) or isinstance(model_z, torch.nn.parallel.DistributedDataParallel):
        z_state_dict = model_z.module.state_dict()  # Access the underlying model
    else:
        z_state_dict = model_z.state_dict()

    savemodel={'model_b':b_state_dict,'model_z':z_state_dict,\
            'optimizer_b': optimizer_b.state_dict(),'optimizer_z': optimizer_z.state_dict(),\
            'epoch':epochin,'loss_record':torch.from_numpy(np.array(loss_record))}
    torch.save(savemodel, model_filename_b)

    return


def load_check_point(model_b,model_z,optimizer_b,optimizer_z,filename):
    print('Loading model from',filename)
    checkpoint = torch.load(filename, map_location='cpu')
    model_state_dict = checkpoint['model_b']
    if isinstance(model_b, torch.nn.DataParallel) or isinstance(model_b, torch.nn.parallel.DistributedDataParallel):
        # If the model is wrapped, we need to load the state_dict into model_b.module
        model_b.module.load_state_dict(model_state_dict)
    else:
        # If the model is not wrapped, load the state_dict directly
        model_b.load_state_dict(model_state_dict)
    model_state_dict = checkpoint['model_z']
    if isinstance(model_z, torch.nn.DataParallel) or isinstance(model_z, torch.nn.parallel.DistributedDataParallel):
        # If the model is wrapped, we need to load the state_dict into model_z.module
        model_z.module.load_state_dict(model_state_dict)
    else:
        # If the model is not wrapped, load the state_dict directly
        model_z.load_state_dict(model_state_dict)

    optimizer_b.load_state_dict(checkpoint['optimizer_b'])
    optimizer_z.load_state_dict(checkpoint['optimizer_z'])


    return checkpoint['loss_record'].tolist()


# def main(config_data,config_model,config_loss,config_train):
def main(config):

    torch.manual_seed(123456)
    np.random.seed(123456)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ",device)

    # Load and process the data
    print("Loading data ...")
    dataset,dataerror=process_data(filename=config['filename'],\
                                    size=config['core_size'],hsize=config['hsize'],\
                                    pad=config['pad'],blockNum=config['blockNum'],device=device)

    dataset = TensorDataset(dataset,dataerror)
    train_loader= DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Create the model for sign of divergence
    print("Setting up the model B and Z ...")

    model_b=ResidualUNet3D(in_channels=config["input_channels"],out_channels=config["output_channels_b"],\
                f_maps=config["nchan"],num_levels=config["deep"],\
                conv_upscale=2,\
                upsample='deconv',\
                layer_order='cbr',\
                final_sigmoid=False, is_segmentation=False,dropout_prob=0.0).to(device)

    model_z=ResidualUNet3D(in_channels=config["input_channels"],out_channels=config["output_channels_z"],\
                f_maps=config["nchan"],num_levels=config["deep"],\
                conv_upscale=2,\
                upsample='deconv',\
                layer_order='cbr',\
                final_sigmoid=False, is_segmentation=False,dropout_prob=0.0).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model_b = torch.nn.DataParallel(model_b)
        model_z = torch.nn.DataParallel(model_z)

    print("Setting up the Custom Loss ...")

    criterion_b = CustomLoss_B(height_reg=config['height_reg'],
                            laplace_reg=config['laplace_reg'],\
                            thickness_max=config['thickness_max'],\
                            thickness_min=config['thickness_min'],\
                            dx=config['dx'],\
                            dy=config['dy'],\
                            dz=config['dz'],\
                            w_div=config['w_div'],\
                            w_div_min=config['w_div_sign_start'],\
                            w_div_max=config['w_div_sign_end'],\
                            w_b=config['w_b'],\
                            w_parallel=config['w_parallel'],\
                            w_height_max=config['w_height_max'],\
                            w_height_min=config['w_height_min'],\
                            w_thick_max=config['w_thick_max'],\
                            w_thick_min=config['w_thick_min'],\
                            w_laplace=config['w_laplace']).to(device)
    criterion_z = CustomLoss_Z(height_reg=config['height_reg'],
                            laplace_reg=config['laplace_reg'],\
                            thickness_max=config['thickness_max'],\
                            thickness_min=config['thickness_min'],\
                            dx=config['dx'],\
                            dy=config['dy'],\
                            dz=config['dz'],\
                            dz_min=config['dz_min'],\
                            w_div=config['w_div'],\
                            w_b=config['w_b'],\
                            w_parallel=config['w_parallel'],\
                            w_height_max=config['w_height_max'],\
                            w_height_min=config['w_height_min'],\
                            w_thick_max=config['w_thick_max'],\
                            w_thick_min=config['w_thick_min'],\
                            w_laplace=config['w_laplace']).to(device)
    criterion_z_b = CustomLoss_Z_B(height_reg=config['height_reg'],
                            laplace_reg=config['laplace_reg'],\
                            thickness_max=config['thickness_max'],\
                            thickness_min=config['thickness_min'],\
                            dx=config['dx'],\
                            dy=config['dy'],\
                            dz=config['dz'],\
                            w_div=config['w_div'],\
                            w_div_min=config['w_div_sign_start'],\
                            w_div_max=config['w_div_sign_end'],\
                            w_b=config['w_b'],\
                            w_parallel=config['w_parallel'],\
                            w_height_max=config['w_height_max'],\
                            w_height_min=config['w_height_min'],\
                            w_thick_max=config['w_thick_max'],\
                            w_thick_min=config['w_thick_min'],\
                            w_laplace=config['w_laplace']).to(device)

    print("Setting up the Optimizer ...")
    if config['optimizer'] == 'Adam':
        optimizer_b = torch.optim.Adam(model_b.parameters(), lr=config['learning_rate'],eps=1e-6)
        optimizer_z = torch.optim.Adam(model_z.parameters(), lr=config['learning_rate'],eps=1e-6)

    elif config['optimizer'] == 'AdamW':
        optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        optimizer_z = torch.optim.AdamW(model_z.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    
    if config['train']:
        print('Initial coupled training AdamW ...')
        loss_record=train_couple(model_b,optimizer_b,model_z,optimizer_z,\
                            criterion_b,criterion_z,criterion_z_b,train_loader,\
                            lr_start=config['lr_start'],lr_end=config['lr_end'],\
                            loss_record=[],num_epochs=config['num_epochs'],step_size=config['step_size'],\
                            config=config,device=device)
        return 0
    if config['continue']:
        loss_record=load_check_point(model_b,model_z,optimizer_b,optimizer_z,config['checkpoint'])
        print('Continue training from checkpoint ... '+config['checkpoint'])
        loss_record=train_couple(model_b,optimizer_b,model_z,optimizer_z,\
                            criterion_b,criterion_z,criterion_z_b,train_loader,\
                            lr_start=config['lr_start'],lr_end=config['lr_end'],\
                            loss_record=[],num_epochs=config['num_epochs'],step_size=config['step_size'],\
                            config=config,device=device)
        print("Finished!") 
        return 0



# if __name__ == "__main__":
#     config_data={
#         'filename':'muram_tau_layers_200g.h5',
#         'core_size':64,
#         'pad':16,
#         'hsize':32,
#         'blockNum':4
#     }
#     config_model={
#         'nchan':32,
#         'deep':2,
#         'input_channels':4,
#         'output_channels_b':3,
#         'output_channels_z':1,
#         'posi_encoding':False,
#         'FCN':False,
#         'Siren':False
#     }
#     config_loss={
#         'w_div':1e8,
#         'w_div_sign':1e8,
#         'w_div_sign_start':1e8,
#         'w_div_sign_end':1e0,
#         'w_b':1e7,
#         'w_parallel':1e10,
#         'w_j':1e1,
#         'w_laplace':1e4,
#         'w_mon':1e15,
#         'w_average':1e3,
#         'w_height_max':1e2,
#         'w_height_min':1e0,
#         'w_thick_max':1e2,
#         'w_thick_min':1e0,
#         'height_reg':0.19105607,
#         'laplace_reg':0.03665518,
#         'thickness_max':0.014761985,
#         'thickness_min':0.0046328306,
#         'dx':0.031999588*0.5,
#         'dy':0.031999588*0.5,
#         'dz':0.014398932,
#         'dz_min':0.0022426844,
#         'num_epochs_1':10,
#         'extend_num_epochs':30000,
#     }
#     config_train={
#         'train':True,
#         'continue':False,
#         'checkpoint':'checkpoint_model_B_muram_v24_fix_b_000_19999.pth',
#         'checkpoint_LBFGS':'checkpoint_model_B_muram_v26_couple_29999.pth',
#         'batch_size':8,
#         'num_epochs_1':10,
#         'num_epochs':20000,
#         'extend_num_epochs':30000,
#         'LBFGS_num_epochs':100,
#         'num_epochs_conti':20000,
#         'step_size':3000,
#         'weight_decay':1e-2,
#         'optimizer':'AdamW',
#         'learning_rate':1e-3,
#         'lr_start':1e-3,
#         'lr_end':1e-6,
#     }
#     main(config_data,config_model,config_loss,config_train)