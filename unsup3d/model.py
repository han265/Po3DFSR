import os
import cv2
import math
import glob
import torch
import numpy as np
import torch.nn as nn
import torchvision
from . import networks
from . import utils
from .renderer import Renderer
import torch.nn.functional as F



EPS = 1e-7


class Unsup3D():
    def __init__(self, cfgs):
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('image_size', 64)
        self.min_depth = cfgs.get('min_depth', 0.9)
        self.max_depth = cfgs.get('max_depth', 1.1)
        self.border_depth = cfgs.get('border_depth', (0.7*self.max_depth + 0.3*self.min_depth))
        self.min_amb_light = cfgs.get('min_amb_light', 0.)
        self.max_amb_light = cfgs.get('max_amb_light', 1.)
        self.min_diff_light = cfgs.get('min_diff_light', 0.)
        self.max_diff_light = cfgs.get('max_diff_light', 1.)
        self.xyz_rotation_range = cfgs.get('xyz_rotation_range', 60)
        self.xy_translation_range = cfgs.get('xy_translation_range', 0.1)
        self.z_translation_range = cfgs.get('z_translation_range', 0.1)
        self.use_conf_map = cfgs.get('use_conf_map', True)
        self.lam_perc = cfgs.get('lam_perc', 1)
        self.lam_flip = cfgs.get('lam_flip', 0.5)
        self.lam_flip_start_epoch = cfgs.get('lam_flip_start_epoch', 0)
        self.lam_depth_sm = cfgs.get('lam_depth_sm', 0)
        self.lam_slow_3d = cfgs.get('lam_slow_3d', 60)
        self.lam_3d = cfgs.get('lam_3d', 1.0)
        self.alpha = cfgs.get('alpha', 1) 
        self.beta = cfgs.get('beta', 0.05)
        self.theta = cfgs.get('theta', 1) 
        self.gamma = cfgs.get('gamma', 0.05)
        self.lr = cfgs.get('lr', 1e-4)
        self.load_gt_depth = cfgs.get('load_gt_depth', False)
        self.renderer = Renderer(cfgs)
        
        ## networks and optimizers
        self.netG_RLS = networks.G_RLS()
        self.netG_DHL = networks.G_DHL()
        self.netG_DSL = networks.G_DSL()
        self.netD_L1 = networks.Discriminator(16)
        self.netD_L2 = networks.Discriminator(16)
        self.netD_H1 = networks.Discriminator(64)
        self.netD_H2 = networks.Discriminator(64)
        self.freezed_D = networks.EDDeconv(cin=3, cout=1, nf=64, zdim=256, activation=None, requires_grad=False)
        self.freezed_A = networks.EDDeconv(cin=3, cout=3, nf=64, zdim=256, requires_grad=False)
        self.freezed_L = networks.Encoder(cin=3, cout=4, nf=32, requires_grad=False)
        self.freezed_V = networks.Encoder(cin=3, cout=6, nf=32, requires_grad=False)
        if self.use_conf_map:
            self.freezed_C = networks.ConfNet(cin=3, cout=2, nf=64, zdim=128, requires_grad=False)
            
        self.network_names = [k for k in vars(self) if 'net' in k]
        self.freezed_network_names = [k for k in vars(self) if 'freezed' in k]
        
        self.make_optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, betas=(0.9, 0.999))
        self.make_lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00001)
        
        ## other parameters
        self.PerceptualLoss = networks.PerceptualLoss(requires_grad=False)
        self.other_param_names = ['PerceptualLoss']
        self.L1Loss = nn.L1Loss(reduction='mean')    

        ## depth rescaler: -1~1 -> min_deph~max_deph
        self.depth_rescaler = lambda d : (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth
        self.amb_light_rescaler = lambda x : (1+x)/2 *self.max_amb_light + (1-x)/2 *self.min_amb_light
        self.diff_light_rescaler = lambda x : (1+x)/2 *self.max_diff_light + (1-x)/2 *self.min_diff_light

    def init_optimizers(self):
        self.optimizer_names = []
        for net_name in self.network_names:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net','optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]
            
    def init_lr_scheduler(self):
        self.lr_scheduler_names = []
        for optimizer_name in self.optimizer_names:
            lr_scheduler = self.make_lr_scheduler(getattr(self, optimizer_name))
            lr_scheduler_name = optimizer_name.replace('optimizer', 'lr_scheduler')
            setattr(self, lr_scheduler_name, lr_scheduler)
            self.lr_scheduler_names += [lr_scheduler_name]

    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names:
                getattr(self, k).load_state_dict(cp[k])

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names:
                getattr(self, k).load_state_dict(cp[k])
    
    def load_lr_scheduler_state(self, cp):
        for k in cp:
            if k and k in self.lr_scheduler_names:
                getattr(self, k).load_state_dict(cp[k])

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states
    
    def get_lr_scheduler_state(self):
        states = {}
        for lr_scheduler_name in self.lr_scheduler_names:
            states[lr_scheduler_name] = getattr(self, lr_scheduler_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        print(device)
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))
        for freezed_net_name in self.freezed_network_names:
            setattr(self, freezed_net_name, getattr(self, freezed_net_name).to(device))
        if self.other_param_names:
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()
        for freezed_net_name in self.freezed_network_names:
            getattr(self, freezed_net_name).eval()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()
        for freezed_net_name in self.freezed_network_names:
            getattr(self, freezed_net_name).eval()

    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None):
        loss = (im1-im2).abs()
        if conf_sigma is not None:
            loss = loss *2**0.5 / (conf_sigma +EPS) + (conf_sigma +EPS).log()
        if mask is not None:
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def backward(self):
        for optim_name in self.optimizer_names:
            getattr(self, optim_name).zero_grad()
        self.loss_D_L1.backward()
        self.loss_D_H1.backward()
        self.loss_D_L2.backward()
        self.loss_D_H2.backward()
        self.loss_G_DHL.backward()
        self.loss_G_DSL.backward()
        self.loss_G_RLS = self.theta * self.loss_G_RLS_1 + self.gamma * self.loss_G_RLS_2 + self.lam_3d * self.loss_3d_Recon
        self.loss_G_RLS.backward()
        for optim_name in self.optimizer_names:
            lr_name = optim_name.replace('optimizer', 'lr_')
            getattr(self, optim_name).step()
            setattr(self, lr_name, next(iter(getattr(self, optim_name).param_groups))['lr']) 
        for lr_scheduler_name in self.lr_scheduler_names:
            getattr(self, lr_scheduler_name).step()

        

    def forward1(self, batch):
        
        if self.load_gt_depth:
            input, depth_gt = input

        self.zs = batch["z"].cuda()
        self.lrs = batch["lr"].cuda()
        self.hrs = batch["hr"].cuda()
        self.downs = batch["hr_down"].cuda()
        self.ups = F.interpolate(self.lrs, size=[64, 64], mode="bicubic", align_corners=True)
        self.ups_detach = self.ups.detach()

        # forward H -> L -> H
        self.lr_gen = self.netG_DHL(self.hrs, self.zs)
        self.lr_gen_detach = self.lr_gen.detach()
        self.hr_gen = self.netG_RLS(self.lr_gen_detach) 
        self.hr_gen_detach = self.hr_gen.detach()
        
        self.input_im_for3d1 = self.hr_gen
        b, c, h, w = self.input_im_for3d1.shape
        
        
        ## predict canonical depth
        self.canon_depth_raw1 = self.freezed_D(self.input_im_for3d1).squeeze(1)  # BxHxW
        self.canon_depth1 = self.canon_depth_raw1 - self.canon_depth_raw1.view(b,-1).mean(1).view(b,1,1)
        self.canon_depth1 = self.canon_depth1.tanh()
        self.canon_depth1 = self.depth_rescaler(self.canon_depth1)

        ## optional depth smoothness loss (only used in synthetic car experiments)
        self.loss_depth_sm1 = ((self.canon_depth1[:,:-1,:] - self.canon_depth1[:,1:,:]) /(self.max_depth-self.min_depth)).abs().mean()
        self.loss_depth_sm1 += ((self.canon_depth1[:,:,:-1] - self.canon_depth1[:,:,1:]) /(self.max_depth-self.min_depth)).abs().mean()

        ## clamp border depth
        depth_border1 = torch.zeros(1,h,w-4).to(self.input_im_for3d1.device)
        depth_border1 = nn.functional.pad(depth_border1, (2,2), mode='constant', value=1)
        self.canon_depth1 = self.canon_depth1*(1-depth_border1) + depth_border1 *self.border_depth
        self.canon_depth1 = torch.cat([self.canon_depth1, self.canon_depth1.flip(2)], 0)  # flip

        ## predict canonical albedo
        self.canon_albedo1 = self.freezed_A(self.input_im_for3d1)  # Bx3xHxW
        self.canon_albedo1 = torch.cat([self.canon_albedo1, self.canon_albedo1.flip(3)], 0)  # flip

        ## predict confidence map
        if self.use_conf_map:
            conf_sigma_l1, conf_sigma_percl = self.freezed_C(self.input_im_for3d1)  # Bx2xHxW
            self.conf_sigma_l11 = conf_sigma_l1[:,:1]
            self.conf_sigma_l1_flip1 = conf_sigma_l1[:,1:]
            self.conf_sigma_percl1 = conf_sigma_percl[:,:1]
            self.conf_sigma_percl_flip1 = conf_sigma_percl[:,1:]
        else:
            self.conf_sigma_l11 = None
            self.conf_sigma_l1_flip1 = None
            self.conf_sigma_percl1 = None
            self.conf_sigma_percl_flip1 = None

        ## predict lighting
        canon_light = self.freezed_L(self.input_im_for3d1).repeat(2,1)  # Bx4
        self.canon_light_a1 = self.amb_light_rescaler(canon_light[:,:1])  # ambience term
        self.canon_light_b1 = self.diff_light_rescaler(canon_light[:,1:2])  # diffuse term
        canon_light_dxy = canon_light[:,2:]
        self.canon_light_d1 = torch.cat([canon_light_dxy, torch.ones(b*2,1).to(self.input_im_for3d1.device)], 1)
        self.canon_light_d1 = self.canon_light_d1 / ((self.canon_light_d1**2).sum(1, keepdim=True))**0.5  # diffuse light direction

        ## shading
        self.canon_normal1 = self.renderer.get_normal_from_depth(self.canon_depth1)
        self.canon_diffuse_shading1 = (self.canon_normal1 * self.canon_light_d1.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shading = self.canon_light_a1.view(-1,1,1,1) + self.canon_light_b1.view(-1,1,1,1)*self.canon_diffuse_shading1
        self.canon_im1 = (self.canon_albedo1/2+0.5) * canon_shading *2-1

        ## predict viewpoint transformation
        self.view1 = self.freezed_V(self.input_im_for3d1).repeat(2,1)
        self.view1 = torch.cat([
            self.view1[:,:3] *math.pi/180 *self.xyz_rotation_range,
            self.view1[:,3:5] *self.xy_translation_range,
            self.view1[:,5:] *self.z_translation_range], 1)

        ## reconstruct input view
        self.renderer.set_transform_matrices(self.view1)
        self.recon_depth1 = self.renderer.warp_canon_depth(self.canon_depth1)
        self.recon_normal1 = self.renderer.get_normal_from_depth(self.recon_depth1)
        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth1)
        self.recon_im1 = nn.functional.grid_sample(self.canon_im1, grid_2d_from_canon, mode='bilinear')

        margin = (self.max_depth - self.min_depth) /2
        recon_im_mask = (self.recon_depth1 < self.max_depth+margin).float()  # invalid border pixels have been clamped at max_depth+margin
        recon_im_mask_both = recon_im_mask[:b] * recon_im_mask[b:]  # both original and flip reconstruction
        recon_im_mask_both = recon_im_mask_both.repeat(2,1,1).unsqueeze(1).detach()
        self.recon_im1 = self.recon_im1 * recon_im_mask_both
        
        
        ## loss function
        self.loss_D_L1 = nn.ReLU()(1.0 - self.netD_L1(self.lrs)).mean() + nn.ReLU()(1 + self.netD_L1(self.lr_gen_detach)).mean()
        self.loss_D_H1 = nn.ReLU()(1.0 - self.netD_H1(self.hrs)).mean() + nn.ReLU()(1 + self.netD_H1(self.hr_gen_detach)).mean()

        self.gan_loss_DHL = -self.netD_L1(self.lr_gen).mean()
        self.L1_loss_DHL = self.L1Loss(self.lr_gen, self.downs)
        self.loss_G_DHL = self.alpha * self.L1_loss_DHL + self.beta * self.gan_loss_DHL

        self.gan_loss_RLS_1 = -self.netD_H1(self.hr_gen).mean()
        self.Cyc_loss_RLS_1 = self.L1Loss(self.hr_gen, self.hrs)
        self.loss_G_RLS_1 = self.alpha * self.Cyc_loss_RLS_1 + self.beta * self.gan_loss_RLS_1

        metrics = {
                    'Cyc_loss_RLS_1': self.Cyc_loss_RLS_1,
                    'gan_loss_RLS_1': self.gan_loss_RLS_1,
                    'loss_net_G_RLS_1': self.loss_G_RLS_1,
                    'gan_loss_DHL': self.gan_loss_DHL,
                    'L1_loss_DHL': self.L1_loss_DHL,
                    'loss_net_G_DHL': self.loss_G_DHL, 
                    'loss_net_DL1': self.loss_D_L1, 
                    'loss_net_DH1': self.loss_D_H1
                }
        
        return metrics
    
    def forward2(self, batch):
        
        
        if self.load_gt_depth:
            input, depth_gt = input
            
        self.ups = F.interpolate(self.lrs, size=[64, 64], mode="bicubic", align_corners=True)
        self.ups_detach = self.ups.detach()
        
        # backward L -> H -> L
        self.hr_gen_r = self.netG_RLS(self.lrs) 
        self.hr_gen_r_detach = self.hr_gen_r.detach()
        self.lr_real_2 = self.netG_DSL(self.hr_gen_r_detach, self.zs)
        self.lr_real_2_detach = self.lr_real_2.detach()
        
        self.input_im_for3d2 = self.hrs
        b, c, h, w = self.input_im_for3d2.shape

        ## predict canonical depth
        self.canon_depth_raw2 = self.freezed_D(self.input_im_for3d2).squeeze(1)  # BxHxW
        self.canon_depth2 = self.canon_depth_raw2 - self.canon_depth_raw2.view(b,-1).mean(1).view(b,1,1)
        self.canon_depth2 = self.canon_depth2.tanh()
        self.canon_depth2 = self.depth_rescaler(self.canon_depth2)

        ## optional depth smoothness loss (only used in synthetic car experiments)
        self.loss_depth_sm2 = ((self.canon_depth2[:,:-1,:] - self.canon_depth2[:,1:,:]) /(self.max_depth-self.min_depth)).abs().mean()
        self.loss_depth_sm2 += ((self.canon_depth2[:,:,:-1] - self.canon_depth2[:,:,1:]) /(self.max_depth-self.min_depth)).abs().mean()

        ## clamp border depth
        depth_border = torch.zeros(1,h,w-4).to(self.input_im_for3d2.device)
        depth_border = nn.functional.pad(depth_border, (2,2), mode='constant', value=1)
        self.canon_depth2 = self.canon_depth2*(1-depth_border) + depth_border *self.border_depth
        self.canon_depth2 = torch.cat([self.canon_depth2, self.canon_depth2.flip(2)], 0)  # flip

        ## predict canonical albedo
        self.canon_albedo2 = self.freezed_A(self.input_im_for3d2)  # Bx3xHxW
        self.canon_albedo2 = torch.cat([self.canon_albedo2, self.canon_albedo2.flip(3)], 0)  # flip

        ## predict confidence map
        if self.use_conf_map:
            conf_sigma_l1, conf_sigma_percl = self.freezed_C(self.input_im_for3d2)  # Bx2xHxW
            self.conf_sigma_l12 = conf_sigma_l1[:,:1]
            self.conf_sigma_l1_flip2 = conf_sigma_l1[:,1:]
            self.conf_sigma_percl2 = conf_sigma_percl[:,:1]
            self.conf_sigma_percl_flip2 = conf_sigma_percl[:,1:]
        else:
            self.conf_sigma_l12 = None
            self.conf_sigma_l1_flip2 = None
            self.conf_sigma_percl2 = None
            self.conf_sigma_percl_flip2 = None

        ## predict lighting
        canon_light = self.freezed_L(self.input_im_for3d2).repeat(2,1)  # Bx4
        self.canon_light_a2 = self.amb_light_rescaler(canon_light[:,:1])  # ambience term
        self.canon_light_b2 = self.diff_light_rescaler(canon_light[:,1:2])  # diffuse term
        canon_light_dxy = canon_light[:,2:]
        self.canon_light_d2 = torch.cat([canon_light_dxy, torch.ones(b*2,1).to(self.input_im_for3d2.device)], 1)
        self.canon_light_d2 = self.canon_light_d2 / ((self.canon_light_d2**2).sum(1, keepdim=True))**0.5  # diffuse light direction

        ## shading
        self.canon_normal2 = self.renderer.get_normal_from_depth(self.canon_depth2)
        self.canon_diffuse_shading2 = (self.canon_normal2 * self.canon_light_d2.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shading = self.canon_light_a2.view(-1,1,1,1) + self.canon_light_b2.view(-1,1,1,1)*self.canon_diffuse_shading2
        self.canon_im2 = (self.canon_albedo2/2+0.5) * canon_shading *2-1

        ## predict viewpoint transformation
        self.view2 = self.freezed_V(self.input_im_for3d2).repeat(2,1)
        self.view2 = torch.cat([
            self.view2[:,:3] *math.pi/180 *self.xyz_rotation_range,
            self.view2[:,3:5] *self.xy_translation_range,
            self.view2[:,5:] *self.z_translation_range], 1)

        ## reconstruct input view
        self.renderer.set_transform_matrices(self.view2)
        self.recon_depth2 = self.renderer.warp_canon_depth(self.canon_depth2)
        self.recon_normal2 = self.renderer.get_normal_from_depth(self.recon_depth2)
        grid_2d_from_canon = self.renderer.get_inv_warped_2d_grid(self.recon_depth2)
        self.recon_im2 = nn.functional.grid_sample(self.canon_im2, grid_2d_from_canon, mode='bilinear')

        margin = (self.max_depth - self.min_depth) /2
        recon_im_mask = (self.recon_depth2 < self.max_depth+margin).float()  # invalid border pixels have been clamped at max_depth+margin
        recon_im_mask_both = recon_im_mask[:b] * recon_im_mask[b:]  # both original and flip reconstruction
        recon_im_mask_both = recon_im_mask_both.repeat(2,1,1).unsqueeze(1).detach()
        self.recon_im2 = self.recon_im2 * recon_im_mask_both
        
        
        
        ## loss function
        self.loss_D_L2 = nn.ReLU()(1.0 - self.netD_L2(self.lrs)).mean() + nn.ReLU()(1 + self.netD_L2(self.lr_real_2_detach)).mean()
        self.loss_D_H2 = nn.ReLU()(1.0 - self.netD_H2(self.hrs)).mean() + nn.ReLU()(1 + self.netD_H2(self.hr_gen_r_detach)).mean()

        self.gan_loss_G_DSL = -self.netD_L2(self.lr_real_2).mean()
        self.Cyc_loss_DSL = self.L1Loss(self.lr_real_2, self.lrs)
        self.loss_G_DSL = self.alpha * self.Cyc_loss_DSL + self.beta * self.gan_loss_G_DSL

        self.gan_loss_RLS_2 = -self.netD_H2(self.hr_gen_r).mean()
        self.L1_loss_RLS_2 = self.L1Loss(self.ups, self.hr_gen_r)
        self.loss_G_RLS_2 = self.alpha * self.L1_loss_RLS_2 + self.beta * self.gan_loss_RLS_2

        self.loss_3d_Recon = self.L1Loss(self.recon_im1[:b], self.recon_im2[:b])
            
        metrics = {
                    'L1_loss_RLS_2': self.L1_loss_RLS_2,
                    'gan_loss_RLS_2': self.gan_loss_RLS_2,
                    'loss_net_G_RLS_2': self.loss_G_RLS_2, 
                    'Cyc_loss_DSL': self.Cyc_loss_DSL,
                    'gan_loss_G_DSL': self.gan_loss_G_DSL,
                    'loss_net_G_DSL': self.loss_G_DSL, 
                    'loss_net_DL2': self.loss_D_L2, 
                    'loss_net_DH2': self.loss_D_H2,
                    'loss_3d_Recon': self.loss_3d_Recon
                }
        return metrics

    def forward_for_test(self, batch, save_dir, epoch, is_val=False):
            
        self.lrs = batch["lr"].cuda()
        self.hr_gen_r = self.netG_RLS(self.lrs)
        self.img_names = batch["img_name"]
        self.input_im = self.hr_gen_r
        b, c, h, w = self.input_im.shape
        hr_gen = self.hr_gen_r[:b].clamp(-1,1).detach().cpu().numpy() /2+0.5  
        img_names = self.img_names
        utils.save_images(save_dir, hr_gen, img_names, epoch=epoch, suffix='sr_image', sep_folder=True, is_val = is_val)




    def visualize(self, logger, total_iter, max_bs=6):
        assert self.input_im_for3d1.shape == self.input_im_for3d2.shape
        b, c, h, w = self.input_im_for3d1.shape
        b0 = min(max_bs, b)

        #forward
        with torch.no_grad():
            v0 = torch.FloatTensor([-0.1*math.pi/180*60,0,0,0,0,0]).to(self.input_im_for3d1.device).repeat(b0,1)
            canon_im_rotate1 = self.renderer.render_yaw(self.canon_im1[:b0], self.canon_depth1[:b0], v_before=v0, maxr=90).detach().cpu() /2.+0.5  # (B,T,C,H,W)
            canon_normal_rotate1 = self.renderer.render_yaw(self.canon_normal1[:b0].permute(0,3,1,2), self.canon_depth1[:b0], v_before=v0, maxr=90).detach().cpu() /2.+0.5  # (B,T,C,H,W)

        lr_gen = self.lr_gen[:b0].detach().cpu() /2+0.5
        input_im_for3d1 = self.input_im_for3d1[:b0].detach().cpu() /2+0.5
        canon_albedo1 = self.canon_albedo1[:b0].detach().cpu() /2.+0.5
        canon_im1 = self.canon_im1[:b0].detach().cpu() /2.+0.5
        recon_im1 = self.recon_im1[:b0].detach().cpu() /2.+0.5
        canon_depth1 = ((self.canon_depth1[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
        recon_depth1 = ((self.recon_depth1[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
        canon_diffuse_shading1 = self.canon_diffuse_shading1[:b0].detach().cpu()
        canon_normal1 = self.canon_normal1.permute(0,3,1,2)[:b0].detach().cpu() /2+0.5
        recon_normal1 = self.recon_normal1.permute(0,3,1,2)[:b0].detach().cpu() /2+0.5
        canon_im_rotate_grid1 = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b0**0.5))) for img in torch.unbind(canon_im_rotate1, 1)]  # [(C,H,W)]*T
        canon_im_rotate_grid1 = torch.stack(canon_im_rotate_grid1, 0).unsqueeze(0)  # (1,T,C,H,W)
        canon_normal_rotate_grid1 = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b0**0.5))) for img in torch.unbind(canon_normal_rotate1, 1)]  # [(C,H,W)]*T
        canon_normal_rotate_grid1 = torch.stack(canon_normal_rotate_grid1, 0).unsqueeze(0)  # (1,T,C,H,W)

        def log_grid_image(label, im, nrow=int(math.ceil(b0**0.5)), iter=total_iter):
            im_grid = torchvision.utils.make_grid(im, nrow=nrow)
            logger.add_image(label, im_grid, iter)
        
        log_grid_image('Image/lr_gen', lr_gen)
        log_grid_image('Image/hr_gen1', input_im_for3d1)
        log_grid_image('Image/canonical_albedo1', canon_albedo1)
        log_grid_image('Image/canonical_image1', canon_im1)
        log_grid_image('Image/recon_image1', recon_im1)
        log_grid_image('Image/recon_side1', canon_im_rotate1[:,0,:,:,:])
        log_grid_image('Depth/canonical_depth1', canon_depth1)
        log_grid_image('Depth/recon_depth1', recon_depth1)
        log_grid_image('Depth/canonical_diffuse_shading1', canon_diffuse_shading1)
        log_grid_image('Depth/canonical_normal1', canon_normal1)
        log_grid_image('Depth/recon_normal1', recon_normal1)

        
        #backforward
        with torch.no_grad():
            v0 = torch.FloatTensor([-0.1*math.pi/180*60,0,0,0,0,0]).to(self.input_im_for3d2.device).repeat(b0,1)
            canon_im_rotate2 = self.renderer.render_yaw(self.canon_im2[:b0], self.canon_depth2[:b0], v_before=v0, maxr=90).detach().cpu() /2.+0.5  # (B,T,C,H,W)
            canon_normal_rotate2 = self.renderer.render_yaw(self.canon_normal2[:b0].permute(0,3,1,2), self.canon_depth2[:b0], v_before=v0, maxr=90).detach().cpu() /2.+0.5  # (B,T,C,H,W)
        lrs = self.lrs[:b0].detach().cpu() /2+0.5
        hr_gen2 = self.hr_gen_r[:b0].detach().cpu() /2+0.5
        input_im_for3d2 = self.input_im_for3d2[:b0].detach().cpu() /2+0.5
        canon_albedo2 = self.canon_albedo2[:b0].detach().cpu() /2.+0.5
        canon_im2 = self.canon_im2[:b0].detach().cpu() /2.+0.5
        recon_im2 = self.recon_im2[:b0].detach().cpu() /2.+0.5
        canon_depth2 = ((self.canon_depth2[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
        recon_depth2 = ((self.recon_depth2[:b0] -self.min_depth)/(self.max_depth-self.min_depth)).detach().cpu().unsqueeze(1)
        canon_diffuse_shading2 = self.canon_diffuse_shading2[:b0].detach().cpu()
        canon_normal2 = self.canon_normal2.permute(0,3,1,2)[:b0].detach().cpu() /2+0.5
        recon_normal2 = self.recon_normal2.permute(0,3,1,2)[:b0].detach().cpu() /2+0.5
        lr_real_2 = self.lr_real_2[:b0].detach().cpu() /2.+0.5
        canon_im_rotate_grid2 = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b0**0.5))) for img in torch.unbind(canon_im_rotate2, 1)]  # [(C,H,W)]*T
        canon_im_rotate_grid2 = torch.stack(canon_im_rotate_grid2, 0).unsqueeze(0)  # (1,T,C,H,W)
        canon_normal_rotate_grid2 = [torchvision.utils.make_grid(img, nrow=int(math.ceil(b0**0.5))) for img in torch.unbind(canon_normal_rotate2, 1)]  # [(C,H,W)]*T
        canon_normal_rotate_grid2 = torch.stack(canon_normal_rotate_grid2, 0).unsqueeze(0)  # (1,T,C,H,W)
        log_grid_image('Image/lrs', lrs)
        log_grid_image('Image/hr_gen2', hr_gen2)
        log_grid_image('Image/lr_real_2', lr_real_2)
        log_grid_image('Image/hrs', input_im_for3d2)
        log_grid_image('Image/canonical_albedo2', canon_albedo2)
        log_grid_image('Image/canonical_image2', canon_im2)
        log_grid_image('Image/recon_image2', recon_im2)
        log_grid_image('Image/recon_side2', canon_im_rotate2[:,0,:,:,:])

        log_grid_image('Depth/canonical_depth2', canon_depth2)
        log_grid_image('Depth/recon_depth2', recon_depth2)
        log_grid_image('Depth/canonical_diffuse_shading2', canon_diffuse_shading2)
        log_grid_image('Depth/canonical_normal2', canon_normal2)
        log_grid_image('Depth/recon_normal2', recon_normal2)
        
        logger.add_scalar('Loss/Cyc_loss_RLS_1', self.Cyc_loss_RLS_1, total_iter)
        logger.add_scalar('Loss/gan_loss_RLS_1', self.gan_loss_RLS_1, total_iter)
        logger.add_scalar('Loss/gan_loss_DHL', self.gan_loss_DHL, total_iter)
        logger.add_scalar('Loss/L1_loss_DHL', self.L1_loss_DHL, total_iter)
        logger.add_scalar('Loss/loss_net_DL1', self.loss_D_L1, total_iter)
        logger.add_scalar('Loss/loss_net_DH1', self.loss_D_H1, total_iter)
        logger.add_scalar('Loss/L1_loss_RLS_2', self.L1_loss_RLS_2, total_iter)
        logger.add_scalar('Loss/gan_loss_RLS_2', self.gan_loss_RLS_2, total_iter)
        logger.add_scalar('Loss/Cyc_loss_DSL', self.Cyc_loss_DSL, total_iter)
        logger.add_scalar('Loss/gan_loss_G_DSL', self.gan_loss_G_DSL, total_iter)
        logger.add_scalar('Loss/loss_net_DL2', self.loss_D_L2, total_iter)
        logger.add_scalar('Loss/loss_net_DH2', self.loss_D_H2, total_iter)
        logger.add_scalar('Loss/loss_3D_Recon', self.loss_3d_Recon, total_iter)

        logger.add_scalar('Lr/lr_G_RLS', self.lr_G_RLS, total_iter)
        logger.add_scalar('Lr/lr_G_DHL', self.lr_G_DHL, total_iter)
        logger.add_scalar('Lr/lr_G_DLS', self.lr_G_DSL, total_iter)
        logger.add_scalar('Lr/lr_D_H1', self.lr_D_H1, total_iter)
        logger.add_scalar('Lr/lr_D_L1', self.lr_D_L1, total_iter)
        logger.add_scalar('Lr/lr_D_H2', self.lr_D_H2, total_iter)
        logger.add_scalar('Lr/lr_D_L2', self.lr_D_L2, total_iter)
        