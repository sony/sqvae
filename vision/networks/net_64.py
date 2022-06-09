import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from networks.util import ResBlock


class EncoderVqResnet64(nn.Module):
    def __init__(self, dim_z, cfgs, flg_bn=True, flg_var_q=False):
        super(EncoderVqResnet64, self).__init__()
        self.flg_variance = flg_var_q
        # Convolution layers
        layers_conv = []
        layers_conv.append(nn.Sequential(nn.Conv2d(3, dim_z // 2, 4, stride=2, padding=1)))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z // 2))
        layers_conv.append(nn.ReLU())
        layers_conv.append(nn.Conv2d(dim_z // 2, dim_z, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU())
        layers_conv.append(nn.Conv2d(dim_z, dim_z, 3, stride=1, padding=1))
        self.conv = nn.Sequential(*layers_conv)
        # Resblocks
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        self.res_m = ResBlock(dim_z)
        if self.flg_variance:
            self.res_v = ResBlock(dim_z)

    def forward(self, x):
        out_conv = self.conv(x)
        out_res = self.res(out_conv)
        mu = self.res_m(out_res)
        if self.flg_variance:
            log_var = self.res_v(out_res)
            return mu, log_var
        else:
            return mu
        

class DecoderVqResnet64(nn.Module):
    def __init__(self, dim_z, cfgs, flg_bn=True):
        super(DecoderVqResnet64, self).__init__()
        # Resblocks
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        # Convolution layers
        layers_convt = []
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z, 3, stride=1, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z))
        layers_convt.append(nn.ReLU())
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 2))
        layers_convt.append(nn.ReLU())
        layers_convt.append(nn.ConvTranspose2d(dim_z // 2, 3, 4, stride=2, padding=1))
        layers_convt.append(nn.Sigmoid())
        self.convt = nn.Sequential(*layers_convt)
        
    def forward(self, z):
        out_res = self.res(z)
        out = self.convt(out_res)

        return out


class EncoderVqResnet64Label(nn.Module):
    def __init__(self, dim_z, cfgs, flg_bn=True, flg_var_q=False):
        super(EncoderVqResnet64Label, self).__init__()
        self.n_class = int(np.ceil(cfgs.num_class / 2) * 2)
        self.flg_variance = flg_var_q
        # Convolution layers
        layers_conv = []
        layers_conv.append(nn.Conv2d(self.n_class, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z // 2))
        layers_conv.append(nn.ReLU())
        layers_conv.append(nn.Conv2d(dim_z // 2, dim_z, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU())
        layers_conv.append(nn.Conv2d(dim_z, dim_z, 4, stride=2, padding=1))
        self.conv = nn.Sequential(*layers_conv)
        # Resblocks
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        self.res_m = ResBlock(dim_z)
        if self.flg_variance:
            self.res_v = ResBlock(dim_z)


    def forward(self, x):
        x_one_hot = (
            F.one_hot(x.to(torch.int).long(), num_classes = self.n_class)
            .type_as(x)
        ).permute(0, 3, 1, 2).contiguous()
        out_conv = self.conv(x_one_hot)
        out_res = self.res(out_conv)
        mu = self.res_m(out_res)
        if self.flg_variance:
            log_var = self.res_v(out_res)
            return mu, log_var
        else:
            return mu


class DecoderVqResnet64Label(nn.Module):
    def __init__(self, dim_z, cfgs, act="linear", flg_bn=True):
        super(DecoderVqResnet64Label, self).__init__()
        self.n_class = int(np.ceil(cfgs.num_class / 2) * 2)
        # Resblocks
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb-1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
		# Convolution layers
        layers_convt = []
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z))
        layers_convt.append(nn.ReLU())
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 2))
        layers_convt.append(nn.ReLU())
        layers_convt.append(nn.ConvTranspose2d(dim_z // 2, self.n_class, 4, stride=2, padding=1))
        if act == "sigmoid":
            layers_convt.append(nn.Sigmoid())
        elif act == "exp":
            layers_convt.append(nn.Softplus())
        elif act == "tanh":
            layers_convt.append(nn.Tanh())
        self.convt = nn.Sequential(*layers_convt)
    
    def forward(self, z):
        out_res = self.res(z)
        out = self.convt(out_res)

        return out
