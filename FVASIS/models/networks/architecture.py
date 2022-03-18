"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE, SPADELight, VariationAwareCLADE, VariationAwareSPADE
from math import pi

# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.opt = opt

        # create conv layers
        add_channels = 1 if ('clade' in opt.norm_mode and not opt.no_instance) else 0
        if opt.pad == 'zero':
            if opt.kernel == 3:
                self.conv_0 = nn.Conv2d(fin+add_channels, fmiddle, kernel_size=3, padding=1)
                self.conv_1 = nn.Conv2d(fmiddle+add_channels, fout, kernel_size=3, padding=1)
            elif opt.kernel == 1:
                self.conv_0 = nn.Conv2d(fin+add_channels, fmiddle, kernel_size=1, padding=0)
                self.conv_1 = nn.Conv2d(fmiddle+add_channels, fout, kernel_size=1, padding=0)
        elif opt.pad == 'reflect':
            assert opt.kernel == 3
            self.pad = nn.ReflectionPad2d(1)
            self.conv_0 = nn.Conv2d(fin+add_channels, fmiddle, kernel_size=3)
            self.conv_1 = nn.Conv2d(fmiddle+add_channels, fout, kernel_size=3)
        else:
            raise NotImplementedError('Error: please check the padding type.')

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin+add_channels, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        opt.config_text = opt.norm_G.replace('spectral', '')
        if opt.norm_mode == 'spade':
            opt.norm_nc = fin
            self.norm_0 = SPADE(opt)
            opt.norm_nc = fmiddle
            self.norm_1 = SPADE(opt)
            if self.learned_shortcut:
                opt.norm_nc = fin
                self.norm_s = SPADE(opt)
        elif opt.norm_mode == 'spade_variation':
            opt.norm_nc = fin
            self.norm_0 = VariationAwareSPADE(opt)
            opt.norm_nc = fmiddle
            self.norm_1 = VariationAwareSPADE(opt)
            if self.learned_shortcut:
                opt.norm_nc = fin
                self.norm_s = VariationAwareSPADE(opt)
        elif opt.norm_mode == 'clade':
            opt.label_nc_ = opt.label_nc + (1 if opt.contain_dontcare_label else 0)
            opt.norm_nc = fin
            self.norm_0 = SPADELight(opt)
            opt.norm_nc = fmiddle
            self.norm_1 = SPADELight(opt)
            if self.learned_shortcut:
                opt.norm_nc = fin
                self.norm_s = SPADELight(opt)
        elif opt.norm_mode == 'clade_variation':
            opt.label_nc_ = opt.label_nc + (1 if opt.contain_dontcare_label else 0)
            opt.norm_nc = fin
            self.norm_0 = VariationAwareCLADE(opt)
            opt.norm_nc = fmiddle
            self.norm_1 = VariationAwareCLADE(opt)
            if self.learned_shortcut:
                opt.norm_nc = fin
                self.norm_s = VariationAwareCLADE(opt)

        else:
            raise ValueError('%s is not a defined normalization method' % opt.norm_mode)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, input_dist=None):
        x_s = self.shortcut(x, seg, input_dist)

        if self.opt.pad == 'zero':
            dx = self.conv_0(self.actvn(self.norm_0(x, seg, input_dist)))
            dx = self.conv_1(self.actvn(self.norm_1(dx, seg, input_dist)))
        elif self.opt.pad == 'reflect':
            dx = self.conv_0(self.pad(self.actvn(self.norm_0(x, seg, input_dist))))
            dx = self.conv_1(self.pad(self.actvn(self.norm_1(dx, seg, input_dist))))
        else:
            dx = None
            raise NotImplementedError('Error: please check the padding type')

        out = x_s + dx

        return out

    def shortcut(self, x, seg, input_dist=None):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg, input_dist))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class MLPBlock(nn.Module):
    def __init__(self, opt, fc_in, fc_out, height, width):
        super(MLPBlock, self).__init__()
        self.opt = opt
        self.fc_in = fc_in
        self.fc_out = fc_out
        self.height, self.width = height, width
        label_nc = opt.label_nc
        # fc_in = fc_in + opt.nc_position
        self.conv = nn.Conv2d(self.opt.nc_position, 1, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.weight = nn.Parameter(torch.randn(label_nc, fc_in * fc_out))
        self.bias = nn.Parameter(torch.zeros(label_nc, fc_out))
        self.init_pos()

    def init_pos(self):
        H, W = self.height, self.width

        # learned position
        self.pl = nn.Parameter(torch.randn(2, H, W))

        # absolute position
        x = torch.tensor(range(H)).view(-1, 1) / H * 2 - 1
        y = torch.tensor(range(W)).view(1, W) / W * 2 - 1
        x = torch.cat([x] * W, dim=1).unsqueeze(0)
        y = torch.cat([y] * H, dim=0).unsqueeze(0)
        self.pa = torch.cat([x, y], dim=0)

        #
        h, w = H, W
        f0 = 3
        f = f0
        while f > 1:
            x = torch.arange(0, w).float()
            y = torch.arange(0, h).float()
            xcos = torch.cos((2 * pi * torch.remainder(x, f).float() / f).float())
            xsin = torch.sin((2 * pi * torch.remainder(x, f).float() / f).float())
            ycos = torch.cos((2 * pi * torch.remainder(y, f).float() / f).float())
            ysin = torch.sin((2 * pi * torch.remainder(y, f).float() / f).float())
            xcos = xcos.view(1, 1, w).repeat(1, h, 1)
            xsin = xsin.view(1, 1, w).repeat(1, h, 1)
            ycos = ycos.view(1, h, 1).repeat(1, 1, w)
            ysin = ysin.view(1, h, 1).repeat(1, 1, w)
            coords_cur = torch.cat([xcos, xsin, ycos, ysin], 0)
            if f < f0:
                coords = torch.cat([coords, coords_cur], 1)
            else:
                coords = coords_cur
            f = f//2
        self.coords = coords

    def affine_layout(self, layout):
        arg_layout = torch.argmax(layout, 1).long() # [b, h, w]
        weight = F.embedding(arg_layout, self.weight, scale_grad_by_freq=True) # [b, h, w, c]
        bias = F.embedding(arg_layout, self.bias, scale_grad_by_freq=True)
        b, h, w, c = weight.size()
        weight = weight.contiguous().view(b, h, w, self.fc_in, self.fc_out)
        bias = bias.contiguous().view(b, h, w, 1, self.fc_out)
        return weight, bias

    def forward(self, x, layout, pr=None):
        """
        x: feature map <B, C, h, w>
        layout: semantic layout, <B, N, H, W>
        pr: relative position <B, 2, H, W>
        """
        if self.opt.check_flop and pr is None:
            pr = torch.randn((x.size(0), 2, self.height, self.width), device=x.get_device())
        B, C, h, w = x.size()
        device = x.get_device()
        pa = self.pa.unsqueeze(0).expand(B, 2, h, w).to(device)
        pl = self.pl.unsqueeze(0).expand(B, 2, h, w).to(device)
        coords = self.coords.unsqueeze(0).expand(B, 4, h, w).to(device)
        pr = F.interpolate(pr, size=[h, w], mode='nearest')
        # noise = torch.randn((B, 20, h, w), device=device)
        pos = torch.cat([pa, pl, pr, coords], dim=1)
        x = x + self.conv(pos)
        x = x.contiguous().permute(0, 2, 3, 1).view(B, h, w, 1, C)
        layout = F.interpolate(layout, size=[h, w], mode='nearest')
        weight, bias = self.affine_layout(layout)
        x = torch.matmul(x, weight) + bias
        x = x.squeeze(3).permute(0, 3, 1, 2)
        x = F.instance_norm(x)
        x = F.leaky_relu(x)
        # x = self.conv(x)
        # x = F.instance_norm(x)
        # x = F.leaky_relu(x)
        return x