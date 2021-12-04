"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
import matplotlib.pyplot as plt
import numpy as np
from models.networks.cal_dist import cal_connectedComponents as compute_dist


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            if opt.pad == 'zero':
                self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)
            elif opt.pad == 'reflect':
                self.fc = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=0)
                )

        opt.height, opt.width = self.sh, self.sw
        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        opt.height, opt.width = self.sh * 2, self.sw * 2
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        if opt.num_upsampling_layers == 'normal':
            opt.height, opt.width = self.sh * 2, self.sw * 2
            self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
            opt.height, opt.width = self.sh * 4, self.sw * 4
            self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
            opt.height, opt.width = self.sh * 8, self.sw * 8
            self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
            opt.height, opt.width = self.sh * 16, self.sw * 16
            self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
            opt.height, opt.width = self.sh * 32, self.sw * 32
            self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)
        elif opt.num_upsampling_layers == 'more':
            opt.height, opt.width = self.sh * 4, self.sw * 4
            self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
            opt.height, opt.width = self.sh * 8, self.sw * 8
            self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
            opt.height, opt.width = self.sh * 16, self.sw * 16
            self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
            opt.height, opt.width = self.sh * 32, self.sw * 32
            self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
            opt.height, opt.width = self.sh * 64, self.sw * 64
            self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        if opt.pad == 'zero':
            self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        elif opt.pad == 'reflect':
            self.conv_img = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(final_nc, 3, 3, padding=0)
            )

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None, input_dist=None):
        seg = input

        # vis
        ctrl_label = 23 # sky
        ctrl_label_1 = 23 # sky
        ctrl_label_2 = 7 # road
        vis_size = [256, 256]
        save_name = f'spade_original_vis{self.opt.vis}'
        save_mean = []
        save_std = []

        feat = []
        if self.opt.vis:
            B, N, H, W = seg.size()
            if self.opt.vis == 1:
                seg[0] = torch.zeros(N, H, W, device=seg.device)
                seg[0, ctrl_label] = torch.ones(H, W, device=seg.device)
            elif self.opt.vis == 2:
                seg[0] = torch.zeros(N, H, W, device=seg.device)
                seg[0, ctrl_label_1, :H//2] = torch.ones(H//2, W, device=seg.device)
                seg[0, ctrl_label_2, H//2:] = torch.ones(H//2, W, device=seg.device)

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        if self.opt.vis:
            img = F.interpolate(x, size=vis_size, mode='nearest')
            feat.append(img[0, 0].detach().cpu().numpy())
            save_mean.append(torch.mean(img[0, 0]).detach().cpu().numpy())
            save_std.append(torch.std(img[0, 0]).detach().cpu().numpy())

        x = self.head_0(x, seg, input_dist)

        if self.opt.vis:
            img = F.interpolate(x, size=vis_size, mode='nearest')
            feat.append(img[0, 0].detach().cpu().numpy())
            save_mean.append(torch.mean(img[0, 0]).detach().cpu().numpy())
            save_std.append(torch.std(img[0, 0]).detach().cpu().numpy())

        x = self.up(x)
        x = self.G_middle_0(x, seg, input_dist)

        if self.opt.vis:
            img = F.interpolate(x, size=vis_size, mode='nearest')
            feat.append(img[0, 0].detach().cpu().numpy())
            save_mean.append(torch.mean(img[0, 0]).detach().cpu().numpy())
            save_std.append(torch.std(img[0, 0]).detach().cpu().numpy())

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg, input_dist)

        if self.opt.vis:
            img = F.interpolate(x, size=vis_size, mode='nearest')
            feat.append(img[0, 0].detach().cpu().numpy())
            save_mean.append(torch.mean(img[0, 0]).detach().cpu().numpy())
            save_std.append(torch.std(img[0, 0]).detach().cpu().numpy())

        x = self.up(x)
        x = self.up_0(x, seg, input_dist)

        if self.opt.vis:
            img = F.interpolate(x, size=vis_size, mode='nearest')
            feat.append(img[0, 0].detach().cpu().numpy())
            save_mean.append(torch.mean(img[0, 0]).detach().cpu().numpy())
            save_std.append(torch.std(img[0, 0]).detach().cpu().numpy())

        x = self.up(x)
        x = self.up_1(x, seg, input_dist)

        if self.opt.vis:
            img = F.interpolate(x, size=vis_size, mode='nearest')
            feat.append(img[0, 0].detach().cpu().numpy())
            save_mean.append(torch.mean(img[0, 0]).detach().cpu().numpy())
            save_std.append(torch.std(img[0, 0]).detach().cpu().numpy())

        x = self.up(x)
        x = self.up_2(x, seg, input_dist)

        if self.opt.vis:
            img = F.interpolate(x, size=vis_size, mode='nearest')
            feat.append(img[0, 0].detach().cpu().numpy())
            save_mean.append(torch.mean(img[0, 0]).detach().cpu().numpy())
            save_std.append(torch.std(img[0, 0]).detach().cpu().numpy())

        x = self.up(x)
        x = self.up_3(x, seg, input_dist)

        if self.opt.vis:
            img = F.interpolate(x, size=vis_size, mode='nearest')
            feat.append(img[0, 0].detach().cpu().numpy())
            save_mean.append(torch.mean(img[0, 0]).detach().cpu().numpy())
            save_std.append(torch.std(img[0, 0]).detach().cpu().numpy())

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, input_dist)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        if self.opt.vis:
            img = F.interpolate(x, size=vis_size, mode='nearest')
            feat.append(img[0].detach().permute(1, 2, 0).cpu().numpy())
            save_mean.append(torch.mean(img[0, 0]).detach().cpu().numpy())
            save_std.append(torch.std(img[0, 0]).detach().cpu().numpy())

        if self.opt.vis:
            for i, img in enumerate(feat):
                img = (img.clip(-1, 1) + 1) / 2
                # plt.imsave(f'img_{i}.jpg', img, cmap='gray')
                import os
                os.makedirs(f'vis/{save_name}', exist_ok=True)
                plt.imsave(f'vis/{save_name}/img_{i}.jpg', img)
            np.savetxt(f'vis/{save_name}/mean.txt', save_mean, fmt='%.4f', delimiter=',')
            np.savetxt(f'vis/{save_name}/std.txt', save_std, fmt='%.4f', delimiter=',')

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)
