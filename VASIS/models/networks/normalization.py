"""
Copyright (C) University of Science and Technology of China.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, args):
        super().__init__()
        config_text = args.config_text
        norm_nc = args.norm_nc
        label_nc = args.semantic_nc
        pad = args.pad

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        if pad == 'zero':
            if args.kernel_norm == 3:
                pw = ks // 2
                self.mlp_shared = nn.Sequential(
                    nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
                    nn.ReLU()
                )
                self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
                self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            elif args.kernel_norm == 1:
                self.mlp_shared = nn.Sequential(
                    nn.Conv2d(label_nc, nhidden, kernel_size=1, padding=0),
                    nn.ReLU()
                )
                self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=1, padding=0)
                self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=1, padding=0)
        elif pad == 'reflect':
            assert args.kernel_norm == 3
            self.mlp_shared = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(label_nc, nhidden, kernel_size=ks),
                nn.ReLU()
            )
            self.mlp_gamma = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(nhidden, norm_nc, kernel_size=ks)
            )
            self.mlp_beta = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(nhidden, norm_nc, kernel_size=ks)
            )
        else:
            raise NotImplementedError('ERROR: please check the padding type')

    def forward(self, x, segmap, input_dist=None):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class VariationAwareSPADE(nn.Module):
    def __init__(self, args,):
        super().__init__()
        # config_text, norm_nc, label_nc
        config_text = args.config_text
        self.norm_nc = args.norm_nc
        self.label_nc = args.semantic_nc
        ctrl_noise = args.noise_nc
        self.pos = args.pos
        self.height = args.height
        self.width = args.width
        pad = args.pad

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(self.norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(self.norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(self.norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # channels for segmentation and noise
        if ctrl_noise == 'zero':
            self.seg_nc = self.norm_nc
            self.noise_nc = 0
        elif ctrl_noise == 'one':
            self.seg_nc = self.norm_nc // 2
            self.noise_nc = 1
        elif ctrl_noise == 'all':
            # self.seg_nc = self.norm_nc
            # self.noise_nc = self.norm_nc
            self.seg_nc = self.norm_nc // 2
            self.noise_nc = self.norm_nc - self.seg_nc
        else:
            raise NotImplementedError('Please check the noise_nc: <zero, one, all>.')
        if self.noise_nc > 0:
            self.gamma_noise_gamma = nn.Parameter(torch.rand(self.label_nc, self.noise_nc))
            self.gamma_noise_beta = nn.Parameter(torch.zeros(self.label_nc, self.noise_nc))
            self.beta_noise_gamma = nn.Parameter(torch.rand(self.label_nc, self.noise_nc))
            self.beta_noise_beta = nn.Parameter(torch.zeros(self.label_nc, self.noise_nc))

        # for position code
        if self.pos == 'no':
            self.gamma_pos = None
            self.beta_pos = None
            self.pos_nc_in = 0
        elif self.pos == 'learn':
            H, W = self.height, self.width
            self.gamma_pos = nn.Parameter(torch.randn(2, H, W))
            self.beta_pos = nn.Parameter(torch.randn(2, H, W))
            self.pos_nc_in = 2
        elif self.pos == 'fix':
            self.pos_nc_in = 2
            H, W = self.height, self.width
            x = torch.tensor(range(H)).view(-1, 1) / H * 2 - 1
            y = torch.tensor(range(W)).view(1, W) / W * 2 - 1
            x = torch.cat([x] * W, dim=1).unsqueeze(0)
            y = torch.cat([y] * H, dim=0).unsqueeze(0)
            self.gamma_pos = torch.cat([x, y], dim=0)
            self.beta_pos = self.gamma_pos
        elif self.pos == 'reflect':
            self.pos_nc_in = 2
        elif self.pos == 'learn_relative':
            self.pos_nc_in = 4
            H, W = self.height, self.width
            self.gamma_pos = nn.Parameter(torch.randn(2, H, W))
            self.beta_pos = nn.Parameter(torch.randn(2, H, W))
        else:
            raise NotImplementedError('ERROR: please check the pos type: learn, fix, no')
        if self.pos_nc_in > 0:
            if args.pos_nc == 'one':
                self.conv_pos_gamma = nn.Conv2d(self.pos_nc_in, 1, 1)
                self.conv_pos_beta = nn.Conv2d(self.pos_nc_in, 1, 1)
            elif args.pos_nc == 'all':
                self.conv_pos_gamma = nn.Conv2d(self.pos_nc_in, self.seg_nc, 1)
                self.conv_pos_beta = nn.Conv2d(self.pos_nc_in, self.seg_nc, 1)
            nn.init.zeros_(self.conv_pos_gamma.weight)
            nn.init.zeros_(self.conv_pos_gamma.bias)
            nn.init.zeros_(self.conv_pos_beta.weight)
            nn.init.zeros_(self.conv_pos_beta.bias)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        if pad == 'zero':
                pw = ks // 2
                self.mlp_shared = nn.Sequential(
                    nn.Conv2d(self.label_nc, nhidden, kernel_size=ks, padding=pw),
                    nn.ReLU()
                )
                self.mlp_gamma = nn.Conv2d(nhidden, self.seg_nc, kernel_size=ks, padding=pw)
                self.mlp_beta = nn.Conv2d(nhidden, self.seg_nc, kernel_size=ks, padding=pw)
        elif pad == 'reflect':
            self.mlp_shared = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(self.label_nc, nhidden, kernel_size=ks),
                nn.ReLU()
            )
            self.mlp_gamma = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(nhidden, self.seg_nc, kernel_size=ks),
            )
            self.mlp_beta = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(nhidden, self.seg_nc, kernel_size=ks)
            )
        else:
            raise NotImplementedError('ERROR: please check the padding type')

    def affine_noise(self, mask):
        arg_mask = torch.argmax(mask, 1).long()
        gamma_noise_gamma = F.embedding(arg_mask, self.gamma_noise_gamma).permute(0, 3, 1, 2)
        gamma_noise_beta = F.embedding(arg_mask, self.gamma_noise_beta).permute(0, 3, 1, 2)
        beta_noise_gamma = F.embedding(arg_mask, self.beta_noise_gamma).permute(0, 3, 1, 2)
        beta_noise_beta = F.embedding(arg_mask, self.beta_noise_beta).permute(0, 3, 1, 2)
        B, _, H, W = mask.size()
        noise_1 = torch.rand((B, self.norm_nc - self.seg_nc, H, W), device=mask.device)
        noise_2 = torch.rand((B, self.norm_nc - self.seg_nc, H, W), device=mask.device)
        gamma_noise = noise_1 * gamma_noise_gamma + gamma_noise_beta
        beta_noise = noise_2 * beta_noise_gamma + beta_noise_beta
        return gamma_noise, beta_noise

    def forward(self, x, segmap, input_dist=None):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # position code
        if self.pos_nc_in == 2:
            if self.pos in ['learn', 'fix']:
                gamma = gamma * (1 + self.conv_pos_gamma(self.gamma_pos.unsqueeze(0).to(x.device)))
                beta = beta * (1 + self.conv_pos_beta(self.beta_pos.unsqueeze(0).to(x.device)))
            else: # 'relative'
                input_dist = F.interpolate(input_dist, size=x.size()[2:], mode='nearest')
                gamma = gamma * (1 + self.conv_pos_gamma(input_dist))
                beta = beta * (1 + self.conv_pos_beta(input_dist))
        elif self.pos_nc_in == 4:
            input_dist = F.interpolate(input_dist, size=x.size()[2:], mode='nearest')
            gamma_pos = self.gamma_pos.unsqueeze(0).expand_as(input_dist)
            beta_pos = self.beta_pos.unsqueeze(0).expand_as(input_dist)
            gamma_pos = torch.cat([gamma_pos.to(x.device), input_dist], dim=1)
            beta_pos = torch.cat([beta_pos.to(x.device), input_dist], dim=1)
            gamma = gamma * (1 + self.conv_pos_gamma(gamma_pos))
            beta = beta * (1 + self.conv_pos_beta(beta_pos))
        # semantic noise
        if self.noise_nc > 0:
            gamma_noise, beta_noise = self.affine_noise(segmap)
            gamma = torch.cat([gamma, gamma_noise], dim=1)
            beta = torch.cat([beta, beta_noise], dim=1)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class VariationAwareCLADE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.no_instance = args.no_instance
        assert args.config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', args.config_text)
        param_free_norm_type = str(parsed.group(1))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(args.norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(args.norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(args.norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        self.class_specified_affine = VariationAffineCLADE(args)

        if not args.no_instance:
            self.inst_conv = nn.Conv2d(1, 1, kernel_size=1, padding=0)

    def forward(self, x, segmap, input_dist=None):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. scale the segmentation mask
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        if not self.no_instance:
            inst_map = torch.unsqueeze(segmap[:, -1, :, :],1)
            segmap = segmap[:, : -1, :, :]
        # Part 3. class affine with noise
        out = self.class_specified_affine(normalized, segmap, input_dist)
        if not self.no_instance:
            inst_feat = self.inst_conv(inst_map)
            out = torch.cat((out, inst_feat), dim=1)
        return out

class VariationAffineCLADE(nn.Module):
    """
    Functions:
        Semantic noise.
            Concatenate with semantic label: one channel or all channels
            Plus with semantic label with learnable weights
        Position
            Fixed with Conv or learnable without Conv
            Fixed after learning
    """
    def __init__(self, args):
        super(VariationAffineCLADE, self).__init__()
        self.args = args
        self.label_nc = args.label_nc_
        self.feature_nc = args.norm_nc
        self.ctrl_noise = args.noise_nc
        self.pos = args.pos
        self.height = args.height
        self.width = args.width

        self.set_nc()
        self.init_net()

    def set_nc(self):
        # set the numbers of channel
        if self.ctrl_noise == 'zero':
            self.seg_nc = self.feature_nc
            self.noise_nc = 0
        elif self.ctrl_noise == 'one':
            self.seg_nc = self.feature_nc // 2
            self.noise_nc = 1
        elif self.ctrl_noise == 'all':
            self.seg_nc = self.feature_nc // 2
            self.noise_nc = self.feature_nc - self.seg_nc
        else:
            raise NotImplementedError('Please check the noise_nc: <zero, one, all>.')

    def init_net(self):
        # first part: between-class variation
        self.gamma_seg = nn.Parameter(torch.Tensor(self.label_nc, self.seg_nc))
        self.beta_seg = nn.Parameter(torch.Tensor(self.label_nc, self.seg_nc))
        nn.init.uniform_(self.gamma_seg)
        nn.init.zeros_(self.beta_seg)

        # second part: in-class variation for semantic noise
        if self.noise_nc > 0:
            self.gamma_noise_gamma = nn.Parameter(torch.Tensor(self.label_nc, self.noise_nc))
            self.gamma_noise_beta = nn.Parameter(torch.Tensor(self.label_nc, self.noise_nc))
            self.beta_noise_gamma = nn.Parameter(torch.Tensor(self.label_nc, self.noise_nc))
            self.beta_noise_beta = nn.Parameter(torch.Tensor(self.label_nc, self.noise_nc))
            nn.init.uniform_(self.gamma_noise_gamma)
            nn.init.zeros_(self.gamma_noise_beta)
            nn.init.uniform_(self.beta_noise_gamma)
            nn.init.zeros_(self.beta_noise_beta)

        # for position code
        if self.pos == 'no':
            self.gamma_pos = None
            self.beta_pos = None
            self.pos_nc_in = 0
        elif self.pos == 'learn':
            H, W = self.height, self.width
            self.gamma_pos = nn.Parameter(torch.randn(2, H, W))
            self.beta_pos = nn.Parameter(torch.randn(2, H, W))
            self.pos_nc_in = 2
        elif self.pos == 'fix':
            self.pos_nc_in = 2
            H, W = self.height, self.width
            x = torch.tensor(range(H)).view(-1, 1) / H * 2 - 1
            y = torch.tensor(range(W)).view(1, W) / W * 2 - 1
            x = torch.cat([x] * W, dim=1).unsqueeze(0)
            y = torch.cat([y] * H, dim=0).unsqueeze(0)
            self.gamma_pos = torch.cat([x, y], dim=0)
            self.beta_pos = self.gamma_pos
        elif self.pos == 'relative':
            self.pos_nc_in = 2
        elif self.pos == 'learn_relative':
            self.pos_nc_in = 4
            H, W = self.height, self.width
            self.gamma_pos = nn.Parameter(torch.randn(2, H, W))
            self.beta_pos = nn.Parameter(torch.randn(2, H, W))
        else:
            raise NotImplementedError('ERROR: please check the pos type: learn, fix, no')
        if self.pos_nc_in > 0:
            if self.args.pos_nc == 'one':
                self.conv_pos_gamma = nn.Conv2d(self.pos_nc_in, 1, 1)
                self.conv_pos_beta = nn.Conv2d(self.pos_nc_in, 1, 1)
            elif self.args.pos_nc == 'all':
                self.conv_pos_gamma = nn.Conv2d(self.pos_nc_in, self.seg_nc, 1)
                self.conv_pos_beta = nn.Conv2d(self.pos_nc_in, self.seg_nc, 1)
            nn.init.zeros_(self.conv_pos_gamma.weight)
            nn.init.zeros_(self.conv_pos_gamma.bias)
            nn.init.zeros_(self.conv_pos_beta.weight)
            nn.init.zeros_(self.conv_pos_beta.bias)

    def affine_seg(self, mask):
        arg_mask = torch.argmax(mask, 1).long() # [n, h, w]
        gamma_seg = F.embedding(arg_mask, self.gamma_seg).permute(0, 3, 1, 2) # [n, c, h, w]
        beta_seg = F.embedding(arg_mask, self.beta_seg).permute(0, 3, 1, 2) # [n, c, h, w]
        return gamma_seg, beta_seg

    def affine_noise(self, mask):
        arg_mask = torch.argmax(mask, 1).long()
        gamma_noise_gamma = F.embedding(arg_mask, self.gamma_noise_gamma).permute(0, 3, 1, 2)
        gamma_noise_beta = F.embedding(arg_mask, self.gamma_noise_beta).permute(0, 3, 1, 2)
        beta_noise_gamma = F.embedding(arg_mask, self.beta_noise_gamma).permute(0, 3, 1, 2)
        beta_noise_beta = F.embedding(arg_mask, self.beta_noise_beta).permute(0, 3, 1, 2)
        B, _, H, W = mask.size()
        noise_1 = torch.randn((B, self.feature_nc - self.seg_nc, H, W), device=mask.device)
        noise_2 = torch.randn((B, self.feature_nc - self.seg_nc, H, W), device=mask.device)
        gamma_noise = noise_1 * gamma_noise_gamma + gamma_noise_beta
        beta_noise = noise_2 * beta_noise_gamma + beta_noise_beta
        return gamma_noise, beta_noise

    def forward(self, input, mask, input_dist=None):
        # input: only include semantic segmentation, no instance map
        gamma_seg, beta_seg = self.affine_seg(mask)

        if self.pos_nc_in == 2:
            gamma_seg = gamma_seg * (1+ self.conv_pos_gamma(self.gamma_pos.unsqueeze(0).to(input.device)))
            beta_seg = beta_seg * (1 + self.conv_pos_beta(self.beta_pos.unsqueeze(0).to(input.device)))
        elif self.pos_nc_in == 4:
            input_dist = F.interpolate(input_dist, size=input.size()[2:], mode='nearest')
            gamma_pos = self.gamma_pos.unsqueeze(0).expand_as(input_dist)
            beta_pos = self.beta_pos.unsqueeze(0).expand_as(input_dist)
            gamma_pos = torch.cat([gamma_pos.to(input.device), input_dist], dim=1)
            beta_pos = torch.cat([beta_pos.to(input.device), input_dist], dim=1)
            gamma_seg = gamma_seg * (1 + self.conv_pos_gamma(gamma_pos))
            beta_seg = beta_seg * (1 + self.conv_pos_beta(beta_pos))

        # semantic noise
        if self.noise_nc > 0:
            gamma_noise, beta_noise = self.affine_noise(mask)
            gamma_seg = torch.cat([gamma_seg, gamma_noise], dim=1)
            beta_seg = torch.cat([beta_seg, beta_noise], dim=1)

        x = input * gamma_seg + beta_seg
        return x


class SPADELight(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.no_instance = args.no_instance
        self.add_dist = args.add_dist
        assert args.config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', args.config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(args.norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(args.norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(args.norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        self.class_specified_affine = ClassAffine(args)

        if not args.no_instance:
            self.inst_conv = nn.Conv2d(1, 1, kernel_size=1, padding=0)

    def forward(self, x, segmap, input_dist=None):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. scale the segmentation mask
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        if not self.no_instance:
            inst_map = torch.unsqueeze(segmap[:,-1,:,:],1)
            segmap = segmap[:,:-1,:,:]

        # Part 3. class affine with noise
        out = self.class_specified_affine(normalized, segmap, input_dist)

        if not self.no_instance:
            inst_feat = self.inst_conv(inst_map)
            out = torch.cat((out, inst_feat), dim=1)

        return out

class ClassAffine(nn.Module):
    def __init__(self, args):
        super(ClassAffine, self).__init__()
        self.add_dist = args.add_dist
        self.affine_nc = args.norm_nc
        self.label_nc = args.label_nc_
        self.weight = nn.Parameter(torch.Tensor(self.label_nc, self.affine_nc))
        self.bias = nn.Parameter(torch.Tensor(self.label_nc, self.affine_nc))
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)
        if args.add_dist:
            self.dist_conv_w = nn.Conv2d(2, 1, kernel_size=1, padding=0)
            nn.init.zeros_(self.dist_conv_w.weight)
            nn.init.zeros_(self.dist_conv_w.bias)
            self.dist_conv_b = nn.Conv2d(2, 1, kernel_size=1, padding=0)
            nn.init.zeros_(self.dist_conv_b.weight)
            nn.init.zeros_(self.dist_conv_b.bias)

    def affine_gather(self, input, mask):
        n, c, h, w = input.shape
        # process mask
        mask2 = torch.argmax(mask, 1) # [n, h, w]
        mask2 = mask2.view(n, h*w).long() # [n, hw]
        mask2 = mask2.unsqueeze(1).expand(n, self.affine_nc, h*w) # [n, nc, hw]
        # process weights
        weight2 = torch.unsqueeze(self.weight, 2).expand(self.label_nc, self.affine_nc, h*w) # [cls, nc, hw]
        bias2 = torch.unsqueeze(self.bias, 2).expand(self.label_nc, self.affine_nc, h*w) # [cls, nc, hw]
        # torch gather function
        class_weight = torch.gather(weight2, 0, mask2).view(n, self.affine_nc, h, w)
        class_bias = torch.gather(bias2, 0, mask2).view(n, self.affine_nc, h, w)
        return class_weight, class_bias

    def affine_einsum(self, mask):
        class_weight = torch.einsum('ic,nihw->nchw', self.weight, mask)
        class_bias = torch.einsum('ic,nihw->nchw', self.bias, mask)
        return class_weight, class_bias

    def affine_embed(self, mask):
        arg_mask = torch.argmax(mask, 1).long() # [n, h, w]
        class_weight = F.embedding(arg_mask, self.weight).permute(0, 3, 1, 2) # [n, c, h, w]
        class_bias = F.embedding(arg_mask, self.bias).permute(0, 3, 1, 2) # [n, c, h, w]
        return class_weight, class_bias

    def forward(self, input, mask, input_dist=None):
        # class_weight, class_bias = self.affine_gather(input, mask)
        # class_weight, class_bias = self.affine_einsum(mask)
        class_weight, class_bias = self.affine_embed(mask)
        if self.add_dist:
            input_dist = F.interpolate(input_dist, size=input.size()[2:], mode='nearest')
            class_weight = class_weight * (1 + self.dist_conv_w(input_dist))
            class_bias = class_bias * (1 + self.dist_conv_b(input_dist))
        x = input * class_weight + class_bias
        return x
