import torch.nn.utils.spectral_norm as spectral_norm
from models.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn as nn
import torch.nn.functional as F
import torch


class SPADE(nn.Module):
    def __init__(self, opt, norm_nc, label_nc, height, width):
        super().__init__()
        self.first_norm = get_norm_layer(opt, norm_nc)
        ks = opt.spade_ks
        nhidden = 128
        pw = ks // 2
        self.norm_nc = norm_nc
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        # original
        # self.mlp_gamma = nn.Conv2d(nhidden, norm_nc // 2, kernel_size=ks, padding=pw)
        # self.mlp_beta = nn.Conv2d(nhidden, norm_nc // 2, kernel_size=ks, padding=pw)
        self.mlp_gamma = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nhidden, norm_nc // 2, kernel_size=3),
        )
        self.mlp_beta = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nhidden, norm_nc // 2, kernel_size=3),
        )

        # position code
        H, W = height, width
        self.gamma_pos_learn = nn.Parameter(torch.randn(2, H, W))
        self.beta_pos_learn = nn.Parameter(torch.randn(2, H, W))
        self.conv_pos_gamma = nn.Conv2d(2, norm_nc // 2, 1)
        self.conv_pos_beta = nn.Conv2d(2, norm_nc // 2, 1)
        nn.init.zeros_(self.conv_pos_gamma.weight)
        nn.init.zeros_(self.conv_pos_gamma.bias)
        nn.init.zeros_(self.conv_pos_beta.weight)
        nn.init.zeros_(self.conv_pos_beta.bias)

        # semantic noise
        self.noise_nc = norm_nc - norm_nc // 2
        self.label_nc = label_nc
        self.gamma_noise_gamma = nn.Parameter(torch.rand(self.label_nc, self.noise_nc))
        self.beta_noise_gamma = nn.Parameter(torch.rand(self.label_nc, self.noise_nc))
        self.gamma_noise_beta = nn.Parameter(torch.zeros(self.label_nc, self.noise_nc))
        self.beta_noise_beta = nn.Parameter(torch.zeros(self.label_nc, self.noise_nc))

    def affine_noise(self, mask):
        arg_mask = torch.argmax(mask, 1).long()
        gamma_noise_gamma = F.embedding(arg_mask, self.gamma_noise_gamma).permute(0, 3, 1, 2)
        beta_noise_gamma = F.embedding(arg_mask, self.beta_noise_gamma).permute(0, 3, 1, 2)
        gamma_noise_beta = F.embedding(arg_mask, self.gamma_noise_beta).permute(0, 3, 1, 2)
        beta_noise_beta = F.embedding(arg_mask, self.beta_noise_beta).permute(0, 3, 1, 2)

        B, _, H, W = mask.size()
        noise_1 = torch.rand((B, self.norm_nc - self.norm_nc // 2, H, W), device=mask.device)
        noise_2 = torch.rand((B, self.norm_nc - self.norm_nc // 2, H, W), device=mask.device)

        gamma_noise = noise_1 * gamma_noise_gamma + gamma_noise_beta
        beta_noise = noise_2 * beta_noise_gamma + beta_noise_beta

        return gamma_noise, beta_noise

    def forward(self, x, segmap):
        normalized = self.first_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        # original
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # position code
        # print(f"error 1 {self.gamma_pos_learn.size()}")
        # print(f"error 2 {x.size()}")
        gamma = gamma * (1 + self.conv_pos_gamma(self.gamma_pos_learn.unsqueeze(0).to(x.device)))
        beta = beta * (1 + self.conv_pos_beta(self.beta_pos_learn.unsqueeze(0).to(x.device)))

        # noise code
        gamma_noise, beta_noise = self.affine_noise(segmap)
        gamma = torch.cat([gamma, gamma_noise], dim=1)
        beta = torch.cat([beta, beta_noise], dim=1)

        out = normalized * (1 + gamma) + beta
        return out


def get_spectral_norm(opt):
    if opt.no_spectral_norm:
        return torch.nn.Identity()
    else:
        return spectral_norm


def get_norm_layer(opt, norm_nc):
    if opt.param_free_norm == 'instance':
        return nn.InstanceNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'syncbatch':
        return SynchronizedBatchNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'batch':
        return nn.BatchNorm2d(norm_nc, affine=False)
    else:
        raise ValueError('%s is not a recognized param-free norm type in SPADE'
                         % opt.param_free_norm)