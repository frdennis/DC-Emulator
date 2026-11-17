import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log2
import math
import numpy as np
from scipy.special import hyp2f1
import itertools

from .layers import *
from .utils import *


class AUNetG(nn.Module):
    def __init__(self, in_chan, out_chan, style_size, scale_factor=16,
                chan_base=512, chan_min=16, chan_max=512, cat_noise=True,
                **kwargs):
        super().__init__()
        # https://github.com/dschaurecker/dl_halo/blob/main/train_test/map2map/models/generator.py

        self.conv_in = ConvStyled3d(in_chan, 32, style_size, kernel_size=1, stride=1)

        self.resblock1 = ResNetBlock(32, 64, 64, style_size) # +2
        self.down1 = DownBlock(64, style_size)

        self.resblock2 = ResNetBlock(64, 128, 128, style_size)  # +4
        self.down2 = DownBlock(128, style_size)

        self.bottleneck1 = ResNetBlock(128, 256, 512, style_size) # +8
        self.bottleneck2 = ResNetBlock(512, 256, 128, style_size)

        self.up1 = UpBlock(128, style_size)
        self.resblock3 = ResNetBlock(128+128, 128, 64, style_size)

        self.up2 = UpBlock(64, style_size)
        self.resblock4 = ResNetBlock(64+64, 64, 32, style_size, last_act=False)

        self.conv_out = ConvStyled3d(32, out_chan, style_size, kernel_size=1, stride=1)

    def forward(self, x, s, noise_arr=[None,]*18):
        x = F.interpolate(x, scale_factor=2,
            mode='trilinear', align_corners=False)

        y = x   # Predict the residual to y

        x = self.conv_in((x, s))
        
        x1 = self.resblock1(x, s, noise1=noise_arr[0], noise2=noise_arr[1])   # +2
        x = self.down1(x1, s, noise=noise_arr[2])       # +1

        x2 = self.resblock2(x, s, noise1=noise_arr[3], noise2=noise_arr[4])   # +2        
        x = self.down2(x2, s, noise=noise_arr[5])       # +1

        x = self.bottleneck1(x, s, noise1=noise_arr[6], noise2=noise_arr[7])  # +2
        x = self.bottleneck2(x, s, noise1=noise_arr[8], noise2=noise_arr[9])  # +2 

        x = self.up1(x, s, noise1=noise_arr[10], noise2=noise_arr[11])          # +2
        x2 = narrow_like_(x2, x)
        x = torch.cat([x2, x], dim=1) 
        x = self.resblock3(x, s, noise1=noise_arr[12], noise2=noise_arr[13])    # +2

        x = self.up2(x, s, noise1=noise_arr[14], noise2=noise_arr[15])          # +2
        x1 = narrow_like_(x1, x)
        x = torch.cat([x1, x], dim=1)
        x = self.resblock4(x, s, noise1=noise_arr[16], noise2=noise_arr[17])    # +2
        
        x = self.conv_out((x, s))

        return y + x
