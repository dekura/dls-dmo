'''
@Author: Guojin Chen
@Date: 2019-11-18 22:25:55
@LastEditTime: 2019-11-18 23:28:59
@Contact: cgjhaha@qq.com
@Description: 
'''
# -*- coding: utf-8 -*-
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 5, padding=1)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 5, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.act_func(out)
        return out


class PoolConv(nn.Module):
    def __init__(self, channels):
        """
        :params:
            channels-- input and out channels
        """
        super().__init__()
        self.pool_conv = nn.Conv2d(channels, channels, kernel_size=4,
                        stride=2, padding=1, bias=False
        )
    def forward(self, input):
        return self.pool_conv(input)

class UpConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(channels, channels,
                                    kernel_size=4, stride=2,
                                    padding=1, bias=False)

    def forward(self, input):
        return self.up_conv(input)

class UNet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc

        nb_filter = [32, 64, 128, 256, 512]
        # nb_filter = [16, 64, 128, 512, 1024]

        # self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.input_nc, nb_filter[0], nb_filter[0])
        self.pool0_0 = PoolConv(nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.pool1_0 = PoolConv(nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.pool2_0 = PoolConv(nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.pool3_0 = PoolConv(nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.up4_0 = UpConv(nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.up3_1 = UpConv(nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2_2 = UpConv(nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1_3 = UpConv(nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], self.output_nc, kernel_size=1)
        self.tanh = nn.Tanh()


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool0_0(x0_0))
        x2_0 = self.conv2_0(self.pool1_0(x1_0))
        x3_0 = self.conv3_0(self.pool2_0(x2_0))
        x4_0 = self.conv4_0(self.pool3_0(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up1_3(x1_3)], 1))

        output = self.final(x0_4)
        output = self.tanh(output)

        return output