# -*- coding: utf-8 -*-
"""
this code modified u++ using dcgan mode
"""
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func_down = act_func
        self.act_func_up = nn.ReLU(True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func_down(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func_up(out)

        return out


class UNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(args.input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

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

class NestedUNet(nn.Module):
    def __init__(self, input_nc, output_nc=1, lambda_o=1, tanh_act=True, deepsupervision=True, upp_scale=2):
        """
        :param args:
            input_channels
            deepsupervison
        """
        super().__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.deepsupervision = deepsupervision
        self.tanh_act = tanh_act
        self.lambda_o = lambda_o

        nb_filter = [64, 128, 256, 512, 1024]
        nb_filter = [int(x / upp_scale) for x in nb_filter]
        self.nb_filter = nb_filter
        """
        change the pooling layer to conv2d stride
        using func self.pool
        """
        # self.pool = nn.MaxPool2d(2, 2)
        # self.pool = nn.Conv2d()

        """
        change the upsampleing layer to conv2dtransposed stride
        using func self.up
        """
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.input_nc, nb_filter[0], nb_filter[0])
        self.pool0_0 = PoolConv(nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.pool1_0 = PoolConv(nb_filter[1])
        self.up1_0 = UpConv(nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.pool2_0 = PoolConv(nb_filter[2])
        self.up2_0 = UpConv(nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.pool3_0 = PoolConv(nb_filter[3])
        self.up3_0 = UpConv(nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.up4_0 = UpConv(nb_filter[4])


        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1_1 = UpConv(nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2_1 = UpConv(nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.up3_1 = UpConv(nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1_2 = UpConv(nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2_2 = UpConv(nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1_3 = UpConv(nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.output_nc, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.output_nc, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.output_nc, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.output_nc, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.output_nc, kernel_size=1)

        if self.tanh_act:
            self.tanh = nn.Tanh()

    def forward(self, input):
        nb_filter = self.nb_filter
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool0_0(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool1_0(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool2_0(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool3_0(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            output = (output1 + output2 + output3 + output4)/4
            output = output * self.lambda_o
            if self.tanh_act:
                return self.tanh(output)
            else:
                return (output1 + output2 + output3 + output4)/4
            # return [output1, output2, output3, output4]

        else:
            if self.tanh_act:
                output = self.final(x0_4)
                output = output * self.lambda_o
                output = self.tanh(output)
            else:
                output = self.final(x0_4)
                output = output * self.lambda_o
            return output