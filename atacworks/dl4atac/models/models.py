#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""Model definitions."""

from atacworks.dl4atac.layers import ConvAct1d, DownBlock, \
    ResBlock, UpBlock, ZeroSamePad1d

import torch
import torch.nn as nn


class DenoisingResNet(nn.Module):
    """Resnet model."""

    def __init__(self, interval_size, in_channels=1, out_channels=15,
                 num_blocks=5,
                 kernel_size=50, dilation=8, bn=False, afunc='relu',
                 num_blocks_class=2,
                 kernel_size_class=50, dilation_class=8,
                 out_channels_class=15):
        """Initialize the class.

        Args:
            interval_size : length (in bp) of the genomic intervals supplied
            to the model.
            in_channels : Number of input channels.
            out_channels: Number of output channels for all residual blocks
            in the regression part of the model.
            num_blocks: Number of residual blocks in the regression part
            of the model.
            kernel_size: Size of the kernel filter for all
            convolutional layers in the regression part of the model.
            dilation: Dilation parameter for all convolutional layers
            in the regression part of the model.
            bn: Batch normalization.
            afunc: activation function.
            num_blocks_class: Number of residual blocks in the
            classification part of the model.
            kernel_size_class: Size of the kernel filter for all
            convolutional layers in the classification part of the model.
            dilation_class: Dilation parameter for all convolutional layers
            in the classification part of the model.
            out_channels_class: Number of output channels for all residual
            blocks in the classification part of the model

        """
        self.interval_size = interval_size
        super(DenoisingResNet, self).__init__()

        self.res_blocks = nn.ModuleList()
        self.res_blocks_class = nn.ModuleList()

        # Residual blocks for regression
        self.res_blocks.append(
            ResBlock(interval_size, in_channels, out_channels, kernel_size,
                     dilation=dilation, bn=bn, afunc=afunc, conv_input=True))
        for _ in range(num_blocks - 1):
            self.res_blocks.append(
                ResBlock(interval_size, out_channels, out_channels,
                         kernel_size,
                         dilation=dilation, bn=bn, afunc=afunc,
                         conv_input=False))
        self.regressor = ConvAct1d(interval_size, in_channels=out_channels,
                                   out_channels=1, kernel_size=1, dilation=1,
                                   bn=bn, afunc=afunc)

        # Residual blocks for classification
        self.res_blocks_class.append(ResBlock(interval_size, in_channels=1,
                                              out_channels=out_channels_class,
                                              kernel_size=kernel_size_class,
                                              dilation=dilation_class, bn=bn,
                                              afunc=afunc, conv_input=True,
                                              bias=True))
        for _ in range(num_blocks_class - 1):
            self.res_blocks_class.append(
                ResBlock(interval_size, out_channels_class, out_channels_class,
                         kernel_size_class, dilation=dilation_class, bn=bn,
                         afunc=afunc, conv_input=False, bias=True))
        self.classifier = ConvAct1d(interval_size, in_channels=out_channels,
                                    out_channels=1, kernel_size=1, dilation=1,
                                    bn=bn, afunc=None, bias=True)

    def forward(self, x):
        """Get regression and classification forward propagated output.

        Args:
            x : Input.

        Return:
            out_reg: Regression output.
            out_cla: Classification output.

        """
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.regressor(x)
        out_reg = x.squeeze(1)
        for res_block in self.res_blocks_class:
            x = res_block(x)
        out_cla = torch.sigmoid(self.classifier(x).squeeze(1))

        return out_reg, out_cla


class DenoisingUNet(nn.Module):
    """U-net model."""

    def __init__(self, interval_size, in_channels=1, afunc='relu', bn=False):
        """Initialize the class.

        Args:
            interval_size : Interval size of chromosomes.
            in_channels : Number of input channels.
            afunc: Activation function.
            bn: Batch Normalization.

        """
        self.interval_size = interval_size
        super(DenoisingUNet, self).__init__()
        self.down1 = DownBlock(interval_size, in_channels=in_channels,
                               out_channels=16, kernel_size=5, bn=bn,
                               afunc=afunc)
        self.down2 = DownBlock(interval_size, in_channels=16,
                               out_channels=32, kernel_size=5, bn=bn,
                               afunc=afunc)
        self.down3 = DownBlock(interval_size, in_channels=32,
                               out_channels=64, kernel_size=25, bn=bn,
                               afunc=afunc)
        self.down4 = DownBlock(interval_size, in_channels=64,
                               out_channels=128, kernel_size=25, bn=bn,
                               afunc=afunc)

        self.conv5 = ConvAct1d(interval_size, in_channels=128,
                               out_channels=256,
                               kernel_size=250, dilation=1, bn=bn, afunc=afunc)

        self.up6 = UpBlock(interval_size, in_channels=256,
                           out_channels=128, kernel_size=5, bn=bn, afunc=afunc)
        self.up7 = UpBlock(interval_size, in_channels=128,
                           out_channels=64, kernel_size=5, bn=bn, afunc=afunc)
        self.up8 = UpBlock(interval_size, in_channels=64,
                           out_channels=32, kernel_size=5, bn=bn, afunc=afunc)
        self.up9 = UpBlock(interval_size, in_channels=32,
                           out_channels=16, kernel_size=5, bn=bn, afunc=afunc)

        self.regressor = ConvAct1d(
            interval_size, in_channels=16, out_channels=1, kernel_size=1,
            dilation=1, bn=bn, afunc=afunc)
        self.classifier = ConvAct1d(
            interval_size, in_channels=16, out_channels=1, kernel_size=1,
            dilation=1, bn=bn, afunc=None)

    def forward(self, input):
        """Forward.

        Args:
            input: Input data.

        Return:
            out_reg: Regression output.
            out_cla: Classification output

        """
        # for readability, keeping itermediate p1 ~ p4 and x5 ~ x9,
        # but actually unnecessary and a waste of memory
        x1, p1 = self.down1(input)
        x2, p2 = self.down2(p1)
        x3, p3 = self.down3(p2)
        x4, p4 = self.down4(p3)

        x5 = self.conv5(p4)

        x6 = self.up6(x5, x4)
        x7 = self.up7(x6, x3)
        x8 = self.up8(x7, x2)
        x9 = self.up9(x8, x1)

        out_reg = self.regressor(x9).squeeze(1)
        out_cla = torch.sigmoid(
            self.classifier(x9).squeeze(1))  # (N, 1, L) => (N, L)

        return out_reg, out_cla


# Baseline models

class DenoisingLinear(nn.Module):
    """Linear regression model."""

    def __init__(self, interval_size, field, in_channels=1, out_channels=1):
        """Initialize the class.

        Args:
            interval_size : Interval size of chromosomes.
            field : Receptive field
            in_channels : Number of input channels
            out_channels:Number of output channels

        """
        super(DenoisingLinear, self).__init__()
        self.padding_layer = ZeroSamePad1d(
            interval_size, kernel_size=field, stride=1, dilation=1)
        self.conv_layer = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=field, stride=1, padding=0, dilation=1, bias=True)

    def forward(self, x):
        """Forward.

        Args:
            x : Input

        Return:
            x: Input

        """
        x = self.padding_layer(x)
        x = self.conv_layer(x).squeeze(1)
        return x


class DenoisingLogistic(nn.Module):
    """Logistic regression model: a linear model with sigmoid activation."""

    def __init__(self, interval_size, field, in_channels=1, out_channels=1):
        """Initialize the class.

        Args:
            interval_size : Interval size of chromosomes.
            field : Receptive field
            in_channels : Number of input channels
            out_channels: Number of output channels

        """
        super(DenoisingLogistic, self).__init__()
        self.denoising_linear = DenoisingLinear(
            interval_size, field, in_channels, out_channels)

    def forward(self, x):
        """Forward.

        Args:
            x : Input

        Return:
            x: Input

        """
        x = torch.sigmoid(self.denoising_linear(x))
        return x
