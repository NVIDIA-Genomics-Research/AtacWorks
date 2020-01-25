#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""Layer module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroSamePad1d(nn.Module):
    """Apply SAME zero padding to input."""

    def __init__(self, interval_size, kernel_size, stride, dilation):
        """Initialize layer.

        Args:
            interval_size : Genome interval size.
            kernel_size : Size of filter.
            stride : Stride for filter.
            dilation : Filter dilation.

        """
        super(ZeroSamePad1d, self).__init__()

        required_total_padding = ZeroSamePad1d._get_total_same_padding(
            interval_size, kernel_size, stride, dilation)
        padding_left = required_total_padding // 2
        padding_right = required_total_padding - padding_left
        self.pad = nn.ConstantPad1d((padding_left, padding_right), 0)

    @staticmethod
    def _get_total_same_padding(interval_size, kernel_size, stride, dilation):
        """Calculate total required padding.

        Args:
            interval_size : Genome interval size.
            kernel_size : Size of filter.
            stride : Stride for filter.
            dilation : Filter dilation.

        Return:
            Total padding required around the input for SAME padding.

        """
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        required_total_padding = (interval_size - 1) * \
            stride + effective_kernel_size - interval_size
        return required_total_padding

    def forward(self, x):
        """Execute layer on input.

        Args:
            x : Input data.

        """
        return self.pad(x)


class Activation(nn.Module):
    """Configurable activation layer."""

    def __init__(self, afunc='relu'):
        """Initialize layer.

        Args:
            afunc : Type of activation function.

        """
        super(Activation, self).__init__()
        self.act_layer = nn.Identity()
        if afunc == 'relu':
            self.act_layer = nn.ReLU()
        elif afunc == 'prelu':
            self.act_layer = nn.PReLU()
        elif afunc is not None:
            raise NotImplementedError

    def forward(self, x):
        """Execute layer on input.

        Args:
            x : Input data.

        """
        return self.act_layer(x)


class ConvAct1d(nn.Module):
    """1D conv layer with same padding.

    Optional batch normalization and activation layer.
    """

    def __init__(self, interval_size, in_channels, out_channels,
                 kernel_size, stride=1, dilation=1, bias=False,
                 bn=False, afunc='relu'):
        """Initialize.

        Args:
            interval_size : Genome interval size
            in_channels : Input channel
            out_channels : Output channels
            kernel_size : Filter size
            stride : Stride for filter
            dilation : Dilation for filter
            bias : Conv layer bias
            bn : Enable batch norm
            afunc : Activation function

        """
        self.interval_size = interval_size
        super(ConvAct1d, self).__init__()

        self.padding_layer = ZeroSamePad1d(
            interval_size, kernel_size, stride, dilation)
        self.conv_layer = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding=0,
            dilation=dilation, bias=bias)
        self.bn_layer = nn.BatchNorm1d(out_channels) if bn else None
        self.act_layer = Activation(afunc) if afunc else None

    def forward(self, x):
        """Execute layer on input.

        Args:
            x : Input data.

        """
        x = self.padding_layer(x)
        x = self.conv_layer(x)
        if self.bn_layer:
            x = self.bn_layer(x)
        if self.act_layer:
            x = self.act_layer(x)
        return x


class ResBlock(nn.Module):
    """Residual block.

    2 conv/activation layers followed by residual connection
    and third activation.

    """

    def __init__(self, interval_size, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, bias=False, bn=False,
                 afunc='relu', conv_input=False):
        """Initialize layer.

        Args:
            interval_size : Genome interval size.
            in_channels : Input channels.
            out_channels : Output channels.
            kernel_size : Filter size.
            stride : Filter stride.
            dilation : Dilation for filter.
            bias : Conv layer bias
            bn : Enable batch norm
            afunc : Activation function
            conv_input : Apply conv to input layer if True, else
            apply Identity

        """
        super(ResBlock, self).__init__()

        if conv_input:
            self.conv_input = ConvAct1d(interval_size, in_channels,
                                        out_channels, kernel_size=1,
                                        bn=bn, afunc=afunc)
        else:
            self.conv_input = nn.Identity()
        self.conv_act1 = ConvAct1d(
            interval_size, in_channels, out_channels, kernel_size,
            stride, dilation, bias, bn, afunc)
        self.conv_act2 = ConvAct1d(
            interval_size, out_channels, out_channels, kernel_size,
            stride, dilation, bias, bn, afunc)
        self.conv_act3 = ConvAct1d(
            interval_size, out_channels, out_channels, kernel_size,
            stride, dilation, bias, bn, afunc=None)
        self.activation = nn.PReLU() if afunc == 'prelu' else nn.ReLU()

    def forward(self, input):
        """Execute layer on input.

        Args:
            input : Input data.

        """
        x = self.conv_act1(input)
        x = self.conv_act2(x)
        x = self.conv_act3(x)
        x = x + self.conv_input(input)
        x = self.activation(x)

        return x


class DownBlock(nn.Module):
    """U-net down block - 2 conv/activation layers followed by max pool."""

    def __init__(self, interval_size, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, bias=False, bn=True, afunc='relu'):
        """Initialize layer.

        Args:
            interval_size : Genome interval size.
            in_channels : Input channels.
            out_channels : Output channels.
            kernel_size : Filter size.
            stride : Filter stride.
            dilation : Filter dilation.
            bias : Conv layer bias
            bn : Enable batch norm
            afunc : Activation function

        """
        super(DownBlock, self).__init__()

        self.conv_act1 = ConvAct1d(
            interval_size, in_channels, out_channels, kernel_size,
            stride, dilation, bias, bn, afunc)
        self.conv_act2 = ConvAct1d(
            interval_size, out_channels, out_channels, kernel_size,
            stride, dilation, bias, bn, afunc)
        self.max_pool = nn.MaxPool1d(2)

    def forward(self, input):
        """Execute layer on input.

        Args:
            input : Input data.

        """
        x = self.conv_act1(input)
        x = self.conv_act2(x)
        xp = self.max_pool(x)

        return x, xp


class UpBlock(nn.Module):
    """U-net up block - upsampling, merge, followed by 2 conv layers."""

    def __init__(self, interval_size, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, bias=False, bn=True, afunc='relu'):
        """Initialize.

        Args:
            interval_size : Genome interval size.
            in_channels : Input channels.
            out_channels : Output channels.
            kernel_size : Filter size.
            stride : Filter stride.
            dilation : Filter dilation.
            bias : Conv layer bias
            bn : Enable batch norm
            afunc : Activation function

        """
        super(UpBlock, self).__init__()

        self.conv_act1 = ConvAct1d(
            interval_size, in_channels, out_channels, kernel_size,
            stride, dilation, bias, bn, afunc)
        self.conv_act2 = ConvAct1d(
            interval_size, out_channels * 2, out_channels, kernel_size, stride,
            dilation, bias, bn, afunc)
        self.conv_act3 = ConvAct1d(
            interval_size, out_channels, out_channels, kernel_size, stride,
            dilation, bias, bn, afunc)

    def forward(self, x_up, x_down):
        """Execute layer on input.

        Args:
            x_up : Input data.
            x_down: Data from previous layer to concatenate.

        """
        x_up = F.interpolate(x_up, scale_factor=2, mode='nearest')
        x_up = self.conv_act1(x_up)
        x_up = torch.cat((x_down, x_up), dim=1)
        x_up = self.conv_act2(x_up)
        x_up = self.conv_act3(x_up)

        return x_up
