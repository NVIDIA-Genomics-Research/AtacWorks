#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import configargparse


def model_params_args():
    parser = configargparse.ArgParser(default_config_files=['configs/model_structure.yaml'])

    # Model architecture args
    parser.add('--model', type=str, help='model type', choices=(
        'unet', 'resnet', 'linear', 'logistic', 'fc2', 'fc3'))
    parser.add('--bn', action='store_true', help='batch norm')
    parser.add('--nblocks', type=int,
                        help='number of regression blocks for resnet')
    parser.add('--dil', type=int,
                        help='dilation for regression blocks in resnet')
    parser.add('--width', type=int,
                        help='kernel size for regression blocks in resnet')
    parser.add('--nfilt', type=int,
                        help='number of filters for regression blocks in resnet')
    parser.add('--nblocks_cla', type=int,
                        help='number of classification blocks for resnet')
    parser.add('--dil_cla', type=int,
                        help='dilation for classification blocks in resnet')
    parser.add('--width_cla', type=int, 
                        help='kernel size for classification blocks in resnet')
    parser.add('--nfilt_cla', type=int,
                        help='number of filters for classification blocks in resnet')
    parser.add('--field', type=int,
                        help='receptive field for linear/logistic regression')

    args = parser.parse_known_args()

    return args
