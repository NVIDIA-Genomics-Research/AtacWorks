#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""Util functions for model parsing and building."""

# system imports
import logging
import os
import sys
import warnings

# module imports
from atacworks.dl4atac.models.models import DenoisingLinear, \
    DenoisingLogistic
from atacworks.dl4atac.models.models import DenoisingResNet, DenoisingUNet

from atacworks.dl4atac.utils import load_model, myprint

import configargparse

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

warnings.filterwarnings("ignore")

# Set up logging
log_formatter = logging.Formatter(
    '%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('AtacWorks-model_utils')
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)


def type_or_none_fn(type):
    """Generate function to interpret a string as a specific type or None.

    Args:
        type: Data type to check for.

    Error:
        None, if val is str(None)
        else, type(val)

    """
    def type_or_none(val):
        if str(val) == "None":
            return None
        else:
            return type(val)
    return type_or_none


def model_args_v1(root_dir):
    """Parse arguments relevant to model structure.

    Arguments are loaded from a default config file which can be overwritten
    by
    environment variables, command line or custom config file. Below is the
    hierarchy.
    command line > environment variables > config file values > defaults

    Args:
        root_dir : Path to the parent directory of the script being run.

    Return:
        args : Parsed argument object.

    """
    parser = configargparse.ArgParser(default_config_files=[os.path.join(
        root_dir, 'configs', 'model_structure.yaml')])
    parser.add('--config_mparams', required=False, is_config_file=True,
               help='config file path')
    parser.add('--model', required=True, type=str,
               help='model type', choices=('unet', 'resnet', 'linear',
                                           'logistic', 'fc2', 'fc3'))
    parser.add('--bn', action='store_true', help='batch norm')
    parser.add('--afunc', required=True, type=str,
               help='activation')
    parser.add('--nblocks', required=True, type=int,
               help='number of regression blocks for resnet')
    parser.add('--dil', required=True, type=int,
               help='dilation for regression blocks in resnet')
    parser.add('--width', required=True, type=int,
               help='kernel size for regression blocks in resnet')
    parser.add('--nfilt', required=True, type=int,
               help='number of filters for regression blocks in resnet')
    parser.add('--nblocks_cla', required=True, type=int,
               help='number of classification blocks for resnet')
    parser.add('--dil_cla', required=True, type=int,
               help='dilation for classification blocks in resnet')
    parser.add('--width_cla', required=True, type=int,
               help='kernel size for classification blocks in resnet')
    parser.add('--nfilt_cla', required=True, type=int,
               help='number of filters for classification blocks in resnet')
    parser.add('--field', required=True, type=type_or_none_fn(int),
               help='receptive field for linear/logistic regression')
    parser.add('--in_channels', required=True, type=int,
               help='number of channels for input data')

    args = parser.parse_known_args()

    return args


def build_model(rank, interval_size, resume,
                infer, evaluate, weights_path,
                gpu, distributed):
    """Build the model with requested parameters.

    Args:
        rank : rank.
        interval_size : Interval size of chromosomes.
        resume : Resume from given model
        infer : Build model for inference.
        evaluate : Build model for evaluation.
        weights_path : Path to weights.
        gpu : Identity of GPU to be used.
        distributed : Distributed.

    Return:
        model : Built model.
        model_args : Parsed argument object containing model parameters.

    """
    # Read model parameters
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), ".."))
    model_args, extra = model_args_v1(root_dir)

    myprint("Building model: {} ...".format(
        model_args.model), color='yellow', rank=rank)
    # TODO: implement a model dic for model instantiation

    if model_args.model == 'unet':  # args.task == 'both'
        model = DenoisingUNet(interval_size=interval_size,
                              afunc=model_args.afunc, bn=model_args.bn)
    elif model_args.model == 'resnet':
        model = DenoisingResNet(interval_size=interval_size,
                                afunc=model_args.afunc,
                                bn=model_args.bn,
                                num_blocks=model_args.nblocks,
                                num_blocks_class=model_args.nblocks_cla,
                                out_channels=model_args.nfilt,
                                out_channels_class=model_args.nfilt_cla,
                                kernel_size=model_args.width,
                                kernel_size_class=model_args.width_cla,
                                dilation=model_args.dil,
                                dilation_class=model_args.dil_cla,
                                in_channels=model_args.in_channels)

    elif model_args.model == 'linear':
        model = DenoisingLinear(
            interval_size=interval_size, field=model_args.field)

    elif model_args.model == 'logistic':
        model = DenoisingLogistic(
            interval_size=interval_size, field=model_args.field)

    # TODO: there is a potential problem with loading model on each device
    #  like this. keep an eye on torch.load()'s map_location arg
    if resume or infer or evaluate:
        model = load_model(model, weights_path, rank)

    model = model.cuda(gpu)

    if distributed:
        _logger.info('Compiling model in DistributedDataParallel')
        model = DistributedDataParallel(model, device_ids=[gpu])
    elif gpu > 1:
        _logger.info('Compiling model in DataParallel')
        model = nn.DataParallel(
            model, device_ids=list(range(gpu))).cuda()

    myprint("Finished building.", color='yellow', rank=rank)
    return model, model_args
