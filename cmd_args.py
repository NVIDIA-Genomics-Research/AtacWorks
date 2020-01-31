#!/usr/bin/env python

#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""Module to parse command line arguments of main.py."""

import os

import configargparse


def check_dependence(
        arg1, arg2, parser, err_msg="Argument dependency test failed."):
    """Ensure provided arguments are both defined. Raise error if not.

    Args:
        arg1 : One of the dependent argument.
        arg2 : Argument to check arg1 against.
        parser: Argument parser object.
        err_msg : Error message to print out if dependency is not met.

    Error:
        Print error if condition is not met.

    """
    if arg1 and not arg2:
        parser.error(err_msg)


def check_mutual_exclusive(
        arg1, arg2, parser, err_msg="Argument mutual exclusive test failed."):
    """Ensure provided arguments are not defined at the same time.

    Args:
        arg1 : One of the mutually exclusive argument.
        arg2 : Argument to check arg1 against.
        parser: Argument parser object.
        err_msg : Error message to print out if both arguments are defined.

    Error:
        Print error if condition is not met.

    """
    if arg1 and arg2:
        parser.error(err_msg)


def type_or_none_fn(type):
    """Ensure provided arguments are not defined at the same time.

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


def parse_args(root_dir):
    """Parse command line arguments.

    Args:
        root_dir : Path to the root directory,
        where the configs folder can be found.

    Return:
        args : parsed argument object.

    """
    config_path = os.path.join(root_dir, 'configs', 'config_params.yaml')
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=[config_path])
    parser.add('--config', required=False,
               is_config_file=True, help='Provide a custom config file path.\
               Default config values will be overwritten with custom config values.')

    # Learning args
    parser.add('--clip_grad', required=True, type=float,
               help='Whether to perform gradient clipping for bad/extreme batches')
    parser.add('--lr', required=True, type=float,
               help='learning rate to use for training.')
    parser.add('--epochs', required=True, type=int,
               help='Number of epochs to train the model for.')
    parser.add('--mse_weight', required=True, type=float,
               help='relative weight of mean squared error loss')
    parser.add('--pearson_weight', required=True, type=float,
               help='relative weight of pearson loss')
    parser.add_argument('--poisson_weight', required=True, type=float,
                        help='relative weight of poisson loss')
# =============================================================================
    # experiment args
    parser.add('--label', required=True, type=str,
               help='label of the experiment; used for naming output folder')
    parser.add('--out_home', required=True, type=str,
               help='parent directory for the experiment folder')
    parser.add('--train', action='store_true',
               help='training; preempt --infer')
    parser.add('--infer', action='store_true',
               help='run inference')
    parser.add('--resume', action='store_true',
               help='resume training')
    parser.add('--eval', action='store_true',
               help='evaluation: inference + result dumping +\
                       metrics evaluation')
    parser.add('--infer_files', required=True, type=str,
               help='list of data files in the form of [file1, file2, ...];'
               'or a single path to a folder of files')
    parser.add('--intervals_file', required=True, type=str,
               help='bed file containing the chr and\
                       interval values for inference')
    parser.add('--sizes_file', required=True, type=str,
               help='bed file containing the chr and max size values')
    parser.add('--infer_threshold', required=True,
               type=type_or_none_fn(float),
               help='threshold the output peaks to this value')
    parser.add('--reg_rounding', required=True, type=int,
               help='rounding values for regression outputs')
    parser.add('--cla_rounding', required=True, type=int,
               help='rounding values for classification outputs')
    parser.add('--batches_per_worker', required=True, type=int,
               help='number of batches to run per worker\
                       during multiprocessing')
    parser.add('--gen_bigwig', action='store_true',
               help='save the inference output to bigiwig\
                       in addition to bedgraph')
    parser.add('--weights_path', required=True, type=str,
               help="checkpoint path to load the model from for\
               inference or resume training")
    parser.add('--result_fname', required=True, type=str,
               help='filename of the inference results')
    parser.add_argument('--deletebg', action='store_true',
                        help='delete output bedGraph file')

    # training args
    parser.add('--task', required=True,
               choices=['regression', 'classification', 'both'],
               help='Task can be regression or\
                       classification or both. (default: %(default)s)')
    parser.add('--train_files', required=True, type=str,
               help='list of data files in the form of [file1, file2, ...];'
               'or a single path to a folder of files')
    parser.add('--print_freq', required=True, type=int,
               help="Logging frequency")
    parser.add('--bs', required=True, type=int,
               help="batch_size")
    parser.add('--num_workers', required=True, type=int,
               help="number of workers for dataloader")
    parser.add('--checkpoint_fname', required=True, type=str,
               help="checkpoint filename to save the model")
    parser.add('--save_freq', required=True, type=int,
               help="model checkpoint saving frequency")

    # Dataset args
    parser.add('--pad', required=True, type=type_or_none_fn(int),
               help="Padding around intervals")
    parser.add('--transform', required=True, type=str, choices=['log', 'None'],
               help='transformation to apply to\
                       coverage tracks before training')

    # validation args
    parser.add('--val_files', required=True, type=str,
               help='list of data files in the form of [file1, file2, ...];'
               'or a single path to a folder of files')
    parser.add('--eval_freq', required=True, type=int,
               help="evaluation frequency")
    parser.add('--threshold', required=True, type=float,
               help="threshold for classification metrics")
    parser.add_argument('--best_metric_choice', required=True,
                        type=str,
                        choices=['BCE', 'MSE', 'Recall',
                                 'Specificity', 'CorrCoef', 'AUROC'],
                        help="metric to be considered for best metric.\
                                Choice is case sensitive.")

    # dist-env args
    parser.add('--gpu', required=True, type=int,
               help='GPU id to use; preempted by --distributed\
                       which uses all available gpus ')
    parser.add('--distributed', action='store_true',
               help='Do distributed training \
               across all available gpus on the node')
    parser.add('--dist-url', required=True, type=str,
               help='url used to set up distributed training')
    parser.add('--dist-backend', required=True, type=str,
               help='distributed backend')

    # debug
    parser.add('--debug', action='store_true',
               help='Enable debug prints')

    args, extra = parser.parse_known_args()

    check_mutual_exclusive(args.train, args.infer, parser,
                           "--train and --infer are mutual exclusive")
    check_mutual_exclusive(args.train, args.eval, parser,
                           "--train and --eval are mutual exclusive")
    check_mutual_exclusive(args.infer, args.eval, parser,
                           "--infer and --eval are mutual exclusive")
    # check_mutual_exclusive(args.infer, args.distributed, parser,\
    #        "--infer and --distributed are mutual exclusive")

    check_dependence(args.train, args.train_files, parser,
                     "--train requires --train_files")
    check_dependence(args.train, args.val_files, parser,
                     "--train requires --val_files")
    check_dependence(args.train, args.checkpoint_fname, parser,
                     "--train requires --checkpoint_fname")

    check_dependence(args.infer, args.infer_files, parser,
                     "--infer requires --infer_files")
    check_dependence(args.infer, args.weights_path, parser,
                     "--infer requires --weights_path")

    check_dependence(args.eval, args.val_files, parser,
                     "--eval requires --val_files")
    check_dependence(args.eval, args.weights_path, parser,
                     "--eval requires --weights_path")

    return args
