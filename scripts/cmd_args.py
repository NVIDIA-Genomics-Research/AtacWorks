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


def add_common_options(parser):
    """Add common options to the parser.

    Args:
        parser : parser to add the arguments to.

    Return:
        parser : After adding the arguments.

    """
    # experiment args
    parser.add('--label', required=True, type=str,
               help='label of the experiment; used for naming output folder')
    parser.add('--out_home', required=True, type=str,
               help='parent directory in which to create the output folder')
    parser.add('--task', required=True,
               choices=['regression', 'classification', 'both'],
               help='Task can be regression or\
                           classification or both. (default: %(default)s)')
    parser.add('--print_freq', required=True, type=int,
               help="Logging frequency")
    parser.add('--bs', required=True, type=int,
               help="batch_size")
    parser.add('--num_workers', required=True, type=int,
               help="number of workers for dataloader")
    # Dataset args
    parser.add('--pad', required=True, type=type_or_none_fn(int),
               help="Number of additional bases to add as padding \
                   on either side of each interval. Use the same --pad \
                   value that was supplied to bw2h5.py when creating \
                   the h5 files for training and validation.")
    parser.add('--transform', required=True, type=str, choices=['log', 'None'],
               help='transformation to apply to\
                           coverage tracks before training')
    parser.add('--layers', type=str,
               help='Names of additional layers to read from h5 file \
                   as input, in the form: "[name1, name2]". \
                   Layers will be concatenated to the noisy ATAC-seq signal \
                   in the order supplied.')
    parser.add('--weights_path', required=True, type=type_or_none_fn(str),
               help="checkpoint path to load the model from for\
                   inference or resume training")
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


def add_train_options(parser):
    """Add training options to the parser.

    Args:
        parser : parser to add the arguments to.

    Return:
        parser : After adding the arguments.

    """
    add_common_options(parser)
    parser.add('--files_train', required=True, type=str,
               help='list of data files in the form of "[file1, file2, '
                    '...]";'
                    'or a single path to a file or folder of files')
    parser.add('--checkpoint_fname', required=True, type=str,
               help="checkpoint filename to save the model")
    parser.add('--save_freq', required=True, type=int,
               help="model checkpoint saving frequency")
    # Learning args
    parser.add('--clip_grad', required=True, type=float,
               help='Grad clipping for bad/extreme batches')
    parser.add('--lr', required=True, type=float,
               help='learning rate')
    parser.add('--epochs', required=True, type=int,
               help='Number of epochs')
    parser.add('--mse_weight', required=True, type=float,
               help='relative weight of mse loss')
    parser.add('--pearson_weight', required=True, type=float,
               help='relative weight of pearson correlation loss')
    parser.add_argument('--poisson_weight', required=True, type=float,
                        help='relative weight of poisson loss')
    # validation args
    parser.add('--val_files', required=True, type=str,
               help='list of data files in the form of [file1, file2, '
                    '...];'
                    'or a single path to a folder of files')
    parser.add('--eval_freq', required=True, type=int,
               help="evaluation frequency")
    parser.add('--threshold', required=True, type=float,
               help="probability threshold above which to call peaks. \
               Used for classification metrics")
    parser.add_argument('--best_metric_choice', required=True,
                        type=str,
                        choices=['BCE', 'MSE', 'Recall',
                                 'Specificity', 'CorrCoef', 'AUROC'],
                        help="metric to select the best model.\
                                Choice is case sensitive.")


def add_inference_options(parser):
    """Add inference options to the parser.

    Args:
        parser : parser to add the arguments to.

    Return:
        parser : After adding the arguments.

    """
    add_common_options(parser)
    parser.add('--config', required=False,
               is_config_file=True, help='config file path')
    parser.add('--files', required=True, type=str,
               help='list of data files in the form of "[file1, file2, '
                    '...]";'
                    'or a single path to a file or folder of files')
    parser.add('--intervals_file', required=True, type=str,
               help='bed file containing the genomic\
                               intervals for inference')
    parser.add('--sizes_file', required=True, type=str,
               help='chromosome sizes file for the genome. \
                       Chromosome sizes files for hg19 and hg38 are \
                       given in the data/reference folder.')
    parser.add('--infer_threshold', required=True,
               type=type_or_none_fn(float),
               help='threshold above which to call peaks from the \
                       predicted probability values.')
    parser.add('--reg_rounding', required=True, type=int,
               help='number of decimal digits to round values \
                       for regression outputs')
    parser.add('--cla_rounding', required=True, type=int,
               help='number of decimal digits to round values \
                       for classification outputs')
    parser.add('--batches_per_worker', required=True, type=int,
               help='number of batches to run per worker\
                               during multiprocessing')
    parser.add('--gen_bigwig', action='store_true',
               help='save the inference output to bigiwig\
                               in addition to bedgraph')
    parser.add('--result_fname', required=True, type=str,
               help='prefix for the inference result files.')
    parser.add('--deletebg', action='store_true',
               help='delete output bedGraph file')


def add_eval_options(parser):
    """Add evaluation options to the parser.

    Args:
        parser : parser to add the arguments to.

    Return:
        parser : After adding the arguments.

    """
    add_inference_options(parser)

    parser.add('--threshold', required=True, type=float,
               help="probability threshold above which to call peaks. \
               Used for classification metrics")
    parser.add('--best_metric_choice', required=True,
               type=str,
               choices=['BCE', 'MSE', 'Recall',
                        'Specificity', 'CorrCoef', 'AUROC'],
               help="metric to select the best model.\
                                Choice is case sensitive.")


def parse_args(root_dir):
    """Parse command line arguments.

    Args:
        root_dir : Path to the root directory,
        where the configs folder can be found.

    Return:
        args : parsed argument object.

    """
    parser = configargparse.ArgParser()
    # =========================================================================
    # training args
    subparsers = parser.add_subparsers(dest="mode")
    train_config_path = os.path.join(root_dir, 'configs', 'train_config.yaml')
    parser_train = subparsers.add_parser(
        'train',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=[train_config_path])
    add_train_options(parser_train)
    # =========================================================================
    # Inference args
    infer_config_path = os.path.join(root_dir, 'configs', 'infer_config.yaml')
    parser_infer = subparsers.add_parser(
        'infer',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=[infer_config_path])
    add_inference_options(parser_infer)
    # =========================================================================
    # Evaluation args
    parser_eval = subparsers.add_parser(
        'eval',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=[infer_config_path])
    add_eval_options(parser_eval)
    # =========================================================================
    args, extra = parser.parse_known_args()

    if args.mode == "infer":
        check_dependence(args.deletebg, args.gen_bigwig, parser,
                         "--deletebg requires --gen_bigwig")

    return args
