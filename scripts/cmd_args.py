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


def add_common_options(parser):
    """Add common options to the parser.

    Args:
        parser : parser to add the arguments to.

    Return:
        parser : After adding the arguments.

    """
    # Pre-processing 1 : Get intervals
    parser.add('--interval_size', type=int, help='Interval size')
    parser.add('--noisybw', type=type_or_none_fn(str),
               help='Path to bigwig file containing noisy \
                        (low coverage/low quality) ATAC-seq signal',
               required=True)
    parser.add('--cleanbw', type=type_or_none_fn(str),
               help='Path to bigwig file containing clean \
                        (high-coverage/high-quality) ATAC-seq signal.\
                            Not used with --nolabel.')
    parser.add('--cleanpeakfile', type=type_or_none_fn(str),
               help='Path to narrowPeak or BED file containing peak calls '
                    'from \
                        MACS2.')
    parser.add('--layersbw', type=type_or_none_fn(str),
               help='Paths to bigWig files containing \
                            additional layers. If single file,  \
                            use format: "name:file". \
                            If there are multiple files, use format: \
                            "[name1:file1, name2:file2,...]"')
    parser.add('--batch_size', type=type_or_none_fn(int),
               help='batch size; number of intervals to read from '
                    'bigWig \
               files at a time. Unrelated to training/inference \
               batch size.', default=1000)
    parser.add('--nonzero', action='store_true',
               help='Only save intervals with nonzero coverage. \
                        Recommended when encoding training data, as intervals \
                        with zero coverage do not help the model to learn.')

    # experiment args
    parser.add('--genome', required=True, type=str,
               help='chromosome sizes file for the genome. You can \
                       specify a path to the sizes file or use the keywords \
                       \"hg19\" or \"hg38\" inorder to use the sizes file for \
                       human genome 19 or human genome 38. Chromosome \
                       sizes files for hg19 and hg38 are pre-installed with '
                    'atacworks.')
    parser.add('--exp_name', required=True, type=str,
               help='Give a unique name to the experiment. \
                     Used for naming output folder.')
    parser.add('--out_home', required=True, type=str,
               help='parent directory in which to create the output folder')
    parser.add('--task', required=True,
               choices=['regression', 'classification', 'both'],
               help='Task can be regression or\
                           classification or both. \
                           Should match the task the model was trained for.')
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
    parser.add('--layers', required=True, type=type_or_none_fn(str),
               help='Names of additional layers to read from h5 file \
                   as input, in the form: "[name1, name2]". \
                   Layers will be concatenated to the noisy ATAC-seq signal \
                   in the order supplied.')
    parser.add('--weights_path', required=True, type=type_or_none_fn(str),
               help="checkpoint path to load the model from for\
                   inference or resume training")
    # dist-env args
    parser.add('--gpu', required=False, type=int,
               help='GPU id to use; preempted by --distributed\
                           which uses all available gpus ')
    parser.add('--distributed', action='store_true',
               help='Do distributed training \
                   across all available gpus on the node. Note that \
                   --gpu and --distributed are mutually exclusive. Either\
                   one of the options must be provided but both cannot be\
                   provided at once.')
    parser.add('--seed', required=True, type=int,
               help='Seed value to set for RNG (Random Number Generators).\
                     Data loading and model initialization are \
                     deterministic with seed setting, model training is not.')


def add_train_options(parser):
    """Add training options to the parser.

    Args:
        parser : parser to add the arguments to.

    Return:
        parser : After adding the arguments.

    """
    add_common_options(parser)
    parser.add('--val_chrom', type=type_or_none_fn(str),
               help='Chromosome for validation')
    parser.add('--holdout_chrom', type=type_or_none_fn(str),
               help='Chromosome to hold out')
    parser.add('--nonpeak', type=type_or_none_fn(int),
               help='Ratio between number of non-peak\
                    intervals and peak intervals. In other words,\
                    no. of nonpeak intervals = nonpeak*(no. of peak intervals')
    # Learning args
    parser.add('--lr', required=True, type=float,
               help='Learning rate to be used for training.')
    parser.add('--epochs', required=True, type=int,
               help='Number of epochs to train the model for.')
    parser.add('--mse_weight', required=True, type=float,
               help='Relative weight of mse loss')
    parser.add('--pearson_weight', required=True, type=float,
               help='Relative weight of pearson correlation loss')
    parser.add('--poisson_weight', required=True, type=float,
               help='Relative weight of poisson loss')
    parser.add('--threshold', required=True,
               type=float,
               help='atacworks outputs probability values for peaks \
                       by default. These are thresholded at 0.5 for \
                       binary output and then classification metrics \
                       are calculated. Threshold value can be a float \
                       number between 0 and 1. \
                       You can set the threshold using this option. \
                       output < infer_threshold = 0,\
                       output > infer_threshold = 1.')
    # validation args
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

    parser.add('--wg', action='store_true',
               help='Set this flag to produce one set of intervals for\
                     whole genome. If you would like to run denoising \
                     on a subset of chromosome instead, take a look at \
                     the --regions option instead.')
    parser.add('--regions', required=True, type=type_or_none_fn(str),
               help='atacworks denoising is done on whole genome by'
                    'default. You can optionally specify a list of \
                     chromosomes separated by comma and no spaces \
                     like [chr1,chr2]. You can aslo provide list of \
                     region indices with each chromosome like \
                     [chr1:0-1000,chr2,chr3:0-500]. Please note \
                     NO SPACES. You can also provide a BED file that \
                     contains equally spaced chromosome intervals. \
                     See documentation for creating your own regions \
                     file.')
    parser.add('--peaks', action='store_true',
               help='Set this flag to output denosied peaks from atacworks. '
                    'If --task is \
                     regression, model only outputs denoised tracks \
                     and this option becomes irrelevant.')
    parser.add('--tracks', action='store_true',
               help='Set this flag to output denosied tracks from atacworks. '
                    'If --task is \
                    classification, model only outputs denoised peaks \
                     and this option becomes irrelevant.')
    parser.add('--threshold', required=True,
               type=type_or_none_fn(float),
               help='atacworks outputs probability values for peaks \
                       by default. OUtput can be thresholded binary \
                       output. You can set the threshold using this \
                       option. Threshold value has to be a float \
                       number between 0 and 1. \
                       output < infer_threshold = 0,\
                       output > infer_threshold = 1.')
    parser.add('--reg_rounding', required=True, type=int,
               help='Number of decimal digits to round values \
                       for regression outputs')
    parser.add('--batches_per_worker', required=True, type=int,
               help='number of batches to run per worker\
                               during multiprocessing')
    parser.add('--gen_bigwig', action='store_true',
               help='Save the inference output to bigiwig\
                               in addition to bedgraph')
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
        'denoise',
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

    if args.mode == "denoise":
        check_dependence(args.deletebg, args.gen_bigwig, parser,
                         "--deletebg requires --gen_bigwig")

    # Both options cannot be provided at once.
    check_mutual_exclusive(args.gpu, args.distributed, "If args.gpu is set,\
            cannot run atacworks in distributed mode. You can only use one \
            of the options.")

    # Either one of the options needs to be provided.
    if not (args.gpu is not None or args.distributed):
        parser.error("Either specify a gpu ID to run atacworks by \
                setting --gpu <ID> or run on ALL available gpus by \
                setting --distributed")

    return args
