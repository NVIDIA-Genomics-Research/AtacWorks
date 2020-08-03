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
    # Pre-processing
    parser.add('--genome', required=True, type=str,
               help='chromosome sizes file for the genome. Sizes \
                     files for human genome 19 (hg19) and human \
                     genome 38 (hg38) are already available. To use \
                     hg19, specify --genome hg19, to use \
                     hg38, specify --genome hg38. Alternatively, \
                     to pass in a path to a different sizes file, \
                     specify --genome <path-to-file>.')
    parser.add('--interval_size', type=int, help='Interval size \
                defines the input feature size for the model. It should \
                be atleast as big as the receptive field of the network.')
    parser.add('--noisybw', required=True, type=type_or_none_fn(str),
               help='Path to bigwig file containing noisy \
                    (low coverage/low quality/ low cell) \
                    ATAC-seq signal')
    parser.add('--layersbw', type=type_or_none_fn(str),
               help='Paths to bigWig files containing \
                     additional layers. If single file,  \
                     use format: "name:file". \
                     If there are multiple files, use format: \
                     "[name1:file1, name2:file2,...]"')
    parser.add('--read_buffer', type=type_or_none_fn(int),
               help='Number of intervals to read from bigWig \
               files at a time, since very big files may not fit \
               in memory if read at once.')
    parser.add('--nonzero', action='store_true',
               help='Only save intervals with nonzero coverage. \
                        Recommended when encoding training data, as intervals \
                        with zero coverage do not help the model to learn.')

    # experiment args
    parser.add('--exp_name', required=True, type=str,
               help='Name of the experiment; used for naming output folder')
    parser.add('--out_home', required=True, type=str,
               help='parent directory in which to create the output folder')
    parser.add('--task', required=True,
               choices=['regression', 'classification', 'both'],
               help='Task can be regression or classification or both. \
                     When using for denoising, this should match the \
                     task the model was trained for.')
    parser.add('--batch_size', required=True, type=int,
               help="batch size to be used for training.")
    parser.add('--num_workers', required=True, type=int,
               help="number of workers for dataloader")
    # Dataset args
    parser.add('--pad', required=True, type=type_or_none_fn(int),
               help="Number of additional bases to add as padding \
                   on either side of each interval.")
    parser.add('--layers', required=True, type=type_or_none_fn(str),
               help='Names of additional layers to read from h5 file \
                   as input, in the form: "[name1, name2]". \
                   Layers will be concatenated to the noisy ATAC-seq signal \
                   in the order supplied.')
    parser.add('--weights_path', required=True, type=type_or_none_fn(str),
               help="checkpoint path to load the model from for\
                   inference or resume training")
    # dist-env args
    parser.add('--gpu_idx', required=False, type=int,
               help='GPU ID to use. ID can be known from nvidia-smi; \
                       preempted by --distributed which uses all \
                       available gpus ')
    parser.add('--distributed', action='store_true',
               help='Do distributed training \
                   across all available gpus on the node')
    parser.add('--dist-url', required=True, type=str,
               help='url used to set up distributed training')
    parser.add('--dist-backend', required=True, type=str,
               help='distributed backend')
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
    parser.add('--cleanbw', type=type_or_none_fn(str),
               help='Path to bigwig file containing clean \
                     (high-coverage/high-quality) ATAC-seq signal.')
    parser.add('--cleanpeakfile', type=type_or_none_fn(str),
               help='Path to narrowPeak or BED file containing peak calls '
                    'from MACS2 on the clean (high-coverage/high-quality) \
                     ATAC-seq signal.')
    parser.add('--val_chrom', type=type_or_none_fn(str),
               help='Chromosome to be reserved for validation')
    parser.add('--holdout_chrom', type=type_or_none_fn(str),
               help='Chromosome to be reserved for hold out')
    parser.add('--nonpeak', type=type_or_none_fn(int),
               help='Ratio between number of non-peak intervals and \
                     peak intervals. In other words, \
                     nonpeak intervals = nonpeak*(peak intervals)')
    parser.add('--checkpoint_fname', required=True, type=str,
               help="checkpoint filename to save the model")
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
    parser.add('--train_h5_files', required=False,
               type=type_or_none_fn(str),
               help="Instead of providing bigwig files, users can \
                     optionally provide pre-processed h5 files \
                     generated by previous runs. If h5 files are \
                     provided, then atacworks will skip re-generating \
                     the h5 files. ONLY MEANT FOR ADVANCED USERS. Can \
                     provide path to the folder containing all h5 \
                     files or a comma separated list of file paths \
                     like [file1,file2,file3]")
    # validation args
    parser.add('--val_h5_files', required=False,
               type=type_or_none_fn(str),
               help="Instead of providing bigwig files, users can \
                     optionally provide pre-processed h5 files \
                     generated by previous runs. If h5 files are \
                     provided, then atacworks will skip re-generating \
                     the h5 files. ONLY MEANT FOR ADVANCED USERS. Can \
                     provide path to the folder containing all h5 \
                     files or a comma separated list of file paths \
                     like [file1,file2,file3]")
    parser.add('--threshold', required=True, type=float,
               help="probability threshold above which to call peaks. \
               Used for classification metrics")
    parser.add('--best_metric_choice', required=True,
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
    parser.add('--regions', required=True, type=type_or_none_fn(str),
               help='atacworks denoising is done on whole genome by \
                     default. You can optionally specify a list of \
                     chromosomes separated by comma and no spaces \
                     like [chr1,chr2]. You can also provide list of \
                     region indices with each chromosome like \
                     [chr1:0-1000,chr2,chr3:0-500]. Please note \
                     NO SPACES. You can also provide a BED file that \
                     contains genomic intervals of length equal to \
                     --interval_size. The BED file should have three \
                     columns (chromosome, start position, end position) \
                     and no header. Specified chromosomes MUST be \
                     present in the --genome file and the number of \
                     base pairs specified by the region i.e, the diff \
                     where diff = (end position - start position) \
                     MUST be a multiple of --interval_size.')
    parser.add('--peaks', action='store_true',
               help='Output denoised peaks from atacworks. \
                     If --task is regression, \
                     model only outputs denoised tracks and \
                     this option becomes irrelevant.')
    parser.add('--tracks', action='store_true',
               help='Output denosied tracks from atacworks. If --task is classification, \
                       model only outputs denoised peaks and \
                       this option becomes irrelevant.')
    parser.add('--threshold', required=True,
               type=type_or_none_fn(float),
               help='threshold above which to call peaks from the \
                     predicted probability values.')
    parser.add('--reg_rounding', required=True, type=int,
               help='number of decimal digits to round values \
                       for regression outputs')
    parser.add('--batches_per_worker', required=True, type=int,
               help='number of batches to run per worker\
                               during multiprocessing')
    parser.add('--gen_bigwig', action='store_true',
               help='save the inference output to bigwig\
                               in addition to bedgraph')
    parser.add('--deletebg', action='store_true',
               help='delete output bedGraph file')
    parser.add('--out_resolution', required=False,
               type=type_or_none_fn(int),
               help='resolution of output files. default 1bp. \
                     Atacworks always denoises at 1 base pair \
                     resolution. If out_resolution is 5, then \
                     the coverage values for every 5 base pairs is \
                     averaged. Keep in mind that the interval_size \
                     provided should be a multiple of the \
                     out_resolution value provided.')
    parser.add('--denoise_h5_files', required=False,
               type=type_or_none_fn(str),
               help="Instead of providing bigwig files, users can \
                     optionally provide pre-processed h5 files \
                     generated by previous runs. If h5 files are \
                     provided, then atacworks will skip re-generating \
                     the h5 files. ONLY MEANT FOR ADVANCED USERS. Can \
                     provide path to the folder containing all h5 \
                     files or a comma separated list of file paths \
                     like [file1,file2,file3]")
    parser.add('--intervals_file', required=False,
               type=type_or_none_fn(str),
               help="ONLY RELEVANT IF USING --denoise_h5_files option.\
                     Provide the path to intervals file that was used \
                     to generate the h5 files.")


def add_eval_options(parser):
    """Add evaluation options to the parser.

    Args:
        parser : parser to add the arguments to.

    Return:
        parser : After adding the arguments.

    """
    add_inference_options(parser)

    parser.add('--cleanbw', type=type_or_none_fn(str),
               help='Path to bigwig file containing clean \
                     (high-coverage/high-quality) ATAC-seq signal.')
    parser.add('--cleanpeakfile', type=type_or_none_fn(str),
               help='Path to narrowPeak or BED file containing peak calls '
                    'from MACS2 on the clean (high-coverage/high-quality) \
                     ATAC-seq signal.')
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

    if args.mode == "train":
        if not (args.val_chrom or args.holdout_chrom):
            if args.train_h5_files or args.val_h5_files:
                check_dependence(args.train_h5_files,
                                 args.val_h5_files,
                                 parser,
                                 "Specify both --train_h5_file and "
                                 "--val_h5_file.")
            else:
                parser.error("val_chrom and holdout_chrom are required for \
                              training.")
        check_dependence(args.cleanbw, args.cleanpeakfile, parser,
                         "cleanbw and cleanpeakfile are required for \
                          training")

    if args.mode == "eval":
        check_dependence(args.cleanbw, args.cleanpeakfile, parser,
                         "cleanbw and cleanpeakfile are required for \
                          eval")

    if not(args.distributed) and (args.gpu_idx is None):
        parser.error("Either specify which GPU to run atacworks on \
                through --gpu_idx, or pass the flag --distributed \
                to run on ALL available GPUs.")

    return args
