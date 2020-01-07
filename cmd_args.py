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


def check_dependence(
        arg1, arg2, parser, err_msg="Argument mutual inclusive test failed."):
    if arg1 and not arg2:
        parser.error(err_msg)


def check_mutual_exclusive(
        arg1, arg2, parser, err_msg="Argument mutual exclusive test failed."):
    if arg1 and arg2:
        parser.error(err_msg)


def parse_args():
    parser = configargparse.ArgParser(default_config_files=['configs/*.yaml'])
    parser.add('--config', required=False, is_config_file=True, help='config file path')

    # Learning args
    parser.add('--clip_grad', type=float, default=0.,
                        help='Grad clipping for bad/extreme batches')
    parser.add('--lr', type=float,
                        help='learning rate', default=0.0001)
    parser.add('--epochs', type=int,
                        help='Number of epochs', default=5)
    parser.add('--afunc', type=str,
                        help='activation', default='relu')
    parser.add('--mse_weight', type=float,
                        help='relative weight of mse loss', default=0.001)
    parser.add('--pearson_weight', type=float,
                        help='relative weight of pearson loss', default=1)
    parser.add_argument('--poisson_weight', type=float,
                        help='relative weight of poisson loss', default=0)
# =========================================================================================
    # experiment args
    parser.add('--label', type=str, default='AtacWorks',
                        help='label of the experiment; used for naming output folder')
    parser.add('--out_home', type=str, default='./Cache',
                        help='parent directory for the experiment folder')
    parser.add('--train', action='store_true',
                        help='training; preempt --infer')
    parser.add('--infer', action='store_true',
                        help='inference')
    parser.add('--resume', action='store_true',
                        help='resume training')
    parser.add('--eval', action='store_true',
                        help='evaluation: inference + result dumping + metrics evaluation')
    parser.add('--infer_files', type=str, default="",
                        help='list of data files in the form of [file1, file2, ...];'
                        'or a single path to a folder of files')
    parser.add('--intervals_file', type=str, default="",
                        help='bed file containing the chr and interval values for inference')
    parser.add('--sizes_file', type=str, default="",
                        help='bed file containing the chr and max size values')
    parser.add('--infer_threshold', type=float,
                        help='threshold the output peaks to this value')
    parser.add('--reg_rounding', type=int, default=0,
                        help='rounding values for regression outputs')
    parser.add('--cla_rounding', type=int, default=3,
                        help='rounding values for classification outputs')
    parser.add('--batches_per_worker', type=int, default=16,
                        help='number of batches to run per worker during multiprocessing')
    parser.add('--gen_bigwig', action='store_true',
                        help='save the inference output to bigiwig in addition to bedgraph')
    parser.add('--weights_path', type=str, default="",
                        help="checkpoint path to load the model from for inference or resume training")
    parser.add('--result_fname', type=str, default='infer_results.h5',
                        help='filename of the inference results')
    parser.add_argument('--deletebg', action='store_true',
                        help='delete output bedGraph file')

    # training args
    parser.add('--task', default='regression', choices=['regression', 'classification', 'both'],
                        help='Task can be regression or classification or both. (default: %(default)s)')
    parser.add('--train_files', type=str, default="",
                        help='list of data files in the form of [file1, file2, ...];'
                        'or a single path to a folder of files')
    parser.add('--print_freq', type=int, default=10,
                        help="Logging frequency")
    parser.add('--bs', type=int, default=32,
                        help="batch_size")
    parser.add('--num_workers', type=int, default=4,
                        help="number of workers for dataloader")
    parser.add('--checkpoint_fname', type=str, default="",
                        help="checkpoint filename to save the model")
    parser.add('--save_freq', type=int, default=5,
                        help="model checkpoint saving frequency")

    # Dataset args
    parser.add('--pad', type=int, help="Padding around intervals")
    parser.add('--transform', default='none', choices=['log', 'none'],
                        help='transformation to apply to coverage tracks before training')

    # validation args
    parser.add('--val_files', type=str, default="",
                        help='list of data files in the form of [file1, file2, ...];'
                        'or a single path to a folder of files')
    parser.add('--eval_freq', type=int, default=2,
                        help="evaluation frequency")
    parser.add('--threshold', type=float, default=0.5,
                        help="threshold for classification metrics")
    parser.add_argument('--best_metric_choice', type=str, default="AUROC", choices=['BCE', 'MSE', 'Recall', 'Specificity', 'CorrCoef', 'AUROC'],
                        help="metric to be considered for best metric. Choice is case sensitive.")

    # dist-env args
    parser.add('--gpu', default=0, type=int,
                        help='GPU id to use; preempted by --distributed which uses all available gpus ')
    parser.add('--distributed', action='store_true',
                        help='Do distributed training across all available gpus on the node')
    parser.add('--dist-url', default='tcp://127.0.0.1:4321', type=str,
                        help='url used to set up distributed training')
    parser.add('--dist-backend', default='gloo', type=str,
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
    # check_mutual_exclusive(args.infer, args.distributed, parser, "--infer and --distributed are mutual exclusive")

    check_dependence(args.train, args.train_files,  parser,
                     "--train requires --train_files")
    check_dependence(args.train, args.val_files,    parser,
                     "--train requires --val_files")
    check_dependence(args.train, args.checkpoint_fname, parser,
                     "--train requires --checkpoint_fname")

    check_dependence(args.infer, args.infer_files,  parser,
                     "--infer requires --infer_files")
    check_dependence(args.infer, args.weights_path, parser,
                     "--infer requires --weights_path")

    check_dependence(args.eval, args.val_files,  parser,
                     "--eval requires --val_files")
    check_dependence(args.eval, args.weights_path, parser,
                     "--eval requires --weights_path")

    return args
