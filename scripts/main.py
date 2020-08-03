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
"""Main script to run training, inference and evaluation."""
# system imports
import glob
import logging
import math
import os
import shutil
import sys
import tempfile
# python imports
import warnings

# module imports
from atacworks.dl4atac.utils import (Timers, assert_device_available,
                                     gather_files_from_cmdline,
                                     make_experiment_dir, save_config,
                                     get_intervals)
from atacworks.io.bedgraphio import df_to_bedGraph, intervals_to_bg
from atacworks.io.bedio import read_intervals, read_sizes
from atacworks.io.bigwigio import bedgraph_to_bigwig

from scripts.cmd_args import parse_args
from atacworks.io.peak2bw import peak2bw
from atacworks.io.bw2h5 import bw2h5

import h5py

import numpy as np

import pandas as pd

import torch

import torch.multiprocessing as mp

from scripts.worker import eval_worker, infer_worker, train_worker

warnings.filterwarnings("ignore")

# Set up logging
log_formatter = logging.Formatter(
    '%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('AtacWorks-main')
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)


def check_intervals(intervals_df, sizes_df, h5_file):
    """Check intervals file used for inference.

    Args:
        intervals_df(Pandas DataFrame): df with cols chrom, start, end
        sizes_df(Pandas dataframe): df with cols chrom, length
        h5_file(str): path to h5 file to match intervals

    """
    # Length of intervals == length of dataset in h5 file
    with h5py.File(h5_file, 'r') as hf:
        len_data = len(hf['input'])
    assert len_data == intervals_df.shape[0], \
        "Infer dataset size ({}) doesn't match the \
        intervals file ({})".format(len_data, intervals_df.shape[0])

    # Intervals do not cover chromosomes outside sizes file.
    interval_chroms = set(intervals_df['chrom'])
    sizes_chroms = set(sizes_df['chrom'])
    assert interval_chroms.issubset(sizes_chroms), \
        "Intervals file contains chromosomes not in sizes file ({})".format(
            interval_chroms.difference(sizes_chroms))

    # Interval bounds do not exceed chromosome lengths
    intervals_sizes = intervals_df.merge(sizes_df, on='chrom')
    excess_intervals = intervals_sizes[
        intervals_sizes['end'] > intervals_sizes['length']]
    assert len(excess_intervals) == 0, \
        "Intervals exceed chromosome sizes in sizes file ({})".format(
            excess_intervals)


def save_to_bedgraph(batch_range, item, task, channel, intervals,
                     outfile, rounding=None, threshold=None,
                     out_resolution=None):
    """Write out the tracks and peaks to bedGraphs.

    Args:
        batch_range : List containing start and end position of batch to write.
        item : Output from the queue.
        task: Whether the task is classification or regression or both.
        channel : Channel to be written out.
        intervals : pandas object containing inference intervals.
        outfile : The output file to write the output to.
        rounding : If not None, round the scores to given value.
        threshold : if not None, threhsold the scores to given value.
        out_resolution : resolution of output files

    """
    keys, batch = item
    start = batch_range[0]
    end = batch_range[1]
    if task == "both":
        scores = batch[start:end, :, channel]
    else:
        scores = batch[start:end, :, 0]

    # Round scores - for regression output
    if rounding is not None:
        scores = scores.astype('float64')
        # Sometimes np.around doesn't work with float32. To investigate.
        scores = np.around(scores, decimals=rounding)

    # Apply thresholding to peaks
    if threshold is not None and channel == 1:
        scores = (scores > threshold).astype(int)

    # if the batch contains values > 0, write them
    if (scores > 0).any():
        # Select intervals corresponding to batch
        batch_intervals = intervals.iloc[keys.numpy()[start:end], :].copy()
        # Add scores to each interval
        batch_intervals['scores'] = np.split(scores, scores.shape[0])
        batch_intervals['scores'] = [x[0] for x in batch_intervals['scores']]
        # Select intervals with scores>0
        batch_intervals = batch_intervals.loc[scores.sum(axis=1) > 0, :]

        # Expand each interval, combine with scores, and contract to smaller
        # intervals
        batch_bg = intervals_to_bg(batch_intervals, out_resolution)
        df_to_bedGraph(batch_bg, outfile)


def writer(infer, intervals_file, exp_dir,
           task, peaks, tracks, num_workers, infer_threshold,
           reg_rounding, batches_per_worker,
           gen_bigwig, sizes_file, res_queue, prefix, deletebg,
           out_resolution):
    """Write out the inference output to specified format.

    BedGraphs are generated by default.
    If bigwig is requested, then bedGraph files are converted to bigwig.

    Args:
        infer: Whether inferring or not. Writing only to be called to write
        inference output.
        intervals_file: Files containing the chromosome intervals.
        exp_dir: Experiment directory.
        task: Regression, classification or both.
        peaks: classification output.
        tracks: regression output.
        num_workers: Number of workers to use, for multi-processing
        infer_threshold: Value to threshold the inference output at.
        reg_rounding: Number of digits to round the regression output.
        batches_per_worker: If using multi processing, how many batches per
        worker.
        gen_bigwig: Whether to generate bigwig file
        sizes_file: The chromosome size file
        res_queue: Inference queue
        prefix: Prefix to use for output files
        deletebg: Delete bedgraph file after generating bigwig
        out_resolution: resolution of output files

    """
    # We only pass one file at a time to the writer as a list.
    if not infer:
        assert False, "writer called but infer = False."

    intervals = pd.read_csv(intervals_file, sep='\t', header=None,
                            names=['chrom', 'start', 'end'],
                            usecols=(0, 1, 2),
                            dtype={'chrom': str, 'start': int, 'end': int})

    channels = []
    # Suffix to give to the output
    result_fname = "infer"
    out_base_path = os.path.join(exp_dir, prefix + "_" + result_fname)

    if task == "both":
        if tracks and not peaks:
            channels = [0]
        elif peaks and not tracks:
            channels = [1]
        elif tracks and peaks:
            channels = [0, 1]
        # If both tracks and peaks are false, turn them to default true.
        elif not (tracks or peaks):
            channels = [0, 1]
    elif task == "classification":
        channels = [1]
    elif task == "regression":
        channels = [0]

    outfiles = [os.path.join(out_base_path + ".track.bedGraph"),
                os.path.join(out_base_path + ".peaks.bedGraph")]
    # Always round to 3 decimal digits. We will threshold it anyway.
    rounding = [reg_rounding, int(3)]

    # Temp dir used to save temp files during multiprocessing.
    temp_dir = tempfile.mkdtemp()
    for channel in channels:
        os.makedirs(os.path.join(temp_dir, str(channel)))

    batch_num = 0
    while True:
        if not res_queue.empty():
            batch_num = batch_num + 1
            item = res_queue.get()
            if item == 'done':
                break
            keys, batch = item
            # No multi processing
            if num_workers == 0:
                start = 0
                end = batch.shape[0]
                for channel in channels:
                    with open(outfiles[channel], "a+") as outfile:
                        save_to_bedgraph([start, end], item, task, channel,
                                         intervals, outfile,
                                         rounding=rounding[channel],
                                         threshold=infer_threshold,
                                         out_resolution=out_resolution)
            else:
                num_jobs = math.ceil(batch.shape[0] / batches_per_worker)
                pool_size = 0

                if num_workers == -1:  # spawn pool of processes
                    num_cpus = mp.cpu_count()
                    pool_size = min(num_jobs, num_cpus)
                else:  # user specified # of processes
                    pool_size = num_workers

                pool = mp.Pool(pool_size)
                tmp_batch_ranges = [[i * batches_per_worker,
                                     (i + 1) * batches_per_worker] for i in
                                    range(num_jobs)]
                # Force the last interval to capture remaining data.
                tmp_batch_ranges[-1][1] = batch.shape[0]
                all_intervals = [intervals] * len(tmp_batch_ranges)
                all_items = [item] * len(tmp_batch_ranges)
                for channel in channels:
                    temp_files = [os.path.join(
                        temp_dir, str(channel),
                        "{0:05}".format(num + batch_num)
                    ) for num in range(num_jobs)]
                    map_args = list(zip(tmp_batch_ranges, all_items,
                                        [task] * len(tmp_batch_ranges),
                                        [channel] * len(
                                            tmp_batch_ranges),
                                        all_intervals,
                                        temp_files,
                                        [rounding[channel]] * len(
                                            tmp_batch_ranges),
                                        [infer_threshold] * len(
                                            tmp_batch_ranges),
                                        [out_resolution] * len(
                                            tmp_batch_ranges)))

                    pool.starmap(save_to_bedgraph, map_args)

                    while True:
                        files = sorted(
                            glob.glob(os.path.join(temp_dir,
                                                   str(channel), "*")))
                        if len(files) == 1:
                            break
                        map_args = [(files[i * 2], files[i * 2 + 1])
                                    for i in range(len(files) // 2)]
                        pool.starmap(combiner, map_args)

                pool.close()
                pool.join()

    if num_workers != 0:
        for channel in channels:
            # Get last file in temp directory which has all the data
            files = glob.glob(os.path.join(temp_dir, str(channel), "*"))
            # Only one file should be left after the combiner stage
            assert (len(files) == 1)
            # move final files out of tmp folder
            shutil.copy(files[0], outfiles[channel])

    # remove tmp folder
    shutil.rmtree(temp_dir)

    if (gen_bigwig):
        print("Writing the output to bigwig files")
        for channel in channels:
            bedgraph_to_bigwig(
                outfiles[channel], sizes_file, prefix=None,
                deletebg=deletebg, sort=True)


def combiner(f1, f2):
    """Copy file f2 into f1 and delete f2.

    Args:
        f1 : File to write to.
        f2 : File to write from.

    """
    _logger.debug("Combining {} and {}...".format(f1, f2))
    with open(f2, 'rb') as f:
        with open(f1, 'ab') as final:
            shutil.copyfileobj(f, final)
    os.remove(f2)


def main():
    """Main."""
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), ".."))

    args = parse_args(root_dir)

    genomes = {"hg19": os.path.join(root_dir, "reference",
                                    "hg19.chrom.sizes"),
               "hg38": os.path.join(root_dir, "reference",
                                    "hg38.chrom.sizes")}
    if args.genome in genomes:
        args.genome = genomes[args.genome]

    # Set log level
    _logger.debug(args)

    # check gpu
    # TODO: add cpu support
    if not torch.cuda.is_available():
        raise Exception("No GPU available. Check your machine configuration.")

    # all output will be written in the exp_dir folder
    args.exp_dir = make_experiment_dir(
        args.exp_name, args.out_home, timestamp=True)

    # Convert layer names to a list
    if args.layers is not None:
        args.layers = args.layers.strip("[]").split(",")

    if args.seed is not None and args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    # train & resume
    ##########################################################################
    if args.mode == "train":

        # If h5 files are provided, load them.
        if args.train_h5_files is not None:
            args.train_files = gather_files_from_cmdline(
                args.train_h5_files,
                extension=".h5")
            args.val_files = gather_files_from_cmdline(
                args.val_h5_files,
                extension=".h5")

        # If h5 files not given, generate them.
        else:
            args.cleanpeakfile = gather_files_from_cmdline(
                args.cleanpeakfile,
                extension=(".bed", ".narrowPeak"))
            args.noisybw = gather_files_from_cmdline(args.noisybw,
                                                     extension=".bw")
            args.cleanbw = gather_files_from_cmdline(args.cleanbw,
                                                     extension=".bw")

            # We have to make sure there is a 1-1 correspondence between files.
            assert len(args.cleanpeakfile) == len(args.noisybw)
            assert len(args.cleanbw) == len(args.noisybw)

            train_files = []
            val_files = []
            for idx in range(len(args.cleanbw)):
                cleanbw = args.cleanbw[idx]
                noisybw = args.noisybw[idx]
                cleanpeakfile = args.cleanpeakfile[idx]
                # Read in the narrowPeak or BED files for clean data peak
                # labels, convert them to bigwig
                out_path = os.path.join(args.exp_dir, "bigwig_peakfiles")
                cleanpeakbw = peak2bw(cleanpeakfile, args.genome, out_path)
                # Generate training, validation, holdout intervals files
                out_path = os.path.join(args.exp_dir, "intervals")
                train_intervals, val_intervals, holdout_intervals = \
                    get_intervals(args.genome, args.interval_size,
                                  out_path,
                                  val=args.val_chrom,
                                  holdout=args.holdout_chrom,
                                  nonpeak=args.nonpeak,
                                  peakfile=cleanpeakbw)

                # Convert the input bigwig files and the clean peak files into
                # h5 for training.
                out_path = os.path.join(args.exp_dir, "bw2h5")
                nonzero = True
                prefix = os.path.basename(cleanbw) + ".train"
                train_file = bw2h5(noisybw, cleanbw, args.layersbw,
                                   cleanpeakbw, args.read_buffer,
                                   nonzero, train_intervals, out_path,
                                   prefix, args.pad)
                train_files.append(train_file)
                prefix = os.path.basename(cleanbw) + ".val"
                val_file = bw2h5(noisybw, cleanbw, args.layersbw, cleanpeakbw,
                                 args.read_buffer,
                                 nonzero, val_intervals, out_path,
                                 prefix, args.pad)
                val_files.append(val_file)

            args.train_files = train_files
            args.val_files = val_files
        _logger.debug("Training data:   " + "\n".join(args.train_files))
        _logger.debug("Validation data: " + "\n".join(args.val_files))

        # Get model parameters
        with h5py.File(args.train_files[0], 'r') as f:
            if args.pad is not None:
                args.interval_size = f['input'].shape[1] - 2 * args.pad
            else:
                args.interval_size = f['input'].shape[1]
            args.batch_size = 1

        ngpus_per_node = torch.cuda.device_count()
        # WAR: gloo distributed doesn't work if world size is 1.
        # This is fixed in newer torch version -
        # https://github.com/facebookincubator/gloo/issues/209
        if ngpus_per_node == 1:
            args.distributed = False
            args.gpu_idx = 0

        config_dir = os.path.join(args.exp_dir, "configs")
        if not os.path.exists(config_dir):
            os.mkdir(config_dir)
        if args.distributed:
            _logger.info('Distributing to %s GPUS' % str(ngpus_per_node))
            args.world_size = ngpus_per_node
            mp.spawn(train_worker, nprocs=ngpus_per_node,
                     args=(ngpus_per_node, args), join=True)
        else:
            assert_device_available(args.gpu_idx)
            _logger.info('Running on GPU: %s' % str(args.gpu_idx))
            args.world_size = 1
            train_worker(args.gpu_idx, ngpus_per_node, args, timers=Timers)

    # infer & eval
    ##########################################################################
    if args.mode == "denoise" or args.mode == "eval":

        files = []
        if args.denoise_h5_files is not None:
            files = gather_files_from_cmdline(args.denoise_h5_files,
                                              extension=".h5")
            infer_intervals = args.intervals_file
        else:
            cleanpeakbw = None
            if args.mode == "eval":
                # Read in the narrowPeak or BED files for clean data peak
                # labels, convert them to bigwig
                out_path = os.path.join(args.exp_dir, "bigwig_peakfiles")
                cleanpeakbw = peak2bw(args.cleanpeakfile, args.genome,
                                      out_path)

            # Generate training, validation, holdout intervals files
            out_path = os.path.join(args.exp_dir, "intervals")
            infer_intervals = get_intervals(args.genome,
                                            args.interval_size,
                                            out_path,
                                            peakfile=cleanpeakbw,
                                            regions=args.regions)

            # Convert the input bigiwg files and the clean peak files into h5
            # for training.
            args.noisybw = gather_files_from_cmdline(args.noisybw,
                                                     extension=".bw")

            for idx in range(len(args.noisybw)):
                out_path = os.path.join(args.exp_dir, "bw2h5")
                nonzero = False
                cleanbw = None
                noisybw = args.noisybw[idx]
                if args.mode == "eval":
                    cleanbw = args.cleanbw[idx]
                prefix = os.path.basename(noisybw) + "." + args.mode
                infer_file = bw2h5(noisybw, cleanbw, args.layersbw, None,
                                   args.read_buffer,
                                   nonzero, infer_intervals, out_path,
                                   prefix, args.pad)
                files.append(infer_file)

        for x in range(len(files)):
            infile = files[x]
            args.input_files = [infile]
            if args.mode == "denoise":
                _logger.debug("Inference data: ", infile)

                # Check that intervals, sizes and h5 file are all compatible.
                _logger.info('Checking input files for compatibility')
                intervals = read_intervals(infer_intervals)
                sizes = read_sizes(args.genome)
                check_intervals(intervals, sizes, infile)

                # Delete intervals and sizes objects in main thread
                del intervals
                del sizes
            else:
                _logger.debug("Evaluation data: ", infile)
            # Get model parameters
            with h5py.File(files[x], 'r') as f:
                if args.pad is not None:
                    args.interval_size = f['input'].shape[1] - 2 * args.pad
                else:
                    args.interval_size = f['input'].shape[1]
                args.batch_size = 1

            # Make sure that interval_size is a multiple of the out_resolution
            if args.out_resolution is not None:
                assert(args.interval_size % args.out_resolution == 0)

            prefix = os.path.basename(infile).split(".")[0]
            # setup queue and kick off writer process
            #############################################################
            manager = mp.Manager()
            res_queue = manager.Queue()
            # Create a keyword argument dictionary to pass into the
            # multiprocessor
            keyword_args = {"infer": args.mode == "denoise",
                            "intervals_file": infer_intervals,
                            "exp_dir": args.exp_dir,
                            "task": args.task,
                            "peaks": args.peaks,
                            "tracks": args.tracks,
                            "num_workers": args.num_workers,
                            "infer_threshold": args.threshold,
                            "reg_rounding": args.reg_rounding,
                            "batches_per_worker": args.batches_per_worker,
                            "gen_bigwig": args.gen_bigwig,
                            "sizes_file": args.genome,
                            "res_queue": res_queue, "prefix": prefix,
                            "deletebg": args.deletebg,
                            "out_resolution": args.out_resolution}
            write_proc = mp.Process(target=writer, kwargs=keyword_args)
            write_proc.start()
            #############################################################

            ngpus_per_node = torch.cuda.device_count()
            # WAR: gloo distributed doesn't work if world size is 1.
            # This is fixed in newer torch version -
            # https://github.com/facebookincubator/gloo/issues/209
            if ngpus_per_node == 1:
                args.distributed = False
                args.gpu_idx = 0

            worker = infer_worker if args.mode == "denoise" else eval_worker
            if args.distributed:
                args.world_size = ngpus_per_node
                mp.spawn(worker, nprocs=ngpus_per_node, args=(
                    ngpus_per_node, args, res_queue), join=True)
            else:
                assert_device_available(args.gpu_idx)
                args.world_size = 1
                worker(args.gpu_idx, ngpus_per_node, args, res_queue)

            # finish off writing
            #############################################################
            res_queue.put("done")
            _logger.info("Waiting for writer to finish...")
            write_proc.join()
            #############################################################
    # Save config parameters
    dst_config_path = os.path.join(args.out_home,
                                   args.mode + "_config.yaml")
    save_config(dst_config_path, args)


if __name__ == '__main__':
    main()
