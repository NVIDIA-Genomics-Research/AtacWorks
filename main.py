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

# system imports
import logging
import os
import random
import sys
import tempfile
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

# module imports
from claragenomics.dl4atac.models.models import *
from claragenomics.dl4atac.utils import *
from claragenomics.io.bedgraphio import intervals_to_bg, df_to_bedGraph
from claragenomics.io.bigwigio import bedgraph_to_bigwig
from cmd_args import parse_args
from worker import train_worker, infer_worker, eval_worker 

# python imports
import warnings
import glob
import math
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


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def check_intervals(intervals_df, sizes_df, h5_file):
    """
    Function to check intervals file used for inference.
    Args:
        intervals_df(Pandas DataFrame): df with cols chrom, start, end
        sizes_df(Pandas dataframe): df with cols chrom, len
        h5_file(str): path to h5 file to match intervals
    """
    # Length of intervals == length of dataset in h5 file
    with h5py.File(h5_file, 'r') as hf:
        len_data = len(hf['data'])
    assert len_data == intervals_df.shape[0], \
        "Size of infer dataset ({}) doesn't match the intervals file ({})".format(len_data, intervals_df.shape[0])

    # Intervals do not cover chromosomes outside sizes file.
    interval_chroms = set(intervals_df['chrom'])
    sizes_chroms = set(sizes_df['chrom'])
    assert interval_chroms.issubset(sizes_chroms), \
         "Intervals file contains chromosomes not in sizes file ({})".format(interval_chroms.difference(sizes_chroms))

    # Interval bounds do not exceed chromosome lengths
    intervals_sizes = intervals_df.merge(sizes_df, on='chrom')
    excess_intervals = intervals_sizes[intervals_sizes['end'] > intervals_sizes['len']]
    assert len(excess_intervals) == 0, \
          "Intervals exceed chromosome sizes in sizes file ({})".format(excess_intervals)


def save_to_bedgraph(batch_range, item, channel, intervals, outfile, rounding=None, threshold=None):
    """
    Function to write out the tracks and peaks to bedGraphs. 
    Args:
        batch_range : List containing start and end position of batch to write.
        item : Output from the queue.
        channel : Channel to be written out.
        intervals : pandas object containing inference intervals.
        outfile : The output file to write the output to.
        rounding : If not None, round the scores to given value.
        threshold : if not None, threhsold the scores to given value.
    """
    keys, batch = item
    start = batch_range[0]
    end = batch_range[1]
    scores = batch[start:end,:,channel]
    # Round scores - for regression output
    if rounding is not None:
        scores = scores.astype('float64')
        # Sometimes np.around doesn't work with float32. To investigate.
        scores = np.around(scores, decimals=rounding)

    # Apply thresholding only to peaks
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
        batch_intervals = batch_intervals.loc[scores.sum(axis=1)>0,:]
            
        # Expand each interval, combine with scores, and contract to smaller intervals
        batch_bg = intervals_to_bg(batch_intervals)
        df_to_bedGraph(batch_bg, outfile)


def writer(infer, intervals_file, exp_dir, result_fname,
           task, num_workers, infer_threshold, reg_rounding, cla_rounding,
           batches_per_worker, gen_bigwig, sizes_file, res_queue, prefix, deletebg):
    """
    Function to write out the inference output to specified format. BedGraphs are generated by default. 
    If bigwig is requested, then bedGraph files are converted to bigwig.
    Args:
        args : Object holding all command line arguments
        res_queue : Queue containing the inference results
        prefix : The prefix to add to the output inference files. Useful when multiple files are
        inferenced together.
        deletebg : Delete the output bedGraph after converting to bigWig
    """

    # We only pass one file at a time to the writer as a list.
    if not infer:
        assert False, "writer called but infer = False. Not sure what file to write?"

    intervals = pd.read_csv(intervals_file, sep='\t', header=None, names=['chrom', 'start', 'end'], 
         usecols=(0,1,2), dtype={'chrom':str, 'start':int, 'end':int})

    channels = []
    outputfiles = []
    out_base_path = os.path.join(exp_dir, prefix + "_" + result_fname)
    if task == "regression":
        channels = [0]
        outfiles = [os.path.join(out_base_path + ".track.bedGraph")]
        rounding = [reg_rounding]
    elif task == "classification":
        channels = [1]
        outfiles = [os.path.join(out_base_path + ".peaks.bedGraph")]
        rounding = [cla_rounding]
    elif task == "both":
        channels = [0, 1]
        outfiles = [os.path.join(out_base_path + ".track.bedGraph"),
                    os.path.join(out_base_path + ".peaks.bedGraph")]
        rounding = [reg_rounding, cla_rounding]

    # Temp dir used to save temp files during multiprocessing.
    temp_dir = tempfile.mkdtemp()
    for channel in channels:
        os.makedirs(os.path.join(temp_dir, str(channel)))

    batch_num = 0
    while 1:
        if not res_queue.empty():
            batch_num = batch_num + 1
            item = res_queue.get()
            if item == 'done':
                break
            keys, batch = item
            #No multi processing
            if num_workers == 0:
                start = 0
                end = batch.shape[0]
                for channel in channels:
                    with open(outfiles[channel], "a+") as outfile:
                        batch_bg = save_to_bedgraph([start, end], item, channel, intervals, outfile,
                                                    rounding=rounding[channel],
                                                    threshold=infer_threshold)
            else:
                num_jobs = math.ceil(batch.shape[0] / batches_per_worker)
                pool_size = 0

                if num_workers == -1: # spawn pool of processes
                    num_cpus = mp.cpu_count()
                    pool_size = min(num_jobs, num_cpus)
                else: # user specified # of processes
                    pool_size = num_workers

                pool = mp.Pool(pool_size)
                tmp_batch_ranges = [[i*batches_per_worker, (i+1)*batches_per_worker] for i in range(num_jobs)]
                # Force the last interval to capture remaining data.
                tmp_batch_ranges[-1][1] = batch.shape[0]
                all_intervals = [intervals]*len(tmp_batch_ranges)
                all_items = [item]*len(tmp_batch_ranges)
                for channel in channels:
                    temp_files = [os.path.join(temp_dir, str(channel), "{0:05}".format(num+batch_num)) for num in range(num_jobs)]
                    if infer_threshold is None:
                        map_args = list(zip(tmp_batch_ranges, all_items,
                                            [channel]*len(tmp_batch_ranges), all_intervals,
                                            temp_files, [rounding[channel]]*len(tmp_batch_ranges)))
                    else:
                        map_args = list(zip(tmp_batch_ranges, all_items,
                                            [channel]*len(tmp_batch_ranges), all_intervals,
                                            temp_files, [rounding[channel]]*len(tmp_batch_ranges),
                                            [infer_threshold]*len(tmp_batch_ranges)))

                    pool.starmap(save_to_bedgraph, map_args)

                    while True:
                        files = sorted(glob.glob(os.path.join(temp_dir, str(channel), "*")))
                        if len(files) == 1:
                            break
                        map_args = [(files[i*2], files[i*2+1]) for i in range(len(files)//2)]
                        res = pool.starmap(combiner, map_args)

                pool.close()
                pool.join()

    if num_workers != 0:
        for channel in channels:
            # Get last file in temp directory which has all the data
            files = glob.glob(os.path.join(temp_dir, str(channel), "*"))
            assert(len(files) == 1) # Only one file should be left after the combiner stage
            # move final files out of tmp folder
            shutil.move(files[0], outfiles[channel])

    # remove tmp folder
    shutil.rmtree(temp_dir)

    if (gen_bigwig):
        print ("Writing the output to bigwig files")
        for channel in channels:
            bedgraph_to_bigwig(outfiles[channel], sizes_file, prefix=None,deletebg=deletebg, sort=True)


def combiner(f1, f2):
    _logger.debug("Combining {} and {}...".format(f1, f2))
    with open(f2, 'rb') as f:
        with open(f1, 'ab') as final:
            shutil.copyfileobj(f, final)
    os.remove(f2)


def main():
    root_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    args = parse_args(root_dir)
    # Set log level
    if args.debug:
        _handler.setLevel(logging.DEBUG)
        _logger.setLevel(logging.DEBUG)

    _logger.debug(args)

    # Random seeds for reproducability.
    # set_random_seed(args.seed)

    # check gpu
    # TODO: add cpu support
    if not torch.cuda.is_available():
        raise Exception("No GPU available. Check your machine configuration.")

    # all output will be written in the exp_dir folder
    args.exp_dir = make_experiment_dir(
        args.label, args.out_home, timestamp=True)

    # train & resume
    ##########################################################################################
    if args.train or args.resume:
        args.train_files = gather_files_from_cmdline(args.train_files)
        args.val_files = gather_files_from_cmdline(args.val_files)
        _logger.debug("Training data:   " + "\n".join(args.train_files))
        _logger.debug("Validation data: " + "\n".join(args.val_files))

        # Get model parameters
        with h5py.File(args.train_files[0], 'r') as f:
            args.interval_size = f['data'].shape[1]
            args.batch_size = 1

        ngpus_per_node = torch.cuda.device_count()
        # WAR: gloo distributed doesn't work if world size is 1.
        # This is fixed in newer torch version - https://github.com/facebookincubator/gloo/issues/209
        args.distributed = False if ngpus_per_node == 1 else args.distributed

        _logger.info('Distributing to %s GPUS' % str(ngpus_per_node))

        if args.distributed:
            args.world_size = ngpus_per_node
            mp.spawn(train_worker, nprocs=ngpus_per_node,
                     args=(ngpus_per_node, args), join=True)
        else:
            assert_device_available(args.gpu)
            args.world_size = 1
            train_worker(args.gpu, ngpus_per_node, args, timers=Timers)


    # infer & eval
    ##########################################################################################
    if args.infer or args.eval:
        files = args.infer_files if args.infer else args.val_files
        files = gather_files_from_cmdline(files)
        for x in range(len(files)):
            infile = files[x]
            if args.infer:
                args.infer_files = [infile]
                _logger.debug("Inference data: ", args.infer_files)

                # Check that intervals, sizes and h5 file are all compatible.
                _logger.info('Checkng input files for compatibility')
                intervals = pd.read_csv(args.intervals_file, sep='\t', header=None, names=['chrom', 'start', 'end'],
                     usecols=(0,1,2), dtype={'chrom':str, 'start':int, 'end':int})
                sizes = pd.read_csv(args.sizes_file, sep='\t', header=None, names=['chrom', 'len'],
                     usecols=(0,1), dtype={'chrom':str, 'len':int}) 
                check_intervals(intervals, sizes, args.infer_files[0])

                # Delete intervals and sizes objects in main thread
                del intervals
                del sizes
            else:
                args.val_files = [infile]
                _logger.debug("Evaluation data: ", args.val_files)
            # Get model parameters
            with h5py.File(files[x], 'r') as f:
                args.interval_size = f['data'].shape[1]
                args.batch_size = 1

            prefix = os.path.basename(infile).split(".")[0]
            # setup queue and kick off writer process
            #############################################################
            manager = mp.Manager()
            res_queue = manager.Queue()
            # Create a keyword argument dictionary to pass into the multiprocessor
            keyword_args = {"infer" : args.infer, "intervals_file" : args.intervals_file, "exp_dir" : args.exp_dir,
                            "result_fname" : args.result_fname, "task" : args.task, "num_workers" : args.num_workers,
                            "infer_threshold" : args.infer_threshold, "reg_rounding" : args.reg_rounding,
                            "cla_rounding" : args.cla_rounding, "batches_per_worker" : args.batches_per_worker,
                            "gen_bigwig" : args.gen_bigwig, "sizes_file" : args.sizes_file,
                            "res_queue" : res_queue, "prefix" : prefix, "deletebg" : args.deletebg}
            write_proc = mp.Process(target=writer, kwargs=keyword_args)
            write_proc.start()
            #############################################################

            ngpus_per_node = torch.cuda.device_count()
            # WAR: gloo distributed doesn't work if world size is 1.
            # This is fixed in newer torch version - https://github.com/facebookincubator/gloo/issues/209
            args.distributed = False if ngpus_per_node == 1 else args.distributed

            worker = infer_worker if args.infer else eval_worker
            if args.distributed:
                args.world_size = ngpus_per_node
                mp.spawn(worker, nprocs=ngpus_per_node, args=(
                    ngpus_per_node, args, res_queue), join=True)
            else:
                assert_device_available(args.gpu)
                args.world_size = 1
                worker(args.gpu, ngpus_per_node, args, res_queue)

            # finish off writing
            #############################################################
            res_queue.put("done")
            _logger.info("Waiting for writer to finish...")
            write_proc.join()
            #############################################################
    # Save config parameters
    dst_config_path = os.path.join(args.out_home, "config_params.yaml")
    save_config(dst_config_path, args)

if __name__ == '__main__':
    main()
