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

"""
postprocess.py:
    Postprocessing of denoised data for genome browser visualization.

Workflow:
    1. Reads hdf5 file containing model predictions
    2. Reads intervals from BED file
    3. Writes intervals and predicted scores to bedGraph file
    4. Converts bedGraph to bigWig using bedGraphToBigWig

Output:
    bigWig file containing model predictions.

Note: Intervals must be non-overlapping and sorted.

TODO:
    1. Delete bedGraph.
    2. Return error if bedGraph file exists.
"""
# import requirements


import argparse
import logging
import subprocess
import numpy as np
import pandas as pd
import h5py
from claragenomics.io.bedgraphio import expand_interval, intervals_to_bg, df_to_bedGraph
from claragenomics.io.bigwigio import bedgraph_to_bigwig
from claragenomics.dl4atac.train.utils import safe_make_dir

import time
import os
import math
import multiprocessing as mp
import glob
import shutil
import tempfile

# Set up logging
log_formatter = \
    logging.Formatter('%(levelname)s:%(asctime)s:%(name)s] %(message)s')
logger = logging.getLogger('AtacWorks-postprocess')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(log_formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='AtacWorks postprocessing script.')
    parser.add_argument('intervals_file', type=str,
                        help='Path to training intervals file')
    parser.add_argument('predictions_file', type=str,
                        help='Path to hdf5 file containing inferred data')
    parser.add_argument('sizes_file', type=str,
                        help='Path to chromosome sizes file')
    parser.add_argument('prefix', type=str,
                        help='Output files saved with this prefix')
    parser.add_argument('--threshold', type=float,
                        help='threshold for peak calling')
    parser.add_argument('--channel', type=int,
                        help='channel of predictions file to read')
    parser.add_argument('--round', type=int,
                        help='number of decimals to round predicted values')
    parser.add_argument('--num_worker', type=int, default=-1,
                        help='number of worker processes')
    args = parser.parse_args()
    return args


args = parse_args()

# Load intervals
logger.info('Loading intervals')
intervals = pd.read_csv(args.intervals_file, sep='\t', header=None)

# Get batch parameters
with h5py.File(args.predictions_file, 'r') as infile:
    num_batches = infile['data'].shape[0]
    interval_size = infile['data'].shape[1]

# TODO: assert that len(intervals) equals number of intervals in predictions file
# TODO: assert that size of intervals in intervals file equals size of intervals in predictions file

# Load predictions, convert to bedGraph and append for each batch
logger.info(
    'Writing scored intervals to bedGraph file {}.bedGraph'.format(args.prefix))

def writer(batch_range, outfilename):
    start, end = batch_range[0], batch_range[1]
    # num_batches = end - start
    with open(outfilename, 'w') as outfile:
        with h5py.File(args.predictions_file, 'r') as infile:
            # Load predictions
            if args.channel is not None:
                scores = infile['data'][start:end, :, args.channel]
            else:
                scores = np.array(infile['data'])

            # Flatten scores
            scores = scores.flatten()

            # Threshold predictions - for classification output
            if args.threshold is not None:
                scores = (scores > args.threshold).astype(int)
            # Round scores - for regression output
            elif args.round is not None:
                scores = scores.astype('float64')
                # Sometimes np.around doesn't work with float32. To investigate.
                scores = np.around(scores, decimals=args.round)

            # if the batch contains values > 0, write them
            if (scores > 0).any():
                # Select intervals corresponding to batch
                batch_intervals = intervals.iloc[start:end, :]
                # Expand each interval and combine with scores
                batch_bg = intervals_to_bg(batch_intervals, scores)

                # Write to file
                df_to_bedGraph(batch_bg, outfile)


def combiner(f1, f2):
    logger.debug("Combining {} and {}...".format(f1, f2))
    with open(f2, 'rb') as f:
        with open(f1, 'ab') as final:
            shutil.copyfileobj(f, final)
    os.remove(f2)


out_bedgraph = args.prefix + '.bedGraph'
# single process
if args.num_worker == 0: 
    writer([0, num_batches], out_bedgraph)
else: # multiprocessing
    if args.num_worker == -1: # spawn pool of processes
        num_cpus = mp.cpu_count()
        if num_cpus**2 > num_batches: 
            pool_size = int(math.sqrt(num_batches))
        else:
            pool_size = num_cpus
    else: # user specified # of processes
        pool_size = args.num_worker

    # writers dump temporary results
    ###############################################################################
    batches_per_process = num_batches // pool_size
    logger.debug("Launching {} writers.".format(pool_size))
    pool = mp.Pool(pool_size)

    args.tmp_dir = tempfile.mkdtemp()
    
    tmp_filenames = [os.path.join(args.tmp_dir, "{0:03}".format(i)) for i in range(pool_size)]
    tmp_batch_ranges = [[i*batches_per_process, (i+1)*batches_per_process] for i in range(pool_size)]
    tmp_batch_ranges[-1][1] = num_batches

    map_args = list(zip(tmp_batch_ranges, tmp_filenames))
    pool.starmap(writer, map_args)

    # combiners merge tmp results
    ###############################################################################
    start = time.time()
    # each combiner merges two tmp files until only 1 final file is left
    while True:
        files = sorted(glob.glob(os.path.join(args.tmp_dir, "*")))
        if len(files) == 1:
            break
        map_args = [(files[i*2], files[i*2+1]) for i in range(len(files)//2)]
        res = pool.starmap(combiner, map_args)

    pool.close()
    pool.join()
    logger.info("Time taken: {}s by combiners".format(time.time()-start))

    # move final file out of tmp folder
    shutil.move(os.path.join(args.tmp_dir, "000"), out_bedgraph)
    # remove tmp folder
    os.rmdir(args.tmp_dir)



# Convert bedGraph to bigWig
logger.info('Writing scored intervals to bigWig file {}'.format(args.prefix + '.bw'))
bedgraph_to_bigwig(out_bedgraph, args.sizes_file)

