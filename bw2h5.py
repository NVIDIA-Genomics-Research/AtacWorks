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
bw2h5.py:
    Reads data from bigWig files in intervals and generates batch data for model.

Workflow:
    1. Reads a BED file containing genomic intervals
    2. Takes as input a bigWig file containing noisy ATAC-Seq data
    3. Optionally, selects intervals with nonzero coverage in the noisy data
    4. Splits intervals into batches of given size
    5. Writes each batch to an hdf5 file
    6. Optionally, includes clean data and clean peaks

Output:
    h5 file containing data for training or testing DL model

Examples:
    Training:
        python bw2h5.py --noisybw noisy.bw --intervals training_intervals.bed
            --batch_size 120 --prefix training_data
            --cleanbw clean.bw --cleanpeakbw clean.narrowPeak.bw --nonzero
    Validation/Testing:
        python bw2h5.py --noisybw noisy.bw --intervals validation_intervals.bed
            --batch_size 120 --prefix validation_data
            --cleanbw clean.bw --cleanpeakbw clean.narrowPeak.bw

"""
# Import requirements
import argparse
import numpy as np
import pyBigWig
import pandas as pd
import logging
import h5py
from claragenomics.io.bigwigio import extract_bigwig_to_numpy, extract_bigwig_intervals, check_bigwig_nonzero, check_bigwig_intervals_nonzero


# Set up logging
log_formatter = logging.Formatter(
    '%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('AtacWorks-bw2h5')
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Data processing for genome-wide denoising models.')
    parser.add_argument('--noisybw', type=str,
                        help='Path to noisy bigwig file', required=True)
    parser.add_argument('--intervals', type=str,
                        help='Path to interval file', required=True)
    parser.add_argument('--batch_size', type=int,
                        help='batch size', required=True)
    parser.add_argument('--pad', type=int, help='padding around interval')
    parser.add_argument('--prefix', type=str,
                        help='output file prefix', required=True)
    parser.add_argument('--nolabel', action='store_true',
                        help='only saving noisy data')
    parser.add_argument('--cleanbw', type=str,
                        help='Path to clean bigwig file. Not used with --predict-only.')
    parser.add_argument('--cleanpeakbw', type=str,
                        help='Path to clean peak bigwig file. Not used with --predict-only.')
    parser.add_argument('--nonzero', action='store_true',
                        help='subset to intervals with nonzero coverage')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug prints')
    args = parser.parse_args()
    return args


args = parse_args()

if args.debug:
    _handler.setLevel(logging.DEBUG)
    _logger.setLevel(logging.DEBUG)

_logger.debug(args)

# Read intervals
_logger.info('Reading intervals')
intervals = pd.read_csv(args.intervals, header=None, sep='\t')
_logger.info('Read {} intervals'.format(len(intervals)))

# Optionally, select intervals with nonzero coverage
if args.nonzero:
    _logger.info('Selecting intervals with nonzero coverage')
    nonzero_intervals = check_bigwig_intervals_nonzero(
        intervals, args.noisybw)
    _logger.info("Retaining {} of {} nonzero noisy intervals".format(
        sum(nonzero_intervals), len(intervals)))
    intervals = intervals[nonzero_intervals]

_logger.debug('Collecting %d intervals' % len(intervals))

# Calculate number of batches
batches_per_epoch = int(np.ceil(len(intervals) / args.batch_size))
_logger.info('Writing data in ' + str(batches_per_epoch) + ' batches.')

# Split intervals into batches
batch_starts = np.array(range(0, len(intervals), args.batch_size))
batch_ends = batch_starts + args.batch_size
batch_ends[-1] = len(intervals)

# Write batches to hdf5 file
_logger.info('Extracting data for each batch and writing to h5 file')
df = None
filename = args.prefix + '.h5'
with h5py.File(filename, 'w') as f:
    # Create a single dataset -- expand locations as we go along
    for i in range(batches_per_epoch):
        if i % 10 == 0:
            _logger.info("batch: " + str(i) + " of " + str(batches_per_epoch))

        # Subset intervals
        batch_intervals = intervals.iloc[batch_starts[i]:batch_ends[i], :]

        # Read noisy data
        batch_data = extract_bigwig_intervals(
            batch_intervals, args.noisybw, pad=args.pad)

        if not args.nolabel:

            # Read clean data: regression labels
            clean_data = extract_bigwig_intervals(
                batch_intervals, args.cleanbw, pad=args.pad
            )

            # Read clean data: classification labels
            clean_peak_data = extract_bigwig_intervals(
                batch_intervals, args.cleanpeakbw, pad=args.pad
            )

            batch_data = np.dstack([batch_data, clean_data, clean_peak_data])

        _logger.debug(batch_data.shape)

        batch_name = 'batch' + str(i)
        _logger.debug('Saving batch: |%s|' % batch_name)

        # Create dataset, or expand and append batch.
        # TODO: Write as named numpy fields
        if df == None:
            df = f.create_dataset("data", data=batch_data, maxshape=(
                None, batch_data.shape[1], batch_data.shape[2]), compression='lzf')
            _logger.debug('Created new dataset! Shape %d -- file %s' %
                          (batch_data.shape[0], filename))
        else:
            df = f["data"]
            d_len = df.shape[0]
            df.resize(
                (d_len+batch_data.shape[0], batch_data.shape[1], batch_data.shape[2]))
            df[d_len:] = batch_data
            _logger.debug('expanded HDF from %d to %d' % (d_len, df.shape[0]))

_logger.info('Done! Saved to %s' % filename)
