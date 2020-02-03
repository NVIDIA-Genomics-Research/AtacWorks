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

"""Read data from bigWig files in intervals and generate batch data for model.

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

import logging

from claragenomics.io.bedio import read_intervals
from claragenomics.io.bigwigio import (check_bigwig_intervals_nonzero,
                                       extract_bigwig_intervals)
from claragenomics.io.h5io import dict_to_h5
from claragenomics.dl4atac.utils import gather_key_files_from_cmdline
import numpy as np


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
    """Parse command line arguments.

    Return:
        args : parsed argument object.

    """
    parser = argparse.ArgumentParser(
        description='Data processing for genome-wide denoising models.')
    parser.add_argument('--noisybw', type=str,
                        help='Path to noisy bigwig file', required=True)
    parser.add_argument('--layersbw', type=str,
                        help='Paths to bigWig files containing \
                            additional layers. If single file,  \
                            use format: "name:file". \
                            If there are multiple files, use format: \
                            "[name1:file1, name2:file2,...]"')
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
                        help='Path to clean bigwig file.\
                            Not used with --nolabel.')
    parser.add_argument('--cleanpeakbw', type=str,
                        help='Path to clean peak bigwig file.\
                            Not used with --nolabel.')
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
intervals = read_intervals(args.intervals)
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

# Get output hdf5 filename
filename = args.prefix + '.h5'

# Write batches to hdf5 file
_logger.info('Extracting data for each batch and writing to h5 file')
for i in range(batches_per_epoch):

    # Print current batch
    if i % 10 == 0:
        _logger.info("batch " + str(i) + " of " + str(batches_per_epoch))

    # Create dictionary to store data
    batch_data = {}

    # Subset intervals
    batch_intervals = intervals.iloc[batch_starts[i]:batch_ends[i], :]

    # Read noisy data
    batch_data['input'] = extract_bigwig_intervals(
        batch_intervals, args.noisybw, pad=args.pad
    )

    # Add other input layers
    if args.layersbw is not None:
        # Read additional layers
        layers = gather_key_files_from_cmdline(args.layersbw, extension='.bw')
        for key in layers.keys():
            batch_data[key] = extract_bigwig_intervals(
                batch_intervals, layers[key], pad=args.pad
            )

    # Add labels
    if not args.nolabel:

        # Read clean data: regression labels
        batch_data['label_reg'] = extract_bigwig_intervals(
            batch_intervals, args.cleanbw, pad=args.pad
        )

        # Read clean data: classification labels
        batch_data['label_cla'] = extract_bigwig_intervals(
            batch_intervals, args.cleanpeakbw, pad=args.pad
        )

    _logger.debug(len(batch_data))
    _logger.debug("Saving batch " + str(i) + " with keys " + str(
        batch_data.keys()))

    # Create dataset, or expand and append batch.
    if i == 0:
        dict_to_h5(batch_data, h5file=filename, create_new=True)
    else:
        dict_to_h5(batch_data, h5file=filename, create_new=False)

_logger.info('Done! Saved to %s' % filename)
