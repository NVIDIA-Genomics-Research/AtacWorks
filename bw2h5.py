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

r"""Read data from bigWig files in intervals and generate batch data for model.

Workflow:
    1. Reads a BED file containing genomic intervals
    2. Takes as input a bigWig file containing noisy ATAC-Seq data
    3. Optionally, selects intervals with nonzero coverage in the noisy data
    4. Splits intervals into batches of given size
    5. Reads noisy ATAC-seq data in these intervals from the bigWig file
    6. Optionally, includes other layers of input in addition to the noisy
       ATAC-seq data
    7. Writes each batch to an hdf5 file
    8. Optionally, includes clean data (regression labels) and clean peaks
       (classification labels)

Output:
    .h5 file containing data for training or testing a DL model

Examples:
    Training:
        python bw2h5.py --noisybw noisy.bw \
            --intervals training_intervals.bed \
            --out_home ./ --prefix training_data \
            --cleanbw clean.bw --cleanpeakbw clean.narrowPeak.bw --nonzero
    Validation:
        python bw2h5.py --noisybw noisy.bw \
            --intervals validation_intervals.bed \
            --out_home ./ --prefix validation_data \
            --cleanbw clean.bw --cleanpeakbw clean.narrowPeak.bw
    Inference:
        python bw2h5.py --noisybw noisy.bw --intervals test_intervals.bed \
            --out_home ./ --prefix test_data \
            --nolabel

"""
# Import requirements

import argparse

import logging

from claragenomics.dl4atac.utils import gather_key_files_from_cmdline
from claragenomics.io.bedio import read_intervals
from claragenomics.io.bigwigio import (check_bigwig_intervals_nonzero,
                                       extract_bigwig_intervals)
from claragenomics.io.h5io import dict_to_h5

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
                        help='Path to bigwig file containing noisy \
                        (low coverage/low quality) ATAC-seq signal',
                        required=True)
    parser.add_argument('--layersbw', type=str,
                        help='Paths to bigWig files containing \
                            additional layers. If single file,  \
                            use format: "name:file". \
                            If there are multiple files, use format: \
                            "[name1:file1, name2:file2,...]"')
    parser.add_argument('--intervals', type=str,
                        help='Path to BED file containing genomic intervals. \
                        ATAC-seq data within these intervals will be read \
                        from the bigWig files. See get_intervals.py for \
                        help generating such a BED file.',
                        required=True)
    parser.add_argument('--batch_size', type=int,
                        help='batch size; number of intervals to read from bigWig \
                        files at a time. Unrelated to training/inference \
                        batch size.', default=1000)
    parser.add_argument('--pad', type=int, help='Number of additional bases to \
                        add as padding on either side of interval. Use the \
                        same value for training, validation and test files.')
    parser.add_argument('--out_home', type=str,
                        help='directory to save output file.', required=True)
    parser.add_argument('--prefix', type=str,
                        help='output file prefix. The output file will be saved \
                        with the name prefix.h5', required=True)
    parser.add_argument('--nolabel', action='store_true',
                        help='only saving noisy ATAC-seq data')
    parser.add_argument('--cleanbw', type=str,
                        help='Path to bigwig file containing clean \
                        (high-coverage/high-quality) ATAC-seq signal.\
                            Not used with --nolabel.')
    parser.add_argument('--cleanpeakbw', type=str,
                        help='Path to bigwig file containing peak calls from \
                        clean (high-coverage/high-quality) ATAC-seq signal.\
                        Use peak2bw.py to generate this file. \
                        Not used with --nolabel.')
    parser.add_argument('--nonzero', action='store_true',
                        help='Only save intervals with nonzero coverage. \
                        Recommended when encoding training data, as intervals \
                        with zero coverage do not help the model to learn.')
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
output_file_path = args.out_home + '/' + args.prefix + '.h5'

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
        dict_to_h5(batch_data, h5file=output_file_path, create_new=True)
    else:
        dict_to_h5(batch_data, h5file=output_file_path, create_new=False)

_logger.info('Done! Saved to %s' % output_file_path)
