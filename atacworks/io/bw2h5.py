#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

r"""Read data from bigWig files and generate encoded data for training."""

# Import requirements

import logging
import os

from atacworks.dl4atac.utils import gather_key_files_from_cmdline
from atacworks.io.bedio import read_intervals
from atacworks.io.bigwigio import (check_bigwig_intervals_nonzero,
                                   extract_bigwig_intervals)
from atacworks.io.h5io import dict_to_h5

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


def bw2h5(noisybw, cleanbw, layersbw, cleanpeakbw, batch_size,
          nonzero, intervals_file, out_dir, prefix, pad):
    """Convert bigwig files to h5.

    Args:
        noisybw: BigWig file containing noisy data.
        cleanbw: BigWig file containing clean data.
        layersbw: BigWig file containing layers data.
        cleanpeakbw: BigWig file containing clean peaks data to be used as
        labels for training.
        batch_size: Number of lines to read at a time, since all of the data
        does not fit in memory.
        nonzero: Only save intervals that have non-zero values.
        intervals_file: File containing the intervals corresponding to the
        training,
        validation or inference.
        out_dir: Directory to save the output files to.
        prefix: Prefix to attach to the name, to make the files unique.
        pad: Padding values.

    Returns:
        Path to output files.

    """
    # Read intervals
    _logger.info('Reading intervals')
    intervals = read_intervals(intervals_file)
    _logger.info('Read {} intervals'.format(len(intervals)))

    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Optionally, select intervals with nonzero coverage
    if nonzero:
        _logger.info('Selecting intervals with nonzero coverage')
        nonzero_intervals = check_bigwig_intervals_nonzero(
            intervals, noisybw)
        _logger.info("Retaining {} of {} nonzero noisy intervals".format(
            sum(nonzero_intervals), len(intervals)))
        intervals = intervals[nonzero_intervals]

    _logger.debug('Collecting %d intervals' % len(intervals))

    # Calculate number of batches
    batches_per_epoch = int(np.ceil(len(intervals) / batch_size))
    _logger.info('Writing data in ' + str(batches_per_epoch) + ' batches.')

    # Split intervals into batches
    batch_starts = np.array(range(0, len(intervals), batch_size))
    batch_ends = batch_starts + batch_size
    batch_ends[-1] = len(intervals)

    # Get output hdf5 filename
    output_file_path = os.path.join(out_dir, prefix + '.h5')

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
            batch_intervals, noisybw, pad=pad
        )

        # Add other input layers
        if layersbw is not None:
            # Read additional layers
            layers = gather_key_files_from_cmdline(layersbw, extension='.bw')
            for key in layers.keys():
                batch_data[key] = extract_bigwig_intervals(
                    batch_intervals, layers[key], pad=pad
                )

        # Add labels
        if cleanbw and cleanpeakbw:
            # Read clean data: regression labels
            batch_data['label_reg'] = extract_bigwig_intervals(
                batch_intervals, cleanbw, pad=pad
            )

            # Read clean data: classification labels
            batch_data['label_cla'] = extract_bigwig_intervals(
                batch_intervals, cleanpeakbw, pad=pad
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
    return output_file_path
