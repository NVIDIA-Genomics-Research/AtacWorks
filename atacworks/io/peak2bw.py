#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

r"""Script to generate peak labels from MACS2 output or BED file."""

import logging
import os

from atacworks.io.bedgraphio import df_to_bedGraph
from atacworks.io.bedio import read_intervals, read_sizes
from atacworks.io.bigwigio import bedgraph_to_bigwig

# Set up logging
log_formatter = logging.Formatter(
    '%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('AtacWorks-peak2bw')
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)


def peak2bw(input_file, sizesfile, out_dir):
    """Convert peak files to bigwig.

    Args:
        input_file: Clean peak file to be converted to bigwig. Needs to be
        either bed or narrowPeak file.
        sizesfile: BED file containing chromosome sizes.
        out_dir: Directory to save the outputs to.

    Returns:
        Path to the output file.

    """
    # Create the output folder
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Set name for output file
    prefix = os.path.basename(input_file)
    out_bg_name = os.path.join(out_dir, prefix + '.bedGraph')

    out_bw_name = os.path.join(out_dir, prefix + '.bw')
    # Read input files
    _logger.info('Reading input file')
    # Skip first line if the file is narrowPeak
    skip = False
    if input_file.endswith("narrowPeak"):
        skip = True
    peaks = read_intervals(input_file, skip=skip)
    _logger.info('Read ' + str(len(peaks)) + ' peaks.')
    sizes = read_sizes(sizesfile)

    # Add score of 1 for all peaks
    _logger.info('Adding score')
    peaks['score'] = 1

    # Write bedGraph
    _logger.info('Writing peaks to bedGraph file')

    # Note: peaks will be subset to chromosomes in sizes file.
    df_to_bedGraph(peaks, out_bg_name, sizes)

    # Write bigWig and delete bedGraph
    _logger.info('Writing peaks to bigWig file {}'.format(out_bw_name))
    bedgraph_to_bigwig(out_bg_name, sizesfile,
                       deletebg=True, sort=True)

    _logger.info('Done!')

    return out_bw_name
