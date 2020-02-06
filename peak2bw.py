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

"""Script to generate peak labels from MACS output or BED file.

Workflow:
    1. Reads narrowPeak file generated by MACS or BED file
    2. Subsets to given chromosomes
    3. Adds a score of 1 for all bases in peaks
    4. Writes to bedGraph file
    5. Converts bedGraph to bigWig file using bedGraphToBigWig
    6. Deletes bedGraph

Output:
    bigWig file containing score of 1 at peak positions

"""

import argparse
import logging

from claragenomics.io.bedgraphio import df_to_bedGraph
from claragenomics.io.bigwigio import bedgraph_to_bigwig
from claragenomics.io.bedio import read_intervals, read_sizes


# Set up logging
log_formatter = logging.Formatter(
    '%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('AtacWorks-peak2bw')
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
        description='Convert BED file or MACS narrowPeak file to bigWig file')
    parser.add_argument('input', type=str, help='Path to narrowPeak file')
    parser.add_argument('sizes', type=str,
                        help='Path to chromosome sizes file')
    parser.add_argument('--prefix', type=str, help='Output file prefix')
    parser.add_argument('--skip', type=int, default=0,
                        help='Rows of input file to skip - set to 1 \
                        for narrowPeak or BED file with header')
    args = parser.parse_args()
    return args


def main():
    """Convert peak files to bigwig."""
    args = parse_args()

    # Read input files
    _logger.info('Reading input files')
    peaks = read_intervals(args.input, skip=args.skip)
    sizes = read_sizes(args.sizes)

    # Add score of 1 for all peaks
    _logger.info('Adding score')
    peaks['score'] = 1

    # Set prefix for output files
    if args.prefix is None:
        # Output file gets name from input
        prefix = args.input
    else:
        prefix = args.prefix

    # Write bedGraph
    _logger.info('Writing peaks to bedGraph file')
    # Note: peaks will be subset to chromosomes in sizes file.
    df_to_bedGraph(peaks, prefix + '.bedGraph', sizes)

    # Write bigWig and delete bedGraph
    _logger.info('Writing peaks to bigWig file {}'.format(prefix + '.bw'))
    bedgraph_to_bigwig(prefix + '.bedGraph', args.sizes,
                       deletebg=True, sort=True)
    _logger.info('Done!')


if __name__ == "__main__":
    main()
