#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""Read peak bigWig file and produce BED file with scored peaks and summits.

Workflow:
    1. Read a bigWig file (produced by main.py) containing peak labels
    at each position
    2. Collapses peaks to BED format using bigWigToBedGraph
    3. For each peak, calculates the summit location, max score and
    average score

Output:
    BED file containing scored peaks and summits

Example:
    python peaksummary.py --peakbw peaks.bw --trackbw tracks.bw --out_dir ./

"""

# Import requirements
import argparse
import logging
import os
import subprocess

from atacworks.io.bedio import df_to_bed, read_intervals
from atacworks.io.bigwigio import extract_bigwig_intervals

import numpy as np

# Set up logging
log_formatter = logging.Formatter(
    '%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('AtacWorks-peaksummary')
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
    parser.add_argument('--peakbw', type=str,
                        help='Path to bigwig file with predicted peak labels',
                        required=True)
    parser.add_argument('--trackbw', type=str,
                        help='Path to bigwig file with predicted \
                        coverage track', required=True)
    parser.add_argument('--out_dir', type=str,
                        help='directory to save output file.',
                        required=True)
    parser.add_argument('--prefix', type=str,
                        help='output file prefix. \
                        Output file will be saved as prefix.bed. \
                        If not supplied, the output file will be \
                        summarized_peaks.bed')
    parser.add_argument('--minlen', type=int, help='minimum peak length',
                        default=20)
    args = parser.parse_args()
    return args


args = parse_args()

# Output file names
if args.prefix is None:
    prefix = 'summarized_peaks'
else:
    prefix = args.prefix
out_bed_path = os.path.join(args.out_dir, prefix + '.bed')
out_bg_path = os.path.join(args.out_dir, prefix + '.bedGraph')

# Collapse peaks
_logger.info('Writing peaks to bedGraph file {}'.format(out_bg_path))
subprocess.call(['bigWigToBedGraph', args.peakbw, out_bg_path])

# Read collapsed peaks
_logger.info('Reading peaks')
peaks = read_intervals(out_bg_path)
peaks.columns = ['#chrom', 'start', 'end']

# Add length of peaks
_logger.info('Calculating peak statistics')
peaks['len'] = peaks['end'] - peaks['start']

# Extract scores in peaks
peakscores = extract_bigwig_intervals(peaks, args.trackbw, stack=False)

# Add mean score in peak
peaks['mean'] = peakscores.apply(np.mean)

# Add max score
peaks['max'] = peakscores.apply(np.max)

# Add summit
# TODO: we might want to make this more complicated - if there
# are multiple positions with same value, pick the central one?
peaks['relativesummit'] = peakscores.apply(np.argmax)
peaks['summit'] = peaks['start'] + peaks['relativesummit']

# Discard peaks below minimum length
if args.minlen is not None:
    num_before_cut = len(peaks)
    peaks = peaks[peaks['len'] >= args.minlen]
    _logger.info("reduced number of peaks from {} to {}.".format(
        num_before_cut, len(peaks)))
# TODO: we may also want to merge small peaks together

# Write to BED
_logger.info('Writing peaks to BED file {}'.format(out_bed_path))
df_to_bed(peaks, out_bed_path, header=True)

# Delete bedGraph
_logger.info('Deleting bedGraph file')
subprocess.call(['rm', out_bg_path])
