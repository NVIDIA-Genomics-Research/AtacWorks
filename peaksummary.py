#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# Import requirements
import argparse
import numpy as np
import pyBigWig
import pandas as pd
import logging
import subprocess

from claragenomics.io.bigwigio import extract_bigwig_to_numpy, extract_bigwig_intervals


"""
peaksummary.py:
    Reads data from peak bigWig file and produces BED file with scored peaks and summits.
Workflow:
    1. Reads a bigWig file (produced by postprocess.py) containing peak labels at each position
    2. Collapses peaks to BED format using bigWigToBedGraph
    3. For each peak, calculates the summit location, max score and average score
Output:
    BED file containing scored peaks and summits
Example:
    python peaksummary.py --peakbw peaks.bw --scorebw scores.bw --prefix peaks
"""

# Set up logging
log_formatter = logging.Formatter(
    '%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('AtacWorks-peaksummary')
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)

# TODO: add optional command to filter by length of peak?
def parse_args():
    parser = argparse.ArgumentParser(
        description='Data processing for genome-wide denoising models.')
    parser.add_argument('--peakbw', type=str, help='Path to bigwig file with peak labels')
    parser.add_argument('--scorebw', type=str, help='Path to bigwig file with coverage track')
    parser.add_argument('--prefix', type=str, help='output file prefix')
    parser.add_argument('--minlen', type=int, help='minimum peak length')
    args = parser.parse_args()
    return args

args = parse_args()

# Collapse peaks
_logger.info('Writing peaks to bedGraph file {}.bedGraph'.format(args.prefix))
subprocess.call(['bigWigToBedGraph', args.peakbw, args.prefix + '.bedGraph'])

# Read collapsed peaks
_logger.info('Reading peaks')
peaks = pd.read_csv(args.prefix + '.bedGraph', header=None, sep='\t', usecols=(0,1,2))
peaks.columns = ['chrom', 'start', 'end']

# Add length of peaks
_logger.info('Calculating peak statistics')
peaks['len'] = peaks['end'] - peaks['start']

# Extract scores in peaks
peakscores = extract_bigwig_intervals(peaks, args.scorebw, stack=False)

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
peaks = peaks[peaks['len'] >= args.minlen]

# Write to BED
_logger.info('Writing peaks to BED file {}.bed'.format(args.prefix))
peaks.to_csv(args.prefix + '.bed', sep='\t', index=None)

# Delete bedGraph
_logger.info('Deleting bedGraph file')
subprocess.call(['rm', args.prefix + '.bedGraph'])
