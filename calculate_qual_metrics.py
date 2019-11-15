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
calculate_baseline_metrics.py:
    Takes noisy data and ground truth, returns metrics
Workflow:
   1. Read labels from .h5 file
   2. Read noisy data from .h5 or .bw file
   2. Calculate metrics
   3. If required, classification metrics at multiple thresholds.
"""

# Import requirements
import argparse
import numpy as np
import h5py
import torch
import pyBigWig
import pandas as pd
import os
import logging

from claragenomics.io.bigwigio import extract_bigwig_intervals

# Set up logging
log_formatter = logging.Formatter(
    '%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('AtacWorks-qual-metrics')
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)

def flip_negative(x, col):
    if x['strand']=='-':
        return np.flip(x[col],0)
    else:
        return x[col]

def parse_args():
    parser = argparse.ArgumentParser(
        description='AtacWorks script to calculate additional signal-quality metrics from a bigWig file.')
    parser.add_argument('--bwfile', type=str,
                        help='Path to bw file containing track.')
    parser.add_argument('--peak_file', type=str,
                        help='Path to bed or narrowPeak file containing peak calls.')
    parser.add_argument('--clean_peak_file', type=str,
                        help='Path to bed or narrowPeak file containing peak calls based on clean data.')
    parser.add_argument('--tss_file', type=str,
                        help='Path to bed file containing tss sites with 1st 4 columns: chr, start, end, strand')
    parser.add_argument('--dhs_file', type=str,
                        help='Path to bed file containing dhs sites')
    parser.add_argument('--out_file', type=str,
                        help='Output file for TSS enrichment profile')
    parser.add_argument('--intervals', type=str,
                        help='bed file of intervals to evaluate on.')
    parser.add_argument('--sizes', type=str,
                        help='chromosome sizes file to evaluate on.')
    args = parser.parse_args()
    return args


args = parse_args()

#Load sizes
sizes = pd.read_csv(args.sizes, header=None, sep='\t', usecols=(0,1))

# Load intervals
_logger.info('Loading intervals')
if args.intervals is not None:
	intervals = pd.read_csv(args.intervals, header=None, sep='\t', usecols=(0,1,2))
# If no intervals are supplied, evaluate on full length of chromosomes provided.
else:
	intervals = sizes.copy()
    # Convert to interval format - add a column of zeros for start value
    intervals[2] = [0]*len(intervals)
    intervals.rename(columns={0: 0, 2: 1, 1: 2}, inplace=True)


# TSS enrichment

if args.tss is not None:
    #Read TSS
    tss = pd.read_csv(args.tss_file, sep='\t', header=None, usecols=(0,1,2,3))

    if args.output_file is not None:
        # Define TSS region
        tss[1] = tss[1] - 2000
        tss[2] = tss[2] + 2000

        # Extract coverage
        tss['coverage'] = extract_bigwig_intervals(tss, args.bw_file)
        tss['flip'] = tss.apply(flip_negative, axis=1, args=('coverage',))
        signal_near_tss = np.sum(np.stack(tss['flip']), axis=0)[:-1]
        enr_tss = signal_near_tss/signal_near_tss[1:200].mean()
        np.save(args.out_file, enr_tss)

        # Get TSS score
        signal_near_tss = signal_near_tss[1000:3000]
    else:
        tss[1] = tss[1] - 1000
        tss[2] = tss[2] + 1000
        signal_near_tss = extract_bigwig_intervals(tss, args.bw_file, stack=True)
        signal_near_tss = np.sum(signal_near_tss)

    # TSS score
    tss_score = np.mean(signal_near_tss)*2/(signal_near_tss[:100] + signal_near_tss[1900:])

#Sum of signal overall
sum_signal=np.sum(extract_bigwig_intervals(intervals, args.trackbw))

#DHS score
if args.dhs is not None:
    dhs = pd.read_csv(args.dhs_file, sep='\t', header=None, usecols=(0,1,2))
    dhs_score = np.sum(extract_bigwig_intervals(dhs, args.bw_file))/sum_signal

# number of peaks
if args.peak_file is not None:
    peaks = pd.read_csv(args.peak_file, sep='\t', header=None, skip=1, usecols=(0,1,2))

    # Number of peaks with fold enrichment over threshold
    len(peaks)

    # FSIP
    np.sum(extract_bigwig_intervals(peaks, args.bw_file))/sum_signal

# FSIP (clean)
if args.clean_peak_file is not None:
    clean_peaks = pd.read_csv(args.clean_peak_file, sep='\t', header=None, skip=1, usecols=(0,1,2))
    np.sum(extract_bigwig_intervals(clean_peaks, args.bw_filew))/sum_signal

