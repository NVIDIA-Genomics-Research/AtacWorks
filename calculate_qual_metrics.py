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
calculate_qual_metrics.py:
    Returns quality metrics for a coverage track in bigWig format
"""

# Import requirements
import argparse
import numpy as np
import pyBigWig
import pandas as pd
import logging

from claragenomics.io.bigwigio import extract_bigwig_intervals, extract_bigwig_positions, extract_bigwig_chromosomes

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


def filter_bed(bed, intervals_df):
    intervals_df = intervals_df[intervals_df[0]==bed.iloc[0,0]]
    if len(intervals_df) > 0:
        in_interval = bed.apply(lambda x:((intervals_df[1]<=x[1])[intervals_df[2] >= x[2]].any()), axis=1)
        bed_filtered = bed[in_interval]
        return bed_filtered


def filter_bed_multichrom(bed, intervals_df=None, sizes_df=None):
    if intervals_df is not None:
        bed_by_chrom = bed.groupby(0, as_index=False)
        bed_filtered = bed_by_chrom.apply(filter_bed, (intervals_df))
    elif sizes_df is not None:
        bed_filtered = bed[bed[0].isin(sizes[0])]
    else:
        raise InputError('Either intervals or sizes must be provided.')
    return bed_filtered


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
    parser.add_argument('--prefix', type=str,
                        help='Output file prefix for TSS enrichment profile')
    parser.add_argument('--intervals', type=str,
                        help='bed file of intervals to evaluate on.')
    parser.add_argument('--sizes', type=str,
                        help='chromosome sizes file to evaluate on.')
    args = parser.parse_args()
    return args


args = parse_args()


# Load intervals
_logger.info('Loading intervals')
if args.intervals is not None:
	intervals = pd.read_csv(args.intervals, header=None, sep='\t', usecols=(0,1,2))
# If no intervals are supplied, evaluate on full length of chromosomes provided.
elif args.sizes is not None:
    sizes = pd.read_csv(args.sizes, header=None, sep='\t', usecols=(0,1))
else:
    parser.error('Either intervals or sizes must be provided')

# TSS enrichment
if args.tss is not None:
    
    # Read TSS
    tss = pd.read_csv(args.tss_file, sep='\t', header=None, usecols=(0,1,2,3))
    
    # Filter TSS
    if args.intervals is not None:
        tss = filter_bed_multichrom(tss, intervals)
    else:
        tss = filter_bed_multichrom(tss, None, sizes)

    if len(tss) > 0:
        tss.columns = ['chrom', 'start', 'end', 'strand']

        # Calculate TSS enrichment over 4000-bp region
        if args.output_file is not None:

            # Define TSS region
            tss['start'] = tss['start'] - 2000
            tss['end'] = tss['end'] + 2000

            # Extract coverage
            tss['coverage'] = extract_bigwig_intervals(tss, args.bw_file, stack=False)
            tss['flip'] = tss.apply(flip_negative, axis=1, args=('coverage',))
            signal_near_tss = np.sum(np.stack(tss['flip']), axis=0)[:-1]
            enr_near_tss = signal_near_tss/signal_near_tss[1:200].mean()
            np.save(args.prefix + '.npy', enr_near_tss)

            # Get TSS score
            signal_near_tss = signal_near_tss[1000:3000]
        else:
            tss['start'] = tss['start'] - 1000
            tss['end'] = tss['end'] + 1000
            signal_near_tss = extract_bigwig_intervals(tss, args.bw_file, stack=True)
            signal_near_tss = np.sum(signal_near_tss)

        # Calculate TSS score
        tss_score = np.mean(signal_near_tss)*2/(signal_near_tss[:100] + signal_near_tss[-100:])
        print("TSS Score: {}".format(tss_score))

# Mean signal overall
if args.intervals is not None:
    sum_signal = np.sum(extract_bigwig_intervals(intervals, args.bw_file))
    mean_signal = np.mean(extract_bigwig_intervals(intervals, args.bw_file))
else:
    sum_signal = np.sum(extract_bigwig_chromosomes(sizes, args.bw_file))
    mean_signal = np.mean(extract_bigwig_chromosomes(sizes, args.bw_file))

#DHS score
if args.dhs is not None:

    # Read DHS
    dhs = pd.read_csv(args.dhs_file, sep='\t', header=None, usecols=(0,1,2))

    # Filter DHS
    if args.intervals is not None:
        dhs = filter_bed_multichrom(dhs, intervals)
    else:
        dhs = filter_bed_multichrom(dhs, None, sizes)

    # Calculate DHS score
    dhs_score = np.sum(extract_bigwig_intervals(dhs, args.bw_file))/sum_signal

# Number of peaks
if args.peak_file is not None:

    # Read peaks
    peaks = pd.read_csv(args.peak_file, sep='\t', header=None, skiprows=1, usecols=(0,1,2,9))

    # Filter peaks
    if args.intervals is not None:
        peaks = filter_bed_multichrom(peaks, intervals)
    else:
        peaks = filter_bed_multichrom(peaks, None, sizes)

    # Number of peaks
    num_peaks = len(peaks)    

    # Summits
    peaks.columns = ['chrom', 'start', 'end', 'relativesummit']
    peaks['summit'] = peaks['start'] + peaks['relativesummit']
    signal_at_summits = extract_bigwig_positions(peaks[['chrom','summit']], args.bw_file)
    fc_at_summits = signal_at_summits/mean_signal

    # Number of peaks with fold enrichment over threshold
    num_peaks_10 = sum(fc_at_summits >= 10)
    num_peaks_20 = sum(fc_at_summits >= 20)

    # FSIP
    signal_in_peaks = extract_bigwig_intervals(peaks, args.bw_file, stack=True)
    fsip = np.sum(signal_in_peaks)/sum_signal

# FSIP (clean)
if args.clean_peak_file is not None:

    # Read clean peaks
    clean_peaks = pd.read_csv(args.clean_peak_file, sep='\t', header=None, skiprows=1, usecols=(0,1,2,6))

    # Filter clean peaks
    if args.intervals is not None:
        clean_peaks = filter_bed_multichrom(clean_peaks, intervals)
    else:
        clean_peaks = filter_bed_multichrom(clean_peaks, None, sizes)

    # FSIP
    signal_in_clean_peaks = extract_bigwig_intervals(clean_peaks, args.bw_file)
    fsip_clean = np.sum(signal_in_clean_peaks)/sum_signal

