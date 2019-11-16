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


def flip_negative(df, col):
    """
    Function to flip values for intervals on the negative strand.
    Args:
        df: Pandas dataframe with a column named 'strand'
        col: name of column in df whose values are to be flipped
    Returns:
        Pandas Series identical to df[col] except that values are 
        flipped for rows where strand='-'.
    """
    if df['strand'] == '-':
        return np.flip(df[col], 0)
    else:
        return df[col]


def filter_bed(bed_df, intervals_df):
    """
    Function to select bed-format entries that are fuly contained within given intervals
    Assumes all entries in bed_df have the same chromosome.
    Args:
        bed_df: Pandas dataframe with columns for chrom, start, end.
        intervals_df: Pandas dataframe with columns for chrom, start, end.
    Returns:
        bed_filtered: Pandas dataframe containing rows of bed_df that are contained within 
        intervals of intervals_df. 
    """
    intervals_df = intervals_df[intervals_df[0] == bed_df.iloc[0,0]]
    if len(intervals_df) > 0:
        in_interval = bed_df.apply(lambda x:((intervals_df[1]<=x[1])[intervals_df[2] >= x[2]].any()), axis=1)
        bed_filtered = bed_df[in_interval]
        return bed_filtered


def filter_bed_multichrom(bed_df, intervals_df=None, sizes_df=None):
    """
    Function to select bed-format entries that are fuly contained within given intervals or chromosomes
    Args:
        bed_df: Pandas dataframe with columns for chrom, start, end.
        intervals_df: Pandas dataframe with columns for chrom, start, end.
        sizes_df: Pandas dataframe with columns for chrom, size.
    Returns:
        bed_filtered: Pandas dataframe containing rows of bed_df that are contained within 
        intervals of intervals_df or chromosomes of sizes_df. 
    """
    if intervals_df is not None:
        bed_by_chrom = bed_df.groupby(0, as_index=False)
        bed_filtered = bed_by_chrom.apply(filter_bed, (intervals_df))
    elif sizes_df is not None:
        bed_filtered = bed_df[bed_df[0].isin(sizes[0])]
    else:
        raise InputError('Either intervals or sizes must be provided.')
    return bed_filtered


def parse_args():
    parser = argparse.ArgumentParser(
        description='AtacWorks script to calculate additional signal-quality metrics from a bigWig file.')
    parser.add_argument('--bw_file', type=str,
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
if args.intervals is not None:
    _logger.info('Loading intervals')
    intervals = pd.read_csv(args.intervals, header=None, sep='\t', usecols=(0,1,2))
# If no intervals are supplied, evaluate on full length of chromosomes provided.
elif args.sizes is not None:
    _logger.info('Loading chromosome sizes')
    sizes = pd.read_csv(args.sizes, header=None, sep='\t', usecols=(0,1))
else:
    parser.error('Either intervals or sizes must be provided')

# TSS enrichment
if args.tss_file is not None:
    
    # Read TSS
    _logger.info('Loading TSS positions')
    tss = pd.read_csv(args.tss_file, sep='\t', header=None, usecols=(0,1,2,3))
    
    # Filter TSS
    if args.intervals is not None:
        tss = filter_bed_multichrom(tss, intervals)
    else:
        tss = filter_bed_multichrom(tss, None, sizes)

    if len(tss) > 0:
        tss.columns = ['chrom', 'start', 'end', 'strand']

        # Calculate TSS enrichment over 4000-bp region
        if args.prefix is not None:

            # Define TSS region
            tss['start'] = tss['start'] - 2000
            tss['end'] = tss['end'] + 2000

            # Extract coverage
            _logger.info('Extracting signal in TSS +/- 2000 bp')
            tss['coverage'] = extract_bigwig_intervals(tss, args.bw_file, stack=False)
            tss['flip'] = tss.apply(flip_negative, axis=1, args=('coverage',))
            signal_near_tss = np.sum(np.stack(tss['flip']), axis=0)[:-1]
            enr_near_tss = signal_near_tss/signal_near_tss[1:200].mean()
            
            # Save local enrichment as .npy file
            _logger.info('Saving enrichment in TSS +/- 2000 bp')
            np.save(args.prefix + '.npy', enr_near_tss)

            # Get TSS score
            signal_near_tss = signal_near_tss[1000:3000]
        else:
            tss['start'] = tss['start'] - 1000
            tss['end'] = tss['end'] + 1000
            _logger.info('Extracting signal in TSS +/- 1000 bp')
            signal_near_tss = extract_bigwig_intervals(tss, args.bw_file, stack=True)
            signal_near_tss = np.sum(signal_near_tss)

        # Calculate TSS score
        _logger.info('Calculating TSS score')
        tss_score = np.mean(signal_near_tss)*2/(signal_near_tss[:100] + signal_near_tss[-100:]).mean()
        print("TSS Score: {}".format(tss_score))

# Mean signal overall
_logger.info('Extracting signal values for all positions in intervals')
if args.intervals is not None:
    full_signal = extract_bigwig_intervals(intervals, args.bw_file)
else:
    full_signal = extract_bigwig_chromosomes(sizes, args.bw_file)
    full_signal = np.concatenate(full_signal)
_logger.info('Calculating mean and total signal overall')
sum_signal = np.sum(full_signal)
mean_signal = sum_signal/len(full_signal)

#DHS score
if args.dhs_file is not None:

    # Read DHS
    _logger.info('Loading DHS positions')
    dhs = pd.read_csv(args.dhs_file, sep='\t', header=None, usecols=(0,1,2))

    # Filter DHS
    if args.intervals is not None:
        dhs = filter_bed_multichrom(dhs, intervals)
    else:
        dhs = filter_bed_multichrom(dhs, None, sizes)

    # Calculate DHS score
    _logger.info('Extracting signal in DHSs')
    signal_in_dhs = extract_bigwig_intervals(dhs, args.bw_file, stack=False)
    signal_in_dhs = np.concatenate(signal_in_dhs)
    _logger.info('Calculating DHS score')
    dhs_score = np.sum(signal_in_dhs)/sum_signal
    print("DHS Score: {}".format(dhs_score))

# Number of peaks
if args.peak_file is not None:

    # Read peaks
    _logger.info('Loading peaks')
    peaks = pd.read_csv(args.peak_file, sep='\t', header=None, skiprows=1, usecols=(0,1,2,9))

    # Filter peaks
    if args.intervals is not None:
        peaks = filter_bed_multichrom(peaks, intervals)
    else:
        peaks = filter_bed_multichrom(peaks, None, sizes)

    # Number of peaks
    _logger.info('Counting peaks')
    num_peaks = len(peaks)    
    print("Number of peaks: {}".format(num_peaks))

    # Summits
    summits = peaks[[0]].copy()
    summits[1] = peaks[1] + peaks[9]
    _logger.info('Extracting signal at summits')
    signal_at_summits = extract_bigwig_positions(summits, args.bw_file)
    fc_at_summits = signal_at_summits/mean_signal

    # Number of peaks with fold enrichment over threshold
    _logger.info('Counting peaks with fold enrichment over threshold')
    num_peaks_10 = sum(fc_at_summits >= 10)
    print("Number of peaks with FC>=10 over global average: {}".format(num_peaks_10))
    num_peaks_20 = sum(fc_at_summits >= 20)
    print("Number of peaks with FC>=20 over global average: {}".format(num_peaks_20))

    # FSIP
    _logger.info('Calculating FSIP')    
    signal_in_peaks = extract_bigwig_intervals(peaks, args.bw_file, stack=False)
    signal_in_peaks = np.concatenate(signal_in_peaks)
    fsip = np.sum(signal_in_peaks)/sum_signal
    print("FSIP: {}".format(fsip))

# FSIP (clean)
if args.clean_peak_file is not None:

    # Read clean peaks
    _logger.info('Loading peaks from clean data')
    clean_peaks = pd.read_csv(args.clean_peak_file, sep='\t', header=None, skiprows=1, usecols=(0,1,2,6))

    # Filter clean peaks
    if args.intervals is not None:
        clean_peaks = filter_bed_multichrom(clean_peaks, intervals)
    else:
        clean_peaks = filter_bed_multichrom(clean_peaks, None, sizes)

    # FSIP
    _logger.info('Calculating FSIP using peaks in clean data')
    signal_in_clean_peaks = extract_bigwig_intervals(clean_peaks, args.bw_file, stack=False)
    signal_in_clean_peaks = np.concatenate(signal_in_clean_peaks)
    fsip_clean = np.sum(signal_in_clean_peaks)/sum_signal
    print("FSIP_clean: {}".format(fsip_clean))


