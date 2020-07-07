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

r"""Creates intervals tiling across the whole genome or given chromosomes."""

# Import requirements
import logging
import os

from atacworks.io.bedio import df_to_bed, read_sizes
from atacworks.io.bigwigio import check_bigwig_intervals_peak

import pandas as pd

# Set up logging
log_formatter = logging.Formatter(
    '%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('AtacWorks-intervals')
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)


def get_tiling_intervals(sizes, intervalsize):
    """Produce intervals of given chromosomes.

    Tile from start to end of given chromosomes, shifting by given length.

    Args:
        sizes: Pandas df containing columns 'chrom' and 'length',
            with name and length of required chromosomes
        intervalsize: length of intervals
        shift: distance between starts of successive intervals.

    Returns:
        Pandas DataFrame containing chrom, start, and end of tiling intervals.

    """
    # Default: non-overlapping intervals
    shift = intervalsize

    # Create empty DataFrame
    intervals = pd.DataFrame()

    # Create intervals per chromosome
    for i in range(len(sizes)):
        chrom = sizes.iloc[i, 0]
        chrend = sizes.iloc[i, 1]
        starts = range(0, chrend - (intervalsize + 1), shift)
        ends = [x + intervalsize for x in starts]
        intervals = intervals.append(pd.DataFrame(
            {'chrom': chrom, 'start': starts, 'end': ends}))

    # Eliminate intervals that extend beyond chromosome size
    intervals = intervals.merge(sizes, on='chrom')
    intervals = intervals[intervals['end'] < intervals['length']]

    return intervals.loc[:, ('chrom', 'start', 'end')]


def get_intervals(sizesfile, intervalsize, out_dir, wg=None, val=None,
                  holdout=None, nonpeak=None, peakfile=None, regions=None):
    """Read chromosome sizes and generate intervals.

     Args:
         sizesfile: BED file containing sizes of each chromosome.
         intervalsize: Size of the intervals at each row.
         out_dir: Directory to save the output files to.
         wg: Create intervals for the whole genome.
         val: Chromosome to reserve for validation.
         holdout: Chromosome to reserve for evaluation.
         nonpeak: Ratio of nonpeak to peak intervals desired in training
         dataset.
         peakfile: File with clean peaks to know which intervals have non-zero
         values. Only useful if nonpeak is greater than one.

    Returns:
         Paths of files saved.

    """
    # Read chromosome sizes
    sizes = read_sizes(sizesfile)

    # Create the output dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Generate intervals
    if wg:

        # Generate intervals tiling across all chromosomes in the sizes file
        _logger.info("Generating intervals tiling across all chromosomes \
            in sizes file: " + sizesfile)
        intervals = get_tiling_intervals(sizes, intervalsize)

        # Write to file
        out_file_name = str(intervalsize) + '.genome_intervals.bed'
        wg_file_path = os.path.join(out_dir, out_file_name)
        df_to_bed(intervals, wg_file_path)
        _logger.info('Done!')
        return wg_file_path

    elif not (val is None or holdout is None):

        # Generate training intervals - can overlap
        _logger.info("Generating training intervals")
        train_sizes = sizes[sizes['chrom'] != val]
        if holdout is not None:
            train_sizes = train_sizes[train_sizes['chrom'] != holdout]
        train = get_tiling_intervals(
            train_sizes, intervalsize)

        # Optional - Set fraction of training intervals to contain peaks
        if nonpeak is not None:
            _logger.info('Finding intervals with peaks')
            train['peak'] = check_bigwig_intervals_peak(train, peakfile)
            _logger.info('{} of {} intervals contain peaks.'.format(
                train['peak'].sum(), len(train)))
            train_peaks = train[train['peak']].copy()
            train_nonpeaks = train[train['peak'] is False].sample(
                nonpeak * len(train_peaks))
            train = train_peaks.append(train_nonpeaks)
            train = train.iloc[:, :3]
            _logger.info('Generated {} peak and {} non-peak\
                     training intervals.'.format(
                len(train_peaks), len(train_nonpeaks)))

        # Write to file
        out_file_name = str(intervalsize) + '.training_intervals.bed'
        train_file_path = os.path.join(out_dir, out_file_name)
        df_to_bed(train, train_file_path)

        # Generate validation intervals - do not overlap
        _logger.info("Generating val intervals")
        val_sizes = sizes[sizes['chrom'] == val]
        val = get_tiling_intervals(
            val_sizes, intervalsize)

        # Write to file
        out_file_name = str(intervalsize) + '.val_intervals.bed'
        val_file_path = os.path.join(out_dir, out_file_name)
        df_to_bed(val, val_file_path)

        # Generate holdout intervals - do not overlap
        if holdout is not None:
            _logger.info("Generating holdout intervals")
            holdout_sizes = sizes[sizes['chrom'] == holdout]
            holdout = get_tiling_intervals(
                holdout_sizes, intervalsize)

            # Write to file
            out_file_name = str(intervalsize) + '.holdout_intervals.bed'
            holdout_file_path = os.path.join(out_dir, out_file_name)
            df_to_bed(holdout, holdout_file_path)
        _logger.info('Done!')
        return train_file_path, val_file_path, holdout_file_path
    elif regions is not None:
        # If given regions is a file, then just return the file path
        if regions.endswith(".bed"):
            return regions
        else:
            final_intervals = pd.DataFrame()
            regions = regions.strip("[]").split(",")
            for region in regions:
                chrom = region
                chrom_range = None
                # If regions are specified with intervals like chr1:0-50
                # Then split the region into chrom and it's range.
                if region.find(":") != -1:
                    chrom, chrom_range = region.split(":")
                chrom_sizes = sizes[sizes['chrom'] == chrom]
                intervals = get_tiling_intervals(chrom_sizes, intervalsize)

                if chrom_range:
                    chrom_range = chrom_range.split("-")
                    intervals = intervals[
                        intervals["start"] >= int(chrom_range[0])]
                    intervals = intervals[
                        intervals["end"] <= int(chrom_range[1])]
                final_intervals = final_intervals.append(intervals,
                                                         ignore_index=True)

            # Write the intervals to file
            out_file_name = str(intervalsize) + '.regions_intervals.bed'
            region_file_path = os.path.join(out_dir, out_file_name)
            df_to_bed(final_intervals, region_file_path)
            return region_file_path
