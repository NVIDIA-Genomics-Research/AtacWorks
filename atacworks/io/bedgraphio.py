#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""Modules to write to bedGraph format."""

# Import requirements
import pandas as pd
import numpy as np


def expand_interval(interval, score=True):
    """Expand an interval to single-base resolution and add scores.

    Args:
        interval : dict or DataFrame containing chrom, start, end and
        optionally scores.
        score : Boolean to specify whether to add score to each base of
        interval

    Returns:
        expanded: pandas DataFrame containing a row for every base in the
        interval

    """
    expanded = pd.DataFrame(columns=['chrom', 'start'])
    expanded['start'] = range(interval['start'], interval['end'])
    expanded['end'] = expanded['start'] + 1
    expanded['chrom'] = interval['chrom']
    # If necessary, assign a score to each base in the interval
    if score:
        expanded['score'] = interval['scores']
    return expanded


def contract_interval(expanded_df, positive=True):
    """Contract a dataframe containing genomic positions.

    Scores into smaller intervals with equal score.

    Args:
        expanded_df: Pandas dataframe containing chrom, start, end, score at
        base resolution.
        positive : if True, only regions with score>0 are retained.

    Returns:
        intervals_df: Pandas dataframe with same columns; bases with same
        score are combined into one line.

    """
    # For each base, attach the score assigned to the previous base
    expanded_df['prevscore'] = [-1] + list(expanded_df['score'])[:-1]
    # Select bases where score changes - or the last base
    intervals_df = expanded_df[
        (expanded_df['score'] != expanded_df['prevscore']) | (
            expanded_df.index == len(expanded_df) - 1)].copy()
    # Each interval ends at the next point where the score changes,
    # or at the last base
    intervals_df['end'] = list(intervals_df['start'])[1:] + [
        intervals_df['end'].iloc[-1]]
    # Only keep intervals where score > 0
    if positive:
        intervals_df = intervals_df[intervals_df['score'] > 0]
    if len(intervals_df) > 0:
        intervals_df = intervals_df.loc[:, ['chrom', 'start', 'end', 'score']]
        return intervals_df


def combine_over_bins(df, resolution):
    """Combine rows of bedGraph format dataFrame into equal-sized bins.

    Args:
        df: Expanded pandas dataframe with columns chrom, start, end, score
        resolution: output dataframe resolution
        aggregate: function to aggregate rows

    Returns:
        binned_df: Binned dataFrame

    """
    binned_df = pd.DataFrame(columns=('chrom', 'start', 'end', 'score'))
    # Check that total interval size is a multiple of resolution
    assert(len(df) % resolution == 0)
    # Split expanded interval
    num_splits = (df.end.iloc[-1] - df.start.iloc[0]) / resolution
    split_df = np.split(df, num_splits)
    # Group rows and aggregate scores
    binned_df = pd.concat([combine_bin(bin) for bin in split_df], axis=1).T
    return binned_df


def combine_bin(df):
    """Combine rows of bedGraph format dataFrame.

    Args:
        df: Expanded pandas dataframe for a single bin, with columns
            chrom, start, end, score

    Returns:
        bin_series: Pandas series describing the bin

    """
    bin_series = pd.Series({'chrom': df.chrom.iloc[0], 'start': min(df.start),
                            'end': max(df.end), 'score': np.mean(df.score)})
    return bin_series


def intervals_to_bg(intervals_df, resolution):
    """Format intervals + scores to bedGraph format.

    Args:
        intervals_df: Pandas dataframe containing columns for chrom, start,
        end and scores.
        resolution: output resolution in bp

    Returns:
        bg: pandas dataframe containing expanded+contracted intervals.

    """
    # Expand each interval to single-base resolution and add scores
    bg = intervals_df.apply(expand_interval, axis=1)
    if resolution is not None:
        # Combine scores over bins
        bg = bg.apply(combine_over_bins, args=(resolution,))
    # Contract regions where score is the same
    bg = bg.apply(contract_interval)
    # Combine into single pandas df
    bg = pd.concat(list(bg))
    return bg


def df_to_bedGraph(df, outfile, sizes=None):
    """Write a dataframe in bedGraph format to a bedGraph file.

    Args:
        df : dataframe to be written.
        outfile : file name or object.
        sizes: dataframe containing chromosome sizes.

    """
    if sizes is not None:
        # Write only entries for the given chromosomes.
        num_drop = sum(~df['chrom'].isin(sizes['chrom']))
        print("Discarding " + str(num_drop) + " entries outside sizes file.")
        df = df[df['chrom'].isin(sizes['chrom'])]
        # Check that no entries exceed chromosome lengths.
        df_sizes = df.merge(sizes, on='chrom')
        excess_entries = df_sizes[
            df_sizes['end'] > df_sizes['length']]
        assert len(excess_entries) == 0, \
            "Entries exceed chromosome sizes ({})".format(excess_entries)
    assert len(df) > 0, "0 entries to write to bedGraph"
    df.to_csv(outfile, sep='\t', header=False, index=False)
