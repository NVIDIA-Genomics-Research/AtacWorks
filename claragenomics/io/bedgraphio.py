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
bedGraphio.py: 
    Contains functions to write to bedGraph format.
"""

# Import requirements
import numpy as np
import pandas as pd


def expand_interval(interval, score=True):
    """
    Function to expand an interval to single-base resolution and add scores
    Args:
        interval (dict or DataFrame containing chrom, start, end and optionally scores)
        score (bool): add score to each base of interval
    Returns:
        expanded: pandas DataFrame containing a row for every base in the interval
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
    """
    Function to contract a dataframe containing genomic positions and scores into smaller intervals with equal score
    Args:
        expanded_df: Pandas dataframe containing chrom, start, end, score at base resolution.
        positive (bool): if True, only regions with score>0 are retained
    Returns:
        intervals_df: Pandas dataframe with same columns; bases with same score are combined into one line.
    """
    # For each base, attach the score assigned to the previous base
    expanded_df['prevscore'] = [-1] + list(expanded_df['score'])[:-1]    
    # Select bases where score changes - or the last base
    intervals_df = expanded_df[(expanded_df['score'] != expanded_df['prevscore']) | (expanded_df.index==len(expanded_df)-1)].copy()
    # Each interval ends at the next point where the score changes, or at the last base
    intervals_df['end'] = list(intervals_df['start'])[1:] + [intervals_df['end'].iloc[-1]]
    # Only keep intervals where score > 0
    if positive:
        intervals_df = intervals_df[intervals_df['score'] > 0]
    if len(intervals_df) > 0:
        intervals_df = intervals_df.loc[:, ['chrom', 'start', 'end', 'score']]
        return intervals_df


def intervals_to_bg(intervals_df):
    """
    Function to format intervals + scores to bedGraph format
    Args:
        intervals_df: Pandas dataframe containing columns for chrom, start, end and scores
    Returns:
        bg: pandas dataframe containing expanded+contracted intervals
    """
    # Expand each interval to single-base resolution and add scores
    bg = intervals_df.apply(expand_interval, axis=1)
    # Contract regions where score is the same
    bg = bg.apply(contract_interval)
    # Combine into single pandas df
    bg = pd.concat(list(bg))
    return bg


def df_to_bedGraph(df, outfile):
    """
    Function to write a dataframe in bedGraph format to a bedGraph file
    Args:
        df (Pandas dataframe): dataframe to be written
        outfile(file name or object)
    """
    # TODO - add checks - legitimate chromosome names, no entries outside chromosome sizes, sorted positions
    df.to_csv(outfile, sep='\t', header=False, index=False)
