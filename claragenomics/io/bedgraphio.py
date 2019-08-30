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


def expand_interval(interval):
    """
    Function to expand an interval to single-base resolution
    Args:
        interval (list or list-like containing chrom, start, and end)
    Returns:
        expanded: pandas dataframe containing a row for every base in the interval
    """
    expanded = pd.DataFrame(columns=['chrom', 'start'])
    expanded['start'] = range(interval[1], interval[2])
    expanded['chrom'] = interval[0]
    return expanded


def intervals_to_bg(intervals_df, scores):
    """
    Function to combine intervals and scores in bedGraph format
    Args:
        intervals_df: Pandas dataframe containing columns for chrom, start, end
        scores: numeric scores (at single-base resolution) to be added to bedGraph
    Returns:
        bg: pandas dataframe containing expanded intervals and scores where score>0.
    """
    bg = intervals_df.apply(expand_interval, axis=1)
    bg = pd.concat(list(bg))
    bg['end'] = bg['start']+1
    bg['score'] = scores
    bg = bg[bg['score'] > 0]
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
