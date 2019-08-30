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
bigwigio.py:
    Contains functions to read and write to bigWig format.
"""

# Import requirements
import numpy as np
import pandas as pd
import pyBigWig
import subprocess
import os


def extract_bigwig_to_numpy(interval, bw):
    """
    Function to read values in an interval from a bigWig file.
    Args:
        interval (list or list-like): containing chrom, start, end
        bw: bigWig file object
    Returns:
        NumPy array containing values in the interval
    """
    result = bw.values(interval[0], interval[1], interval[2])
    result = np.array(result, dtype='float32')
    result = np.nan_to_num(result)
    return result


def extract_bigwig_intervals(intervals_df, bwfile, stack=True):
    """
    Function to read values in multiple intervals from a bigWig file.
    Args:
        intervals_df (Pandas DataFrame): containing columns chrom, start, end
        bwfile: bigWig file path
        stack (bool): if True, stack the values into a 2D NumPy array. Only works for equal-sized intervals.
    Returns:
        NumPy array containing values in all intervals
    """
    with pyBigWig.open(bwfile) as bw:
        result = intervals_df.apply(
            extract_bigwig_to_numpy, axis=1, args=(bw, ))
    if stack:
        result = np.stack(result)
    return result


def check_bigwig_nonzero(interval, bw):
    """
    Function to chck whether an interval has nonzero coverage in a bigWig file.
    Args:
        interval (list or list-like): containing chrom, start, end
        bw: bigWig file object
    Returns:
        boolean: does the interval have nonzero coverage
    """
    result = bw.values(interval[0], interval[1], interval[2])
    return (~np.isnan(result)).any()


def check_bigwig_intervals_nonzero(intervals_df, bwfile):
    """
    Function to check whether multiple intervals have nonzero coverage
    in a bigWig file.
    Args:
        intervals_df (Pandas DataFrame): containing columns chrom, start, end
        bwfile: bigWig file path
    Returns:
        Pandas Series containing boolean value for each interval
    """
    with pyBigWig.open(bwfile) as bw:
        result = intervals_df.apply(
            check_bigwig_nonzero, axis=1, args=(bw, ))
    return result


def check_bigwig_peak(interval, bw):
    """
    Function to check whether an interval contains a peak.
    Args:
        interval (list or list-like): containing chrom, start, end
        bw: bigWig file object containing peaks
    Returns:
        boolean: does the interval contain a peak
    """
    result = bw.values(interval[0], interval[1], interval[2])
    try:
        result.index(1)
    except ValueError:
        return False
    else:
        return True


def check_bigwig_intervals_peak(intervals_df, bwfile):
    """
    Function to check whether multiple intervals contain peaks.
    Args:
        intervals_df (Pandas DataFrame): containing columns chrom, start, end
        bwfile: bigWig file path
    Returns:
        Pandas Series containing boolean value for each interval
    """
    with pyBigWig.open(bwfile) as bw:
        result = intervals_df.apply(
            check_bigwig_peak, axis=1, args=(bw, ))
    return result


def bedgraph_to_bigwig(bgfile, sizesfile, prefix=None, deletebg=False):
    """
    Function to convert bedGraph file to bigWig file
    Args:
        bgfile (str): path to bedGraph file
        sizesfile (str): path to chromosome sizes file
        prefix (str): optional prefix to name bigWig file
        delete (bool): delete bedGraph file after conversion
    Writes:
        bigWig file
    """
    if prefix is not None:
        bwfile = prefix + '.bw'
    else:
        bwfile = os.path.splitext(bgfile)[0] + '.bw'
    subprocess.call(['bedGraphToBigWig', bgfile, sizesfile, bwfile])
    if deletebg:
        subprocess.call(['rm', bgfile])
