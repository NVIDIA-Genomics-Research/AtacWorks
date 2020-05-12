#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""Read and write to bigWig format."""

# Import requirements
import os
import subprocess

import numpy as np

import pyBigWig


def extract_bigwig_to_numpy(interval, bw, pad, sizes, dtype='float32'):
    """Read values in an interval from a bigWig file.

    Args:
        interval : containing chrom, start, end.
        bw : bigWig file object.
        pad : padding around interval.
        sizes : dictionary of chromosome sizes.
        dtype : numpy dtype to return.

    Returns:
        NumPy array containing values in the interval.

    """
    if pad is None:
        result = bw.values(interval[0], interval[1], interval[2])
    else:
        # Add padding on both sides of interval.
        result = bw.values(interval[0], max(
            0, interval[1] - pad), min(interval[2] + pad, sizes[interval[0]]))
        # If padding goes beyond chromosome bounds, fill the empty spaces
        # with zeros.
        if interval[1] < pad:
            left_zero_pad = np.zeros(pad - interval[1])
            result = np.concatenate([left_zero_pad, result])
        if interval[2] + pad > sizes[interval[0]]:
            right_zero_pad = np.zeros(interval[2] + pad - sizes[interval[0]])
            result = np.concatenate([result, right_zero_pad])
        assert (len(result) == interval[2] - interval[1] + 2 * pad)
    result = np.nan_to_num(result)
    result = result.astype(dtype)
    return result


def extract_bigwig_intervals(intervals_df, bwfile, stack=True, pad=None,
                             dtype='float32'):
    """Read values in multiple intervals from a bigWig file.

    Args:
        intervals_df : containing columns chrom, start, end
        bwfile : bigWig file path
        stack : if True, stack the values into a 2D NumPy array. Only works
        for equal-sized intervals.
        pad : padding to add around interval edges
        dtype : numpy dtype to return
    Returns:
        NumPy array containing values in all intervals.

    """
    with pyBigWig.open(bwfile) as bw:
        result = intervals_df.apply(
            extract_bigwig_to_numpy, axis=1,
            args=(bw, pad, bw.chroms(), dtype))
    if stack:
        result = np.stack(result)
    return result


def check_bigwig_nonzero(interval, bw):
    """Check whether an interval has nonzero coverage in a bigWig file.

    Args:
        interval : containing chrom, start, end
        bw : bigWig file object
    Returns:
        boolean : does the interval have nonzero coverage

    """
    result = bw.values(interval[0], interval[1], interval[2])
    return (~np.isnan(result)).any()


def check_bigwig_intervals_nonzero(intervals_df, bwfile):
    """Check whether multiple intervals have nonzero coverage in a bigwig file.

    Args:
        intervals_df : Pandas dataframe containing columns chrom, start, end
        bwfile : bigWig file path
    Returns:
        Pandas Series containing boolean value for each interval.

    """
    with pyBigWig.open(bwfile) as bw:
        result = intervals_df.apply(
            check_bigwig_nonzero, axis=1, args=(bw,))
    return result


def check_bigwig_peak(interval, bw):
    """Check whether an interval contains a peak.

    Args:
        interval : List containing chrom, start, end
        bw: bigWig file object containing peaks
    Returns:
        boolean: does the interval contain a peak.

    """
    result = bw.values(interval[0], interval[1], interval[2])
    try:
        result.index(1)
    except ValueError:
        return False
    else:
        return True


def check_bigwig_intervals_peak(intervals_df, bwfile):
    """Check whether multiple intervals contain peaks.

    Args:
        intervals_df : Pandas dataframe containing columns chrom, start, end
        bwfile: bigWig file path
    Returns:
        Pandas Series containing boolean value for each interval.

    """
    with pyBigWig.open(bwfile) as bw:
        result = intervals_df.apply(
            check_bigwig_peak, axis=1, args=(bw,))
    return result


def bedgraph_to_bigwig(bgfile, sizesfile, prefix=None, deletebg=False,
                       sort=False):
    """Convert bedGraph file to bigWig file.

    Args:
        bgfile : path to bedGraph file
        sizesfile : path to chromosome sizes file
        prefix : optional prefix to name bigWig file
        deletebg : delete bedGraph file after conversion
        sort : sort bedGraph before conversion. Chromosomes sorted
        alphabetically.

    """
    if prefix is not None:
        bwfile = prefix + '.bw'
    else:
        bwfile = os.path.splitext(bgfile)[0] + '.bw'
    if sort:
        sort_env = os.environ.copy()
        sort_env['LC_COLLATE'] = 'C'
        subprocess.check_call(
            ['sort', '-u', '-k1,1', '-k2,2n', bgfile, '-o', bgfile],
            env=sort_env)
    subprocess.check_call(['bedGraphToBigWig', bgfile, sizesfile, bwfile])
    if deletebg:
        subprocess.check_call(['rm', bgfile])


def df_to_bigwig(intervals, sizes_file, batch_data, outputfile):
    """Write a pandas dataframe object into bigiwg file.

    Args:
        intervals : Pandas dataframe containing columns, chrom, start, end
        sizes_file : path to chromosome sizes file
        batch_data : Pandas dataframe containing scores
        outputfile : Path to output bigwig file

    """
    bw = pyBigWig.open(outputfile, "w")
    uniq_chroms = intervals["chrom"].unique()
    sizetuple = []
    with open(sizes_file, "r") as sizefile:
        sizes = sizefile.readlines()
        for size in sizes:
            size = size.strip().split("\t")
            if size[0] in uniq_chroms:
                sizetuple.append((size[0], int(size[1])))
    bw.addHeader(sizetuple, maxZooms=10)
    bw.addEntries(list(batch_data["chrom"]), list(batch_data["start"]),
                  ends=list(batch_data["end"]),
                  values=list(batch_data["score"]))
    bw.close()
