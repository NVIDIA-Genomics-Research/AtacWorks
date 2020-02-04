#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""Contains functions to read and write to BED format."""

# Import requirements
import pandas as pd


def read_intervals(bed_file, skip=0):
    """Read genomic intervals from a BED file into a DataFrame.

    Args:
        bed_file: Path to BED file
        skip: Number of header lines to skip

    Returns:
        df: Pandas DataFrame containing intervals.

    """
    df = pd.read_csv(bed_file, sep='\t', header=None,
                     names=['chrom', 'start', 'end'],
                     usecols=(0, 1, 2),
                     dtype={'chrom': str, 'start': int, 'end': int},
                     skiprows=skip)
    return df


def read_sizes(sizes_file, as_intervals=False):
    """Read chromosome sizes into a DataFrame.

    Args:
        sizes_file(str): Path to sizes file
        as_intervals(bool): Format the DataFrame as 0-indexed intervals

    Returns:
        df: Pandas DataFrame

    """
    df = pd.read_csv(sizes_file, sep='\t', header=None, usecols=(0, 1),
                     names=['chrom', 'length'],
                     dtype={'chrom': str, 'length': int})
    if as_intervals:
        df['start'] = [0] * len(df)
        df = df[['chrom', 'start', 'length']]
        df.rename(columns={"length": "end"}, inplace=True)
    return df
