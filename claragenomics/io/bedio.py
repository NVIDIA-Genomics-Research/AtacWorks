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


def read_intervals(bed_file):
    """Function to read genomic intervals from a BED file into a DataFrame.

    Args:
        bed_file(str): Path to BED file

    Returns:
        df: Pandas DataFrame containing intervals.

    """
    df = pd.read_csv(bed_file, sep='\t', header=None,
                     names=['chrom', 'start', 'end'],
                     usecols=(0, 1, 2),
                     dtype={'chrom': str, 'start': int, 'end': int})
    return df


def read_sizes(sizes_file, as_intervals=False):
    """Function to read chromosome sizes into a DataFrame.

    Args:
        sizes_file(str): Path to sizes file
        as_intervals(bool): Format the DataFrame as 0-indexed intervals

    """
    df = pd.read_csv(sizes_file, sep='\t', header=None, usecols=(0, 1),
                     dtype={0: str, 1: int})
    if as_intervals:
        df[2] = [0] * len(df)
        df.rename(columns={0: 0, 2: 1, 1: 2}, inplace=True)
        df.columns = ['chrom', 'start', 'end']
    else:
        df.columns = ['chrom', 'length']
    return df


def df_to_bed(df, out_file_path, header=False):
    """Write dataframe to BED file.

    Args:
        df: pandas DataFrame
        out_file_path: path to output BED file
        header: include column names as header

    """
    df.to_csv(out_file_path, sep='\t', index=False, header=header)
