#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""Unit tests for atacworks/io/bedio.py module."""
import os

import pandas as pd
import pytest

from atacworks.io import bedio


@pytest.mark.cpu
def test_read_intervals(tmpdir):
    """Write intervals to a file and read it through read_intervals \
    API. Compare the output of read_intervals with original data."""
    intervals = {"chrom": ["chr1", "chr2"],
                 "start": [0, 100],
                 "end": [100, 500]}
    input_df = pd.DataFrame(intervals)
    bedfile = os.path.join(tmpdir, "intervals.bed")
    input_df.to_csv(bedfile, sep="\t", header=False, index=False)
    output_df = bedio.read_intervals(bedfile)
    assert input_df.equals(output_df)


@pytest.mark.cpu
def test_read_intervals_skip(tmpdir):
    """Write intervals to a file and read it using read_intervals \
    API, by skipping the first row. Compare the output with \
    original data with first row skipped."""
    intervals = {"chrom": ["chr1", "chr2"],
                 "start": [0, 100],
                 "end": [100, 500]}
    intervals_skip = {"chrom": ["chr2"],
                      "start": [100],
                      "end": [500]}
    input_df = pd.DataFrame(intervals)
    input_df_skip = pd.DataFrame(intervals_skip)
    bedfile = os.path.join(tmpdir, "intervals.bed")
    input_df.to_csv(bedfile, sep="\t", header=False, index=False)
    output_df = bedio.read_intervals(bedfile, skip=1)
    assert input_df_skip.equals(output_df)


@pytest.mark.cpu
def test_sizes(tmpdir):
    """Write sizes to a file and read it using read_sizes API. \
    Compare the output of read_sizes with original data."""
    sizes = {"chrom": ["chr1", "chr2"],
             "length": [1000, 200]}
    input_df = pd.DataFrame(sizes)
    bedfile = os.path.join(tmpdir, "sizes.bed")
    input_df.to_csv(bedfile, sep="\t", header=False, index=False)
    output_df = bedio.read_sizes(bedfile)
    assert input_df.equals(output_df)


@pytest.mark.cpu
def test_sizes_as_intervals(tmpdir):
    """Write sizes to a file and read it as intervals using \
    read_sizes API. Compare the output of read_sizes with \
    original data."""
    sizes = {"chrom": ["chr1", "chr2"],
             "length": [1000, 200]}
    sizes_intervals = {"chrom": ["chr1", "chr2"],
                       "start": [0, 0],
                       "end": [1000, 200]}
    input_df = pd.DataFrame(sizes)
    sizes_intervals_df = pd.DataFrame(sizes_intervals)
    bedfile = os.path.join(tmpdir, "sizes.bed")
    input_df.to_csv(bedfile, sep="\t", header=False, index=False)
    output_df = bedio.read_sizes(bedfile, as_intervals=True)
    assert sizes_intervals_df.equals(output_df)


@pytest.mark.cpu
def test_df_to_bed(tmpdir):
    """Create a pandas dataframe of intervals and write it to a file \
    using df_to_bed API. Read the output file and compare with \
    original data."""
    sizes_intervals = {"chrom": ["chr1", "chr2"],
                       "start": [0, 0],
                       "end": [1000, 200]}
    sizes_df = pd.DataFrame(sizes_intervals)
    bedfile = os.path.join(tmpdir, "sizes.bed")
    bedio.df_to_bed(sizes_df, bedfile)
    read_output = pd.read_csv(bedfile, sep="\t", header=None)
    sizes_df.columns = [0, 1, 2]
    assert sizes_df.equals(read_output)


@pytest.mark.cpu
def test_df_to_bed_header(tmpdir):
    """Create a pandas dataframe of intervals and write it to a file, \
    along with the header using df_to_bed API. Read the output file \
    and compare with original data."""
    sizes_intervals = {"chrom": ["chr1", "chr2"],
                       "start": [0, 0],
                       "end": [1000, 200]}
    sizes_df = pd.DataFrame(sizes_intervals)
    bedfile = os.path.join(tmpdir, "sizes.bed")
    bedio.df_to_bed(sizes_df, bedfile, header=True)
    read_output = pd.read_csv(bedfile, sep="\t")
    assert sizes_df.equals(read_output)
