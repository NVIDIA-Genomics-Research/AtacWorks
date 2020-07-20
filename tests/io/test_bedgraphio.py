#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""Unit tests for atacworks/io/bedgraphio.py module."""
import os

import pandas as pd
import pytest

from atacworks.io import bedgraphio


@pytest.mark.cpu
def test_expand_interval():
    """
    Tests behavior of expanding an interval of size K to K distinct \
    intervals at single base-pair resolution. Uses a dict describing \
    an interval range as input, and compares with expected output. \
    Scores provides a score for *each* base-pair.
    """
    interval = {'chrom': 'chr1', 'start': 1, 'end': 5,
                'scores': [0.2, 0.5, 0.5, 0.2]}
    output_df = bedgraphio.expand_interval(interval, score=True)
    expected_df_dict = {'chrom': ['chr1'] * 4,
                        'start': [1, 2, 3, 4],
                        'end': [2, 3, 4, 5],
                        'score': [0.2, 0.5, 0.5, 0.2]}
    expected_df = pd.DataFrame(expected_df_dict)
    pd.testing.assert_frame_equal(expected_df, output_df)


@pytest.mark.cpu
def test_contract_interval():
    """
    Tests behavior of 'contracting' an interval DataFrame at single-bp \
    resolution. Contracting entails combining neighboring bases with the \
    same score into a single DataFrame entry.
    Note: the last basepair always ends up as its own interval, even if it \
    has the same score as previous, as reflected in the test's expected DF.
    """
    expanded_df_dict = {'chrom': ['chr1'] * 7,
                        'start': range(1, 8),
                        'end': range(2, 9),
                        'score': [0.2, 0.2, 0.5, -0.1, -0.1, 0.5, 0.5]}
    expanded_df = pd.DataFrame(expanded_df_dict)
    output_df = bedgraphio.contract_interval(expanded_df, positive=True)
    output_df.reset_index(drop=True, inplace=True)
    expected_df_dict = {'chrom': ['chr1'] * 4,
                        'start': [1, 3, 6, 7],
                        'end': [3, 4, 7, 8],
                        'score': [0.2, 0.5, 0.5, 0.5]}
    expected_df = pd.DataFrame(expected_df_dict)
    pd.testing.assert_frame_equal(expected_df, output_df)


@pytest.mark.cpu
def test_intervals_to_bg_basepair():
    """
    Tests behavior of function that maps an intervals DF to bedGraph format. \
    An arbitrary number of intervals is first expanded to single-bp \
    resolution, and then these intervals are contracted where possible.
    Computed DF compared with expected DF.
    """
    intervals_df_dict = {'chrom': ['chr1', 'chr1', 'chr2'],
                         'start': [1, 4, 2],
                         'end': [4, 7, 5],
                         'scores': [[0.5, 0.5, 0.1],
                                    [0.5, 0.5, 0.1],
                                    [0.1, 0.1, 0.0]]}
    intervals_df = pd.DataFrame(intervals_df_dict)
    output_df = bedgraphio.intervals_to_bg(intervals_df, resolution=None)
    output_df.reset_index(drop=True, inplace=True)
    expected_df_dict = {'chrom': ['chr1'] * 4 + ['chr2'],
                        'start': [1, 3, 4, 6, 2],
                        'end': [3, 4, 6, 7, 4],
                        'score': [0.5, 0.1, 0.5, 0.1, 0.1]}
    expected_df = pd.DataFrame(expected_df_dict)
    pd.testing.assert_frame_equal(expected_df, output_df)


@pytest.mark.cpu
def test_intervals_to_bg_with_resolution():
    """
    Tests behavior of function that maps an intervals DF to bedGraph format. \
    An arbitrary number of intervals is first expanded to single-bp \
    resolution, these intervals are combined into size 3bp bins,
    and these bins are contracted where possible. Computed DF compared \
    with expected DF.
    """
    intervals_df_dict = {'chrom': ['chr1', 'chr1', 'chr2'],
                         'start': [1, 10, 2],
                         'end': [10, 16, 8],
                         'scores': [[0.5, 0.5, 0.1, 0.1, 0.5, 0.5,
                                     0.5, 0.1, 0.5],
                                    [0.2, 0.4, 0.3, 0.3, 0.4, 0.2],
                                    [0.1, 0.1, 0.0, 0.0, 0.0, 0.0]]}
    intervals_df = pd.DataFrame(intervals_df_dict)
    output_df = bedgraphio.intervals_to_bg(intervals_df, resolution=3)
    output_df.reset_index(drop=True, inplace=True)
    expected_df_dict = {'chrom': ['chr1'] * 4 + ['chr2'],
                        'start': [1, 7, 10, 13, 2],
                        'end': [7, 10, 13, 16, 5],
                        'score': [1.1 / 3, 1.1 / 3, 0.3, 0.3, 0.2 / 3]}
    expected_df = pd.DataFrame(expected_df_dict)
    # dtype check is False, because output_df returns all dtypes as "objects"
    # Pandas 1.0.0 supports convert_dtypes() which infers the closest possible
    # dtype. Since atacworks calls for 0.25.0, that feature doesn't exist.
    pd.testing.assert_frame_equal(expected_df, output_df, check_dtype=False)


@pytest.mark.cpu
def test_df_to_bedgraph(tmpdir):
    """
    Creates sample bedgraph-formatted DataFrame, writes to file, loads in \
    written output, and compares expected output to loaded in DF.
    """
    intervals_df_dict = {'chrom': ['chr1', 'chr1', 'chr2', 'chr10'],
                         'start': [1, 13, 1, 1],
                         'end': [4, 18, 5, 3],
                         'score': [[0.2] * 3, [0.3] * 5, [0.5] * 4, [0.1] * 2]}
    intervals_df = pd.DataFrame(intervals_df_dict)
    sizes_df_dict = {'chrom': ['chr1', 'chr2'],
                     'length': [27, 10]}
    sizes_df = pd.DataFrame(sizes_df_dict)
    bedgraphfile = os.path.join(tmpdir, "intervals.bedgraph")
    bedgraphio.df_to_bedGraph(intervals_df, bedgraphfile, sizes_df)
    output_df = pd.read_csv(bedgraphfile, sep="\t",
                            names=['chrom', 'start', 'end', 'score'],
                            dtype={'chrom': str, 'start': int,
                                   'end': int, 'score': str})
    expected_df_dict = {'chrom': ['chr1', 'chr1', 'chr2'],
                        'start': [1, 13, 1],
                        'end': [4, 18, 5],
                        'score': [str([0.2] * 3),
                                  str([0.3] * 5),
                                  str([0.5] * 4)]}
    expected_df = pd.DataFrame(expected_df_dict)
    pd.testing.assert_frame_equal(expected_df, output_df)
