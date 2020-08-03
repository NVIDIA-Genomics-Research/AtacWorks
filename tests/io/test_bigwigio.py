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
import numpy as np
import pyBigWig
import pytest

from atacworks.io import bigwigio


@pytest.mark.cpu
def test_extract_bigwig_to_numpy(tmpdir):
    """
    Tests extract_bigwig_to_numpy function with padding values. \
    Uses pyBigWig to construct sample bigWig to draw values from. \
    Compares expected numpy array with computed numpy array.
    """
    tmpbigwig = os.path.join(tmpdir, "tmp.bigwig")
    bw = pyBigWig.open(tmpbigwig, "w")
    bw.addHeader([('chr1', 20)], maxZooms=0)
    bw.addEntries(['chr1', 'chr1'],
                  [0, 11],
                  ends=[5, 20],
                  values=[3.0, 7.0])
    bw.close()

    bw = pyBigWig.open(tmpbigwig)
    sizes = {'chr1': 20}
    pad = 5
    interval1 = ['chr1', 2, 4]  # -3 to 9
    interval2 = ['chr1', 14, 17]  # 9 to 22
    output1 = bigwigio.extract_bigwig_to_numpy(interval1, bw, pad, sizes)
    output2 = bigwigio.extract_bigwig_to_numpy(interval2, bw, pad, sizes)
    expected1 = np.array([0, 0, 0, 3.0, 3.0, 3.0, 3.0, 3.0, 0, 0, 0, 0])
    expected2 = np.array([0, 0, 7.0, 7.0, 7.0, 7.0, 7.0,
                          7.0, 7.0, 7.0, 7.0, 0, 0])
    assert np.allclose(expected1, output1)
    assert np.allclose(expected2, output2)


@pytest.mark.cpu
def test_extract_bigwig_intervals(tmpdir):
    """
    Tests extracting multiple intervals in a DataFrame from a bigwig. \
    Compares expected stacked NumPy array with computed one. \
    Uses consistent interval size (in order to stack) and padding.
    """
    tmpbigwig = os.path.join(tmpdir, "tmp.bigwig")
    bw = pyBigWig.open(tmpbigwig, "w")
    bw.addHeader([('chr1', 20)], maxZooms=0)
    bw.addEntries(['chr1', 'chr1'],
                  [0, 11],
                  ends=[5, 20],
                  values=[3.0, 7.0])
    bw.close()

    intervals_df_dict = {'chrom': ['chr1', 'chr1'],
                         'start': [3, 15],
                         'end': [5, 17]}
    intervals_df = pd.DataFrame(intervals_df_dict)
    output = bigwigio.extract_bigwig_intervals(intervals_df, tmpbigwig,
                                               stack=True, pad=5)
    expected = np.array([[0, 0, 3.0, 3.0, 3.0, 3.0,
                          3.0, 0, 0, 0, 0, 0],  # -2 to 10
                         [0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
                          7.0, 7.0, 0.0, 0.0]])  # 10 to 22
    assert np.allclose(expected, output)


@pytest.mark.cpu
def test_check_bigwig_nonzero(tmpdir):
    """
    Tests method that checks if an interval in a bigwig file is nonzero. \
    Uses two intervals on a sample bigwig, one which should return True and \
    another which should return False.
    """
    tmpbigwig = os.path.join(tmpdir, "tmp.bigwig")
    bw = pyBigWig.open(tmpbigwig, "w")
    bw.addHeader([('chr1', 20)], maxZooms=0)
    bw.addEntries(['chr1', 'chr1'],
                  [0, 11],
                  ends=[5, 20],
                  values=[3.0, 7.0])
    bw.close()

    bw = pyBigWig.open(tmpbigwig)
    interval1 = ['chr1', 9, 12]
    interval2 = ['chr1', 7, 10]
    assert bigwigio.check_bigwig_nonzero(interval1, bw)
    assert not bigwigio.check_bigwig_nonzero(interval2, bw)


@pytest.mark.cpu
def test_df_to_bigwig(tmpdir):
    """
    Tests writing a DataFrame of intervals with scores to a bigwig file. \
    Creates sample chromosome sizes file, intervals DF, and scores DF. \
    Writes output bigwig, and checks that values in this bigwig are correct.
    """
    sizes_fp = os.path.join(tmpdir, 'chroms.sizes')
    sizes_file = open(sizes_fp, 'w')
    sizes_file.write('chr1\t20\nchr2\t16')
    sizes_file.close()

    intervals_df_dict = {'chrom': ['chr1', 'chr1', 'chr2'],
                         'start': [0, 11, 3],
                         'end': [5, 20, 6]}
    intervals_df = pd.DataFrame(intervals_df_dict)

    scores_to_write_df = {'chrom': ['chr1', 'chr1'],
                          'start': [0, 13],
                          'end': [5, 20],
                          'score': [3.0, 7.0]}
    scores_to_write = pd.DataFrame(scores_to_write_df)

    outputfile = os.path.join(tmpdir, "output.bigwig")
    bigwigio.df_to_bigwig(intervals_df, sizes_fp, scores_to_write, outputfile)

    bw = pyBigWig.open(outputfile)
    bw_chr1 = bw.values('chr1', 0, 20)
    bw_chr2 = bw.values('chr2', 0, 16)
    expected_chr1 = np.array([3.0] * 5 + [np.nan] * 8 + [7.0] * 7)
    expected_chr2 = np.full(16, np.nan)
    assert np.allclose(expected_chr1, bw_chr1, equal_nan=True)
    assert np.allclose(expected_chr2, bw_chr2, equal_nan=True)
