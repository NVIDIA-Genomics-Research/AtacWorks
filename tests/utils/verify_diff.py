#!/usr/bin/env python

#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""Test different output files against expected files."""

import argparse
import subprocess
import sys

import h5py

import numpy as np

import torch


VERIFY_DIFF_EPS = 0.001


def parse_args():
    """Parse command line arguments.

    Return:
        args : parsed argument object.

    """
    parser = argparse.ArgumentParser(
        "Utils scripts for verifying identicality of different files.")
    parser.add_argument('--result_path', type=str,
                        help='Path to generated result file.')
    parser.add_argument('--expected_path', type=str,
                        help='Path to expected result file.')
    parser.add_argument('--format', type=str,
                        choices=["torch", "text", "h5", "general_diff"],
                        help='File formats being tested.')
    args = parser.parse_args()
    return args


def verify_torch(result_path, expected_path):
    """Verify if the two torch files are identical.

    Args:
        result_path: Path to generated result file.
        expected_path: Path to expected result file.

    """
    rank = 0
    result_model = torch.load(result_path, map_location="cuda:" + str(rank))
    expected_model = torch.load(expected_path,
                                map_location="cuda:" + str(rank))
    val = 0
    for key, _ in result_model['state_dict'].items():
        val = val + torch.sum(
            (result_model['state_dict'][key] - expected_model[
                'state_dict'][key]) ** 2).item()

    if (val > VERIFY_DIFF_EPS):
        raise ValueError("Models not identical!")
        sys.exit(-1)


def _get_metrics_dict(logfile, strings_to_check):
    """Parse a log file and look for specific strings.

    Args:
        logfile: file to be parsed.
        strings_to_check: List of strings to look for and parse.

    Return:
        Dictionary with relevant strings as keys and the parsed
        string as values.

    """
    metrics_dict = {}
    with open(logfile, "r") as infile:
        line = infile.readline()
        while line:
            for str_check in strings_to_check:
                if (line.find(str_check) == 0):
                    newstr = line.replace(str_check + " : ", '')
                    metric_list = newstr.strip().split("|")
                    metrics_dict[str_check] = metric_list
            line = infile.readline()
    return metrics_dict


def verify_text(result_path, expected_path):
    """Parse text file for particular metrics.

    Compare with expected results.

    Args:
        result_path: Path to generated result file.
        expected_path: Path to expected result file.

    """
    strings_to_check = ["Regression metrics on full data",
                        "Regression metrics in peaks",
                        "Regression metrics outside peaks",
                        "Classification metrics at threshold 0.5",
                        "AUC metrics"]

    result_metrics = _get_metrics_dict(result_path, strings_to_check)
    expected_metrics = _get_metrics_dict(expected_path, strings_to_check)

    if (result_metrics.keys() != expected_metrics.keys()):
        raise ValueError(
            "Result file and expected file do not have matching metrics!")

    for key in result_metrics.keys():
        result_dict = {}
        expected_dict = {}

        for metric in result_metrics[key]:
            metric = metric.strip().split(":")
            result_dict[metric[0]] = float(metric[1])
        for metric in expected_metrics[key]:
            metric = metric.strip().split(":")
            expected_dict[metric[0]] = float(metric[1])

        if result_dict.keys() != expected_dict.keys():
            raise ValueError(
                "Metric keys for result and expected do not match!")

        for key, value in result_dict.items():
            diff = abs(value - expected_dict[key])
            if diff > VERIFY_DIFF_EPS:
                print(value, expected_dict[key])
                err_msg = "Metrics for " + key + " differ more than " + str(
                    VERIFY_DIFF_EPS) + " between result and expected metrics!"
                raise ValueError(err_msg)


def verify_h5(result_path, expected_path):
    """Compare h5 files using h5diff.

    Args:
        result_path: Path to generated result file.
        expected_path: Path to expected result file.

    """
    result_keys = []
    result_data = {}
    expected_keys = []
    expected_data = {}

    with h5py.File(result_path, 'r') as result:
        result_keys = set(list(result.keys()))
        for key in result_keys:
            result_data[key] = np.array(result[key]).flatten()

    with h5py.File(expected_path, 'r') as expected:
        expected_keys = set(list(expected.keys()))
        for key in expected_keys:
            expected_data[key] = np.array(expected[key]).flatten()

    if expected_keys != result_keys:
        msg = expected_path + " and " + result_path + " are not identical!"
        raise ValueError(msg)

    for key in expected_keys:
        result_array = result_data[key]
        expected_array = expected_data[key]
        if not np.array_equal(result_array, expected_array):
            msg = expected_path + " and " + result_path + " are not identical!"
            raise ValueError(msg)


def verify_general_diff(result_path, expected_path):
    """Compare h5 files using h5diff.

    Args:
        result_path: Path to generated result file.
        expected_path: Path to expected result file.

    """
    cmd = "diff " + result_path + " " + expected_path
    ret = subprocess.call(cmd, shell=True)
    err_msg = result_path + " and " + expected_path + " are not identical!"
    if ret != 0:
        raise ValueError(err_msg)


args = parse_args()
if args.format == "torch":
    verify_torch(args.result_path, args.expected_path)
elif args.format == "text":
    verify_text(args.result_path, args.expected_path)
elif args.format == "h5":
    verify_h5(args.result_path, args.expected_path)
elif args.format == "general_diff":
    verify_general_diff(args.result_path, args.expected_path)
else:
    msg = "Requested format " + args.format + " is not supported!"
    print(msg)
