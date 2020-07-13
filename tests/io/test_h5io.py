#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""Unit tests for atacworks/io/h5io.py module."""
import os

import numpy as np
import h5py
import pytest

from atacworks.io import h5io


@pytest.mark.cpu
def test_dict_to_h5(tmpdir):
    """Create a dict of numpy arrays and write it to a file using \
    the dict_to_h5 API. Read the output file and compare it with \
    the original data."""
    input_dict = {"features": np.array([0, 1, 2, 3]),
                  "scores": np.array([1, 2, 1, 1])}
    h5file = os.path.join(tmpdir, "h5file.h5")
    h5io.dict_to_h5(input_dict, h5file)
    with h5py.File(h5file, 'r') as f:
        for key in input_dict.keys():
            data = np.array(f[key]).flatten().tolist()
            assert data == input_dict[key].flatten().tolist()

    # Append to existing dataset and test whether the API updates
    # the dataset accordingly.
    new_dict = {"features": np.array([0, 1, 2, 3, 4, 5, 6])}
    h5io.dict_to_h5(new_dict, h5file, create_new=False)
    # Update new dictionary to reflect the new changes.
    input_dict["features"] = np.array([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6])
    with h5py.File(h5file, 'r') as f:
        for key in input_dict.keys():
            data = np.array(f[key]).flatten().tolist()
            input_data = input_dict[key].flatten().tolist()
            assert data == input_data


@pytest.mark.cpu
def test_h5_to_array(tmpdir):
    """Create a dict of numpy arrays and write it to a h5 file. \
    Using the h5_to_array API, read the h5 file and compare \
    the output with the original data."""
    input_dict = {"features": np.array([0, 1, 2, 3]),
                  "scores": np.array([1, 2, 1, 1])}
    h5file = os.path.join(tmpdir, "h5file.h5")
    with h5py.File(h5file) as f:
        for key in input_dict.keys():
            f.create_dataset(key, data=input_dict[key])

    for key in input_dict.keys():
        data = h5io.h5_to_array(h5file, key, None)
        assert data.tolist() == input_dict[key].flatten().tolist()
