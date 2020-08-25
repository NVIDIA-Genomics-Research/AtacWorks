#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""Unit tests for atacworks/dl4atac/dataset.py module."""
import os

import numpy as np
import pytest

from atacworks.io import h5io
from atacworks.dl4atac.dataset import DatasetInfer, DatasetTrain


@pytest.mark.cpu
def test_dataset_train(tmpdir):
    """Create a dict of numpy arrays and write it to a file using \
    the dict_to_h5 API. Send the file to DatasetTrain API and verify \
    that samples are being read as expected."""
    input_dict = {"input": np.array([0, 1, 3, 4]),
                  "label_reg": np.array([1, 2, 10, 1]),
                  "label_cla": np.array([1, 1, 0, 0])}
    h5file = os.path.join(tmpdir, "h5file.h5")
    h5io.dict_to_h5(input_dict, h5file)
    train_dataset = DatasetTrain(files=[h5file], layers=None)
    for idx in range(0, len(train_dataset)):
        expected_dict = {"idx": idx,
                         "input": input_dict["input"][idx],
                         "label_reg": input_dict["label_reg"][idx],
                         "label_cla": input_dict["label_cla"][idx]}
        result_dict = train_dataset[idx]

        assert expected_dict.keys() == result_dict.keys()
        for key, value in expected_dict.items():
            assert result_dict[key] == value


@pytest.mark.cpu
def test_dataset_infer(tmpdir):
    """Create a dict of numpy arrays and write it to a file using \
    the dict_to_h5 API. Send the file to DatasetInfer API and verify \
    that samples are being read as expected."""
    input_dict = {"input": np.array([0, 1, 3, 4]),
                  "label_reg": np.array([1, 2, 10, 1]),
                  "label_cla": np.array([1, 1, 0, 0])}
    h5file = os.path.join(tmpdir, "h5file.h5")
    h5io.dict_to_h5(input_dict, h5file)
    infer_dataset = DatasetInfer(files=[h5file], layers=None)
    for idx in range(0, len(infer_dataset)):
        expected_dict = {"idx": idx,
                         "input": input_dict["input"][idx]}
        result_dict = infer_dataset[idx]

        assert expected_dict.keys() == result_dict.keys()
        for key, value in expected_dict.items():
            assert result_dict[key] == value
