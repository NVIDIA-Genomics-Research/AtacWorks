#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""Contains functions to read and write to hdf5 format."""

import h5py

import numpy as np


def dict_to_h5(x, h5file, create_new=True, compression='lzf'):
    """Write a dictionary to an h5 file.

    Args:
        x: dictionary with numpy arrays
        h5file: path to hdf5 file to write
        create_new: Create new datasets in the h5 file
        compression: compression type to use for h5 file.
            Only valid with create_new=True.

    """
    with h5py.File(h5file) as f:
        if create_new:
            # Create new datasets
            for key in x.keys():
                max_shape = list(x[key].shape)
                max_shape[0] = None
                df = f.create_dataset(key, data=x[key],
                                      maxshape=max_shape,
                                      compression=compression)
        else:
            # Extend existing datasets
            for key in x.keys():
                df = f[key]
                d_len = df.shape[0]
                data_dimension = list(x[key].shape)
                data_dimension[0] += d_len
                df.resize(data_dimension)
                df[d_len:] = x[key]


def h5_to_array(h5file, dataset, pad, flatten=True):
    """Read test data into a NumPy array.

    Args:
        h5file: path to hdf5 file containing data
        dataset: dataset in hdf5 file to read
        pad: interval padding in h5 file
        flatten: Flatten the output into a 1-D array

    Returns:
        data: NumPy array containing a channel of the data.

    """
    with h5py.File(h5file, 'r') as f:
        data = np.array(f[dataset])
    # ignore padding
    if pad is not None:
        center = range(pad, data.shape[1] - pad)
        print("Remove padding and reduce interval size from {} to {}".format(
            data.shape[1], len(center)))
        data = data[:, center]
    # Flatten data
    if flatten:
        data = data.flatten()
        return data
