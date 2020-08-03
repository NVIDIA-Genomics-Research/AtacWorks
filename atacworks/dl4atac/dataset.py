#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""Dataset class. Load data for training, inference, evaluation."""

import sys

import h5py

import numpy as np

from torch.utils.data import Dataset


class DatasetBase(Dataset):
    """Base class."""

    def __init__(self, files, layers):
        """Initialize base class.

        Args:
            files: Dataset files to load.
            layers: list of names of additional input layers to read.

        """
        self.files = files
        self.layers = layers
        self._h5_gen = None
        assert len(files) > 0, \
            "Need to supply at least one file for dataset loading"
        self.running_counts = [0]
        for file in self.files:
            with h5py.File(file, 'r') as f:
                self.running_counts.append(
                    self.running_counts[-1] + f["input"].shape[0])

    def __len__(self):
        """Lengths of all datasets loaded."""
        return (self.running_counts[-1])

    # Just do linear search to find which file to access.
    # Return file number and relative ID...
    # Assume IDX ranges from 0 to (total_len - 1)
    def _get_file_id(self, idx):
        """Return file ID based on index mapping.

        Args:
            idx: file index to fetch ID for.

        """
        for i in range(len(self.files)):
            if idx < self.running_counts[i + 1]:
                return (i, idx - self.running_counts[i])
        # If not found, we have an error
        return None

    def __getitem__(self, idx):
        """Throw error if called.

        idx: File index to fetch the ID for.

        """
        raise NotImplementedError("Abstract class method called")


class DatasetTrain(DatasetBase):
    """Custom class to load data from disk and allow random indexing.

    Args:
        files: list of data file paths.
        layers: list of names of additional input layers to read.

    """

    def __getitem__(self, idx):
        """Return indexed example.

        Args:
            idx: Index for which the batch is to be returned.

        """
        if self._h5_gen is None:
            self._h5_gen = self._get_generator()
            next(self._h5_gen)
        return self._h5_gen.send(idx)

    def _get_generator(self):
        """Generate example."""
        # Support 2+ datasets
        # Assumption - all files have the same set of named fields
        # All fields will be read
        # List fields and create dictionary
        hrecs = {'input': [], 'label_reg': [], 'label_cla': []}
        if self.layers is not None:
            for layer_key in self.layers:
                hrecs[layer_key] = []
        for i, filename in enumerate(self.files):
            # print('loading H5Py file %s' % filename)
            hf = h5py.File(filename, 'r')
            # Read noisy data and labels
            for key in hf.keys():
                hrecs[key].append(hf[key])
        idx = yield
        while True:
            # Find correct dataset, given idx
            file_id, local_idx = self._get_file_id(idx)
            assert file_id < len(
                hrecs['input']), "No file reference %d" % file_id
            rec = {'idx': idx}
            rec['input'] = hrecs['input'][file_id][local_idx]
            rec['label_reg'] = hrecs['label_reg'][file_id][local_idx]
            rec['label_cla'] = hrecs['label_cla'][file_id][local_idx]
            if self.layers is not None:
                for layer_key in self.layers:
                    rec['input'] = np.vstack((
                        rec['input'],
                        hrecs[layer_key][file_id][local_idx]))
                rec['input'] = np.swapaxes(rec['input'], 0, 1)
            yield rec


class DatasetInfer(DatasetBase):
    """Infer Dataset.

    Not intended to be shuffled

    """

    def __init__(self, files, layers, prefetch_size=256):
        """Initialize class.

        Args:
            files: Files to infer on.
            layers: list of names of additional input layers to read.
            prefetch_size: Number of samples to prefetch.

        """
        super(DatasetInfer, self).__init__(files, layers)
        self.fh_indices = {}
        for i in range(len(self.files)):
            self.fh_indices[i] = (0, 0)
        self.fh_data = {}
        self.prefetch_size = prefetch_size

    def __getitem__(self, idx):
        """Get data from requested index.

        Args:
            idx: Index to fetch the data from.

        Return:
            Fetched data.

        """
        if self._h5_gen is None:
            self._h5_gen = self._get_generator()
            next(self._h5_gen)
        return self._h5_gen.send(idx)

    def _get_generator(self):
        """Get generator."""
        idx = yield
        while True:
            # Find correct dataset, given idx
            file_id, local_idx = self._get_file_id(idx)
            filename = self.files[file_id]
            assert file_id < len(self.files)
            if (local_idx < self.fh_indices[file_id][0]) or (
                    local_idx >= self.fh_indices[file_id][1]):
                # Data is not pre loaded, so load new data
                # Read additional layers
                hf = h5py.File(filename, 'r')
                prefetch_hdrecs = hf["input"][
                    local_idx: local_idx + self.prefetch_size]
                if self.layers is not None:
                    for layer_key in self.layers:
                        prefetch_hdrecs = np.dstack(
                            (prefetch_hdrecs,
                             hf[layer_key][local_idx:
                                           local_idx + self.prefetch_size]))
                sys.stdout.flush()
                self.fh_indices[file_id] = (
                    local_idx, local_idx + self.prefetch_size)
                self.fh_data[file_id] = prefetch_hdrecs
                sys.stdout.flush()
            rec = self.fh_data[file_id][
                local_idx - self.fh_indices[file_id][0]]
            sys.stdout.flush()
            idx = yield {'idx': idx, 'input': rec}
