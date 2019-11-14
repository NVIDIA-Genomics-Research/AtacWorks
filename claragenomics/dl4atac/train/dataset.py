#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import time
import sys


MAX_FILES = 30
class DatasetBase(Dataset):
    def __init__(self, files):
        self.files = files
        self._h5_gen = None
        assert len(files) > 0, "Need to supply at least one file for dataset loading"
        assert len(files) < MAX_FILES, "Only tested for up to %d files in dataset" % MAX_FILES
        self.running_counts = [0]
        for file in self.files:
            with h5py.File(file, 'r') as f:
                self.running_counts.append(
                    self.running_counts[-1] + f["data"].shape[0])

    def __len__(self):
        return (self.running_counts[-1])

    # Just do linear search to find which file to access.
    # Return file number and relative ID...
    # Assume IDX ranges from 0 to (total_len - 1)
    def _get_file_id(self, idx):
        for i in range(len(self.files)):
            if idx < self.running_counts[i+1]:
                return (i, idx - self.running_counts[i])
        # If not found, we have an error
        return None


    def __getitem__(self, idx):
        raise NotImplementedError("Abstract class method called")


class DatasetTrain(DatasetBase):
    ''' Custom DatasetTrain class to load data from disk and allow random indexing

    __len__ method: returns the number of batches in the dataset
    __getitem__ method: allows indexing of the dataset and returns the indexed batch

    Args:
        files: list of data file paths

    '''

    def __getitem__(self, idx):
        if self._h5_gen is None:
            self._h5_gen = self._get_generator()
            next(self._h5_gen)
        return self._h5_gen.send(idx)

    def _get_generator(self):
        # Support 2+ datasets
        hdrecs = []
        for i,filename in enumerate(self.files):
            #print('loading H5Py file %s' % filename)
            hf = h5py.File(filename, 'r')
            hd = hf["data"]
            #print('shape %s' % str(hd.shape))
            hdrecs.append(hd)
        idx = yield
        while True:
            # Find correct dataset, given idx
            file_id, local_idx = self._get_file_id(idx)
            assert file_id < len(hdrecs), "No file reference %d" % file_id
            rec = hdrecs[file_id][local_idx]
            if len(rec.shape) == 1:
                # When no labels, return just the input data
                idx = yield {'idx':idx, 'x':rec}
            else:
                # Return 4 items -- IDX (for saving/tracing), input data, upsampled data, peaks/classifications
                idx = yield {'idx':idx, 'x':rec[:,0], 'y_reg':rec[:,1], 'y_cla':rec[:,2]}

class DatasetInfer(DatasetBase):
    ''' Infer Dataset
        1. Not intended to be shuffled
    '''
    def __init__(self, files, prefetch_size=256):
        super(DatasetInfer, self).__init__(files)
        self.fh_indices = {}
        for i in range(len(self.files)):
            self.fh_indices[i] = (0, 0)
        self.fh_data = {}
        self.prefetch_size = prefetch_size

    def __getitem__(self, idx):
        if self._h5_gen is None:
            self._h5_gen = self._get_generator()
            next(self._h5_gen)
        return self._h5_gen.send(idx)

    def _get_generator(self):
        hdrecs = []
        for i,filename in enumerate(self.files):
            hf = h5py.File(filename, 'r')
            hd = hf["data"]
            hdrecs.append(hd)
            sys.stdout.flush()
        idx = yield
        while True:
            # Find correct dataset, given idx
            file_id, local_idx = self._get_file_id(idx)
            assert file_id < len(self.files)
            if (local_idx < self.fh_indices[file_id][0]) or (local_idx >= self.fh_indices[file_id][1]):
                # Data is not pre loaded, so load new data
                self.fh_indices[file_id] = (local_idx, local_idx + self.prefetch_size)
                self.fh_data[file_id] = hdrecs[file_id][local_idx : local_idx + self.prefetch_size]
                sys.stdout.flush()
            rec = self.fh_data[file_id][local_idx - self.fh_indices[file_id][0]]
            sys.stdout.flush()
            if len(rec.shape) == 1:
                # When no labels, return just the input data
                idx = yield {'idx':idx, 'x':rec}
            else:
                # Return 4 items -- IDX (for saving/tracing), input data, upsampled data, peaks/classifications
                idx = yield {'idx':idx, 'x':rec[:,0], 'y_reg':rec[:,1], 'y_cla':rec[:,2]}

"""
# Is this even used?
class DatasetEval(DatasetBase):

    def __getitem__(self, idx):
        for i in range(len(self.running_counts) - 1):
            low = self.running_counts[i]
            high = self.running_counts[i+1]
            if idx >= low and idx < high:
                with h5py.File(self.files[i]) as f:
                    batch_key = self.batch_name_prefix + str(idx-low)
                    batch = torch.from_numpy(np.array(f[batch_key]))
                    x = batch[..., 0].type(torch.float32)
                    y_reg = batch[..., 1].type(torch.float32)
                    y_cla = batch[..., 2].type(torch.float32)
                break

        return self.batch_name_prefix + str(idx), x, y_reg, y_cla


def custom_collate_train(batch):
    '''
        Assumes batch to be a sequence of tuples (x, y1, y2)
    '''
    elem = batch[0]
    assert len(elem[0].shape) == 2
    zipped = zip(*batch)
    return [torch.cat(samples, 0) for samples in zipped]


def custom_collate_infer(batch):
    '''
        Assumes batch to be a sequence of tensors
    '''
    if len(batch) != 1:
        raise AttributeError("Inference must be done one batch at a time, for now.")

    key, tensor = batch[0]
    assert len(tensor.shape) == 2
    return (key, tensor.contiguous())


def custom_collate_eval(batch):
    '''
        Assumes batch to be a sequence of tensors
    '''
    if len(batch) != 1:
        raise AttributeError("Evaluation (with result dumping) must be done one batch at a time, for now.")

    key, x, y_reg, y_cla = batch[0]
    assert len(x.shape) == 2
    return (key, x.contiguous(), y_reg.contiguous(), y_cla.contiguous())
"""
