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


MAX_FILES = 10
class DatasetBase(Dataset):
    def __init__(self, files, args={}):
        self.files = files
        # Easy way to pass command line args, w/o enurating all
        self.args = args
        self._h5_gen = None
        assert len(files) > 0, "Need to supply at least one file for dataset loading"
        assert len(files) < MAX_FILES, "Only tested for up to %d files in dataset" % MAX_FILES
        self.running_counts = [0]
        for file in self.files:
            with h5py.File(file, 'r') as f:
                self.running_counts.append(
                    self.running_counts[-1] + f["data"].shape[0])
        # Options for implied cuts and downsampling
        self.test_correct_cuts = True
        self.downsample_rate = args.sample_rate
        print('Generating data with dynamic downsampling rate %.3f' % self.downsample_rate)
        self.debug = False

    def __len__(self):
        return self.running_counts[-1]

    def __getitem__(self, idx):
        raise NotImplementedError("Abstract class method called")


class DatasetTrain(DatasetBase):
    ''' Custom DatasetTrain class to load data from disk and allow random indexing

    __len__ method: returns the number of batches in the dataset
    __getitem__ method: allows indexing of the dataset and returns the indexed batch

    Args:
        files: list of data file paths
        batch_name_prefix: key prefix of the batches in data files

    '''

    def __getitem__(self, idx):
        if self._h5_gen is None:
            self._h5_gen = self._get_generator()
            next(self._h5_gen)
        return self._h5_gen.send(idx)

    # Just do linear search to find which file to access.
    # Return file number and relative ID...
    # Assume IDX ranges from 0 to (total_len - 1)
    def _get_file_id(self, idx):
        for i in range(len(self.files)):
            if idx < self.running_counts[i+1]:
                return (i, idx - self.running_counts[i])
        # If not found, we have an error
        return None

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
            all_reads = rec[:,1]
            peaks = rec[:,2]
            downsampled_reads = rec[:,0]

            # Two options:
            # A. Return fixed downsampled reads
            # B. Compute *implied* cut sites, downsample, regenerate
            # Why find implied cut cites? Because we can substitute for actual cut sites in future data...
            compute_implied = True
            smooth_size = 200
            if compute_implied:
                implied_cuts = self.get_implied_cuts(h=all_reads, w=smooth_size)
                if self.debug:
                    print('Found %s cuts in the original data' % str(implied_cuts.shape))

                # Test correctness. Optional, adds time but really just negligible.
                # NOTE: Will throw error in *some* cases -- namely two reads back to back no gap.
                # NOTE: We could handle these, but more processing.
                if self.test_correct_cuts:
                    expanded_reads = self.expand_cuts(cuts=implied_cuts,h_shape=all_reads.shape[0], w=smooth_size)
                    # Assert we got our data back!
                    """
                    print(np.sum(all_reads))
                    print(all_reads)
                    print(expanded_reads)
                    print(all_reads - expanded_reads)
                    print(np.sum(all_reads - expanded_reads))
                    print(np.nonzero(all_reads - expanded_reads))
                    """
                    assert np.array_equal(all_reads, expanded_reads), "Not getting original image back from implied cuts!"

                # Now downsample...
                # HACK -- get downsample rate from the single downsample...
                # NOTE: Display purposes only -- rate is inconsistent...
                implied_downsample_cuts = self.get_implied_cuts(h=downsampled_reads, w=smooth_size)
                if self.debug:
                    print('data has downsampled from %s to %s cuts...' % (str(implied_cuts.shape), str(implied_downsample_cuts.shape)))

                sample_rate = self.downsample_rate
                # TODO: Add wiggle to the cuts? Better if we had true cuts... they come pre-wiggled
                sampled_cuts = np.random.choice(implied_cuts, int(np.ceil(len(implied_cuts) * sample_rate)), replace=False)
                if self.debug:
                    print('Produced %d sampled cuts' % sampled_cuts.shape[0])
                expanded_reads = self.expand_cuts(cuts=sampled_cuts,h_shape=all_reads.shape[0], w=smooth_size)
                x = expanded_reads

            # Return 4 items -- IDX (for saving/tracing), downsampled data (to train), full data, peaks/classifications
            idx = yield {'idx':idx, 'x':downsampled_reads, 'y_reg':all_reads, 'y_cla':downsampled_reads}

    # Get histogram, look for cuts, smoothed to width w
    # Return array of cut locations
    def get_implied_cuts(self,h,w, debug=False):
        if debug:
            print('getting implied cuts')
            print(h.shape)
            print(h)
            print(np.sum(h))
            print('estimated cuts:')
            print(np.sum(h)/(w+1))

        # 1. Expand h to head and tail values -- account for w-length at the ends
        # 2. Compare each pos with pos+1
        # 3. ^^ that should tell you all the cuts
        # TODO: This *almost* works -- we won't notice cuts if one starts exactly where another ends
        # (could be fixed with iterative step...)
        # 4. Adjust for padding
        # 5. Flatten to list
        h_expand = np.concatenate((np.full((w+1,),h[0]), h, np.full((w+1,),h[-1])))
        if debug:
            print(h_expand)
            print(h_expand.shape)

        # Debugging
        start = 12000
        end = start+50

        # forward cuts
        h_diff = h_expand[1:] - h_expand[:-1]
        h_diff = np.clip(h_diff, a_min=0, a_max=None)
        h_diff = h_diff[:-(w+1)]
        h_diff_for = h_diff
        if debug:
            print(h_diff)
            print('Total forward cuts!')
            print(h_diff[start:end])
            print(np.sum(h_diff))

        # backward cuts
        h_diff = h_expand[:-1] - h_expand[1:]
        h_diff = np.clip(h_diff, a_min=0, a_max=None)
        h_diff = h_diff[(w+1):]
        h_diff_back = h_diff
        if debug:
            print(h_diff)
            print('Total back cuts!')
            print(h_diff[start:end])
            print(np.sum(h_diff))

        # Max of forward and back cuts [removes edge effects]
        h_diff = np.maximum(h_diff_back, h_diff_for)
        # NOTE: max() will capture the *end* of each sequence. Cuts are in the middle...
        # NOTE: Cuts can occur... before or after the sequence
        cuts = h_diff #h_diff[int(w/2):-int(w/2)]
        if debug:
            print(cuts.shape)
            print(cuts)
            print(h_diff)
            print('Total (max) cuts!')
            print(h_diff[start:end])
            print(np.sum(h_diff))

            print(np.nonzero(h_diff))
            print(np.nonzero(cuts))

        """
        print(h[:100])
        print(h[100:200])
        print((147-2,147+2))
        print(h[147-2:147+2])
        print(h[200:300])
        print(h[300:400])
        print(h[400:500])
        print(h[500:600])
        """

        # cuts -- count by location (could be 2x or 3x)
        cuts_nonzero = np.nonzero(cuts)
        cuts_list = []
        for cut_id in cuts_nonzero[0].flat:
            # NOTE: Cuts can occurr before or after the sequence!
            cuts_list += [cut_id-int(w/2)]*int(cuts[cut_id])
        if debug:
            print(cuts_list)

        # NOTE: occasionally there is extra data left over. How? Cuts are consecutive, at exactly (w) size.
        # Solutoin: run this cutting, recursively...
        expanded_reads = self.expand_cuts(cuts=cuts_list,h_shape=h.shape[0], w=w)
        remain_reads = h - expanded_reads
        while np.sum(remain_reads) > 0.:
            if debug:
                print('remainder: %f' % np.sum(remain_reads))
                print('Calling cuts on remaining reads, recursively.')
            #time.sleep(2)
            extra_cuts = self.get_implied_cuts(h=remain_reads,w=w)
            cuts_list += list(extra_cuts)
            if debug:
                print('cuts after appending...')
                print(cuts_list)
            expanded_reads = self.expand_cuts(cuts=cuts_list,h_shape=h.shape[0], w=w)
            remain_reads = h - expanded_reads
        if np.sum(remain_reads) < 0.:
            assert False, 'Error in deconstructing reads with implied cuts! Check dimensions, or ignore'
            pass

        return np.array(cuts_list)

    # Get cuts, expand them -- into shape from h
    def expand_cuts(self,cuts,h_shape,w,debug=False):
        if debug:
            print('expanding cuts')
            print(cuts.shape)
            print(h_shape)

        expand_data = np.zeros((h_shape,), dtype=float)
        # Logic is re-implemneted from ATAC padding in different repo
        # Tested for w=200. We want *100* on each side, inclusively.
        # TODO: Better rule or definition.
        pad = int(w/2)
        for cut in cuts:
            expand_data[max(0,cut-pad):cut+pad+1] += 1.
        return expand_data












