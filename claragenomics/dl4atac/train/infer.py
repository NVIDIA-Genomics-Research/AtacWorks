#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from collections import Iterable, OrderedDict
import time
import gc
import sys
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel.scatter_gather import gather
from claragenomics.dl4atac.train.utils import myprint, progbar, dump_results


def infer(*, rank, gpu, task, model, infer_loader, print_freq, res_queue, pad):
    ''' The infer function

    Args:
        rank: rank of current process
        gpu: GPU id to use
        task: task among 'regression', 'classification' and 'both'
        model: trained model
        infer_loader: dataloader
        print_freq: logging frequency
        res_queue: network predictions will be put in the queue for result dumping
        pad: padding on ends of interval

    '''

    # inference
    #################################################################################
    num_batches = len(infer_loader)
    model.eval()
    start = time.time()

    count = 0
    with torch.no_grad():
        for i, batch in enumerate(infer_loader):
            idxes = batch['idx']
            x = batch['x']
            
            # model forward pass
            x = x.unsqueeze(1)  # (N, 1, L)
            x = x.cuda(gpu, non_blocking=True)
            count += x.shape[0]

            pred = model(x)

            if task == 'both':
                batch_res = np.stack([x.cpu().numpy() for x in pred], axis=-1)
            else:
                batch_res = np.expand_dims(pred.cpu().numpy(), axis=-1)                    
            
            # Remove padding before writing results
            if pad is not None:
                center = range(pad, x.shape[2] - pad)
                batch_res = batch_res[:, center, :]

            # HACK -- replacing "key" with i=index.
            # TODO: Remove the write queue
            res_queue.put((idxes, batch_res))

            if rank == 0 and i % print_freq == 0:
                progbar(curr=i, total=num_batches, progbar_len=20,
                        pre_bar_msg="Inference", post_bar_msg="")

    myprint("Inference time taken: {:8.3f}s".format(
        time.time()-start), color='yellow', rank=rank)
