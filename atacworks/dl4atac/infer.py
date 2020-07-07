#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""Infer module."""

import time

from atacworks.dl4atac.utils import myprint, progbar

import numpy as np

import torch


def infer(*, rank, gpu, task, model, infer_loader, print_freq,
          res_queue, pad):
    """Run inference.

    Args:
        rank: rank of current process
        gpu: GPU id to use
        task: task among 'regression', 'classification' and 'both'
        model: trained model
        infer_loader: dataloader
        print_freq: logging frequency
        res_queue: network predictions will be put in the queue for
        result dumping
        pad: padding on ends of interval

    """
    # inference
    ##########################################################################
    num_batches = len(infer_loader)
    model.eval()
    start = time.time()
    local_init = time.time()

    pred_total_time = 0
    load_total_time = 0

    count = 0
    with torch.no_grad():
        for i, batch in enumerate(infer_loader):
            load_time = (time.time() - local_init)
            load_total_time += load_time
            local_init = time.time()

            idxes = batch['idx']
            x = batch['input']

            # move input to GPU for model forward pass
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # (N, 1, L)
            else:
                x = np.swapaxes(x, 1, 2)
            x = x.cuda(gpu, non_blocking=True)
            count += x.shape[0]

            # Model forward pass
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

            pred_time = (time.time() - local_init)
            pred_total_time += pred_time
            local_init = time.time()

    myprint("Inference time taken: {:8.3f}s (Load {:8.3f}s,"
            "Prediction {:8.3f}s)".format(time.time() - start,
                                          load_total_time, pred_total_time),
            color='yellow', rank=rank)
