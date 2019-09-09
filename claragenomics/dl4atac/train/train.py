#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from claragenomics.dl4atac.train.utils import myprint, progbar, equal_width_formatter
import h5py
import numpy as np


def train(*, rank, gpu, task, model, train_loader, loss_func, optimizer, pad,
          epoch, epochs, clip_grad, print_freq, distributed, world_size):

    num_batches = len(train_loader)
    epoch_formatter = "Epoch " + \
        equal_width_formatter(total=epochs).format(epoch)
    start = time.time()
    forward_time = 0.
    backward_time = 0.
    print_time = 0.

    model.train()

    print('Num_batches %d; rank %s, gpu %s' % (num_batches, str(rank), str(gpu)))

    # Loop training data
    for i, batch in enumerate(train_loader):
        x = batch['x']
        y_reg = batch['y_reg']
        y_cla = batch['y_cla']
        # model forward pass
        x = x.unsqueeze(1)  # (N, 1, L)
        x = x.cuda(gpu, non_blocking=True)

        if task == 'regression':
            y = y_reg.cuda(gpu, non_blocking=True)
        elif task == 'classification':
            y = y_cla.cuda(gpu, non_blocking=True)
        elif task == 'both':
            y_reg = y_reg.cuda(gpu, non_blocking=True)
            y_cla = y_cla.cuda(gpu, non_blocking=True)

        t = time.time()
        pred = model(x)

        # Calculate losses
        # Remove padding
        cen = list(range(pad, x.shape[2] - pad))
        if task == 'regression' or task == 'classification':
            total_loss_value, losses_values = loss_func(pred[:, cen], y[:, cen])
        elif task == 'both':
            total_loss_value_reg, losses_values_reg = loss_func[0](
                pred[0][:, cen], y_reg[:, cen])
            total_loss_value_cla, losses_values_cla = loss_func[1](
                pred[1][:, cen], y_cla[:, cen])
            # Combine loss values
            losses_values = losses_values_reg.copy()
            losses_values.update(losses_values_cla)
            # Combine total loss
            total_loss_value = total_loss_value_reg + total_loss_value_cla
            losses_values['total_loss'] = total_loss_value

        forward_time += time.time() - t

        # one gradient descent step
        optimizer.zero_grad()
        t = time.time()
        total_loss_value.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        backward_time += time.time() - t

        # Loss is only reduced every X batches?
        if (i % print_freq == 0) or (i == num_batches-1):
            t = time.time()
            if (dist.is_initialized()):
                for loss_type, value in losses_values.items():
                    # update inplace, ReduceOp=SUM
                    dist.reduce_multigpu([value], dst=0)
                    losses_values[loss_type] = value / world_size

            if rank == 0:
                post_bar_msg = " | ".join(
                    [k + ':{:8.3f}'.format(v.cpu().item()) for k, v in losses_values.items()])
                progbar(curr=i, total=num_batches, progbar_len=20,
                        pre_bar_msg=epoch_formatter, post_bar_msg=post_bar_msg)
            print_time += time.time() - t

    myprint(epoch_formatter +
            " Time Taken: {:7.3f}s".format(time.time()-start), color='yellow', rank=rank)

    # Time breakdown for the epoch...
    total_time = time.time() - start
    remainder_time = total_time - forward_time - backward_time - print_time
    #print('Total train time: %.3f\tFor time: %.3f\tBack time: %.3f\tPrint time: %.3f\tRemain (data) time: %.3f' % (total_time, forward_time, backward_time, print_time, remainder_time))
