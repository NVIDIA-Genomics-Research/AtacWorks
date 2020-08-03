#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""Train module."""

import time

from atacworks.dl4atac.utils import myprint, progbar, equal_width_formatter

import torch.distributed as dist

import numpy as np


def train(*, rank, gpu, task, model, train_loader, loss_func, optimizer, pad,
          epoch, epochs, print_freq, distributed, world_size,
          ):
    """Train with given data.

    Args:
        rank: rank of current process
        gpu: GPU id to use
        task: task among 'regression', 'classification' and 'both'
        model: trained model
        train_loader : dataloader
        loss_func : Loss function
        optimizer : Optimization object
        pad : Padding
        epoch : Current epoch
        epochs : Total epochs to train for
        print_freq : How frequently to print training information.
        distributed : Distributed training
        world_size : World size

    """
    num_batches = len(train_loader)
    epoch_formatter = "Epoch " + \
                      equal_width_formatter(total=epochs).format(epoch)
    start = time.time()
    forward_time = 0.
    backward_time = 0.
    print_time = 0.

    model.train()

    print(
        'Num_batches %d; rank %s, gpu %s' % (num_batches, str(rank), str(gpu)))

    # Loop training data
    for i, batch in enumerate(train_loader):
        x = batch['input']
        y_reg = batch['label_reg']
        y_cla = batch['label_cla']

        # move data and labels to GPU for forward pass
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (N, 1, L)
        else:
            x = np.swapaxes(x, 1, 2)
        x = x.cuda(gpu, non_blocking=True)

        if task == 'regression':
            y = y_reg.cuda(gpu, non_blocking=True)
        elif task == 'classification':
            y = y_cla.cuda(gpu, non_blocking=True)
        elif task == 'both':
            y_reg = y_reg.cuda(gpu, non_blocking=True)
            y_cla = y_cla.cuda(gpu, non_blocking=True)

        # Model forward pass
        t = time.time()
        pred = model(x)

        # Remove padding
        if pad is not None:
            center = range(pad, x.shape[2] - pad)
            if task == 'regression' or task == 'classification':
                y = y[:, center]
                pred = pred[:, center]
            elif task == 'both':
                y_reg = y_reg[:, center]
                y_cla = y_cla[:, center]
                pred = [x[:, center] for x in pred]

        # Calculate losses
        if task == 'regression' or task == 'classification':
            total_loss_value, losses_values = loss_func(pred, y)
        elif task == 'both':
            total_loss_value_reg, losses_values_reg = loss_func[0](
                pred[0], y_reg)
            total_loss_value_cla, losses_values_cla = loss_func[1](
                pred[1], y_cla)
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
        optimizer.step()
        backward_time += time.time() - t

        # Loss is only reduced every X batches?
        if (i % print_freq == 0) or (i == num_batches - 1):
            t = time.time()
            if (dist.is_initialized()):
                for loss_type, value in losses_values.items():
                    # update inplace, ReduceOp=SUM
                    dist.reduce_multigpu([value], dst=0)
                    losses_values[loss_type] = value / world_size

            if rank == 0:
                post_bar_msg = " | ".join(
                    [k + ':{:8.3f}'.format(v.cpu().item()) for k, v in
                     losses_values.items()])
                progbar(curr=i, total=num_batches, progbar_len=20,
                        pre_bar_msg=epoch_formatter, post_bar_msg=post_bar_msg)
            print_time += time.time() - t

    myprint(
        epoch_formatter + " Time Taken: {:7.3f}s".format(time.time() - start),
        color='yellow', rank=rank)

    # Time breakdown for the epoch...
    total_time = time.time() - start
    remainder_time = total_time - forward_time - backward_time - print_time
    print(
        'Total train time: %.3f\tFor time: %.3f\tBack time: %.3f\tPrint '
        'time: %.3f\tRemain (data) time: %.3f' % (total_time, forward_time,
                                                  backward_time, print_time,
                                                  remainder_time))
