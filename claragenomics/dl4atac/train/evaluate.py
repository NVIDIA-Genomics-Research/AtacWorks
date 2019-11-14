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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel.scatter_gather import gather
from claragenomics.dl4atac.train.utils import myprint, gather_tensor
from claragenomics.dl4atac.train.metrics import CorrCoef


def evaluate(*, rank, gpu, task, model, val_loader, metrics_reg, metrics_cla, world_size, distributed, pad, best_metric=None, res_queue=None):
    ''' The evaluate function

    Args:
        rank: rank of current process
        gpu: GPU id to use
        task: task among 'regression', 'classification' and 'both'
        model: trained model
        val_loader: dataloader
        metrics_reg: Regression metrics objects
        metrics_cla : Classification metrics objects
        world_size: number of gpus used for evaluation
        distributed: distributed
        best_metric: metric object for comparison
        res_queue: network predictions will be put in the queue for result dumping
        pad: padding around intervals

    '''

    model.eval()
    start = time.time()

    ###################################################################
    y_reg_list = []
    y_cla_list = []
    pred_reg_list = []
    pred_cla_list = []

    ###################################################################
    print('Eval for %d batches' % len(val_loader))
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            idxes = batch['idx']
            x = batch['x']
            y_reg = batch['y_reg']
            y_cla = batch['y_cla']

            """
            if res_queue: # res_queue indicates the mode we are in
                (key, x, y_reg, y_cla) = batch
            else:
                (x, y_reg, y_cla) = batch
                """
            # model forward pass
            x = x.unsqueeze(1)  # (N, 1, L)
            x = x.cuda(gpu, non_blocking=True)

            # log normalize tracks
            x_log = torch.log(x+1)

            # predict using log track
            pred = model(x_log)

            ###################################################################
            # Remove padding before evaluation
            if pad is not None:            
                center = range(pad, x.shape[2] - pad)
                if task == 'regression' or task =='both':
                    y_reg = y_reg[:, center]
                if task == 'classification' or task == 'both':
                    y_cla = y_cla[:, center]
                if task == 'both':
                    pred = [x[:, center] for x in pred]
                else:
                    pred = pred[:, center]

            ###################################################################
            # dump results in eval mode
            """
            if res_queue:
                if task == "both":
                    batch_res = np.stack([p.cpu().numpy()
                                          for p in pred], axis=-1)
                else:
                    batch_res = np.expand_dims(pred.cpu().numpy(), axis=-1)
                res_queue.put((idxes, batch_res))
            """
            ###################################################################
            # Store all the batch predictions and labels in a list
            if task == 'both':
                y_reg_list.append(y_reg.detach())
                y_cla_list.append(y_cla.detach())
                pred_reg_list.append(pred[0].cpu().detach())
                pred_cla_list.append(pred[1].cpu().detach())
            elif task == 'classification':
                y_cla_list.append(y_cla.detach())
                pred_cla_list.append(pred.cpu().detach())
            else:
                y_reg_list.append(y_reg.detach())
                pred_reg_list.append(pred.cpu().detach())

        ###################################################################
        # on each device, concat result tensors together for later gathering
        if task == 'both' or task == 'regression':
            ys_reg = torch.cat(y_reg_list, dim=0)
            preds_reg = torch.cat(pred_reg_list, dim=0)
            #print(preds_reg)
            preds_reg = torch.exp(preds_reg) - 1
            #print(preds_reg)
            del y_reg_list
            del pred_reg_list
        if task == 'both' or task == 'classification':
            ys_cla = torch.cat(y_cla_list, dim=0)
            preds_cla = torch.cat(pred_cla_list, dim=0)
            del y_cla_list
            del pred_cla_list

        # gather_start = time.time()
        # gather the results across all devices
        if distributed:
            if task == 'both' or task == 'regression':
                ys_reg = gather_tensor(ys_reg, world_size=world_size, rank=rank)
                preds_reg = gather_tensor(
                    preds_reg, world_size=world_size, rank=rank)
            if task == 'both' or task == 'classification':
                ys_cla = gather_tensor(ys_cla, world_size=world_size, rank=rank)
                preds_cla = gather_tensor(
                    preds_cla, world_size=world_size, rank=rank)
            #myprint("Gathering takes {}s".format(time.time()-gather_start), rank=rank)

        # now with the results of whole dataset, compute metrics on device 0
        if rank == 0:
            if task == 'both' or task == 'classification':
                for metric in metrics_cla:
                    metric(preds_cla, ys_cla)
            if task == 'both' or task == 'regression':
                for metric in metrics_reg:
                    metric(preds_reg, ys_reg)
        ###################################################################

        metrics = metrics_reg + metrics_cla
        
        if rank == 0:
            result_str = " | ".join([str(metric) for metric in metrics])
            if task == 'regression' or task == 'both':
                myprint("Evaluating on {} points.".format(preds_reg.shape[1]))
            else:
                myprint("Evaluating on {} points.".format(preds_cla.shape[1]))
            myprint("Evaluation result: " + result_str, rank=rank)
            myprint("Evaluation time taken: {:7.3f}s".format(time.time()-start), color='yellow', rank=rank)
