#!/usr/bin/env python

#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""Worker functions for training, inference and evaluation."""

# system imports
import logging
import os
# python imports
import warnings

from atacworks.dl4atac import metrics
# module imports
from atacworks.dl4atac.dataset import DatasetInfer, DatasetTrain
from atacworks.dl4atac.evaluate import evaluate
from atacworks.dl4atac.infer import infer
from atacworks.dl4atac.losses import MultiLoss
from atacworks.dl4atac.metrics import (AUROC, BCE, CorrCoef, MSE,
                                       Recall, Specificity)
from atacworks.dl4atac.models.model_utils import build_model
from atacworks.dl4atac.train import train
from atacworks.dl4atac.utils import myprint, save_config, save_model

import torch

import torch.distributed as dist

from torch.optim import Adam
warnings.filterwarnings("ignore")

# Set up logging
log_formatter = logging.Formatter(
    '%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('AtacWorks-worker')
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)


def get_losses(task, mse_weight, pearson_weight, gpu, poisson_weight):
    """Return loss function.

    Args:
        task : Whether the task is regression or classification or both.
        mse_weight : Mean squared error weight.
        pearson_weight : Pearson correlation loss weight.
        poisson_weight : Poisson loss weight.
        gpu : Number of gpus.

    Return:
        loss_func : list of loss functions for each task.

    """
    reg_loss_func = MultiLoss('poissonloss', poisson_weight, device=gpu) \
        if poisson_weight > 0 else MultiLoss(['mse', 'pearsonloss'],
                                             [mse_weight, pearson_weight],
                                             device=gpu)
    cla_loss_func = MultiLoss('bce', 1, device=gpu)

    if task == "regression":
        loss_func = reg_loss_func
    elif task == "classification":
        loss_func = cla_loss_func
    else:
        loss_func = [reg_loss_func, cla_loss_func]

    return loss_func


def get_metrics(task, threshold, best_metric_choice):
    """Get metrics.

    Args:
        task : Whether the task is regression or classification or both.
        threshold : the threshold for classification.
        best_metric_choice : which metric to use for best metric.

    Return:
        metrics_reg : List of metrics to calculate for regression.
        metrics_cla : List of metrics to calculate for classification
        best_metric : Metric to choose best model from.

    """
    metrics_reg = []
    metrics_cla = []
    best_metric = []
    if task == "regression":
        metrics_reg = [MSE(), CorrCoef()]
    elif task == "classification":
        metrics_cla = [BCE(), Recall(threshold),
                       Specificity(threshold), AUROC()]
    elif task == 'both':
        metrics_reg = [MSE(), CorrCoef()]
        metrics_cla = [BCE(), Recall(threshold),
                       Specificity(threshold), AUROC()]
    try:
        best_metric_class = getattr(metrics, best_metric_choice)

        if metrics_reg:
            for obj in metrics_reg:
                if isinstance(obj, best_metric_class):
                    best_metric = obj

        if metrics_cla:
            for obj in metrics_cla:
                if isinstance(obj, best_metric_class):
                    best_metric = obj
    except AttributeError as e:
        print(e)
    return metrics_reg, metrics_cla, best_metric


def get_model(args, gpu, rank):
    """Build and return the model.

    Args:
        args : Parsed argument object.
        gpu : GPU identity, if using specific GPU.
        rank : .

    Return:
        model : built model
        model_params : model parameters

    """
    torch.cuda.set_device(gpu)
    _logger.debug('Rank %s' % str(rank))

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size, rank=rank)

    # Why is model & optimizer built in spawned function?
    resume = (args.weights_path is not None)
    model, model_params = build_model(rank=rank,
                                      interval_size=args.interval_size,
                                      resume=resume,
                                      infer=args.mode == "infer",
                                      evaluate=args.mode == "eval",
                                      weights_path=args.weights_path,
                                      gpu=gpu, distributed=args.distributed)
    return model, model_params


def train_worker(gpu, ngpu_per_node, args, timers=None):
    """Build models and run training.

    Args:
        gpu : GPU identity, if using specific GPU.
        ngpu_per_node : Number of GPUs per node.
        args : argument object.
        timers : .

    """
    print_freq = 50
    # fix random seed so models have the same starting weights
    if args.seed is not None and args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    rank = gpu if args.distributed else 0

    model, model_params = get_model(args, gpu, rank)

    optimizer = Adam(model.parameters(), lr=args.lr)

    config_dir = os.path.join(args.exp_dir, "configs")
    dst_path = os.path.join(config_dir, "model_structure.yaml")
    save_config(dst_path, model_params)
    # TODO: LR schedule
    train_dataset = DatasetTrain(files=args.train_files, layers=args.layers)
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True,
        sampler=train_sampler,
        drop_last=False)

    # TODO: need DatasetVal? Not for now
    val_dataset = DatasetTrain(files=args.val_files, layers=args.layers)
    val_sampler = None
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        # collate_fn=custom_collate_train,
        num_workers=args.num_workers, pin_memory=True, sampler=val_sampler,
        drop_last=False
        # drop_last=True # need to drop irregular batch for distributed
        # evaluation due to limitation of dist.all_gather
    )

    loss_func = get_losses(args.task, args.mse_weight,
                           args.pearson_weight, gpu, args.poisson_weight)

    current_best = None
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(rank=rank, gpu=gpu, task=args.task, model=model,
              train_loader=train_loader,
              loss_func=loss_func, optimizer=optimizer, epoch=epoch,
              epochs=args.epochs,
              print_freq=print_freq, pad=args.pad,
              distributed=args.distributed, world_size=args.world_size,
              )

        # either create new objects or call reset on each metric obj
        metrics_reg, metrics_cla, best_metric = get_metrics(
            args.task, args.threshold, args.best_metric_choice)

        # best_metric is the metric used to compare results
        # across different evaluation runs. It's modified in place.
        evaluate(rank=rank, gpu=gpu, task=args.task,
                 model=model, val_loader=val_loader,
                 metrics_reg=metrics_reg, metrics_cla=metrics_cla,
                 world_size=args.world_size, distributed=args.distributed,
                 best_metric=best_metric, pad=args.pad,
                 print_freq=print_freq)

        if rank == 0:
            new_best = best_metric.better_than(current_best)
            if new_best:
                current_best = best_metric
                myprint("New best metric found - {}".format(current_best),
                        color='yellow', rank=rank)
            # give it the module attribute of the model
            # (DistributedDataParallel wrapper)
            if args.distributed:
                save_model(
                    model.module, args.exp_dir, args.checkpoint_fname,
                    epoch=epoch, is_best=new_best)
            else:
                save_model(
                    model, args.exp_dir, args.checkpoint_fname,
                    epoch=epoch, is_best=new_best)


def infer_worker(gpu, ngpu_per_node, args, res_queue=None):
    """Run inference.

    Args:
        gpu : GPU identity number, if using specific GPU.
        ngpu_per_node : Number of GPUs per node.
        args : argument object.
        res_queue : Inference queue.

    """
    print_freq = 50
    # fix random seed so models have the same starting weights
    if args.seed is not None and args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    rank = gpu if args.distributed else 0

    model, _ = get_model(args, gpu, rank)

    infer_dataset = DatasetInfer(files=args.input_files, layers=args.layers)
    infer_sampler = None

    if args.distributed:
        infer_sampler = torch.utils.data.distributed.DistributedSampler(
            infer_dataset, shuffle=False)

    infer_loader = torch.utils.data.DataLoader(
        infer_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, sampler=infer_sampler,
        drop_last=False
    )

    infer(rank=rank, gpu=gpu, task=args.task, model=model,
          infer_loader=infer_loader,
          print_freq=print_freq, res_queue=res_queue,
          pad=args.pad)


def eval_worker(gpu, ngpu_per_node, args, res_queue=None):
    """Run evaluation.

    Args:
        gpu : GPU identity number, if using specific GPU.
        ngpu_per_node : Number of GPUs per node.
        args : argument object.
        res_queue : Evaluate queue.

    """
    print_freq = 50
    # fix random seed so models have the same starting weights
    if args.seed is not None and args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    rank = gpu if args.distributed else 0

    model, _ = get_model(args, gpu, rank)

    eval_dataset = DatasetTrain(args.input_files, layers=args.layers)
    eval_sampler = None

    if args.distributed:
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, sampler=eval_sampler,
        drop_last=False
    )

    metrics_reg, metrics_cla, best_metric = get_metrics(
        args.task, args.threshold, args.best_metric_choice)
    evaluate(rank=rank, gpu=gpu, task=args.task,
             model=model, val_loader=eval_loader,
             metrics_reg=metrics_reg,
             metrics_cla=metrics_cla,
             world_size=args.world_size,
             distributed=args.distributed,
             best_metric=best_metric, res_queue=res_queue,
             pad=args.pad,
             print_freq=print_freq)
