#!/usr/bin/env python

#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# local imports
from cmd_args import parse_args

# module imports
from claragenomics.dl4atac.train.models import *
from claragenomics.dl4atac.train.losses import MultiLoss
from claragenomics.dl4atac.train.dataset import DatasetTrain #, DatasetInfer, DatasetEval #, custom_collate_train, custom_collate_infer, custom_collate_eval
from claragenomics.dl4atac.train.utils import * # myprint, save_model, load_model, gather_files_from_cmdline, assert_device_available, make_experiment_dir
from claragenomics.dl4atac.train.train import train
from claragenomics.dl4atac.train.evaluate import evaluate
from claragenomics.dl4atac.train.infer import infer
from claragenomics.dl4atac.train.metrics import BCE, MSE, Recall, Specificity, CorrCoef, AUROC

# python imports
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os, sys
import time
import random
import h5py
import multiprocessing
# pytorch imports
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
import torch.multiprocessing as mp
import logging

# Set up logging
log_formatter = logging.Formatter(
    '%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('AtacWorks-main')
_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)

def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)

def get_losses(args):

    if args.task == "regression":
        loss_func = MultiLoss(['mse', 'pearsonloss'], [args.mse_weight, args.pearson_weight], device=args.gpu)
    elif args.task == "classification":
        loss_func = MultiLoss('bce', 1, device=args.gpu)
    elif args.task == 'both':  #shouldn't reach here for now
        loss_func = [MultiLoss(['mse', 'pearsonloss'], [args.mse_weight, args.pearson_weight], device=args.gpu),
                     MultiLoss('bce', 1, device=args.gpu)]
    return loss_func

def get_metrics(args):

    metrics_reg = [] 
    cla_metics = []
    best_metric = []
    if args.task == "regression":
        metrics_reg = [MSE(), CorrCoef()]
        best_metric = metrics_reg[0]
    elif args.task == "classification":
        metrics_cla = [BCE(), Recall(args.threshold), Specificity(args.threshold), AUROC()]
        best_metric = metrics_cla[0]
    elif args.task == 'both':  #shouldn't reach here for now
        metrics_reg = [MSE(), CorrCoef()]
        metrics_cla = [BCE(), Recall(args.threshold), Specificity(args.threshold), AUROC()]
        best_metric = metrics_cla[0]

    return metrics_reg, metrics_cla, best_metric

# build_model now does build, load, distribute in one go
def build_model(args):
    myprint("Building model: {} ...".format(args.model), color='yellow', rank=args.rank)
    # TODO: implement a model dic for model instantiation

    if args.model == 'unet':  # args.task == 'both'
        model = DenoisingUNet(interval_size=args.interval_size,
                              afunc=args.afunc, bn=args.bn)
    elif args.model == 'fc2':  # args.task == 'classification'
        model = FC2(interval_size=args.interval_size)

    elif args.model == 'resnet':
        model = DenoisingResNet(interval_size=args.interval_size,
                              afunc=args.afunc, bn=args.bn, num_blocks = args.nblocks,
                              out_channels=args.nfilt, kernel_size=args.width, dilation=args.dil,
                              num_blocks_class=args.nblocksc)

    elif args.model == 'linear':
        model = DenoisingLinear(interval_size=args.interval_size, field=args.field)

    elif args.model == 'logistic':
        model = DenoisingLogistic(interval_size=args.interval_size, field=args.field)

    # TODO: there is a potential problem with loading model on each device like this. keep an eye on torch.load()'s map_location arg
    if args.resume or args.infer or args.eval:
        model = load_model(model, args.weights_path, args.rank)

    model = model.cuda(args.gpu)

    if args.distributed:
        _logger.info('Compiling model in DistributedDataParallel')
        model = DistributedDataParallel(model, device_ids=[args.gpu])
    elif args.gpu > 1:
        _logger.info('Compiling model in DataParallel')
        model = nn.DataParallel(model,device_ids=list(range(args.gpus))).cuda()

    myprint("Finished building.", color='yellow', rank=args.rank)
    return model

def train_worker(gpu, ngpu_per_node, args, timers=None):
    # fix random seed so models have the same starting weights
    torch.manual_seed(42)

    args.rank = gpu if args.distributed else 0
    args.gpu = gpu

    torch.cuda.set_device(args.gpu)
    _logger.debug('Rank %s' % str(args.rank))

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Why is model & optimizer built in spawned function?
    model = build_model(args)
    optimizer = Adam(model.parameters(), lr=args.lr)
    # TODO: LR schedule

    train_dataset = DatasetTrain(args.train_files)
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler,# collate_fn=custom_collate_train,
        drop_last=False
    )

    # TODO: need DatasetVal? Not for now
    val_dataset = DatasetTrain(args.val_files)
    val_sampler = None
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, sampler=val_sampler,# collate_fn=custom_collate_train,
        drop_last=False
        #drop_last=True # need to drop irregular batch for distributed evaluation due to limitation of dist.all_gather
    )


    loss_func = get_losses(args)

    current_best = None
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(rank=args.rank, gpu=args.gpu, task=args.task, model=model, train_loader=train_loader, 
              loss_func=loss_func, optimizer=optimizer, epoch=epoch, epochs=args.epochs, 
              clip_grad=args.clip_grad, print_freq=args.print_freq, distributed=args.distributed, world_size=args.world_size)

        if epoch % args.eval_freq == 0:
            # either create new objects or call reset on each metric obj
            metrics_reg, metrics_cla, best_metric = get_metrics(args)

            # best_metric is the metric used to compare results across different evaluation runs
            # it's modified in place
            evaluate(rank=args.rank, gpu=args.gpu, task=args.task,
                     model=model, val_loader=val_loader,
                     metrics_reg=metrics_reg, metrics_cla=metrics_cla,
                     world_size=args.world_size, distributed=args.distributed,
                     best_metric=best_metric)

            if args.rank == 0:
                new_best = best_metric.better_than(current_best)
                if new_best:
                    current_best = best_metric
                    myprint("New best metric found - {}".format(current_best), color='yellow', rank=args.rank)
                if new_best or epoch % args.save_freq == 0:
                    # give it the module attribute of the model (DistributedDataParallel wrapper)
                    if args.distributed:
                        save_model(model.module, args.exp_dir, args.checkpoint_fname, epoch=epoch, is_best=new_best)
                    else:
                        save_model(model, args.exp_dir, args.checkpoint_fname, epoch=epoch, is_best=new_best)

def infer_worker(gpu, ngpu_per_node, args, res_queue=None):

    args.rank = gpu if args.distributed else 0
    args.gpu = gpu

    torch.cuda.set_device(args.gpu)

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = build_model(args)

    #infer_dataset = DatasetInfer(args.infer_files)
    infer_dataset = DatasetTrain(args.infer_files)
    infer_sampler = None

    if args.distributed:
        infer_sampler = torch.utils.data.distributed.DistributedSampler(infer_dataset)

    infer_loader = torch.utils.data.DataLoader(
        infer_dataset, batch_size=args.bs, shuffle=False, #(infer_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=infer_sampler, drop_last=False#,
        #collate_fn=custom_collate_infer
    )

    infer(rank=args.rank, gpu=args.gpu, task=args.task, model=model, infer_loader=infer_loader,
          print_freq=args.print_freq, res_queue=res_queue)

# Is Eval ever called???
def eval_worker(gpu, ngpu_per_node, args, res_queue=None):

    args.rank = gpu if args.distributed else 0
    args.gpu = gpu

    torch.cuda.set_device(args.gpu)
    _logger.debug('Initialize Eval worker for Node %s' % (str(args.gpu)))

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = build_model(args)

    #eval_dataset = DatasetEval(args.val_files)
    eval_dataset = DatasetTrain(args.val_files)
    eval_sampler = None

    if args.distributed:
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.bs, shuffle=False, #(eval_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=eval_sampler, drop_last=False#,
        #collate_fn=custom_collate_eval
    )

    metrics_reg, metrics_cla, best_metric = get_metrics(args)
    evaluate(rank=args.rank, gpu=args.gpu, task=args.task,
             model=model, val_loader=eval_loader,
             metrics_reg=metrics_reg, metrics_cla=metrics_cla,
             world_size=args.world_size, distributed=args.distributed,
             best_metric=best_metric, res_queue=res_queue)


def writer(args, res_queue):
    result_path = os.path.join(args.exp_dir, args.result_fname)
    myprint("Writer process standing by... %s" % result_path, color='yellow')

    # We need dimensions of the inference dataset.
    # TODO: Clean up this mess...
    if args.infer:
        files_path = args.infer_files
    else:
        assert False, "writer called but infer = False. Not sure what file to write?"
    total_size = 0
    write_dimension = 0
    for fname in files_path:
        with h5py.File(fname, 'r') as f:
            total_size = f["data"].shape[0]
            write_dimension = f["data"].shape[1]
    # Write one or two outputs per item
    total_outputs = 2 if args.task == 'both' else 1
    output_data = np.zeros((total_size, write_dimension,total_outputs), dtype=float)

    count = 0
    outputs_written = np.zeros((total_size), dtype=int)
    with h5py.File(result_path, 'w') as f:
        # Create dataset, of the right size (so we don't have to keep extending)
        df = f.create_dataset("data", data=output_data)
        while 1:
            item = res_queue.get()
            if item == 'done':
                break
            keys, batch = item
            # Write each element to the empty file
            for idx, item in zip(keys, batch):
                if outputs_written[idx] > 0:
                    _logger.error('Danger! Key %d already written in dataset' % idx)
                    continue
                df[idx] = item
                count += 1
                outputs_written[idx] += 1

    # Assert that file is properly populated, all keys are written
    assert np.array_equal(outputs_written, np.ones_like(outputs_written, dtype=int)), "Not all items were inferred. Should have thrown an error..."
    myprint("Dumped results of {} examples to {}".format(count, result_path), color='yellow')

def main():
    args = parse_args()

    # Set log level
    if args.debug:
        _handler.setLevel(logging.DEBUG)
        _logger.setLevel(logging.DEBUG)

    _logger.debug(args)

    # Random seeds for reproducability.
    #set_random_seed(args.seed)

    # check gpu
    # TODO: add cpu support
    if not torch.cuda.is_available():
        raise Exception("No GPU available. Check your machine configuration.")

    # all output will be written in the exp_dir folder
    args.exp_dir = make_experiment_dir(args.label, args.out_home, timestamp=True)

    # train & resume
    ##########################################################################################
    if args.train or args.resume:
        args.train_files = gather_files_from_cmdline(args.train_files)
        args.val_files = gather_files_from_cmdline(args.val_files)
        _logger.debug("Training data:   " + "\n".join(args.train_files))
        _logger.debug("Validation data: " + "\n".join(args.val_files))

        # Get model parameters
        with h5py.File(args.train_files[0], 'r') as f:
            args.interval_size = f['data'].shape[1]
            args.batch_size = 1

        ngpus_per_node = torch.cuda.device_count()

        _logger.info('Distributing to %s GPUS' % str(ngpus_per_node))

        if args.distributed:
            args.world_size = ngpus_per_node
            mp.spawn(train_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args), join=True)
        else:
            assert_device_available(args.gpu)
            args.world_size = 1
            train_worker(args.gpu, ngpus_per_node, args, timers=Timers)

    # infer & eval
    ##########################################################################################
    if args.infer or args.eval:
        files = args.infer_files if args.infer else args.val_files
        files = gather_files_from_cmdline(files)

        # Get model parameters
        with h5py.File(files[0], 'r') as f:
            args.interval_size = f['data'].shape[1]
            args.batch_size = 1

        if args.infer:
            args.infer_files = files
            _logger.debug("Inference data: ", args.infer_files)
        else:
            args.val_files = files
            _logger.debug("Evaluation data: ", args.val_files)


        # setup queue and kick off writer process
        #############################################################
        manager = mp.Manager()
        res_queue = manager.Queue()
        write_proc = mp.Process(target=writer, args=(args, res_queue))
        write_proc.start()
        #############################################################

        ngpus_per_node = torch.cuda.device_count()
        worker = infer_worker if args.infer else eval_worker
        if args.distributed:
            args.world_size = ngpus_per_node
            mp.spawn(worker, nprocs=ngpus_per_node, args=(
                ngpus_per_node, args, res_queue), join=True)
        else:
            assert_device_available(args.gpu)
            args.world_size = 1
            worker(args.gpu, ngpus_per_node, args, res_queue)

        # finish off writing
        #############################################################
        res_queue.put("done")
        _logger.info("Waiting for writer to finish...")
        write_proc.join()
        #############################################################

if __name__ == '__main__':
    main()
