#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""Utility functions."""
import shutil
import os
import time

import h5py
import numpy as np
from datetime import datetime
from termcolor import colored
import torch
import torch.distributed as dist

import yaml


def myprint(msg, color=None, rank=0):
    """Custom print function.

    Args:
        msg: Message to print.
        color: Color to print in.
        rank: --

    """
    if rank == 0:
        print(colored(msg, color))


def safe_make_dir(path):
    """Make directory if it does not exist.

    Args:
        path: Path to the directory to be created.

    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except (OSError, ValueError, TypeError):
        print('Failed to create directory {}: already exists!'.format(path))


def make_experiment_dir(label, out_home, timestamp=True,
                        create_latest_symlink=True):
    """Make output directory. Create appropriate symlinks.

    Args:
        label: Label to attach to the experiment directory.
        out_home: Parent directory to save the experiment directory at.
        timestamp: Attach timestamp to output folder.
        create_latest_symlink: Whether to create a symlink folder with
        latest timestamped folder.

    Return:
        Experiment path string.

    """
    exp_name = label
    if timestamp:
        exp_name = exp_name + datetime.now().strftime("_%Y.%m.%d_%H.%M")

    exp_path = os.path.join(out_home, exp_name)
    safe_make_dir(exp_path)
    # Python doesn't support forced smlink update.
    # Therefore, removing and then making a new symlink.
    if create_latest_symlink:
        latest_symlink = os.path.join(out_home,
                                      "{}_{}".format(label, "latest"))
        if (os.path.islink(latest_symlink)):
            os.remove(latest_symlink)
        os.symlink(os.path.abspath(exp_path), latest_symlink)
    return exp_path


def save_config(config_path, config_params):
    """Save config files used for training.

    Args:
        config_path : Path to config file.
        config_params : object containing config parameters.

    """
    # Only write if config doesn't already exist
    if not os.path.exists(config_path):
        myprint("Saving config file to {}...".format(config_path))
        with open(config_path, "w") as outfile:
            yaml.dump(vars(config_params), outfile)


def save_model(model, exp_dir, filename, epoch, is_best=True):
    """Save model.

    Args:
        model: Model.
        exp_dir: Experiment directory where to store the model
        filename: Filename to store the model by.
        epoch: Current epoch.
        is_best: Whether this is the current best model.

    """
    filename = "epoch{}_{}".format(epoch, filename)
    checkpoint_path = os.path.join(exp_dir, filename)
    myprint("Saving model ckpt to {}...".format(checkpoint_path),
            color='yellow')
    states = {'state_dict': model.state_dict()}
    torch.save(states, checkpoint_path)
    if is_best:
        best_checkpoint_path = os.path.join(exp_dir, 'model_best.pth.tar')
        myprint("Saving best model to {}...".format(best_checkpoint_path),
                color='yellow')
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


def load_model(model, weights_path, rank=None):
    """Load model.

    Args:
        model: Model to load.
        weights_path: Path to where weights of the model are stored.
        rank: Rank.

    """
    try:
        myprint("Loading model weights from {}...".format(weights_path),
                color='yellow', rank=rank)
        checkpoint = torch.load(weights_path, map_location="cuda:" + str(rank))
        model.load_state_dict(checkpoint['state_dict'])
        myprint("Finished loading.", color='yellow', rank=rank)
    except (OSError, ValueError) as e:
        raise Exception(
            "Failed to load weights from path {}: {}".format(weights_path, e))

    return model


def assert_device_available(gpu):
    """Assert that requested GPU device is available.

    Args:
        gpu: Identitiy of GPU device.

    Raise:
        Error if requested device is not available.

    """
    ngpus_per_node = torch.cuda.device_count()
    if gpu >= ngpus_per_node:
        raise ValueError(
            "GPU:{} is not available. "
            "The node has {} GPUs".format(gpu, ngpus_per_node))


def equal_width_formatter(total):
    """Equal width formatter.

    Args:
        total: total width.

    Return:
        formatted string.

    """
    num_digits = len(str(total // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(total) + ']'


def progbar(*, curr, total, progbar_len, pre_bar_msg, post_bar_msg):
    """Progress bar.

    Args:
        curr: Current epoch?
        total: total epochs
        progbar_len: progress bar length
        pre_bar_msg: Message.
        post_bar_msg: Message.

    """
    frac = curr / total
    filled_progbar = round(frac * progbar_len)
    fmt = equal_width_formatter(total)

    print(pre_bar_msg,
          '#' * filled_progbar + '-' * (progbar_len - filled_progbar),
          fmt.format(curr),
          post_bar_msg, end='\n', flush=True)

    # sys.stdout.flush()


def gather_files_from_cmdline(input, extension=".h5"):
    """Gather all input files and return as list.

    Args:
        input: Input file or list of files.
        extension: extension of files

    Return:
        List of input files.

    """
    path = input.strip("[]")
    res = None
    if path == input:
        # a single path is provided; not wrapped in []
        # could be a regular file or a directory
        if os.path.isfile(path):
            assert path.endswith(extension)
            res = [path]
        elif os.path.isdir(path):
            paths = [os.path.join(path, f)
                     for f in os.listdir(path)]
            paths = [f for f in paths if
                     os.path.isfile(f) and f.endswith(extension)]
            res = paths
    else:
        # multiple regular files wrapped in []
        paths = [f.strip() for f in path.split(',')]
        paths = [f for f in paths if os.path.isfile(f) and f.endswith(
            extension)]
        res = paths

    if not res:
        raise Exception("Invalid format for file paths provided.")

    return res


def gather_tensor(tensor, world_size, rank):
    """Gather tensor.

    Args:
        tensor: tensor.
        world_size: world_size.
        rank: rank.

    Return:
        res.

    """
    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    res = None
    if rank == 0:  # only rank 0 needs the result tensor
        res = torch.cat(gather_list, dim=0)
    del gather_list
    return res


def dump_results(task, res, result_path, print_freq=5):
    """Dump results into a h5 file.

    Args:
        task: Regression or classification or both.
        res: result to be dumped
        result_path: Path to h5.
        print_freq: Print every 5 epochs.

    """
    start = time.time()
    num_batches = len(res)
    with h5py.File(result_path, 'w') as f:
        for i, batch_res in enumerate(res):
            if task == "both":
                batch_res = np.stack(batch_res, axis=-1)
            else:
                batch_res = np.expand_dims(batch_res, axis=-1)

            f.create_dataset('batch' + str(i), data=batch_res,
                             compression='lzf')

            if i % print_freq == 0:
                progbar(curr=i, total=num_batches, progbar_len=20,
                        pre_bar_msg="Dumping", post_bar_msg="")

    myprint("Inference result saved to {}".format(result_path), color='yellow')
    myprint("Result dumping time taken: {:8.3f}s".format(time.time() - start),
            color='yellow')


# NOTE -- Compied from Megatron
class Timers:
    """Group of timers."""

    class Timer:
        """Timer."""

        def __init__(self, name):
            """Initialize.

            Args:
                name: Name.

            """
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            """Start the timer."""
            assert not self.started_, 'timer has already been started'
            torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self):
            """Stop the timer."""
            assert self.started_, 'timer is not started'
            torch.cuda.synchronize()
            self.elapsed_ += (time.time() - self.start_time)
            self.started_ = False

        def reset(self):
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self.elapsed_
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        """Initialize."""
        self.timers = {}

    def __call__(self, name):
        """Callable object.

        Args:
            name: Name.

        """
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        print(string)
