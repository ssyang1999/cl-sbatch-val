# Copyright (C) Shaoshu Yang. All Rights Reserved.
# email: shaoshuyang2020@outlook.com
from engine.utils.metric_logger import MetricLogger
from engine.inference import inference

import datetime
import logging
import time
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_worker(device, ngpus_per_node, cfg):
    r"""Distributed parallel training worker

    """
    # suppress logging display if not master
    if cfg.MULTIPROC_DIST and device != 0:
        # Capture display logger
        logger = logging.getLogger("kknight-mute")
    else:
        logger = logging.getLogger("kknight")

    if device is not None:
        logger.info("Use GPU: {id} for training".format(id=device))

    if cfg.DISTRIBUTED:
        if cfg.DIST_URL == "envs://" and cfg.RANK == -1:
            cfg.RANK = int(os.environ["RANK"])
        if cfg.MULTIPROC_DIST:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg.RANK = cfg.RANK * ngpus_per_node + device
        dist.init_process_group(backend=cfg.DIST_BACKEND, init_method=cfg.DIST_URL,
                                world_size=cfg.WORLD_SIZE, rank=cfg.RANK)

    # Create model
    logger.info("Creating model.")
    # TODO: call model builder
    model = nn.Conv2d(1, 1, 1)

    if cfg.DISTRIBUTED:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if device is not None:
            torch.cuda.set_device(device)
            model.cuda(device)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            cfg.SOLVER.BATCH_PER_NODE = int(cfg.SOLVER.BATCH_SIZE / ngpus_per_node)
            cfg.DATALOADER.NUM_WORKERS = int((cfg.DATALOADER.NUM_WORKERS + ngpus_per_node - 1)
                                             / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif device is not None:
        torch.cuda.set_device(device)
        model = model.cuda(device)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    # TODO: criterion factory
    criterion = nn.CrossEntropyLoss().cuda(device=device)

    # TODO: optimizer factory
    # TODO: checkpointer
    # TODO: dataset and transforms
    # TODO: dataloader and sampler

    # TODO: epoch-wise training pipeline


def do_contrastive_train(
        cfg,
        model,
        data_loader,
        criterion,
        optimizer,
        epoch,
        device = None,
        meters = None,
        logger = None,
):
    r"""Contrastive training implementation:

    Args:
        cfg: (CfgNode) Specified configuration. The final config settings
    integrates the initial default setting and user defined settings given
    by argparse.
        model: (nn.Module) Under the context of this repository, the model
    can be a simple PyTorch neural network or the instance wrapped by
    ContrastiveWrapper.
        data_loader:
        criterion:
        optimizer:
        device:
        epoch:
        meters:

    Returns:

    """
    # Capture display logger
    # logger = logging.getLogger("kknight")
    logger.info("Epoch {epoch} now started.".format(epoch = epoch))

    # Switch to train mode
    model.train()

    # Timers
    end = time.time()
    data_time, batch_time = 0, 0

    # Gradient accumulation interval and statistic display interval
    n_accum_grad = cfg.SOLVER.ACCUM_GRAD
    n_print_intv = n_accum_grad * cfg.SOLVER.BATCH_SIZE
    max_iter = len(data_loader)

    # Estimated time of arrival of remaining epoch
    total_eta = meters.time.global_avg * max_iter * (cfg.SOLVER.EPOCH - epoch)

    for iteration, ((xis, xjs), _) in enumerate(data_loader):
        data_time += time.time() - end

        if device is not None:
            xis = xis.cuda(device, non_blocking=True)
            xjs = xjs.cuda(device, non_blocking=True)

        # Compute embedding and target label
        output, target, extra = model(xis, xjs)
        loss = criterion(output, target)

        # acc1/acc5 are (k + 1)-way constant classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        meters.update(loss=loss, **extra)
        meters.update(acc1=acc1, acc5=acc5)
        loss.backward()

        # Compute batch time
        batch_time += time.time() - end
        end = time.time()

        eta_seconds = meters.time.global_avg * (max_iter - iteration) + total_eta
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % n_accum_grad == 0 or iteration == max_iter:
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            # Record batch time and data sampling time
            meters.update(time=batch_time, data=data_time)
            data_time, batch_time = 0, 0

        if iteration % n_print_intv == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.joint(
                    [
                        "eta: {eta}",
                        "epoch: {epoch}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}"
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024. / 1024.,
                )
            )
