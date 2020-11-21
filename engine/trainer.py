# Copyright (C) Shaoshu Yang. All Rights Reserved.
# email: shaoshuyang2020@outlook.com
from engine.data.transforms import GaussianBlur, TwoCropsTransfrom
from engine.data.build import build_dataset, DatasetCatalog
from engine.utils.metric_logger import MetricLogger
from engine.utils.logger import GroupedLogger
from engine.solver import WarmupMultiStepLR
from engine.utils.checkpoint import Checkpointer
from engine.contrastive.build import build_contrastive_model
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
import torch.nn.modules.loss as loss
import torch.optim as optim
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


def build_metric_logger():
    pass


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
    model = build_contrastive_model(cfg, device=device)

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
    # Build up criterion and deploy on corresponding device
    factory = getattr(loss, cfg.CRITERION)
    criterion = factory().cuda(device=device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), cfg.SOLVER.BASE_LR, betas=cfg.SOLVER.BETAS,
                                 weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    # Learn rate scheduler
    scheduler = WarmupMultiStepLR(optimizer,
                                  milestones=cfg.SOLVER.MILESTONES,
                                  gamma=cfg.SOLVER.GAMMA,
                                  warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                                  warmup_iters=cfg.SOLVER.WARMUP_EPOCHES,
                                  warmup_method=cfg.SOLVER.WARMUP_METHOD)

    # Checkpoint the model, optimizer and learn rate scheduler if is master node
    arguments = dict()
    arguments["epoch"] = 0
    # checkpointer = None
    # if not cfg.MULTIPROC_DIST or (cfg.MULTIPROC_DIST and cfg.RANK % ngpus_per_node == 0):
    checkpointer = Checkpointer(
        model, optimizer, scheduler, save_dir=cfg.OUTPUT_DIR, save_to_disk=True
    )

    # read from checkpoint
    extra_checkpoint_data = checkpointer.load(cfg.CHECKPOINT)
    arguments.update(extra_checkpoint_data)
    start_epoch = arguments["epoch"]

    # TODO: dataset and transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentation = [
        transforms.RandomResizedCrop(112, scale=(0.7, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    trans = TwoCropsTransfrom(transforms.Compose(augmentation))
    train_dataset = build_dataset(cfg.DATA.TRAIN, trans, DatasetCatalog, is_train=True)

    # TODO: dataloader and sampler
    if cfg.DISTRIBUTED:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_PER_NODE,
        shuffle=(train_sampler is None),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    # TODO: metic logger or tensorboard logger
    meters = MetricLogger(delimiter="  ")

    # TODO: epoch-wise training pipeline
    for epoch in range(start_epoch, cfg.SOLVER.EPOCH):
        if cfg.DISTRIBUTED:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        do_contrastive_train(
            cfg=cfg,
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            meters=meters,
            logger=logger
        )

        scheduler.step(epoch=epoch)

        # Produce checkpoint
        if not cfg.MULTIPROC_DIST or (cfg.MULTIPROC_DIST and cfg.RANK % ngpus_per_node == 0):
            checkpointer.save(
                "checkpoint_{:03d}".format(epoch),
            )


def do_contrastive_train(
        cfg,
        model,
        data_loader,
        criterion,
        optimizer,
        epoch,
        device=None,
        meters=None,
        logger=None,
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
        # output, target, extra = model(xis, xjs)
        output, target = model(xis, xjs)
        loss = criterion(output, target)
        loss.backward()

        # acc1/acc5 are (k + 1)-way constant classifier accuracy
        # measure accuracy and record loss
        if meters:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            meters.update(loss=loss)
            # meters.update(loss=loss, **extra)
            meters.update(acc1=acc1, acc5=acc5)

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
            if meters:
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
