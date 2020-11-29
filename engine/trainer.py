# Copyright (C) Shaoshu Yang. All Rights Reserved.
# email: shaoshuyang2020@outlook.com
from engine.data.transforms import GaussianBlur, TwoCropsTransfrom
from engine.data.build import build_dataset, DatasetCatalog
from engine.utils.metric_logger import MetricLogger
from engine.utils.logger import GroupedLogger
from engine.solver import WarmupMultiStepLR
from engine.utils.checkpoint import Checkpointer
from engine.contrastive.build import build_contrastive_model
from engine.utils.logger import setup_logger, setup_mute_logger
from engine.inference import contrastive_inference, lincls_inference
from engine.data.eval import contrastive_accuracy
from engine.data.samplers import DropLastDistributedSampler
from engine.contrastive.simclr import SimCLRModel

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


def build_metric_logger():
    pass


def lincls_train_worker(device, ngpus_per_node, cfg):
    r"""Distributed parallel training worker

        """
    # suppress logging display if not master
    if cfg.MULTIPROC_DIST and device != 0:
        # Capture display logger
        logger = setup_mute_logger("kknight-mute")
    else:
        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)
        logger = setup_logger("kknight", cfg.OUTPUT_DIR)
    logger.info(cfg)

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
    # TODO: call model builderi
    if cfg.MODEL.CONTRASTIVE == "moco":
        model = models.__dict__[cfg.MODEL.ARCH](num_classes=10)

        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False

        # initialize fc layer
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
    elif cfg.MODEL.CONTRASTIVE == "simclr":
        model = SimCLRModel(num_classes=10)
        # _state_dict = model.state_dict()

        for name, param in model.named_parameters():
            if name not in ['headding.wight', 'headding.bias']:
                param.requires_grad = False

    # model = build_contrastive_model(cfg, device=device)

    if os.path.isfile(cfg.PRETRAINED):
        logger.info("Loading pretrained model from {}".format(cfg.PRETRAINED))
        checkpoint = torch.load(cfg.PRETRAINED, map_location="cpu")

        # Load pretrained model
        state_dict = checkpoint["model"]
        if cfg.MODEL.CONTRASTIVE == "moco":
            for k in list(state_dict.keys()):
                # Copy the module named module
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        elif cfg.MODEL.CONTRASTIVE == "simclr":
            local_dict = model.state_dict()

            for k in list(state_dict.keys()):
                if not k.startswith('features'):
                    del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"headding.0.weight", "headding.0.bias"}

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
            cfg.EVAL.BATCH_PER_NODE = int(cfg.EVAL.BATCH_SIZE / ngpus_per_node)
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
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    # Build up criterion and deploy on corresponding device
    factory = getattr(loss, cfg.CRITERION)
    criterion = factory().cuda(device=device)

    # Optimizer
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(parameters, cfg.SOLVER.BASE_LR, betas=cfg.SOLVER.BETAS,
                                 weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    # Learn rate scheduler
    scheduler = WarmupMultiStepLR(optimizer,
                                  milestones=cfg.SOLVER.MILESTONES,
                                  gamma=cfg.SOLVER.GAMMA,
                                  warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                                  warmup_iters=cfg.SOLVER.WARMUP_EPOCHS,
                                  warmup_method=cfg.SOLVER.WARMUP_METHOD)

    # Checkpoint the model, optimizer and learn rate scheduler if is master node
    arguments = dict()
    arguments["epoch"] = 0
    # checkpointer = None
    # if not cfg.MULTIPROC_DIST or (cfg.MULTIPROC_DIST and cfg.RANK % ngpus_per_node == 0):
    checkpointer = Checkpointer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=cfg.OUTPUT_DIR,
        save_to_disk=True,
        logger=logger
    )

    # read from checkpoint
    extra_checkpoint_data = checkpointer.load(cfg.CHECKPOINT)
    arguments.update(extra_checkpoint_data)
    start_epoch = arguments["epoch"]

    # TODO: dataset and transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trans = transforms.Compose([
            transforms.RandomResizedCrop(112, scale=(0.7, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    train_dataset = build_dataset(cfg.DATA.TRAIN, trans, DatasetCatalog, is_train=True)


    trans = transforms.Compose([
            transforms.Resize(112),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            normalize,
        ])
    test_dataset = build_dataset(cfg.DATA.TEST, trans, DatasetCatalog, is_train=False)

    # TODO: dataloader and sampler
    if cfg.DISTRIBUTED:
        train_sampler = DropLastDistributedSampler(train_dataset, batch_size=cfg.SOLVER.BATCH_SIZE)
        test_sampler = DropLastDistributedSampler(test_dataset, batch_size=cfg.SOLVER.BATCH_SIZE)
    else:
        train_sampler = None
        test_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.SOLVER.BATCH_PER_NODE,
        shuffle=(train_sampler is None),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.EVAL.BATCH_PER_NODE,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=test_sampler,
    )

    # TODO: metic logger or tensorboard logger
    meters = MetricLogger(delimiter="  ")
    meters_val = MetricLogger(delimiter="  ")

    # TODO: epoch-wise training pipeline
    for epoch in range(start_epoch, cfg.SOLVER.EPOCH):
        if cfg.DISTRIBUTED:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        do_lincls_train(
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

        if (epoch + 1) % cfg.EVAL.EVAL_INTERVAL == 0:
            metric = lincls_inference(
                cfg=cfg,
                model=model,
                data_loader=test_loader,
                device=device,
                logger=logger,
            )
            meters_val.update(**metric)
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Evaluation result]: ",
                        "epoch: {epoch}",
                        "{meters}",
                    ]
                ).format(
                    epoch=epoch,
                    meters=str(meters_val),
                )
            )

        scheduler.step()
        arguments["epoch"] = epoch
        # Produce checkpoint
        if not cfg.MULTIPROC_DIST or (cfg.MULTIPROC_DIST and cfg.RANK % ngpus_per_node == 0):
            checkpointer.save(
                "checkpoint_{:03d}".format(epoch), **arguments
            )
        if epoch == start_epoch:
            sanity_check(cfg.MODEL.CONTRASTIVE, model.state_dict(), cfg.PRETRAINED)


def sanity_check(contrastive, state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['model']

    if contrastive == "moco":
        for k in list(state_dict.keys()):
            # only ignore fc layer
            if 'fc.weight' in k or 'fc.bias' in k:
                continue

            # name in pretrained model
            k_pre = 'module.encoder_q.' + k[len('module.'):] \
                if k.startswith('module.') else 'module.encoder_q.' + k

            assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
                '{} is changed in linear classifier training.'.format(k)
    elif contrastive == "simclr":
        for k in list(state_dict.keys()):
            if 'headding' in k:
                continue

            assert ((state_dict[k].cpu() == state_dict_pre[k]).all()), \
                '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


def do_lincls_train(
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
    # Capture display logger
    # logger = logging.getLogger("kknight")
    logger.info("Epoch {epoch} now started.".format(epoch=epoch))

    # Switch to train mode
    model.eval()

    # Timers
    end = time.time()
    data_time, batch_time = 0, 0

    # Gradient accumulation interval and statistic display interval
    n_accum_grad = cfg.SOLVER.ACCUM_GRAD
    n_print_intv = n_accum_grad * cfg.SOLVER.DISP_INTERVAL
    max_iter = len(data_loader)

    for iteration, (images, target) in enumerate(data_loader):
        data_time += time.time() - end

        if device is not None:
            images = images.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)

        # Compute embedding and target label
        # output, target, extra = model(xis, xjs)
        output = model(images)
        loss = criterion(output, target)

        # acc1/acc5 are (k + 1)-way constant classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = contrastive_accuracy(output, target, topk=(1, 5))
        # meters.update(loss=loss, **extra)
        meters.update(loss=loss)
        meters.update(acc1=acc1, acc5=acc5)
        loss.backward()

        # Compute batch time
        batch_time += time.time() - end
        end = time.time()

        if (iteration + 1) % n_accum_grad == 0 or iteration + 1 == max_iter:
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            # Record batch time and data sampling time
            meters.update(time=batch_time, data=data_time)
            data_time, batch_time = 0, 0

        if (iteration + 1) % n_print_intv == 0 or iteration == max_iter:
            # Estimated time of arrival of remaining epoch
            total_eta = meters.time.global_avg * max_iter * (cfg.SOLVER.EPOCH - epoch)
            eta_seconds = meters.time.global_avg * (max_iter - iteration) + total_eta
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            logger.info(
                meters.delimiter.join(
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


def train_worker(device, ngpus_per_node, cfg):
    r"""Distributed parallel training worker

    """
    # suppress logging display if not master
    if cfg.MULTIPROC_DIST and device != 0:
        # Capture display logger
        logger = setup_mute_logger("kknight-mute")
    else:
        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)
        logger = setup_logger("kknight", cfg.OUTPUT_DIR)
    logger.info(cfg)

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
            cfg.EVAL.BATCH_PER_NODE = int(cfg.EVAL.BATCH_SIZE / ngpus_per_node)
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
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
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
                                  warmup_iters=cfg.SOLVER.WARMUP_EPOCHS,
                                  warmup_method=cfg.SOLVER.WARMUP_METHOD)

    # Checkpoint the model, optimizer and learn rate scheduler if is master node
    arguments = dict()
    arguments["epoch"] = 0
    # checkpointer = None
    # if not cfg.MULTIPROC_DIST or (cfg.MULTIPROC_DIST and cfg.RANK % ngpus_per_node == 0):
    checkpointer = Checkpointer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=cfg.OUTPUT_DIR,
        save_to_disk=True,
        logger=logger
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
    test_dataset = build_dataset(cfg.DATA.TEST, trans, DatasetCatalog, is_train=False)

    # TODO: dataloader and sampler
    if cfg.DISTRIBUTED:
        train_sampler = DropLastDistributedSampler(train_dataset, batch_size=cfg.SOLVER.BATCH_SIZE)
        test_sampler = DropLastDistributedSampler(test_dataset, batch_size=cfg.SOLVER.BATCH_SIZE)
    else:
        train_sampler = None
        test_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.SOLVER.BATCH_PER_NODE,
        shuffle=(train_sampler is None),
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.EVAL.BATCH_PER_NODE,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        sampler=test_sampler,
        drop_last=True,
    )

    # TODO: metic logger or tensorboard logger
    meters = MetricLogger(delimiter="  ")
    meters_val = MetricLogger(delimiter="  ")

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

        if (epoch + 1) % cfg.EVAL.EVAL_INTERVAL == 0:
            metric = contrastive_inference(
                cfg=cfg,
                model=model,
                data_loader=test_loader,
                device=device,
                logger=logger,
            )
            meters_val.update(**metric)
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Evaluation result]: ",
                        "epoch: {epoch}",
                        "{meters}",
                    ]
                ).format(
                    epoch=epoch,
                    meters=str(meters_val),
                )
            )

        scheduler.step()
        arguments["epoch"] = epoch + 1
        # Produce checkpoint
        if not cfg.MULTIPROC_DIST or (cfg.MULTIPROC_DIST and cfg.RANK % ngpus_per_node == 0):
            checkpointer.save(
                "checkpoint_{:03d}".format(epoch), **arguments
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
    n_print_intv = n_accum_grad * cfg.SOLVER.DISP_INTERVAL
    max_iter = len(data_loader)

    for iteration, ((xis, xjs), _) in enumerate(data_loader):
        data_time += time.time() - end

        if device is not None:
            xis = xis.cuda(device, non_blocking=True)
            xjs = xjs.cuda(device, non_blocking=True)

        # Compute embedding and target label
        # output, target, extra = model(xis, xjs)
        output, target = model(xis, xjs)
        loss = criterion(output, target)

        # acc1/acc5 are (k + 1)-way constant classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = contrastive_accuracy(output, target, topk=(1, 5))
        # meters.update(loss=loss, **extra)
        meters.update(loss=loss)
        meters.update(acc1=acc1, acc5=acc5)
        loss.backward()

        # Compute batch time
        batch_time += time.time() - end
        end = time.time()

        if (iteration + 1) % n_accum_grad == 0 or iteration + 1 == max_iter:
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            # Record batch time and data sampling time
            meters.update(time=batch_time, data=data_time)
            data_time, batch_time = 0, 0

        if (iteration + 1) % n_print_intv == 0 or iteration == max_iter:
            # Estimated time of arrival of remaining epoch
            total_eta = meters.time.global_avg * max_iter * (cfg.SOLVER.EPOCH - epoch)
            eta_seconds = meters.time.global_avg * (max_iter - iteration) + total_eta
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            logger.info(
                meters.delimiter.join(
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
