# Copyright (C) Shaoshu Yang. All Rights Reserved.
# email: shaoshuyang2020@outlook.com
from engine.utils.metric_logger import MetricLogger
from engine.inference import inference

import logging
import torch
import time


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


def train_worker(cfg, device=None):
    # suppress logging display if not master
    if cfg.MULTIPROC_DIST and device != 0:
        # Capture display logger
        logger = logging.getLogger("kknight-mute")
    else:
        logger = logging.getLogger("kknight")

    if device is not None:
        logger.info("Use GPU: {id} for training".format(id=device))


def do_contrastive_train(
        cfg,
        model,
        data_loader,
        criterion,
        scheduler,
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
        scheduler:
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
    start_epoch_time = time.time()

    # Gradient accumulation interval and statistic display interval
    n_accum_grad = cfg.SOLVER.ACCUM_GRAD
    n_print_intv = n_accum_grad * cfg.SOLVER.BATCH_SIZE
    max_iter = len(data_loader)

    for iteration, (images, _) in enumerate(data_loader):
        data_time += time.time() - end

        if device is not None:
            images[0] = images[0].cuda(device, non_blocking=True)
            images[1] = images[1].cuda(device, non_blocking=True)

        # Compute embedding and target label
        output, target = model(images)
        loss = criterion(output, target)

        # acc1/acc5 are (k + 1)-way constant classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss.backward()


        batch_time += time.time() - end
        end = time.time()

        if iteration % n_accum_grad == 0 or iteration == max_iter:
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

            data_time, batch_time = 0, 0

        if iteration % n_print_intv == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.joint(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}"
                    ]
                )
            )


    epoch_time = time.time() - start_epoch_time