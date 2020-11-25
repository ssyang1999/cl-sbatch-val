from engine.utils.timer import Timer, get_time_str
from engine.data.eval import contrastive_accuracy
from engine.utils.metric_logger import MetricLogger
from collections import OrderedDict

import datetime
import torch.distributed as distributed
import torch


def lincls_inference(
        cfg,
        model,
        data_loader,
        # dataset_name,
        device=None,
        logger=None
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    dataset = data_loader.dataset

    n_print_intv = cfg.SOLVER.DISP_INTERVAL
    max_iter = len(dataset)
    logger.info("Start evaluation on {} images".format(len(dataset)))
    # logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    model.eval()

    meters = MetricLogger(delimiter="  ")
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    for iter_val, (images, target) in enumerate(data_loader):

        if device is not None:
            images = images.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)

        # Compute embedding and target label
        # output, target, extra = model(xis, xjs)
        inference_timer.tic()
        output = model(images)
        inference_timer.toc()

        # acc1/acc5 are (k + 1)-way constant classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = contrastive_accuracy(output, target, topk=(1, 5))
        meters.update(acc1=acc1, acc5=acc5)

        if (iter_val + 1) % n_print_intv == 0 or iter_val == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "[Evaluation]",
                        "iter: {iter}",
                        "{meters}",
                        "max mem: {memory:.0f}"
                    ]
                ).format(
                    iter=iter_val,
                    meters=str(meters),
                    memory=torch.cuda.max_memory_allocated() / 1024. / 1024.,
                )
            )

    # wait for all processes to complete before measuring the time
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {}".format(
            total_time_str,
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ".format(
            total_infer_time,
        )
    )

    if cfg.DISTRIBUTED:
        metrics = torch.tensor(list(map(lambda q: q.global_avg, meters.meters.values()))).cuda(device)
        distributed.all_reduce(metrics)
        metrics = metrics.cpu() / cfg.WORLD_SIZE

        return {k: v.item() for k, v in zip(meters.meters.keys(), metrics)}

    return {k: v.global_avg for k, v in meters.meters.items()}


def contrastive_inference(
        cfg,
        model,
        data_loader,
        # dataset_name,
        device=None,
        logger=None
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    dataset = data_loader.dataset

    n_print_intv = cfg.SOLVER.DISP_INTERVAL
    max_iter = len(data_loader)
    logger.info("Start evaluation on {} images".format(len(dataset)))
    # logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    model.eval()

    meters = MetricLogger(delimiter="  ")
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    for iter_val, ((xis, xjs), _) in enumerate(data_loader):

        if device is not None:
            xis = xis.cuda(device, non_blocking=True)
            xjs = xjs.cuda(device, non_blocking=True)

        # Compute embedding and target label
        # output, target, extra = model(xis, xjs)
        inference_timer.tic()
        output, target = model(xis, xjs)
        inference_timer.toc()

        # acc1/acc5 are (k + 1)-way constant classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = contrastive_accuracy(output, target, topk=(1, 5))
        meters.update(acc1=acc1, acc5=acc5)

        if (iter_val + 1) % n_print_intv == 0 or iter_val == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "[Evaluation]",
                        "iter: {iter}",
                        "{meters}",
                        "max mem: {memory:.0f}"
                    ]
                ).format(
                    iter=iter_val,
                    meters=str(meters),
                    memory=torch.cuda.max_memory_allocated() / 1024. / 1024.,
                )
            )

    # wait for all processes to complete before measuring the time
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {}".format(
            total_time_str,
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ".format(
            total_infer_time,
        )
    )

    if cfg.DISTRIBUTED:
        metrics = torch.tensor(list(map(lambda q: q.global_avg, meters.meters.values()))).cuda(device)
        distributed.all_reduce(metrics, op=distributed.ReduceOp.SUM)
        metrics = metrics.cpu() / cfg.WORLD_SIZE

        return {k: v.item() for k, v in zip(meters.meters.keys(), metrics)}

    return {k: v.global_avg for k, v in meters.meters.items()}
