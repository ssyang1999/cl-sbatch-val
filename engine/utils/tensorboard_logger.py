# Copyright (c) Shaoshu Yang, 2020
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from collections import deque

import torch


class TensorboardLogger(object):
    def __init__(self, log_dir=None, purge_step=None, max_queue=10, flush_secs=120):
        self.writer = SummaryWriter(
            log_dir=log_dir,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
        )

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)