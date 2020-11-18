# Copyright (C) Shaoshu Yang. All Rights Reserved.
# email: shaoshuyang2020@outlook.com
from engine.utils.metric_logger import MetricLogger
from engine.inference import inference


def train(
        cfg,
        model,
        data_loader,
        criterion,
        scheduler,
        optimizer,
        device,
        epoch,
        logger = None,
):
    '''
    Contrastive training implementation.

    :param cfg: (CfgNode) Specified configuration. The final config settings
    integrates the initial default setting and user defined settings given
    by argparse.
    :param model: (nn.Module) Under the context of this repository, the model
    can be a simple PyTorch neural network or the instance wrapped by
    ContrastiveWrapper.
    :param data_loader:
    :param criterion:
    :param scheduler:
    :param optimizer:
    :param device:
    :param epoch:
    :param logger:
    :return:
    '''
    pass
