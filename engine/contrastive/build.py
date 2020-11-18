# Copyright (c) Shaoshu Yang 2020

from .moco import MoCo
from .simclr import SimCLR

_CONTRASTIVE_METHODS = {
    "moco": MoCo,
    "simclr": SimCLR,
}


def build_contrastive_model(cfg, **kwargs):
    meta_arch = _CONTRASTIVE_METHODS[cfg.MODEL.CONTRASTIVE]
    return meta_arch(cfg, **kwargs)