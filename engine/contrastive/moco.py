import engine.modeling.build as build

import torch.nn as nn
import torch


class MoCo(nn.Module):
    def __init__(self, cfg):
        super(MoCo, self).__init__()

        # MoCo parameters
        self.K = cfg.MODEL.MOCO_K
        self.m = cfg.MODEL.MOCO_M
        self.T = cfg.MODEL.MOCO_T

        # Encoders
        self.encoder_q = build.__dict__[cfg.MODEL.ARCH](num_classes=cfg.MODEL.MOCO_DIM)
        self.encoder_k = build.__dict__[cfg.MODEL.ARCH](num_classes=cfg.MODEL.MOCO_DIM)

