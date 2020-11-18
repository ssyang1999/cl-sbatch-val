import engine.modeling.build as build

import torch.nn as nn
import torch


class SimCLR(nn.Module):
    def __init__(self, cfg):
        self.features = build.__dict__[cfg.MODEL.ARCH](num_classes=cfg.MODEL.SIMCLR_DIM)
