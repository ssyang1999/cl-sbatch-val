
import torch
import torch.nn as nn


class ContrastiveWrapper(nn.Module):
    def __init__(self, cfg, similarity_function):
        super(ContrastiveWrapper, self).__init__()
        # self.model =
        self.similarity_function = similarity_function

    def forward(self, xis, xjs):
        raise NotImplementedError(".")