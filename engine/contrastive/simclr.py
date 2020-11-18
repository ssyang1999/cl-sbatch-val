from engine.contrastive.moco import concat_all_gather
import engine.modeling.build as build

import torch.nn as nn
import torch

import numpy as np


class SimCLR(nn.Module):
    def __init__(self, cfg, device):
        super(SimCLR, self).__init__()

        # SimCLR options
        self.T = cfg.MODEL.SIMCLR_T

        # Build up feature extractor, while erasing the last fc layer
        features = build.__dict__[cfg.MODEL.ARCH](num_classes=cfg.MODEL.SIMCLR_DIM)

        # Get hidden vector dimension
        n_features = features.fc.weight.shape[1]
        self.features = nn.Sequential(*list(features.children())[:-1])

        self.headding = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, cfg.MODEL.SIMCLR_DIM)
        )

        # Similarity function
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.batch_size = cfg.SOLVER.BATCH_SIZE

        # Negative logits mask
        self.device = device
        self.mask = self._get_correlated_mask().type(torch.bool)

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.cuda(self.device)

    def forward(self, xis, xjs):
        # Compute hidden embedding and representation
        his = self.features(xis)
        his = torch.flatten(his, 1)
        # Representation dimension: N x dim
        xis = self.headding(his)

        hjs = self.features(xjs)
        hjs = torch.flatten(hjs, 1)
        xjs = self.headding(hjs)    # N x dim

        # Gather all features from a batch
        concat_all_gather(xis)
        concat_all_gather(xjs)

        # Compute similarity function
        represnetations = torch.cat([xis, xjs], dim=0)
        similarity_matrix = self.similarity(represnetations, represnetations)

        # Extract logits
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)

        # Positive logits: 2N x 1
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        # Negative logits: 2N x (2N - 1)
        negatives = similarity_matrix[self.mask].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.T

        # Labels: 2N x 1
        labels = torch.zeros(2 * self.batch_size).cuda(self.device).long()
        return logits, labels

