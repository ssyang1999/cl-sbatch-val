# This implementation has referred to the original MoCo.
import engine.modeling.build as build

import torch.nn as nn
import torch


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class MoCo(nn.Module):
    def __init__(self, cfg, device):
        super(MoCo, self).__init__()

        # MoCo parameters
        self.K = cfg.MODEL.MOCO_K
        self.m = cfg.MODEL.MOCO_M
        self.T = cfg.MODEL.MOCO_T
        self.mlp = cfg.MODEL.MOCO_MLP

        self.device = device

        # Encoders
        self.encoder_q = build.__dict__[cfg.MODEL.ARCH](num_classes=cfg.MODEL.MOCO_DIM)
        self.encoder_k = build.__dict__[cfg.MODEL.ARCH](num_classes=cfg.MODEL.MOCO_DIM)

        if self.mlp:    # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),
                                              self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),
                                              self.encoder_k.fc)

        # Initialize encoder k with parameters of q
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # MoCo negative key queue. Initialized with normalized random vector
        self.register_buffer("queue", torch.randn(cfg.MODEL.MOCO_DIM, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda(self.device)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, xis, xjs):
        # query embeddings
        q = self.encoder_q(xis)         # queires: N x dim
        q = nn.functional.normalize(q, dim=1)

        # key embeddings
        with torch.no_grad():
            # update encoder k before computing
            self._momentum_update_key_encoder()

            # shuffle batch normalization
            im_k, idx_unshuffle = self._batch_shuffle_ddp(xjs)

            k = self.encoder_k(im_k)    # keys: N x dim
            k = nn.functional.normalize(k, dim=1)

            # unshuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # Cosine similarity, computed by Einstein sum
        # Positive logits N x 1
        l_pos = torch.einsum('ij, ij -> i', [q, k]).unsqueeze(-1)

        # Negative logits N x K
        l_neg = torch.einsum('ij, jk -> ik', [q, self.queue.clone().detach()])

        # logits and ground truth
        logits = torch.cat([l_pos, l_neg], dim=1)   # N x (k + 1)
        logits /= self.T

        # Labels N x 1
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(self.device)

        # enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
