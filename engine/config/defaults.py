import os
from yacs.config import CfgNode as CN

r"""Learning algorithm settings 
"""

_C = CN()

# --------------------------------------------------------------------- #
#                    computation device & media options                 #
# --------------------------------------------------------------------- #
_C.GPU = None                  # Only gpu mode is supported now
_C.DISTRIBUTED = True

_C.MULTIPROC_DIST = True
_C.DIST_URL = 'tcp://224.66.41.62:23456'
_C.DIST_BACKEND = 'nccl'
_C.WORLD_SIZE = 1
_C.RANK = 0

# --------------------------------------------------------------------- #
#                          miscellaneous options                        #
# --------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.CHECKPOINT = "."
_C.LOGGER = ('MetricLogger', 'TensorboardLogger')

# --------------------------------------------------------------------- #
#                              model options                            #
# --------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.CONTRASTIVE = "moco"
_C.MODEL.SIMILARITY = "consine"

# MoCo settings
_C.MODEL.MOCO_DIM = 128     # feature dimension
_C.MODEL.MOCO_K = 65536     # queue size
_C.MODEL.MOCO_M = 0.999     # momentum
_C.MODEL.MOCO_T = 0.07      # softmax temperature
_C.MODEL.MOCO_MLP = True    # use mlp projection header

# SimCLR settings
_C.MODEL.SIMCLR_DIM = 128   # feature dimension
_C.MODEL.SIMCLR_T = 0.5

# Architecture definition mode, can be torchvision/script/mdg
_C.MODEL.ARCH_MODE = "torchvision"
_C.MODEL.ARCH = "resnet18"

# --------------------------------------------------------------------- #
#                           criterion options                           #
# --------------------------------------------------------------------- #
_C.CRITERION = "CrossEntropyLoss"

# --------------------------------------------------------------------- #
#                     solver & optimization options                     #
# --------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCH = 200
_C.SOLVER.BATCH_SIZE = 256
_C.SOLVER.BATCH_PER_NODE = 256
_C.SOLVER.ACCUM_GRAD = 1

_C.SOLVER.OPTIMIZER = "Adam"
_C.SOLVER.BASE_LR = 0.001

# Adam optimizer setting
_C.SOLVER.BETAS = (0.9, 0.999)
_C.SOLVER.WEIGHT_DECAY = 1e-6

# Multi-step learn rate scheduler
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.MILESTONES = (120, 160)

# Learn rate warm up options
_C.SOLVER.WARMUP_FACTOR = 1.0 / 5
_C.SOLVER.WARMUP_EPOCHS = 10
_C.SOLVER.WARMUP_METHOD = "linear"      # linear or constant

# --------------------------------------------------------------------- #
#                            input options                              #
# --------------------------------------------------------------------- #
_C.INPUT = CN()

# --------------------------------------------------------------------- #
#                             data options                              #
# --------------------------------------------------------------------- #
_C.DATA = CN()
_C.DATA.TRAIN = ("cifar10",)

# --------------------------------------------------------------------- #
#                         data loader options                           #
# --------------------------------------------------------------------- #
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 32

# --------------------------------------------------------------------- #
#                          evaluation options                           #
# --------------------------------------------------------------------- #
_C.EVAL = CN()
