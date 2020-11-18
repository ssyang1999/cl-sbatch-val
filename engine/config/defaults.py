import os
from yacs.config import CfgNode as CN

r"""Learning algorithm settings 
"""

_C = CN()

# --------------------------------------------------------------------- #
#                    computation device & media options                 #
# --------------------------------------------------------------------- #
_C.GPU = None
_C.DISTRIBUTED = True

_C.MULTIPROC_DIST = True
_C.DIST_URL = 'tcp://224.66.41.62:23456'
_C.DIST_BACKEND = 'nccl'
_C.WORLD_SIZE = 1
_C.RANK = 1

# --------------------------------------------------------------------- #
#                          miscellaneous options                        #
# --------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.LOGGER = ('MetricLogger', 'TensorboardLogger')

# --------------------------------------------------------------------- #
#                              model options                            #
# --------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.CONTRASTIVE = "moco"
_C.MODEL.SIMILARITY = "consine"

# Architecture definition mode, can be script/mdg
_C.MODEL.ARCH_MODE = "script"

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
_C.SOLVER.ACCUM_GRAD = 1

_C.SOLVER.OPTIMIZER = "SGD"
_C.SOLVER.BASE_LR = 0.001


# --------------------------------------------------------------------- #
#                            input options                              #
# --------------------------------------------------------------------- #
_C.INPUT = CN()

# --------------------------------------------------------------------- #
#                             data options                              #
# --------------------------------------------------------------------- #
_C.DATA = CN()

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
