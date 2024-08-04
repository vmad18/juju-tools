import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional, Iterable, Callable, Dict
from math import sqrt, ceil, log, pi
import math
from einops import rearrange

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -> %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

true, false, null = True, False, None
DEF_MAX_TOK = 2048
DEF_MAX_BSZ = 8
