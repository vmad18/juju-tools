from juju_tools.utils.layers.quant.bit_net import *
from juju_tools.utils.layers.quant.mhq_attn import *
from juju_tools.utils.layers.quant.nfk import *
from juju_tools.utils.layers.quant.qlora import *
from juju_tools.utils.layers.quant.quant_int8 import *

def compute_normal_fpk(bits, e=.967752) -> Tensor:
    half_1 = 2 ** bits // 2
    half_2 = 2 ** bits // 2 + 1

    v1 = (-norm.ppf(torch.linspace(.5, e, half_1, dtype=torch.float16))).tolist()[1:]
    v2 = (norm.ppf(torch.linspace(.5, e, half_2, dtype=torch.float16))).tolist()

    g = v1 + v2
    g.sort()
    nf4 = torch.tensor(g)
    nf4 /= nf4.amax()

    return nf4.to(torch.bfloat16)

PRECOMPUTED_NORMAL_FP4 = compute_normal_fpk(bits=4)