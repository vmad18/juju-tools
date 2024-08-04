from juju_tools.utils.consts import *

from scipy.stats import norm
from enum import Enum

# from juju_tools.utils.layers import *

"""

Estimates the Hessian for Pre-conditioning

"""


class Estimator:

    def __init__(self) -> None: pass

    def compute(self, params: List[Tensor], loss: Tensor, logits: Optional[Tensor] = null) -> List[Tensor]:
        pass


class ModuleConfig(Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.bsz = DEF_MAX_BSZ
        self.max_tokens = DEF_MAX_TOK
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.B_q: int = 1024
        self.B_kv: int = 2048
        self.__dict__.update(kwargs)


class Config(object):

    def __init__(self, **kwargs):
        self.bsz = DEF_MAX_BSZ
        self.max_tokens = DEF_MAX_TOK
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.B_q: int = 1024
        self.B_kv: int = 2048

        self.dim = 512

        self.attn_heads = 32
        self.head_dim = self.dim // self.attn_heads

        self.n_layers = 32
        self.eps = 1e-5
        self.rope_base = 5e5

        self.bias = False
        self.drop_r = 0.1

        self.scale: float = 2
        self.nl: Callable[[Tensor], Tensor] = F.silu
        self.gate: bool = True

        self.quant_modules: Tuple[str, ...] = ()
        self.quant_layers: Tuple[str, ...] = ()

        self.pad_tok_idx = -1

        self.__dict__.update(kwargs)


class LLaMaConfig(Config):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dim = 4096

        self.max_tokens = 8192

        self.attn_heads = 32
        self.kv_heads = 8  # Grouped Query Attention Heads

        self.head_dim = self.dim // self.attn_heads

        self.n_groups = self.attn_heads // self.kv_heads

        self.n_layers = 32
        self.eps = 1e-5
        self.rope_base = 500000.0

        self.bias = False
        self.attn_bias = False

        self.drop_r = 0.1

        self.vocab_size = 128256
        self.pad_tok_idx = -1
        

        self.scale: float = 7 / 2
        self.nl: Callable[[Tensor], Tensor] = F.silu
        self.gate: bool = True

        self.quant_modules: Tuple[str, ...] = ("layers", "embed", "cls_head", "pre_norm")
        self.quant_layers: Tuple[str, ...] = ("gq_attn", "ffn", "attn_prenorm", "ffn_prenorm")

        self.__dict__.update(kwargs)


class Phi3MiniConfig(Config):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LModelConfig(Config):

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

        self.dim = 256
        self.heads = 8
        self.vocab_size = 500000
        self.n_layers = 2
        self.scale = 4
        self.rope = True
        self.kv_cache = True
        self.drop_r = .1
        self.nl = F.silu
        self.base = 5e5
        self.__dict__.update(kwargs)


def linear_weight(
        in_features,
        out_features,
        requires_grad=True) -> Tensor:
    return torch.nn.init.xavier_uniform_(torch.zeros((out_features, in_features), requires_grad=requires_grad))


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


# class QuantType(Enum):
#     NFK = 1
#     INT8 = 2


# class Quantizer(object):
#
#     def __init__(self, data: Tensor) -> None:
#         self.data = data
#
#     def quant(self) -> Tuple[Tensor, ...]:
#         pass
#
#     def dequant(self) -> Tuple[Tensor, ...]:
#         pass


PRECOMPUTED_NORMAL_FP4 = compute_normal_fpk(bits=4)
