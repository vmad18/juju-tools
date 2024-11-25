import torch

from juju_tools.utils.consts import *

class Config(object):

    def __init__(self, **kwargs):
        self.bsz = DEF_MAX_BSZ
        self.max_tokens = DEF_MAX_TOK
        self.vocab_size = 100000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dim = 512

        self.attn_heads = 32
        self.head_dim = self.dim // self.attn_heads

        self.n_layers = 32
        self.eps = 1e-5
        self.rope = True
        self.rope_base = 5e5
        self.sm_scale = 1 / sqrt(self.head_dim)

        self.bias = False
        self.drop_r = 0.1

        self.expansion_scale: float = 2  # expansion factor
        self.hidden_state: int = int(self.expansion_scale * self.dim)
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


class nGPTConfig(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rope = True
        self.sm_scale = sqrt(self.head_dim)

        self.__dict__.update(kwargs)
        self.norm_dim = -1

        # eigen learning rate and scaling factors
        self.alpha_a_scale = 1. / sqrt(self.dim)
        self.alpha_a_init =  1. / self.n_layers

        self.alpha_m_scale = 1. / sqrt(self.dim)
        self.alpha_m_init = 1. / self.n_layers

        self.s_qk_scale = 1. / (self.dim ** 0.5)
        self.s_qk_init = 1.

        self.s_u_scale = 1.
        self.s_u_init = 1.

        self.s_v_scale = 1.
        self.s_v_init = 1.
        self.v_scale = sqrt(self.dim)

        self.s_z_scale = 1. / (self.dim ** 0.5)
        self.s_z_init = 1.

        self.dtype = torch.bfloat16


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


class ModuleConfig(Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.bsz = DEF_MAX_BSZ
        self.max_tokens = DEF_MAX_TOK
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.__dict__.update(kwargs)

def linear_weight(
        in_features,
        out_features,
        requires_grad=True) -> Tensor:
    return torch.nn.init.xavier_uniform_(torch.zeros((out_features, in_features), requires_grad=requires_grad))
