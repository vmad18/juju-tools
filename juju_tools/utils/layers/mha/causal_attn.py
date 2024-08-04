from juju_tools.utils import *

from juju_tools.utils.layers import RoPE

from juju_tools.utils.layers.transforms.embeds import DynNTKRoPE
from juju_tools.utils.math import scaled_dot_product, FlashAttention, causal_mask
from juju_tools.utils.layers.mha.attn_utils import DynKVCache
from juju_tools.kernels.py.fused_attn import flash_attn

# TODO make sure the masking stuff is done correctly
# TODO remove forward parameters for flash attn
class CausalAttention(ModuleConfig):
    r"""
    
    Causal attention 

    Args:
        dim(int): - Input model dimension
        heads(int): - Numbers of attention heads
        rope(bool): - Enables rotary position embedding
        flash(bool): - Enables flash attention (default = False)
        drop_r(float): - Drop out rate (default = 0.)
        bias(bool): - Adds projection bias (default = False)

    """

    def __init__(self,
                 dim: int,
                 heads: int,
                 rope: bool = True,
                 flash: bool = False,
                 kv_cache: bool = True,
                 drop_r: float = 0.,
                 bias: bool = False,
                 base: float = 1e4,
                 **kwargs, ):
        super().__init__(**kwargs)

        self.h = heads

        self.to_q = nn.Linear(dim, dim, bias=bias, device=self.device)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=bias, device=self.device)
        self.o_proj = nn.Linear(dim, dim, device=self.device)

        self.rope = RoPE(dim // heads, base=base, **kwargs) if rope else RoPE.pass_qk
        self.flash = flash
        self.kv_cache = kv_cache

        if kv_cache:
            self.k_cache = torch.zeros((self.bsz, heads, self.max_tokens, dim // heads), device=self.device)
            self.v_cache = torch.zeros((self.bsz, heads, self.max_tokens, dim // heads), device=self.device)

        self.drop_r = drop_r

    def forward(self,
                x: Tensor,
                ctx: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                shift: int = 0, ) -> Tensor:
        b, s, *_ = x.shape

        if ctx is None:
            qkv = (self.to_q(x), *self.to_kv(x).chunk(2, -1))
        else:
            qkv = (self.to_q(x), *self.to_kv(ctx).chunk(2, -1))

        q, k, v = map(
            lambda w: rearrange(w, "b s (h d) -> b h s d", h=self.h),
            qkv
        )

        q, k = self.rope(q, k, shift=shift)

        if self.kv_cache:
            self.k_cache[:b, :, shift:s + shift, :] = k
            self.v_cache[:b, :, shift:s + shift, :] = v

            k = self.k_cache[:b, :, :s + shift]
            v = self.v_cache[:b, :, :s + shift]

        if mask is None:
            mask = causal_mask(q.shape[-2], k.shape[-2], shift=shift).to(self.device)

        if self.flash:
            assert self.B_q != 0 and self.B_kv != 0

            fa = FlashAttention.apply
            x = fa(q, k, v, self.B_q, self.B_kv, mask)
        else:
            x = scaled_dot_product(q, k, v, mask=mask, drop_r=self.drop_r)

        x = rearrange(x, "b h s d -> b s (h d)", h=self.h)

        return F.dropout(x, p=self.drop_r, training=True)


class LLaMaGQA(Module):

    def __init__(self,
                 config: LLaMaConfig,
                 layer_idx: Optional[int] = None) -> None:
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.rope = DynNTKRoPE(self.config)

        self.proj_q = nn.Linear(config.dim, config.attn_heads * config.head_dim,
                                bias=config.attn_bias, device=self.config.device)
        self.proj_k = nn.Linear(config.dim, config.kv_heads * config.head_dim,
                                bias=config.attn_bias, device=self.config.device)
        self.proj_v = nn.Linear(config.dim, config.kv_heads * config.head_dim,
                                bias=config.attn_bias, device=self.config.device)

        self.proj_o = nn.Linear(config.attn_heads * config.head_dim, config.dim,
                                bias=config.attn_bias, device=self.config.device)

    def _repeat_heads(self,
                      states: Tensor) -> Tensor:
        b, h, s, d = states.shape

        assert h == self.config.kv_heads, "Head dimension does not equal KV head dimension!"

        if self.config.n_groups <= 1:
            return states

        states = rearrange(states, "b h s d -> b h 1 s d").expand(b, h, self.config.n_groups, s, d)
        states = states.reshape(b, h * self.config.n_groups, s, d)
        return states

    def forward(self,
                x: Tensor,
                attn_mask: Optional[Tensor] = None,
                kv_cache: Optional[DynKVCache] = None,
                shift: int = 0) -> Tensor:
        b, s, d = x.shape

        q_states, k_states, v_states = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        q_states = rearrange(q_states, "b s (h d) -> b h s d", h=self.config.attn_heads)
        k_states = rearrange(k_states, "b s (h d) -> b h s d", h=self.config.kv_heads)
        v_states = rearrange(v_states, "b s (h d) -> b h s d", h=self.config.kv_heads)

        # RoPE stuff
        if kv_cache is not None:
            q_states, k_states = self.rope(q_states, k_states, shift)
            k_states, v_states = kv_cache.update(k_states, v_states, self.layer_idx)
        else:
            q_states, k_states = self.rope(q_states, k_states, shift)

        k_states = self._repeat_heads(k_states)
        v_states = self._repeat_heads(v_states)

        A = scaled_dot_product(q_states.to(torch.float32), k_states.to(torch.float32), v_states.to(torch.float32), mask=attn_mask).transpose(1, 2).reshape(b, s, -1).to(torch.bfloat16)

        # A = flash_attn(q_states.to(torch.float32), k_states.to(torch.float32), v_states.to(torch.float32), training=self.training, causal=True).transpose(1, 2).reshape(b, s, -1).to(torch.bfloat16)

        return self.proj_o(A)
