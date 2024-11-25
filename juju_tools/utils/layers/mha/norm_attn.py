from juju_tools.utils import *

from juju_tools.utils.layers import RoPE, L2Norm
from juju_tools.utils.math import scaled_dot_product

class NormAttention(Module):

    def __init__(self, config: nGPTConfig,
                 layer_idx: Optional[int]):
        super().__init__()

        self.to_q = nn.Linear(config.dim, config.head_dim * config.attn_heads,
                                bias=config.bias, dtype=config.dtype, device=config.device)
        self.to_k = nn.Linear(config.dim, config.head_dim * config.attn_heads,
                                bias=config.bias, dtype=config.dtype, device=config.device)
        self.to_v = nn.Linear(config.dim, config.head_dim * config.attn_heads,
                                bias=config.bias, dtype=config.dtype, device=config.device)
        self.o_proj = nn.Linear(config.head_dim * config.attn_heads, config.dim,
                                bias=config.bias, dtype=config.dtype, device=config.device)

        self.rope = RoPE(config) if config.rope else RoPE.pass_qk

        self.norm = L2Norm()
        self.s_qk = nn.Parameter(torch.ones(config.head_dim * config.attn_heads, dtype=config.dtype, device=config.device) * config.s_qk_scale)
        self.alpha = nn.Parameter(torch.ones(config.dim, dtype=config.dtype, device=config.device) * config.alpha_a_scale)

        self.config = config
        self.layer_idx = layer_idx

    def forward(self, h: Tensor):
        h_qkv = (self.to_q(h), self.to_k(h), self.to_v(h))

        q, k, v = map(
            lambda x: rearrange(x, "b s (h d) -> b h s d", h=self.config.attn_heads),
            h_qkv
        )

        s_qk = (self.s_qk * self.config.s_qk_init / self.config.s_qk_scale).view(1, self.config.attn_heads, 1, self.config.head_dim)

        q, k = self.rope(q, k)

        q = self.norm(q) * s_qk
        k = self.norm(k) * s_qk

        attn = scaled_dot_product(q, k, v, scaling_norm=self.config.sm_scale, training=True).to(q.dtype) # perform sdp
        h_a = self.norm(self.o_proj(rearrange(attn, "b h s d -> b s (h d)", h = self.config.attn_heads)))  # concat and project out

        alpha_a = torch.abs(self.alpha * (self.config.alpha_a_init / self.config.alpha_a_scale))

        h = self.norm(h + alpha_a * (h_a - h))  # take step over hypersphere and inject into residual stream - SLERP
        return h
