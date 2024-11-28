from juju_tools.utils import *

from juju_tools.utils.layers import RoPE, FlashAttention
from juju_tools.utils.math import scaled_dot_product


class CrossAttention(Module):

    r"""
    
    Cross Attention 

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
                 **kwargs,):
        super().__init__(**kwargs)

        self.h = heads

        self.to_qk = nn.Linear(dim, 2*dim, bias=bias, device=self.device)
        self.to_v = nn.Linear(dim, dim, bias=bias, device=self.device)
        self.o_proj = nn.Linear(dim, dim, device=self.device)

        self.rope = RoPE(dim // heads, base, **kwargs) if rope else RoPE.pass_qk
        self.flash = flash
        self.kv_cache = kv_cache


        if kv_cache:
            self.k_cache = torch.zeros((self.bsz, heads, self.max_tokens, dim//heads), device=self.device)
            self.v_cache = torch.zeros((self.bsz, heads, self.max_tokens, dim//heads), device=self.device)


        self.drop_r = drop_r

    def forward(self,
                ctx: Tensor, 
                x: Tensor, 
                mask: Optional[Tensor] = None, 
                shift: int = 1,)-> Tensor:
        b, s, *_ = x.shape

        if ctx is None:
            qkv = (*self.to_qk(x).chunk(2, -1), self.to_v(x))
        else:
            qkv = (*self.to_qk(ctx).chunk(2, -1), self.to_v(x))

        q, k, v = map(
            lambda w: rearrange(w, "b s (h d) -> b h s d", h=self.h),
            qkv
        )

        q, k = self.rope(q, k, shift=shift)
        
        if self.kv_cache:
            self.k_cache[:b, :, shift:s+shift, :] = k 
            self.v_cache[:b, :, shift:s+shift, :] = v 

            k = self.k_cache[:b, :, :s+shift]
            v = self.v_cache[:b, :, :s+shift]
        
        if self.flash:
            assert self.B_q != 0 and self.B_kv != 0

            fa = FlashAttention.apply
            x = fa(q, k, v, self.B_q, self.B_kv, mask)
        else:
            x = scaled_dot_product(q, k, v, mask=mask, drop_r=self.drop_r)

        x = rearrange(x, "b h s d -> b s (h d)", h=self.h)

        return F.dropout(x, p=self.drop_r, training=True)
