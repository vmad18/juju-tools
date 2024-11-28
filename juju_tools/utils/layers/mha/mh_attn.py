from juju_tools.utils import *

from juju_tools.utils.layers import RoPE
from juju_tools.utils.math import scaled_dot_product, FlashAttention

class MultiHeadAttention(Module):

    r"""
    
    Multi-Head Attention, proposed in AIAYN, is a method to attend 
    tokens with each other across heads creating an attention mask. 
    Mask is applied across input. Formulated as:
    output = softmax(q * k_T / sqrt(k_dim)) * v 

    Args:
        dim (int): - The model dimensions
        heads (int): - The number of heads to peform attention across
        flash (bool): - Sets if flash attention should be used (default False)
        rope (bool): - Sets if rotary encodings (RoPE) should be used (default True)
        drop_r (float): - The dropout rate value
        bias (bool): - Sets if bias should be used in q, k, v projections (default = False)
        **kwargs (dict) - Other variables: max_tokens

    """
    
    def __init__(self,
                 dim: int,
                 heads: int,
                 flash: bool = False,
                 rope: bool = True,
                 drop_r: float = 0.,
                 bias: bool = False,
                 base: float = 1e4,
                 kv_cache: bool = True,
                 **kwargs,):
        super().__init__(**kwargs)
 
        self.__dict__.update(kwargs)

        self.h = heads
        
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias, device=self.device)
        self.o_proj = nn.Linear(dim, dim, device=self.device)
        
        self.rope = RoPE(dim // heads, base=base, **kwargs) if rope else RoPE.pass_qk

        self.flash = flash
        self.drop_r = drop_r

    def forward(self, 
                x: Tensor, 
                shift: int = 0,
                mask: Optional[Tensor] = None,
                ) -> Tensor:
        qkv = self.to_qkv(x).chunk(3, -1)

        q, k, v = map(
            lambda w: rearrange(w, "b s (h d) -> b h s d", h=self.h),
            qkv
        )

        q, k = self.rope(q, k, shift=shift)

        if self.flash:
            assert self.B_q != 0 and self.B_kv != 0

            fa = FlashAttention.apply
            x = fa(q, k, v, self.B_q, self.B_kv, mask)
        else:
            x = scaled_dot_product(q, k, v, mask=mask, drop_r=self.drop_r)

        x = rearrange(x, "b h s d -> b s (h d)", h=self.h)

        return F.dropout(x, p=self.drop_r, training=True)


if __name__ == "__main__":
    mha = MultiHeadAttention(20, 5)
    x = torch.randn((3, 10, 20))
    print(mha(x).shape)
