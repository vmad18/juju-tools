from juju_tools.utils import *

from juju_tools.utils.layers.quant import BitLinear
from juju_tools.utils.math import scaled_dot_product, FlashAttention

# Remove 158?
class MHQAttention158(Module):

    '''

    :param dim - Input model dimension
    :param heads - Number of attention heads
    :param flash - Flag that enables flash attention (default: False)
    :param drop_r - Drop out rate

    '''

    def __init__(self,
                 dim: int,
                 heads: int,
                 flash: bool = False,
                 drop_r: float = 0.):
        super().__init__()

        self.h = heads

        self.to_qkv = BitLinear(dim, 3 * dim)
        self.o_proj = BitLinear(dim, dim)

        self.flash = flash
        self.drop_r = drop_r

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, B_q: int = 0, B_kv: int = 0) -> Tensor:
        qkv = self.to_qkv(x).chunk(3, -1)

        q, k, v = map(
            lambda w: rearrange(w, "b s (h d) -> b h s d", h=self.h),
            qkv
        )

        if self.flash:
            assert B_q != 0 and B_kv != 0

            fa = FlashAttention.apply
            x = fa(q, k, v, B_q, B_kv, mask)
        else:
            x = scaled_dot_product(q, k, v, mask=mask, drop_r=self.drop_r)

        x = rearrange(x, "b h s d -> b s (h d)", h=self.h)

        return F.dropout(x, p=self.drop_r, training=True)
