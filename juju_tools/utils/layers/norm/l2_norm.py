from juju_tools.utils.consts import *

# L2 Normalization
class L2Norm(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, dim = -1) -> Tensor:
        dtype = x.dtype
        x = x.float()
        return (x / x.norm(p=2, dim=dim, keepdim=True)).to(dtype)

