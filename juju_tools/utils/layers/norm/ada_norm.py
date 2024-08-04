from juju_tools.utils import *


class AdaNorm(Module):

    def __init__(self, C: int, k: float = 1. / 10):
        super().__init__()

        self.C = C
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        z_val = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True))
        return (self.C * (1 - self.k * z_val)).dot(z_val)
