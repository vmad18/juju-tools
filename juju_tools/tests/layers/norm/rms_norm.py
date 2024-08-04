from utils.layers.norm.rms_norm import RMSNorm
from utils.consts import *


class rms_norm_test(Module):

    def __init__(self, dim: int):
        super().__init__()

        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


if __name__ == '__main__':
    x = torch.randn((3, 20))
    rn = rms_norm_test(dim=20)
    print(rn(x))
