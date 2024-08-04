from juju_tools.utils import *


class RMSNorm(Module):

    def __init__(self,
                 config: Config,
                 layer_idx: Optional[int] = None):
        super().__init__()

        self.config = config

        self.gate = nn.Parameter(torch.ones(self.config.dim, dtype=torch.float32, device=config.device),
                                 requires_grad=True)
        self.layer_idx = layer_idx

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        rrms = (x.to(torch.float32).pow(2).mean(-1, keepdim=True) + self.config.eps).rsqrt()
        return (x * self.gate * rrms).to(dtype)
