from juju_tools.utils import *


class FeedForward(Module):

    def __init__(self, 
                 config: Config,
                 layer_idx: Optional[int] = None):
        super().__init__()
        
        self.config = config

        hidden = int(self.config.dim * self.config.scale)

        self.proj_up = nn.Linear(self.config.dim, hidden, bias=self.config.bias, device=self.config.device)
        self.gate = nn.Linear(self.config.dim, hidden, bias=self.config.bias, device=self.config.device) \
            if self.config.gate else nn.Identity()
        self.proj_down = nn.Linear(hidden, self.config.dim, bias=self.config.bias, device=self.config.device)
        
        self.layer_idx = layer_idx

    def forward(self, x: Tensor) -> Tensor:
        o, gate = self.proj_up(x), self.gate(x)
        return self.proj_down(self.config.nl(o) * gate)
