from juju_tools.utils import *
from juju_tools.utils.layers.norm.l2_norm import L2Norm

class NormFeedForward(Module):

    def __init__(self,
                 config: nGPTConfig,
                 layer_idx: Optional[int] = None):
        super().__init__()

        self.norm = L2Norm()

        self.proj_up = nn.Linear(config.dim, config.hidden_state, bias=config.bias, device=config.device, dtype=config.dtype)
        self.gate = nn.Linear(config.dim, config.hidden_state, bias=config.bias, device=config.device, dtype=config.dtype) \
            if config.gate else nn.Identity()

        self.proj_down = nn.Linear(config.hidden_state, config.dim, bias=config.bias, device=config.device, dtype=config.dtype)

        self.s_u = nn.Parameter(torch.ones(config.hidden_state, dtype=config.dtype, device=config.device)) * config.s_u_scale
        self.s_v = nn.Parameter(torch.ones(config.hidden_state, dtype=config.dtype, device=config.device)) * config.s_v_scale
        self.alpha = nn.Parameter(torch.ones(config.dim, dtype=config.dtype, device=config.device)) * config.alpha_m_scale

        self.config = config
        self.layer_idx = layer_idx

    def forward(self, h: Tensor) -> Tensor:
        u, v = self.proj_up(h), self.gate(h)

        s_u = self.s_u * self.config.s_u_init / self.config.s_u_scale
        s_v = self.s_v * self.config.s_v_init / self.config.s_v_scale

        u = u * s_u
        v = v * s_v * self.config.v_scale

        alpha_m = torch.abs(self.alpha * self.config.alpha_m_init / self.config.alpha_m_scale)

        h_m = self.norm(self.proj_down(self.norm(self.config.nl(v) * u)))
        h = self.norm(h + alpha_m * (h_m - h)) # SLERP
        return h