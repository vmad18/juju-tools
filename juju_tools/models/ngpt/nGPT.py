from juju_tools.utils import nGPTConfig
from juju_tools.utils.consts import *
from juju_tools.utils.layers import NormAttention, NormFeedForward, L2Norm


class nGPTBlock(Module):

    def __init__(self, config: nGPTConfig, layer_idx: Optional[int]):
        super().__init__()

        self.norm_attn = NormAttention(config, layer_idx)
        self.norm_ffn = NormFeedForward(config, layer_idx)

        self.config = config
        self.layer_idx = layer_idx

    def forward(self, h: Tensor) -> Tensor:
        h = self.norm_attn(h)
        h = self.norm_ffn(h)
        return h


class nGPT(Module):

    def __init__(self, config: nGPTConfig):
        super().__init__()

        self.layers = nn.ModuleList([nGPTBlock(config, i) for i in range(config.n_layers)])

    def forward(self, h: Tensor):
        # pass through residual stream
        for layer in self.layers:
            h = layer(h)
        return h


class nGPTAutoRegressive(Module):

    def __init__(self, config: nGPTConfig):
        super().__init__()

        self.embed = nn.Embedding(config.vocab_size, config.dim, device=config.device, dtype=config.dtype)
        self.stream = nGPT(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False, device=config.device, dtype=config.dtype)
        self.s_z = nn.Parameter(torch.ones(config.vocab_size, dtype=config.dtype, device=config.device)) * config.s_z_scale

        self.config = config
        self.apply(self._init_weights)

        self.norm = L2Norm()

    def forward(self, h: Tensor, sm: bool = True):
        h = self.embed(h)
        h = self.stream(h)
        z = self.lm_head(h)
        s_z = self.s_z * self.config.s_z_init / self.config.s_z_scale
        z = z * s_z
        return F.softmax(z, dim=-1) if sm else z

    def _norm_mats(self):
        self.embed.weight.copy_(self.norm(self.embed.weight, -1))
        for layer in self.stream.layers:
            if isinstance(layer, nGPTBlock):
                # normalizes over embedding dimension
                layer.norm_attn.to_q.weight.copy_(self.norm(layer.norm_attn.to_q.weight, dim=0))
                layer.norm_attn.to_k.weight.copy_(self.norm(layer.norm_attn.to_k.weight, dim=0))
                layer.norm_attn.to_v.weight.copy_(self.norm(layer.norm_attn.to_v.weight, dim=0))
                layer.norm_attn.o_proj.weight.copy_(self.norm(layer.norm_attn.o_proj.weight, dim=-1))

                layer.norm_ffn.proj_up.weight.copy_(self.norm(layer.norm_ffn.proj_up.weight, dim=0))
                layer.norm_ffn.gate.weight.copy_(self.norm(layer.norm_ffn.gtate.weight, dim=0))
                layer.norm_ffn.proj_down.weight.copy_(self.norm(layer.norm_ffn.proj_up.weight, dim=-1))
        self.lm_head.weight.copy_(self.norm(self.lm_head.weight, 0))

    def _init_weights(self, module: Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1. / sqrt(self.config.dim))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1. / sqrt(self.config.dim))
