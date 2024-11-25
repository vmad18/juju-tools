from juju_tools.utils import *


class PositionalEncoding(Module):

    def __init__(self,
                 config: Config,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.config = config

        self.pos = torch.arange(0, self.config.max_tokens, device=self.config.device)[..., None]

        self.freqs = self.pos * \
                     torch.exp(-log(1e4) * torch.arange(0, self.config.dim, 2, device=self.device) / self.config.dim)[
                         None, ...]

        self.pos = torch.zeros((self.max_tokens, self.config.dim), device=self.device)
        self.pos[..., ::2] = self.freqs.sin()
        self.pos[..., 1::2] = self.freqs.cos()
        self.pos = self.pos[None, ...]

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x + self.pos[0, :x.shape[-2], :], training=self.training)


class PositionalEmbedding(ModuleConfig):

    def __init__(self,
                 dim: int,
                 drop_r: float = 0.,
                 **kwargs, ) -> None:
        super().__init__(**kwargs)

        self.embeds = nn.Parameter(torch.randn((1, self.max_tokens, dim), device=self.device), requires_grad=True)
        self.drop = nn.Dropout(p=drop_r)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(x + self.embeds[0, :x.shape[-2], :])


class RoPE(Module):

    def __init__(self,
                 config: Config,
                 scaling: float = 1.) -> None:
        super().__init__()

        self.config = config

        self.dim = self.config.head_dim
        self.base = self.config.rope_base
        self.scaling = scaling

    def comp_rots(self) -> Tensor:
        theta = torch.exp(
            -log(self.base) * torch.arange(0, self.dim, 2, device=self.config.device, dtype=torch.int64) / self.dim)[
            None, ...]

        m = torch.arange(self.config.max_tokens, device=self.config.device, dtype=torch.float32)[..., None].float()
        freqs = m * theta
        mag = torch.ones_like(freqs, device=self.config.device)

        return torch.polar(mag, freqs)

    @staticmethod
    def pass_qk(q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        return q, k

    def _to_complex(self, x: Tensor) -> Tensor:
        return torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))

    def _to_real(self, x: Tensor) -> Tensor:
        return torch.view_as_real(x)

    @torch.no_grad()
    def forward(self,
                q: Tensor,
                k: Tensor,
                shift: int = 0) -> Tuple[Tensor, Tensor]:
        *_, s, d = q.shape

        dtype = q.dtype

        q, k = q.float(), k.float()

        rots = self.comp_rots()[shift : shift + s].reshape(1, 1, s, d // 2)  # self.rotations[shift:shift+s]

        _q = self._to_complex(q) * rots
        _k = self._to_complex(k) * rots

        rq = self._to_real(_q).to(dtype)
        rk = self._to_real(_k).to(dtype)

        return rq.reshape(*rq.shape[:-2], d).to(dtype), rk.reshape(*rk.shape[:-2], d).to(dtype)


class DynNTKRoPE(RoPE):

    def comp_rots(self) -> Tensor:
        return super().comp_rots()

    def forward(self,
                q: Tensor,
                k: Tensor,
                shift: int = 0) -> Tuple[Tensor, Tensor]:
        seq = shift + q.shape[-2] + 1

        if seq > self.config.max_tokens:
            self.base = (self.base * self.scaling * seq / self.config.max_tokens) ** (self.dim / (self.dim - 2))

        return super().forward(q, k, shift)
