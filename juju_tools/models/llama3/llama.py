from juju_tools.models.modeling_utils import BaseModel
from juju_tools.utils import LLaMaConfig
from juju_tools.utils.consts import *

from juju_tools.utils.layers import LLaMaGQA, FeedForward, RMSNorm
from juju_tools.utils.layers.mha.attn_utils import DynKVCache
from juju_tools.utils.math import causal_mask


class LLaMaBlockDecoder(Module):

    def __init__(self,
                 config: LLaMaConfig,
                 layer_idx: Optional[int]) -> None:
        super().__init__()

        self.config = config

        self.gq_attn = LLaMaGQA(config, layer_idx)

        self.ffn = FeedForward(config, layer_idx)

        self.attn_prenorm = RMSNorm(config, layer_idx)
        self.ffn_prenorm = RMSNorm(config, layer_idx)

        self.layer_idx = layer_idx

    def forward(self,
                x: Tensor,
                mask: Optional[Tensor] = None,
                kv_cache: Optional[DynKVCache] = None,
                shift: int = 0) -> Tensor:
        h = x + self.gq_attn(self.attn_prenorm(x), mask, kv_cache, shift)
        o = h + self.ffn(self.ffn_prenorm(h))
        return o


class LLaMa(BaseModel):

    def __init__(self,
                 config: LLaMaConfig,
                 use_cache: bool = True,
                 load_all: bool = True, **kwargs) -> None:
        super().__init__(config, load_all, **kwargs)

        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.dim,
                                  padding_idx=config.pad_tok_idx, device="meta")  # .cpu()

        self.kv_cache = DynKVCache(config) if use_cache else None
        self.pre_norm = RMSNorm(config)

        self._add_block()

        if self.load_all:
            self.load()

    def clear_cache(self) -> None:
        if self.kv_cache is None:
            return
        self.kv_cache.clear()

    def load(self, device: Optional[Tensor] = None) -> None:

        conf_device = self.config.device
        device = device if device != None else conf_device

        self.config.device = device

        for i in range(self.config.n_layers):
            try:
                self.layers.append(LLaMaBlockDecoder(self.config, i))
            except:
                logger.info("Going to fallback - Out of Memory")
                self.config.device = "meta"
                self.layers.append(LLaMaBlockDecoder(self.config, i))
        super().load()

        self.config.device = conf_device

    def forward(self,
                input: Tensor,
                attn_mask: Optional[Tensor] = None,
                shift: int = 0) -> Tensor:
        assert self.loaded, "Cannot pass forward - model has not been loaded!"

        shift = self.kv_cache.past_toks if self.kv_cache is not None else shift

        h = self.embed(input)

        if input.shape[-1] > 1:
            attn_mask = self._gen_mask(input, shift).to(h.device) if attn_mask == None else attn_mask

        for layer in self.layers:
            h = layer(h, attn_mask, self.kv_cache, shift)

        return self.pre_norm(h)

    def _gen_mask(self,
                  input: Tensor,
                  shift: int = 0) -> Tensor:
        curr_seq = input.shape[-1]

        target = curr_seq + shift
        shift = shift + 1

        return causal_mask(curr_seq,
                           target,
                           shift=shift)[None, None, ...]


class LLaMaAutoRegressive(LLaMa):

    def __init__(self,
                 config: LLaMaConfig,
                 use_cache: bool = True,
                 load_all: bool = True,
                 **kwargs) -> None:
        super().__init__(config, use_cache, load_all, **kwargs)
        self.cls_head = nn.Linear(self.config.dim, self.config.vocab_size,
                                  bias=False, device="meta")

        if self.loaded():
            self._init_weights()

    def clear_cache(self) -> None:
        if self.kv_cache is None: return

        super().clear_cache()

    def forward(self,
                input: Tensor,
                attn_mask: Optional[Tensor] = None,
                shift: int = 0) -> Tensor:
        h = super().forward(input, attn_mask, shift)  # .to(torch.get_default_dtype())  # .cpu()
        return self.cls_head(h.cpu()).cuda().float()
