from juju_tools.utils import *
from juju_tools.utils.configs import Config

class DynKVCache:

    def __init__(self,
                 config: Config,
                 auto_crop: bool = True) -> None:

        self.key_caches: List[Tensor] = []
        self.value_caches: List[Tensor] = []

        self.auto_crop = auto_crop
        self.config = config

        self.past_toks: int = 0

    def update(self,
               key_cache: Tensor,
               value_cache: Tensor,
               layer_idx: int) -> Tuple[Tensor, Tensor]:

        if len(self.key_caches) <= layer_idx:
            self.key_caches.append(key_cache)
            self.value_caches.append(value_cache)
        else:
            self.key_caches[layer_idx] = torch.cat([self.key_caches[layer_idx], key_cache], dim=-2)
            self.value_caches[layer_idx] = torch.cat([self.value_caches[layer_idx], value_cache], dim=-2)

        if self.auto_crop:
            self.crop(layer_idx)

        if layer_idx == 0:
            self.past_toks = min(self.past_toks + key_cache.shape[-2], self.config.max_tokens)

        return self.key_caches[layer_idx], self.value_caches[layer_idx]

    def crop(self,
             layer_idx: int) -> None:

        if self.key_caches[layer_idx].shape[-2] <= self.config.max_tokens:
            return

        diff = self.key_caches[layer_idx].shape[-2] - self.config.max_tokens

        self.key_caches[layer_idx] = self.key_caches[layer_idx][..., diff:, :]
        self.value_caches[layer_idx] = self.value_caches[layer_idx][..., diff:, :]

    def clear(self) -> None:
        self.key_caches = []
        self.value_caches = []
        self.past_toks = 0
