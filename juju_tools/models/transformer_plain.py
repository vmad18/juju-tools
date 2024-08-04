from juju_tools.utils import *

from juju_tools.utils.layers import (
                                    MultiHeadAttention, 
                                    CrossAttention, 
                                    CausalAttention, 
                                    RMSNorm, 
                                    FeedForward, 
                                    PositionalEncoding)


class Encoder(LModelConfig):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        
        self.mha = MultiHeadAttention(heads=self.heads, drop_r=self.drop_r, rope=self.rope, **kwargs)
        self.ffn = FeedForward(scale=self.scale, nl=self.nl, **kwargs)

        self.norm1 = RMSNorm(**kwargs)
        self.norm2 = RMSNorm(**kwargs) 
            
    def forward(self, 
                x: Tensor, 
                mask: Optional[Tensor]=None) -> Tensor:
        x = x + self.norm1(self.mha(x, mask=mask))
        x = x + self.norm2(self.ffn(x))
        return x 

class Decoder(LModelConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.mmha = CausalAttention(heads=self.heads, drop_r=self.drop_r, kv_cache=self.kv_cache, rope=self.rope, **kwargs)
        self.cmha = CrossAttention(heads=self.heads, drop_r=self.drop_r, **kwargs)
        self.ffn = FeedForward(scale=self.scale, nl=self.nl, **kwargs)
        
        self.norm1 = RMSNorm(**kwargs)
        self.norm2 = RMSNorm(**kwargs)
        self.norm3 = RMSNorm(**kwargs)

    def forward(self, 
                x: Tensor, 
                ctx: Tensor, 
                mask: Optional[Tensor]=None,
                shift: int = 0,) -> Tensor:
        x = x + self.norm1(self.mmha(x, shift=shift, mask=mask))
        x = x + self.norm2(self.cmha(x, ctx))
        x = x + self.norm3(self.ffn(x))
        return x

class TransformerBlock(LModelConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.enc = Encoder(**kwargs)
        self.dec = Decoder(**kwargs)

    def forward(self, 
                inp: Tensor, 
                out: Tensor,
                shift: int) -> Tuple[Tensor, Tensor]:
        ctx = self.enc(inp)
        y = self.dec(out, ctx, shift)
        return ctx, y

class Transformer(LModelConfig):

    def __init__(self, head: bool = True, prob: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim, device=self.device)
        self.pe = PositionalEncoding(drop_r=self.drop_r, **kwargs)
        self.layers = nn.ModuleList([TransformerBlock(**kwargs) for _ in range(self.n_layers)])
        
        self.cls_head = nn.Identity()

        if head:
            self.cls_head = nn.Linear(self.dim, self.vocab_size, device=self.device)

        self.prob = prob

    def forward(self, 
                inp: Tensor, 
                out: Tensor, 
                shift: int = 1, 
                temp: float = 1.) -> Tensor:
        inp = self.pe(self.embed(inp))
        out = self.pe(self.embed(out))
        

 
        for layer in self.layers: 
            inp, out = layer(inp, out, shift)
  
        out = self.cls_head(out) 
        if self.prob:
            out = F.softmax(out / temp, dim=-1)
        
        return out

