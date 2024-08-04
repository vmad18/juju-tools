from juju_tools.utils import *

from juju_tools.utils.layers import RoPE
from juju_tools.utils.math import scaled_dot_product, FlashAttention

class InfiniAttention(ModuleConfig):

    r"""
    
    InfiniAttention module that enables attention across a much longer context window. Previous segments are saved 
    in a memory matrix, M. 

    Args:
        dim (int): - Hidden state input dimension
        heads (int): - Number of attention heads
        dim_heads (int | None): - Number of dimensions per head, if None then dim_heads == dim (default = None)
        rope (bool): - Set to true to use rotary embeddings (default = True)
        flash (bool): - Set to true to use flash attention (default = False)
        drop_r (float): - Dropout rate of attention (default = 0.)
        bias (bool): - Set to use bias for q, k, v projections (default = False)
        delta (bool): - Set to use delta rule for memory updates (default = True)
        layer_idx (int | None): - Layer index of the module (default = None)
        save_mem_norm (bool): - Set to save previous memory and normalization states internally (default = False)

    """

    def __init__(self, 
                 dim: int, 
                 heads: int, 
                 dim_heads: Optional[int] = None,
                 rope: bool = True, 
                 flash: bool = False, 
                 drop_r: float = 0.,
                 bias: bool = False,
                 base: float = 1e4,
                 delta: bool = True,
                 layer_idx: Optional[int] = None,
                 save_mem_norm: bool = False,
                 **kwargs,) -> None:
        super().__init__(**kwargs)
        
        # self.bsz = DEF_MAX_BSZ

        # self.__dict__.update(kwargs)

        self.dim = dim

        self.dim_heads = dim // heads if dim_heads is None else dim_heads
        self.h = heads

        self.to_qkv = nn.Linear(dim, 3 * self.h * self.dim_heads, bias=bias, device=self.device)
        self.o_proj = nn.Linear(self.h * self.dim_heads, self.dim, bias=bias, device=self.device)

        self.rope = RoPE(self.dim_heads, base=base, **kwargs) if rope else RoPE.pass_qk

        self.flash = flash
        self.drop_r = drop_r

        self.beta = nn.Parameter(-9 * torch.ones(1, self.h, 1, 1, device=self.device))  # gating factor - init -> sigmoid(b) ~ 0

        self.layer_idx = layer_idx
        self.save_states = save_mem_norm

        self.delta = delta
        self.memory, self.z_norm = None, None
    
    def _retrieve_memory(self, 
                         q: Tensor, 
                         memory: Tensor, 
                         z_norm: Tensor) -> Tensor:
        q_elu = F.elu(q) + 1.
        return (q_elu @ memory) / (q_elu * z_norm + 1e-6)

    def _update_memory(self, 
                       k: Tensor, 
                       v: Tensor, 
                       memory: Tensor, 
                       z_norm: Tensor) -> Tuple[Tensor, Tensor]:
        k_elu = F.elu(k) + 1.
        
        if self.delta:
            v = v - (k_elu @ memory) / (k_elu * z_norm + 1e-6)

        new_mem = memory + k_elu.transpose(-2, -1) @ v
        new_z_norm = z_norm + k_elu.sum(dim=-2, keepdim=True)

        return new_mem, new_z_norm

    def forward(self, 
                segment: Tensor,
                mask: Optional[Tensor] = None,
                prev_states: Dict[int, Tuple[Tensor, Tensor]] = None,
                update_mem: bool = True,
                erase_mem: bool = False,
                ) -> Tuple[Tensor, Tensor, Tensor, Dict[int, Tuple[Tensor, Tensor]]]:
        b, s, d = segment.shape

        x = segment

        qkv = self.to_qkv(x).chunk(3, -1)
        q, k, v = map(
                    lambda w: rearrange(w, "b s (h d) -> b h s d", h = self.h), qkv)
        
        assert self.save_states != None or self.layer_idx != None, "Must have past memory saved internally or passed through"

        if self.save_states:
            if self.memory == None or self.z_norm == None or erase_mem:
                self.memory = q.new_zeros((self.bsz, self.h, self.dim_heads, self.dim_heads), device=self.device)
                self.z_norm = q.new_zeros((self.bsz, self.h, 1, self.dim_heads), device=self.device)
                
            memory = self.memory
            z_norm = self.z_norm

        if self.layer_idx != None:
            if prev_states == None or erase_mem:
                prev_states = {} if prev_states is None else prev_states
                if not self.save_states or erase_mem:
                    memory = q.new_zeros((self.bsz, self.h, self.dim_heads, self.dim_heads), device=self.device)
                    z_norm = q.new_zeros((self.bsz, self.h, 1, self.dim_heads), device=self.device)

                prev_states[self.layer_idx] = (memory, z_norm)
            
            if not self.save_states: 
                memory, z_norm = prev_states[self.layer_idx]

        A_mem = self._retrieve_memory(q, memory[:b, ...], z_norm[:b, ...])
        
        if update_mem:
            new_mem, new_z_norm = self._update_memory(k, v, memory[:b, ...], z_norm[:b, ...])
            
            memory[:b, ...] = new_mem
            z_norm[:b, ...] = new_z_norm

            if self.save_states:
                self.memory = new_mem
                self.z_norm = new_z_norm
            if self.layer_idx != None:
                prev_states[self.layer_idx] = (memory.detach(), z_norm.detach())

        xq, xk = self.rope(q, k)  # apply rotational embeddings

        if self.flash:
            fa = FlashAttention.apply
            A_dot = fa(q, k, v, self.B_q, self.B_kv, mask)
        else:
            A_dot = scaled_dot_product(xq, xk, v, mask=mask, drop_r=self.drop_r)  # perform regular SDP

        # A_mem, A_dot = map(
        #                 lambda a: rearrange(a, "b h s d -> b s (h d)", h=self.h), (A_mem, A_dot))  # concatenate heads
        A = F.sigmoid(self.beta) * A_mem + (1 - F.sigmoid(self.beta)) * A_dot
        
        A = rearrange(A, "b h s d -> b s (h d)")

        out = self.o_proj(A)
        return (out,
                self.memory,
                self.z_norm,
                prev_states)
