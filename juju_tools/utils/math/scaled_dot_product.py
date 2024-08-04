from torch.autograd import Function
from juju_tools.utils import *


def scaled_dot_product(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        scaling_norm: Optional[float] = None,
        mask: Optional[Tensor] = None,
        drop_r: float = 0.,
        training: bool = False) -> Tensor:

    r"""

    As proposed in AIAYN

    Attends each token in k with q, creating an attention mask.
    Attention mask is placed over v.

    Args:
        q(Tensor): - query tensor [..., S, D]
        k(Tensor): - key tensor [..., W, D]
        v(Tensor): - value tensor [..., W, D]
        scaling_norm(float | None): - scales the attention map (default = None)
        mask(Tensor | None): - mask tensor [..., S, W] (default = None)
        drop_p(float) - dropout rate (defualt = 0.)

    Returns:
        Tensor - [..., S, D]
    
    """

    norm = scaling_norm if not (scaling_norm is None) else 1. / sqrt(q.shape[-1])
    mask = torch.zeros((1, 1, q.shape[-2], k.shape[-2]), device=q.device, dtype=torch.float32) if mask is None else mask

    attends = (q @ k.transpose(-2, -1)).float() * norm + mask

    attn = F.softmax(attends, dtype=torch.float32, dim=-1).to(q.dtype)
    attn = F.dropout(attn, p=drop_r, training=training)
    
    return attn @ v





# WORKS?!
# TODO Use pntr reference to tensor instead of split
# TODO make sure masking is implemented correctly
# PyTorch version of Flash Attention 2 (don't use this :p)
class FlashAttention(Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                B_q: int,
                B_kv: int,
                mask: Tensor | None = None) -> Tensor:
        norm_scale = 1. / sqrt(k.shape[-1])
        # diff = abs(k.shape[-2] - q.shape[-2])
        # m_shape = min(k.shape[-2], q.shape[-2])

        # if mask.ndim <= 2:
        #     mask = rearrange(mask, "")

        q_blocks = torch.split(q, B_q, -2)  # Use pntrs instead of torch.split
        k_blocks = torch.split(k, B_kv, -2)
        v_blocks = torch.split(v, B_kv, -2)

        T_q = len(q_blocks)  # ceil(q.shape[-2] / self.B_q)
        T_kv = len(k_blocks)  # ceil(k.shape[-2] / self.B_kv)

        mask = torch.zeros((*q.shape[:-1], B_kv)).cuda() if mask is None else mask
        mask_blocks = torch.split(mask, B_q, -2)

        o = list(torch.split((torch.zeros_like(q)).to(q.device), B_q,
                             -2))  # might need to change if k and q are different Ns
        l = list(torch.split((torch.zeros((*q.shape[:-1], 1))).to(q.device), B_q, -2))  # [..., N / T_q, 1]
        m = list(
            torch.split((torch.zeros((*q.shape[:-1], 1)) + float("-inf")).to(q.device), B_q, -2))  # [..., N / T_q, 1]
        lsexp = list(torch.split((torch.zeros((*q.shape[:-1], 1)) + float("-inf")).to(q.device), B_q,
                                 -2))  # [..., N / T_q, 1]... T_q

        for i in range(T_q):
            q_i = q_blocks[i]
            mask_i = mask_blocks[i]
            o_i = o[i]
            l_i = l[i]
            m_i = m[i]
            lsexp_i = lsexp[i]
            for j in range(T_kv):
                k_j, v_j = k_blocks[j], v_blocks[j]

                s_ij = torch.einsum("... i d, ... j d -> ... i j", q_i,
                                    k_j) * norm_scale  # q_i @ k_j.transpose(-2, -1) * norm_scale #
                s_ij = torch.where(mask_i == 0, s_ij, float("-inf"))

                rowmax = torch.amax(s_ij, dim=-1, keepdim=True)

                m_ij = torch.maximum(rowmax, lsexp_i)
                p_ij = torch.exp(s_ij - m_ij)

                l_ij = torch.exp(m_i - m_ij) * l_i + torch.sum(p_ij, dim=-1, keepdim=True) + 1e-10
                o_ij = torch.exp(lsexp_i - m_ij) * o_i + p_ij @ v_j

                o_i = o_ij
                l_i = l_ij
                m_i = m_ij

                lsexp_i = m_ij + torch.log(torch.exp(lsexp_i - m_ij) + l_i)

            lsexp[i] = lsexp_i
            o[i] = torch.exp(m_i - lsexp_i) * o_i

        lsexp = torch.cat(lsexp, dim=-2)
        o = torch.cat(o, dim=-2)

        ctx.save_for_backward(o, lsexp, q, k, v)
        ctx.args = (B_q, B_kv, T_q, T_kv, mask)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx,
                 grad):

        o, lsexp, q, k, v = ctx.saved_tensors
        B_q, B_kv, T_q, T_kv, mask = ctx.args

        scale = 1. / sqrt(k.shape[-1])

        do = grad

        mask = torch.zeros((*q.shape[:-1], B_kv), requires_grad=True).cuda() if mask is None else mask
        mask_blocks = list(torch.split(mask, B_q, -2))

        q_blocks = torch.split(q, B_q, -2)  # Use pntrs instead of torch.split
        k_blocks = torch.split(k, B_kv, -2)
        v_blocks = torch.split(v, B_kv, -2)

        o_blocks = list(torch.split(o, B_q, -2))
        do_blocks = list(torch.split(do, B_q, -2))

        lsexp_blocks = list(torch.split(lsexp, B_q, -2))

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        dq_blocks = list(torch.split(dq, B_q, -2))
        dk_blocks = list(torch.split(dk, B_kv, -2))
        dv_blocks = list(torch.split(dv, B_kv, -2))

        d_blocks = list(torch.split(torch.sum(do * o, dim=-1, keepdim=True), B_q, -2))

        for j in range(T_kv):
            k_j, v_j, dk_j, dv_j = k_blocks[j], v_blocks[j], dk_blocks[j], dv_blocks[j]

            for i in range(T_q):
                q_i = q_blocks[i]
                dq_i = dq_blocks[i]

                o_i = o_blocks[i]
                do_i = do_blocks[i]

                lsexp_i = lsexp_blocks[i]
                d_i = d_blocks[i]

                mask_i = mask_blocks[i]

                s_ij = torch.einsum("... i d, ... j d -> ... i j", q_i, k_j) * scale
                s_ij = torch.where(mask_i == 0, s_ij, float("-inf"))

                p_ij = torch.exp(s_ij - lsexp_i)

                dv_blocks[j] = dv_j + torch.einsum("... d i, ... d j -> ... i j", p_ij, do_i)

                dp_ij = torch.einsum("... i d, ... j d -> ... i j", do_i, v_j)

                ds_ij = p_ij * (dp_ij - d_i)

                dq_blocks[i] = dq_i + ds_ij @ k_j
                dk_blocks[j] = dk_j + torch.einsum("... d i, ... d j -> ... i j", ds_ij, q_i)

        dq = torch.cat(dq_blocks, -2)
        dk = torch.cat(dk_blocks, -2)
        dv = torch.cat(dv_blocks, -2)

        return dq, dk, dv, None, None, None, None
