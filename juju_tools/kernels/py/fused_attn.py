from juju_tools.utils.consts import *

import triton
import triton.language as tl

import torch.autograd

# this doesn't work locally - i have a 1080 :(
# TODO compute exp in exp2

@triton.jit
def attn_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        l_m_ptr,
        sm_scale,
        q_stride_b, q_stride_h, q_stride_m,
        k_stride_b, k_stride_h, k_stride_n,
        v_stride_b, v_stride_h, v_stride_n,
        o_stride_b, o_stride_h, o_stride_m,
        PAST_LEN,
        q_rounded,
        k_len,
        HEAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        EVEN_HEAD_DIM: tl.constexpr,
        TRAINING: tl.constexpr,
        CAUSAL: tl.constexpr):
    m_id = tl.program_id(0)
    bh_id = tl.program_id(1)

    offset_b = bh_id // HEAD
    offset_h = bh_id % HEAD

    offset_m = m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = tl.arange(0, BLOCK_N)
    offset_dim = tl.arange(0, HEAD_DIM)

    q_len = k_len - PAST_LEN

    q_ptrs = q_ptr + offset_b * q_stride_b + offset_h * q_stride_h + (
            offset_m[:, None] * q_stride_m + offset_dim[None, :])
    k_ptrs = k_ptr + offset_b * k_stride_b + offset_h * k_stride_h + (
            offset_n[None, :] * k_stride_n + offset_dim[:, None])  # shape [d, s] | loads k transposed
    v_ptrs = v_ptr + offset_b * v_stride_b + offset_h * v_stride_h + (
            offset_n[:, None] * v_stride_n + offset_dim[None, :])

    if EVEN_M:
        if EVEN_HEAD_DIM:
            q = tl.load(q_ptrs)  # load q to sram
        else:
            q = tl.load(q_ptrs, mask=offset_dim[None, :] < HEAD_DIM, other=0.)
    else:
        if EVEN_HEAD_DIM:
            q = tl.load(q_ptrs, mask=offset_m[:, None] < q_len, other=0.)
        else:
            q = tl.load(q_ptrs, mask=(offset_m[:, None] < q_len) & (offset_dim[None, :] < HEAD_DIM), other=0.)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + float("-inf")  # SM row max term
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # SM denominator

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)  # accumulator

    start_blck = 0
    end_blck = k_len if not CAUSAL else tl.minimum((m_id + 1) * BLOCK_M, k_len)

    for start_n_blck in range(start_blck, end_blck, BLOCK_N):
        start_n = tl.multiple_of(start_n_blck, BLOCK_N)

        s_ij = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        if EVEN_N:
            if EVEN_HEAD_DIM:
                k_t = tl.load(k_ptrs + start_n * k_stride_n)  # load k_t to SRAM
                v = tl.load(v_ptrs + start_n * v_stride_n)  # load v to SRAM
            else:
                k_t = tl.load(k_ptrs + start_n * k_stride_n, mask=offset_dim[:, None] < HEAD_DIM, other=0.)
                v = tl.load(v_ptrs + start_n * v_stride_n, mask=offset_dim[None, :] < HEAD_DIM, other=0.)
        else:
            if EVEN_HEAD_DIM:
                k_t = tl.load(k_ptrs + start_n * k_stride_n, mask=(offset_n + start_n)[None, :] < k_len, other=0.)
                v = tl.load(v_ptrs + start_n * v_stride_n, mask=(offset_n + start_n)[:, None] < k_len, other=0.)
            else:
                k_t = tl.load(k_ptrs + start_n * k_stride_n,
                              mask=((offset_n + start_n)[None, :] < k_len) & (offset_dim[:, None] < HEAD_DIM), other=0.)
                v = tl.load(v_ptrs + start_n + v_stride_n,
                            mask=((offset_n + start_n)[:, None] < k_len) & (offset_dim[None, :] < HEAD_DIM), other=0.)

        s_ij += tl.dot(q, k_t)
        s_ij *= sm_scale

        if CAUSAL:
            s_ij = tl.where(offset_m[:, None] + PAST_LEN >= (start_n + offset_n)[None, :], s_ij, float("-inf"))
        elif not EVEN_N:
            s_ij = tl.where((start_n + offset_n)[None, :] < k_len, s_ij, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(s_ij, 1))
        fix_m_i = tl.exp(m_i - m_i_new)

        p_ij = tl.exp(s_ij - m_i_new[:, None])
        l_i = fix_m_i * l_i + tl.sum(p_ij, 1)

        acc = fix_m_i[:, None] * acc + tl.dot(p_ij, v)

        m_i = m_i_new

    acc = 1 / l_i[:, None] * acc

    if TRAINING:
        logsum_l_m = m_i + tl.log(l_i)
        l_m_ptrs = l_m_ptr + bh_id * q_rounded + offset_m

        if EVEN_M:
            tl.store(l_m_ptrs, logsum_l_m)
        else:
            tl.store(l_m_ptrs, logsum_l_m, mask=offset_m < q_len)

    out_ptrs = o_ptr + offset_b * o_stride_b + offset_h * o_stride_h + (
            offset_m[:, None] * o_stride_m + offset_dim[None, :])
    if EVEN_M:
        if EVEN_HEAD_DIM:
            tl.store(out_ptrs, acc)
        else:
            tl.store(out_ptrs, acc, mask=offset_dim[None, :] < HEAD_DIM)
    else:
        if EVEN_HEAD_DIM:
            tl.store(out_ptrs, acc, mask=offset_m[:, None] < q_len)
        else:
            tl.store(out_ptrs, acc, mask=(offset_m[:, None] < q_len) & (offset_dim[None, :] < HEAD_DIM))


@triton.jit
def attn_bckwd_inner_do_o_kernel(
        do_pntr, o_pntr, d_pntr,
        do_stride_m, o_stride_m,
        BLOCK_M: tl.constexpr,
        HEAD_DIM: tl.constexpr,

):
    bhm_id = tl.program_id(0)

    offset_m = bhm_id * BLOCK_M + tl.arange(0, BLOCK_M)

    offset_d = tl.arange(0, HEAD_DIM)

    do_pntrs = do_pntr + offset_m[:, None] * do_stride_m + offset_d[None, :]
    o_pntrs = o_pntr + offset_m[:, None] * o_stride_m + offset_d[None, :]

    do = tl.load(
        do_pntrs).to(tl.float32)  # load dO to SRAM
    o = tl.load(
        o_pntrs).to(tl.float32)  # load O to SRAM

    D = tl.sum(do * o, axis=1)  # inner prod.

    d_pntrs = d_pntr + offset_m
    tl.store(d_pntrs, D)  # write back to DRAM


@triton.jit
def attn_bckwd(
        q_ptr, k_ptr, v_ptr,
        dq_ptr, dk_ptr, dv_ptr, do_ptr,
        lse_ptr, d_ptr,
        q_stride_b, q_stride_h, q_stride_m,
        k_stride_b, k_stride_h, k_stride_n,
        v_stride_b, v_stride_h, v_stride_n,
        dq_stride_b, dq_stride_h, dq_stride_m,
        dk_stride_b, dk_stride_h, dk_stride_n,
        dv_stride_b, dv_stride_h, dv_stride_n,
        do_stride_b, do_stride_h, do_stride_m,
        PAST_LEN,
        q_rounded,
        k_len,
        sm_scale,
        HEAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        EVEN_HEAD_DIM: tl.constexpr,
        CAUSAL: tl.constexpr,
):
    n_id = tl.program_id(0)
    bh_id = tl.program_id(1)

    offset_b = bh_id // HEAD
    offset_h = bh_id % HEAD

    offset_m = tl.arange(0, BLOCK_M)
    offset_n = n_id * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_d = tl.arange(0, HEAD_DIM)

    k_ptrs = k_ptr + offset_b * k_stride_b + offset_h * k_stride_h + (
            offset_n[:, None] * k_stride_n + offset_d[None, :])
    v_ptrs = v_ptr + offset_b * v_stride_b + offset_h * v_stride_h + (
            offset_n[:, None] * v_stride_n + offset_d[None, :])

    if EVEN_N:
        if EVEN_HEAD_DIM:
            k = tl.load(k_ptrs)  # load k to SRAM
            v = tl.load(v_ptrs)  # load v to SRAM
        else:
            k = tl.load(k_ptrs, mask=offset_d[None, :] < HEAD_DIM, other=0.)
            v = tl.load(v_ptrs, mask=offset_d[None, :] < HEAD_DIM, other=0.)
    else:
        if EVEN_HEAD_DIM:
            k = tl.load(k_ptrs, mask=offset_n[:, None] < k_len, other=0.)
            v = tl.load(v_ptrs, mask=offset_n[:, None] < k_len, other=0.)
        else:
            k = tl.load(k_ptrs,
                        mask=((offset_n[:, None] < k_len) & (offset_d[None, :] < HEAD_DIM)), other=0.)
            v = tl.load(v_ptrs,
                        mask=(offset_n[:, None] < k_len) & (offset_d[None, :] < HEAD_DIM), other=0.)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    q_ptrs_base = q_ptr + offset_b * q_stride_b + offset_h * q_stride_h
    dq_ptrs_base = dq_ptr + offset_b * dq_stride_b + offset_h * dq_stride_h
    do_ptrs_base = do_ptr + offset_b * do_stride_b + offset_h * do_stride_h

    d_ptrs_base = d_ptr + bh_id * q_rounded
    lse_ptrs_base = lse_ptr + bh_id * q_rounded

    q_len = k_len - PAST_LEN

    start = 0
    end_blck = q_len

    for start_m_blck in range(start, end_blck, BLOCK_M):
        start_m = tl.multiple_of(start_m_blck, BLOCK_M) + offset_m

        q_ptrs = q_ptrs_base + (start_m[:, None] * q_stride_m + offset_d[None, :])
        dq_ptrs = dq_ptrs_base + (start_m[:, None] * dq_stride_m + offset_d[None, :])
        do_ptrs = do_ptrs_base + (start_m[:, None] * do_stride_m + offset_d[None, :])

        if EVEN_M:
            if EVEN_HEAD_DIM:
                q = tl.load(q_ptrs)  # load q to SRAM
                do = tl.load(do_ptrs)  # load do to SRAM
            else:
                q = tl.load(q_ptrs, mask=offset_d[None, :] < HEAD_DIM, other=0.)
                do = tl.load(do_ptrs, mask=offset_d[None, :] < HEAD_DIM, other=0.)
        else:
            if EVEN_HEAD_DIM:
                q = tl.load(q_ptrs, mask=start_m[:, None] < q_len, other=0.)
                do = tl.load(do_ptrs, mask=start_m[:, None] < q_len, other=0.)
            else:
                q = tl.load(q_ptrs,
                            mask=((start_m[:, None] < q_len) & (offset_d[None, :] < HEAD_DIM)), other=0.)
                do = tl.load(do_ptrs,
                             mask=(start_m[:, None] < q_len) & (offset_d[None, :] < HEAD_DIM), other=0.)

        lse_ptrs = lse_ptrs_base + start_m
        d_ptrs = d_ptrs_base + start_m

        s_i = tl.dot(q, tl.trans(k))
        s_i *= sm_scale

        if CAUSAL:
            s_i = tl.where(start_m[:, None] + PAST_LEN >= offset_n[None, :], s_i, float("-inf"))
        elif not EVEN_N:
            s_i = tl.where(offset_n[None, :] < k_len, s_i, float("-inf"))  # CHANGE THIS TO s_i = ... (..., s_i, ...) // save w/ in fwd_attn

        if EVEN_M:
            lse = tl.load(lse_ptrs)  # load LSE to SRAM
            d = tl.load(d_ptrs)  # load D to SRAM
        else:
            lse = tl.load(lse_ptrs, start_m < q_len, other=0.)
            d = tl.load(d_ptrs, start_m < q_len, other=0.)

        p_i = tl.exp(s_i - lse[:, None])  # SM of q * k^T

        dv += tl.dot(tl.trans(p_i.to(q_ptrs.dtype.element_ty)), do)

        dp = tl.dot(do, tl.trans(v))
        ds = p_i * (dp - d[:, None]) * sm_scale

        if EVEN_M:
            if EVEN_HEAD_DIM:
                dq = tl.dot(ds.to(q_ptrs.dtype.element_ty), k)
                tl.atomic_add(dq_ptrs, dq)
            else:
                dq = tl.dot(ds.to(q_ptrs.dtype.element_ty), k)
                tl.atomic_add(dq_ptrs, dq, mask=offset_d[None, :] < HEAD_DIM)
        else:
            if EVEN_HEAD_DIM:
                dq = tl.dot(ds.to(q_ptrs.dtype.element_ty), k)
                tl.atomic_add(dq_ptrs, dq, mask=start_m[:, None] < q_len)
            else:
                dq = tl.dot(ds.to(q_ptrs.dtype.element_ty), k)
                tl.atomic_add(dq_ptrs, dq, mask=(start_m[:, None] < q_len) & (offset_d[None, :] < HEAD_DIM))

        dk += tl.dot(tl.trans(ds.to(q_ptrs.dtype.element_ty)), q)

    dk_ptrs = dk_ptr + offset_b * dk_stride_b + offset_h * dk_stride_h + (
            offset_n[:, None] * dk_stride_n + offset_d[None, :])
    dv_ptrs = dv_ptr + offset_b * dv_stride_b + offset_h * dv_stride_h + (
            offset_n[:, None] * dv_stride_n + offset_d[None, :])

    if EVEN_N:
        if EVEN_HEAD_DIM:
            tl.store(dk_ptrs, dk)  # write dk to DRAM
            tl.store(dv_ptrs, dv)  # write dv to DRAM
        else:
            tl.store(dk_ptrs, dk, mask=offset_d[None, :] < HEAD_DIM)
            tl.store(dv_ptrs, dv, mask=offset_d[None, :] < HEAD_DIM)
    else:
        if EVEN_HEAD_DIM:
            tl.store(dk_ptrs, dk, mask=offset_n[:, None] < k_len)
            tl.store(dv_ptrs, dv, mask=offset_n[:, None] < k_len)
        else:
            tl.store(dk_ptrs, dk,
                     mask=((offset_n[:, None] < k_len) & (offset_d[None, :] < HEAD_DIM)))
            tl.store(dv_ptrs, dv,
                     mask=(offset_n[:, None] < k_len) & (offset_d[None, :] < HEAD_DIM))


def attn_forward(ctx, q, k, v, training: bool = False, causal: bool = True):
    BLOCK_M = 32
    BLOCK_N = 64
    HEAD = q.shape[1]

    HEAD_DIM = q.shape[-1]

    EVEN_HEAD = HEAD_DIM == HEAD_DIM

    o = torch.empty_like(q).contiguous().cuda()

    EVEN_M = q.shape[-2] % BLOCK_M == 0
    EVEN_N = k.shape[-2] % BLOCK_N == 0
    TRAINING = training
    CAUSAL = causal

    q_len = q.shape[-2]
    k_len = k.shape[-2]
    past_len = k_len - q_len

    grid = (triton.cdiv(q_len, BLOCK_M), q.shape[0] * q.shape[1], )
    q_rounded = grid[0] * BLOCK_M

    logsum_l_m = torch.empty((q.shape[0] * q.shape[1], q_rounded),
                             device=q.device, dtype=q.dtype)

    num_warps = max(1, 2 ** int(math.log2(max(BLOCK_M, BLOCK_N, HEAD_DIM) / 32)))

    sm_scale = 1. / math.sqrt(HEAD_DIM)

    attn_fwd_kernel[grid](
        q, k, v, o,
        logsum_l_m, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        past_len, q_rounded, k_len,
        HEAD, BLOCK_M, BLOCK_N, EVEN_M, EVEN_N,
        HEAD_DIM, EVEN_HEAD, TRAINING, CAUSAL, num_warps=num_warps, num_stages=1)

    ctx.save_for_backward(q, k, v, o, logsum_l_m)

    ctx.BLOCK_M = BLOCK_M
    ctx.BLOCK_N = BLOCK_N
    ctx.EVEN_M = EVEN_M
    ctx.EVEN_N = EVEN_N
    ctx.HEAD_DIM = HEAD_DIM
    ctx.EVEN_HEAD = EVEN_HEAD
    ctx.grid = grid
    ctx.sm_scale = sm_scale
    ctx.PAST_LEN = past_len
    ctx.q_rounded = q_rounded
    ctx.CAUSAL = CAUSAL
    ctx.num_warps = num_warps

    return o


class attn_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, training: bool, causal: bool):
        return attn_forward(ctx, q, k, v, training, causal)

    @staticmethod
    def backward(ctx, do):
        with torch.no_grad():
            if do.stride(0) == 0:
                do = do.contiguous()

            q, k, v, o, logsum_l_m = ctx.saved_tensors

            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

            d = torch.empty_like(logsum_l_m)

            pre_grid = (ctx.grid[0] * ctx.grid[1],)
            grid = (triton.cdiv(k.shape[-2], ctx.BLOCK_N), k.shape[0] * k.shape[1],)

            attn_bckwd_inner_do_o_kernel[pre_grid](do, o, d, do.stride(2), o.stride(2),
                                                   ctx.BLOCK_M, ctx.HEAD_DIM)  # computes doT * o

            attn_bckwd[grid](q, k, v, dq, dk, dv, do, logsum_l_m, d,
                             q.stride(0), q.stride(1), q.stride(2),
                             k.stride(0), k.stride(1), k.stride(2),
                             v.stride(0), v.stride(1), v.stride(2),
                             dq.stride(0), dq.stride(1), dq.stride(2),
                             dk.stride(0), dk.stride(1), dk.stride(2),
                             dv.stride(0), dv.stride(1), dv.stride(2),
                             do.stride(0), do.stride(1), do.stride(2),
                             ctx.PAST_LEN,
                             ctx.q_rounded,
                             k.shape[-2],
                             ctx.sm_scale,
                             q.shape[1],
                             ctx.BLOCK_M,
                             ctx.BLOCK_N,
                             ctx.EVEN_M,
                             ctx.EVEN_N,
                             q.shape[-1],
                             ctx.EVEN_HEAD,
                             ctx.CAUSAL, num_warps=ctx.num_warps, num_stages=1)

        return dq, dk, dv, None, None, None


def flash_attn(q, k, v, training, causal):
    return attn_func.apply(q, k, v, training, causal)
