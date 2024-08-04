from juju_tools.utils import *

import triton
import triton.language as tl


@triton.jit
def add_kernel(
                x_ptr, 
                y_ptr, 
                out_ptr,
                b_stride,
                num_elm, 
                BLOCK_SIZE: tl.constexpr):
    c_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    curr_stride = b_idx.to(tl.int64) * b_stride + c_idx * BLOCK_SIZE
    
    offset = curr_stride + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_elm 

    x = tl.load(x_ptr + offset, mask=mask) # load x to SRAM
    y = tl.load(y_ptr + offset, mask=mask) # load y to SRAM

    result = x+y

    tl.store(out_ptr + offset, result, mask=mask) # load output back to DRAM/HBM


def vec_add_wrap(x: Tensor, y: Tensor):
    
    assert x.shape[-1] == y.shape[-1] and x.dim() == 2, "Vectors are not of the same shape!"
    
    BLOCK_SIZE = 128 
    num_elm = x.numel()
    out = torch.empty_like(x)
    b = x.shape[0]
    grid = lambda meta: (triton.cdiv(num_elm, meta["BLOCK_SIZE"]), b, )

    add_kernel[grid](x, y, out, x.stride(0), num_elm, BLOCK_SIZE, num_warps=2)

    return out 


