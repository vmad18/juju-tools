import time

from einops.layers import torch

from juju_tools.utils import *
from scipy.stats import norm

def compute_normal_fpk(bits, e=.967752) -> Tensor:
    half_1 = 2 ** bits // 2
    half_2 = 2 ** bits // 2 + 1

    v1 = (-norm.ppf(torch.linspace(.5, e, half_1, dtype=torch.float16))).tolist()[1:]
    v2 = (norm.ppf(torch.linspace(.5, e, half_2, dtype=torch.float16))).tolist()

    g = v1 + v2
    g.sort()
    nf4 = torch.tensor(g)
    nf4 /= nf4.amax()

    return nf4.to(torch.bfloat16)

PRECOMPUTED_NORMAL_FP4 = compute_normal_fpk(bits=4)

# TODO rename this to TensorNFK or something
# TODO act. rename this to just TensorN4K, will not work for...
# TODO well maybe I can do something w/ padding and the number of bits.
# TODO Modulo bits with blocksize

class NFK:
    r"""
    
    Double blockwise quantization technique that respects the distirbution of model weights (gaussian).

    Data is quantized to k bits. Absmax scales are quantized to 8 bits

    Args:
        data (Tensor): - The data to be quantized
        bits (int): - The number of bits of quantized data
        W_BLOCKS (int): - The size of each quantized data block (lower, more percison) - default = 64
        C_BLOCKS (int): - The size of each quantized absmax scale block (lower, more percision) - default = 256 

    """

    def __init__(self,
                 data: Tensor,
                 bits: int,
                 W_BLOCKS: int = 64,
                 C_BLOCKS: int = 256,
                 device="cuda"):

        self.bits = bits
        self.Q_b = 2 ** self.bits

        self.W_BLOCKS = W_BLOCKS
        self.C_BLOCKS = C_BLOCKS

        self.nfk_vals = compute_normal_fpk(bits).to(device)

        self.out_shape = data.shape

        assert data.numel() % W_BLOCKS == 0 and data[..., 0].numel() % C_BLOCKS == 0

        self.W_quant, self.c_scales_quant, self.c_scales_scales, self.c_scales_mean = None, None, None, None

        self.quant(data)

    def _scales_quant(self, c_scales: Tensor) -> Tuple[Tensor, Tensor]:

        c_scales_mean = c_scales.mean()

        c_scales = c_scales - c_scales_mean

        n = c_scales.numel() // self.C_BLOCKS

        c_scales_blocked = c_scales.view(n, self.C_BLOCKS)
        c_scales_scales = (127 / c_scales_blocked.abs().amax(-1, keepdim=True)).to(torch.float32)

        c_scales_quant = (c_scales_blocked * c_scales_scales).round().clamp(-128, 127).to(torch.int8)

        return c_scales_quant, c_scales_scales, c_scales_mean

    def _scales_dequant(self) -> Tensor:
        return (self.c_scales_quant.to(torch.float32) / self.c_scales_scales).view(-1, 1).to(torch.bfloat16) + self.c_scales_mean

    def quant(self, data: Tensor) -> Tuple[Tensor, ...]:
        n = data.numel() // self.W_BLOCKS

        W_blocked = data.view(n, self.W_BLOCKS)
        c_scales = W_blocked.abs().amax(-1, keepdim=True)
        W_scaled = (W_blocked / c_scales).view(-1, 1)

        diff = W_scaled - self.nfk_vals.view(1, -1)

        try:
            diff = diff.abs()
        except:
            # idk why this needs to be done to avoid a CUDA out of memory error
            print("CUDA OUT OF MEMORY!")
            for i in range(0, diff.shape[0], diff.shape[0] // 4):
                end = min(diff.shape[0], diff.shape[0] // 4)
                diff[i:i + end, :] = diff[i:i + end, :].abs()

        # del data
        # torch.cuda.empty_cache()
        # diff = torch.abs(diff)

        W_quant = diff.argmin(-1).view(-1, 1)  # .view(n, self.W_BLOCKS)
        c_scales_quant, c_scales_scales, c_scales_mean = self._scales_quant(c_scales)

        # logger.info(f"W_QUANT {W_quant}")
        # logger.info(f"C SCALES {c_scales}")
        # logger.info(f"QUANT SCALES {c_scales_quant}")
        # logger.info(f"SCALE SCALES {c_scales_scales}")
        # logger.info(f"MEAN {c_scales_mean}")
        # to avoid having to create a custom 4 bit float datatype, we can recognize that
        # two 4-bit values stores the same amount of space as one 8-bit value
        # thus, we fuse each adjacent feature by shifting the first value by
        # 4 bits then joining the adj. 4 bits to the end of the value.
        # first 4 bits -> abcd
        # shift by 4 bits -> abcd0000
        # add the adj. 4 bits -> abcd0000 | efgh
        # result is eq. to two stacked 4 bits abcdefgh
        W_quant = (W_quant[::2] << 4).to(torch.uint8) | (W_quant[1::2]).to(torch.uint8)

        self.W_quant = W_quant  # .cuda()
        self.c_scales_quant = c_scales_quant
        self.c_scales_scales = c_scales_scales
        self.c_scales_mean = c_scales_mean

        return W_quant, c_scales_quant, c_scales_scales, c_scales_mean

    def dequant(self) -> Tensor:
        c_scales = self._scales_dequant()

        #(-1, self.W_BLOCKS)

        unwrapped = torch.empty((2 * self.W_quant.numel(), 1), device="cuda").to(torch.uint8) # .cuda()
        unwrapped[::2] = self.W_quant >> 4  # get the first 4 bits.
        unwrapped[1::2] = self.W_quant & 0b1111  # get the last 4 bits

        W_near_nf4 = self.nfk_vals[unwrapped.to(torch.long)].view(-1, self.W_BLOCKS)
        W_dequant = W_near_nf4.to(torch.bfloat16) * c_scales.to(torch.bfloat16)

        return W_dequant.view(self.out_shape)


# TODO should remove this, there doesn't seem to be a use for this
class NormalKBitParameter(NFK):

    def __init__(self, data: Tensor, bits: int, W_BLOCKS: int, C_BLOCKS: int, device="cpu") -> None:
        NFK.__init__(self, data.to(device), bits, W_BLOCKS, C_BLOCKS)


class LinearNFK(Module):
    r"""

    LinearNFK or Linear Normal Float K-Bit performs a projection of input
    vector, x, by normally distributed 4-bit float weights. Weights are 
    quantized to 4-bit floats then dequantized to blfloat16 in the forward pass.

    Args:
        weights(Tensor | None): - Provided transformation weights
        in_features(int | None): - Input feature dimension
        out_features(int | None): - Output projected space
        bits(int): - Number of bits to represents Weights (default 4)
        W_BLOCKS(int): - Size of blocks to split the weights for quantization, lower more percision (default 64)
        C_BLOCKS(int): - Size of blocks for weight absmax scales quantization, lower more percision (default 256)
        requires_grad(bool): - Keep the weights frozen if False (default True) 

    """

    def __init__(self,
                 in_features: int = None,
                 out_features: int = None,
                 bits: int = 4,
                 W_BLOCKS: int = 64,
                 C_BLOCKS: int = 256,
                 requires_grad=False,
                 device: str = "cuda", **kwargs) -> None:
        super().__init__()

        self.weight = None
        self.bits = bits
        self.W_BLOCKS = W_BLOCKS
        self.C_BLOCKS = C_BLOCKS

        self.device = device
        self.__dict__.update(kwargs)

        if in_features is not None and out_features is not None:
            self.weight = torch.randn((out_features, in_features), requires_grad=requires_grad, device=device).detach()

        # if kwargs.keys().__contains__("weight"):
        #     if isinstance(kwargs["weight"], Tensor):
        #         self.weight = NormalKBitParameter(kwargs["weight"], bits, W_BLOCKS, C_BLOCKS)

        assert self.weight is not None and isinstance(self.weight,
                                                      Tensor), "Need in_features and out_features or provided weights (tensor)"

        self.weight = NFK(self.weight, bits, W_BLOCKS, C_BLOCKS, device)

        # def get_data(self) -> Tensor:

    #     return self.weight.data

    # def load_weight(self, 
    #                 weight: Tensor, 
    #                 is_quant: bool = False) -> None:
    #     if is_quant:
    #         self.weight = weight 
    #         return 
    #     
    #     self.weight = NormalKBitParameter(weight, self.bits, 
    #                                       self.W_BLOCKS, self.C_BLOCKS, device=self.device)

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype

        x = x.to(torch.bfloat16)
        # if cuda.is_available():
        #     return (x @ self.weight.dequant().T.to("cuda")).to(dtype)
        # logger.info(f"{self.weight.dequant().T}")
        # start = time.time()
        # w = self.weight.dequant()
        # logger.info(f"This took  {time.time() - start}  seconds to compute")

        result = (x @ self.weight.dequant().T).to(dtype)

        return result


class FeedForwardNF4(Module):
    r"""
    FFN based normal 4-bit floating point layer. 

    Args:
        dim(int): - Input dimension
        scale(int): - Scaling factor for up projected dimension
        nl(Callable[[Tensor], Tensor]]): - Non liearity function
        drop_r(float): - Dropout rate value (default = 0.)
    
    """

    def __init__(self, dim: int, scale: int, nl: Callable[[Tensor], Tensor] = F.silu, drop_r: float = 0.):
        super().__init__()

        self.proj = LinearNFK(dim, 2 * scale * dim)
        self.proj_out = LinearNFK(scale * dim, dim)

        self.nl = nl

        self.drop_r = drop_r

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, -1)
        return F.dropout(self.proj_out(self.nl(x * gate)), p=self.drop_r, training=self.training)
