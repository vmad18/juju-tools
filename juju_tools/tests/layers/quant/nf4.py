import torch
from scipy.stats import norm
from utils.consts import *
from utils.layers.quant.NFK import NormalKBitParameter, LinearNFK, NFK, FeedForwardNF4
from utils.layers import linear_weight

W_BLOCK = 64
C_BLOCK = 256


def scales_quant(c_scales: Tensor) -> Tuple[Tensor, Tensor]:
    n = c_scales.numel() // C_BLOCK

    c_scales_blocked = c_scales.view(n, C_BLOCK)
    c_scales_scales = 127 / c_scales_blocked.abs().amax(-1, keepdim=True)

    c_scales_quant = (c_scales_blocked * c_scales_scales).round().clamp(-128, 127)

    return c_scales_quant, c_scales_scales


def scales_dequant(c_scales_quant: Tensor, c_scales_scales: Tensor) -> Tensor:
    return (c_scales_quant / c_scales_scales).view(-1, 1)


def quant_weights_scales(W: Tensor, normal_floats: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    n = W.numel() // W_BLOCK

    W_blocked = W.view(n, W_BLOCK)
    c_scales = W_blocked.abs().amax(-1, keepdim=True)

    W_scaled = W_blocked / c_scales

    diff = (W_scaled.flatten().view(-1, 1) - normal_floats.view(1, -1)).abs()

    W_quant = normal_floats[diff.argmin(-1)].view(n, W_BLOCK)

    c_scales_quant, c_scales_scales = scales_quant(c_scales)

    return W_quant, c_scales_quant, c_scales_scales


def double_dequant(W_quant: Tensor, c_scales_quant: Tensor, c_scales_scales: Tensor) -> Tensor:
    c_scales = scales_dequant(c_scales_quant, c_scales_scales)
    W_dequant = W_quant * c_scales
    return W_dequant


def main() -> None:
    # e = .967752
    #
    # v1 = (-norm.ppf(torch.linspace(.5, e, 8))).tolist()[1:]
    # v2 = (norm.ppf(torch.linspace(.5, e, 9))).tolist()
    #
    # g = v1 + v2
    # g.sort()
    # nf4 = torch.tensor(g)
    # nf4 /= nf4.amax()
    #
    # nf4 = compute_normal_fpk(bits=4)
    # print(nf4)
    # w = torch.randn((20, 512, 1024))
    #
    # W_quant, c_scales_quant, c_scales_scales = quant_weights_scales(w, nf4)
    # W_dequant = double_dequant(W_quant, c_scales_quant, c_scales_scales).to(torch.bfloat16)
    # print(W_dequant.view(w.shape))
    # print(w)

    # x = linear_weight(512, 1024)  # torch.randn((1024, 512), requires_grad=True)

    # nfk = NFK(x, 4, 64, 256)

    x = torch.randn((10, 512)).cuda()
    ffn = FeedForwardNF4(512, 2)
    
    print(ffn(x))

    # proj = LinearNFK(weight=x)

    # x = proj(x)

    # x.sum().backward()

    # W = torch.randn((512, 1024))
    #
    # nfk = NormalKBitParameter(W, bits=4, W_BLOCKS=W_BLOCK, C_BLOCKS=C_BLOCK)
    #
    # print(W)
    # print(nfk.dequant())

    # n_w = w.numel() // W_BLOCK
    # w_blocked = w.flatten().view(n_w, W_BLOCK)
    #
    # c_scales = w_blocked.abs().amax(-1, keepdim=True)
    #
    # c_scales_quant, c_scales_scales = scales_quant(c_scales)
    #
    # c_scales_dequant = scales_dequant(c_scales_quant, c_scales_scales)
    # print(c_scales_dequant)
    # print(c_scales)

    # c_mean = c.view(1, n_w).mean(-1, keepdim=True)
    #
    # c_sub = c - c_mean
    #
    # n_c = c_sub.numel() // c_block
    #
    # c_blocked = c_sub.view((n_c, c_block))
    #
    # c_blocked_amax = c_blocked.abs().amax(-1, keepdim=True)
    #
    # print(c_blocked)
    #
    # c_blocked = (128 * c_blocked / c_blocked_amax).round().clamp(-128, 127)
    #
    # print(c_blocked.shape)
    #
    # c_dequant = ((c_blocked / (128 / c_blocked_amax)).view(1, n_w) + c_mean).view()
    # print(c)
    # print(c_dequant)
    # # c_sub = c_sub.flatten()
    #
    # w_blocked = w_blocked / c
    #
    # w_quant = nf4[(nf4.view(1, -1) - w_blocked.flatten()[..., None]).abs().argmin(-1)].view(n_w, w_block)
    #
    # w_back = (w_quant * c).view((512, 1024))
    # # print(w_back - w)
    # #
    # # print(w_quant.mean(-1))
    # # print(w_quant.var(-1))


if __name__ == '__main__':
    main()
