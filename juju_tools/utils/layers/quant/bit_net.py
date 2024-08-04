from juju_tools.utils import *

def round_clip(x: Tensor, bits: int = 1, round: bool = True, eps: float = 0.) -> Tensor:
    Q_b = 2 ** (bits - 1)

    if round:
        x = x.round()

    return x.clamp(-Q_b + eps, Q_b - eps)


def standardize(x: Tensor, eps: float = 1e-6) -> Tensor:
    return (x - x.mean()) / (x.std() + eps)


class BitLinear(Module):
    
    r"""

    BitLinear is a special forward projection where the fp32 weights are represented as 
    1.58 bits. During training, the layer learns how to quantize the fp32 weights 
    to 1.58 bit having values in the set {-1, 0, 1} (log2(3) = 1.58). During inference
    all all we need are the 1.58 bit weight matricies. 

    Args:
       in_features (int): - The input dimension
       out_features (int): - The output dimension to project to 
       bits (int): - The number of bits to quantize down to


    """

    def __init__(self, in_features: int, out_features: int, bits: int = 8) -> None:
        super().__init__()

        self.W = nn.Parameter(torch.randn((out_features, in_features)), requires_grad=True)

        self.bits = bits
        self.Q_b = 2 ** (bits - 1)

    def forward(self, x: Tensor) -> Tensor:
        W = self.W

        beta = W.abs().mean()
        W_quant = round_clip(W / (beta + 1e-6), bits=1)  # quantize weights - w_ij in {-1, 0, 1}

        z = standardize(x)

        gamma = z.abs().amax(dim=-1, keepdim=True)  # inf norm
        z_quant = round_clip(z * self.Q_b / gamma, bits=self.bits, round=False, eps=1e-6)

        y = (z_quant @ W_quant.T) * beta * gamma / self.Q_b  # de-quant
        return y
