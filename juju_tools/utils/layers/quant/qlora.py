from juju_tools.utils import *
from juju_tools.utils.layers.quant.nfk import LinearNFK

class QLoRA(Module):

    r"""

    QLoRA or Quantized Low Rank Adapter is a method for efficiently fine-tuning
    large-language models with less compute resources. The weights of the LLM are quantized and 
    frozen and only two matrices are optimized: A and B. QLoRA is formalized as: 
    output = dequant(quant(W_model)) * x + s * BA * x
    s is a scaling parameter, and it is the ratio of the hyperparameters: alpha, r

    Should be used to adapt to the attention layer weights of LLMs.

    Args:
        dim (int): - The dimensions of the weights
        weights (Tensor): - The weights of the frozen model
        r (float): - The lower dimension output projection dimension
        alpha (float): - The Scaling factor
        drop_r (float): - The dropout rate value

    """

    def __init__(self, dim: int, weights: Tensor, r: float, alpha: float = 1., drop_r: float = 0.):
        super().__init__()
        
        assert r > 0, "r must be greater than 0"

        self.scale = alpha / r

        weights.requires_grad = False
        self.linear_n4k = LinearNFK(weights=weights, requires_grad=False)

        self.A = nn.Parameter(weights.new_zeros((r, dim)), requires_grad=True)
        self.B = nn.Parameter(weights.new_zeros((r, dim)), requires_grad=True)        
        
        self.drop_r = drop_r

        self.init()

    def init(self) -> None:
        nn.init.normal_(self.A)

    def forward(self, x: Tensor) -> Tensor:
        x_m = self.nfk(F.dropout(x, p=self.drop_r))
        h = x_m + self.scale * ((x @ self.A.T) @ self.B.T)
        return h
