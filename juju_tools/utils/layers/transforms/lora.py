from juju_tools.utils import *


class LoRA(Module):

    r"""

    LoRA or Low Rank Adapter is a method for efficiently fine-tuning
    large-language models. We keep the weights of LLM frozen and only tune
    two matrices: A and B. LoRA is formalized as: output = W_model * x + s * BA * x

    s is a scaling parameter, and it is the ratio of the hyperparameters: alpha, r

    Adapts to the attention layer weights of LLMs.

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

        self.A = nn.Parameter(weights.new_zeros((r, dim)), requires_grad=True)
        self.B = nn.Paramter(weights.new_zeros((dim , r)), requires_grad=True)
        self.weight = weights

        self.weight.requires_grad = False

        self.drop_r = drop_r
    
    def init() -> None:
        nn.init.normal_(self.A)

    def forward(self, x: Tensor) -> Tensor:
        x_m = F.linear(F.dropout(x, p=self.drop_r), self.weight)
        h = x_m + self.scale * (x @ self.A.T) @ self.B.T
        return h
