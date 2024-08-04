from juju_tools.utils import *

from juju_tools.utils.math import Estimator


# TODO test this
class GaussNewtonBartlett(Estimator):

    def __init__(self, loss_func: Callable[[Tensor], Tensor]):
        super().__init__()

        self.loss_func = loss_func

    def compute(self, params: List[Tensor], loss: Tensor, logits: Optional[Tensor]) -> List[Tensor]:
        
        if logits != None:
            B, _ = logits.shape
        
            y_hat = F.softmax(logits, dim=-1)
            loss_sum = self.loss_func(y_hat, logits).sum() * 1 / B
            g_hat: List[Tensor] = list(torch.autograd.grad(loss_sum, inputs=params, retain_graph=True))
        else:
            g_hat: List[Tensor] = list(torch.autograd.grad(params, inputs=params, retain_graph=True))

        for i, g in enumerate(g_hat):
            g_hat[i] = B * g.square()

        return g_hat[i]
