from juju_tools.utils import *
from juju_tools.utils.math.estimator import Estimator


class Hutchinson(Estimator):
    
    """

    Computes the Diagonal of Hessian

    :param params - model parameters
    :param loss - model loss
    :param batch - mini batch

    """

    def __init__(self) -> None:
        super().__init__()

    def compute(self, params: List[Tensor], loss: Tensor, logits: Optional[Tensor] = null) -> \
            List[Tensor]:

        u: List[Tensor] = [torch.randn_like(p) for p in params]

        J: List[Tensor] = [p.grad for p in params]

        # J: Tuple[Tensor] = torch.autograd.grad(loss, params, create_graph=true)  # compute jacobian, idk why i did this when we already computed gradients in backward pass

        gradu: List[Tensor] = []  # dot products with gradient and noise

        for i, g in enumerate(J):
            gradu.append((g * u[i]).sum())

        hvp: Tuple[Tensor, ...] = torch.autograd.grad(gradu, params,
                                                      retain_graph=True)  # compute hessian vector product

        hessian: List[Tensor] = []

        for i, grad in enumerate(hvp):
            hessian.append(grad * u[i])
        return hessian

