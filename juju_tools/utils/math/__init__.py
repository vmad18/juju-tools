from juju_tools.utils.math.causal_mask import *
from juju_tools.utils.math.gauss_newton_bartlett import *
from juju_tools.utils.math.hutchinson import *
from juju_tools.utils.math.scaled_dot_product import *


"""

Estimates the Hessian for Pre-conditioning

"""


class Estimator:

    def __init__(self) -> None: pass

    def compute(self, params: List[Tensor], loss: Tensor, logits: Optional[Tensor] = null) -> List[Tensor]:
        pass
