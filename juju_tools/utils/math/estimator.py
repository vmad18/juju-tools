from juju_tools.utils.consts import *


"""

Estimates the Hessian for Pre-conditioning

"""


class Estimator:

    def __init__(self) -> None: pass

    def compute(self, params: List[Tensor], loss: Tensor, logits: Optional[Tensor] = null) -> List[Tensor]:
        pass
