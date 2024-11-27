from juju_tools.utils import *
from torch.optim import Optimizer

class ADOPT(Optimizer):

    def __init__(self, params, lr: float,
                 betas: tuple[float, float], eps: float,
                 weight_decay: float):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.params = params


    def step(self, loss: Tensor = None, batch: Tensor = None) -> None:
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                g = p.grad  # grad theta_i
                state = self.state[p] # state of theta_i

                if len(state) == 0:
                    state["step"] = 0
                    