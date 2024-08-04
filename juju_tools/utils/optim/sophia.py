from juju_tools.utils import *
from torch.optim import Optimizer
from juju_tools.utils.math import Estimator


class Sophia(Optimizer):

    def __init__(self, params, estimator: Estimator, lr: float = 8e-4, betas: tuple[float, float] = (.965, .99),
                 eps: float = 1e-8, k: int = 10, weight_decay: float = 1e-1,
                 rho: float = .022) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rho=rho, k=k, est=estimator)
        super().__init__(params, defaults)

        self.params = params

        self.estimator = estimator
        self.hessian = []

        self.s = 0
        self.k = k

    def estimate_hessian(self, loss: Tensor = null, batch: Tensor = null, params: List[Tensor] = null) -> None:
        if self.s % self.k == 1:
            self.hessian = self.estimator.compute(params, loss, batch)

    def step(self, loss: Tensor = null, batch: Tensor = null) -> None:
        for group in self.param_groups:
            self.estimate_hessian(loss, batch, group["params"])
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                g = p.grad  # parameters' gradient

                state = self.state[p]

                # init state
                if len(state) == 0:
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)  # EMA Momentum
                    state["h"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)  # Hessian

                step = state["step"]

                b1, b2 = group["betas"]
                m: Tensor = state["ema"]  # momentum
                h: Tensor = state["h"]  # hessian

                lr = group["lr"]  # learning rate

                if step % group["k"] == 1:
                    h_hat = self.hessian[i]  # get computed hessian
                    h.mul_(b2).add_(h_hat, alpha=1 - b2)

                m.mul_(b1).add_(g, alpha=1 - b1)

                # Update Parameters
                with torch.no_grad():
                    p.mul_(1 - lr * group["weight_decay"])

                    m_sign = m.sign()

                    dp = (m.abs() / (h.clamp(min=group["eps"]))).clamp(
                        max=group["rho"])

                    p.addcmul_(value=-lr, tensor1=m_sign, tensor2=dp)

                state["step"] += 1

            self.s += 1
