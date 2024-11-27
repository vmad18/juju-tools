from juju_tools.utils.consts import *
from juju_tools.utils.optim.scheduler import LRScheduler


class CosineLR(LRScheduler):
    """
    Cosine Annealing Learning Rate Scheduler w/ Warmup
    """
    def __init__(self,
                 max_lr: float,
                 min_lr: float,
                 iters: int,
                 warmup_iters: int = 0, **kwargs):
        super().__init__(kwargs)

        self.max_lr = max_lr
        self.min_lr = min_lr
        self.iters = iters
        self.warmup_iters = warmup_iters

        self.use_warmup = warmup_iters != 0

    def compute_lr(self) -> float:
        if self.iter < self.warmup_iters:
            return self.max_lr * self.iter / self.warmup_iters
        if self.iter - self.warmup_iters > self.iters:
            return self.min_lr

        gate = (self.iter - self.warmup_iters) / (self.iters - self.warmup_iters)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(gate * pi))

    def step(self):
        for group in self.param_groups:
            group["lr"] = self.compute_lr()
        self.iter += 1
