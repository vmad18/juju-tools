from juju_tools.utils.consts import *


class LRScheduler(object):
    def __init__(self, param_groups):
        self.param_groups = param_groups
        self.iter = 0
        self.iters = 0

    def compute_lr(self) -> float:
        pass

    def step(self):
        pass

    def sweep(self):
        for _ in range(self.iters):
            yield self.iter, self.compute_lr()
            self.iter+=1
        self.iter = 0