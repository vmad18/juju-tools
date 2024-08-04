from juju_tools.utils import *


class S4DKernel(Module):
    
    def __init__(self, 
                 dim: int,
                 state_size: int = 64,
                 dt_min: float = 0.01,
                 dt_max: float = 0.1):
        
        self.delta = 

        self.A_real = nn.Parameter(-.5 * torch.ones(dim, state_size//2), requires_grad=True)
        self.A_img = nn.Parameter(pi * torch.arange(state_size//2).repeat(dim, 1), requires_grad=True)
        
        self.C = nn.Parameter(torch.normal(0, .5 ** .5, (dim, state_size, 2)))

