from juju_tools.utils import *
from juju_tools.utils.layers.quant import LinearNFK 

class BasicQuantizer(Module):

    def __init__(self, model: Module, quant_method: str = "nf4"):
        super().__init__() 
        
