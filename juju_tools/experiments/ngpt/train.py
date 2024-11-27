from juju_tools.utils.consts import *

from juju_tools.models import nGPTAutoRegressive
from juju_tools.utils import nGPTConfig

def create_model():
    config = nGPTConfig(dim=512, n_layers=20)
    logger.info("Creating model with torch compile...")
    model = torch.compile(nGPTAutoRegressive(config=config))
    x = torch.arange(100).cuda()
    x = x.view(10, 10)
    logger.info(x)
    model(x)

if __name__ == "__main__":
    create_model()
