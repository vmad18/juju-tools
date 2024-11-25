from juju_tools.utils.consts import *

from juju_tools.models import nGPTAutoRegressive
from juju_tools.utils import nGPTConfig

def create_model():
    config = nGPTConfig(dim=20, n_layers=20)
    logger.info(config.alpha_a_scale)
    logger.info(config.n_layers)
    # model = nGPTAutoRegressive(config=config)

if __name__ == "__main__":
    create_model()