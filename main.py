from juju_tools.utils import *
from juju_tools.experiments.ngpt.train import *

def main() -> Optional[Tensor]:
    return create_model()

if __name__ == "__main__":
    logger.info("Starting task... ")
    main()
    logger.info("...Finished")
