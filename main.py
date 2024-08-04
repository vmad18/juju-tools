from juju_tools.utils import *
from juju_tools.models.llama3.test_model import run_local_model, get_hf_weights


def main(**kwargs) -> Optional[Tensor]:
    return run_local_model()


if __name__ == "__main__":
    logger.info("Starting task... ")
    main()
    logger.info("...Finished")
