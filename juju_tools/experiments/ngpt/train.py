from contextlib import nullcontext

from juju_tools.utils.consts import *

from juju_tools.models import nGPTAutoRegressive
from juju_tools.utils import nGPTConfig, Config
from juju_tools.utils.optim import LRScheduler

from torch.amp import GradScaler

class TrainerArgs:

    def __init__(self, **kwargs):

        self.bsz = DEF_MAX_BSZ
        self.max_tokens = DEF_MAX_TOK

        self.iterations = 1e6
        self.gradient_accumulate_steps = 64
        self.grad_clip = 0.0

        self.opt_type = "adam"

        self.lr = 1e-4
        self.scheduler: Optional[LRScheduler] = None

        self.weight_decay: float = 0.0
        self.betas: Optional[tuple[int, int]] = None

        self.loss_func = F.cross_entropy

        self.is_cuda = torch.cuda.is_available()

        self.__dict__.update(kwargs)

def create_model(compile: bool = True, **kwargs) -> Tuple:
    logger.info("Creating model...")
    config = nGPTConfig(**kwargs)
    model = nGPTAutoRegressive(config=config)
    if compile:
        model = torch.compile(model)

    return config, model

def train(config: Config,
          model: Module,
          train_args: TrainerArgs):
    torch.manual_seed(1747)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    grad_scalar = GradScaler(enabled=(config.dtype!=torch.float32))
    opt = model.configure_optimizers(train_args.opt_type, train_args.weight_decay, train_args.lr, train_args.betas, config.device)

    ctx = nullcontext() if config.device == 'cpu' else torch.amp.autocast(device_type=config.device, dtype=config.dtype)

    iter_num = 0

    while True:

        if train_args.scheduler is not None:
            train_args.scheduler.step()

        for idx in range(train_args.gradient_accumulate_steps):
            x, y = None, None # get batch

            # deal with DDP all reduce operation
            with ctx:
                y_hat = model(x)
                loss = train_args.loss_func(y, y_hat)
                loss /= train_args.gradient_accumulate_steps

            grad_scalar.scale(loss).backward()

        if train_args.grad_clip != 0:
            grad_scalar.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
        grad_scalar.step(opt)
        grad_scalar.update()

        opt.zero_grad(set_to_none=True)

        iter_num+=1

        if iter_num > train_args.iterations:
            break

if __name__ == "__main__":
    train_args = TrainerArgs(grad_clip = 1.0, betas=(0.94, 0.95))
    args = dict(dim=512, n_layers=20)
    config, model = create_model(compile=True, **args)