from juju_tools.utils.consts import *

def linear_weight(
        in_features,
        out_features,
        requires_grad=True) -> Tensor:
    return torch.nn.init.xavier_uniform_(torch.zeros((out_features, in_features), requires_grad=requires_grad))
