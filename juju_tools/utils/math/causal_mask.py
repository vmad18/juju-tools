from juju_tools.utils import *

'''

Gives causual attention mask

:param S - Rows dim
:param W - Cols dim
:param shift - Where mask shift starts
:param value - Value to fill mask

:return causal relation mask

'''


def causal_mask(
        S: int,
        W: int,
        shift: int | None = 1,
        value: float = None 
    ) -> Tensor:
    shift = shift if shift != None else 1
    value = value if value != None else float("-inf") # torch.finfo(torch.get_default_dtype()).min
    mask = torch.full((S, W), fill_value=value).triu(diagonal=shift)
    return mask


if __name__ == "__main__":
    pass
