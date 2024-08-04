from utils.consts import *
from utils.layers.transforms.feed_forward import FeedForward

if __name__ == '__main__':
    x = torch.randn((3, 10))
    l = FeedForward(10, scale=2, nl=F.silu)
    print(l(x).shape)