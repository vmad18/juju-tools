from utils.consts import *
from utils.layers.quant.bit_net import round_clip, BitLinear




if __name__ == "__main__":
    net = BitLinear(512, 6)
    x = torch.randn((512, 512))
    print(net(x))

