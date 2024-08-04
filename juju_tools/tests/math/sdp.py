from utils.consts import *
from utils.math.scaled_dot_product import FlashAttention, scaled_dot_product
import time


class mha_test(Module):

    def __init__(self,
                 dim: int,
                 heads: int,
                 drop_r: float = 0.,
                 bias: bool = True,
                 flash: bool = False):
        super().__init__()

        self.h = heads

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias, device="cuda")
        self.o_proj = nn.Linear(dim, dim, device="cuda")

        self.flash = flash
        self.drop_r = drop_r

    def forward(self, x: Tensor, B_q=0, B_kv=0, mask=None) -> Tensor:
        qkv = self.to_qkv(x).chunk(3, -1)

        q, k, v = map(
            lambda w: rearrange(w, "b s (h d) -> b h s d", h=self.h),
            qkv
        )

        if self.flash:
            fa = FlashAttention.apply
            x = fa(q, k, v, B_q, B_kv, mask)
        else:
            x = scaled_dot_product(q, k, v, mask=mask)

        x = rearrange(x, "b h s d -> b s (h d)", h=self.h)

        return F.dropout(x, p=self.drop_r, training=True)


def flash_test():
    N, dim = 16384 // 2, 128

    x = torch.randn((1, N, dim), requires_grad=True).cuda()

    mha_flash = mha_test(128, heads=8, drop_r=0, flash=True)

    start = time.time_ns()
    g = mha_flash(x, 1024, 2048)
    end = time.time_ns()

    print((end - start) / (10 ** 9))

    del x
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    #################################################################
    x = torch.randn((1, N, dim), requires_grad=True).cuda()

    mha_flash = mha_test(128, heads=8, drop_r=0, flash=False)

    start = time.time_ns()
    g = mha_flash(x)
    end = time.time_ns()

    print((end - start) / (10 ** 9))

    del x
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def backward_test():
    N, dim = 16384 // 2, 128

    x = torch.randn((1, N, dim), requires_grad=True).cuda()

    mha_flash = mha_test(128, heads=8, drop_r=0, flash=True)

    g = mha_flash(x, 1024, 2048)

    start = time.time_ns()
    g.sum().backward()
    end = time.time_ns()

    print((end - start) / (10 ** 9))


if __name__ == "__main__":
    flash_test()
    backward_test()

# class wrapper(Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, q, k, v):
#         return scaled_dot_product(q, k, v)
#
#
# class test_nn(Module):
#
#     def __init__(self):
#         super().__init__()
#
#         self.l1 = nn.Linear(10, 20)
#         self.nl = nn.SiLU()
#         self.l2 = nn.Linear(20, 30)
#         self.nl2 = nn.SiLU()
#         self.l3 = nn.Linear(30, 10)
#         self.nl3 = nn.Softmax(dim=-1)
#
#     # testnn = test_nn()
#     #
#     # x = torch.randn((3, 10), requires_grad=True)
#     # o = testnn(x)
#     # o.retain_grad()
#     # o.cumprod(0).backward()
#     # print(x.grad)
#     # print(o.grad)
#
#     def forward(self, q: Tensor):
#         return self.nl3(self.l3(self.nl2(self.l2(self.nl(self.l1(q))))))
#
#
# if __name__ == "__main__":
#     q = torch.randn((1, 16384, 128)).cuda()
#     k = torch.randn((1, 16384, 128)).cuda()
#     v = torch.randn((1, 16384, 128)).cuda()
#     # print(q, k, v)
#     fa = FlashAttention.apply
#     wrap = wrapper()
#
#     a = 1
#     '''
#     .052  .043  .063
#     .0321   .028
#     '''
#
#     if a == 1:
#         start = time.time_ns()
#         g = fa(q, k, v, 1024, 2048)
#         end = time.time_ns()
#
#         print((end - start) / (10 ** 9))
#     else:
#         # print(torch.cat(g[0], dim=-2)[0, 1, :])
#         start = time.time_ns()
#         g = wrap(q, k, v)  # F.scaled_dot_product_attention(q, k, v)
#         end = time.time_ns()
#
#         print((end - start) / (10 ** 9))
