from juju_tools.utils import *

class PatchEmbed2d(ModuleConfig):

    r"""
    
    Converts input spatial data to N x M patches. If specified, 
    adds a class token to the front and learnable positional 
    information to the patchified data. 

    Args:
        dim (int): - Input dimension (ex. images dim = 3)
        patch_size (int | Tuple[int, int]) - Size of spatial patches (int - N x N)
        H (int): - Input spatial dimension height
        W (int): - Input spatial dimension width
        edim (int): - Embedding dimension to project to 
        cls (bool): - Set to add class token (default = True)
        pos_embed (bool): - Set to add positional information (defualt = True)

    """

    def __init__(self,
                 dim: int, 
                 patch_size: int | Tuple[int, int], 
                 H: int, 
                 W: int,
                 edim: Optional[int] = None,
                 cls: bool = True,
                 pos_embed: bool = True,
                 **kwargs,):
        super().__init__(**kwargs)
        
        patch_size = (patch_size, patch_size) if type(patch_size) is int else patch_size

        assert H % patch_size[0] == 0 and W % patch_size[1] == 0, "Number of patches must be divisible by the height and width"

        
        count = H // patch_size[0] * W // patch_size[1]

        self.patchify = nn.Conv2d(dim, 
                               dim if edim == None else edim, 
                               kernel_size=patch_size, 
                               stride=patch_size, 
                               device=self.device)
        self.pos = None
        if pos_embed:
            assert edim != None, "Embedding dimension cannot be None"
            count = count + 1 if cls else count
            self.pos = nn.Parameter(torch.randn((count, edim), device=self.device))
        
        self.cls_token = None

        if cls:
            self.cls_token = torch.randn((1, 1, edim if edim != None else dim), device=self.device)
        
    def forward(self, x: Tensor) -> Tensor:
        b, *_ = x.shape
        x = rearrange(x, "... h w c -> ... c h w")
        x = self.patchify(x).flatten(-2).transpose(-2, -1)

        if self.cls_token != None:
            x = torch.cat([self.cls_token.repeat(b, 1, 1), x], -2)

        if self.pos != None:
            x = x + self.pos

        return x
