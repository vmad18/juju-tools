from juju_tools.utils import *

#TODO make sure complex implementation was done correctly
#TODO learn cuda :P so it is as optimized as what the real paper purposes.
#TODO gotta have that mamba mentaity
class MambaBlock(Module): 
    
    r"""

    MambaBlock fuses an MLP with sequence transformations within the projections. The transformations
    consist of a convolution over the sequence and select state space modeling over the sequence. 
    Compared to MHA it linearly scales with sequence length. 

    Args:
        dim(int): - Input dimension of the data
        scale(int): - The scale to project the dimensions by (scale * dim)
        state_size(int): - The latent size of the state space
        conv_size(int): - The kernel size of the 1D convolution performed across the sequence
        A_init(str): - The initialization scheme of the A state matrix
        d_size(str | None): - The initial size of the discretization parameter
        nl(Callable[[Tensor], Tensor]): - The non-linearity function to apply after higher space projections (default SiLU)

    """

    def __init__(self,
                 dim: int,
                 scale: int,
                 state_size: int,
                 conv_size: int,
                 A_init: str = "auto",
                 d_size: int | None = None,
                 nl: Callable[[Tensor], Tensor] = F.silu,
                 save_state: bool = False,
                 layer_idx: Optional[int] = None,
                 ) -> None:
        super().__init__() 

        self.proj_dim = scale * dim
        self.R = d_size if d_size != None else self.proj_dim // state_size  
        self.state_size = state_size

        self.proj_x_res = nn.Linear(dim, 2*self.proj_dim) # projects x and residual connection
        self.proj_seq = nn.Conv1d(
                        in_channels=self.proj_dim,
                        out_channels=self.proj_dim,
                        kernel_size=conv_size,
                        groups=self.proj_dim,
                        padding=conv_size-1
        ) # projects the sequence

        self.proj_s = nn.Linear(self.proj_dim, 2 * state_size + self.R) # projects x to B, C, and delta
        self.proj_tau = nn.Linear(self.R, self.proj_dim)
        
        self.A_init = A_init

        if self.A_init == "auto" or self.A_init == "real_linear":
            A = torch.arange(1, state_size + 1, dtype=torch.float32).repeat(self.proj_dim, 1)
            self.A = nn.Parameter(A.log(), requires_grad=True)
        elif A_init == "complex_linear":
            self.A_real = nn.Parameter(torch.log(-.5 * torch.ones((self.proj_dim, self.state_size // 2))), requires_grad=True)
            self.A_img = nn.Parameter(pi * torch.arange(self.state_size // 2).repeat(self.proj_dim, 1), requires_grad=True)

        self.D = nn.Parameter(torch.zeros(self.proj_dim, 1), requires_grad=True) # S4 says initialize as 0/N(0, 1)
        
        self.proj_o = nn.Linear(self.proj_dim, dim)

        self.nl = nl

        self.save_state = save_state
        self.layer_idx = layer_idx
        
        self.hidden_state: Tensor = None

        self.weight_init()

    def weight_init(self) -> None:
        nn.init.uniform_(self.proj_tau.bias.data, .001, .1)

    def forward(self, 
                x: Tensor, 
                ctx: Tensor = None,
                past_states: Dict[int, Tensor] = None) -> Tuple[Tensor, Tensor, Dict[int, Tensor]]:

        b, l, d = x.shape

        x, res = self.proj_x_res(x).chunk(2, -1)
        x = self.proj_seq(rearrange(x, "b l d -> b d l"))[..., :l]
        x = rearrange(x, "b d l -> b l d")
        
        x = self.nl(x)

        if self.A_init == "auto" or self.A_init == "real_linear":
            A = -self.A.exp()
        elif self.A_init == "complex_linear":
            A = self.A_real.exp() + 1.0j * self.A_img 
        
        B, C, delta = self.proj_s(x).split([self.state_size, self.state_size, self.R], -1)
        delta = F.softplus(self.proj_tau(delta))
    
        if self.A_init == "auto" or self.A_init == "real_linear":
            C = C 
        elif self.A_init == "complex_linear":
            C = torch.view_as_complex(rearrange(C, "b l (n, c) -> b l n c", c=2))

        dA = torch.einsum("b l d, d n -> b l d n", delta, A) # A * delta[..., None]
        A_bar = dA.exp()
        B_bar = 1./dA * (dA.exp() - torch.eye(dA.shape[-2], dA.shape[-1])) * torch.einsum("b l d, b l n -> b l d n", delta, B)
        
        if ctx != None:
            x, h = self._ssm(A_bar, B_bar, C, x, ctx)
        elif self.save_state:
            if self.hidden_state == None:
                x, h = self._ssm(A_bar, B_bar, C, x, None)
                self.hidden_state = h
            else:
                x, h = self._ssm(A_bar, B_bar, C, x, self.hidden_state)
                self.hidden_state = h
        
        if self.layer_idx != None:
            if past_states == None:
                past_states = {}
                if self.save_state:
                    pass
                else:
                    x, h = self._ssm(A_bar, B_bar, C, x, None)
            elif self.save_state:
                pass
            else:
                x, h = self._ssm(A_bar, B_bar, C, x, past_states[self.layer_idx])

            past_states[self.layer_idx] = h

        x = x * self.nl(res)
        return self.proj_o(x), h, past_states

    def _ssm(self, A: Tensor, B: Tensor, C: Tensor, x_t_prev: Tensor, h: Tensor | None) -> Tuple[Tensor, Tensor]:
        
        r"""
        
        Performs SSM or state space modeing over a sequence. Linearly iterates through 
        each element in the sequence in x_t_prev and aggregates a hidden state, h which 
        can be used for predicting the next token. 

        Args: 
            A (Tensor): - A is the selection matrix that fuses the previous state's signal - similar to a gating the hidden state in an RNN. Shape: (B, L, D, N) if real
            B (Tensor): - B is the selection matrix that provides information from the input sequence - similar to gating the input sequence in an RNN. Shape: (B, L, D, N) if real
            C (Tensor): - C is the selection matrix that determines how much of the hidden state should go into the output, y. Shape: (B, L, N) if real
            x_t_prev (Tensor): - Input sequence. Shape: (B, L, D)
            h (Tensor | None): - Initial context/hidden state to be used for SSM. Shape: (B, L, D)
        
        Returns: Tuple[Tensor, Tensor]

        """

        b, l, d = x_t_prev.shape
        
        if self.A_init == "auto" or self.A_init == "real_linear":
            B = B
        elif self.A_init == "complex_linear":
            B = torch.view_as_complex(rearrange(B, "b l d (n c) -> b l d n c", c=2)) 
        else:
            B = B
            
        B_x_t_prev = x_t_prev[..., None] * B

        h = x_t_prev.new_zeros((b, d, self.state_size)) if h is None else h
        y = []
        for i in range(l):
            h = h * A[:, i] + B_x_t_prev[:, i]
            y_t = torch.einsum("b d n, b n -> b d", h, C[:, i])
            y.append(y_t)
        
        y = torch.stack(y, dim=1) + x_t_prev * self.D.T 
        return y, h
