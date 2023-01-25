from torch import nn, Tensor

class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x:Tensor):
        return x