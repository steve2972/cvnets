from torch import nn
from .identity import Identity

def get_activation(activation:str, **kwargs):
    avail = {'relu', 'relu6', 'leaky', 'elu', 'selu', 'celu', 'linear', 'silu', 'sigmoid', 'hswish', 'hsigmoid'}
    if activation.lower() not in avail:
        raise NameError(f"Make sure activation is one of {avail}. Received {activation}")
    act = activation.lower()
    if act   == 'relu': act = nn.ReLU(inplace=True)
    elif act =='relu6': act = nn.ReLU6(inplace=True)
    elif act =='leaky': act = nn.LeakyReLU(negative_slope=kwargs["neg_slope"], inplace=True)
    elif act == 'elu' : act = nn.ELU(alpha=1,inplace=True)
    elif act == 'selu': act = nn.SELU(inplace=True)
    elif act == 'silu': act = nn.SiLU(inplace=True)
    elif act == 'celu': act = nn.CELU(alpha=1,inplace=True)
    elif act =='sigmoid':act = nn.Sigmoid()
    elif act =='hsigmoid': act=nn.Hardsigmoid(inplace=True)
    elif act =='hswish':act = nn.Hardswish(inplace=True)
    elif act =='linear': act = Identity()
    return act

class Activation(nn.Module):
    def __init__(self, activation:str='relu', **kwargs) -> None:
        super().__init__()
        act = get_activation(activation, **kwargs)
        self.act = act

    def forward(self, x):
        return self.act(x)