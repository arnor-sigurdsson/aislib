import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Swish(Module):

    __constants__ = ["num_parameters"]

    def __init__(self, num_parameters=1, init=1):
        self.num_parameters = num_parameters
        super(Swish, self).__init__()
        self.weight = Parameter(
            torch.Tensor(num_parameters).fill_(init), requires_grad=True
        )

    def forward(self, input_):
        return input_ * torch.sigmoid(self.weight * input_)

    def extra_repr(self):
        return "num_parameters={}".format(self.num_parameters)


class Mish(Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return mish(input)


@torch.jit.script
def mish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    """
    return input * torch.tanh(F.softplus(input))
