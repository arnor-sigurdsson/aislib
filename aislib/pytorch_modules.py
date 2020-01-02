import torch
from torch.nn import Module
from torch.nn.parameter import Parameter


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
