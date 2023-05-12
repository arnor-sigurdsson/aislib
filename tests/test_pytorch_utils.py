import pytest
from torch import nn
import torch

from aislib import pytorch_utils


def test_calc_size_after_conv_sequence():
    class SimpleBlock(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv = nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1
            )
            self.bn = nn.BatchNorm2d(16)
            self.act = nn.ReLU(True)

        def forward(self, x):
            out = self.conv(x)
            out = self.bn(out)
            out = self.act(out)
            return out

    conv_seq = nn.Sequential(*[SimpleBlock()] * 3)
    width, height = pytorch_utils.calc_size_after_conv_sequence(
        input_width=224, input_height=8, conv_sequence=conv_seq
    )

    assert width == 28
    assert height == 1

    input_tensor = torch.rand(1, 16, 224, 8)
    output_tensor = conv_seq(input_tensor)
    assert output_tensor.shape == (1, 16, 28, 1)

    conv_seq_bad = nn.Sequential(*[SimpleBlock()] * 10)
    with pytest.raises(ValueError):
        pytorch_utils.calc_size_after_conv_sequence(
            input_width=224, input_height=8, conv_sequence=conv_seq_bad
        )


@pytest.mark.parametrize(
    "test_input,expected",
    [  # Even input and kernel
        ((1000, 10, 4, 1), (10, 1)),
        ((1000, 10, 4, 3), (10, 0)),
        ((250, 4, 4, 1), (4, 1)),
        # # Odd input, odd kernel
        ((1001, 11, 2, 1), (11, 0)),
        ((1001, 11, 1, 1), (11, 0)),
        ((1001, 11, 4, 2), (11, 0)),
        # # Odd input, mixed kernels
        ((1001, 11, 11, 1), (11, 0)),
        ((1001, 10, 10, 1), (9, 4)),
        ((1001, 11, 3, 2), (12, 0)),
    ],
)
def test_calc_conv_padding_needed_pass(test_input, expected):
    """
    input_width, kernel_size, stride, dilation
    """
    kernel_size, padding = pytorch_utils.calc_conv_params_needed(*test_input)

    assert kernel_size == expected[0]
    assert padding == expected[1]


def test_calc_padding_needed_fail():
    with pytest.raises(ValueError):
        pytorch_utils.calc_conv_params_needed(-1000, 10, 4, 1)
