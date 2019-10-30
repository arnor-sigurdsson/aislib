import pytest
from torch import nn

from aislib import pytorch_utils


def test_calc_size_after_conv_sequence():
    class SimpleBlock(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv = nn.Conv2d(16, 16, 4, 2, 1)
            self.bn = nn.BatchNorm2d(16)
            self.act = nn.ReLU(True)

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    conv_seq = nn.Sequential(*[SimpleBlock()] * 3)
    size = pytorch_utils.calc_size_after_conv_sequence(224, conv_seq)

    assert size == 28

    conv_seq_bad = nn.Sequential(*[SimpleBlock()] * 10)
    with pytest.raises(ValueError):
        pytorch_utils.calc_size_after_conv_sequence(224, conv_seq_bad)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ((1000, 10, 4, 1), 3),
        ((1000, 10, 4, 3), 3),
        ((250, 4, 4, 1), 1),
        ((1001, 11, 2, 1), 5),
        ((1001, 11, 1, 1), 5),
    ],
)
def test_calc_conv_padding_needed_pass(test_input, expected):
    assert pytorch_utils.calc_conv_padding_needed(*test_input) == expected


def test_calc_padding_needed_fail():
    with pytest.raises(ValueError):
        pytorch_utils.calc_conv_padding_needed(-1000, 10, 4, 1)
