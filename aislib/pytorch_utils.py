import math
from typing import Tuple, Union

from sympy import Symbol
from sympy.solvers import solve
from torch import nn


def calc_size_after_conv_sequence(
    input_width: int, input_height: int, conv_sequence: nn.Sequential
) -> Tuple[int, int]:
    current_width = input_width
    current_height = input_height
    for block in conv_sequence:
        conv_operations = [i for i in vars(block)["_modules"] if i.find("conv") != -1]

        for operation in conv_operations:
            conv_layer = vars(block)["_modules"][operation]

            current_width = _calc_output_size(
                size=current_width, layer=conv_layer, axis=1
            )
            current_height = _calc_output_size(
                size=current_height, layer=conv_layer, axis=0
            )

    if int(current_width) == 0 or int(current_height) == 0:
        raise ValueError(
            "Calculated size after convolution sequence is 0, "
            "check the number of convolutions and their parameters."
        )

    return int(current_width), int(current_height)


def _calc_output_size(size: int, layer: nn.Module, axis: int):
    kernel_size = layer.kernel_size[axis]
    padding = layer.padding[axis]
    stride = layer.stride[axis]
    dilation = layer.dilation[axis]

    output_size = (
        (size + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride
    ) + 1

    return output_size


def calc_conv_params_needed(
    input_width: int, kernel_size: int, stride: int, dilation: int
) -> Tuple[int, int]:
    if input_width < 0:
        raise ValueError("Got negative size for input width: %d", input_width)

    target_width = math.ceil((input_width / stride))

    for t_width in [target_width, target_width - 1, target_width + 1]:
        for k_size in [kernel_size, kernel_size - 1, kernel_size + 1]:
            padding = _solve_for_padding(
                input_width=input_width,
                target_width=t_width,
                dilation=dilation,
                stride=stride,
                kernel_size=k_size,
            )

            if padding is not None:
                assert isinstance(padding, int)
                return k_size, padding

    raise AssertionError(
        f"Could not find a solution for padding with the supplied conv "
        f"parameters: {locals()}."
    )


def _solve_for_padding(
    input_width: int, target_width: int, dilation: int, stride: int, kernel_size: int
) -> Union[int, None]:
    p = Symbol("p", integer=True)
    padding = solve(
        ((input_width + (2 * p) - dilation * (kernel_size - 1) - 1) / stride + 1)
        - target_width,
        p,
    )

    if len(padding) > 0:
        return int(padding[0])

    return None
