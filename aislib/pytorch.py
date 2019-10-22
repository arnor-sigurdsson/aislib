from torch import nn
from sympy import Symbol
from sympy.solvers import solve


def calc_size_after_conv_sequence(
    input_width: int, conv_sequence: nn.Sequential, axis: int = 1
) -> int:
    """
    Calculates the final width of the input channels going into the fully
    connected layer after encoding.

    TODO:
        - Currently we raise ``ValueError`` if calculated size is 0 according
          after sequentally applying the conv size function calculation.
          Possibly we could do ``current_size = max(current_size, 1)``?
        - Make this function more general / robust.

    :param input_width: Input width before going through convolutions.
    :param conv_sequence: Sequence of convolutions applied to input.
    :param axis: Whether we have tensor height (axis = 0) or width (axis = 1).
    :return: The width of each channel after all convolutions.
    """

    def calc_output_size(size, layer):
        return (
            size - layer.kernel_size[axis] + 2 * layer.padding[axis]
        ) / layer.stride[axis] + 1

    current_size = input_width
    for block in conv_sequence:

        # find all conv operations
        conv_operations = [i for i in vars(block)["_modules"] if i.find("conv") != -1]

        # go over each conv layer, calculating running size
        for operation in conv_operations:
            conv_layer = vars(block)["_modules"][operation]

            current_size = calc_output_size(current_size, conv_layer)

    if int(current_size) == 0:
        raise ValueError(
            "Calculated size after convolution sequence is 0,"
            "check the number of convolutions and their params."
        )

    return int(round(current_size))


def calc_conv_padding_needed(input_width: int, kernel_size: int, stride: int):

    if [i for i in locals().values() if i < 0]:
        raise ValueError(
            f"Got negative value when expected positive in the"
            f"following args passed in: {locals()}."
        )

    p = Symbol("p")
    target_width = int((input_width / stride) + 0.5)
    padding = solve(
        ((input_width - kernel_size + 2 * p) / stride + 1) - target_width, p
    )

    return padding[0]
