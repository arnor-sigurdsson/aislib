from pathlib import Path

import numpy as np
import pytest

from aislib import plink_utils
from aislib.misc_utils import ensure_path_exists

ONE_HOT_MAPPING_DICT = {
    0: np.array([1, 0, 0, 0]),
    1: np.array([0, 1, 0, 0]),
    2: np.array([0, 0, 1, 0]),
    9: np.array([0, 0, 0, 1]),
}
OHMP = ONE_HOT_MAPPING_DICT


@pytest.fixture()
def create_test_data(tmp_path: Path) -> Path:
    """
    Format:
        The header line contains only the rs_IDs.
        FID	        Family ID
        IID	        Within-family ID
        PAT	        Paternal within-family ID
        MAT	        Maternal within-family ID
        SEX	        Sex (1 = male, 2 = female, 0 = unknown)
        PHENOTYPE	Main phenotype value
    """
    test_data_list = [
        ["rs1", "rs2", "rs3", "rs4", "rs5", "rs6"],
        ["FID_1", "IID_1", "PAT_1", "MAT_1", "M", "1"] + ["0", "1", "2", "NA"],
        ["FID_2", "IID_2", "PAT_2", "MAT_2", "M", "1"] + ["0", "0", "0", "0"],
        ["FID_3", "IID_3", "PAT_3", "MAT_3", "M", "1"] + ["0", "1", "2", "NA"],
        ["FID_4", "IID_4", "PAT_4", "MAT_4", "M", "1"] + ["0", "1", "2", "NA"],
        ["FID_5", "IID_5", "PAT_5", "MAT_5", "M", "1"] + ["2", "2", "2", "2"],
    ]

    test_file_raw_path = tmp_path / "testfile.raw"

    with open(str(test_file_raw_path), "w") as outfile:
        for line in test_data_list:
            outfile.write(" ".join(line) + "\n")

    return test_file_raw_path


@pytest.mark.parametrize("test_input", [0, 1, 2, 9])
def test_get_plink_raw_encoder(test_input):
    encoder = plink_utils.get_plink_raw_encoder()

    test_input_array = [[test_input]]
    expected = OHMP[test_input]
    assert (encoder.transform(test_input_array) == expected).all()

    with pytest.raises(ValueError):
        encoder.transform([[5]])


# ohmp_keys are grabbed from create_test_data fixture expected genotypes
@pytest.mark.parametrize(
    "test_input_file,ohmp_keys",
    [
        ("IID_1.npy", [0, 1, 2, 9]),
        ("IID_2.npy", [0, 0, 0, 0]),
        ("IID_3.npy", [0, 1, 2, 9]),
        ("IID_4.npy", [0, 1, 2, 9]),
        ("IID_5.npy", [2, 2, 2, 2]),
    ],
)
def test_plink_raw_to_one_hot(test_input_file, ohmp_keys, create_test_data):
    test_raw_path = create_test_data

    test_output_folder = test_raw_path.parent / "encoded_outputs"
    ensure_path_exists(test_output_folder, is_folder=True)

    encoder = plink_utils.get_plink_raw_encoder()
    plink_utils.plink_raw_to_one_hot(test_raw_path, test_output_folder, encoder)

    gotten = np.load(test_output_folder / test_input_file)
    expected = np.array([OHMP[integer] for integer in ohmp_keys]).T
    assert (gotten == expected).all()
