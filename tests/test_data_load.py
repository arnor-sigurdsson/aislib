import numpy as np
import pytest

from aislib import data_load


@pytest.fixture()
def create_test_arr_files(request, tmp_path):
    test_arr_base = np.zeros((4, 100), dtype=np.uint8)
    test_arr_base[0, :] = 1

    if hasattr(request, "param"):
        hook = request.param
    else:

        def hook(x):
            return x

    for i in range(10):
        np.save(tmp_path / f"{i}_-_label.npy", hook(test_arr_base))

    return tmp_path, test_arr_base


def test_iter_loadtxt(tmp_path):
    test_arr = np.zeros((10, 10))
    for i in range(test_arr.shape[0]):
        test_arr[i, :] = i

    arr_path = tmp_path / "arr.npy"
    np.savetxt(arr_path, test_arr, fmt="%d", delimiter=",")
    loader = data_load.iter_loadtxt

    loaded_arr = loader(arr_path, delimiter=",")
    np.testing.assert_array_equal(test_arr, loaded_arr)

    loaded_arr_skipped = loader(arr_path, delimiter=",", skiprows=5, dtype=float)

    assert loaded_arr_skipped.dtype == float
    np.testing.assert_array_equal(test_arr[5:], loaded_arr_skipped)

    loaded_arr_custom = loader(arr_path, custom_splitter=lambda x: x.strip().split(","))
    np.testing.assert_array_equal(test_arr, loaded_arr_custom)


@pytest.mark.parametrize("create_test_arr_files", (np.packbits,), indirect=True)
def test_load_np_packbits_from_folder(create_test_arr_files):
    arr_path, arr_base = create_test_arr_files

    loader = data_load.load_np_packbits_from_folder
    loaded_arrs, loaded_ids = loader(arr_path, 4, dtype=np.uint8)

    for arr in loaded_arrs:
        np.testing.assert_array_equal(arr, arr_base)

    assert sorted(loaded_ids) == [str(i) + "_-_label" for i in range(10)]


def test_load_np_arrays_from_folder(create_test_arr_files):
    arr_path, arr_base = create_test_arr_files

    loader = data_load.load_np_arrays_from_folder
    loaded_arrs = loader(arr_path, dtype=np.uint8)

    for arr in loaded_arrs:
        np.testing.assert_array_equal(arr, arr_base)


def test_get_labels_from_folder(create_test_arr_files):
    arr_path, _ = create_test_arr_files

    labels = data_load.get_labels_from_folder(arr_path, "_-_")
    assert set(labels) == {"label"}


def test_get_labels_from_iterable():
    test_label_list_1 = [f"{i}_-_label" for i in range(10)]

    labels_1 = data_load.get_labels_from_iterable(test_label_list_1, "_-_")

    assert set(labels_1) == {"label"}

    test_label_list_2 = [f"{i}_-_some_info_-_label" for i in range(10)]

    labels_2 = data_load.get_labels_from_iterable(test_label_list_2, "_-_", 2)

    assert set(labels_2) == {"label"}
