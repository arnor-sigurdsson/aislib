import pytest

from aislib import misc_utils


def test_ensure_path_exists(tmp_path):
    file_path = tmp_path / 'folder1/subfolder1/file.txt'
    misc_utils.ensure_path_exists(file_path)

    assert file_path.parent.exists()
    assert len([i for i in file_path.parent.iterdir()]) == 0

    folder_path = tmp_path / 'folder2/subfolder1/'

    misc_utils.ensure_path_exists(folder_path, is_folder=True)
    assert folder_path.exists()
    assert len([i for i in folder_path.iterdir()]) == 0

    extra_path = folder_path / 'subsubfolder1/file.txt'
    misc_utils.ensure_path_exists(extra_path)
    assert extra_path.parent.exists()
    assert len([i for i in extra_path.parent.iterdir()]) == 0
