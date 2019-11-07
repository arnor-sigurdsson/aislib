from aislib import misc_utils
import logging


def test_get_logger():
    logger_basic = misc_utils.get_logger("logger_basic")
    assert logger_basic.level == 10
    assert len(logger_basic.handlers) == 1
    assert isinstance(logger_basic.handlers[0], logging.StreamHandler)

    logger_tqdm = misc_utils.get_logger("logger_tqdm", tqdm_compatible=True)
    assert logger_tqdm.level == 10
    assert len(logger_tqdm.handlers) == 1
    assert isinstance(logger_tqdm.handlers[0], misc_utils.TQDMLoggingHandler)


def test_ensure_path_exists(tmp_path):
    file_path = tmp_path / "folder1/subfolder1/file.txt"
    misc_utils.ensure_path_exists(file_path)

    assert file_path.parent.exists()
    assert len([i for i in file_path.parent.iterdir()]) == 0

    folder_path = tmp_path / "folder2/subfolder1/"

    misc_utils.ensure_path_exists(folder_path, is_folder=True)
    assert folder_path.exists()
    assert len([i for i in folder_path.iterdir()]) == 0

    extra_path = folder_path / "subsubfolder1/file.txt"
    misc_utils.ensure_path_exists(extra_path)
    assert extra_path.parent.exists()
    assert len([i for i in extra_path.parent.iterdir()]) == 0
