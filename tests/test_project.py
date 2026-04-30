import os
import importlib.util


def test_project_files_exist():
    assert os.path.exists("dataset.py")
    assert os.path.exists("training.py")
    assert os.path.exists("plottings.py")
    assert os.path.exists("Makefile")


def test_required_packages_installed():
    assert importlib.util.find_spec("pandas") is not None
    assert importlib.util.find_spec("numpy") is not None
    assert importlib.util.find_spec("sklearn") is not None
    assert importlib.util.find_spec("matplotlib") is not None


def test_readme_exists():
    assert os.path.exists("README.md")
