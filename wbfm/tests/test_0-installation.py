import os
import pytest


def test_core_package_imports():
    print("import wbfm")

    print("import scipy")

    print("import numpy")

    print("import skimage")

    print("import sacred")

    print("import tifffile")

    print("import napari")

    print("import zarr")

    print("Successfully imported everything! Your core environment is properly setup")


def test_custom_package_imports():
    # Import something from each file

    pass


@pytest.mark.skipif(os.environ.get('CONDA_DEFAULT_ENV', '') != "segmentation", reason="incorrect conda env")
def test_segmentation_package_imports():

    print("import segmentation")

    print("Successfully imported everything! Your segmentation environment is properly setup")


@pytest.mark.skipif(os.environ.get('CONDA_DEFAULT_ENV', '') != "wbfm", reason="incorrect conda env")
def test_wbfm_package_imports():
    print("import pytorch")

    print("Successfully imported everything! Your environment is properly setup")


@pytest.mark.skipif(os.environ.get('CONDA_DEFAULT_ENV', '') != "torch", reason="incorrect conda env")
def test_wbfm_package_imports():
    print("import torch")

    print("Successfully imported everything! Your environment is properly setup")
