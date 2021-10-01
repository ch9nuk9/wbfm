import os

import pytest


def test_core_package_imports():
    print("import DLC_for_WBFM")
    import DLC_for_WBFM

    print("import scipy")
    import scipy

    print("import numpy")
    import numpy

    print("import skimage")
    import skimage

    print("import sacred")
    import sacred

    print("import tifffile")
    import tifffile

    print("import napari")
    import napari

    print("import zarr")
    import zarr

    print("Successfully imported everything! Your core environment is properly setup")


@pytest.mark.skipif(os.environ.get('CONDA_DEFAULT_ENV', '') != "segmentation", reason="incorrect conda env")
def test_segmentation_package_imports():

    print("import segmentation")
    import segmentation

    print("Successfully imported everything! Your segmentation environment is properly setup")


@pytest.mark.skipif(os.environ.get('CONDA_DEFAULT_ENV', '') != "wbfm", reason="incorrect conda env")
def test_wbfm_package_imports():
    print("import deeplabcut")
    import deeplabcut

    print("import tensorflow")
    import tensorflow

    print("Successfully imported everything! Your environment is properly setup")