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


def test_custom_package_imports():
    # Import something from each file
    import DLC_for_WBFM.utils.external.centerline_utils
    import DLC_for_WBFM.utils.external.utils_cv2
    import DLC_for_WBFM.utils.external.utils_logging
    import DLC_for_WBFM.utils.external.utils_networkx
    import DLC_for_WBFM.utils.external.utils_pandas
    import DLC_for_WBFM.utils.external.utils_zarr

    from DLC_for_WBFM.utils.general.custom_errors import AnalysisOutOfOrderError
    from DLC_for_WBFM.utils.general.distance_functions import calc_global_track_to_tracklet_distances
    from DLC_for_WBFM.utils.neuron_matching.class_frame_pair import FramePairOptions
    from DLC_for_WBFM.utils.neuron_matching.class_reference_frame import ReferenceFrame
    from DLC_for_WBFM.utils.neuron_matching.feature_pipeline import build_tracklets_full_video
    from DLC_for_WBFM.utils.neuron_matching.long_range_matching import long_range_matches_from_config
    from DLC_for_WBFM.utils.neuron_matching.matches_class import MatchesWithConfidence
    from DLC_for_WBFM.utils.neuron_matching.utils_candidate_matches import calc_all_bipartite_matches
    from DLC_for_WBFM.utils.neuron_matching.utils_features import convert_to_grayscale
    from DLC_for_WBFM.utils.neuron_matching.utils_gaussian_process import calc_matches_using_gaussian_process
    from DLC_for_WBFM.utils.neuron_matching.utils_keypoint_matching import calc_all_tracklet_features
    from DLC_for_WBFM.utils.neuron_matching.utils_reference_frames import add_all_good_components
    from DLC_for_WBFM.utils.neuron_matching.utils_rigid_alignment import filter_image

    from DLC_for_WBFM.utils.nn_utils.superglue import KeypointEncoder
    from DLC_for_WBFM.utils.nn_utils.worm_with_classifier import WormWithNeuronClassifier

    from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData
    from DLC_for_WBFM.utils.projects.physical_units import PhysicalUnitConversion
    from DLC_for_WBFM.utils.projects.plotting_classes import TrackletAndSegmentationAnnotator
    from DLC_for_WBFM.utils.projects.project_config_classes import ConfigFileWithProjectContext
    from DLC_for_WBFM.utils.projects.utils_filenames import resolve_mounted_path_in_current_os
    from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name_neuron
    from DLC_for_WBFM.pipeline.project_initialization import build_project_structure_from_config
    from DLC_for_WBFM.utils.projects.utils_project_status import check_all_needed_data_for_step

    from DLC_for_WBFM.pipeline.traces import extract_traces_using_config

    from DLC_for_WBFM.utils.tracklets.high_performance_pandas import PaddedDataFrame
    from DLC_for_WBFM.utils.tracklets.tracklet_class import NeuronComposedOfTracklets
    from DLC_for_WBFM.utils.tracklets.tracklet_pipeline import match_all_adjacent_frames_using_config
    from DLC_for_WBFM.utils.tracklets.tracklet_to_DLC import best_tracklet_covering_from_my_matches
    from DLC_for_WBFM.utils.tracklets.utils_splitting import TrackletSplitter
    from DLC_for_WBFM.utils.tracklets.utils_tracklets import create_new_track

    from DLC_for_WBFM.utils.visualization.napari_from_config import napari_of_full_data
    from DLC_for_WBFM.utils.visualization.napari_utils import napari_labels_from_traces_dataframe
    from DLC_for_WBFM.utils.visualization.plot_traces import make_grid_plot_using_project
    from DLC_for_WBFM.pipeline.traces import reindex_segmentation_using_config


@pytest.mark.skipif(os.environ.get('CONDA_DEFAULT_ENV', '') != "segmentation", reason="incorrect conda env")
def test_segmentation_package_imports():

    print("import segmentation")
    import segmentation

    print("Successfully imported everything! Your segmentation environment is properly setup")


@pytest.mark.skipif(os.environ.get('CONDA_DEFAULT_ENV', '') != "wbfm", reason="incorrect conda env")
def test_wbfm_package_imports():
    print("import pytorch")
    import torch

    print("Successfully imported everything! Your environment is properly setup")


@pytest.mark.skipif(os.environ.get('CONDA_DEFAULT_ENV', '') != "torch", reason="incorrect conda env")
def test_wbfm_package_imports():
    print("import torch")
    import torch

    print("Successfully imported everything! Your environment is properly setup")
