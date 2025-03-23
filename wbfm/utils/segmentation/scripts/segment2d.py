"""
The top level functions for segmenting a full (WBFM) recording.
"""

from wbfm.utils.segmentation.util.utils_pipeline import segment_video_using_config_2d
import os
# Experiment tracking
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver

# Initialize sacred experiment
ex = Experiment()
ex.add_config(r'config\segment_config.yaml')
ex.observers.append(FileStorageObserver('runs')) # TODO: smarter location


@ex.config
def cfg(video_path):
    # Check paths
    if video_path is None:
        print("Must path a valid video path!")
    assert os.path.exists(video_path)


@ex.automain
def segment2d(_config, _run):
    """
    Function to segment, stitch and curate a recording on a volume-to-volume basis.

    Parameters
    ----------
    video_path : str
        path to recording
    start_volume : int
        starting frame in absolute number starting from 0
    num_frames : int
        number of volumes to be analyzed
    stardist_model_name : str
        Name/alias of the StarDist model to be used for segmentation. I.e. 'lukas', 'charlie', 'versatile', 'Charlie100-3d'
        see also: get_stardist_model()
    preprocessing : yaml file for preprocessing settings
        see also PreprocessingSettings
    num_slices : int
        number of total Z-slices per volume
    to_remove_border : boolean
        Flag for zeroing edge areas, because of edge effect artefacts in segmentations of prealigned volumes
    flag_3d : Boolean
        Flag for using Stardists' 3D segmentation
    options : object
        contains further necessary options for running
    verbose : int
        flag for print statements. Increasing by 1, increase depth by 1
    DEBUG : bool
        flag for a 10 second run; segments one volume and writes small versions of all files

    Returns
    -------
    output_fname : str
        path to the TIF-file of segmented masks
    metadata : dict
        dictionary containing metadata per volume and neuron. Each volume's dataframe has N rows, where
        N = number of unique neurons in that slice and 3 columns = ('total_brightness', 'neuron_volume', 'centroids')
        {volume_number : [dataframe('total_brightness', 'neuron_volume', 'centroids')]}

        Use the column name to index directly into the dataframe to get all e.g. centroid values
    """
    sacred.commands.print_config(_run)

    segment_video_using_config_2d(_config)
