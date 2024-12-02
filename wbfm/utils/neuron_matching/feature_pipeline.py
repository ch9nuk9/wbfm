import concurrent.futures
from typing import Tuple, Dict
import numpy as np
from segmentation.util.utils_metadata import DetectedNeurons
from tqdm.auto import tqdm

from wbfm.utils.external.custom_errors import NoMatchesError, NoNeuronsError
from wbfm.utils.general.preprocessing.utils_preprocessing import PreprocessingSettings
from wbfm.utils.neuron_matching.class_frame_pair import FramePair, calc_FramePair_from_Frames, \
    FramePairOptions
from wbfm.utils.neuron_matching.class_reference_frame import ReferenceFrame, \
    build_reference_frame_encoding
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import ModularProjectConfig


##
## Full traces function
##


def build_tracklets_full_video(video_data, video_fname: str, start_volume: int = 0, num_frames: int = 10,
                               z_depth_neuron_encoding: float = 5.0,
                               frame_pair_options: FramePairOptions = None,
                               external_detections: str = None,
                               project_config: ModularProjectConfig = None,
                               use_superglue: bool = True,
                               verbose: int = 0) -> Tuple[Dict[Tuple[int, int], FramePair], Dict[int, ReferenceFrame]]:
    """
    Detects and tracks neurons using opencv-based feature matching
    Note: only compares adjacent frames
        Thus, if a neuron is lost in a single frame, the track ends

    New: uses and returns my class of features
    """

    # Build frames, then match them
    preprocessing_settings = project_config.get_preprocessing_class()
    end_volume = start_volume + num_frames
    frame_range = list(range(start_volume, end_volume))
    all_frame_dict = calculate_frame_objects_full_video(video_data, external_detections, frame_range,
                                                        video_fname, z_depth_neuron_encoding,
                                                        preprocessing_settings=preprocessing_settings)

    try:
        if use_superglue:
            from wbfm.utils.tracklets.tracklet_pipeline import build_frame_pairs_using_superglue
            project_data = ProjectData.load_final_project_data_from_config(project_config)
            project_data.raw_frames = all_frame_dict
            all_frame_pairs = build_frame_pairs_using_superglue(all_frame_dict, frame_pair_options, project_data)
        else:
            all_frame_pairs = match_all_adjacent_frames(all_frame_dict, end_volume, frame_pair_options, start_volume)
        return all_frame_pairs, all_frame_dict
    except (ValueError, NoNeuronsError, NoMatchesError) as e:
        project_config.logger.warning("Error in frame pair matching; quitting gracefully and saving the frame pairs:")
        print(e)
        return None, all_frame_dict


def match_all_adjacent_frames(all_frame_dict, end_volume, frame_pair_options, start_volume):
    all_frame_pairs = {}
    frame_range = range(start_volume + 1, end_volume)
    for i_frame in tqdm(frame_range):
        key = (i_frame - 1, i_frame)
        frame0, frame1 = all_frame_dict[key[0]], all_frame_dict[key[1]]
        this_pair = calc_FramePair_from_Frames(frame0, frame1, frame_pair_options=frame_pair_options)

        all_frame_pairs[key] = this_pair
    return all_frame_pairs


def calculate_frame_objects_full_video(video_data, external_detections, frame_range, video_fname,
                                       z_depth_neuron_encoding, encoder_opt=None, max_workers=8,
                                       preprocessing_settings=None,
                                       logger=None, **kwargs):
    # Get initial volume; settings are same for all
    vol_shape = video_data[0, ...].shape
    all_detected_neurons = DetectedNeurons(external_detections)
    all_detected_neurons.setup()

    def _build_frame(frame_ind: int) -> ReferenceFrame:
        metadata = {'frame_ind': frame_ind,
                    'vol_shape': vol_shape,
                    'video_fname': video_fname,
                    'z_depth': z_depth_neuron_encoding,
                    'alpha_red': preprocessing_settings.alpha_red,
                    '_raw_data': np.array(video_data[frame_ind, ...])}
        f = build_reference_frame_encoding(metadata=metadata, all_detected_neurons=all_detected_neurons,
                                           encoder_opt=encoder_opt)
        return f

    # Build all frames initially, then match
    all_frame_dict = dict()
    # logger.info(f"Calculating Frame objects for frames: {frame_range[0]} to {frame_range[-1]}")
    with tqdm(total=len(frame_range)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_build_frame, i): i for i in frame_range}
            for future in concurrent.futures.as_completed(futures):
                i_frame = futures[future]
                all_frame_dict[i_frame] = future.result()
                pbar.update(1)
    return all_frame_dict
