from DLC_for_WBFM.bin.configuration_definition import load_config
from DLC_for_WBFM.utils.postprocessing.base_DLC_utils import xy_from_dlc_dat
from DLC_for_WBFM.utils.postprocessing.base_cropping_utils import get_crop_from_avi
from DLC_for_WBFM.utils.postprocessing.postprocessing_utils import get_crop_from_ometiff_virtual
# from DLC_for_WBFM.utils.postprocessing.postprocessing_utils import *


def _get_crop_from_avi(config_file,
                       which_neuron,
                       num_frames,
                       use_red_channel=True):

    c = load_config(config_file)

    # Get track
    this_xy, this_prob = xy_from_dlc_dat(c.tracking.annotation_fname,
                                        which_neuron=which_neuron,
                                        num_frames=num_frames)
    # Get data
    if use_red_channel:
        fname = c.datafiles.red_avi_fname
        flip_x = False
    else:
        fname = c.datafiles.green_avi_fname
        flip_x = c.preprocessing.red_and_green_mirrored

    cropped_dat = get_crop_from_avi(fname, this_xy, num_frames, c.traces.crop_sz)

    return cropped_dat




def _get_crop_from_ometiff_virtual(config_file,
                                   which_neuron,
                                   num_frames,
                                   use_red_channel=True):
    """
    See also: get_crop_from_ometiff_virtual

    By default flips the green channel
    """
    c = load_config(config_file)

    # Get track
    this_xy, this_prob = xy_from_dlc_dat(c.tracking.annotation_fname,
                                         which_neuron=which_neuron,
                                         num_frames=num_frames)
    # Get data
    if use_red_channel:
        fname = c.datafiles.red_bigtiff_fname
        flip_x = False
    else:
        fname = c.datafiles.green_bigtiff_fname
        flip_x = c.preprocessing.red_and_green_mirrored
    cropped_dat = get_crop_from_ometiff_virtual(fname,
                                                this_xy,
                                                which_z=c.preprocessing.center_slice,
                                                num_frames=num_frames,
                                                crop_sz=c.traces.crop_sz,
                                                num_slices=c.preprocessing.num_total_slices,
                                                actually_create=True,
                                                alpha=c.preprocessing.alpha,
                                                start_volume=c.preprocessing.start_volume,
                                                actually_crop=True,
                                                flip_x=flip_x,
                                                verbose=c.verbose)
    return cropped_dat
