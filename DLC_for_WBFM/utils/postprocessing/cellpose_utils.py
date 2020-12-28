## Utilities for trace extraction via cellpose-based segmentation


# Use the project config file
from DLC_for_WBFM.bin.configuration_definition import *
from DLC_for_WBFM.utils.postprocessing.postprocessing_utils import _get_crop_from_ometiff_virtual
from DLC_for_WBFM.utils.postprocessing.postprocessing_utils import *
from cellpose import models
from cellpose import utils as cellutils
from scipy.ndimage import center_of_mass


##
## Functions to extract traces
##

def extract_all_traces_cp(config_file,
                          which_neurons=None,
                          num_frames=None,
                          crop_sz=None,
                          is_3d=None,
                          params=None,
                          trace_fname='test_cellpose.pickle',
                          overwrite_trace_settings=True):
    """
    Extracts all traces using cellpose

    Parameters
    ----------
    config_file : str
        Path to the config file
    which_neurons : None or list
        The indices for the used neurons; 'None' = all neurons
    num_frames : int
        Integer for the number of frames
    crop_sz : tuple
        3 or 2 length tuple to describe the final cropped frame or cube
    is_3d : bool
        Whether the input data is 3d or 2d
    params : dict
        Dictionary of parameters for cellpose function... TODO
    trace_fname : str
        Name to save the final traces as
    """

    c = load_config(config_file)

    # Get needed fields
    if num_frames is None:
        num_frames = c.preprocessing.num_frames
    if which_neurons is None:
        num_neurons, which_neurons, tmp = get_number_of_annotations(c.tracking.annotation_fname)
    # Assume if these aren't set that there is already a traces subobject
    if not overwrite_trace_settings:
        trace_fname = c.traces.traces_fname
    else:
        # Traces will be saved in overall config file folder
        # TODO
        trace_fname = os.path.join(c.get_dirname(), trace_fname)

        # Save configuration
        traces_config = DLCForWBFMTraces(is_3d,
                                         crop_sz,
                                         trace_fname,
                                         which_neurons)
        c.traces = traces_config
        save_config(c)

    # Actually calculate
    # TODO: Cellpose options
    start = time.time()
    all_traces = []
    for neuron in which_neurons:
        all_traces.append(extract_single_trace_cp(c,
                                                  which_neuron=neuron,
                                                  num_frames=num_frames))

    end = time.time()
    if c.verbose >= 1:
        print('Finished in ' + str(end-start) + ' seconds')

    # Save traces
    pickle.dump(all_traces, open(trace_fname, 'wb'))

    return all_traces


def extract_single_trace_cp(config_filename,
                            which_neuron=None,
                            num_frames=100,
                            cellpose_opt={'diameter':8}):
    """
    Extracts single trace of a single neuron via cellpose segmentation

    Returns:

    all_traces : [dict,...]
        Array of dicts, where the keys are 'red', 'green', and 'num_pixels'
        Each final element is a 1d array
    """

    c = load_config(config_filename)

    # Two channels
    cropped_dat_red = _get_crop_from_ometiff_virtual(c,
                                                     which_neuron=which_neuron,
                                                     num_frames=num_frames,
                                                     use_red_channel=True)

    cropped_dat_green = _get_crop_from_ometiff_virtual(c,
                                                       which_neuron=which_neuron,
                                                       num_frames=num_frames,
                                                       use_red_channel=False)
    # Do the segmentation on red only
    channels = [0,0]
    model = models.Cellpose(gpu=False, model_type='nuclei')
    # Get time series of segmentations
    all_masks = []

    for i in range(num_frames):
        this_vol = np.squeeze(cropped_dat_red[i,...])
        m, f, s, d = model.eval(this_vol,
                                channels=channels,
                                do_3D=True,
                                **cellpose_opt)

        if c.verbose >= 2:
            print(f"Segmenting Volume: {i}/{num_frames}")
        all_masks.append(m)

    # TODO: better saving
    # fname = f'test_masks_neuron_{which_neuron}'
    # pickle.dump(all_masks, open(fname, 'wb'))

    # Link segmentations in time
    # For now, discard all but the center neuron
    initial_neuron, _ = calc_center_neuron(all_masks[0])
    _, _, this_neuron_masks = calc_all_overlaps(initial_neuron,all_masks)

    # Finally, get traces
    trace_red = np.zeros(num_frames)
    trace_green = np.zeros(num_frames)
    num_pixels = np.zeros(num_frames)

    for i, m in enumerate(this_neuron_masks):
        this_vol_red = np.squeeze(cropped_dat_red[i,...])
        this_vol_green = np.squeeze(cropped_dat_green[i,...])
        # TODO: Assume there is only one neuron detected
        trace_red[i] = brightness_from_roi(this_vol_red, m, 1)
        trace_green[i] = brightness_from_roi(this_vol_green, m, 1)
        num_pixels[i] = np.count_nonzero(m)

    final_trace = {'red': trace_red,
                   'green': trace_green,
                   'num_pixels': num_pixels}

    return final_trace


def brightness_from_roi(img, all_masks, which_neuron):
    """
    Just averages the pixels given a 3d mask

    TODO: better determination of brightness
    """
    mask = all_masks==which_neuron
    return np.mean(img[mask])


##
## Functions for linking across time
##


def calc_best_overlap(mask_v0, # Only one mask
                      masks_v1,
                      verbose=1):
    """
    Calculates the best overlap between an initial mask and all subsequent masks
        Note: calculates pairwise for adjacent time points

    Parameters
    ----------
    mask_v0 : array_like
        Mask of the original neuron
    masks_v1 : list
        Masks of all neurons detected in the next frame

    """
    best_overlap = 0
    best_ind = None
    best_mask = np.zeros_like(masks_v1)
    all_vals = np.unique(masks_v1)
    for i,val in enumerate(all_vals):
        if val == 0:
            continue
        this_neuron_mask = masks_v1==val
        overlap = np.count_nonzero(mask_v0*this_neuron_mask)
        if overlap > best_overlap:
            best_overlap = overlap
            best_ind = i
            best_mask = this_neuron_mask
    if verbose >= 2:
        print(f'Best Neuron: {best_ind}, overlap between {best_overlap} and original')
    return best_ind, best_overlap, best_mask


def calc_all_overlaps(start_neuron,
                      all_multi_masks,
                      verbose=1):
    """
    Get the "tube" of a neuron through time via most overlapping pixels

    Parameters
    ----------
    start_neuron : int
        Which neuron to take as the initial one
    all_multi_masks : list
        List of all masks across time

    See also: calc_best_overlap
    """
    num_frames = len(all_multi_masks)
    all_masks = []
    all_neurons = np.zeros(num_frames)
    all_overlaps = np.zeros(num_frames)

    # Initial neuron
    all_masks.append(all_multi_masks[0] == start_neuron)
    all_neurons[0] = start_neuron

    has_track = True
    for i, masks_v1 in enumerate(all_multi_masks):
        if i==0:
            continue
        if has_track:
            # Retain the last successful mask
            prev_mask = all_multi_masks[i-1] == all_neurons[i-1]

        all_neurons[i], all_overlaps[i], this_mask = calc_best_overlap(prev_mask, masks_v1)
        if all_overlaps[i]==0:
            if c.verbose >= 1:
                print("Lost neuron tracking, attempting to find...")
            all_neurons[i], this_mask, has_track = attempt_to_refind_neuron(prev_mask, masks_v1)
        else:
            has_track = True
        all_masks.append(this_mask)

    return all_neurons, all_overlaps, all_masks


def calc_center_neuron(initial_mask,
                       verbose=1):
    """
    Calculates neuron that is closest to the center

    Uses: scipy.ndimage.center_of_mass
    """
    center_point = np.array(initial_mask.shape) / 2.0
    all_vals = np.unique(initial_mask)
    closest_neuron = 0
    best_dist = np.inf
    for i in all_vals:
        if i==0:
            continue
        this_center = center_of_mass(initial_mask==i)
        dist = np.linalg.norm(this_center-center_point)
        if dist < best_dist:
            closest_neuron = i
            best_dist = dist
    if verbose >= 1:
        print(f'Found closest neuron to be {closest_neuron}')

    return closest_neuron, best_dist


def attempt_to_refind_neuron(prev_mask, masks_v1, verbose=1):
    """
    Try to refind the neuron when there is no overlap
    """

    # First: are there any objects detected?
    # If so, Get the object closest to center
    closest_neuron, best_dist = calc_center_neuron(masks_v1)
    if closest_neuron==0:
        if verbose >= 1:
            print("No neurons detected, hopefully the tracking will succeed later")
        return 0, np.zeros_like(masks_v1), False
    else:
        this_mask = masks_v1==closest_neuron

    # TODO: Confirm if it is a similar size
    sz0 = np.count_nonzero(prev_mask)
    sz1 = np.count_nonzero(this_mask)
    if (sz0 < 2*sz1) and (sz0 > sz1/2):
        if c.verbose >= 1:
            print("Re-found neuron!")
        return closest_neuron, this_mask, True
    else:
        if c.verbose >= 1:
            print(f"New Object size ({sz1}) was too different ({sz0}); rejecting")
        # print("Hopefully the tracking will succeed later")
        return 0, np.zeros_like(masks_v1), False
