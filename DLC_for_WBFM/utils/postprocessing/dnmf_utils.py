from dNMF.Demix.dNMF import dNMF
import torch
import h5py
import pandas as pd
from DLC_for_WBFM.utils.postprocessing.postprocessing_utils import *
from DLC_for_WBFM.utils.postprocessing.base_cropping_utils import *
# from DLC_for_WBFM.config.class_configuration import *
from DLC_for_WBFM.utils.postprocessing.base_DLC_utils import xy_from_dlc_dat

##
## Full workflow
##

def _extract_all_traces(config_file,
                        which_neurons=None,
                        num_frames=None,
                        crop_sz=None,
                        is_3d=False,
                        params=None,
                        trace_fname='test.pickle'):
    """
    Extracts all traces using a project-level config file

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
        Dictionary of parameters for dnmf function
    trace_fname : str
        Name to save the final traces as

    Returns
    -------
    all_traces : list
        List of neurons, with two arrays: 'gcamp' (signal) and 'mcherry' (tracking)

    See also: extract_all_traces
    """

    c = load_config(config_file)

    # Get needed fields
    annotation_fname = c.tracking.annotation_fname
    video_fname_red = c.datafiles.red_bigtiff_fname
    video_fname_green = c.datafiles.green_bigtiff_fname

    z_params = (c.preprocessing.center_slice, c.preprocessing.num_total_slices,
                c.preprocessing.alpha, c.preprocessing.start_volume)

    if num_frames is None:
        num_frames = c.preprocessing.num_frames

    # Save configuration
    # WARNING: overwrite old
    traces_config = DLCForWBFMTraces(is_3d,
                                        crop_sz,
                                        trace_fname,
                                        which_neurons)
    c.traces = traces_config
    save_config(config_file)

    # Actually get traces
    all_traces = extract_all_traces(annotation_fname,
                           video_fname_red,
                           video_fname_green,
                           which_neurons,
                           num_frames,
                           crop_sz,
                           params,
                           is_3d,
                           z_params)

    # Save traces
    pickle.dump(all_traces, open(trace_fname, 'wb'))

    return all_traces


def get_defaults_from_dlc(annotation_fname, num_frames, which_neurons):
    # Some files can only be read a different way... not sure what's up

    try:
        with h5py.File(annotation_fname, 'r') as dlc_dat:
            dlc_table = dlc_dat['df_with_missing']['table']
            # Each table entry has: x, y, probability
            if which_neurons is None:
                num_neurons = len(dlc_table[0][1])//3
                which_neurons = range(num_neurons)
            if num_frames is None:
                num_frames = len(dlc_table)
    except:
        dlc_table = pd.read_hdf(annotation_fname)
        if which_neurons is None:
            num_neurons = len(dlc_table.columns)//3
            which_neurons = range(num_neurons)
        if num_frames is None:
            num_frames = len(dlc_table)
    print(f'Found annotations for {num_neurons} neurons and {num_frames} frames')

    return which_neurons, num_neurons, num_frames


def extract_all_traces(annotation_fname,
                       video_fname_mcherry,
                       video_fname_gcamp,
                       which_neurons=None,
                       num_frames=None,
                       crop_sz=(19,19),
                       params=None,
                       is_3d=False,
                       z_params=None):
    """
    Extracts a trace from a single neuron in 2d using dNMF from one movie

    Input
    ----------
    annotation_fname : str
        .h5 produced by DeepLabCut with annotations

    video_fname_mcherry : str
        .avi file with comparison channel.
        As of 16.10.2020 this is 'mcherry'

    video_fname_gcamp : str
        .avi file with actual neuron activities
        As of 16.10.2020 this is 'gcamp'

    which_neuron : [int,..]
        Indices of the neurons, as determined by the original annotation
        By default, extracts all tracked neurons

    num_frames : int
        How many frames to extract

    crop_sz : (int, int)
        Number of pixels to use for traces determination.
        A Gaussian is fit within this size, so it should contain the entire neuron

    params : dict
        Parameters for final trace extraction, using a Gaussian.
        See 'dNMF' docs for explanation of parameters

    Output
    ----------
    all_traces : [dict,...]
        Array of dicts, where the keys are 'mcherry' and 'gcamp'
        Each final element is a 1d array
    """

    # Get the number of neurons
    if which_neurons is None or num_frames is None:
        out = get_defaults_from_dlc(annotation_fname, num_frames, which_neurons)
        which_neurons, num_neurons, num_frames = out

    # Initialize
    all_traces = []
    start = time.time()

    # Loop through and get traces of gcamp and mcherry
    for which_neuron in which_neurons:
        print(f'Starting analysis of neuron {which_neuron}/{len(which_neurons)}...')
        mcherry_dat = extract_single_trace(annotation_fname,
                                 video_fname_mcherry,
                                 which_neuron=which_neuron,
                                 num_frames=num_frames,
                                 crop_sz=crop_sz,
                                 params=params,
                                 is_3d=is_3d,
                                 z_params=z_params)
        print('Finished extracting mCherry')
        gcamp_dat = extract_single_trace(annotation_fname,
                                  video_fname_gcamp,
                                  which_neuron=which_neuron,
                                  num_frames=num_frames,
                                  crop_sz=crop_sz,
                                  params=params,
                                  is_3d=is_3d,
                                  flip_x=True, # OPTIMIZE: needed as of 09.11.2020
                                  z_params=z_params)
        print('Finished extracting GCaMP')
        all_traces.append({'mcherry':mcherry_dat,
                           'gcamp':gcamp_dat})
    end = time.time()
    print('Finished in ' + str(end-start) + ' seconds')

    return all_traces


def extract_single_trace(annotation_fname,
                         video_fname,
                         which_neuron=0,
                         num_frames=500,
                         crop_sz=(19,19),
                         params=None,
                         is_3d=False,
                         flip_x=False,
                         z_params=None):
    """
    Extracts a trace from a single neuron in 2d using dNMF from one movie

    Input
    ----------
    annotation_fname : str
        .h5 produced by DeepLabCut with annotations

    video_fname : str
        .avi file with neuron activities.
        Intended use is red or green channel

    which_neuron : int
        Index of the neuron, as determined by the original annotation

    num_frames : int
        How many frames to extract

    crop_sz : (int, int)
        Number of pixels to use for traces determination.
        A Gaussian is fit within this size, so it should contain the entire neuron

    params : dict
        Parameters for final trace extraction, using a Gaussian.
        See 'dNMF' docs for explanation of parameters

    flip_x : bool
        To flip the video in x

    is_3d : bool
        2d or 3d trace extraction

    Output
    ----------
    trace : np.array()
        1d array of trace activity
    """

    # Get the positions, and crop the full video
    this_xy, this_prob = xy_from_dlc_dat(annotation_fname,
                                         which_neuron=which_neuron,
                                         num_frames=num_frames)
    if not is_3d:
        cropped_dat = get_crop_from_avi(video_fname,
                                        this_xy,
                                        num_frames,
                                        sz=crop_sz)
    else:
        assert len(crop_sz)==3, "Crop must be 3d"
        which_z, num_slices, alpha, start_volume = z_params
        cropped_dat = get_crop_from_ometiff_virtual(video_fname,
                                                    this_xy,
                                                    this_prob,
                                                    which_z,
                                                    num_frames,
                                                    crop_sz=crop_sz,
                                                    num_slices=num_slices,
                                                    alpha=alpha,
                                                    flip_x=flip_x,
                                                    start_volume=start_volume)
        cropped_dat = np.transpose(cropped_dat, axes=(2,3,1,0))


    # Get parameters and run dNMF
    dnmf_obj = dNMF_default_from_DLC(cropped_dat, crop_sz, params, is_3d)
    dnmf_obj.optimize(lr=1e-4,n_iter=20,n_iter_c=2)

    return dnmf_obj.C[0,:]




def dNMF_default_from_DLC(dat, crop_sz, params=None, is_3d=False):
    """
    Prepares the parameters and data files for consumption by dNMF
    """
    # Defaults that work decently
    if params is None:
        params = {'n_trials':5, 'noise_level':1e-2, 'sigma_inv':.2,
                  'radius':10, 'step_S':.1, 'gamma':0, 'stride_factor':2, 'density':.1, 'varfact':5,
                  'traj_means':[.0,.0,.0], 'traj_variances':[2e-4,2e-4,1e-5], 'sz':[20,20,1],
                  'K':20, 'T':100, 'roi_window':[4,4,0]}


    # Convert the data
    dat_torch = torch.tensor(dat).float()

    # Finalize the parameters
    if not is_3d:
        # Build position and convert to pytorch
        positions =[list(crop_sz + (0,)),[1, 1, 0]] # Add a dummy position
        positions = np.expand_dims(positions,2)/2.0 # Return the center of the crop
        positions =  torch.tensor(positions).float()
        params = {'positions':positions[:,:,0][:,:,np.newaxis],\
                  'radius':params['radius'],'step_S':params['step_S'],'gamma':params['gamma'],\
                  'use_gpu':False,'initial_p':positions[:,:,0],'sigma_inv':params['sigma_inv'],\
                  'method':'1->t', 'verbose':False}
    else:
        positions =[list(crop_sz),[1, 1, 0]] # Add a dummy position
        positions = np.expand_dims(positions,2)/2.0 # Return the center of the crop
        positions =  torch.tensor(positions).float()
        params = {'positions':positions,\
                  'radius':params['radius'],'step_S':params['step_S'],'gamma':params['gamma'],\
                  'use_gpu':False,'initial_p':positions[:,:,0],'sigma_inv':params['sigma_inv'],\
                  'method':'1->t', 'verbose':False}

    # Finally, create the analysis object
    dnmf_obj = dNMF(dat_torch, params=params)

    return dnmf_obj
