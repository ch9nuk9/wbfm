import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.mixture import GaussianMixture

from wbfm.utils.projects.utils_filenames import pickle_load_binary
from wbfm.utils.projects.utils_neuron_names import name2int_neuron_and_tracklet


def double_gaussian_mixture_model_to_histogram(neuron, pixel_values_dict_red):
    """gives back auc for the gaussian curve with higher mean

    example for input that is expected as pixel_values_dict_red:
    file = open(
        "S:/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/C-exp12_worm3-2022_08_01//visualization/pixel_values_all_neurons_red.pickle",
        "rb")
    pixel_values_dict_red = pickle.load(file, encoding='bytes')"""

    auc_trace = []
    num_timepoints = len(pixel_values_dict_red)
    neuron_int = name2int_neuron_and_tracklet(neuron)

    for timepoint in tqdm(range(num_timepoints)):
        try:
            X = np.array(pixel_values_dict_red[timepoint][neuron_int]).reshape(-1, 1)
            gm = GaussianMixture(n_components=2).fit(X)
            label = np.where((gm.means_ == np.max(gm.means_)).flatten())[0][0]
            amplitude = np.max(gm.predict_proba(gm.means_)[label]) * np.sum(gm.predict(X) == label)
            auc = amplitude * gm.covariances_[label] * np.sqrt(2 * np.pi)
            auc_trace.append(auc[0][0])
        except KeyError:
            auc_trace.append(np.nan)

    return auc_trace


def top_percentage(project_data, pixel_values_dict_red, pixel_values_dict_green,
                   percentage=0.25, neuron_names=None,
                   DEBUG=False):
    """gives back two dataframes (red;green) with new trace for every neuron

    example for input that is expected as pixel_values_dict_red:
    file = open(
        "S:/neurobiology/zimmer/fieseler/wbfm_projects/exposure_12ms/C-exp12_worm3-2022_08_01//visualization/pixel_values_all_neurons_red.pickle",
        "rb")
    pixel_values_dict_red = pickle.load(file, encoding='bytes')"""

    if neuron_names is None:
        neuron_names = project_data.neuron_names
    num_neurons = len(neuron_names)
    num_timepoints = project_data.red_traces.shape[0]

    extracted_traces_green = np.array([np.array([np.nan] * num_timepoints)] * num_neurons)
    extracted_traces_red = np.array([np.array([np.nan] * num_timepoints)] * num_neurons)

    for i_neuron, neuron_name in enumerate(tqdm(neuron_names, leave=False)):
        mean_vol = np.mean(project_data.red_traces[neuron_name]["area"])
        num_pixel = int(percentage * mean_vol)
        if DEBUG:
            print(num_pixel)

        for timepoint in np.sort(list(pixel_values_dict_red.keys())):

            # red
            dic = pixel_values_dict_red[timepoint]
            if DEBUG:
                print(dic[neuron_name])

            if neuron_name in dic.keys():
                extracted_traces_red[i_neuron, timepoint] = np.sum(np.sort(dic[neuron_name])[-num_pixel:])

            # green
            dic = pixel_values_dict_green[timepoint]

            if neuron_name in dic.keys():
                extracted_traces_green[i_neuron, timepoint] = np.sum(np.sort(dic[neuron_name])[-num_pixel:])

    df_extracted_red = pd.DataFrame(extracted_traces_red, neuron_names).T
    df_extracted_green = pd.DataFrame(extracted_traces_green, neuron_names).T
    return df_extracted_red, df_extracted_green


def save_alternate_trace_dataframes(project_data, percentage_list=None):
    """
    Uses histograms of pixels to extract an alternate method of calculating traces

    Uses a constant number of pixels across time (different per neuron)

    Parameters
    ----------
    project_data - FinishedProjectData class
    percentage_list - list of fractions of pixels to keep. Default is [0.1, 0.25, 0.5]

    Returns
    -------

    """
    if percentage_list is None:
        percentage_list = [0.1, 0.25, 0.5]

    dirname = project_data.project_config.get_visualization_config(make_subfolder=True).absolute_subfolder

    red_fname = os.path.join(dirname, 'pixel_values_all_neurons_red.pickle')
    red_dict = pickle_load_binary(red_fname)
    green_fname = os.path.join(dirname, 'pixel_values_all_neurons_green.pickle')
    green_dict = pickle_load_binary(green_fname)

    for percentage in tqdm(percentage_list):
        df_extracted_red, df_extracted_green = top_percentage(project_data, red_dict, green_dict,
                                                              percentage=percentage)

        fname = f"df_top_{percentage}"
        fname = fname.replace('.', '-')
        fname = os.path.join('visualization', f"{fname}")
        red_fname = f"{fname}_red.h5"
        project_data.project_config.save_data_in_local_project(df_extracted_red, red_fname)
        green_fname = f"{fname}_green.h5"
        project_data.project_config.save_data_in_local_project(df_extracted_green, green_fname)
