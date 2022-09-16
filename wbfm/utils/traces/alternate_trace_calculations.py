import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.mixture import GaussianMixture


def gaussian_mixture_model(project_data, neuron, pixel_values_dict_red):
    """gives back auc for the gaussian curve with higher mean

    example for input that is expected as pixel_values_dict_red:
    file = open(
        "S:/neurobiology/zimmer/Charles/dlc_stacks/exposure_12ms/C-exp12_worm3-2022_08_01//visualization/pixel_values_all_neurons_red.pickle",
        "rb")
    pixel_values_dict_red = pickle.load(file, encoding='bytes')"""

    auc_trace = []
    num_timepoints = project_data.red_traces.shape[0]
    neuron_int = int(neuron[-3:])

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


def top_percentage(project_data, pixel_values_dict_red, pixel_values_dict_green, percentage=0.25):
    """gives back two dataframes (red;green) with new trace for every neuron

    example for input that is expected as pixel_values_dict_red:
    file = open(
        "S:/neurobiology/zimmer/Charles/dlc_stacks/exposure_12ms/C-exp12_worm3-2022_08_01//visualization/pixel_values_all_neurons_red.pickle",
        "rb")
    pixel_values_dict_red = pickle.load(file, encoding='bytes')"""

    num_neurons = int(project_data.red_traces.shape[1] / 6)
    num_timepoints = project_data.red_traces.shape[0]

    neuron_names = []
    for neuron_int in tqdm(range(1, num_neurons)):
        neuron = "neuron_" + str(neuron_int).zfill(3)
        neuron_names.append(neuron)

    extracted_traces_green = np.array([np.array([np.nan] * num_timepoints)] * num_neurons)
    extracted_traces_red = np.array([np.array([np.nan] * num_timepoints)] * num_neurons)
    num_pixel = 10

    for neuron in tqdm(range(num_neurons)):
        neuron_name = "neuron_" + str(neuron + 1).zfill(3)
        mean_vol = np.mean(project_data.red_traces[neuron_name]["area"])
        num_pixel = int(percentage * mean_vol)

        for timepoint in np.sort(list(pixel_values_dict_red.keys())):

            # red
            dic = pixel_values_dict_red[timepoint]

            if neuron in dic.keys():
                extracted_traces_red[neuron, timepoint] = np.sum(np.sort(dic[neuron])[-num_pixel:])

            # green
            dic = pixel_values_dict_green[timepoint]

            if neuron in dic.keys():
                extracted_traces_green[neuron, timepoint] = np.sum(np.sort(dic[neuron])[-num_pixel:])

    df_extracted_red = pd.DataFrame(extracted_traces_red[1:, :], neuron_names)
    df_extracted_green = pd.DataFrame(extracted_traces_green[1:, :], neuron_names)
    return df_extracted_red, df_extracted_green

