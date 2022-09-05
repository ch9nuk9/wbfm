import matplotlib.pyplot as plt
import os
import numpy as np
from wbfm.utils.projects.finished_project_data import ProjectData
import sklearn
import pandas as pd
from tqdm.auto import tqdm
from wbfm.utils.general.postprocessing.position_postprocessing import impute_missing_values_in_dataframe
from wbfm.utils.external.utils_pandas import fill_missing_indices_with_nan
import scipy
from lmfit.models import ExponentialModel
import pickle
from sklearn.mixture import GaussianMixture
import scipy.stats
import math
from wbfm.utils.visualization.plot_traces import make_grid_plot_using_project
from statsmodels.graphics.regressionplots import plot_partregress_grid
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from wbfm.utils.traces.bleach_correction import detrend_exponential_iter
import cv2
from segmentation.util.utils_pipeline import _create_or_continue_zarr
import zarr


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


def full_lm_with_windowed_regression_vol(project_data, neuron, window_size=5):
    """" gives back corrected trace for the selected neuron
    calculates alpha for every timepoint considering all neighbours with distance < window size"""


    num_timepoints = project_data.red_traces.shape[0]

    green = np.array(project_data.green_traces[neuron]["intensity_image"])
    vol = np.array(project_data.green_traces[neuron]["area"])
    red = np.array(project_data.red_traces[neuron]["intensity_image"])
    remove_nan = np.logical_and(np.invert(np.isnan(green)), np.invert(np.isnan(vol)))
    green = green[remove_nan]
    vol = vol[remove_nan]
    red = red[remove_nan]

    alpha_green = []
    for i in range(window_size, len(red) - window_size - 1):
        y = green[i - window_size:i + window_size]
        x = vol[i - window_size:i + window_size].reshape(-1, 1)
        model = sklearn.linear_model.LinearRegression(fit_intercept=False)
        model.fit(x, y)
        alpha_green.append(model.coef_)

    alpha_red = []
    for i in range(window_size, len(red) - window_size - 1):
        y = red[window_size:len(red) - window_size - 1]
        x = vol[window_size:len(red) - window_size - 1].reshape(-1, 1)
        model = sklearn.linear_model.LinearRegression(fit_intercept=False)
        model.fit(x, y)
        alpha_red.append(model.coef_)

    red_corrected = red[window_size:len(red) - window_size - 1] - np.array(alpha_red).flatten() * vol[window_size:len(
        red) - window_size - 1]
    green_corrected = green[window_size:len(red) - window_size - 1] - np.array(alpha_green).flatten() * vol[
                                                                                                        window_size:len(
                                                                                                            red) - window_size - 1]

    vol = project_data.red_traces[neuron]["area"][remove_nan][window_size:len(red) - window_size - 1]
    x = project_data.red_traces[neuron]["x"][remove_nan][window_size:len(red) - window_size - 1]
    y = project_data.red_traces[neuron]["y"][remove_nan][window_size:len(red) - window_size - 1]
    z = project_data.red_traces[neuron]["z"][remove_nan][window_size:len(red) - window_size - 1]
    t = np.array(range(num_timepoints))[remove_nan][window_size:len(red) - window_size - 1]
    X = [red_corrected, vol, x, y, z, t]

    X = np.c_[np.array(X).T]
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, green_corrected)
    x_pred = model.predict(X)

    res = green_corrected - x_pred
    return res

#gaussian blur functions

def gaussian_blur_volume(volume, kernel=(5, 5)):

    """ takes volume """
    restored = volume.copy()
    for z in tqdm(range(volume.shape[0])):
        restored[z, :, :] = cv2.GaussianBlur(volume[z, :, :], kernel, 0)

    return restored


def gaussian_blur_video(video, fname, kernel=(5, 5)):
    """takes video"""

    restored_video = _create_or_continue_zarr(fname + ".zarr", num_frames=video.shape[0], num_slices=video.shape[1],
                                              x_sz=video.shape[2], y_sz=video.shape[3], mode='w-')

    for i in tqdm(range(video.shape[0])):
        volume = gaussian_blur_volume(video[i, :, :], kernel=kernel)
        restored_video[i, :, :, :] = volume

    return restored_video


def gaussian_blur_using_config(project_cfg, fname_for_saving_red, fname_for_saving_green, kernel=(5, 5)):
    """takes config file"""
    # Open the file
    project_dat = ProjectData.load_final_project_data_from_config(project_cfg)
    video_dat_red = project_dat.red_data
    video_dat_green = project_dat.green_data

    gaussian_blur_video(video_dat_red, fname=fname_for_saving_red, kernel=kernel)
    gaussian_blur_video(video_dat_green, fname=fname_for_saving_green, kernel=kernel)