from collections import defaultdict

import numpy as np
import pandas as pd
from fDNC.src.DNC_predict import filter_matches
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns

from wbfm.utils.general.postprocessing.postprocessing_utils import filter_dataframe_using_likelihood
from wbfm.utils.projects.utils_neuron_names import int2name_neuron
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df


def calc_true_positive(gt: dict, test: dict):
    num_tp = 0
    for k, v in gt.items():
        if test.get(k, None) == v:
            num_tp += 1
    return num_tp


def calc_mismatches(gt: dict, test: dict):
    num_mm = 0
    for k, v in test.items():
        if gt.get(k, None) != v:
            num_mm += 1
    return num_mm


def calc_missing_matches(gt: dict, test: dict):
    num_mm = 0
    for k, v in gt.items():
        if k not in test:
            num_mm += 1
    return num_mm


def calc_summary_scores_for_training_data(m_final,
                                          min_confidence=0.0,
                                          max_possible=None):
    """
    Assumes the true matches are trivial, e.g. (1,1)

    max_possible defaults to assuming the maximum match in the first column (template) is all that is possible
    """
    m_final = filter_matches(m_final, min_confidence)
    if max_possible is None:
        max_possible = np.max(m_final[:, 0]).astype(int) + 1
    m0to1_dict = {m[0]: m[1] for m in m_final}

    num_tp = 0
    num_outliers = 0
    for m0, m1 in m0to1_dict.items():
        if m0 > max_possible:
            continue
        if m0 == m1:
            num_tp += 1
        else:
            num_outliers += 1
    num_missing = max_possible - num_tp - num_outliers

    return num_tp, num_outliers, num_missing, max_possible


def get_confidences_of_tp_and_outliers(m_final):
    """
    Assumes the true matches are trivial, e.g. (1,1)

    max_possible defaults to assuming the maximum match in the first column (template) is all that is possible
    """
    m0to1_dict = {m[0]: m[1] for m in m_final}
    m0toconf_dict = {m[0]: m[2] for m in m_final}

    conf_tp = []
    conf_outliers = []
    for m0, m1 in m0to1_dict.items():
        if m0 == m1:
            conf_tp.append(m0toconf_dict[m0])
        else:
            conf_outliers.append(m0toconf_dict[m0])

    return conf_tp, conf_outliers

##
## Using manually_tracked ground truth
##
# TRACKED_IND = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16, 21, 26, 30, 31, 32, 33, 34, 35, 39, 41, 42, 43, 44, 45, 46, 47, 49, 53, 55, 56, 61, 62, 71, 72, 75, 82, 84, 86, 95]
TRACKED_IND = [1, 2, 3, 4, 5,
               6, 8, 9, 10,
               11, 12, 14,
               16,
               21,
               26, 27, 28, 29, 30,
               31, 32, 33, 34, 35,
               37, 39,
               41, 42, 43, 44, 45,
               46, 47, 48, 49, 50,
               51, 52, 53, 55,
               56,
               61, 62, 65,
               71, 72, 75,
               78, 79, 80,
               82, 84,
               86, 90,
               95]


def calc_all_dist(df1, df2):
    # Check if they are same neuron, i.e. right on top of each other
    df1.replace(0.0, np.nan, inplace=True)
    df2.replace(0.0, np.nan, inplace=True)
    df_norm = np.sqrt(np.square(df1 - df2).sum(axis=1, min_count=1))

    num_total1 = df1.count()[0]
    num_total2 = df2.count()[0]
    num_total_total = df_norm.count()

    return df_norm.to_numpy(), num_total1, num_total2, num_total_total


def calc_accuracy(all_dist, dist_tol=1e-2):
    # Due to nan values, num_matches + num_mismatches != num_total
    num_matches = len(np.where(all_dist < dist_tol)[0])
    num_mismatches = len(np.where(all_dist > dist_tol)[0])
    num_nan_total = len(np.where(np.isnan(all_dist))[0])

    return num_matches, num_mismatches, num_nan_total


def calculate_column_of_differences(df_gt, df_test,
                                    column_to_check='raw_neuron_ind_in_list', neurons_that_are_finished=None):
    """
    Compares the neuron indices between a ground truth and a test dataframe
    Returns a list of columns with name 'true_neuron_ind' that can be concatenated to the original dataframe using:
    ```
    df_list.insert(0, df_test)
    df_with_true_neuron_ind = pd.concat(df_list, axis=1)
    ```

    Note that df_test can be a dataframe of tracks or tracklets
    """
    if neurons_that_are_finished is None:
        lookup = df_gt.loc[:, (slice(None), column_to_check)]
    else:
        lookup = df_gt.loc[:, (neurons_that_are_finished, column_to_check)]

    names = get_names_from_df(neurons_that_are_finished)

    df_list = []
    for name in tqdm(names):
        track = df_test[name][column_to_check]

        # Note: nan evaluates to false
        mask = lookup.apply(lambda col: col == track)
        idx, true_neuron_ind = np.where(mask)

        # Remove duplicates
        idx, idx_unique = np.unique(idx, return_index=True)
        true_neuron_ind = true_neuron_ind[idx_unique]

        df_list.append(pd.DataFrame(data=true_neuron_ind, columns=[(name, 'true_neuron_ind')], index=idx))

    # Construct full dataframe with these new columns
    # df_list.insert(0, df_test)
    # df_with_true_neuron_ind = pd.concat(df_list, axis=1)
    return df_list

##
## Plotting
##


def plot_histogram_at_likelihood_thresh(df1, df2, likelihood_thresh):
    """Assumes that the neurons have the same name; see rename_columns_using_matching"""
    df2_filter = filter_dataframe_using_likelihood(df2, likelihood_thresh)
    df_all_acc = calculate_accuracy_from_dataframes(df1, df2_filter)

    dat = [df_all_acc['matches_to_gt_nonnan'], df_all_acc['mismatches'], df_all_acc['nan_in_fdnc']]

    sns.histplot(dat, common_norm=False, stat="percent", multiple="stack")
    # sns.histplot(dat, multiple="stack")
    plt.ylabel("Percent of true neurons")
    plt.xlabel("Accuracy (various metrics)")

    plt.title(f"Likelihood threshold: {likelihood_thresh}")

    return dat


def calculate_accuracy_from_dataframes(df_gt: pd.DataFrame, df2_filter: pd.DataFrame) -> pd.DataFrame:
    tracked_names = get_names_from_df(df_gt)

    all_dist_dict, all_total1, all_total2 = calculate_distance_pair_of_dataframes(df_gt, df2_filter)

    num_t = df_gt.shape[0]
    all_acc_dict = defaultdict(list)
    for name in tqdm(tracked_names, leave=False):
        matches, mismatches, nan = calc_accuracy(all_dist_dict[name])
        num_total1, num_total2 = all_total1[name], all_total2[name]
        all_acc_dict['matches'].append(matches / num_t)
        all_acc_dict['matches_to_gt_nonnan'].append(matches / num_total1)
        all_acc_dict['mismatches'].append(mismatches / num_t)
        all_acc_dict['nan_in_fdnc'].append((num_t - num_total2) / num_t)
    df_all_acc = pd.DataFrame(all_acc_dict, index=tracked_names)
    return df_all_acc


def calculate_distance_pair_of_dataframes(df_gt, df2_filter):
    # Calculate distance between neuron positions in two dataframes with the SAME COLUMN NAMES
    coords = ['z', 'x', 'y']
    tracked_names = get_names_from_df(df_gt)
    all_dist_dict = {}
    all_total1 = {}
    all_total2 = {}
    for name in tqdm(tracked_names, leave=False):
        this_df_gt, this_df2 = df_gt[name][coords].copy(), df2_filter[name][coords].copy()
        all_dist_dict[name], all_total1[name], all_total2[name], _ = calc_all_dist(this_df_gt, this_df2)
    return all_dist_dict, all_total1, all_total2


def calculate_confidence_of_mismatches(df_gt: pd.DataFrame, df2_filter: pd.DataFrame) -> pd.DataFrame:
    """
    Returns all confidence values, instead of summary statistics (see: calculate_accuracy_from_dataframes)

    Returns an extended dataframe df2_filter, which has added columns ('is_correct') corresponding to correctness:
        0 - mismatch
        1 - match
        2 - ground truth was nan

    Parameters
    ----------
    df_gt
    df2_filter

    Returns
    -------

    """
    tracked_names = get_names_from_df(df_gt)

    all_dist_dict, all_total1, all_total2 = calculate_distance_pair_of_dataframes(df_gt, df2_filter)

    df2_with_classes = df2_filter.copy()

    dist_tol = 1e-2
    num_t = df2_filter.shape[0]
    for name in tqdm(tracked_names, leave=False):
        this_dist = all_dist_dict[name]

        # ind_matches = np.where(this_dist < dist_tol)[0]
        ind_mismatches = np.where(this_dist > dist_tol)[0]
        ind_nan = np.where(np.isnan(this_dist))[0]

        new_col = np.ones(num_t, dtype=int)
        new_col[ind_mismatches] = 0
        new_col[ind_nan] = 2

        df2_with_classes.loc[:, (name, 'is_correct')] = new_col

    return df2_with_classes.copy()  # To defragment
