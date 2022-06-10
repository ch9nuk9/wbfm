from typing import List

from tqdm.auto import tqdm

from DLC_for_WBFM.utils.external.utils_pandas import df_to_matches, get_column_name_from_time_and_column_value
from DLC_for_WBFM.utils.general.custom_errors import NoMatchesError
from DLC_for_WBFM.utils.neuron_matching.matches_class import MatchesWithConfidence
from DLC_for_WBFM.utils.projects.finished_project_data import ProjectData


def calc_mismatch_between_ground_truth_and_pairs(all_matches, df_gt, t0):
    pair = (t0, t0 + 1)

    model_matches = all_matches[pair].final_matches
    # model_matches = all_matches[pair].feature_matches
    gt_matches = df_to_matches(df_gt, t0)

    model_obj = MatchesWithConfidence.matches_from_array(model_matches)
    gt_obj = MatchesWithConfidence.matches_from_array(gt_matches, 1)
    gt_matches_different_model, model_matches_different_gt, model_matches_no_gt, gt_matches_no_model = get_mismatches(
        gt_obj, model_obj)

    return gt_matches_different_model, model_matches_different_gt, model_matches_no_gt, gt_matches_no_model


def calc_all_mismatches_between_ground_truth_and_pairs(project_data: ProjectData):
    dict_of_all_mismatches = dict()
    all_matches = project_data.raw_matches
    df_gt = project_data.get_final_tracks_only_finished_neurons()

    project_data.logger.info("Calculating mismatches between ground truth and automatic matches")
    for t0 in tqdm(range(project_data.num_frames - 2)):
        # Only need the first of the pair here, i.e. the index on t0, not t0+1
        try:
            gt_matches_different_model, _, _, _ = calc_mismatch_between_ground_truth_and_pairs(all_matches, df_gt, t0)
        except NoMatchesError:
            continue

        if gt_matches_different_model:
            for mismatch in gt_matches_different_model:
                raw_neuron_ind_in_list = mismatch[0]
                ind, neuron_name = get_column_name_from_time_and_column_value(df_gt, t0, raw_neuron_ind_in_list,
                                                                              col_name='raw_neuron_ind_in_list')

                mask_ind = project_data.segmentation_metadata.i_in_array_to_mask_index(t0, raw_neuron_ind_in_list)
                tracklet_name = project_data.tracklets_and_neurons_class.get_tracklet_from_segmentation_index(t0,
                                                                                                              mask_ind)
                # Convert to segmentation to use the tracklet class preallocated dict
                dict_of_all_mismatches[(t0, neuron_name, tracklet_name)] = True

    return dict_of_all_mismatches


def get_mismatches(gt_matches: MatchesWithConfidence, model_matches: MatchesWithConfidence, verbose=0):
    """
    Get mismatches of different types:
    1. Match is different between model and gt:
        1a. The match that the model gave
        1b. The match that the gt gave
    2. Match doesn't exist in gt, but does in model
    3. Match exists in gt, but doesn't in model

    Parameters
    ----------
    gt_matches
    model_matches

    Returns
    -------
    gt_matches_different_model
    model_matches_different_gt
    model_matches_no_gt
    gt_matches_no_model

    """

    dict_of_model_matches = model_matches.get_mapping_0_to_1(unique=True)
    dict_of_gt_matches = gt_matches.get_mapping_0_to_1(unique=True)
    list_of_model_matches: List[list] = model_matches.matches_without_conf.tolist()
    list_of_gt_matches: List[list] = gt_matches.matches_without_conf.tolist()
    inverse_dict_of_model_matches = model_matches.get_mapping_1_to_0()

    model_matches_no_gt = []
    gt_matches_no_model = []
    gt_matches_different_model = []
    model_matches_different_gt = []

    for gt_m in list_of_gt_matches:
        if gt_m in list_of_model_matches:
            if verbose >= 3:
                print(f"{gt_m} in {list_of_model_matches}")
            # Do not explicitly save correct matches
            continue
        elif gt_m[0] != inverse_dict_of_model_matches.get(gt_m[1], gt_m[0]):
            # The first time point had the wrong match
            gt_matches_different_model.append(gt_m)
            model_matches_different_gt.append([inverse_dict_of_model_matches[gt_m[1]], gt_m[1]])
        elif gt_m[1] != dict_of_model_matches.get(gt_m[0], gt_m[1]):
            # The second time point had the wrong match
            gt_matches_different_model.append(gt_m)
            model_matches_different_gt.append([gt_m[0], dict_of_model_matches[gt_m[0]]])
        else:
            # Rare; usually the model has many more matches
            gt_matches_no_model.append(gt_m)

    for model_m in list_of_model_matches:
        if model_m[0] not in dict_of_gt_matches:
            model_matches_no_gt.append(model_m)

    return gt_matches_different_model, model_matches_different_gt, model_matches_no_gt, gt_matches_no_model
