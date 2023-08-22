from pathlib import Path

import pandas as pd

from wbfm.utils.projects.utils_filenames import pickle_load_binary


##
# "Final" set of good datasets
##


def get_summary_visualization_dir():
    """
    Directory to save overall files, e.g. from anything using load_good_datasets

    Returns
    -------

    """

    return "/scratch/neurobiology/zimmer/fieseler/multiproject_visualizations"


def get_project_parent_folder():
    return "/scratch/neurobiology/zimmer/fieseler/wbfm_projects"


def load_paper_datasets(genotype='gcamp', require_behavior=True) -> dict:
    """

    As of Dec 2022, these are the datasets we will use, with this condition:
        gcamp7b
        spacer
        2 percent agar

    Parameters
    ----------
    genotype
    require_behavior

    Returns
    -------

    """
    from wbfm.utils.projects.finished_project_data import load_all_projects_from_list, load_all_projects_in_folder

    # Build a dictionary of all
    if genotype == 'gcamp':
        folder_and_id_dict = {
            "2022-11-23_spacer_7b_2per_agar": [8, 9, 10, 11, 12],
            "2022-11-27_spacer_7b_2per_agar": [1, 3, 4, 5, 6],
            "2022-11-30_spacer_7b_2per_agar": [1, 2],
            "2022-12-05_spacer_7b_2per_agar": [3, 9, 10],
            "2022-12-10_spacer_7b_2per_agar": [1, 2, 3, 4, 5, 6, 7, 8]
        }
        list_of_all_projects = _resolve_project_from_worm_id(folder_and_id_dict)

        good_projects = load_all_projects_from_list(list_of_all_projects)
    elif genotype == 'gcamp_good':
        # Determined by looking at the data and deciding which ones are good
        folder_and_id_dict = {
            "2022-11-27_spacer_7b_2per_agar": [1, 3, 5, 6],
            "2022-11-30_spacer_7b_2per_agar": [1, 2],
            "2022-12-05_spacer_7b_2per_agar": [3, 10],
            "2022-12-10_spacer_7b_2per_agar": [2, 5, 7, 8]
        }
        list_of_all_projects = _resolve_project_from_worm_id(folder_and_id_dict)
        good_projects = load_all_projects_from_list(list_of_all_projects)
    elif genotype == 'gfp':
        folder_path = '/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-10_spacer_7b_2per_agar_GFP'
        good_projects = load_all_projects_in_folder(folder_path)
    elif genotype == 'immob':
        folder_path = '/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-03_immob_adj_settings_2'
        require_behavior = False  # No annotation of behavior here
        good_projects = load_all_projects_in_folder(folder_path)
        # Second folder, which extends above dictionary
        folder_path = '/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-12_immob'
        good_projects.update(load_all_projects_in_folder(folder_path))
    else:
        raise NotImplementedError

    if require_behavior:
        good_projects_filtered = {k: p for k, p in good_projects.items() if p.worm_posture_class.has_beh_annotation}
        if len(good_projects_filtered) < len(good_projects):
            print("Removed some designated 'good' projects because they didn't have behavior")
    else:
        good_projects_filtered = good_projects

    print(f"Loaded {len(good_projects_filtered)} projects")

    return good_projects_filtered


def _resolve_project_from_worm_id(folder_and_id_dict):
    list_of_all_projects = []
    parent_folder = Path(get_project_parent_folder())
    for rel_group_folder, worm_id_list in folder_and_id_dict.items():
        group_folder = parent_folder.joinpath(rel_group_folder)
        for worm_id in worm_id_list:
            worm_id_str = f"worm{worm_id}"
            for this_project_folder in group_folder.iterdir():
                if worm_id_str in this_project_folder.name:
                    list_of_all_projects.append(this_project_folder.resolve())
    return list_of_all_projects

# def get_path_to_double_exponential_model():
#     """Model fit to the forward duration distribution. See fit_multi_exponential_model"""


def forward_distribution_statistics():
    fname = "/scratch/neurobiology/zimmer/wbfm/DistributionsOfBehavior/forward_duration.pickle"
    forward_duration_dict = pickle_load_binary(fname)
    return forward_duration_dict


def reverse_distribution_statistics():
    fname = "/scratch/neurobiology/zimmer/wbfm/DistributionsOfBehavior/reversal_duration.pickle"
    duration_dict = pickle_load_binary(fname)
    return duration_dict


def names_of_neurons_to_id():
    fname = "/lisc/scratch/neurobiology/zimmer/wbfm/id_resources/neurons_to_id.csv"
    if Path(fname).exists():
        df = pd.read_csv(fname)
    else:
        df = None
    return df
