from pathlib import Path

from wbfm.utils.projects.finished_project_data import load_all_projects_from_list


##
# "Final" set of good datasets
##


def get_summary_visualization_dir():
    """
    Directory to save overall files, e.g. from anything using load_good_datasets

    Returns
    -------

    """

    return "/scratch/neurobiology/zimmer/Charles/multiproject_visualizations"


def get_project_parent_folder():
    return "/scratch/neurobiology/zimmer/Charles/dlc_stacks"


def load_good_datasets():
    """
    As of Dec 2022, these are the datasets we will use, with this condition:
        gcamp7b
        spacer
        2 percent agar

    Returns
    -------

    """

    # Build a dictionary of all
    folder_and_id_dict = {
        "2022-11-30_spacer_7b_2per_agar": [1, 2],
        "2022-11-27_spacer_7b_2per_agar": [1, 3, 4, 6],
        "2022-11-23_spacer_7b_2per_agar": [8, 9, 10, 11, 12],
        "2022-12-05_spacer_7b_2per_agar": [3, 9, 10]
    }
    list_of_all_projects = []
    parent_folder = Path(get_project_parent_folder())
    for rel_group_folder, worm_id_list in folder_and_id_dict.items():
        group_folder = parent_folder.joinpath(rel_group_folder)
        for worm_id in worm_id_list:
            worm_id_str = f"worm{worm_id}"
            for this_project_folder in group_folder.iterdir():
                if worm_id_str in this_project_folder.name:
                    list_of_all_projects.append(this_project_folder.resolve())

    good_projects = load_all_projects_from_list(list_of_all_projects)
    good_projects_filtered = [p for p in good_projects if p.worm_posture_class.has_beh_annotation]
    if len(good_projects_filtered) < len(good_projects):
        print("Removed some designated 'good' projects because they didn't have behavior")

    return good_projects_filtered
