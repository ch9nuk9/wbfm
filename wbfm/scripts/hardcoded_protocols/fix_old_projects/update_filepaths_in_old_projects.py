from tqdm.auto import tqdm
from wbfm.utils.general.hardcoded_paths import load_paper_datasets
from wbfm.utils.general.utils_filenames import correct_mounted_path_prefix


def main():
    """
    Loads all the projects being used for the paper, and updates the paths in the config files

    Returns
    -------

    """

    # Load all the projects, including ones where I don't have permissions
    all_projects_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'])
    all_projects_gfp = load_paper_datasets('gfp')
    all_projects_immob = load_paper_datasets('immob')

    def _update_paths_in_project(_p):
        for k, v in _p.project_config.config.items():
            is_updated = False
            v_new = None
            if isinstance(v, str):
                v_new, is_updated = correct_mounted_path_prefix(v)
            if is_updated and v_new is not None:
                _p.project_config.config[k] = v_new
        _p.project_config.update_self_on_disk()

    for _, p in tqdm(all_projects_gcamp.items()):
        try:
            _update_paths_in_project(p)
        except PermissionError as e:
            print(f'Could not update project {p.project_name} due to permission error')
    for _, p in tqdm(all_projects_gfp.items()):
        _update_paths_in_project(p)
    for _, p in tqdm(all_projects_immob.items()):
        _update_paths_in_project(p)

    print('Done updating paths in all projects')


if __name__ == '__main__':
    main()
