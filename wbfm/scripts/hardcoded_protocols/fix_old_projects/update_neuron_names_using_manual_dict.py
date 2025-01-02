from tqdm.auto import tqdm
from wbfm.utils.general.hardcoded_paths import load_all_paper_datasets
from wbfm.utils.projects.finished_project_data import rename_manual_ids_in_project
from wbfm.utils.external.custom_errors import NoNeuronsError

if __name__ == '__main__':
    # Load ALL of the paper projects and update the neuron names
    # This is a one-time operation, and will take a while
    all_projects_dict = load_all_paper_datasets()
    name_mapping = {'IL2LL': 'IL2L', 'IL1LL': 'IL1L', 'IL2LR': 'IL2R', 'IL1LR': 'IL1R'}
    all_errors = []
    for project_name, project in tqdm(all_projects_dict.items()):
        print(project.project_dir)
        try:
            rename_manual_ids_in_project(project, name_mapping)
        except PermissionError as e:
            all_errors.append(e)
        except ValueError as e:
            # This one is unusual, and should be investigated
            print(project.df_manual_ids.columns)
            raise e
        except (NoNeuronsError, AttributeError) as e:
            print(f"No neurons found for project: {project_name}")
            all_errors.append(e)
    print(f"All errors: {all_errors}")
