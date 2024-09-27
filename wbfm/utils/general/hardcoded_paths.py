import logging
import os
from pathlib import Path
from typing import Union
import pkgutil
import pandas as pd
from collections import defaultdict

from wbfm.utils.external.custom_errors import IncompleteConfigFileError
from wbfm.utils.general.utils_filenames import pickle_load_binary
from wbfm.utils.external.utils_yaml import load_config
from ruamel.yaml import YAML


##
## Code for loading various neural networks and things needed for a new user
##

def load_hardcoded_neural_network_paths() -> dict:
    """
    Loads everything that might be needed for a new user. Note that the paths are not defined here, but rather in

    Fundamentally tries to read from a config file that is stored in the user's home directory. If that file does not
    exist, it will then search for a environment variable that contains the path to the config file. If that does not
    exist, it will then try to create the config file using defaults... but this will be empty and throw an error.
    Specifically, the order is this:
    1. Look in ~/.wbfm/config.yaml
    2. Look in the environment variable WBFM_CONFIG_PATH, which should point to a .yaml
    3. Load from the package, which has defaults that only work for the zimmer lab (tries to check if this is the zimmer lab)
    4. Create a default config file in ~/.wbfm/config.yaml, and raise IncompleteConfigFileError

    """
    # First, try to load from the config file
    which_method_worked = None
    default_config_path = Path.home().joinpath('.wbfm/config.yaml')
    try:
        config = load_config(default_config_path)
        which_method_worked = default_config_path
    except FileNotFoundError:
        logging.debug(f"Could not find config file at {default_config_path}; continuing search")

    # If that didn't work, try to load from the environment variable
    if which_method_worked is None:
        try:
            config_path = Path(os.environ['WBFM_CONFIG_PATH'])
            config = load_config(config_path)
            which_method_worked = "WBFM_CONFIG_PATH"
        except (KeyError, FileNotFoundError):
            logging.debug("Could not find WBFM_CONFIG_PATH in environment variables; continuing search")

    # If that didn't work, load from the zimmer-lab defaults
    if which_method_worked is None and is_zimmer_lab():
        try:
            config_dict_str = pkgutil.get_data('wbfm', 'utils/projects/wbfm_config.yaml')
            config = YAML().load(config_dict_str)
            which_method_worked = "defaults imported from package"
        except FileNotFoundError as e:
            logging.debug("Could not find config file within package... is the code properly installed?")
            raise e  # If we are in the zimmer lab and this fails, it's a real error

    # If that didn't work, try to create a default config file
    if which_method_worked is None:
        try:
            config_dict_str = pkgutil.get_data('wbfm', 'utils/projects/wbfm_config.yaml')
            # Create folder if needed, then make sure this is a valid yaml file
            default_config_path.parent.mkdir(parents=True, exist_ok=True)
            config_dict = YAML().load(config_dict_str)
            with open(default_config_path, "w") as f:
                YAML().dump(config_dict, f)
            which_method_worked = "new intialization"
            raise IncompleteConfigFileError(f"Created new config file at: {default_config_path}. "
                                            f"Please fill this out with the correct paths. ")
        except PermissionError:
            raise IncompleteConfigFileError(f"Could not create a default config file at {default_config_path}. "
                                            f"Please make sure you have permissions there, or create one manually. "
                                            f"Note: either way, this config file will have to be filled out manually.")

    logging.debug(f"Loaded config file from {which_method_worked}")

    return config


def is_zimmer_lab():
    """Loose check to see if the code is running on the lisc cluster, from the zimmer lab"""
    return Path('/lisc/scratch/neurobiology/zimmer').exists()

##
# "Final" set of good datasets
##


def get_summary_visualization_dir():
    """
    Directory to save overall files, e.g. from anything using load_good_datasets

    Returns
    -------

    """

    return "/lisc/scratch/neurobiology/zimmer/fieseler/multiproject_visualizations"


def get_project_parent_folder():
    return "/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects"


def get_hierarchical_modeling_dir(gfp=False, immobilized=False, o2_stimulus=False, mutant=False,
                                  suffix=None):
    parent_folder = "/lisc/scratch/neurobiology/zimmer/fieseler/paper/"
    base_name = "hierarchical_modeling"
    if suffix is None:
        if gfp:
            base_name += "_gfp"
        elif immobilized:
            base_name += "_immob"
        if mutant:
            base_name += "_mutant"
        if o2_stimulus:
            base_name += "_o2"
    else:
        if not suffix.startswith('_'):
            base_name += '_'
        base_name += suffix
    return os.path.join(parent_folder, base_name)


def get_triggered_average_modeling_dir():
    return "/lisc/scratch/neurobiology/zimmer/fieseler/paper/triggered_average_dataframes"


def load_all_data_as_dataframe():
    # Load each type of data, and then concatenate
    fname = os.path.join(get_hierarchical_modeling_dir(), 'data.h5')
    Xy_fm = pd.read_hdf(fname).assign(dataset_type='wbfm')

    fname = os.path.join(get_hierarchical_modeling_dir(immobilized=True), 'data.h5')
    Xy_immob = pd.read_hdf(fname).assign(dataset_type='immob')

    fname = os.path.join(get_hierarchical_modeling_dir(gfp=True), 'data.h5')
    Xy_gfp = pd.read_hdf(fname).assign(dataset_type='gfp')

    return pd.concat([Xy_fm, Xy_immob, Xy_gfp])


def load_paper_datasets(genotype: Union[str, list] = 'gcamp', require_behavior=False, only_load_paths=False,
                        **kwargs) -> dict:
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

    if isinstance(genotype, str) and genotype == '':
        # Load default gcamp paper projects
        genotype = ['gcamp', 'hannah_O2_fm']

    if isinstance(genotype, list):
        good_projects = {}
        for this_genotype in genotype:
            good_projects.update(load_paper_datasets(this_genotype, require_behavior=require_behavior,
                                                     only_load_paths=only_load_paths, **kwargs))
        return good_projects

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

        good_projects = load_all_projects_from_list(list_of_all_projects, only_load_paths=only_load_paths, **kwargs)
    elif genotype == 'gcamp_good':
        # Determined by looking at the data and deciding which ones are good
        folder_and_id_dict = {
            "2022-11-27_spacer_7b_2per_agar": [1, 3, 5, 6],
            "2022-11-30_spacer_7b_2per_agar": [1, 2],
            "2022-12-05_spacer_7b_2per_agar": [3, 10],
            "2022-12-10_spacer_7b_2per_agar": [2, 5, 7, 8]
        }
        list_of_all_projects = _resolve_project_from_worm_id(folder_and_id_dict)
        good_projects = load_all_projects_from_list(list_of_all_projects, only_load_paths=only_load_paths, **kwargs)
    elif genotype == 'gfp':
        folder_path = '/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-10_spacer_7b_2per_agar_GFP'
        good_projects = load_all_projects_in_folder(folder_path, only_load_paths=only_load_paths, **kwargs)
    elif genotype == 'immob':
        folder_path = '/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-03_immob_adj_settings_2'
        require_behavior = False  # No annotation of behavior here
        good_projects = load_all_projects_in_folder(folder_path, only_load_paths=only_load_paths, **kwargs)
        # Second folder, which extends above dictionary
        folder_path = '/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-12_immob'
        good_projects.update(load_all_projects_in_folder(folder_path, only_load_paths=only_load_paths, **kwargs))
    elif genotype == 'hannah_O2_fm':
        folder_path = '/lisc/scratch/neurobiology/zimmer/brenner/wbfm_projects/analyze/freely_moving_wt'
        good_projects = load_all_projects_in_folder(folder_path, only_load_paths=only_load_paths, **kwargs)
        folder_path = '/lisc/scratch/neurobiology/zimmer/brenner/wbfm_projects/analyze/IM_to_FM_freely_moving'
        good_projects.update(load_all_projects_in_folder(folder_path, only_load_paths=only_load_paths, **kwargs))
    elif genotype == 'hannah_O2_immob' or genotype == 'immob_o2':
        folder_path = '/lisc/scratch/neurobiology/zimmer/brenner/wbfm_projects/analyze/immobilized_wt'
        good_projects = load_all_projects_in_folder(folder_path, only_load_paths=only_load_paths, **kwargs)
    elif genotype == 'hannah_O2_fm_mutant' or genotype == 'mutant':
        folder_path = '/lisc/scratch/neurobiology/zimmer/brenner/wbfm_projects/analyze/freely_moving_mutant'
        good_projects = load_all_projects_in_folder(folder_path, only_load_paths=only_load_paths, **kwargs)
    elif genotype == 'hannah_O2_immob_mutant' or genotype == 'immob_mutant_o2':
        folder_path = '/lisc/scratch/neurobiology/zimmer/brenner/wbfm_projects/analyze/immobilized_mutant'
        good_projects = load_all_projects_in_folder(folder_path, only_load_paths=only_load_paths, **kwargs)
    elif genotype == 'O2_hiscl' or genotype == 'immob_o2_hiscl':
        folder_path = '/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects/muscle_hiscl_o2_stimulation'
        good_projects = load_all_projects_in_folder(folder_path, only_load_paths=only_load_paths, **kwargs)
    else:
        raise NotImplementedError

    if require_behavior and not only_load_paths:
        print("Filtering out projects without behavior")
        good_projects_filtered = {k: p for k, p in good_projects.items() if p.worm_posture_class.has_beh_annotation}
        if len(good_projects_filtered) < len(good_projects):
            print("Removed some designated 'good' projects because they didn't have behavior")
    else:
        good_projects_filtered = good_projects

    # Change setting to use physical time for all
    if not only_load_paths:
        for project in good_projects_filtered.values():
            project.use_physical_time = True

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
    fname = "/lisc/scratch/neurobiology/zimmer/wbfm/DistributionsOfBehavior/forward_duration.pickle"
    forward_duration_dict = pickle_load_binary(fname)
    return forward_duration_dict


def reverse_distribution_statistics():
    fname = "/lisc/scratch/neurobiology/zimmer/wbfm/DistributionsOfBehavior/reversal_duration.pickle"
    duration_dict = pickle_load_binary(fname)
    return duration_dict


def read_names_of_neurons_to_id() -> pd.Series:
    fname = "/lisc/scratch/neurobiology/zimmer/wbfm/id_resources/neurons_to_id.csv"
    if Path(fname).exists():
        df = pd.read_csv(fname, header=None)
    else:
        df = list_of_neurons_to_id()
    return df


def list_of_neurons_to_id() -> pd.Series:
    """See names_of_neurons_to_id for higher priority list"""
    items = [
        "AIBL",
        "AIBR",
        "ALA",
        "AVAL",
        "AVAR",
        "AVBL",
        "AVBR",
        "AVEL",
        "AVER",
        "BAGL",
        "BAGR",
        "DA01",
        "DB01",
        "OLQDL",
        "OLQDR",
        "OLQVL",
        "OLQVR",
        "RIBL",
        "RIBR",
        "RID",
        "RIML",
        "RIMR",
        "RIS",
        "RIVL",
        "RIVR",
        "RMED",
        "RMEL",
        "RMER",
        "RMEV",
        "SIADL",
        "SIADR",
        "SIAVL",
        "SIAVR",
        "SMDDL",
        "SMDDR",
        "SMDVL",
        "SMDVR",
        "URADL",
        "URADR",
        "URAVR",
        "URAVL",
        "URYDL",
        "URYDR",
        "URYVL",
        "URYVR",
        "VA01",
        "VA02",
        "VB02",
        "VB03"
    ]
    return pd.Series(items)


def list_of_gas_sensing_neurons(include_non_suffix_names=False):
    neuron_list = []
    unilateral_neurons = list_of_unilateral_neurons()
    for raw_neuron in ['AQR', 'IL1L', 'IL2L', 'BAG', 'AUA', 'URX']:
        for suffix in ['L', 'R']:
            if raw_neuron not in unilateral_neurons:
                neuron = f"{raw_neuron}{suffix}"
            elif suffix == 'L':
                neuron = raw_neuron
            else:
                continue
            neuron_list.append(neuron)
        if include_non_suffix_names and raw_neuron not in unilateral_neurons:
            neuron_list.append(raw_neuron)
    return neuron_list


def list_neurons_manifold_in_immob():
    neuron_list = []
    unilateral_neurons = list_of_unilateral_neurons()
    for raw_neuron in ['AIB', 'AVA', 'AVB', 'AVE', 'BAG', 'OLQD', 'OLQV', 'RIB', 'RID', 'RIM', 'RIS',
                       'RME', 'RMED', 'RMEV', 'SIAD', 'SIAV', 'URAD', 'URAV', 'URYD', 'URYV',
                       'VA01', 'VA02', 'VB02', 'VB03', 'DA01']:
        for suffix in ['L', 'R']:
            if raw_neuron not in unilateral_neurons:
                neuron = f"{raw_neuron}{suffix}"
            elif suffix == 'L':
                neuron = raw_neuron
            else:
                # Do not add unilateral neurons twice
                continue
            neuron_list.append(neuron)
    return neuron_list


def list_of_unilateral_neurons():
    unilateral_neurons = ['AQR', 'RID', 'RIS', 'RMED', 'RMEV']
    # Also all neurons like 'DB0X' and 'VA0X'
    unilateral_neurons.extend([f"{dv}{ab}{i:02d}" for dv in ['D', 'V'] for ab in ['A', 'B'] for i in range(1, 5)])
    return unilateral_neurons


def default_raw_data_config():
    """As of Feb 2024"""
    return {'num_z_planes': 22,
            'flyback_saved': False,
            'num_flyback_planes_discarded': 2,
            'z_step_size': 1.5,
            'laser_561': 260,
            'laser_488': 985,
            'exposure_time': 12,
            'agar': 2,
            'recording_length_minutes': 8,
            'ventral': 'left',
            'strain': 'ZIM2165'
            }


def neurons_with_confident_ids(combine_left_right=False):
    neuron_names = ['AVAL', 'AVAR', 'BAGL', 'BAGR', 'RIMR', 'RIML', 'AVEL', 'AVER',
                    #'URAVL', 'URAVR',
                    'URYVL', 'URYVR', 'URADL', 'URADR', 'URYDL', 'URYDR',
                    'RIVR', 'RIVL', 'SMDVL', 'SMDVR', 'SMDDR', 'SMDDL',
                    'SIAVL', 'SIAVR', 'SAAVL', 'SAAVR', 'SIADL', 'SIADR', 'RIAL', 'RIAR',
                    'RMDDL', 'RMDDR', #'AVFL', 'AVFR',
                    'AWBL', 'AWBR',
                    'AWAL', 'AWAR',
                    'IL1LL', 'IL1LR', 'IL2LL', 'IL2LR', #'OLQDL', 'OLQDR', 'OLQVL', 'OLQVR',
                    'ALA', 'RIS', 'AQR', 'RMDVL', 'RMDVR', 'URXL', 'URXR',
                    'VB01', 'VB02', 'VB03', 'DB01', 'DB02', 'VA01', 'VA02', 'DA01', 'DD01',
                    'RIBL', 'RIBR', 'RMEL', 'RMER', 'RMED', 'RMEV', 'RID', 'AVBL', 'AVBR']
    if combine_left_right:
        neuron_names = [n[:-1] if (n[-1] in ['L', 'R'] and len(n) > 3) else n for n in neuron_names ]
    return neuron_names


def _role_of_neurons():
    """Dictionary with role of all neurons... lol chatgpt was mostly right"""
    c_elegans_neurons = {
    'Sensory': [
            'ASI', 'AWA', 'AWB', 'AWC', 'ASH', 'ASJ', 'ASG', 'ASK', 'ADF', 'PDE',
            'IL2L', 'IL2R', 'IL2', 'IL1L', 'IL1R', 'IL1', 'OLQD', 'OLQV', 'CEP', 'ADE', 'PVD', 'FLP', 'PHA', 'PHB', 'URX', 'BAG',
            'AQR', 'SDQ', 'PQR', 'AUA', 'URB', 'SAA', 'URYD', 'URYV', 'URAD', 'URAV'
        ],
    'Interneuron': [
            'AIA', 'AIB', 'AIZ', 'AVA', 'AVE', 'AVB', 'AVD', 'AVG', 'RIM', 'ALA',
            'RIB', 'RIC', 'RIA', 'RIG', 'RIF', 'RIS', 'AIM', 'DVA', 'HSN', 'PVQ', 'RID'
        ],
    'Motor': [
            'DA1-DA9', 'DB1-DB7', 'DD1-DD6', 'VD1-VD13', 'VA1-VA12', 'VB1-VB11',
            'AS1-AS11', 'VC1-VC6', 'SAB', 'DVB', 'PDA', 'SIAD', 'SIAV', 'SMDD', 'SMDV',
            'PDB', 'PVC', 'SMB', 'SIB', 'RMF', 'RMDD', 'RMDV', 'RMEV', 'RMED', 'RME', 'RIV'
        ],
    'Modulatory': [
            'RID', 'RIM'
        ],
    'Forward': [
        'AVB', 'RIB', 'DB1-DB7', 'VB1-VB11', 'RME', 'RMEV', 'RMED', 'RID'
        ],
    'Reverse': [
        'AVA', 'AIB', 'RIM', 'DA1-DA9', 'VA1-VA12', 'AVE'
        ],
    'Dorsal': [
        'DA1-DA9', 'DB1-DB7', 'DD1-DD6', 'SMDD', 'RMEV', 'SIAD'
        ],
    'Ventral': [
        'VA1-VA9', 'VB1-VB7', 'VD1-VD6', 'SMDV', 'RMED', 'SIAV'
        ],
    # 'Turning': [
    #     'SMDV', 'SMDD', 'RIV'
    #     ]
    # 'Pharyngeal': [
    #         'M1', 'M2', 'M3', 'M4', 'M5', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6',
    #         'MC', 'NSM', 'MI', 'RAP', 'RIC', 'RID', 'SIA', 'SIB', 'SMB'
    #     ]
    }
    return c_elegans_neurons


def role_of_neuron_dict(only_fwd_rev=False, include_modulatory=False, include_ventral_dorsal=False):
    # Build a dictionary with the role of each neuron, from names to roles
    # Use the high level info in _role_of_neurons
    role_dict = defaultdict(list)
    for role, info in _role_of_neurons().items():

        if role in ['Forward', 'Reverse']:
            if not only_fwd_rev:
                continue
        elif only_fwd_rev:
            continue

        if role in ['Ventral', 'Dorsal'] and not include_ventral_dorsal:
            continue
        if role == 'Modulatory' and not include_modulatory:
            continue

        for neuron in info:
            keys = [neuron]
            if '-' in neuron:
                keys = []
                # Then this is a range of neurons, and it should be expanded
                # e.g. 'DA1-DA9' should be expanded to 'DA1', 'DA2', ..., 'DA9'
                first_id, last_id = neuron.split('-')
                base_name = first_id[:2]  # The base is always 2 characters
                i_end = int(last_id[2:])
                for i in range(1, i_end + 1):
                    keys.append(f"{base_name}{i:02d}")
            elif neuron not in list_of_unilateral_neurons():
                keys.append(neuron + 'L')
                keys.append(neuron + 'R')
            for k in keys:
                role_dict[k].append(role)
    # Combine all roles, if multiple, into a single string
    for k, v in role_dict.items():
        role_dict[k] = ', '.join(v) if len(v) > 1 else v[0]

    return role_dict
