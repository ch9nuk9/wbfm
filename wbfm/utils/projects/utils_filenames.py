import logging
import os
import pickle
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict

import numpy as np
import pandas as pd
from pip._internal.commands.show import search_packages_info

from wbfm.utils.general.custom_errors import UnknownValueError


def check_exists(abs_path, allow_overwrite):
    if Path(abs_path).exists():
        if allow_overwrite:
            logging.warning("Overwriting existing file")
        else:
            raise FileExistsError


def resolve_mounted_path_in_current_os(raw_path: str, verbose: int = 0) -> str:
    """
    Removes windows-specific mounted drive names (Y:, D:, etc.) and replaces them with the networked system equivalent

    Does nothing if the path is relative

    Note: This is specific to the Zimmer lab, as of 23.06.2021 (at the IMP)
    """
    is_abs = PurePosixPath(raw_path).is_absolute() or PureWindowsPath(raw_path).is_absolute()
    if not is_abs:
        return raw_path

    if verbose >= 1:
        print(f"Checking path {raw_path} on os {os.name}...")

    is_linux = "ix" in os.name.lower()
    is_windows = os.name.lower() == "windows" or os.name.lower() == "nt"

    # Check for silly things
    if os.path.exists(raw_path):
        return raw_path

    # Check for unreachable local drives
    local_drives = ['C:', 'D:']
    if is_linux:
        for drive in local_drives:
            if raw_path.startswith(drive):
                raise FileNotFoundError("File mounted to local drive; network system can't find it")

    # Swap mounted drive locations
    # UPDATE REGULARLY
    mounted_drive_dict = {
        'Y:': "/groups/zimmer",
        'Z:': "/scratch",
        'S:': "/scratch"
    }

    # Loop through drive name matches, and test each one
    for win_drive, linux_drive in mounted_drive_dict.items():
        is_windows_style = raw_path.startswith(win_drive)
        is_linux_style = raw_path.startswith(linux_drive)

        path = None
        if is_linux and is_windows_style:
            path = raw_path.replace(win_drive, linux_drive)
            path = str(Path(path).resolve())
        elif is_windows and is_linux_style:
            path = raw_path.replace(linux_drive, win_drive)
            path = str(Path(path).resolve())

        if path and os.path.exists(path):
            # For example on windows, tries Z: and if not found, tries S:
            if verbose >= 1:
                print(f"Successfully resolved path to {path}")
            break
    else:
        path = raw_path
        if verbose >= 1:
            print(f"Did not successfully resolve path; returning raw {raw_path}")

    assert path is not None

    return path


def pandas_read_any_filetype(filename, **kwargs):
    if filename is None:
        return None
    elif os.path.exists(filename):
        if filename.endswith('.h5'):
            return pd.read_hdf(filename, **kwargs)
        elif filename.endswith('.pickle'):
            return pickle_load_binary(filename, **kwargs)
        elif filename.endswith('.csv'):
            return pd.read_csv(filename, **kwargs)
        elif filename.endswith('.xlsx'):
            return pd.read_excel(filename, **kwargs)
        else:
            raise NotImplementedError
    else:
        logging.debug(f"Did not find file {filename}")
        return None


def read_if_exists(filename: str, reader=pandas_read_any_filetype, **kwargs):
    if filename is None:
        return None
    elif os.path.exists(filename):
        try:
            return reader(filename, **kwargs)
        except UnicodeDecodeError:
            logging.warning(f"Encountered unicode error; aborting read of {filename}")
            return None
    else:
        logging.debug(f"Did not find file {filename}")
        return None


def pickle_load_binary(fname, verbose=0):
    with open(fname, 'rb') as f:
        try:
            dat = pickle.load(f)
        except (ValueError, AttributeError):
            # Pickle saved in 3.8 has a new protocol
            import pickle5
            dat = pickle5.load(f)
        except ModuleNotFoundError:
            # Pandas 2.0.0 can break compatibility
            # https://stackoverflow.com/questions/75953279/modulenotfounderror-no-module-named-pandas-core-indexes-numeric-using-metaflo
            dat = pd.read_pickle(fname)
            # logging.warning(f"Using pandas to read pickle file {fname}; "
            #                 f"May have unknown format changes due to pandas version")

    if verbose >= 1:
        logging.info(f"Read from pickle file: {fname}")
    return dat


def lexigraphically_sort(strs_with_numbers):
    # From: https://stackoverflow.com/questions/35728760/python-sorting-string-numbers-not-lexicographically
    # Note: works with strings like 'neuron0' 'neuron10' etc.
    return sorted(sorted(strs_with_numbers), key=len)


def load_file_according_to_precedence(fname_precedence: list,
                                      possible_fnames: Dict[str, str],
                                      this_reader: callable = read_if_exists, dryrun=False, **kwargs):
    """
    Load a file according to a dict of possible filenames, ordered by fname_precedence

    Parameters
    ----------
    fname_precedence
    possible_fnames
    this_reader
    dryrun
    kwargs

    Returns
    -------

    """
    most_recent_modified_key = get_most_recently_modified(possible_fnames)

    for i, key in enumerate(fname_precedence):
        if key in possible_fnames:
            fname = possible_fnames[key]
        elif key == 'newest':
            fname = possible_fnames[most_recent_modified_key]
        else:
            raise UnknownValueError(key)

        if fname is not None and Path(fname).exists():
            logging.debug(f"File for mode {key} exists at precendence: {i + 1}/{len(possible_fnames)}")
            if dryrun:
                data = None
                logging.debug(f"Dryrun: would have read data from: {fname}")
            else:
                data = this_reader(fname, **kwargs)
                logging.debug(f"Read data from: {fname}")
            if key != most_recent_modified_key:
                logging.debug(f"Not using most recently modified file (mode {most_recent_modified_key})")
            else:
                logging.debug(f"Using most recently modified file")
            break
    else:
        logging.debug(f"Found no files of possibilities: {possible_fnames}")
        data = None
        fname = None
    return data, fname


def get_most_recently_modified(possible_fnames: Dict[str, str]) -> str:
    all_mtimes, all_keys = [], []
    for k, f in possible_fnames.items():
        if f is not None and os.path.exists(f):
            all_mtimes.append(os.path.getmtime(f))
        else:
            all_mtimes.append(0.0)
        all_keys.append(k)
    most_recent_modified = np.argmax(all_mtimes)
    most_recent_modified_key = all_keys[most_recent_modified]
    return most_recent_modified_key


def get_sequential_filename(fname: str, verbose=1) -> str:
    """
    Check if the file or dir exists, and if so, append an integer

    Also check if this function has been used before, and remove the suffix

    Example if the files test.h5 and test-1.h5 exist:
        get_sequential_filename('test.h5') -> 'test-2.h5'
    """
    i = 1
    fpath = Path(fname)
    if fpath.exists():
        if verbose >= 1:
            print(f"Original fname {fpath} exists, so will be suffixed")
        base_fname, suffix_filetype = fpath.stem, fpath.suffix
        # Check for previous application of this function
        regex = r"-\d+$"
        matches = list(re.finditer(regex, base_fname))
        if len(matches) > 0:
            base_fname = base_fname[:matches[0].start()]
            base_fname = base_fname.strip('-')
            if verbose >= 1:
                print(f"Removed suffix {matches[0].group()}, so the basename is taken as: {base_fname}")

        new_base_fname = f"{str(base_fname)}-{i}"
        candidate_fname = fpath.with_name(new_base_fname + str(suffix_filetype))
        while Path(candidate_fname).exists():
            # Increment the suffix
            i += 1
            str_len = len(str(i))
            # Remove the previous suffix
            new_base_fname = new_base_fname[:-str_len].strip('-')
            # Add the new suffix and get the full path
            new_base_fname = f"{new_base_fname}-{i}"
            candidate_fname = fpath.with_name(f"{new_base_fname}{suffix_filetype}")

            if verbose >= 2:
                print(f"Trying {candidate_fname}...")
    else:
        candidate_fname = fname
    return str(candidate_fname)


def get_absname(project_path, fname):
    # Builds the absolute filepath using a project config filepath
    project_dir = Path(project_path).parent
    fname = Path(project_dir).joinpath(fname)
    return str(fname)


def add_name_suffix(path: str, suffix='-1'):
    """
    Add a suffix to a filename, before the filetype

    Example:
        add_name_suffix('test.h5', suffix='-1') -> 'test-1.h5'

    Parameters
    ----------
    path
    suffix

    Returns
    -------

    """
    fpath = Path(path)
    base_fname, suffix_fname = fpath.stem, fpath.suffix
    new_base_fname = str(base_fname) + f"{suffix}"
    candidate_fname = fpath.with_name(new_base_fname + str(suffix_fname))

    # Check for existence?
    return candidate_fname


def generate_output_data_names(cfg):
    """
    Location of preprocessed data

    Update: should be within the project folder itself!

    Parameters
    ----------
    cfg

    Returns
    -------

    """
    fname = cfg.resolve_mounted_path_in_current_os('red_bigtiff_fname')
    out_fname_red = str(cfg.resolve_relative_path(os.path.join("dat", f"{fname.stem}_preprocessed.zarr")))
    fname = cfg.resolve_mounted_path_in_current_os('green_bigtiff_fname')
    out_fname_green = str(cfg.resolve_relative_path(os.path.join("dat", f"{fname.stem}_preprocessed.zarr")))
    return out_fname_red, out_fname_green


def get_location_of_installed_project():
    package_info = next(search_packages_info(['wbfm']))
    return package_info.location


def get_location_of_new_project_defaults():
    parent_folder = Path(get_location_of_installed_project())
    target_folder = parent_folder.joinpath('wbfm').joinpath('new_project_defaults').resolve()
    return str(target_folder)


def get_bigtiff_fname_from_folder(folder_fname, channel_to_check=0):
    fname = None
    str_pattern = f'_Ch{channel_to_check}bigtiff.btf'
    alt_str_pattern = f'_Ch{channel_to_check}_bigtiff.btf'
    for item in Path(folder_fname).iterdir():
        if str(item).endswith(str_pattern) or str(item).endswith(alt_str_pattern):
            fname = str(item)
            break
    else:
        logging.warning(f"Did not find pattern {str_pattern} in folder {folder_fname}")
    return fname


def get_both_bigtiff_fnames_from_parent_folder(parent_data_folder):
    green_bigtiff_fname, red_bigtiff_fname = None, None
    for subfolder in Path(parent_data_folder).iterdir():
        if subfolder.is_file():
            continue
        if subfolder.name.endswith('_Ch0'):
            red_bigtiff_fname = get_bigtiff_fname_from_folder(subfolder, channel_to_check=0)
        if subfolder.name.endswith('_Ch1'):
            green_bigtiff_fname = get_bigtiff_fname_from_folder(subfolder, channel_to_check=1)

    if green_bigtiff_fname is None or red_bigtiff_fname is None:
        logging.warning(f"Did not find one of: {(green_bigtiff_fname, red_bigtiff_fname)}")

    return green_bigtiff_fname, red_bigtiff_fname
