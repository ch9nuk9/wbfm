import logging
import os
import pickle
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict

import numpy as np
import pandas as pd

from DLC_for_WBFM.utils.general.custom_errors import UnknownValueError


def check_exists(abs_path, allow_overwrite):
    if Path(abs_path).exists():
        if allow_overwrite:
            logging.warning("Overwriting existing file")
        else:
            raise FileExistsError


def resolve_mounted_path_in_current_os(path: str, verbose: int = 1) -> str:
    """
    Removes windows-specific mounted drive names (Y:, D:, etc.) and replaces them with the networked system equivalent

    Does nothing if the path is relative

    Note: This is specific to the Zimmer lab, as of 23.06.2021 (at the IMP)
    """
    is_abs = PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute()
    if not is_abs:
        return path

    if verbose >= 1:
        print(f"Checking path {path} on os {os.name}...")

    # Swap mounted drive locations
    # UPDATE REGULARLY
    mounted_drive_dict = {
        'Y:': "/groups/zimmer"
    }

    for win_drive, linux_drive in mounted_drive_dict.items():
        is_linux = "ix" in os.name.lower()
        is_windows_style = path.startswith(win_drive)
        is_windows = os.name.lower() == "windows" or os.name.lower() == "nt"
        is_linux_style = path.startswith(linux_drive)

        if is_linux and is_windows_style:
            path = path.replace(win_drive, linux_drive)
            path = str(Path(path).resolve())
        if is_windows and is_linux_style:
            path = path.replace(linux_drive, win_drive)
            path = str(Path(path).resolve())

    # Check for unreachable local drives
    local_drives = ['C:', 'D:']
    if "ix" in os.name.lower():
        for drive in local_drives:
            if path.startswith(drive):
                raise FileNotFoundError("File mounted to local drive; network system can't find it")

    if verbose >= 1:
        print(f"Resolved path to {path}")
    return path


def read_if_exists(filename, reader=pd.read_hdf):
    if filename is None:
        return None
    elif os.path.exists(filename):
        return reader(filename)
    else:
        logging.warning(f"Did not find file {filename}")
        return None


def pandas_read_any_filetype(filename):
    if filename is None:
        return None
    elif os.path.exists(filename):
        if filename.endswith('.h5'):
            return pd.read_hdf(filename)
        elif filename.endswith('.pickle'):
            return pd.read_pickle(filename)
        else:
            raise NotImplementedError
    else:
        logging.warning(f"Did not find file {filename}")
        return None


def pickle_load_binary(fname, verbose=0):
    with open(fname, 'rb') as f:
        dat = pickle.load(f)
    if verbose >= 1:
        logging.info(f"Read from pickle file: {fname}")
    return dat


def lexigraphically_sort(strs_with_numbers):
    # From: https://stackoverflow.com/questions/35728760/python-sorting-string-numbers-not-lexicographically
    # Note: works with strings like 'neuron0' 'neuron10' etc.
    return sorted(sorted(strs_with_numbers), key=len)


def load_file_according_to_precedence(fname_precedence: list,
                                      possible_fnames: Dict[str, str],
                                      this_reader: callable = read_if_exists):
    most_recent_modified_key = get_most_recently_modified(possible_fnames)

    for i, key in enumerate(fname_precedence):
        if key in possible_fnames:
            fname = possible_fnames[key]
        elif key == 'newest':
            fname = possible_fnames[most_recent_modified_key]
        else:
            raise UnknownValueError(key)

        if fname is not None and Path(fname).exists():
            data = this_reader(fname)
            print(f"File for mode {key} exists at precendence: {i+1}/{len(possible_fnames)}")
            print(f"Read data from: {fname}")
            if key != most_recent_modified_key:
                logging.warning(f"Not using most recently modified file (mode {most_recent_modified_key})")
            else:
                logging.info(f"Using most recently modified file")
            break
    else:
        logging.info(f"Found no files of possibilities: {possible_fnames}")
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
    """
    i = 1
    fpath = Path(fname)
    if fpath.exists():
        if verbose >= 1:
            print(f"Original fname {fpath} exists, so will be suffixed")
        base_fname, suffix_fname = fpath.stem, fpath.suffix
        # Check for previous application of this function
        regex = r"-\d+$"
        matches = list(re.finditer(regex, base_fname))
        if len(matches) > 0:
            base_fname = base_fname[:matches[0].start()]
            if verbose >= 1:
                print(f"Removed suffix {matches[0].group()}, so the basename is taken as: {base_fname}")

        new_base_fname = str(base_fname) + f"-{i}"
        candidate_fname = fpath.with_name(new_base_fname + str(suffix_fname))
        # TODO: should work even if i > 9 (i.e. is 2 digits long)
        while Path(candidate_fname).exists():
            i += 1
            new_base_fname = new_base_fname[:-2] + f"-{i}"
            candidate_fname = fpath.with_name(new_base_fname + str(suffix_fname))
            if verbose >= 2:
                print(f"Trying {candidate_fname}...")
        # new_fname = fpath.with_name(candidate_fname)
    else:
        candidate_fname = fname
    return str(candidate_fname)


def get_absname(project_path, fname):
    # Builds the absolute filepath using a project config filepath
    project_dir = Path(project_path).parent
    fname = Path(project_dir).joinpath(fname)
    return str(fname)


def add_name_suffix(path: str, suffix='-1'):
    fpath = Path(path)
    base_fname, suffix_fname = fpath.stem, fpath.suffix
    new_base_fname = str(base_fname) + f"{suffix}"
    candidate_fname = fpath.with_name(new_base_fname + str(suffix_fname))

    # Check for existence?
    return candidate_fname
